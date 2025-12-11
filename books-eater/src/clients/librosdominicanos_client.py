"""Web scraper client for librosdominicanos.com."""

import logging
import time
from typing import List, Optional, Dict
import requests
from bs4 import BeautifulSoup
from mega import Mega

logger = logging.getLogger(__name__)


class LibrosDominicanosScraper:
    """
    Client for scraping book data from librosdominicanos.com.
    """
    
    BASE_URL = "http://librosdominicanos.com"
    BOOKS_URL = f"{BASE_URL}/books"
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize the scraper.
        
        Args:
            delay: Delay between requests in seconds (be respectful)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        self.mega = Mega()
    
    def get_total_pages(self) -> int:
        """
        Get the total number of pages.
        
        Returns:
            Number of pages
        """
        try:
            response = self.session.get(self.BOOKS_URL, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html5lib')
            
            pagination = soup.find('ul', class_='pagination')
            if pagination:
                last_page_link = pagination.find_all('a', class_='page-link')
                for link in reversed(last_page_link):
                    href = link.get('href', '')
                    if '/page/' in href:
                        page_num = href.split('/page/')[-1]
                        if page_num.isdigit():
                            return int(page_num)
                    text = link.get_text(strip=True)
                    if text.isdigit():
                        return int(text)
            
            return 1
            
        except Exception as e:
            logger.error(f"Error getting total pages: {e}")
            return 1
    
    def scrape_book_listing_page(self, page: int = 1) -> List[Dict[str, str]]:
        """
        Scrape a single listing page to get basic book information.
        
        Args:
            page: Page number to scrape
            
        Returns:
            List of dictionaries with basic book info
        """
        try:
            if page == 1:
                url = self.BOOKS_URL
            else:
                url = f"{self.BOOKS_URL}/page/{page}"
            
            logger.info(f"Scraping page {page}: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html5lib')
            books = []
            
            book_divs = soup.find_all('div', class_='book')
            
            for book_div in book_divs:
                try:
                    title_elem = book_div.find('div', class_='book-title')
                    if not title_elem:
                        continue
                    
                    title_link = title_elem.find('a')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    book_url = title_link.get('href', '')
                    
                    if book_url:
                        book_url = f"{self.BASE_URL}{book_url}"
                    
                    author_elem = book_div.find('span', class_='book-author')
                    author = author_elem.get_text(strip=True) if author_elem else "Unknown"
                    
                    year_elem = book_div.find('span', class_='book-publishing-year')
                    year = year_elem.get_text(strip=True).rstrip(',').strip() if year_elem else "N/A"
                    
                    books.append({
                        'title': title,
                        'author': author,
                        'year': year,
                        'book_url': book_url
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing book on page {page}: {e}")
                    continue
            
            time.sleep(self.delay)
            return books
            
        except Exception as e:
            logger.error(f"Error scraping page {page}: {e}")
            return []
    
    def scrape_all_listings(self) -> List[Dict[str, str]]:
        """
        Scrape all listing pages to get all books.
        
        Returns:
            List of all books with basic info
        """
        total_pages = self.get_total_pages()
        logger.info(f"Found {total_pages} pages to scrape")
        
        all_books = []
        for page in range(1, total_pages + 1):
            books = self.scrape_book_listing_page(page)
            all_books.extend(books)
            logger.info(f"Page {page}/{total_pages}: Found {len(books)} books")
        
        return all_books
    
    def scrape_book_detail_page(self, book_url: str) -> Optional[Dict[str, str]]:
        """
        Scrape a book's detail page to get PDF URL and other info.
        
        Args:
            book_url: URL of the book detail page
            
        Returns:
            Dictionary with PDF URL and additional info, or None
        """
        try:
            logger.debug(f"Scraping detail page: {book_url}")
            response = self.session.get(book_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html5lib')
            
            pdf_url = None
            download_link = soup.find('a', class_='download-link')
            if download_link:
                pdf_url = download_link.get('id', '')
            
            description = ""
            desc_div = soup.find('div', class_='book-description')
            if desc_div:
                description = desc_div.get_text(strip=True)
            
            time.sleep(self.delay)
            
            return {
                'pdf_url': pdf_url if pdf_url else None,
                'description': description
            }
            
        except Exception as e:
            logger.error(f"Error scraping detail page {book_url}: {e}")
            return None
    
    def download_pdf(self, pdf_url: str, output_path: str) -> Optional[str]:
        """
        Download a PDF file from MEGA URL.
        
        Args:
            pdf_url: MEGA URL of the PDF file (format: https://mega.nz/file/XXX#YYY)
            output_path: Path to save the PDF (directory path for MEGA, full path for others)
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        try:
            if not pdf_url or pdf_url == "N/A":
                return None
            
            if "mega.nz" not in pdf_url:
                logger.warning(f"Not a MEGA URL: {pdf_url}")
                return None
            
            logger.debug(f"Downloading PDF from MEGA: {pdf_url}")
            
            from pathlib import Path
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = self.mega.download_url(pdf_url, dest_path=str(output_dir))
            
            if file_path:
                logger.debug(f"PDF downloaded: {file_path}")
                time.sleep(self.delay)
                return str(file_path)
            else:
                logger.error(f"Failed to download from MEGA: {pdf_url}")
                return None
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {pdf_url}: {e}")
            return None
