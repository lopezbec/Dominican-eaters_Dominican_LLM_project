"""
Main script for scraping books from librosdominicanos.com.

Usage:
    python scrape_books.py [--no-pdfs] [--no-youtube] [--output-dir DIR]
"""

import logging
import argparse
from pathlib import Path

from src.services.book_scraper_service import BookScraperService


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scraper.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for the book scraper."""
    parser = argparse.ArgumentParser(
        description='Scrape books from librosdominicanos.com'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scraped_books',
        help='Output directory for scraped data (default: scraped_books)'
    )
    parser.add_argument(
        '--no-pdfs',
        action='store_true',
        help='Skip PDF download and conversion'
    )
    parser.add_argument(
        '--no-youtube',
        action='store_true',
        help='Skip YouTube audiobook search'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='book_content_scrape.xlsx',
        help='Output Excel filename (default: book_content_scrape.xlsx)'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("=" * 70)
    print("LIBROSDOMINICANOS.COM BOOK SCRAPER")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Download PDFs: {not args.no_pdfs}")
    print(f"  Search YouTube: {not args.no_youtube}")
    print(f"  Output file: {args.output_file}")
    print()
    
    service = BookScraperService(
        output_dir=args.output_dir,
        download_pdfs=not args.no_pdfs,
        search_youtube=not args.no_youtube
    )
    
    print("Starting scraping process...")
    print("-" * 70)
    
    books = service.scrape_all_books()
    
    print("\n" + "-" * 70)
    print("Saving results to Excel...")
    
    service.save_to_excel(books, args.output_file)
    
    stats = service.generate_statistics(books)
    
    print("\n" + "=" * 70)
    print("SCRAPING STATISTICS")
    print("=" * 70)
    print(f"Total books scraped: {stats['total_books']}")
    print(f"PDFs found: {stats['pdfs_found']}")
    print(f"PDFs downloaded: {stats['pdfs_downloaded']}")
    print(f"YouTube audiobooks found: {stats['youtube_found']}")
    print(f"Content extracted: {stats['content_extracted']}")
    print("=" * 70)
    print("\n✓ Scraping completed successfully!\n")


if __name__ == "__main__":
    main()
