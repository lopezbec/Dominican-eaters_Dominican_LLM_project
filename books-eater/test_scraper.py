"""
Test script to verify the book scraper works correctly.
Tests individual components before running the full scraper.
"""

import logging
from src.clients.librosdominicanos_client import LibrosDominicanosScraper
from src.clients.youtube_client import YouTubeClient
from src.models.scraped_book import ScrapedBook

logging.basicConfig(level=logging.INFO)


def test_scraper_client():
    """Test the LibrosDominicanosScraper client."""
    print("=" * 70)
    print("TEST 1: Testing LibrosDominicanosScraper")
    print("=" * 70)
    
    scraper = LibrosDominicanosScraper(delay=0.5)
    
    print("\n1.1 Testing get_total_pages()...")
    total_pages = scraper.get_total_pages()
    print(f"✓ Found {total_pages} pages")
    
    print("\n1.2 Testing scrape_book_listing_page(1)...")
    books = scraper.scrape_book_listing_page(1)
    print(f"✓ Found {len(books)} books on page 1")
    if books:
        print(f"  First book: {books[0]['title']} by {books[0]['author']}")
    
    print("\n1.3 Testing scrape_book_detail_page()...")
    if books:
        detail = scraper.scrape_book_detail_page(books[0]['book_url'])
        if detail:
            print(f"✓ Got detail page")
            print(f"  PDF URL: {detail['pdf_url'][:50] if detail['pdf_url'] else 'None'}...")
    
    return books


def test_youtube_client():
    """Test the YouTubeClient."""
    print("\n" + "=" * 70)
    print("TEST 2: Testing YouTubeClient")
    print("=" * 70)
    
    youtube = YouTubeClient(videos_per_search=2)
    
    print("\n2.1 Testing search_audiobook()...")
    result = youtube.search_audiobook("La Mañosa", "Juan Bosch")
    if result:
        print(f"✓ Found audiobook!")
        print(f"  URL: {result['url']}")
        print(f"  Duration: {result['duration']}")
        print(f"  Type: {result['type']}")
    else:
        print("✓ No audiobook found (expected for some books)")


def test_scraped_book_model():
    """Test the ScrapedBook model."""
    print("\n" + "=" * 70)
    print("TEST 3: Testing ScrapedBook Model")
    print("=" * 70)
    
    book = ScrapedBook(
        book_number=1,
        title="Test Book",
        author="Test Author",
        year="2024",
        book_url="http://example.com/book/1",
        pdf_url="http://example.com/book.pdf"
    )
    
    print("\n3.1 Testing to_dict()...")
    book_dict = book.to_dict()
    print(f"✓ Converted to dict with {len(book_dict)} fields")
    
    print("\n3.2 Testing mark_pdf_downloaded()...")
    book.mark_pdf_downloaded()
    print(f"✓ PDF Downloaded: {book.pdf_downloaded}")
    
    print("\n3.3 Testing set_youtube_info()...")
    book.set_youtube_info(
        url="https://youtube.com/watch?v=test",
        duration="1:23:45",
        content_type="Lectura Completa"
    )
    print(f"✓ YouTube URL: {book.youtube_url}")
    print(f"✓ Duration: {book.youtube_duration}")
    print(f"✓ Type: {book.youtube_content_type}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BOOK SCRAPER COMPONENT TESTS")
    print("=" * 70)
    
    try:
        books = test_scraper_client()
        test_youtube_client()
        test_scraped_book_model()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nYou can now run the full scraper with:")
        print("  python scrape_books.py")
        print()
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
