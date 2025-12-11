# Book Content Scraper for LibrosDominicanos.com

This module scrapes book data from http://librosdominicanos.com, downloads PDFs, converts them to markdown, and searches for audiobook versions on YouTube.

## Features

- ✅ Scrapes all 165 books across 11 pages from librosdominicanos.com
- ✅ Extracts book metadata (title, author, year, PDF URL)
- ✅ Downloads PDF files (where available)
- ✅ Converts PDF content to markdown using pdfplumber
- ✅ Searches YouTube for audiobook versions
- ✅ Exports all data to Excel with proper formatting
- ✅ Follows SOLID principles with clean architecture
- ✅ Progress bars with tqdm for long-running operations
- ✅ Proper error handling and logging

## Architecture

The scraper follows SOLID principles with clear separation of concerns:

```
books-eater/
├── src/
│   ├── clients/
│   │   ├── librosdominicanos_client.py  # Web scraper for the site
│   │   └── youtube_client.py            # YouTube search (existing)
│   ├── models/
│   │   └── scraped_book.py              # Data model for scraped books
│   └── services/
│       ├── pdf_processor.py             # PDF to markdown converter
│       └── book_scraper_service.py      # Main orchestration service
└── scrape_books.py                      # Main entry point script
```

## Installation

Install the required dependencies:

```bash
cd books-eater
pip install -r requirements.txt
```

## Usage

### Basic Usage

Scrape all books with default settings:

```bash
python scrape_books.py
```

This will:
1. Scrape all book listings from librosdominicanos.com
2. Visit each book's detail page to get the PDF URL
3. Download all available PDFs to `scraped_books/pdfs/`
4. Convert PDFs to markdown
5. Search YouTube for audiobook versions
6. Save results to `scraped_books/book_content_scrape.xlsx`

### Advanced Options

```bash
# Skip PDF downloads
python scrape_books.py --no-pdfs

# Skip YouTube searches
python scrape_books.py --no-youtube

# Custom output directory
python scrape_books.py --output-dir my_books

# Custom Excel filename
python scrape_books.py --output-file my_results.xlsx

# Combine options
python scrape_books.py --output-dir data --no-youtube --output-file results.xlsx
```

## Output Format

The Excel file contains the following columns:

| Column | Description |
|--------|-------------|
| Book Number | Sequential number (1-165) |
| Title | Book title |
| Author | Author name |
| Year | Publication year |
| PDF URL | URL to download the PDF |
| PDF Downloaded | Yes/No status with color coding |
| Content | Markdown text extracted from PDF |
| YouTube URL | URL of audiobook video (if found) |
| YouTube Duration | Duration of the video |
| YouTube Content Type | Classification (Lectura Completa, Dramatización, etc.) |

## Components

### LibrosDominicanosScraper

Web scraper client for librosdominicanos.com:
- Scrapes paginated book listings
- Extracts book detail pages
- Downloads PDF files
- Implements respectful rate limiting

### PDFProcessor

Converts PDF files to markdown:
- Uses pdfplumber for text extraction
- Formats output as markdown with page markers
- Handles errors gracefully

### BookScraperService

Main orchestration service:
- Coordinates all scraping operations
- Manages file I/O
- Integrates with YouTube search
- Generates statistics
- Exports to formatted Excel

### ScrapedBook

Data model representing a scraped book:
- Stores all book metadata
- Handles PDF and YouTube information
- Converts to dictionary for export

## Error Handling

The scraper includes comprehensive error handling:
- Continues processing if individual books fail
- Logs all errors to `scraper.log`
- Provides detailed progress information
- Generates statistics at completion

## Logging

Logs are saved to `scraper.log` with the following information:
- Scraping progress
- PDF download status
- YouTube search results
- Errors and warnings

## Examples

### Example 1: Quick scrape without downloads

```bash
python scrape_books.py --no-pdfs
```

This is useful for quickly getting book metadata and YouTube links without downloading large PDFs.

### Example 2: Full scrape with all features

```bash
python scrape_books.py --output-dir complete_data
```

Downloads everything and saves to a custom directory.

### Example 3: Metadata only

```bash
python scrape_books.py --no-pdfs --no-youtube
```

Just scrapes book information without external lookups.

## Performance

- Implements rate limiting (1 second between requests)
- Uses streaming for large PDF downloads
- Progress bars show real-time status
- Typical runtime: 30-60 minutes for full scrape (depending on network speed)

## Notes

- The scraper is respectful of the source website with proper delays
- PDFs are stored in `output_dir/pdfs/` as `book_001.pdf`, `book_002.pdf`, etc.
- Some PDFs may be hosted on external services (e.g., MEGA) and may require special handling
- YouTube search uses scrapetube (no API key required)
- Excel file uses color coding: Green (PDF downloaded), Red (PDF not available)

## Troubleshooting

### PDFs not downloading
Some PDFs are hosted on external services like MEGA and may require authentication or special handling.

### YouTube not found
Not all books have audiobook versions on YouTube. The scraper tries multiple search strategies.

### Memory issues
For large PDFs, the content extraction may use significant memory. Consider using `--no-pdfs` for initial testing.

## Dependencies

- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML parser
- `pdfplumber` - PDF text extraction
- `requests` - HTTP requests
- `pandas` - Data manipulation
- `openpyxl` - Excel file creation
- `tqdm` - Progress bars
- `scrapetube` - YouTube search
