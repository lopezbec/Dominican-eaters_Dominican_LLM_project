# Quick Start Guide - LibrosDominicanos.com Scraper

## Installation

```bash
cd books-eater
pip install -r requirements.txt
```

## Basic Usage

```bash
# Full scrape (all features)
python scrape_books.py

# Test components first (recommended)
python test_scraper.py
```

## Common Use Cases

### 1. Quick metadata scrape (no PDFs, no YouTube)
```bash
python scrape_books.py --no-pdfs --no-youtube
```
**Time:** ~5 minutes  
**Output:** Book metadata only

### 2. Metadata + YouTube (no PDFs)
```bash
python scrape_books.py --no-pdfs
```
**Time:** ~15 minutes  
**Output:** Book metadata + YouTube audiobook links

### 3. Full scrape (everything)
```bash
python scrape_books.py
```
**Time:** ~45-60 minutes  
**Output:** Book metadata + PDFs + YouTube links + PDF content as markdown

### 4. Custom output location
```bash
python scrape_books.py --output-dir my_books --output-file results.xlsx
```

## Output Files

```
scraped_books/
├── book_content_scrape.xlsx    # Main Excel file
└── pdfs/                       # Downloaded PDFs
    ├── book_001.pdf
    ├── book_002.pdf
    └── ...
```

## Excel File Columns

1. **Book Number** - Sequential number (1-165)
2. **Title** - Book title
3. **Author** - Author name
4. **Year** - Publication year
5. **PDF URL** - URL to download PDF
6. **PDF Downloaded** - Yes/No (color coded: green/red)
7. **Content** - Full book text in markdown format
8. **YouTube URL** - Audiobook video link
9. **YouTube Duration** - Video duration (e.g., "1:23:45")
10. **YouTube Content Type** - Classification (e.g., "Lectura Completa")

## Troubleshooting

### "No module named 'beautifulsoup4'"
```bash
pip install -r requirements.txt
```

### PDFs not downloading
Some PDFs are hosted externally (e.g., MEGA) and may require authentication.

### Slow scraping
This is normal. The scraper implements rate limiting (1s delay) to be respectful to the server.

### Memory issues
Use `--no-pdfs` to skip PDF downloads and conversion.

## Tips

- **Start with test:** Run `python test_scraper.py` first to verify everything works
- **Use --no-pdfs for testing:** Get results faster during development
- **Check logs:** View `scraper.log` for detailed information
- **Be patient:** Full scrape takes 45-60 minutes due to rate limiting

## Architecture

```
LibrosDominicanosScraper ──> Scrapes website
         │
         ├──> PDFProcessor ──> Converts PDFs to markdown
         │
         ├──> YouTubeClient ──> Finds audiobooks
         │
         └──> BookScraperService ──> Orchestrates everything
                     │
                     └──> Exports to Excel
```

## Support

- **README:** See `SCRAPER_README.md` for full documentation
- **Implementation Details:** See `IMPLEMENTATION_SUMMARY.md`
- **Test Components:** Run `python test_scraper.py`

## Expected Statistics

After a successful run, you should see something like:

```
Total books scraped: 165
PDFs found: 140-145
PDFs downloaded: 135-140
YouTube audiobooks found: 20-30
Content extracted: 130-140
```

## One-Line Quick Start

```bash
cd books-eater && pip install -r requirements.txt && python test_scraper.py && python scrape_books.py
```
