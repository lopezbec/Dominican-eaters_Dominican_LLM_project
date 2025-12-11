# Books Eater - Dominican Audiobooks Scraper

A specialized tool to find, catalog, and process Dominican literature audiobooks from YouTube for LLM training.

## Overview

Books Eater searches for audiobooks by Dominican authors on YouTube, collects comprehensive metadata, and integrates with the audio processing pipeline for transcription and alignment.

## Features

- **Automated YouTube Search**: Uses scrapetube for keyless, unlimited searches
- **Curated Literature Catalog**: Built-in database of classic Dominican works
- **Metadata Extraction**: Title, author, year, duration, URL, availability
- **Excel Export**: Professional formatted output with statistics
- **Pipeline Integration**: Seamless connection to transcription and alignment stages
- **Custom Lists**: Support for user-defined book lists

## Architecture

```
books-eater/
├── src/
│   ├── clients/
│   │   └── youtube_client.py      # YouTube search interface
│   ├── models/
│   │   └── book.py                # Book dataclass definition
│   ├── services/
│   │   ├── audiobook_service.py   # Core scraping logic
│   │   └── transcription_exporter.py  # Data export utilities
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── dominican_books.py     # Curated book database
│       └── file_handler.py        # File I/O operations
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup

### Prerequisites
- Python 3.8+
- Internet connection for YouTube access

### Installation

1. **Navigate to the project root**
   ```bash
   cd Dominican-eaters_Dominican_LLM_project
   ```

2. **Install dependencies** (if not already done)
   ```bash
   pip install -r requirements.txt
   # or specifically for this module:
   pip install -r books-eater/requirements.txt
   ```

3. **Verify setup**
   ```bash
   python books-eater/main.py --help
   ```

## Usage

### Via Main CLI (Recommended)

```bash
# Scrape books only
python dominican-eater.py scrape --type books

# Run full pipeline (scrape + download + transcribe + validate)
python dominican-eater.py pipeline --type books

# Run full pipeline with specific Whisper model
python dominican-eater.py pipeline --type books --model large-v3
```

### Direct Module Execution

```bash
# Run standalone
python books-eater/main.py
```

### Custom Book Lists

Create a `books-eater/books_list.txt` file with format: `Title | Author | Year`

```
La Mañosa | Juan Bosch | 1936
Over | Ramón Marrero Aristy | 1940
Enriquillo | Manuel de Jesús Galván | 1882
```

The program will automatically use this file if it exists.

## Output

The script generates `audio_processing/data/dominican_audiobooks.xlsx` with:

| Column | Description |
|--------|-------------|
| **Título** | Book title |
| **Autor** | Author name |
| **Año** | Publication year |
| **Género** | Literary genre |
| **URL YouTube** | Video link |
| **Duración** | Video duration |
| **Disponible** | Availability status |

### Output Location

All outputs are centralized in the audio_processing directory:

```
audio_processing/
├── data/
│   └── dominican_audiobooks.xlsx  # Scraped metadata
├── audio/books/                   # Downloaded audio files
├── transcriptions/books/          # Whisper transcriptions
├── reference_texts/books/         # Reference texts for alignment
└── reports/books/                 # Processing reports
```

## Pipeline Integration

Books Eater integrates seamlessly with the audio processing pipeline:

1. **Scraping** (this module): Searches YouTube and exports metadata
2. **Download**: Audio files downloaded via yt-dlp
3. **Transcription**: Full audio transcription with Whisper (no partial mode)
4. **Alignment**: Optional alignment with reference book texts
5. **Validation**: Quality reports and metrics

### Processing Commands

```bash
# Step-by-step processing
python dominican-eater.py scrape --type books
python dominican-eater.py download --type books-eater
python dominican-eater.py transcribe --type books-eater --model base
python dominican-eater.py validate --type books-eater

# Or run complete pipeline
python dominican-eater.py pipeline --type books
```

## Configuration

Module configuration is in `audio_processing/config.yaml`:

```yaml
modules:
  books-eater:
    excel_path: audio_processing/data/dominican_audiobooks.xlsx
    url_column: "URL YouTube"
    audio_dir: audio_processing/audio/books
    transcriptions_dir: audio_processing/transcriptions/books
    reference_texts_dir: audio_processing/reference_texts/books
    reports_dir: audio_processing/reports/books
```

## Adding New Books

Edit `books-eater/src/utils/dominican_books.py` to add to the curated list:

```python
DOMINICAN_BOOKS = [
    {
        "title": "Book Title",
        "author": "Author Name",
        "year": 2024,
        "genre": "Fiction"
    },
    # ... more books
]
```

## Dependencies

- `scrapetube` - Keyless YouTube searching
- `pandas` - Data manipulation and export
- `openpyxl` - Excel file writing
- `python-dotenv` - Configuration management

---

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The authors gratefully acknowledge this support.

Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of MESCyT.