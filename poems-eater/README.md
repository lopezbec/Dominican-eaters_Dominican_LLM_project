# Poems Eater - Dominican Poetry Recitations Scraper

An automated tool to find, catalog, and process YouTube videos of Dominican poetry recitations for LLM training. Uses keyless web scraping to find performances, dramatizations, and readings.

## Overview

Poems Eater searches for Dominican poetry recitations on YouTube, collects comprehensive metadata, and integrates with the audio processing pipeline for transcription and alignment. The module includes a curated database of 100+ classic and contemporary Dominican poems.

## Features

- **Keyless Scraping**: Uses `scrapetube` for unlimited YouTube searches (no API key required)
- **Curated Poetry Database**: 100+ poems from Dominican literary canon
- **Smart Search**: Finds recitations, dramatizations, and readings
- **Multiple Export Formats**: Excel and CSV outputs
- **Configurable Search**: Adjust number of videos analyzed per poem
- **Detailed Statistics**: Success rates, popular genres, most-found authors
- **Pipeline Integration**: Seamless connection to transcription and alignment
- **Custom Lists**: Support for user-defined poem lists

## Architecture

```
poems-eater/
├── src/
│   ├── clients/
│   │   └── youtube_client.py      # YouTube search interface
│   ├── models/
│   │   └── poem.py                # Poem dataclass definition
│   ├── services/
│   │   ├── poem_service.py        # Core scraping orchestration
│   │   └── transcription_exporter.py  # Data export utilities
│   └── utils/
│       ├── config.py              # Configuration management
│       ├── dominican_poems.py     # Curated poem database
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

1. **Navigate to project root**
   ```bash
   cd Dominican-eaters_Dominican_LLM_project
   ```

2. **Install dependencies** (if not already done)
   ```bash
   pip install -r requirements.txt
   # or specifically for this module:
   pip install -r poems-eater/requirements.txt
   ```

3. **Verify setup**
   ```bash
   python poems-eater/main.py --help
   ```

## Usage

### Via Main CLI (Recommended)

```bash
# Scrape poems only
python dominican-eater.py scrape --type poems

# Run full pipeline (scrape + download + transcribe + validate)
python dominican-eater.py pipeline --type poems

# Run pipeline with specific Whisper model
python dominican-eater.py pipeline --type poems --model large-v3

# Skip scraping (use existing data)
python dominican-eater.py pipeline --type poems --skip-scrape
```

### Direct Module Execution

```bash
# Run standalone scraper (uses built-in poem database)
python poems-eater/main.py
```

### Individual Pipeline Steps

```bash
# 1. Scrape metadata
python dominican-eater.py scrape --type poems

# 2. Download audio files
python dominican-eater.py download --type poems-eater

# 3. Transcribe (full mode for poems)
python dominican-eater.py transcribe --type poems-eater --model base

# 4. Validate outputs
python dominican-eater.py validate --type poems-eater
```

### Custom Poem Lists

Create a `poems-eater/poems_list.txt` file with format: `Title | Author | Year | Genre`

```
A la Patria | Salomé Ureña | 1874 | Patriotic
Hay un País en el Mundo | Pedro Mir | 1949 | Epic
Letanía | Manuel del Cabral | 1930 | Social
```

The program will automatically use this file if it exists, otherwise it uses the built-in database.

## Output

The scraper generates `audio_processing/data/dominican_poems.xlsx` and `dominican_poems.csv` with:

| Column | Description |
|--------|-------------|
| **Título** | Poem title |
| **Autor** | Poet name |
| **Año** | Publication year |
| **Género** | Literary genre (Patriotic, Epic, Social, Romantic, etc.) |
| **URL YouTube** | Video link |
| **Duración** | Video duration |
| **Tipo** | Content type (recitation, dramatization, reading) |

### Output Locations

All outputs are centralized in the audio_processing directory:

```
audio_processing/
├── data/
│   └── dominican_poems.xlsx       # Scraped metadata
├── audio/poems/                   # Downloaded audio files
├── transcriptions/poems/          # Whisper transcriptions
├── reference_texts/poems/         # Reference poem texts (manual)
└── reports/poems/                 # Processing reports
    ├── download_report.json
    ├── transcription_report.json
    └── validation_report.json
```

### Statistics Included

The output includes comprehensive statistics:
- Total poems searched
- Successfully found videos
- Success rate by author
- Most popular genres
- Average video duration

## Pipeline Integration

Poems Eater integrates seamlessly with the audio processing pipeline:

1. **Scraping** (this module): Searches YouTube and exports metadata
2. **Download**: Audio files downloaded via yt-dlp
3. **Transcription**: Full audio transcription with Whisper (no partial mode)
4. **Alignment**: Optional alignment with reference poem texts (requires manual setup)
5. **Validation**: Quality reports and metrics

### Reference Texts (Optional)

For alignment, reference poem texts can be added manually:

1. Create text files in `audio_processing/reference_texts/poems/`
2. Match filename with transcription: `poems-eater_001_title.txt`
3. Run alignment: `python dominican-eater.py align --type poems-eater`

*Note: Unlike lyrics (automated via Genius API), poem texts require manual collection.*

## Configuration

Module configuration is in `audio_processing/config.yaml`:

```yaml
modules:
  poems-eater:
    excel_path: audio_processing/data/dominican_poems.xlsx
    url_column: "URL YouTube"
    audio_dir: audio_processing/audio/poems
    transcriptions_dir: audio_processing/transcriptions/poems
    reference_texts_dir: audio_processing/reference_texts/poems
    alignments_dir: audio_processing/alignments/poems
    reports_dir: audio_processing/reports/poems
```

## Adding New Poems

Edit `poems-eater/src/utils/dominican_poems.py` to add to the curated list:

```python
DOMINICAN_POEMS = [
    {
        "title": "Poem Title",
        "author": "Poet Name",
        "year": 1950,
        "genre": "Social"
    },
    # ... more poems
]
```

## Search Configuration

Adjust search parameters in the poem service:

- **Videos per poem**: Number of results to analyze per search
- **Search terms**: Variations like "recitación", "declamación", "poema"
- **Quality filters**: Duration limits, view counts

## Curated Database

The module includes 100+ poems from major Dominican poets:

### Featured Authors
- **Salomé Ureña**: Patriotic poetry, education advocate
- **Pedro Mir**: National poet, epic social poetry
- **Manuel del Cabral**: Social and existential themes
- **Aída Cartagena Portalatín**: Feminist and social poetry
- **Héctor Incháustegui Cabral**: Historical and social themes
- **Franklin Mieses Burgos**: Modernist poetry
- **And many more...**

### Genres Covered
- Patriotic poetry
- Epic and social poetry
- Romantic poetry
- Modernist poetry
- Contemporary poetry
- Feminist poetry

## Dependencies

- `scrapetube` - Keyless YouTube searching
- `pandas` - Data manipulation and export
- `openpyxl` - Excel file writing
- `python-dotenv` - Configuration management

## Troubleshooting

### No videos found for poem

- Some poems may not have recitations on YouTube
- Try adding variations to search terms
- Check spelling in poem title/author

### Download failures

- Some videos may be age-restricted or region-locked
- yt-dlp handles most cases automatically
- Check internet connection

### Transcription quality

- Poems use full transcription (no partial mode)
- Recitations may have background music affecting quality
- Consider using larger Whisper models for better accuracy

---

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The authors gratefully acknowledge this support.

Any opinions, findings, conclusions, or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of MESCyT.