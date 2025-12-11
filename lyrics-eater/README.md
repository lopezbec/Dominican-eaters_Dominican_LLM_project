# Lyrics Eater - Dominican Song Lyrics Scraper

An automated tool to fetch Dominican song lyrics from Genius.com, find corresponding YouTube audio, and process them through an optimized alignment pipeline for LLM training.

## Overview

Lyrics Eater scrapes song lyrics from the Genius API, searches for matching YouTube videos, and integrates with an optimized audio processing pipeline that uses **partial transcription** for fast alignment verification during data collection.

## Features

- 🔍 **Genius API Integration**: Automated song and lyric retrieval
- 📝 **Rich Metadata**: Artist, album, genre, record label, release date
- 🎵 **Full Lyric Scraping**: Complete lyrics with section markers
- 🎬 **YouTube Linking**: Automatic video discovery via scrapetube (no API key)
- 📊 **Excel Export**: Professional formatted datasets
- 📦 **Batch Processing**: Process multiple artists from text file
- ⚡ **Optimized Pipeline**: 45-second partial transcription for fast verification
- 🎯 **Smart Alignment**: Early-stop search with 4 similarity metrics
- 🔄 **Robust Error Handling**: Automatic retries and timeout management

## Architecture

```
lyrics-eater/
├── src/
│   ├── clients/
│   │   ├── genius_client.py       # Genius API integration
│   │   └── youtube_client.py      # YouTube search (keyless)
│   ├── models/
│   │   └── song.py                # Song dataclass definition
│   ├── services/
│   │   ├── lyrics_service.py      # Core scraping orchestration
│   │   └── transcription_exporter.py  # Data export utilities
│   └── utils/
│       ├── config.py              # Configuration & environment
│       └── file_handler.py        # File I/O operations
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── main.py                        # CLI entry point
├── requirements.txt               # Python dependencies
├── searches.txt.example           # Artist search template
└── README.md                      # This file
```

## Setup

### Prerequisites
- Python 3.8+
- Genius API Token (free from https://genius.com/api-clients)
- Internet connection

### Installation

1. **Navigate to project root**
   ```bash
   cd Dominican-eaters_Dominican_LLM_project
   ```

2. **Install dependencies** (if not already done)
   ```bash
   pip install -r requirements.txt
   # or specifically for this module:
   pip install -r lyrics-eater/requirements.txt
   ```

3. **Configure Genius API Token**
   ```bash
   cd lyrics-eater
   cp .env.example .env
   ```
   
   Edit `.env` and add your Genius API token:
   ```env
   GENIUS_ACCESS_TOKEN=your_genius_token_here
   ```
   
   To get a token:
   - Go to https://genius.com/api-clients
   - Create an application
   - Copy the "Client Access Token"

4. **Create Artist Search List**
   ```bash
   cp searches.txt.example searches.txt
   ```
   
   Edit `searches.txt` with one artist per line:
   ```
   Juan Luis Guerra
   Romeo Santos
   Aventura
   Prince Royce
   Toby Love
   ```

5. **Verify Setup**
   ```bash
   python lyrics-eater/main.py --help
   ```

## Usage

### Via Main CLI (Recommended)

```bash
# Scrape lyrics only
python dominican-eater.py scrape --type lyrics

# Run full pipeline with optimizations (scrape + download + partial transcribe + align + validate)
python dominican-eater.py pipeline --type lyrics

# Run pipeline with specific Whisper model and explicit partial mode
python dominican-eater.py pipeline --type lyrics --model base --partial

# Skip scraping (use existing data)
python dominican-eater.py pipeline --type lyrics --skip-scrape

# Force re-processing
python dominican-eater.py pipeline --type lyrics --force
```

### Direct Module Execution

```bash
# Run standalone scraper
python lyrics-eater/main.py

# The scraper will:
# 1. Read artists from searches.txt
# 2. Search Genius for each artist's songs
# 3. Scrape lyrics for found songs
# 4. Find YouTube videos for each song
# 5. Export to audio_processing/data/dominican_songs.xlsx
```

### Individual Pipeline Steps

```bash
# 1. Scrape metadata and lyrics
python dominican-eater.py scrape --type lyrics

# 2. Download audio files
python dominican-eater.py download --type lyrics-eater

# 3. Transcribe with partial mode (45 seconds, 3-4x faster)
python dominican-eater.py transcribe --type lyrics-eater --partial

# 4. Align and verify with smart optimization
python dominican-eater.py align --type lyrics-eater

# 5. Generate validation reports
python dominican-eater.py validate --type lyrics-eater
```

## Output

The scraper generates `audio_processing/data/dominican_songs.xlsx` with:

| Column | Description |
|--------|-------------|
| **genero** | Musical genre(s) |
| **artista** | Artist name |
| **cancion** | Song title |
| **letras** | Complete lyrics text |
| **enlace_genius** | Genius.com URL |
| **enlace_youtube** | YouTube video URL |
| **album** | Album name |
| **discografica** | Record label |
| **fecha_lanzamiento** | Release date |

### Output Locations

All outputs are centralized in the audio_processing directory:

```
audio_processing/
├── data/
│   └── dominican_songs.xlsx      # Scraped metadata & lyrics
├── audio/lyrics/                 # Downloaded audio files (.m4a)
├── transcriptions/lyrics/        # Whisper transcriptions (.json)
├── reference_texts/lyrics/       # Lyrics reference files (.txt)
├── alignments/lyrics/            # Alignment data (optional)
└── reports/lyrics/               # Processing & alignment reports
    ├── download_report.json
    ├── transcription_report.json
    ├── alignment_report.json
    └── validation_report.json
```

## Pipeline Optimization (Data Collection Phase)

Lyrics Eater uses **partial transcription** and **smart alignment** for 3-4x faster processing during data collection:

### Optimizations Applied

#### 1. Partial Transcription (Default in Pipeline)
- **What**: Only first 45 seconds transcribed (configurable)
- **Why**: Song lyrics typically start within first 45 seconds
- **Benefit**: 4x faster transcription (~30-45s vs 2-3 minutes)
- **Quality**: Same alignment verification accuracy
- **Auto-enabled**: Pipeline automatically adds `--partial` flag

#### 2. Smart Alignment Search
- **Limited Window**: Searches only first 200 words for alignment start
- **Early Stop**: Stops at first match ≥0.8 similarity (usually 2-3 iterations)
- **Window Size**: 50-word sliding window for efficiency
- **Benefit**: 5x faster alignment (~2-3s vs 10-15s)

#### 3. Four Alignment Metrics
1. **WER (Word Error Rate)**: Word-level accuracy
2. **Character Similarity**: Levenshtein-based character matching
3. **Jaccard Similarity**: Vocabulary overlap
4. **Cosine Similarity**: Word frequency distribution

### Performance Comparison

| Mode | Transcription | Alignment | Total/Song | Use Case |
|------|--------------|-----------|------------|----------|
| **Full** | 2-3 min | 10-15 sec | ~2-3 min | Final dataset |
| **Partial** | 30-45 sec | 2-3 sec | **35-50 sec** | **Data collection** |
| **Speedup** | 4x | 5x | **3-4x** | Current phase |

### When Full Processing Happens

Full transcription will be performed:
- On server infrastructure (more compute)
- During model training phase (need complete audio)
- For production dataset (final quality check)

### Configuration

In `audio_processing/config.yaml`:

```yaml
whisper:
  partial_duration: 45  # Seconds for partial mode

alignment:
  window_size: 50       # Words in sliding window
  min_match_length: 30  # Minimum words for match
  min_similarity: 0.7   # Minimum acceptable similarity
```

### Implementation References

- Partial mode: `audio_processing/src/transcriber.py:89-92`
- Limited search: `audio_processing/src/aligner.py:157`
- Early stop: `audio_processing/src/aligner.py:177-179`
- Auto-enable: `dominican-eater.py:86`

## Alignment Quality Metrics

### Metric Definitions

| Metric | Range | Good Threshold | Description |
|--------|-------|----------------|-------------|
| **WER** | 0.0 to ∞ | < 0.2 | Word Error Rate - lower is better |
| **Char Sim** | 0.0 to 1.0 | > 0.85 | Character similarity - higher is better |
| **Jaccard** | 0.0 to 1.0 | > 0.7 | Vocabulary overlap - higher is better |
| **Cosine** | 0.0 to 1.0 | > 0.8 | Word frequency similarity - higher is better |

### Interpretation Examples

**Excellent Alignment** (ready for training):
```
WER: 0.15, Char Sim: 0.92, Jaccard: 0.78, Cosine: 0.85
→ Only 15% word errors, high character match, good vocabulary coverage
```

**Good Alignment** (acceptable):
```
WER: 0.35, Char Sim: 0.75, Jaccard: 0.65, Cosine: 0.72
→ Some errors but acceptable quality, may have intro/outro sections
```

**Poor Alignment** (needs review):
```
WER: 0.68, Char Sim: 0.54, Jaccard: 0.42, Cosine: 0.58
→ High error rate, possible wrong video or language mismatch
```

### Viewing Alignment Reports

```bash
# View alignment report
cat audio_processing/reports/lyrics/alignment_report.json

# Example report structure:
{
  "module": "lyrics-eater",
  "total_files": 150,
  "successful": 142,
  "failed": 8,
  "success_rate": 0.947,
  "average_wer": 0.18,
  "average_char_similarity": 0.89,
  "average_jaccard_similarity": 0.74,
  "average_cosine_similarity": 0.82,
  "results": [...]
}
```

## Troubleshooting

### Genius API Issues

**Error: "Invalid token"**
- Verify token in `.env` file
- Ensure no extra spaces or quotes
- Get new token at https://genius.com/api-clients

**Error: "Rate limit exceeded"**
- Genius API has rate limits
- Add delays between requests (handled automatically)
- Consider processing in smaller batches

### YouTube Search Issues

**No YouTube links found**
- Some songs may not have videos
- Try different search terms in `searches.txt`
- Check internet connection

**Wrong video matched**
- YouTube search uses best effort matching
- Manual verification recommended for critical datasets
- Check alignment metrics to identify mismatches

### Alignment Issues

**High WER scores**
- Video may have intro/outro dialogue
- Wrong video matched (check alignment_start_offset)
- Different language or remix version
- Check reference lyrics quality

**No reference text found**
- Ensure lyrics were scraped successfully
- Check `audio_processing/reference_texts/lyrics/`
- Re-run scraper if needed

## Adding More Artists

Edit `lyrics-eater/searches.txt` and add one artist per line:

```
# Dominican artists
Juan Luis Guerra
Romeo Santos
Aventura
Prince Royce
Toby Love
Chichi Peralta
Milly Quezada
Frank Reyes

# Add more...
```

Then re-run:
```bash
python dominican-eater.py scrape --type lyrics
```

## Module Configuration

Configuration in `audio_processing/config.yaml`:

```yaml
modules:
  lyrics-eater:
    excel_path: audio_processing/data/dominican_songs.xlsx
    url_column: "enlace_youtube"
    lyrics_column: "letras"
    audio_dir: audio_processing/audio/lyrics
    transcriptions_dir: audio_processing/transcriptions/lyrics
    reference_texts_dir: audio_processing/reference_texts/lyrics
    alignments_dir: audio_processing/alignments/lyrics
    reports_dir: audio_processing/reports/lyrics
```

## Dependencies

- `requests` - HTTP client for Genius API
- `beautifulsoup4` - HTML parsing for lyrics scraping
- `pandas` - Data manipulation and export
- `openpyxl` - Excel file writing
- `python-dotenv` - Environment variable management
- `scrapetube` - Keyless YouTube searching

---