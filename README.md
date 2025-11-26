# Dominican Eaters - LLM Training Dataset Project

A comprehensive pipeline for collecting, processing, and aligning Dominican Spanish audio data from YouTube for Large Language Model training. This project focuses on three content domains: audiobooks, song lyrics, and poetry.

## Project Overview

Dominican Eaters automates the collection and processing of Dominican Spanish content through:
- **Web scraping**: Automated search and metadata extraction from YouTube
- **Audio download**: Bulk downloading of audio files via yt-dlp
- **Transcription**: Whisper-based speech-to-text conversion
- **Text alignment**: Word Error Rate (WER) calculation for quality validation
- **Dataset generation**: Export to multiple formats (JSONL, CSV, Excel)

## Project Structure

```
Dominican-eaters_Dominican_LLM_project/
├── dominican-eater.py          # Main CLI entry point
├── audio_processing/            # Core processing pipeline
│   ├── src/                    # Processing modules
│   │   ├── downloader.py       # YouTube audio download
│   │   ├── transcriber.py      # Whisper transcription
│   │   ├── aligner.py          # Text alignment & WER
│   │   └── validator.py        # Quality validation
│   ├── config.yaml             # Pipeline configuration
│   └── utilities/setup.py      # Dependency installer
├── books-eater/                # Audiobook scraper
├── lyrics-eater/               # Song lyrics scraper
└── poems-eater/                # Poetry scraper
```

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dominican-eaters_Dominican_LLM_project.git
cd Dominican-eaters_Dominican_LLM_project
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
# Install all dependencies
python audio_processing/utilities/setup.py

# Or manually for each module
pip install -r audio_processing/requirements.txt
pip install -r books-eater/requirements.txt
pip install -r lyrics-eater/requirements.txt
pip install -r poems-eater/requirements.txt
```

4. Configure environment (lyrics module only):
```bash
cd lyrics-eater
cp .env.example .env
# Edit .env and add your Genius API token
```

5. Create project directories:
```bash
python dominican-eater.py setup
```

## Usage

### Quick Start - Full Pipeline

Process all modules (books, lyrics, poems):
```bash
python dominican-eater.py pipeline --type all
```

Process individual module:
```bash
python dominican-eater.py pipeline --type lyrics
```

### Individual Commands

**1. Scrape Content Metadata**
```bash
python dominican-eater.py scrape --type all
python dominican-eater.py scrape --type books
```

**2. Download Audio**
```bash
python dominican-eater.py download --type lyrics-eater
python dominican-eater.py download --type all --force
```

**3. Transcribe Audio**
```bash
python dominican-eater.py transcribe --type books-eater --model base
python dominican-eater.py transcribe --type all --model large
```

Available Whisper models: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`

**4. Align Text (Lyrics Only)**
```bash
python dominican-eater.py align --type lyrics-eater
```

**5. Validate Outputs**
```bash
python dominican-eater.py validate --type all
```

### Pipeline Options

```bash
# Skip steps in pipeline
python dominican-eater.py pipeline --type lyrics --skip-scrape --skip-download

# Force re-processing
python dominican-eater.py pipeline --type all --force

# Use specific Whisper model
python dominican-eater.py pipeline --type books --model large-v3
```

## Modules

### Books Eater
Scrapes Dominican literature audiobooks from YouTube using a curated list of authors and titles.

**Configuration**: `books-eater/src/utils/dominican_books.py`

**Output**: `audio_processing/data/dominican_audiobooks.xlsx`

### Lyrics Eater
Fetches song lyrics from Genius API and finds corresponding YouTube audio.

**Configuration**: 
- Create `lyrics-eater/searches.txt` with artist names (one per line)
- Add Genius API token to `lyrics-eater/.env`

**Output**: `audio_processing/data/dominican_songs.xlsx`

### Poems Eater
Searches for Dominican poetry recitations on YouTube.

**Configuration**: `poems-eater/src/utils/dominican_poems.py`

**Output**: `audio_processing/data/dominican_poems.xlsx`

## Output Structure

After running the pipeline, outputs are organized as:

```
audio_processing/
├── audio/
│   ├── books/          # Downloaded audiobook files (.m4a)
│   ├── lyrics/         # Downloaded song files (.m4a)
│   └── poems/          # Downloaded poetry files (.m4a)
├── transcriptions/
│   ├── books/          # Whisper transcriptions (.json)
│   ├── lyrics/         # Whisper transcriptions (.json)
│   └── poems/          # Whisper transcriptions (.json)
├── reference_texts/
│   └── lyrics/         # Reference lyrics for alignment (.txt)
├── reports/
│   ├── books/          # Processing reports
│   ├── lyrics/         # Processing + alignment reports
│   └── poems/          # Processing reports
└── data/
    ├── dominican_audiobooks.xlsx
    ├── dominican_songs.xlsx
    └── dominican_poems.xlsx
```

## Configuration

Main configuration file: `audio_processing/config.yaml`

```yaml
modules:
  books-eater:
    audio_dir: audio_processing/audio/books
    transcriptions_dir: audio_processing/transcriptions/books
    reports_dir: audio_processing/reports/books
  
  lyrics-eater:
    audio_dir: audio_processing/audio/lyrics
    transcriptions_dir: audio_processing/transcriptions/lyrics
    reference_texts_dir: audio_processing/reference_texts/lyrics
    reports_dir: audio_processing/reports/lyrics
  
  poems-eater:
    audio_dir: audio_processing/audio/poems
    transcriptions_dir: audio_processing/transcriptions/poems
    reports_dir: audio_processing/reports/poems
```

## Text Alignment (Lyrics)

The lyrics module includes advanced alignment features:

- **WER Calculation**: Measures transcription accuracy against reference lyrics
- **Intro Detection**: Automatically detects and skips video intros/dialogue
- **Substring Matching**: Finds where lyrics begin in the transcription
- **Quality Metrics**: Generates detailed alignment reports

Average WER for Dominican Spanish lyrics: ~1.5 (98.4% success rate)

## Reports

Each module generates JSON reports tracking:
- Download success/failure rates
- Transcription quality metrics
- Alignment scores (lyrics only)
- Processing timestamps
- Error logs

View reports: `audio_processing/reports/[module]/`

## Troubleshooting

**FFmpeg not found**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Genius API errors (lyrics module)**
- Verify token in `lyrics-eater/.env`
- Get token at https://genius.com/api-clients

**Out of memory during transcription**
- Use smaller Whisper model: `--model tiny` or `--model base`
- Process fewer files at once

**YouTube download failures**
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Some videos may be region-locked or removed

## Development

### Code Style
- Python 3.8+ type hints
- 4-space indentation
- Max 100 character line length
- Use `logging` module for debug output

### Adding New Content

**Add books**: Edit `books-eater/src/utils/dominican_books.py`

**Add poems**: Edit `poems-eater/src/utils/dominican_poems.py`

**Add songs**: Add artist names to `lyrics-eater/searches.txt`

## License

This project is developed for academic research in Dominican Spanish NLP.

## Acknowledgments

- **FONDOCYT**: Funding support for Dominican LLM research
- **Whisper**: OpenAI's speech recognition model
- **yt-dlp**: YouTube download tool
- **Genius API**: Lyrics database

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.
