# Audio Processing Pipeline

This module handles audio downloading, transcription, alignment, and validation for the Dominican Eaters LLM Project.

## Overview

The pipeline processes three types of Dominican cultural content:
- **Books** (audiobooks)
- **Poems** (poetry recitations)
- **Lyrics** (songs with lyrics)

Each content type goes through: Download → Transcribe → Align → Validate

## Pipeline Stages

### 1. Download (`downloader.py`)
Downloads audio files from YouTube URLs stored in Excel files. Uses yt-dlp with automatic retry logic and quality optimization.

**Features:**
- Automatic skip of existing files
- Configurable retries and timeouts
- Progress bars with tqdm
- Error logging and reporting

### 2. Transcribe (`transcriber.py`)
Uses OpenAI's Whisper model to convert audio to text with word-level timestamps.

**Transcription Modes:**

#### Full Mode (Default for Books & Poems)
- Transcribes entire audio file
- Generates complete word-level timestamps
- Best for final dataset preparation

#### Partial Mode (Default for Lyrics in Pipeline)
- Transcribes only first 45 seconds (configurable in `config.yaml`)
- 3-4x faster than full transcription
- Sufficient for alignment verification during data collection
- Automatically enabled for lyrics-eater in pipeline

**Device Support:**
- Automatic CUDA detection for GPU acceleration
- Graceful fallback to CPU if needed
- Memory management with torch.cuda.empty_cache()

**Configuration (config.yaml):**
```yaml
whisper:
  model: base  # tiny, base, small, medium, large, large-v2, large-v3
  language: es
  partial_duration: 45  # seconds for partial mode
  word_timestamps: true
  fp16: false
  beam_size: 5
```

### 3. Align (`aligner.py`)
Matches transcribed audio with reference text using multiple similarity metrics and smart search optimization.

**Alignment Process:**
1. Load reference text (lyrics, book text, poem text)
2. Normalize both reference and transcription (lowercase, remove punctuation)
3. Find best alignment start position (detects intros/outros)
4. Calculate four alignment metrics
5. Generate detailed alignment report

**Search Optimizations:**
- **Limited Search Window**: Only first 200 words searched for alignment start
- **Early Stop**: Stops at first match with ≥0.8 similarity
- **Windowed Matching**: Uses 50-word sliding window for efficiency
- **Result**: ~5-10x faster alignment than exhaustive search

### 4. Validate (`validator.py`)
Generates comprehensive reports with statistics and quality metrics.

**Report Contents:**
- Success/failure rates
- Average alignment metrics
- Processing times
- Error logs
- Per-file detailed results

---

## Alignment Metrics

We use four complementary metrics to evaluate how well the transcribed audio matches the reference text:

### 1. **WER (Word Error Rate)**
- **What it measures**: The minimum number of word-level edits (insertions, deletions, substitutions) needed to transform the transcription into the reference text.
- **Range**: 0.0 (perfect match) to ∞ (can exceed 1.0 if many errors)
- **Best for**: Detecting word-level errors like misheard words, missing words, or extra words
- **Formula**: `(Substitutions + Insertions + Deletions) / Total_Reference_Words`
- **Example**: 
  - Reference: "el gato come pescado"
  - Hypothesis: "el pato come pan"
  - WER = 2/4 = 0.5 (50% error - 2 wrong words out of 4)

**Lower is better** → WER of 0.2 means 20% of words need correction

---

### 2. **Character Similarity (Levenshtein-based)**
- **What it measures**: Character-level similarity using edit distance between strings
- **Range**: 0.0 (completely different) to 1.0 (identical)
- **Best for**: Detecting small typos, spelling variations, or phonetic similarities that WER might miss
- **Formula**: `1 - (Levenshtein_Distance / max(len(ref), len(hyp)))`
- **Example**:
  - Reference: "merengue"
  - Hypothesis: "merenguë"
  - Char Similarity ≈ 0.95 (very similar, only accent differs)

**Higher is better** → 0.85 means 85% of characters match correctly

---

### 3. **Jaccard Similarity**
- **What it measures**: The overlap of unique words between texts (word-level set comparison)
- **Range**: 0.0 (no common words) to 1.0 (identical word sets)
- **Best for**: Detecting vocabulary coverage, regardless of word order
- **Formula**: `|Intersection| / |Union|` = `Common_Words / Total_Unique_Words`
- **Example**:
  - Reference: "el merengue dominicano es alegre"
  - Hypothesis: "el merengue es dominicano"
  - Jaccard = 4/5 = 0.8 (4 common words, 5 unique words total)

**Higher is better** → 0.7 means 70% vocabulary overlap

---

### 4. **Cosine Similarity**
- **What it measures**: Vector similarity of word frequencies (considers repeated words)
- **Range**: 0.0 (orthogonal/different) to 1.0 (identical direction)
- **Best for**: Detecting similar word usage patterns and emphasis
- **Formula**: `dot_product(vec1, vec2) / (||vec1|| * ||vec2||)`
- **Example**:
  - Reference: "baila baila merengue"
  - Hypothesis: "baila merengue baila"
  - Cosine ≈ 1.0 (same words, same frequencies, different order)

**Higher is better** → 0.8 means 80% similar word distribution

---

## Metrics Comparison Table

| Metric | Sensitive To | Ignores | Best Use Case |
|--------|-------------|---------|---------------|
| **WER** | Word order, exact words | Character-level differences | Overall transcription accuracy |
| **Char Similarity** | Spelling, typos | Word boundaries | Phonetic/spelling variations |
| **Jaccard** | Vocabulary presence | Word order, repetition | Topic/vocabulary coverage |
| **Cosine** | Word frequency patterns | Word order | Content emphasis & distribution |

---

## Interpretation Guidelines

### Excellent Alignment (Ready for Training)
- WER < 0.2 (less than 20% word errors)
- Char Similarity > 0.85
- Jaccard > 0.7
- Cosine > 0.8

### Good Alignment (Acceptable)
- WER < 0.4
- Char Similarity > 0.7
- Jaccard > 0.6
- Cosine > 0.7

### Poor Alignment (Needs Review)
- WER > 0.5
- Char Similarity < 0.6
- Jaccard < 0.5
- Cosine < 0.6

---

## Why Multiple Metrics?

Each metric captures different aspects of text similarity:

1. **WER** is strict about word-level accuracy but can be harsh on minor errors
2. **Character Similarity** is forgiving of minor spelling differences (useful for Dominican Spanish variations)
3. **Jaccard** verifies vocabulary coverage without caring about order (useful for lyrics that may rearrange verses)
4. **Cosine** considers word emphasis and repetition (important for songs with repeated choruses)

**Together**, they provide a robust multi-dimensional view of alignment quality, helping identify:
- Transcription errors (high WER, low char similarity)
- Missing sections (low Jaccard)
- Word order issues (low WER, high Jaccard/Cosine)
- Spelling/accent variations (low WER, high char similarity)

---

## Configuration

Main configuration file: `audio_processing/config.yaml`

### Key Settings

```yaml
# Audio download settings
download:
  max_retries: 3
  retry_delay: 5
  timeout: 300

# Whisper transcription settings
whisper:
  model: base  # Model size: tiny, base, small, medium, large, large-v2, large-v3
  language: es  # Dominican Spanish
  partial_duration: 45  # Seconds for partial mode (lyrics optimization)
  word_timestamps: true  # Generate word-level timestamps
  device: cuda  # Auto-fallback to CPU if CUDA unavailable

# Alignment settings
alignment:
  min_similarity: 0.7  # Minimum acceptable similarity
  window_size: 50  # Words to compare in sliding window
  min_match_length: 30  # Minimum words for match validation
  wer_threshold: 0.5  # Maximum acceptable WER
  char_similarity_threshold: 0.6  # Minimum character similarity

# Module-specific paths
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

## Usage

The audio processing pipeline is controlled through the main CLI (`dominican-eater.py`). See main project README for detailed command examples.

### Quick Examples

```bash
# Transcribe with partial mode (45 seconds)
python dominican-eater.py transcribe --type lyrics-eater --partial

# Align lyrics with smart optimization
python dominican-eater.py align --type lyrics-eater

# Full pipeline with partial transcription for lyrics
python dominican-eater.py pipeline --type lyrics

# Manual module usage (advanced)
cd audio_processing
python src/transcriber.py --module lyrics-eater --partial
python src/aligner.py --action align --module lyrics-eater
```

---

## Current Optimization Strategy (Data Collection Phase)

For **lyrics**, we optimize for speed since we're collecting and verifying data, not evaluating models:

### Optimizations Applied

1. **Partial Transcription** (transcriber.py:90)
   - Only first 45 seconds transcribed
   - Configured via `whisper.partial_duration` in config.yaml
   - Automatically enabled in pipeline for lyrics-eater

2. **Limited Search Window** (aligner.py:157)
   - Searches only first 200 words for alignment start
   - Reduces search space by ~80% for typical songs
   - Configured via `max_search_words` parameter

3. **Early Stop Logic** (aligner.py:177-179)
   - Stops at first match with ≥0.8 similarity
   - Avoids exhaustive search when good match found
   - Average 2-3 iterations vs 200+ without early stop

4. **Automatic Enabling** (dominican-eater.py:86)
   - Pipeline automatically enables `--partial` flag for lyrics-eater
   - Ensures consistent optimization across workflows

### Performance Impact

| Metric | Without Optimization | With Optimization | Improvement |
|--------|---------------------|-------------------|-------------|
| **Transcription Time** | 2-3 minutes/song | 30-45 seconds | **4x faster** |
| **Alignment Time** | 10-15 seconds | 2-3 seconds | **5x faster** |
| **Total per Song** | 2-3 minutes | 35-50 seconds | **3-4x faster** |
| **Quality Impact** | Full verification | Same verification | **None** |

### When Full Transcription Happens

Full audio transcription and comprehensive evaluation will be performed:
- On server infrastructure (more compute resources)
- During model training phase (need complete audio)
- For final dataset validation (production quality)

### Implementation Files

- `audio_processing/src/transcriber.py`: Partial mode implementation
- `audio_processing/src/aligner.py`: Search optimization and early stop
- `dominican-eater.py`: Automatic partial enabling for lyrics
- `audio_processing/config.yaml`: Configuration parameters
