# Testing Suite - Dominican Spanish STT/TTS Models

## 1. Introduction

This testing suite evaluates **4 Speech-to-Text (STT)** models and **3 Text-to-Speech (TTS)** models for Dominican Spanish audio processing. Use this suite to transcribe Dominican Spanish audio files into text with timestamps (STT) or synthesize Spanish speech from text (TTS).

**Models included:**
- **STT:** Whisper, Parakeet TDT, Canary 1B Flash, SeamlessM4T v2
- **TTS:** XTTS-v2, Qwen3-TTS, Chatterbox

---

## 2. Server Setup & Dependencies

### System Requirements

- **OS:** Linux, macOS, or Windows with WSL
- **Python:** 3.8+ (3.11 required for NeMo models: Parakeet, Canary)
- **GPU:** Optional but recommended (NVIDIA GPU with 4-8GB VRAM)
- **Disk:** 10-20GB free space for model downloads
- **RAM:** 8GB minimum, 16GB+ recommended

### System Dependencies

Install FFmpeg and build tools (required for audio processing):

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg python3-dev build-essential python3.11 python3.11-venv

# macOS
brew install ffmpeg python@3.11

# Verify installation
ffmpeg -version
python3 --version
python3.11 --version
```

### Installing Model Dependencies

#### Option A: Automated Installation (Recommended)

The shell scripts (`run_all_stt.sh`, `run_all_tts.sh`) automatically create isolated virtual environments for each model and install dependencies. **No manual setup required.**

```bash
cd testing/stt
./run_all_stt.sh --max 3  # Creates venvs and runs all STT models

cd testing/tts
./run_all_tts.sh --text "Hola" --language es --speaker-wav ref.wav  # Creates venvs and runs all TTS models
```

#### Option B: Manual Installation

If you prefer to install dependencies manually for individual models:

**STT Models:**

```bash
cd testing/stt

# Whisper
python3 -m venv .venvs/.venv-whisper
source .venvs/.venv-whisper/bin/activate
pip install openai-whisper torch tqdm
deactivate

# Parakeet & Canary (Python 3.11 required)
python3.11 -m venv .venvs/.venv-nemo
source .venvs/.venv-nemo/bin/activate
pip install nemo-toolkit[asr] torch tqdm
deactivate

# SeamlessM4T v2
python3 -m venv .venvs/.venv-seamless
source .venvs/.venv-seamless/bin/activate
pip install transformers torch torchaudio scipy tqdm
deactivate
```

**TTS Models:**

```bash
cd testing/tts

# XTTS-v2
python3 -m venv .venvs/.venv-xtts
source .venvs/.venv-xtts/bin/activate
pip install TTS torch transformers==4.35.2
deactivate

# Qwen3-TTS
python3 -m venv .venvs/.venv-qwen
source .venvs/.venv-qwen/bin/activate
pip install qwen-tts torch transformers==4.57.3 soundfile
deactivate

# Chatterbox
python3 -m venv .venvs/.venv-chatterbox
source .venvs/.venv-chatterbox/bin/activate
pip install chatterbox-tts torch torchaudio transformers==4.46.3
deactivate
```

### CUDA Setup (Optional - for GPU Acceleration)

Check if CUDA is available and install PyTorch with GPU support:

```bash
# Check NVIDIA driver
nvidia-smi

# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Environment Variables

For XTTS-v2, you must accept the Coqui Public Model License:

```bash
export COQUI_TOS_AGREED=1
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent.

---

## 3. Running Speech-to-Text Models

All STT models process audio files and generate JSON transcripts with timestamps.

**Default input directory:** `lyrics-eater/audio/`  
**Default output directory:** `testing/stt/transcriptions/{model_name}/`

### Whisper (`whisper_model.py`)

**What it does:** General-purpose transcription with multiple model sizes.

**Command:**
```bash
cd testing/stt
python whisper_model.py --audio-dir ../../lyrics-eater/audio --max 5 --model base
```

**Key Arguments:**
- `--audio-dir`: Path to audio directory (default: `../../lyrics-eater/audio`)
- `--max`: Number of files to transcribe (default: 10)
- `--model`: Model size: `tiny`, `base`, `small`, `medium`, `large` (default: `base`)
- `--device`: Force `cpu` or `cuda` (auto-detected if not specified)
- `--out-dir`: Output directory (default: `transcriptions/whisper/`)

**Output:** `testing/stt/transcriptions/whisper/{filename}.json`

---

### Parakeet TDT 0.6B (`parakeet.py`)

**What it does:** High-accuracy Spanish transcription with word-level timestamps. Python 3.11 required.

**Command:**
```bash
cd testing/stt
python parakeet.py --audio-dir ../../lyrics-eater/audio --max 5
```

**Key Arguments:**
- `--audio-dir`: Path to audio directory (default: `../../lyrics-eater/audio`)
- `--max`: Number of files to transcribe (default: 10)
- `--out-dir`: Output directory (default: `transcriptions/parakeet/`)

**Output:** `testing/stt/transcriptions/parakeet/{filename}.json`

---

### Canary 1B Flash (`canary.py`)

**What it does:** Transcription AND translation (es→en). Python 3.11 required.

**Command (Transcription mode):**
```bash
cd testing/stt
python canary.py --audio-dir ../../lyrics-eater/audio --max 5 --source-lang es
```

**Command (Translation mode - es→en):**
```bash
python canary.py --audio-dir ../../lyrics-eater/audio --max 5 --source-lang es --target-lang en
```

**Key Arguments:**
- `--audio-dir`: Path to audio directory (default: `../../lyrics-eater/audio`)
- `--max`: Number of files to process (default: 10)
- `--source-lang`: Source language code: `es`, `en`, `de`, `fr` (default: `es`)
- `--target-lang`: Target language for translation: `es`, `en`, `de`, `fr` (optional, for AST mode)
- `--out-dir`: Output directory (default: `transcriptions/canary/`)

**Output:** `testing/stt/transcriptions/canary/{filename}.json`

---

### SeamlessM4T v2 (`seamless_m4t.py`)

**What it does:** Speech-to-speech translation (generates translated audio files).

**Command:**
```bash
cd testing/stt
python seamless_m4t.py --audio-dir ../../lyrics-eater/audio --max 3 --tgt-lang eng
```

**Key Arguments:**
- `--audio-dir`: Path to audio directory (default: `../../lyrics-eater/audio`)
- `--max`: Number of files to process (default: 10)
- `--tgt-lang`: Target language code: `eng`, `spa`, `rus`, `fra`, etc. (required)
- `--device`: Force `cpu` or `cuda` (auto-detected if not specified)
- `--out-dir`: Output directory (default: `transcriptions/seamless_m4t/`)

**Output:** 
- `testing/stt/transcriptions/seamless_m4t/{filename}_to_{lang}.wav` (translated audio)
- `testing/stt/transcriptions/seamless_m4t/{filename}_to_{lang}.json` (metadata)

---

### Run All STT Models (Batch Processing)

Process multiple audio files with all STT models at once:

```bash
cd testing/stt

# Process first 3 files with all models
./run_all_stt.sh --max 3

# Process 10 files with custom audio directory
./run_all_stt.sh --max 10 --audio-dir /path/to/audio
```

The script creates virtual environments automatically and runs all 4 models sequentially.

---

## 4. Running Text-to-Speech Models

All TTS models generate WAV audio files from input text.

**Default output directory:** `testing/tts/outputs/`

### XTTS-v2 (`xtts_v2.py`)

**What it does:** Voice cloning from a 6-30 second reference audio (required).

**Command:**
```bash
cd testing/tts
python xtts_v2.py --text "Te regalo una rosa" --language es --speaker-wav reference.wav --agree-license --output outputs/xtts_test.wav
```

**Key Arguments:**
- `--text`: Text to synthesize (required)
- `--language`: Language code: `es`, `en`, `fr`, `de`, etc. (default: `es`)
- `--speaker-wav`: Path to reference audio for voice cloning (REQUIRED, 6-30 seconds)
- `--agree-license`: Accept Coqui Public Model License (required on first use)
- `--output`: Output WAV file path (default: `output_xtts_v2.wav`)
- `--device`: Force `cpu` or `cuda` (auto-detected if not specified)

**Output:** WAV file at specified path (24kHz sampling rate)

**Note:** XTTS-v2 requires a speaker reference audio file. Extract a clean 6-30 second clip:
```bash
ffmpeg -i input.mp3 -ss 00:00:10 -t 00:00:15 -ar 22050 reference.wav
```

---

### Qwen3-TTS (`qwen3_tts.py`)

**What it does:** High-quality TTS with 9 built-in premium speakers (no reference audio required).

**Command:**
```bash
cd testing/tts
python qwen3_tts.py --text "Te regalo una rosa" --language es --speaker Ryan --output outputs/qwen_test.wav
```

**Key Arguments:**
- `--text`: Text to synthesize (required)
- `--language`: Language code: `es`, `en`, `zh`, etc. (default: `es`)
- `--speaker`: Speaker name: `Ryan`, `Aiden`, `Vivian`, `Sarah`, `Elijah`, `Sophia`, `Liam`, `Olivia`, `Noah` (default: `Ryan`)
- `--output`: Output WAV file path (default: `output_qwen3_tts.wav`)
- `--device`: Force `cpu` or `cuda` (auto-detected if not specified)

**Output:** WAV file at specified path (12kHz sampling rate)

---

### Chatterbox (`chatterbox_tts.py`)

**What it does:** Expressive TTS with emotion control and optional zero-shot voice cloning.

**Command (Basic usage):**
```bash
cd testing/tts
python chatterbox_tts.py --text "Te regalo una rosa" --language es --output outputs/chatterbox_test.wav
```

**Command (With voice cloning and emotion control):**
```bash
python chatterbox_tts.py --text "Te regalo una rosa" --language es --voice-prompt speaker.wav --exaggeration 0.7 --output outputs/chatterbox_clone.wav
```

**Key Arguments:**
- `--text`: Text to synthesize (required)
- `--language`: Language code: `es`, `en`, `fr`, `de`, etc. (default: `es`)
- `--output`: Output WAV file path (default: `output_chatterbox.wav`)
- `--voice-prompt`: Path to reference audio for zero-shot voice cloning (optional)
- `--exaggeration`: Emotion intensity (0.0-1.0, default: 0.5)
- `--device`: Force `cpu` or `cuda` (auto-detected if not specified)

**Output:** WAV file at specified path (24kHz sampling rate)

---

### Run All TTS Models (Batch Processing)

Generate audio with all 3 TTS models at once:

```bash
cd testing/tts

# Basic usage (XTTS requires speaker-wav)
./run_all_tts.sh --text "Hola, soy dominicano" --language es --speaker-wav reference.wav

# With custom output directory
./run_all_tts.sh --text "Test" --language es --speaker-wav ref.wav --output-dir outputs/
```

The script creates virtual environments automatically and runs all 3 models sequentially.

**Output files:**
- `outputs/chatterbox.wav`
- `outputs/qwen.wav`
- `outputs/xtts.wav`

---

## 5. Understanding the Outputs

### STT Output Format

All STT models generate JSON files with transcription and metadata.

**File location:** `testing/stt/transcriptions/{model_name}/{filename}.json`

**Example JSON structure:**
```json
{
  "file": "lyrics-eater_001_Romeo_Santos_Propuesta_Indecente.m4a",
  "transcript": "Hola me llaman Romeo es un placer conocerla...",
  "language": "es",
  "segments": [
    {
      "start": 38.0,
      "end": 39.2,
      "text": "Hola"
    },
    {
      "start": 39.2,
      "end": 40.5,
      "text": "me llaman Romeo"
    }
  ]
}
```

**Key fields:**
- `file`: Original audio filename
- `transcript`: Full transcription text
- `language`: Detected or specified language code
- `segments`: Time-aligned text segments (sentence-level)
- `timestamp`: Word-level timestamps (model-dependent)

**SeamlessM4T** also generates translated audio files:
- `{filename}_to_eng.wav`: Translated audio in target language
- `{filename}_to_eng.json`: Metadata about translation

---

### TTS Output Format

All TTS models generate WAV audio files.

**File location:** `testing/tts/outputs/{model_name}.wav` or custom `--output` path

**Audio specifications:**
- **XTTS-v2:** 24kHz sample rate, mono, WAV format
- **Qwen3-TTS:** 12kHz sample rate, mono, WAV format
- **Chatterbox:** 24kHz sample rate, mono, WAV format

**Playing audio files:**
```bash
# Linux
ffplay outputs/chatterbox.wav

# macOS
afplay outputs/chatterbox.wav

# Convert to MP3
ffmpeg -i outputs/chatterbox.wav outputs/chatterbox.mp3
```

---

### Expected Processing Results

**STT Models:**
- **Whisper (base):** ~1-2x realtime on GPU, ~5-10x slower on CPU
- **Parakeet TDT:** ~10-100x realtime on GPU (very fast)
- **Canary 1B:** ~10-100x realtime on GPU
- **SeamlessM4T:** Slower (large model), ~0.5-1x realtime on GPU

**TTS Models:**
- **XTTS-v2:** ~2-5 seconds per sentence on GPU
- **Qwen3-TTS:** ~3-8 seconds per sentence on GPU/CPU
- **Chatterbox:** ~2-5 seconds per sentence on GPU

**Disk usage:**
- Each STT model: 500MB - 3GB
- Each TTS model: 1GB - 5GB
- Output JSON files: ~1-10KB per audio file
- Output WAV files: ~100KB - 2MB per synthesis

---

## 6. Data Requirements

### Input Audio Files (for STT)

**Location:** Place audio files in `lyrics-eater/audio/` or specify custom path with `--audio-dir`

**Supported formats:** `.m4a` (primary), `.wav`, `.mp3`, `.flac`

**Content:** Dominican Spanish audio (music, speech, lyrics)

**Example structure:**
```
lyrics-eater/audio/
├── lyrics-eater_001_Romeo_Santos_Propuesta_Indecente.m4a
├── lyrics-eater_002_Juan_Luis_Guerra_Bachata_Rosa.m4a
└── ...
```

---

### Reference Audio Files (for TTS Voice Cloning)

**For XTTS-v2 (REQUIRED):** 6-30 second clean speech sample

**For Chatterbox/Qwen3 (OPTIONAL):** 3-10 second reference for zero-shot cloning

**Format:** `.wav` recommended (22050 Hz or 16000 Hz)

**Quality requirements:**
- Single speaker only
- Clear speech, no background noise
- No music or overlapping voices

**Creating reference audio:**
```bash
# Extract 15-second clip starting at 10 seconds
ffmpeg -i input.mp3 -ss 00:00:10 -t 00:00:15 -ar 22050 -ac 1 reference.wav
```

---

## 7. Quick Reference Commands

### STT (Speech-to-Text)

```bash
cd testing/stt

# Run all models (automated)
./run_all_stt.sh --max 5

# Individual models
python whisper_model.py --audio-dir ../../lyrics-eater/audio --max 5 --model base
python parakeet.py --audio-dir ../../lyrics-eater/audio --max 5
python canary.py --audio-dir ../../lyrics-eater/audio --max 5 --source-lang es
python seamless_m4t.py --audio-dir ../../lyrics-eater/audio --max 3 --tgt-lang eng
```

### TTS (Text-to-Speech)

```bash
cd testing/tts

# Run all models (automated)
./run_all_tts.sh --text "Hola mundo" --language es --speaker-wav ref.wav

# Individual models
python chatterbox_tts.py --text "Te regalo una rosa" --language es --output outputs/test.wav
python qwen3_tts.py --text "Te regalo una rosa" --language es --speaker Ryan --output outputs/test.wav
python xtts_v2.py --text "Te regalo una rosa" --language es --speaker-wav ref.wav --agree-license --output outputs/test.wav
```

### Check Outputs

```bash
# STT transcriptions
ls testing/stt/transcriptions/whisper/
ls testing/stt/transcriptions/parakeet/
ls testing/stt/transcriptions/canary/
ls testing/stt/transcriptions/seamless_m4t/

# TTS audio files
ls testing/tts/outputs/
```
