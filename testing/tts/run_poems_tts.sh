#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_BASE="$SCRIPT_DIR/.venvs"
VENV="$VENV_BASE/.venv-poems-benchmark"

POEMS_DIR="$PROJECT_ROOT/all-poems"
OUTPUT_DIR="$SCRIPT_DIR/outputs/poems_benchmark"
PROMPTS_JSONL="$OUTPUT_DIR/poems_prompts.jsonl"

MAX_PROMPTS=24
LANGUAGE="es"
DEVICE=""

XTTS_SPEAKER_WAV=""
XTTS_SPEAKER_WAVS=""
AGREE_LICENSE=0

RUN_F5=0
F5_REF_AUDIO=""
F5_REF_TEXT=""
F5_MODEL="F5TTS_v1_Base"
F5_SPEED="1.0"
F5_NFE_STEP="32"

RUN_KOKORO=0
KOKORO_VOICE="ef_dora"
KOKORO_SPEED="1.0"

TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but not installed."
  echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --poems-dir) POEMS_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; PROMPTS_JSONL="$2/poems_prompts.jsonl"; shift 2 ;;
    --max-prompts) MAX_PROMPTS="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;

    --xtts-speaker-wav) XTTS_SPEAKER_WAV="$2"; shift 2 ;;
    --xtts-speaker-wavs) XTTS_SPEAKER_WAVS="$2"; shift 2 ;;
    --agree-license) AGREE_LICENSE=1; shift ;;

    --run-f5) RUN_F5=1; shift ;;
    --f5-ref-audio) F5_REF_AUDIO="$2"; shift 2 ;;
    --f5-ref-text) F5_REF_TEXT="$2"; shift 2 ;;
    --f5-model) F5_MODEL="$2"; shift 2 ;;
    --f5-speed) F5_SPEED="$2"; shift 2 ;;
    --f5-nfe-step) F5_NFE_STEP="$2"; shift 2 ;;

    --run-kokoro) RUN_KOKORO=1; shift ;;
    --kokoro-voice) KOKORO_VOICE="$2"; shift 2 ;;
    --kokoro-speed) KOKORO_SPEED="$2"; shift 2 ;;

    --help)
      echo "Usage: ./run_poems_tts.sh [OPTIONS]"
      echo ""
      echo "Core options:"
      echo "  --poems-dir DIR            Path to all-poems directory"
      echo "  --output-dir DIR           Output directory"
      echo "  --max-prompts N            Number of prompts (default: 24)"
      echo "  --language CODE            Language code (default: es)"
      echo "  --device DEV               Device override (cpu/cuda/...)"
      echo ""
      echo "XTTS options:"
      echo "  --xtts-speaker-wav FILE    Single reference clip for XTTS"
      echo "  --xtts-speaker-wavs CSV    Multi-reference CSV for XTTS"
      echo "  --agree-license            Set COQUI_TOS_AGREED=1"
      echo ""
      echo "F5 options:"
      echo "  --run-f5                   Enable F5 condition"
      echo "  --f5-ref-audio FILE        F5 reference audio"
      echo "  --f5-ref-text TEXT         F5 reference transcript"
      echo "  --f5-model NAME            F5 model (default: F5TTS_v1_Base)"
      echo ""
      echo "Kokoro options:"
      echo "  --run-kokoro               Enable Kokoro condition"
      echo "  --kokoro-voice NAME        Kokoro voice id (default: ef_dora)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "$VENV_BASE"

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi

echo "Installing shared dependencies..."
UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$VENV/bin/python" -r "$SCRIPT_DIR/requirements-xtts.txt"

if [ "$RUN_F5" -eq 1 ]; then
  echo "Installing F5 dependencies..."
  UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$VENV/bin/python" -r "$SCRIPT_DIR/requirements-f5.txt"
fi

if [ "$RUN_KOKORO" -eq 1 ]; then
  echo "Installing Kokoro dependencies..."
  UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$VENV/bin/python" -r "$SCRIPT_DIR/requirements-kokoro.txt"
fi

echo "Building prompts from: $POEMS_DIR"
uv run --no-project --python "$VENV/bin/python" "$SCRIPT_DIR/poem_prompt_builder.py" \
  --poems-dir "$POEMS_DIR" \
  --output "$PROMPTS_JSONL" \
  --max-prompts "$MAX_PROMPTS"

RUN_CMD=(
  uv run --no-project --python "$VENV/bin/python" "$SCRIPT_DIR/run_poems_tts_benchmark.py"
  --prompts-jsonl "$PROMPTS_JSONL"
  --output-dir "$OUTPUT_DIR"
  --language "$LANGUAGE"
)

if [ -n "$DEVICE" ]; then
  RUN_CMD+=(--device "$DEVICE")
fi
if [ -n "$XTTS_SPEAKER_WAV" ]; then
  RUN_CMD+=(--xtts-speaker-wav "$XTTS_SPEAKER_WAV")
fi
if [ -n "$XTTS_SPEAKER_WAVS" ]; then
  RUN_CMD+=(--xtts-speaker-wavs "$XTTS_SPEAKER_WAVS")
fi
if [ "$AGREE_LICENSE" -eq 1 ]; then
  RUN_CMD+=(--agree-license)
fi

if [ "$RUN_F5" -eq 1 ]; then
  RUN_CMD+=(--run-f5 --f5-ref-audio "$F5_REF_AUDIO" --f5-ref-text "$F5_REF_TEXT" --f5-model "$F5_MODEL" --f5-speed "$F5_SPEED" --f5-nfe-step "$F5_NFE_STEP")
fi

if [ "$RUN_KOKORO" -eq 1 ]; then
  RUN_CMD+=(--run-kokoro --kokoro-voice "$KOKORO_VOICE" --kokoro-speed "$KOKORO_SPEED")
fi

echo "Running benchmark..."
"${RUN_CMD[@]}"

echo "Done. Manifest: $OUTPUT_DIR/run_manifest_poems.json"
