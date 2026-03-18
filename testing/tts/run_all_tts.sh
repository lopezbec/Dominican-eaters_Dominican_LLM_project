#!/bin/bash
set -euo pipefail

DEFAULT_TEXT="Te regalo una rosa, la encontré en el camino No sé si está desnuda o tiene un solo vestido No, no lo sé Si la riega el verano o se embriaga de olvido Si alguna vez fue amada o tiene amores"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BASE="$SCRIPT_DIR/.venvs"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
REQ_CHATTERBOX="$SCRIPT_DIR/requirements-chatterbox.txt"
REQ_QWEN="$SCRIPT_DIR/requirements-qwen.txt"
REQ_XTTS="$SCRIPT_DIR/requirements-xtts.txt"

TEXT="$DEFAULT_TEXT"
LANGUAGE="es"
SPEAKER_WAV=""
QWEN_SPEAKER="Ryan"
MANIFEST_PATH=""
TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required but not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --text) TEXT="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --speaker-wav) SPEAKER_WAV="$2"; shift 2 ;;
        --qwen-speaker) QWEN_SPEAKER="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --torch-backend) TORCH_BACKEND="$2"; shift 2 ;;
        --help)
            echo "Usage: ./run_all_tts.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --text TEXT          Text to synthesize"
            echo "  --language CODE      Language code for XTTS/Chatterbox (default: es)"
            echo "  --speaker-wav FILE   Reference audio for XTTS voice cloning"
            echo "  --qwen-speaker NAME  Qwen speaker (default: Ryan)"
            echo "  --output-dir DIR     Output directory (default: testing/tts/outputs)"
            echo "  --torch-backend B    Torch backend for dependency install: auto|cpu|cu128..."
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

MANIFEST_PATH="$OUTPUT_DIR/run_manifest.json"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$VENV_BASE"

run_model() {
    local name=$1
    local venv="$VENV_BASE/.venv-$name"
    local output="$OUTPUT_DIR/${name}.wav"
    
    echo "Running $name..."
    
    if [ ! -x "$venv/bin/python" ]; then
        uv venv --python 3.12 "$venv"
    fi
    
    case $name in
        chatterbox)
            UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$venv/bin/python" -r "$REQ_CHATTERBOX"
            uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/chatterbox_tts.py" --text "$TEXT" --language "$LANGUAGE" --output "$output"
            ;;
        qwen)
            UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$venv/bin/python" -r "$REQ_QWEN"
            local qwen_lang="Spanish"
            [ "$LANGUAGE" = "en" ] && qwen_lang="English"
            uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/qwen3_tts.py" --text "$TEXT" --language "$qwen_lang" --speaker "$QWEN_SPEAKER" --output "$output"
            ;;
        xtts)
            UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$venv/bin/python" -r "$REQ_XTTS"
            if [ -n "$SPEAKER_WAV" ] && [ -f "$SPEAKER_WAV" ]; then
                uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/xtts_v2.py" --text "$TEXT" --language "$LANGUAGE" --speaker-wav "$SPEAKER_WAV" --agree-license --output "$output"
            else
                echo "XTTS-v2 skipped: provide --speaker-wav"
                return 0
            fi
            ;;
    esac
    
    if [ -f "$output" ]; then
        echo "Created: $output ($(du -h "$output" | cut -f1))"
    fi
}

echo "Text: $TEXT"
echo ""

run_model "chatterbox"
echo ""
run_model "qwen"
echo ""
run_model "xtts"

echo ""
echo "Done. Outputs in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.wav 2>/dev/null || echo "No files generated"

TEXT_ENV="$TEXT" LANGUAGE_ENV="$LANGUAGE" QWEN_SPEAKER_ENV="$QWEN_SPEAKER" SPEAKER_WAV_ENV="$SPEAKER_WAV" OUTPUT_DIR_ENV="$OUTPUT_DIR" uv run --no-project --python "$VENV_BASE/.venv-qwen/bin/python" - <<PY > "$MANIFEST_PATH"
import json
import os

payload = {
    "text": os.environ["TEXT_ENV"],
    "language": os.environ["LANGUAGE_ENV"],
    "qwen_speaker": os.environ["QWEN_SPEAKER_ENV"],
    "speaker_wav": os.environ["SPEAKER_WAV_ENV"],
    "outputs": {
        "chatterbox": f"{os.environ['OUTPUT_DIR_ENV']}/chatterbox.wav",
        "qwen": f"{os.environ['OUTPUT_DIR_ENV']}/qwen.wav",
        "xtts": f"{os.environ['OUTPUT_DIR_ENV']}/xtts.wav",
    },
}

print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "Wrote run manifest: $MANIFEST_PATH"
