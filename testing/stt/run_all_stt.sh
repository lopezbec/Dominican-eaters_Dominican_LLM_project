#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BASE="$SCRIPT_DIR/.venvs"
OUTPUT_BASE="$SCRIPT_DIR/transcriptions"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_AUDIO_DIR="$REPO_ROOT/lyrics-eater/audio"
REQ_WHISPER="$SCRIPT_DIR/requirements-whisper.txt"
REQ_NEMO="$SCRIPT_DIR/requirements-nemo.txt"
REQ_SEAMLESS="$SCRIPT_DIR/requirements-seamless.txt"

AUDIO_DIR="$DEFAULT_AUDIO_DIR"
MAX_FILES=3
WHISPER_MODEL="base"
INCLUDE_SEAMLESS=0
SEAMLESS_TGT_LANG="eng"
TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
NEMO_DEVICE="cpu"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required but not installed."
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
        --max) MAX_FILES="$2"; shift 2 ;;
        --whisper-model) WHISPER_MODEL="$2"; shift 2 ;;
        --include-seamless) INCLUDE_SEAMLESS=1; shift ;;
        --seamless-tgt-lang) SEAMLESS_TGT_LANG="$2"; shift 2 ;;
        --torch-backend) TORCH_BACKEND="$2"; shift 2 ;;
        --nemo-device) NEMO_DEVICE="$2"; shift 2 ;;
        --help) 
            echo "Usage: ./run_all_stt.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --audio-dir DIR    Audio directory (default: lyrics-eater/audio)"
            echo "  --max N           Max files to process (default: 3)"
            echo "  --whisper-model M Whisper model (default: base)"
            echo "  --include-seamless Run SeamlessM4T v2 (disabled by default)"
            echo "  --seamless-tgt-lang CODE Target language for SeamlessM4T (default: eng)"
            echo "  --torch-backend B Torch backend for dependency install: auto|cpu|cu128..."
            echo "  --nemo-device D Device for Parakeet/Canary: auto|cpu|cuda (default: cpu)"
            echo ""
            echo "This script uses uv only for venv and package management."
            exit 0
            ;;
        *) echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

if [ ! -d "$AUDIO_DIR" ]; then
    echo "Error: Audio directory not found: $AUDIO_DIR"
    exit 1
fi

mkdir -p "$VENV_BASE"
mkdir -p "$OUTPUT_BASE"

echo "Audio directory: $AUDIO_DIR"
echo "Max files: $MAX_FILES"
echo "Whisper model: $WHISPER_MODEL"
echo "Include seamless: $INCLUDE_SEAMLESS"
echo "Torch backend: $TORCH_BACKEND"
echo "NeMo device: $NEMO_DEVICE"
echo ""

ensure_venv() {
    local venv="$1"
    local py_version="$2"

    if [ ! -x "$venv/bin/python" ]; then
        uv venv --python "$py_version" "$venv"
    fi
}

sync_requirements() {
    local venv="$1"
    local req_file="$2"
    echo "Installing dependencies (backend: $TORCH_BACKEND)..."

    if UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --python "$venv/bin/python" -r "$req_file"; then
        return 0
    fi

    echo "Install failed (attempt 1/3). Cleaning cache and retrying..."
    uv cache prune || true
    uv cache clean numpy sympy torch || true
    sleep 2

    if UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --no-cache --python "$venv/bin/python" -r "$req_file"; then
        return 0
    fi

    echo "Install failed (attempt 2/3). Retrying with refresh flags..."
    sleep 4
    if UV_TORCH_BACKEND="$TORCH_BACKEND" uv pip install --refresh --refresh-package numpy --refresh-package torch --python "$venv/bin/python" -r "$req_file"; then
        return 0
    fi

    echo "Failed to install dependencies after 3 attempts."
    return 1
}

run_whisper() {
    local venv="$VENV_BASE/.venv-whisper"
    local output_dir="$OUTPUT_BASE/whisper"
    
    echo "=== Whisper ==="

    ensure_venv "$venv" "3.12"
    sync_requirements "$venv" "$REQ_WHISPER"
    
    mkdir -p "$output_dir"
    
    uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/whisper_model.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES" \
        --model "$WHISPER_MODEL"
}

run_parakeet() {
    local venv="$VENV_BASE/.venv-nemo"
    local output_dir="$OUTPUT_BASE/parakeet"
    
    echo "=== Parakeet ==="
    
    ensure_venv "$venv" "3.11"
    sync_requirements "$venv" "$REQ_NEMO"
    
    uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/parakeet.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES" \
        --device "$NEMO_DEVICE"
}

run_canary() {
    local venv="$VENV_BASE/.venv-nemo"
    local output_dir="$OUTPUT_BASE/canary"
    
    echo "=== Canary ==="
    
    ensure_venv "$venv" "3.11"
    sync_requirements "$venv" "$REQ_NEMO"
    
    uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/canary.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES" \
        --source-lang es \
        --device "$NEMO_DEVICE"
}

run_seamless() {
    local venv="$VENV_BASE/.venv-seamless"
    local output_dir="$OUTPUT_BASE/seamless_m4t"

    echo "=== SeamlessM4T v2 ==="

    ensure_venv "$venv" "3.12"
    sync_requirements "$venv" "$REQ_SEAMLESS"

    uv run --no-project --python "$venv/bin/python" "$SCRIPT_DIR/seamless_m4t.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES" \
        --tgt-lang "$SEAMLESS_TGT_LANG"
}

run_whisper
echo ""
run_parakeet
echo ""
run_canary
if [ "$INCLUDE_SEAMLESS" -eq 1 ]; then
    echo ""
    run_seamless
fi

echo ""
echo "=== Results ==="
for model_dir in "$OUTPUT_BASE"/*/; do
    model_name=$(basename "$model_dir")
    file_count=$(find "$model_dir" -name "*.json" -type f 2>/dev/null | wc -l || true)
    if [ "$file_count" -gt 0 ]; then
        echo "✓ $model_name: $file_count files"
        find "$model_dir" -name "*.json" -type f -printf "  %f (%s bytes)\n" 2>/dev/null
    else
        echo "✗ $model_name: No output"
    fi
done
