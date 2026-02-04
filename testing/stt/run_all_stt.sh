#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BASE="$SCRIPT_DIR/.venvs"
OUTPUT_BASE="$SCRIPT_DIR/transcriptions"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_AUDIO_DIR="$REPO_ROOT/lyrics-eater/audio"

AUDIO_DIR="$DEFAULT_AUDIO_DIR"
MAX_FILES=3

# Detect Python 3.11 (prefer pyenv, then system)
if command -v pyenv &> /dev/null && pyenv versions --bare | grep -q "^3\.11"; then
    PYTHON311="$(pyenv root)/versions/$(pyenv versions --bare | grep '^3\.11' | head -1)/bin/python3"
elif command -v python3.11 &> /dev/null; then
    PYTHON311="python3.11"
else
    PYTHON311=""
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio-dir) AUDIO_DIR="$2"; shift 2 ;;
        --max) MAX_FILES="$2"; shift 2 ;;
        --help) 
            echo "Usage: ./run_all_stt.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --audio-dir DIR    Audio directory (default: lyrics-eater/audio)"
            echo "  --max N           Max files to process (default: 3)"
            echo ""
            echo "Python versions:"
            echo "  Default: $(python3 --version)"
            if [ -n "$PYTHON311" ]; then
                echo "  Python 3.11: $($PYTHON311 --version) (for NeMo models)"
            else
                echo "  Python 3.11: Not found (NeMo models will be skipped)"
            fi
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
echo ""

run_whisper() {
    local venv="$VENV_BASE/.venv-whisper"
    local output_dir="$OUTPUT_BASE/whisper"
    
    echo "=== Whisper ==="
    
    if [ ! -d "$venv" ]; then
        python3 -m venv "$venv"
    fi
    
    source "$venv/bin/activate"
    
    if ! python -c "import whisper" 2>/dev/null; then
        pip install -q torch openai-whisper
    fi
    
    mkdir -p "$output_dir"
    
    python "$SCRIPT_DIR/whisper_model.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES" \
        2>&1 | grep -E "(Saved|ERROR|Transcribing|Processing)" || true
}

run_parakeet() {
    local venv="$VENV_BASE/.venv-nemo"
    local output_dir="$OUTPUT_BASE/parakeet"
    
    echo "=== Parakeet ==="
    
    if [ -z "$PYTHON311" ]; then
        echo "⚠️  Python 3.11 not found - skipping (NeMo requires Python ≤3.13)"
        return 0
    fi
    
    if [ ! -d "$venv" ]; then
        echo "Creating Python 3.11 virtual environment..."
        "$PYTHON311" -m venv "$venv"
    fi
    
    source "$venv/bin/activate"
    
    if ! python -c "import nemo" 2>/dev/null; then
        echo "Installing NeMo (this may take several minutes)..."
        pip install -q torch nemo-toolkit[asr]
    fi
    
    python "$SCRIPT_DIR/parakeet.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES"
}

run_canary() {
    local venv="$VENV_BASE/.venv-nemo"
    local output_dir="$OUTPUT_BASE/canary"
    
    echo "=== Canary ==="
    
    if [ -z "$PYTHON311" ]; then
        echo "⚠️  Python 3.11 not found - skipping (NeMo requires Python ≤3.13)"
        return 0
    fi
    
    if [ ! -d "$venv" ]; then
        echo "Creating Python 3.11 virtual environment..."
        "$PYTHON311" -m venv "$venv"
    fi
    
    source "$venv/bin/activate"
    
    if ! python -c "import nemo" 2>/dev/null; then
        echo "Installing NeMo (this may take several minutes)..."
        pip install -q torch nemo-toolkit[asr]
    fi
    
    python "$SCRIPT_DIR/canary.py" \
        --audio-dir "$AUDIO_DIR" \
        --out-dir "$output_dir" \
        --max "$MAX_FILES" \
        --source-lang es
}

run_whisper
echo ""
run_parakeet
echo ""
run_canary

echo ""
echo "=== Results ==="
for model_dir in "$OUTPUT_BASE"/*/; do
    model_name=$(basename "$model_dir")
    file_count=$(find "$model_dir" -name "*.json" -type f 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        echo "✓ $model_name: $file_count files"
        find "$model_dir" -name "*.json" -type f -printf "  %f (%s bytes)\n" 2>/dev/null | head -5
    else
        echo "✗ $model_name: No output"
    fi
done
