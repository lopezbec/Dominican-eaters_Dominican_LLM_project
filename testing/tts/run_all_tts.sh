#!/bin/bash
set -e

DEFAULT_TEXT="Te regalo una rosa, la encontré en el camino No sé si está desnuda o tiene un solo vestido No, no lo sé Si la riega el verano o se embriaga de olvido Si alguna vez fue amada o tiene amores"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BASE="$SCRIPT_DIR/.venvs"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

TEXT="$DEFAULT_TEXT"
LANGUAGE="es"
SPEAKER_WAV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --text) TEXT="$2"; shift 2 ;;
        --language) LANGUAGE="$2"; shift 2 ;;
        --speaker-wav) SPEAKER_WAV="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "$VENV_BASE"

run_model() {
    local name=$1
    local venv="$VENV_BASE/.venv-$name"
    local output="$OUTPUT_DIR/${name}.wav"
    
    echo "Running $name..."
    
    if [ ! -d "$venv" ]; then
        python3 -m venv "$venv"
    fi
    
    source "$venv/bin/activate"
    
    case $name in
        chatterbox)
            if ! python -c "import chatterbox" 2>/dev/null; then
                pip install -q torch torchaudio transformers==4.46.3 chatterbox-tts
            fi
            python "$SCRIPT_DIR/chatterbox_tts.py" --text "$TEXT" --language "$LANGUAGE" --output "$output" 2>&1 | tail -5
            ;;
        qwen)
            if ! python -c "import qwen_tts" 2>/dev/null; then
                pip install -q torch transformers==4.57.3 soundfile qwen-tts
            fi
            local qwen_lang="Spanish"
            [ "$LANGUAGE" = "en" ] && qwen_lang="English"
            python "$SCRIPT_DIR/qwen3_tts.py" --text "$TEXT" --language "$qwen_lang" --speaker Ryan --output "$output" 2>&1 | tail -5
            ;;
        xtts)
            if ! python -c "from TTS.api import TTS" 2>/dev/null; then
                pip install -q torch transformers==4.35.2 TTS
            fi
            if [ -n "$SPEAKER_WAV" ] && [ -f "$SPEAKER_WAV" ]; then
                python "$SCRIPT_DIR/xtts_v2.py" --text "$TEXT" --language "$LANGUAGE" --speaker-wav "$SPEAKER_WAV" --agree-license --output "$output" 2>&1 | tail -5
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
