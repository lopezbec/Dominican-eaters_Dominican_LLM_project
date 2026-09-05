#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
REPO_NAME="$(basename "$ROOT_DIR")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/_handoff_zips"
STAGE_DIR="$OUT_DIR/${REPO_NAME}_review_${STAMP}_stage"
ZIP_PATH="$OUT_DIR/${REPO_NAME}_review_${STAMP}.zip"

mkdir -p "$OUT_DIR"

echo "Repo: $REPO_NAME"
echo "Stage: $STAGE_DIR"
echo "Zip: $ZIP_PATH"
echo

# Remove old bad zips only if you want:
# rm -f "$OUT_DIR"/*.zip

rsync -a --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.venvs/' \
  --exclude='venv/' \
  --exclude='venvs/' \
  --exclude='env/' \
  --exclude='envs/' \
  --exclude='virtualenv/' \
  --exclude='virtualenvs/' \
  --exclude='*/.venv/' \
  --exclude='*/.venvs/' \
  --exclude='*/venv/' \
  --exclude='*/venvs/' \
  --exclude='*/env/' \
  --exclude='*/envs/' \
  --exclude='*/virtualenv/' \
  --exclude='*/virtualenvs/' \
  --exclude='*/site-packages/' \
  --exclude='*/dist-packages/' \
  --exclude='__pycache__/' \
  --exclude='*/__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.so' \
  --exclude='*.dylib' \
  --exclude='*.dll' \
  --exclude='*.m4a' \
  --exclude='*.mp3' \
  --exclude='*.wav' \
  --exclude='*.mp4' \
  --exclude='*.mov' \
  --exclude='*.webm' \
  --exclude='*.zip' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.tgz' \
  --exclude='*.7z' \
  --exclude='CDD ACTAS PDFs/' \
  --exclude='models/' \
  --exclude='checkpoints/' \
  --exclude='outputs/' \
  --exclude='data/' \
  --exclude='datasets/' \
  --exclude='audio/' \
  --exclude='synthetic_audio/' \
  --exclude='testing/tts/outputs/' \
  --exclude='testing/stt/outputs/' \
  --exclude='testing/stt/transcriptions/' \
  --exclude='lyrics-eater/transcriptions/' \
  --exclude='books-eater/transcriptions/' \
  --exclude='poems-eater/transcriptions/' \
  --exclude='_handoff_zips/' \
  "$ROOT_DIR/" "$STAGE_DIR/"

echo "Checking for virtual environments or dependency folders..."
BAD_PATHS="$(find "$STAGE_DIR" \( \
  -name '.venv' -o \
  -name '.venvs' -o \
  -name 'venv' -o \
  -name 'venvs' -o \
  -name 'env' -o \
  -name 'envs' -o \
  -name 'site-packages' -o \
  -name 'dist-packages' \
\) -print | head -50)"

if [[ -n "$BAD_PATHS" ]]; then
  echo "ERROR: These unwanted dependency folders are still present:"
  echo "$BAD_PATHS"
  echo
  echo "Zip was not created. Add more excludes above."
  exit 1
fi

echo "Checking stage size..."
du -sh "$STAGE_DIR"

echo
echo "Creating zip..."
(
  cd "$OUT_DIR"
  zip -r -9 "$(basename "$ZIP_PATH")" "$(basename "$STAGE_DIR")" >/dev/null
)

echo
echo "Created:"
ls -lh "$ZIP_PATH"

echo
echo "Zip sanity check for venv/site-packages:"
if unzip -l "$ZIP_PATH" | grep -E '(\.venv|\.venvs|/venv/|/venvs/|site-packages|dist-packages)' | head; then
  echo "ERROR: zip still contains venv/dependency files."
  exit 1
else
  echo "OK: no venv or site-packages detected."
fi

echo
echo "Preview:"
unzip -l "$ZIP_PATH" | head -80

echo
echo "Done."
