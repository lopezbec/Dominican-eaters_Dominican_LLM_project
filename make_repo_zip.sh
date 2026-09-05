#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-review}"
ROOT_DIR="$(pwd)"
REPO_NAME="$(basename "$ROOT_DIR")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/_handoff_zips"
mkdir -p "$OUT_DIR"

ZIP_PATH="$OUT_DIR/${REPO_NAME}_${MODE}_${STAMP}.zip"

echo "Repo: $REPO_NAME"
echo "Mode: $MODE"
echo "Output: $ZIP_PATH"
echo

COMMON_EXCLUDES=(
  "*.git*"
  "*/.git/*"
  "*/.venv/*"
  "*/venv/*"
  "*/env/*"
  "*/__pycache__/*"
  "*.pyc"
  "*.pyo"
  ".pytest_cache/*"
  ".mypy_cache/*"
  ".ruff_cache/*"
  ".ipynb_checkpoints/*"
  ".DS_Store"
  "_handoff_zips/*"
)

REVIEW_EXCLUDES=(
  # Heavy raw media / archives
  "*.m4a"
  "*.mp3"
  "*.wav"
  "*.mp4"
  "*.zip"
  "*.tar"
  "*.tar.gz"
  "*.tgz"
  "*.7z"

  # Heavy PDFs / generated data
  "CDD ACTAS PDFs/*"
  "testing/tts/outputs/*"
  "testing/tts/logs/*"
  "testing/stt/transcriptions/*"

  # Model/checkpoint/data cache dirs
  "models/*"
  "checkpoints/*"
  "outputs/*"
  "data/*"
  "datasets/*"
  "synthetic_audio/*"
  "audio/*"

  # Optional: huge transcript dumps, keep reports/reference texts/code instead
  "lyrics-eater/transcriptions/*"
  "poems-eater/transcriptions/*"
  "books-eater/transcriptions/*"
)

FULL_EXCLUDES=(
  # Still exclude things that should never be shared
  "*.git*"
  "*/.git/*"
  "*/.venv/*"
  "*/venv/*"
  "*/env/*"
  "*/__pycache__/*"
  "*.pyc"
  "*.pyo"
  ".pytest_cache/*"
  ".mypy_cache/*"
  ".ruff_cache/*"
  ".ipynb_checkpoints/*"
  ".DS_Store"
  "_handoff_zips/*"
  "models/*"
  "checkpoints/*"
)

build_zip() {
  local -n excludes_ref=$1
  local args=()

  for pattern in "${excludes_ref[@]}"; do
    args+=("-x" "$pattern")
  done

  zip -r -9 "$ZIP_PATH" . "${args[@]}"
}

if [[ "$MODE" == "review" ]]; then
  EXCLUDES=("${COMMON_EXCLUDES[@]}" "${REVIEW_EXCLUDES[@]}")
  build_zip EXCLUDES
elif [[ "$MODE" == "full" ]]; then
  EXCLUDES=("${FULL_EXCLUDES[@]}")
  build_zip EXCLUDES
else
  echo "Unknown mode: $MODE"
  echo "Use: ./make_repo_zip.sh review"
  echo " or: ./make_repo_zip.sh full"
  exit 1
fi

echo
echo "Created:"
ls -lh "$ZIP_PATH"

echo
echo "Zip contents preview:"
unzip -l "$ZIP_PATH" | head -80

echo
echo "Done."
