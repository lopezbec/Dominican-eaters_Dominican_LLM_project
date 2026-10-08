#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

MAIN_PYTHON="${DOMINICAN_EATERS_PYTHON:-$ROOT/.venv/bin/python}"
MANIFEST="${1:-$ROOT/data/manifests/stt-all.json}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${STT_RUN_ROOT:-$ROOT/artifacts/stt/batch-$RUN_ID}"
SUMMARY="$RUN_ROOT/summary.tsv"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/batch.log") 2>&1
printf "preset\tstatus\toutput\n" > "$SUMMARY"

pick_worker() {
  local candidate
  for candidate in "$@"; do
    if [[ -x "$candidate" ]]; then
      realpath "$candidate"
      return 0
    fi
  done
  return 1
}

NEMO_PYTHON="${DOMINICAN_EATERS_NEMO_PYTHON:-$(
  pick_worker "$ROOT/.venvs/nemo/bin/python" "$ROOT/.venv-nemo/bin/python" || true
)}"
GRANITE_PYTHON="${DOMINICAN_EATERS_GRANITE_PYTHON:-$(
  pick_worker "$ROOT/.venvs/granite/bin/python" "$ROOT/.venv-granite/bin/python" || true
)}"
QWEN3_PYTHON="${DOMINICAN_EATERS_QWEN3_ASR_PYTHON:-$(
  pick_worker \
    "$ROOT/.venvs/qwen3-asr/bin/python" \
    "$ROOT/.venvs/qwen3_asr/bin/python" \
    "$ROOT/.venv-qwen3-asr/bin/python" || true
)}"
VOXTRAL_PYTHON="${DOMINICAN_EATERS_VOXTRAL_PYTHON:-$(
  pick_worker "$ROOT/.venvs/voxtral/bin/python" "$ROOT/.venv-voxtral/bin/python" || true
)}"

run_model() {
  local preset="$1"
  local worker_python="$2"
  local output="$RUN_ROOT/$preset"
  local log="$RUN_ROOT/$preset.log"
  local worker_args=()

  printf "\n%s\nSTARTING: %s\nOUTPUT: %s\n%s\n" \
    "============================================================" \
    "$preset" \
    "$output" \
    "============================================================"

  if [[ "$worker_python" == "REQUIRED" ]]; then
    echo "SKIPPED: worker environment was not found" | tee "$log"
    printf "%s\tskipped-missing-worker\t%s\n" "$preset" "$output" >> "$SUMMARY"
    return
  fi
  if [[ -n "$worker_python" ]]; then
    worker_args=(--worker-python "$worker_python")
  fi

  if ! "$MAIN_PYTHON" -m dominican_eaters stt preflight \
    "$MANIFEST" \
    --preset "$preset" \
    --device cuda \
    --precision fp16 \
    --verify-hashes \
    "${worker_args[@]}" 2>&1 | tee "$log"; then
    echo "PREFLIGHT FAILED: $preset" | tee -a "$log"
    printf "%s\tpreflight-failed\t%s\n" "$preset" "$output" >> "$SUMMARY"
    return
  fi

  if "$MAIN_PYTHON" -m dominican_eaters stt benchmark \
    "$MANIFEST" \
    --output-dir "$output" \
    --preset "$preset" \
    --device cuda \
    --precision fp16 \
    --warmup-runs 1 \
    --request-timeout 1800 \
    --short-audio-policy reject \
    --minimum-audio-seconds 0.1 \
    --verify-hashes \
    "${worker_args[@]}" 2>&1 | tee -a "$log"; then
    printf "%s\tcomplete\t%s\n" "$preset" "$output" >> "$SUMMARY"
  else
    printf "%s\tfailed\t%s\n" "$preset" "$output" >> "$SUMMARY"
  fi
}

if [[ ! -x "$MAIN_PYTHON" ]]; then
  echo "Core Python is not executable: $MAIN_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "STT manifest does not exist: $MANIFEST" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

run_model "whisper-base" ""
run_model "whisper-large-v3" ""
run_model "whisper-turbo" ""
run_model "parakeet-tdt-0.6b-v3" "${NEMO_PYTHON:-REQUIRED}"
run_model "canary-1b-v2" "${NEMO_PYTHON:-REQUIRED}"
run_model "granite-speech-4.1-2b" "${GRANITE_PYTHON:-REQUIRED}"
run_model "qwen3-asr-1.7b" "${QWEN3_PYTHON:-REQUIRED}"
run_model "voxtral-mini-3b-2507" "${VOXTRAL_PYTHON:-REQUIRED}"

echo
echo "ALL RUNS FINISHED"
echo "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
