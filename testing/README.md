# Testing Suite - UV-only STT/TTS Benchmarks

This folder benchmarks ASR and TTS models on Dominican songs in `lyrics-eater/audio`.
All setup and execution in this guide uses **uv only**.

## Scope

- **ASR models:** Whisper, Parakeet, Canary (optional: SeamlessM4T for S2ST output)
- **TTS models:** XTTS-v2, Qwen3-TTS, Chatterbox
- **Objective STT evaluation:** local audio + human transcript pairs
- **Legacy STT evaluation:** Genius lyrics (fetched live and cached)

## Requirements

- `uv` installed: <https://docs.astral.sh/uv/>
- `ffmpeg` available on PATH
- `GENIUS_ACCESS_TOKEN` exported only if you want the legacy Genius scoring flow
- Audio files under `lyrics-eater/audio` (already present in this repository)

Example environment setup:

```bash
export GENIUS_ACCESS_TOKEN="your_token_here"
export COQUI_TOS_AGREED=1
```

`COQUI_TOS_AGREED` is required when running XTTS-v2.

## 1) Run ASR models with uv

From repository root:

```bash
./testing/stt/run_all_stt.sh --audio-dir "./lyrics-eater/audio" --max 10 --whisper-model base
```

Optional SeamlessM4T run:

```bash
./testing/stt/run_all_stt.sh --audio-dir "./lyrics-eater/audio" --max 5 --include-seamless --seamless-tgt-lang eng
```

Outputs are written to:

- `testing/stt/transcriptions/whisper/`
- `testing/stt/transcriptions/parakeet/`
- `testing/stt/transcriptions/canary/`
- `testing/stt/transcriptions/seamless_m4t/` (if enabled)

## 2) Score ASR with local human transcript pairs

This is the objective STT evaluation path. It uses a local JSON manifest that maps
each audio file to a human-reviewed transcript and reports **WER + CER**.

Example manifest:

```json
{
  "rows": [
    {
      "audio_file": "lyrics-eater_218_Tercer_Cielo_Yo_Te_Extrañare_Video_Oficial.m4a",
      "reference_text_path": "../../../lyrics-eater/reference_texts/lyrics-eater_218_Tercer_Cielo_Yo_Te_Extrañare_Video_Oficial.txt",
      "split": "eval",
      "source": "human-transcript",
      "alignment_label": {
        "match_second": 53.525,
        "matched_word": "Yo te"
      }
    }
  ]
}
```

Run the evaluator:

```bash
uv run testing/stt/evaluate_with_local_refs.py \
  --transcriptions-dir testing/stt/transcriptions \
  --manifest testing/stt/ground_truth/local_reference_manifest.json \
  --models whisper parakeet canary \
  --output-dir testing/stt/results_local
```

Main outputs:

- `testing/stt/results_local/asr_scores_by_file.csv`
- `testing/stt/results_local/asr_leaderboard.csv`
- `testing/stt/results_local/asr_leaderboard.json`

Results list complete evaluations before incomplete ones, then sort by **corpus WER**. CER and file-level means/medians are supporting metrics. Compare systems only on matching sample coverage; missing outputs are reported, not silently ignored.

The manifest supports either:

- `reference_text_path` for transcript files already stored in the repo
- `reference_text` for inline references in lightweight experiments/tests

Optional `alignment_label`, `match_second`, and `matched_word` fields are carried into the per-file report so you can share manual alignment evidence alongside metric results.

## 3) Score ASR with Genius lyrics ground truth (legacy)

This script fetches Genius lyrics per audio item (using filename-derived queries),
caches references, computes WER/char/Jaccard/cosine, and ranks models.

```bash
uv run --with-requirements testing/stt/requirements-eval.txt \
  testing/stt/evaluate_with_genius.py \
  --transcriptions-dir testing/stt/transcriptions \
  --models whisper parakeet canary \
  --output-dir testing/stt/results \
  --cache-file testing/stt/ground_truth/genius_lyrics_cache.json
```

Main outputs:

- `testing/stt/results/asr_scores_by_file.csv`
- `testing/stt/results/asr_leaderboard.csv`
- `testing/stt/results/asr_leaderboard.json`

Ranking rule: **lowest median WER wins**.

## 4) Run TTS models with uv

Use song-related text prompt (for example, a lyric line) and optionally a song clip as reference audio.

```bash
./testing/tts/run_all_tts.sh \
  --text "Te regalo una rosa" \
  --language es \
  --speaker-wav "./lyrics-eater/audio/reference.wav" \
  --qwen-speaker Ryan
```

This produces:

- `testing/tts/outputs/chatterbox.wav`
- `testing/tts/outputs/qwen.wav`
- `testing/tts/outputs/xtts.wav` (if `--speaker-wav` provided)
- `testing/tts/outputs/run_manifest.json`

## 5) Score TTS by intelligibility (ASR-backtranscription)

This compares each generated WAV to the input prompt text by transcribing with Whisper.

```bash
uv run --with-requirements testing/tts/requirements-eval.txt \
  testing/tts/evaluate_tts_with_whisper.py \
  --manifest testing/tts/outputs/run_manifest.json \
  --output-dir testing/tts/results \
  --whisper-model base
```

Outputs:

- `testing/tts/results/tts_scores_by_file.csv`
- `testing/tts/results/tts_leaderboard.csv`
- `testing/tts/results/tts_leaderboard.json`

Ranking rule: **lowest WER wins**.

## 6) Poem-driven TTS benchmark (all-poems)

For a spoken-text benchmark using `all-poems` prompts instead of lyric lines:

```bash
./testing/tts/run_poems_tts.sh \
  --max-prompts 24 \
  --xtts-speaker-wav "./books-eater_000_La_Mañosa-Juan_Bosch_Audiolibro_(voz_humana).m4a" \
  --agree-license \
  --run-f5 \
  --f5-ref-audio "./books-eater_000_La_Mañosa-Juan_Bosch_Audiolibro_(voz_humana).m4a" \
  --f5-ref-text "Texto de referencia del audio narrado" \
  --run-kokoro \
  --kokoro-voice ef_dora
```

This writes:

- `testing/tts/outputs/poems_benchmark/poems_prompts.jsonl`
- `testing/tts/outputs/poems_benchmark/run_manifest_poems.json`
- Per-model WAVs in `testing/tts/outputs/poems_benchmark/*/`

Evaluate with Whisper backtranscription:

```bash
uv run --with-requirements testing/tts/requirements-eval.txt \
  testing/tts/evaluate_poems_tts_with_whisper.py \
  --manifest testing/tts/outputs/poems_benchmark/run_manifest_poems.json \
  --output-dir testing/tts/results/poems_benchmark \
  --whisper-model base
```

## 7) Actas PDF + YouTube voices -> XTTS synthetic dataset

This flow prepares a synthetic dataset by:

1. reading an Excel with `Nombre y Fecha del Documento`, `Link pdf`, and `Enlace youtube`
2. matching each spreadsheet row to a PDF file
3. extracting text from PDFs
4. chunking long actas into XTTS-friendly prompt files
5. downloading one candidate reference WAV per YouTube row (authorized/non-identifiable voice references only)
6. creating a Cartesian synthesis plan: `voice x acta_chunk`

> Safety note: use this workflow only with **authorized** or **non-identifiable** voice references.
>
> Important: XTTS clones a **single speaker reference**. If a YouTube video contains multiple people, noise, or session audio with interruptions, treat the downloaded WAV as a **candidate** reference that may need later curation.

Prepare only (recommended first):

```bash
uv run --no-project --python 3.11 \
  --with-requirements testing/tts/requirements-actas-xtts.txt \
  testing/tts/actas_xtts_dataset.py \
  --stage prepare \
  --excel-path "ARCHIVOS RELACIONALES (DEPURADO).xlsx" \
  --pdf-zip "CDD ACTAS PDFs-20260415T040612Z-3-001.zip" \
  --workspace-dir testing/tts/outputs/actas_xtts \
  --selection-mode random \
  --chunks-per-doc 2 \
  --random-seed 42 \
  --limit-voices 3 \
  --limit-docs 5
```

That writes:

- `testing/tts/outputs/actas_xtts/texts/`
- `testing/tts/outputs/actas_xtts/chunks/`
- `testing/tts/outputs/actas_xtts/chunks/<document_id>/chunks_manifest.jsonl`
- `testing/tts/outputs/actas_xtts/voice_refs/`
- `testing/tts/outputs/actas_xtts/manifests/documents.json`
- `testing/tts/outputs/actas_xtts/manifests/voices.json`
- `testing/tts/outputs/actas_xtts/manifests/synthesis_plan.json`
- `testing/tts/outputs/actas_xtts/manifests/synthesis_plan.jsonl`
- `testing/tts/outputs/actas_xtts/manifests/synthesis_plan.csv`

Chunk manifest rows now preserve traceability metadata before synthesis, including:

- `chunk_text_original` — exact extracted source text before TTS normalization
- `chunk_text_normalized_for_tts` — normalized text sent to XTTS
- `source_document_id`, `source_document_name`, `source_document_path`, `source_document_type`
- `source_page_number` for PDFs when available
- `source_char_start` / `source_char_end` as document-level character offsets
- `source_line_start` / `source_line_end` set to `null` when the source type cannot provide them

Chunk selection behavior:

- Default behavior is `--selection-mode random --chunks-per-doc 10`, so the pipeline samples chunks instead of synthesizing every chunk from every acta by default.
- `--selection-mode all` uses all chunks unless `--chunks-per-doc` is set (then it keeps the first N chunks per doc).
- `--selection-mode random` selects `--chunks-per-doc` chunks per doc with reproducibility from `--random-seed`.
- `--selection-mode manual` filters chunks via `--selected-chunk-ids` and/or `--manual-selection-file` (`{"document_id": ["chunk_0001", ...]}`).

Run XTTS from the generated plan:

```bash
uv run --no-project --python 3.11 \
  --with-requirements testing/tts/requirements-actas-xtts.txt \
  testing/tts/actas_xtts_dataset.py \
  --stage synthesize \
  --excel-path "ARCHIVOS RELACIONALES (DEPURADO).xlsx" \
  --workspace-dir testing/tts/outputs/actas_xtts \
  --agree-license \
  --device cpu \
  --model "/home/joserodm/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2" \
  --limit-pairs 20 \
  --skip-existing \
  --resume-from-metadata
```

Each synthesized voice/document folder now also writes a UTF-8 `metadata.jsonl` file next to the WAV files:

- `testing/tts/outputs/actas_xtts/synthetic_audio/<voice_reference_id>/<source_document_id>/metadata.jsonl`

Each JSONL row corresponds to one chunk/audio pair and includes generation status (`generated`, `failed`, or `skipped`), error details, model/reference metadata, the exact original chunk text, the normalized XTTS text, and source location fields.

Deterministic WAV filenames now follow this pattern:

- `<voice_reference_id>__<source_document_id>__p<page_number>__<chunk_id>.wav`
- or `<voice_reference_id>__<source_document_id>__<chunk_id>.wav` when page number is unavailable

Or run both stages in one go:

```bash
uv run --no-project --python 3.11 \
  --with-requirements testing/tts/requirements-actas-xtts.txt \
  testing/tts/actas_xtts_dataset.py \
  --stage all \
  --excel-path "ARCHIVOS RELACIONALES (DEPURADO).xlsx" \
  --pdf-zip "CDD ACTAS PDFs-20260415T040612Z-3-001.zip" \
  --workspace-dir testing/tts/outputs/actas_xtts \
  --agree-license \
  --device cpu \
  --model "/home/joserodm/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2" \
  --limit-voices 2 \
  --limit-docs 2 \
  --selection-mode random \
  --chunks-per-doc 1 \
  --limit-pairs 10
```

Dry-run/small-run command (prepare only, no voice download):

```bash
uv run --no-project --python 3.11 \
  --with-requirements testing/tts/requirements-actas-xtts.txt \
  testing/tts/actas_xtts_dataset.py \
  --stage prepare \
  --excel-path "ARCHIVOS RELACIONALES (DEPURADO).xlsx" \
  --pdf-zip "CDD ACTAS PDFs-20260415T040612Z-3-001.zip" \
  --workspace-dir testing/tts/outputs/actas_xtts_small \
  --skip-voice-download \
  --limit-rows 10 \
  --limit-voices 2 \
  --limit-docs 2 \
  --selection-mode random \
  --chunks-per-doc 1
```

This prepare-only run is the fastest metadata dry-run: it generates chunk manifests plus a synthesis plan with traceability fields, without downloading voices or running XTTS.

Resumable server command (metadata + outputs aware):

```bash
uv run --no-project --python 3.11 \
  --with-requirements testing/tts/requirements-actas-xtts.txt \
  testing/tts/actas_xtts_dataset.py \
  --stage synthesize \
  --excel-path "ARCHIVOS RELACIONALES (DEPURADO).xlsx" \
  --workspace-dir testing/tts/outputs/actas_xtts \
  --agree-license \
  --device cpu \
  --skip-existing \
  --resume-from-metadata
```

Validation pass:

```bash
uv run --no-project --python 3.11 \
  --with-requirements testing/tts/requirements-actas-xtts.txt \
  testing/tts/actas_xtts_dataset.py \
  --stage validate \
  --excel-path "ARCHIVOS RELACIONALES (DEPURADO).xlsx" \
  --workspace-dir testing/tts/outputs/actas_xtts
```

For server runs, start with small limits, review `voice_refs/` quality (authorized/non-identifiable references only), and only then scale up.

Recommended working stack for this workflow:

- `TTS==0.22.0`
- `transformers==4.39.3`
- `torchcodec`
- Python `3.11`

If your server has an older NVIDIA driver, use `--device cpu` first to validate the workflow end-to-end before attempting GPU inference.

## Reproducibility notes

- Each runner uses uv-managed isolated venvs in `testing/stt/.venvs` and `testing/tts/.venvs`.
- Dependencies are pinned per model family via local requirements files.
- Genius references are cached to avoid repeated network variance in the legacy flow:
  `testing/stt/ground_truth/genius_lyrics_cache.json`.

## Suggested benchmark protocol for this dataset

1. Run ASR models on `--max 30` files from `lyrics-eater/audio`.
2. Evaluate against local human transcript pairs and inspect `results_local/asr_leaderboard.csv`.
3. Use manual alignment labels from `lyrics-eater/reports/manual_word_matches.json` as supporting evidence when sharing examples.
4. Select 10 lyric lines from scored songs and run all TTS models.
5. Run TTS backtranscription scoring as an experimental proxy only.
6. Pick winners:
   - **Best ASR:** lowest median WER.
   - **Best TTS overall:** lowest backtranscription WER (proxy, not objective ground truth).
   - **Best zero-shot cloning:** compare XTTS and Chatterbox with the same speaker clip.

## Reproducible STT baseline with reviewed references

The first evaluation milestone is **speech-to-text**, using multilingual Whisper
`base` with Spanish selected. No TTS generation is involved. See
[metrics and sources](../docs/speech-evaluation-metrics.md).

Create a separate benchmark environment (Python 3.12 recommended for this repo):

```bash
python3.12 -m venv .venv-stt
source .venv-stt/bin/activate
python -m pip install -r testing/stt/requirements-benchmark.txt click
```

Whisper audio decoding also requires `ffmpeg` on PATH. The first model load may
download weights. The code does not install system packages or run against the
server automatically.

Copy `testing/stt/manifest.example.json` next to your dataset and replace its
placeholder paths with actual files. Paths are relative to the manifest, not the
working directory. Each `audio_file` must be unique. Use `reference_text` for
inline text or `reference_text_path` for a UTF-8 file. References must match the
exact audio segment and be human-reviewed. Do not use model-generated transcripts
as ground truth. Reuse `group_id` for related clips from the same recording or
speaker; confidence intervals resample those groups. `source` is provenance, not
the grouping key. The runner rejects mixed splits unless one is selected.

Preflight checks file existence, reference loading, duplicate keys, and hashes;
it does not decode audio, load models, or verify that annotations are accurate:

```bash
python cli.py benchmark-stt --manifest /path/to/dataset/manifest.json \
  --split eval --max 2 --output-dir /path/to/runs/whisper-smoke --preflight
```

Run a smoke test, then a full evaluation in a different, empty output directory:

```bash
python cli.py benchmark-stt --manifest /path/to/dataset/manifest.json \
  --split eval --max 2 --model base --device cuda --precision fp16 \
  --output-dir /path/to/runs/whisper-smoke

python cli.py benchmark-stt --manifest /path/to/dataset/manifest.json \
  --split eval --model base --device cuda --precision fp16 \
  --bootstrap 2000 --seed 42 --output-dir /path/to/runs/whisper-eval
```

The standalone entry point is `python testing/stt/benchmark.py` with the same
arguments. `--device cuda` fails if CUDA is unavailable; it never silently falls
back to CPU. `--device cpu --precision fp32` is supported for functional checks,
but CPU timing is not a T4 performance result. Existing model-script arguments
and output formats remain unchanged.

Outputs:

- `manifest.json`: selected references, group IDs, absolute audio paths and hashes.
- `run.json`: configuration, checkpoint identity when available, package versions,
  device, loading measurements, success/failure counts, total time and aggregate RTF.
- `transcriptions/whisper/*.json`: raw text, duration, inference time, RTF, sampled
  process RAM, CUDA allocator peaks, or an explicit error. Numeric filenames avoid
  collisions when different directories contain the same basename.
- `scores/asr_scores_by_file.{json,csv}`: raw/normalized text, S/D/I at word and
  character levels, denominators, WER and CER.
- `scores/asr_leaderboard.{json,csv}`: corpus scores and descriptive file means and
  medians. JSON also includes grouped 95% bootstrap intervals and coverage.

Rates are fractions, not percentages. Corpus scores use summed edits divided by
summed reference lengths. Empty successful transcripts are scored as deletions.
Missing/failed outputs remain unscored and are reported; a partial run exits with
status 1. Empty-reference rates and unsupported GPU measurements are JSON null.
One independent group is insufficient for a confidence interval. Comparisons need
matching sample coverage; paired model significance testing is not implemented.

Normalization now preserves spoken parenthetical text, accents and ñ, uses NFC
and lowercasing, and replaces non-alphanumeric characters with whitespace. CER
excludes spaces. This deliberately changes the old evaluator's normalization, so
rerun previous outputs before comparing old and new scores. JSON metadata records
the normalization version.

Timing includes transcription from already decoded 16 kHz audio; it excludes
file decoding, hashing, scoring and output writes. One synthetic-silence warmup
is excluded from scores; it does not guarantee every speech decoder path is warm.
Loading is measured separately. CUDA timings synchronize the device. GPU peaks
cover PyTorch allocations/reservations, not total device memory. RAM is sampled
current-process RSS (default 20 ms), excluding subprocesses; brief peaks may be
missed. FP16 describes decoding configuration, not a guarantee that every weight
or operation uses FP16. A seed does not ensure identical GPU results across
hardware or library versions.

Score existing model outputs without installing inference dependencies:

```bash
python testing/stt/evaluate_with_local_refs.py \
  --manifest /path/to/dataset/manifest.json \
  --transcriptions-dir /path/to/transcriptions --models whisper \
  --output-dir /path/to/scores --bootstrap 2000 --seed 42
```

Lightweight checks (no weights, GPU, audio dataset, or network required):

```bash
python tests/test_stt_local_evaluation.py
```

These include mocked backend and telemetry tests. They do not constitute a real
Whisper or T4 benchmark. Dataset inference remains to be run on the server.
