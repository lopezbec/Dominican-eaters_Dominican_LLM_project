# Testing Suite - UV-only STT/TTS Benchmarks

This folder benchmarks ASR and TTS models on Dominican songs in `lyrics-eater/audio`.
All setup and execution in this guide uses **uv only**.

## Scope

- **ASR models:** Whisper, Parakeet, Canary (optional: SeamlessM4T for S2ST output)
- **TTS models:** XTTS-v2, Qwen3-TTS, Chatterbox
- **ASR ground truth:** Genius lyrics (fetched live and cached)

## Requirements

- `uv` installed: <https://docs.astral.sh/uv/>
- `ffmpeg` available on PATH
- `GENIUS_ACCESS_TOKEN` exported for Genius scoring
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

## 2) Score ASR with Genius lyrics ground truth

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

## 3) Run TTS models with uv

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

## 4) Score TTS by intelligibility (ASR-backtranscription)

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

## Reproducibility notes

- Each runner uses uv-managed isolated venvs in `testing/stt/.venvs` and `testing/tts/.venvs`.
- Dependencies are pinned per model family via local requirements files.
- Genius references are cached to avoid repeated network variance:
  `testing/stt/ground_truth/genius_lyrics_cache.json`.

## Suggested benchmark protocol for this dataset

1. Run ASR models on `--max 30` files from `lyrics-eater/audio`.
2. Run Genius scoring and inspect `asr_leaderboard.csv`.
3. Select 10 lyric lines from scored songs and run all TTS models.
4. Run TTS backtranscription scoring.
5. Pick winners:
   - **Best ASR:** lowest median WER.
   - **Best TTS overall:** lowest backtranscription WER.
   - **Best zero-shot cloning:** compare XTTS and Chatterbox with the same speaker clip.
