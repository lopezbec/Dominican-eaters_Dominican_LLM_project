# Dominican Eaters

Dominican Eaters is being rebuilt as an installable Python package for collecting Dominican
Spanish data and evaluating speech systems. This is a clean architecture migration: the new
runtime has one package, one CLI, one configuration schema, and one STT data contract. Replaced
interfaces will be removed instead of retained through compatibility wrappers.

## Current supported slice

The new `src/dominican_eaters` package currently provides:

- strict application configuration with explicit data and artifact roots;
- a versioned, replayable STT manifest with stable sample IDs and optional SHA-256 checks;
- a small ASR backend protocol with explicit load, warmup, transcribe, and close ownership;
- coverage-aware corpus WER/CER, silence controls, and grouped bootstrap intervals;
- atomic manifest, incremental checkpoint, and benchmark-result artifacts;
- a lazy OpenAI Whisper adapter with strict device and precision selection;
- a strict JSONL worker protocol and lifecycle-safe subprocess controller;
- isolated NeMo workers for NVIDIA Parakeet and Canary Spanish ASR;
- model-load and per-sample timing, audio duration/RTF, sampled host/process-tree RSS, and CUDA allocator peaks;
- a first books collection domain with stable IDs, strict source manifests, explicit outcomes,
  retry-aware resume, and atomic checkpoints;
- dependency-light lyrics and poems domains with typed provider ports, deterministic matching,
  explicit partial/error states, strict manifests, and interruption-safe ledgers;
- a lazy official YouTube Data API adapter shared by the book, lyrics, and poem workflows;
- a lazy Genius API and lyrics-page adapter with canonical-host validation;
- credential-safe collection commands that read API credentials only from environment variables;
- a lightweight installed CLI that does not import Torch, Whisper, or NeMo until its backend loads.

The live-provider adapters and CLI wiring have offline contract coverage, but no credentialed
Genius or YouTube run has been verified. CSV/XLSX projections and the one-time legacy-data
converter are not implemented yet. The root `cli.py` and old top-level modules are migration
inputs, not supported interfaces for new development.

## Install for development

Python 3.11 or newer is required. Python 3.12 is the repository's development version.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . pytest ruff mypy build types-PyYAML types-psutil
```

Install the optional HTTP client only in environments that run live collectors:

```bash
.venv/bin/python -m pip install -e '.[providers]'
```

CLI help and all manifest preflight commands require only the base installation.

Inspect the installed CLI and validate the canonical configuration:

```bash
.venv/bin/dominican-eaters --help
.venv/bin/dominican-eaters config validate config/default.yaml
```

## STT manifest

An STT dataset uses one JSON schema. `dataset_root` may be absolute or relative to the manifest
file on disk. Each `audio_path` is a portable POSIX path below that root. A `sample_id` identifies
the utterance independently of its filename.

```json
{
  "schema_version": 1,
  "dataset_root": "../data/stt-evaluation",
  "samples": [
    {
      "sample_id": "speaker-01-utterance-001",
      "audio_path": "speaker-01/utterance-001.wav",
      "reference_text": "Buenos días",
      "group_id": "speaker-01",
      "split": "test",
      "source": "curated-evaluation-set",
      "sha256": null
    }
  ]
}
```

Validate structure and audio availability before loading a model:

```bash
.venv/bin/dominican-eaters stt preflight path/to/manifest.json
.venv/bin/dominican-eaters stt preflight path/to/manifest.json --verify-hashes
```

Remapping a manifest to another dataset location is explicit and verifies every declared hash:

```bash
.venv/bin/dominican-eaters stt preflight path/to/manifest.json \
  --dataset-root /srv/datasets/stt-evaluation
```

## Collection manifests and runs

Books use a strict JSON source catalog with canonical IDs derived from normalized title and author
identity. Validate it without making provider calls:

```bash
.venv/bin/dominican-eaters collect books preflight path/to/books.json
```

The collection engine retains every source book, including pending, not-found, and provider-error
outcomes. Resume skips only valid found/partial records, retries other states, merges newly added
books by ID, and atomically checkpoints after every attempt. CSV/XLSX will be derived exports,
never resume state.

Lyrics and poems have the same dependency-free preflight boundary:

```bash
.venv/bin/dominican-eaters collect lyrics preflight path/to/lyrics.json
.venv/bin/dominican-eaters collect poems preflight path/to/poems.json
```

Lyrics retain one result per stable request and distinguish complete, partial, not-found, and
provider-error outcomes. Poems retain the full current source catalog, rank candidates
deterministically, and classify fragment markers before presentation styles. All three collection
writers use adjacent advisory locks so concurrent processes cannot silently overwrite progress.

The strict JSON manifests use `schema_version: 1`. Book entries require `book_id`, `title`,
`author`, `publication_year`, and `source`. Lyrics requests require `request_id` and `query`, with
nullable `expected_title` and `expected_artist`. Poem entries require `source_id`, `title`,
`author`, `publication_year`, `genre`, `reference_text`, and `provenance`. Unknown fields are
rejected rather than silently ignored.

Live collection reads secrets only from the environment. Secrets are not accepted as CLI options
and are not stored in collection artifacts:

```bash
export YOUTUBE_API_KEY='...'
export GENIUS_ACCESS_TOKEN='...'  # lyrics only

.venv/bin/dominican-eaters collect books run path/to/books.json \
  --output-dir artifacts/books
.venv/bin/dominican-eaters collect lyrics run path/to/lyrics.json \
  --output-dir artifacts/lyrics
.venv/bin/dominican-eaters collect poems run path/to/poems.json \
  --output-dir artifacts/poems
```

These commands collect metadata, lyrics, and selected media links; they do not download media.
Their canonical resume artifacts are `books-collection.json`, `lyrics-collection.json`, and
`poems-collection.json`. `--force` reprocesses the current manifest. Provider errors leave the
checkpoint or ledger intact and produce exit status 1.

## Whisper benchmark environment

Install Whisper into a dedicated environment; it remains outside the base package dependencies:

```bash
python3.12 -m venv .venv-whisper
.venv-whisper/bin/python -m pip install -e '.[whisper]'
```

Run one immutable benchmark. The output directory must not already exist:

```bash
.venv-whisper/bin/dominican-eaters stt benchmark path/to/manifest.json \
  --output-dir artifacts/whisper-base-run \
  --backend whisper \
  --model base \
  --language es \
  --device auto \
  --precision auto
```

The command exits zero only for a complete run. Partial inference, load failures, scoring errors,
and cleanup errors produce a nonzero exit while retaining structured evidence in `result.json`.
Explicit CUDA requests never silently fall back to CPU.

The frozen `manifest.json` and latest `checkpoint.json` are written before model work. If a run is
interrupted, the checkpoint retains completed samples and the in-flight sample ID; a final
`result.json` exists only after normal lifecycle completion.

## NeMo benchmark environment

NeMo is isolated because its Python and native dependency constraints differ from the core. Use
Python 3.11 or 3.12, install the core first, then install the worker:

```bash
python3.12 -m venv .venv-nemo
.venv-nemo/bin/python -m pip install -e .
.venv-nemo/bin/python -m pip install -e ./workers/nemo
```

Pass the absolute worker interpreter to the lightweight host CLI:

```bash
.venv/bin/dominican-eaters stt benchmark path/to/manifest.json \
  --output-dir artifacts/parakeet-run \
  --backend parakeet \
  --language es \
  --worker-python "$PWD/.venv-nemo/bin/python"

.venv/bin/dominican-eaters stt benchmark path/to/manifest.json \
  --output-dir artifacts/canary-run \
  --backend canary \
  --language es \
  --worker-python "$PWD/.venv-nemo/bin/python"
```

Defaults are `nvidia/parakeet-tdt-0.6b-v3` and `nvidia/canary-1b-v2`. Both workers currently
accept Spanish only; Canary explicitly requests Spanish-to-Spanish recognition. The default
short-audio policy rejects clips below 0.1 seconds and inputs whose duration cannot be read. Use
`--short-audio-policy allow` only when that is intentional.

Timing includes adapter audio loading in each transcription call. Model load is measured
separately; warmup, scoring, and artifact writes are excluded. Host-only and host-plus-descendant
RSS are reported separately, so NeMo worker memory is included in the latter. CUDA fields are
PyTorch allocator peaks rather than total GPU use.

## Architecture

```text
CLI / future jobs
        |
        v
application runner ----> versioned manifest and atomic artifacts/checkpoints
        |                              |
        v                              v
ASRBackend protocol             pure evaluation policy
        |
        v
direct lazy adapter or JSONL subprocess boundary
                                      |
                                      v
                         isolated model worker/environment
```

Runtime code lives under `src/dominican_eaters`. Configuration and manifests are loaded at the
edge, and resolved paths or typed contracts are passed inward. Model dependencies belong only in
backend-specific environments. Result artifacts record model identity, requested and effective
runtime settings, language, decoding policy, and package versions once per run.

## Verification

```bash
ruff format --check src tests/core
ruff check src tests/core
mypy src
pytest -q
python -m build
python -m build workers/nemo
```

The migration decisions and sequence are recorded in
[`docs/architecture-refactor-plan.md`](docs/architecture-refactor-plan.md) and
[`docs/adr`](docs/adr).

## Acknowledgment

This project has been partially supported by the Ministerio de Educación Superior, Ciencia y
Tecnología (MESCyT) of the Dominican Republic through the FONDOCYT grant. The views expressed by
the project do not necessarily represent MESCyT.
