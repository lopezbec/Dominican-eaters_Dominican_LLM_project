# Dominican Eaters NeMo worker

This package isolates NVIDIA NeMo, Torch, and their native dependencies from the
main `dominican-eaters` environment. It communicates with the parent process only
through the versioned JSON Lines protocol on standard input and standard output.
Diagnostics and model logs are written to standard error.

Install this package with Python 3.11 or 3.12 in its own environment. The exact `dominican-eaters==0.2.0`
dependency keeps both sides of the protocol on the same contract version.

```bash
python3.12 -m venv .venv-nemo
.venv-nemo/bin/pip install .
.venv-nemo/bin/pip install ./workers/nemo
```

The worker executable is a protocol process that waits for JSONL requests on stdin; it is not an
interactive command. Its stdout is reserved for protocol frames and diagnostics go to stderr.

Canary always runs Spanish speech recognition with explicit `source_lang="es"`
and `target_lang="es"`. Short audio handling is explicit: the default `reject`
policy refuses clips shorter than 0.1 seconds and any input whose duration
cannot be determined. Select `--short-audio-policy allow` on the host benchmark
command only when unknown or very short durations are intentional.

The host invokes the installed worker with an absolute interpreter path:

```bash
dominican-eaters stt benchmark manifest.json \
  --output-dir artifacts/parakeet-run \
  --backend parakeet \
  --worker-python "$PWD/.venv-nemo/bin/python"
```

Backend defaults are `nvidia/parakeet-tdt-0.6b-v3` and `nvidia/canary-1b-v2`.
Both backends currently support Spanish only. Replace `parakeet` with `canary` in the host example
to run Canary. The host also exposes precision, timestamp, request-timeout, and short-audio
options through `dominican-eaters stt benchmark --help`.

The first load may download model weights. The host reports both its own RSS and combined RSS for
itself plus descendants, which includes this worker while it is alive. CUDA values are
per-transcription PyTorch allocator peaks. No real model, GPU, or server benchmark is claimed by
the offline package tests.
