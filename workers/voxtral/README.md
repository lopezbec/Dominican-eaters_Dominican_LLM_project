# Dominican Eaters Voxtral worker

This independently installed worker runs the local
`mistralai/Voxtral-Mini-3B-2507` checkpoint through the official Transformers
transcription request flow. Standard output is reserved for JSONL protocol v2;
runtime diagnostics go to standard error.

Install it in a Python 3.11 or 3.12 environment separate from the core and other
model ecosystems:

```bash
python3.12 -m venv .venv-voxtral
.venv-voxtral/bin/pip install .
.venv-voxtral/bin/pip install ./workers/voxtral
```

The initial profile is intentionally narrow: Spanish, CUDA FP16, batch size one,
at most 10 minutes of audio, at most 500 generated tokens, and no timestamps.
This leaves more headroom than the checkpoint's advertised 30-minute context on
a 16 GB Tesla T4. Inputs above the limit, unknown-duration non-WAV inputs, CPU,
BF16/FP32, and timestamp requests are rejected instead of silently changing the
benchmark configuration.

`preflight` imports Transformers, Torch, and SoundFile, verifies the Voxtral API,
and checks CUDA without calling `from_pretrained` or downloading weights. Passing
offline tests or preflight does not promote the preset: a reviewed Spanish smoke
test and process-tree memory measurement on the target T4 remain required.
