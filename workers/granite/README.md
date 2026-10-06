# Dominican Eaters Granite Speech worker

This independently packaged worker isolates the Transformers stack required by
`ibm-granite/granite-speech-4.1-2b`. It reserves stdout for the shared JSONL protocol and writes
all diagnostics to stderr.

Create a dedicated Python 3.11 or 3.12 environment and install the core package before the worker:

```bash
python3.12 -m venv .venv-granite
.venv-granite/bin/pip install .
.venv-granite/bin/pip install ./workers/granite
export DOMINICAN_EATERS_GRANITE_PYTHON="$PWD/.venv-granite/bin/python"
```

The protocol-v2 `preflight` imports the installed runtime and checks its APIs, Python version,
CUDA availability, and the FP16 policy without calling `from_pretrained` or downloading weights.
The supported initial profile uses the official Transformers processor/model flow, batch size one,
mono 16 kHz input, deterministic generation, and FP16 on CUDA. CPU is available only with FP32 for
development and contract testing. Quantized GGUF profiles are intentionally outside this worker.

The first `load` may download the model when it is not already cached. Offline tests inject a fake
runtime and never access the model hub. No real GPU compatibility or benchmark result is claimed
until the separate T4 smoke-run gate is completed.
