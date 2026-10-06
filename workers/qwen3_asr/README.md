# Qwen3-ASR worker

Isolated JSONL worker for the native Transformers checkpoint
`Qwen/Qwen3-ASR-1.7B-hf`. The first profile is offline, batch-one inference with
FP16 on CUDA and no forced aligner.

The worker imports Torch and Transformers lazily. Its `preflight` operation
checks the installed runtime and CUDA without loading or downloading model
weights.

```bash
python -m venv .venv-qwen3-asr
.venv-qwen3-asr/bin/pip install -e . -e workers/qwen3_asr
DOMINICAN_EATERS_QWEN3_ASR_PYTHON=.venv-qwen3-asr/bin/python \
  dominican-eaters stt preflight --preset qwen3-asr-1.7b
```

The model card requires Transformers 5.13 or newer. `language=es` forces
Spanish; `language=auto` keeps Qwen's language identification enabled.
