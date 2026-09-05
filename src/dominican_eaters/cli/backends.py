"""ASR backend construction at the command-line composition boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

from dominican_eaters.speech.asr import (
    ASRBackend,
    JsonlSubprocessBackend,
    SubprocessBackendSettings,
    WhisperBackend,
    WhisperSettings,
)

BackendName = Literal["whisper", "parakeet", "canary"]

DEFAULT_MODELS: dict[BackendName, str] = {
    "whisper": "base",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "canary": "nvidia/canary-1b-v2",
}


def create_asr_backend(
    *,
    backend: BackendName,
    model: str | None,
    language: str,
    device: str,
    precision: str,
    worker_python: Path | None,
    request_timeout_seconds: float,
    timestamps: bool,
    short_audio_policy: str = "reject",
    minimum_audio_seconds: float = 0.1,
) -> ASRBackend:
    """Construct a side-effect-free backend; loading happens in the runner."""

    selected_model = model or DEFAULT_MODELS[backend]
    if backend == "whisper":
        if precision == "bf16":
            raise ValueError("Whisper does not support the bf16 CLI precision")
        return WhisperBackend(
            WhisperSettings(
                model=selected_model,
                language=language,
                device=cast(Literal["auto", "cpu", "cuda"], device),
                precision=cast(Literal["auto", "fp16", "fp32"], precision),
            )
        )

    if worker_python is None:
        raise ValueError(f"--worker-python is required for the {backend} backend")
    interpreter = worker_python.expanduser().resolve()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"worker Python is not an executable file: {interpreter}")
    return JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=interpreter,
            worker_module="dominican_eaters_nemo",
            backend=backend,
            model=selected_model,
            language=language,
            device=device,
            precision=precision,
            options={
                "timestamps": timestamps,
                "short_audio_policy": short_audio_policy,
                "minimum_audio_seconds": minimum_audio_seconds,
            },
            request_timeout_seconds=request_timeout_seconds,
        )
    )
