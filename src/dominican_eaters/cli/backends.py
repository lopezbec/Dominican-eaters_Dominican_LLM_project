"""ASR backend construction at the command-line composition boundary."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

from dominican_eaters.speech.asr import (
    ASRBackend,
    BackendDescriptor,
    JsonlSubprocessBackend,
    SubprocessBackendSettings,
    Transcript,
    WhisperBackend,
    WhisperSettings,
    WorkerPreflightReport,
)
from dominican_eaters.speech.asr.registry import (
    CURRENT_BACKEND_SPECS,
    BackendName,
    ModelPreset,
)
from dominican_eaters.speech.asr.worker_protocol import JSONValue


class _PresetBackend:
    """Attach registry identity without coupling runtime adapters to the registry."""

    def __init__(self, backend: ASRBackend, preset: ModelPreset) -> None:
        self._backend = backend
        self._preset = preset

    @property
    def backend_id(self) -> str:
        return self._backend.backend_id

    @property
    def descriptor(self) -> BackendDescriptor:
        return replace(
            self._backend.descriptor,
            preset_id=self._preset.preset_id,
            runtime_id=self._preset.runtime_id,
            model_revision_requested=self._preset.model_revision,
            quantization=self._preset.quantization,
        )

    def load(self) -> None:
        self._backend.load()

    def warmup(self) -> None:
        self._backend.warmup()

    def transcribe(self, audio_path: Path) -> Transcript:
        return self._backend.transcribe(audio_path)

    def close(self) -> None:
        self._backend.close()

    def preflight(self, *, close_after: bool = False) -> WorkerPreflightReport:
        if not isinstance(self._backend, JsonlSubprocessBackend):
            raise TypeError("inline backends do not expose worker preflight")
        return self._backend.preflight(close_after=close_after)


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
    worker_stderr_sink: Callable[[str], None] | None = None,
    preset: ModelPreset | None = None,
) -> ASRBackend:
    """Construct a side-effect-free backend; loading happens in the runner."""

    try:
        spec = CURRENT_BACKEND_SPECS[backend]
    except KeyError as error:
        raise ValueError(f"Backend {backend!r} is not currently runnable") from error
    selected_model = model or spec.default_model
    if spec.execution == "inline":
        if precision == "bf16":
            raise ValueError("Whisper does not support the bf16 CLI precision")
        selected_backend: ASRBackend = WhisperBackend(
            WhisperSettings(
                model=selected_model,
                language=language,
                device=cast(Literal["auto", "cpu", "cuda"], device),
                precision=cast(Literal["auto", "fp16", "fp32"], precision),
            )
        )
        return _PresetBackend(selected_backend, preset) if preset is not None else selected_backend

    if worker_python is None:
        raise ValueError(f"--worker-python is required for the {backend} backend")
    interpreter = Path(os.path.abspath(worker_python.expanduser()))
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"worker Python is not an executable file: {interpreter}")
    worker_options: dict[str, JSONValue]
    if backend in {"parakeet", "canary"}:
        worker_options = {
            "timestamps": timestamps,
            "short_audio_policy": short_audio_policy,
            "minimum_audio_seconds": minimum_audio_seconds,
        }
    elif backend in {"qwen3_asr", "voxtral"}:
        worker_options = {"timestamps": timestamps}
    else:
        worker_options = {}
    selected_backend = JsonlSubprocessBackend(
        SubprocessBackendSettings(
            interpreter=interpreter,
            worker_module=cast(str, spec.worker_module),
            backend=backend,
            model=selected_model,
            language=language,
            device=device,
            precision=precision,
            options=worker_options,
            request_timeout_seconds=request_timeout_seconds,
            preset=None if preset is None else preset.preset_id,
            model_revision=None if preset is None else preset.model_revision,
            quantization=None if preset is None else preset.quantization,
        ),
        stderr_sink=worker_stderr_sink,
    )
    return _PresetBackend(selected_backend, preset) if preset is not None else selected_backend
