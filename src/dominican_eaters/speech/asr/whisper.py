"""Lazy OpenAI Whisper adapter for an isolated model environment."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import platform
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from .contracts import BackendDescriptor, Transcript


class WhisperDependencyError(ImportError):
    """Raised when the adapter environment lacks its model dependencies."""


class _CudaRuntime(Protocol):
    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...

    def synchronize(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def max_memory_allocated(self) -> int: ...

    def max_memory_reserved(self) -> int: ...


class _TorchRuntime(Protocol):
    cuda: _CudaRuntime


class _WhisperModel(Protocol):
    def transcribe(self, audio: object, **options: object) -> Mapping[str, object]: ...


class _WhisperRuntime(Protocol):
    def load_model(self, name: str, *, device: str) -> _WhisperModel: ...

    def load_audio(self, path: str) -> object: ...


class _NumpyRuntime(Protocol):
    def zeros(self, size: int, *, dtype: str) -> object: ...


@dataclass(frozen=True, slots=True)
class WhisperRuntimeBundle:
    whisper: _WhisperRuntime
    torch: _TorchRuntime
    numpy: _NumpyRuntime


RuntimeLoader = Callable[[], WhisperRuntimeBundle]


def load_whisper_runtime() -> WhisperRuntimeBundle:
    """Import heavyweight dependencies only when a run starts loading its model."""

    try:
        whisper = importlib.import_module("whisper")
        torch = importlib.import_module("torch")
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise WhisperDependencyError(
            "The Whisper backend requires its isolated environment. "
            "Install environments/whisper/requirements.in."
        ) from error
    return WhisperRuntimeBundle(
        whisper=cast(_WhisperRuntime, whisper),
        torch=cast(_TorchRuntime, torch),
        numpy=cast(_NumpyRuntime, numpy),
    )


@dataclass(frozen=True, slots=True)
class WhisperSettings:
    model: str = "base"
    language: str = "es"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp16", "fp32"] = "auto"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Whisper model must not be empty")
        if not self.language.strip():
            raise ValueError("Whisper language must not be empty")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Unsupported Whisper device: {self.device}")
        if self.precision not in ("auto", "fp16", "fp32"):
            raise ValueError(f"Unsupported Whisper precision: {self.precision}")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("Whisper fp16 precision requires CUDA")


class WhisperBackend:
    """OpenAI Whisper implementation of the canonical ASR lifecycle."""

    def __init__(
        self,
        settings: WhisperSettings,
        *,
        runtime_loader: RuntimeLoader = load_whisper_runtime,
    ) -> None:
        self._settings = settings
        self._runtime_loader = runtime_loader
        self._runtime: WhisperRuntimeBundle | None = None
        self._model: _WhisperModel | None = None
        self._resolved_device: Literal["cpu", "cuda"] | None = None
        self._fp16: bool | None = None
        self._runtime_versions = {"python": platform.python_version()}

    @property
    def backend_id(self) -> str:
        return f"openai-whisper/{self._settings.model}"

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            model=self._settings.model,
            model_revision=None,
            language=self._settings.language,
            requested_device=self._settings.device,
            requested_precision=self._settings.precision,
            effective_device=self._resolved_device,
            effective_precision=(None if self._fp16 is None else "fp16" if self._fp16 else "fp32"),
            runtime_versions=dict(self._runtime_versions),
            options={
                "task": "transcribe",
                "temperature": 0.0,
                "condition_on_previous_text": False,
            },
        )

    def load(self) -> None:
        if self._model is not None:
            return
        runtime = self._runtime_loader()
        self._runtime = runtime
        for package in ("openai-whisper", "torch", "numpy"):
            try:
                self._runtime_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._runtime_versions[package] = "unknown"
        cuda_available = runtime.torch.cuda.is_available()
        if self._settings.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested for Whisper but is unavailable")
        device: Literal["cpu", "cuda"] = (
            "cuda"
            if self._settings.device == "cuda"
            or (self._settings.device == "auto" and cuda_available)
            else "cpu"
        )
        if self._settings.precision == "fp16" and device != "cuda":
            raise RuntimeError("Whisper fp16 precision requires CUDA")
        self._resolved_device = device
        self._fp16 = device == "cuda" and self._settings.precision != "fp32"
        self._model = runtime.whisper.load_model(self._settings.model, device=device)

    def warmup(self) -> None:
        model, runtime = self._require_loaded()
        model.transcribe(runtime.numpy.zeros(16_000, dtype="float32"), **self._options())

    def transcribe(self, audio_path: Path) -> Transcript:
        model, runtime = self._require_loaded()
        cuda = self._resolved_device == "cuda"
        if cuda:
            runtime.torch.cuda.synchronize()
            runtime.torch.cuda.reset_peak_memory_stats()
        audio = runtime.whisper.load_audio(str(audio_path))
        sample_count = len(cast(Sized, audio))
        if sample_count <= 0:
            raise ValueError("Whisper audio must not be empty")
        result = model.transcribe(audio, **self._options())
        if cuda:
            runtime.torch.cuda.synchronize()
        text = result.get("text")
        if not isinstance(text, str):
            raise TypeError("Whisper result must contain a string text field")
        language = result.get("language", self._settings.language)
        if not isinstance(language, str):
            raise TypeError("Whisper result language must be a string")
        return Transcript(
            text=text,
            language=language,
            audio_duration_seconds=sample_count / 16_000,
            gpu_peak_allocated_bytes=(runtime.torch.cuda.max_memory_allocated() if cuda else None),
            gpu_peak_reserved_bytes=(runtime.torch.cuda.max_memory_reserved() if cuda else None),
            metadata={
                "model": self._settings.model,
                "device": self._resolved_device,
                "precision": "fp16" if self._fp16 else "fp32",
            },
        )

    def close(self) -> None:
        self._model = None
        gc.collect()
        runtime = self._runtime
        if runtime is not None and self._resolved_device == "cuda":
            runtime.torch.cuda.empty_cache()
        self._runtime = None

    def _require_loaded(self) -> tuple[_WhisperModel, WhisperRuntimeBundle]:
        if self._model is None or self._runtime is None:
            raise RuntimeError("Whisper backend is not loaded")
        return self._model, self._runtime

    def _options(self) -> dict[str, object]:
        if self._fp16 is None:
            raise RuntimeError("Whisper backend is not loaded")
        return {
            "language": self._settings.language,
            "task": "transcribe",
            "fp16": self._fp16,
            "verbose": None,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        }
