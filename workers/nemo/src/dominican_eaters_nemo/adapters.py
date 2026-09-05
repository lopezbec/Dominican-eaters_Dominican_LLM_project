"""Lazy, injection-friendly adapters for NVIDIA NeMo ASR models."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import platform
import tempfile
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast

from dominican_eaters.speech.asr import BackendDescriptor, Transcript


class NeMoDependencyError(ImportError):
    """Raised when this isolated environment lacks NeMo runtime dependencies."""


class ShortAudioError(ValueError):
    """Raised when a sample violates the configured minimum-duration policy."""


class ShortAudioPolicy(StrEnum):
    REJECT = "reject"
    ALLOW = "allow"


class _CudaRuntime(Protocol):
    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def synchronize(self) -> None: ...

    def max_memory_allocated(self) -> int: ...

    def max_memory_reserved(self) -> int: ...


class _TorchRuntime(Protocol):
    cuda: _CudaRuntime


class _NeMoModel(Protocol):
    def transcribe(self, audio: Sequence[str], **options: object) -> Sequence[object]: ...

    def to(self, device: str) -> _NeMoModel: ...

    def eval(self) -> object: ...

    def half(self) -> _NeMoModel: ...

    def bfloat16(self) -> _NeMoModel: ...

    def float(self) -> _NeMoModel: ...


class _ASRModelFactory(Protocol):
    def from_pretrained(self, *, model_name: str) -> _NeMoModel: ...


class _Models(Protocol):
    ASRModel: _ASRModelFactory


class _NeMoASRRuntime(Protocol):
    models: _Models


@dataclass(frozen=True, slots=True)
class NeMoRuntimeBundle:
    nemo_asr: _NeMoASRRuntime
    torch: _TorchRuntime


RuntimeLoader = Callable[[], NeMoRuntimeBundle]


def load_nemo_runtime() -> NeMoRuntimeBundle:
    """Load heavyweight dependencies only when the worker receives ``load``."""

    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise NeMoDependencyError(
            "The NeMo backend requires the isolated workers/nemo environment"
        ) from error
    return NeMoRuntimeBundle(
        nemo_asr=cast(_NeMoASRRuntime, nemo_asr),
        torch=cast(_TorchRuntime, torch),
    )


@dataclass(frozen=True, slots=True)
class NeMoSettings:
    model: str
    language: str = "es"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    short_audio_policy: ShortAudioPolicy = ShortAudioPolicy.REJECT
    minimum_audio_seconds: float = 0.1
    timestamps: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("NeMo model must not be empty")
        if self.language != "es":
            raise ValueError("The NeMo worker currently supports Spanish ('es') only")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Unsupported NeMo device: {self.device}")
        if self.precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError(f"Unsupported NeMo precision: {self.precision}")
        if self.device == "cpu" and self.precision in ("fp16", "bf16"):
            raise ValueError(f"NeMo {self.precision} precision requires CUDA")
        if self.minimum_audio_seconds <= 0:
            raise ValueError("minimum_audio_seconds must be positive")


@dataclass(slots=True)
class NeMoBackend:
    """Common lifecycle and normalization for a single NeMo model."""

    settings: NeMoSettings
    runtime_loader: RuntimeLoader = load_nemo_runtime
    _runtime: NeMoRuntimeBundle | None = field(default=None, init=False)
    _model: _NeMoModel | None = field(default=None, init=False)
    _resolved_device: Literal["cpu", "cuda"] | None = field(default=None, init=False)
    _resolved_precision: Literal["fp32", "fp16", "bf16"] | None = field(default=None, init=False)
    _runtime_versions: dict[str, str] = field(
        default_factory=lambda: {"python": platform.python_version()}, init=False
    )

    backend_name: ClassVar[str] = "nemo"

    @property
    def backend_id(self) -> str:
        return f"nvidia-{self.backend_name}/{self.settings.model}"

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            model=self.settings.model,
            model_revision=None,
            language=self.settings.language,
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device=self._resolved_device,
            effective_precision=self._resolved_precision,
            runtime_versions=dict(self._runtime_versions),
            options=self._descriptor_options(),
        )

    def load(self) -> None:
        if self._model is not None:
            return
        runtime = self.runtime_loader()
        self._runtime = runtime
        self._capture_versions()
        cuda_available = runtime.torch.cuda.is_available()
        if self.settings.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested for NeMo but is unavailable")
        device: Literal["cpu", "cuda"] = (
            "cuda"
            if self.settings.device == "cuda" or (self.settings.device == "auto" and cuda_available)
            else "cpu"
        )
        if device == "cpu" and self.settings.precision in ("fp16", "bf16"):
            raise RuntimeError(f"NeMo {self.settings.precision} precision requires CUDA")
        precision: Literal["fp32", "fp16", "bf16"] = (
            "fp16"
            if self.settings.precision == "auto" and device == "cuda"
            else "fp32"
            if self.settings.precision == "auto"
            else self.settings.precision
        )
        model = runtime.nemo_asr.models.ASRModel.from_pretrained(model_name=self.settings.model)
        model = model.to(device)
        model = self._apply_precision(model, precision)
        model.eval()
        self._resolved_device = device
        self._resolved_precision = precision
        self._model = model

    def warmup(self) -> None:
        model = self._require_loaded()
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\0\0" * 16_000)
            model.transcribe([temporary.name], **self._transcribe_options())

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        model = self._require_loaded()
        if not audio_path.is_absolute():
            raise ValueError("audio_path must be absolute")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        duration = duration_seconds if duration_seconds is not None else _wav_duration(audio_path)
        self._enforce_short_audio_policy(duration)
        runtime = self._runtime
        if runtime is not None and self._resolved_device == "cuda":
            runtime.torch.cuda.reset_peak_memory_stats()
        output = model.transcribe([str(audio_path)], **self._transcribe_options())
        if runtime is not None and self._resolved_device == "cuda":
            runtime.torch.cuda.synchronize()
        if len(output) != 1:
            raise RuntimeError(f"NeMo must return exactly one result; received {len(output)}")
        text, language, metadata = _normalize_hypothesis(output[0], self.settings.language)
        metadata.update(
            {
                "model": self.settings.model,
                "device": self._resolved_device,
                "precision": self._resolved_precision,
                "duration_seconds": duration,
            }
        )
        allocated: int | None = None
        reserved: int | None = None
        if runtime is not None and self._resolved_device == "cuda":
            allocated = int(runtime.torch.cuda.max_memory_allocated())
            reserved = int(runtime.torch.cuda.max_memory_reserved())
        return Transcript(
            text=text,
            language=language,
            audio_duration_seconds=duration,
            gpu_peak_allocated_bytes=allocated,
            gpu_peak_reserved_bytes=reserved,
            metadata=metadata,
        )

    def close(self) -> None:
        self._model = None
        gc.collect()
        runtime = self._runtime
        if runtime is not None and self._resolved_device == "cuda":
            runtime.torch.cuda.empty_cache()
        self._runtime = None

    def _require_loaded(self) -> _NeMoModel:
        if self._model is None:
            raise RuntimeError("NeMo backend is not loaded")
        return self._model

    def _capture_versions(self) -> None:
        for package in ("nemo_toolkit", "torch"):
            try:
                self._runtime_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._runtime_versions[package] = "unknown"

    def _apply_precision(
        self, model: _NeMoModel, precision: Literal["fp32", "fp16", "bf16"]
    ) -> _NeMoModel:
        if precision == "fp16":
            return model.half()
        if precision == "bf16":
            return model.bfloat16()
        return model.float()

    def _enforce_short_audio_policy(self, duration_seconds: float | None) -> None:
        if self.settings.short_audio_policy is ShortAudioPolicy.ALLOW:
            return
        if duration_seconds is None:
            raise ShortAudioError("Audio duration is required by the reject short-audio policy")
        if duration_seconds < self.settings.minimum_audio_seconds:
            raise ShortAudioError(
                f"Audio duration {duration_seconds:.6g}s is below the configured minimum "
                f"of {self.settings.minimum_audio_seconds:.6g}s"
            )

    def _descriptor_options(self) -> dict[str, object]:
        return {
            "short_audio_policy": self.settings.short_audio_policy.value,
            "minimum_audio_seconds": self.settings.minimum_audio_seconds,
            "timestamps": self.settings.timestamps,
        }

    def _transcribe_options(self) -> dict[str, object]:
        return {"timestamps": self.settings.timestamps}


class ParakeetBackend(NeMoBackend):
    backend_name = "parakeet"

    def __init__(
        self,
        settings: NeMoSettings,
        *,
        runtime_loader: RuntimeLoader = load_nemo_runtime,
    ) -> None:
        super().__init__(settings, runtime_loader)


class CanaryBackend(NeMoBackend):
    backend_name = "canary"

    def __init__(
        self,
        settings: NeMoSettings,
        *,
        runtime_loader: RuntimeLoader = load_nemo_runtime,
    ) -> None:
        super().__init__(settings, runtime_loader)

    def _descriptor_options(self) -> dict[str, object]:
        options = super()._descriptor_options()
        options.update({"source_lang": "es", "target_lang": "es"})
        return options

    def _transcribe_options(self) -> dict[str, object]:
        options = super()._transcribe_options()
        options.update({"source_lang": "es", "target_lang": "es"})
        return options


def _wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            if rate <= 0:
                return None
            return audio.getnframes() / rate
    except (EOFError, wave.Error):
        return None


def _normalize_hypothesis(
    hypothesis: object, default_language: str
) -> tuple[str, str, dict[str, object]]:
    if isinstance(hypothesis, str):
        return hypothesis, default_language, {}
    if isinstance(hypothesis, Mapping):
        text = hypothesis.get("text")
        language = hypothesis.get("language", default_language)
        timestamp = hypothesis.get("timestamp")
    else:
        text = getattr(hypothesis, "text", None)
        language = getattr(hypothesis, "language", default_language)
        timestamp = getattr(hypothesis, "timestamp", None)
    if not isinstance(text, str):
        raise TypeError("NeMo result must contain string text")
    if not isinstance(language, str):
        raise TypeError("NeMo result language must be a string")
    metadata: dict[str, object] = {}
    if timestamp is not None:
        metadata["timestamps"] = _json_primitive(timestamp)
    return text, language, metadata


def _json_primitive(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("NeMo metadata mapping keys must be strings")
            result[key] = _json_primitive(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_primitive(item) for item in value]
    if hasattr(value, "item"):
        scalar = value.item()
        if scalar is value:
            raise TypeError(f"Unsupported NeMo metadata type: {type(value).__name__}")
        return _json_primitive(scalar)
    raise TypeError(f"Unsupported NeMo metadata type: {type(value).__name__}")
