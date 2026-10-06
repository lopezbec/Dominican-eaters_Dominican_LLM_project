"""Lazy native-Transformers adapter for Qwen3-ASR."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import platform
import tempfile
import wave
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from dominican_eaters.speech.asr import BackendDescriptor, Transcript


class Qwen3ASRDependencyError(ImportError):
    """Raised when the isolated environment lacks its runtime dependencies."""


class _CudaRuntime(Protocol):
    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def synchronize(self) -> None: ...

    def max_memory_allocated(self) -> int: ...

    def max_memory_reserved(self) -> int: ...


class _TorchRuntime(Protocol):
    cuda: _CudaRuntime
    float16: object
    float32: object

    def inference_mode(self) -> AbstractContextManager[None]: ...


class _QwenModel(Protocol):
    device: object
    dtype: object

    def to(self, device: str) -> _QwenModel: ...

    def eval(self) -> object: ...

    def generate(self, **inputs: Any) -> Any: ...


class _ModelFactory(Protocol):
    def from_pretrained(self, model: str, **kwargs: object) -> _QwenModel: ...


class _Processor(Protocol):
    def apply_transcription_request(self, **kwargs: object) -> Any: ...

    def decode(self, token_ids: object, *, return_format: str) -> list[object]: ...


class _ProcessorFactory(Protocol):
    def from_pretrained(self, model: str, **kwargs: object) -> _Processor: ...


@dataclass(frozen=True, slots=True)
class Qwen3RuntimeBundle:
    torch: _TorchRuntime
    model_factory: _ModelFactory
    processor_factory: _ProcessorFactory


RuntimeLoader = Callable[[], Qwen3RuntimeBundle]


def load_qwen3_runtime() -> Qwen3RuntimeBundle:
    """Import heavyweight dependencies only when the worker receives ``load``."""

    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise Qwen3ASRDependencyError(
            "Qwen3-ASR requires the isolated workers/qwen3_asr environment"
        ) from error
    return Qwen3RuntimeBundle(
        torch=cast(_TorchRuntime, torch),
        model_factory=cast(_ModelFactory, transformers.AutoModelForMultimodalLM),
        processor_factory=cast(_ProcessorFactory, transformers.AutoProcessor),
    )


@dataclass(frozen=True, slots=True)
class Qwen3Settings:
    model: str
    model_revision: str | None = None
    language: Literal["es", "auto"] = "es"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Qwen3-ASR model must not be empty")
        if self.language not in ("es", "auto"):
            raise ValueError("Qwen3-ASR language must be 'es' or 'auto'")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Unsupported Qwen3-ASR device: {self.device}")
        if self.precision not in ("auto", "fp32", "fp16"):
            raise ValueError(f"Unsupported Qwen3-ASR precision: {self.precision}")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("Qwen3-ASR FP16 precision requires CUDA")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(slots=True)
class Qwen3ASRBackend:
    settings: Qwen3Settings
    runtime_loader: RuntimeLoader = load_qwen3_runtime
    _runtime: Qwen3RuntimeBundle | None = field(default=None, init=False)
    _model: _QwenModel | None = field(default=None, init=False)
    _processor: _Processor | None = field(default=None, init=False)
    _resolved_device: Literal["cpu", "cuda"] | None = field(default=None, init=False)
    _resolved_precision: Literal["fp32", "fp16"] | None = field(default=None, init=False)
    _runtime_versions: dict[str, str] = field(
        default_factory=lambda: {"python": platform.python_version()}, init=False
    )

    @property
    def backend_id(self) -> str:
        return f"qwen3-asr/{self.settings.model}"

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            model=self.settings.model,
            model_revision=self.settings.model_revision,
            language=self.settings.language,
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device=self._resolved_device,
            effective_precision=self._resolved_precision,
            runtime_versions=dict(self._runtime_versions),
            options={
                "batch_size": 1,
                "max_new_tokens": self.settings.max_new_tokens,
                "attention_implementation": "sdpa",
                "timestamps": False,
            },
        )

    def load(self) -> None:
        if self._model is not None:
            return
        runtime = self.runtime_loader()
        self._runtime = runtime
        self._capture_versions()
        cuda_available = runtime.torch.cuda.is_available()
        if self.settings.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA was requested for Qwen3-ASR but is unavailable")
        device: Literal["cpu", "cuda"] = (
            "cuda"
            if self.settings.device == "cuda" or (self.settings.device == "auto" and cuda_available)
            else "cpu"
        )
        if device == "cpu" and self.settings.precision == "fp16":
            raise RuntimeError("Qwen3-ASR FP16 precision requires CUDA")
        precision: Literal["fp32", "fp16"] = (
            "fp16"
            if self.settings.precision == "auto" and device == "cuda"
            else "fp32"
            if self.settings.precision == "auto"
            else self.settings.precision
        )
        dtype = runtime.torch.float16 if precision == "fp16" else runtime.torch.float32
        model_kwargs: dict[str, object] = {
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        processor_kwargs: dict[str, object] = {}
        if self.settings.model_revision is not None:
            model_kwargs["revision"] = self.settings.model_revision
            processor_kwargs["revision"] = self.settings.model_revision
        processor = runtime.processor_factory.from_pretrained(
            self.settings.model, **processor_kwargs
        )
        model = runtime.model_factory.from_pretrained(self.settings.model, **model_kwargs)
        model = model.to(device)
        model.eval()
        self._processor = processor
        self._model = model
        self._resolved_device = device
        self._resolved_precision = precision

    def warmup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\0\0" * 16_000)
            self.transcribe(Path(temporary.name), duration_seconds=1.0)

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        model, processor, runtime = self._require_loaded()
        if not audio_path.is_absolute():
            raise ValueError("audio_path must be absolute")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        duration = duration_seconds if duration_seconds is not None else _wav_duration(audio_path)
        if self._resolved_device == "cuda":
            runtime.torch.cuda.reset_peak_memory_stats()
        request: dict[str, object] = {"audio": str(audio_path)}
        if self.settings.language == "es":
            request["language"] = "Spanish"
        inputs = processor.apply_transcription_request(**request).to(model.device, model.dtype)
        with runtime.torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
            )
        if self._resolved_device == "cuda":
            runtime.torch.cuda.synchronize()
        input_ids = inputs["input_ids"]
        generated_ids = output_ids[:, input_ids.shape[1] :]
        decoded = processor.decode(generated_ids, return_format="parsed")
        text, detected_language = _normalize_decoded(decoded, self.settings.language)
        allocated: int | None = None
        reserved: int | None = None
        if self._resolved_device == "cuda":
            allocated = int(runtime.torch.cuda.max_memory_allocated())
            reserved = int(runtime.torch.cuda.max_memory_reserved())
        return Transcript(
            text=text,
            language=detected_language,
            audio_duration_seconds=duration,
            gpu_peak_allocated_bytes=allocated,
            gpu_peak_reserved_bytes=reserved,
            metadata={
                "model": self.settings.model,
                "device": self._resolved_device,
                "precision": self._resolved_precision,
                "language_mode": "forced" if self.settings.language == "es" else "detected",
                "batch_size": 1,
                "timestamps": False,
            },
        )

    def close(self) -> None:
        self._processor = None
        self._model = None
        gc.collect()
        if self._runtime is not None and self._resolved_device == "cuda":
            self._runtime.torch.cuda.empty_cache()
        self._runtime = None

    def _require_loaded(self) -> tuple[_QwenModel, _Processor, Qwen3RuntimeBundle]:
        if self._model is None or self._processor is None or self._runtime is None:
            raise RuntimeError("Qwen3-ASR backend is not loaded")
        return self._model, self._processor, self._runtime

    def _capture_versions(self) -> None:
        for package in ("torch", "transformers"):
            try:
                self._runtime_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._runtime_versions[package] = "unknown"


def _wav_duration(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            return audio.getnframes() / rate if rate > 0 else None
    except (EOFError, wave.Error):
        return None


def _normalize_decoded(decoded: list[object], requested_language: str) -> tuple[str, str]:
    if len(decoded) != 1:
        raise RuntimeError(f"Qwen3-ASR must return exactly one result; received {len(decoded)}")
    result = decoded[0]
    if not isinstance(result, Mapping):
        raise TypeError("Qwen3-ASR parsed result must be a mapping")
    text = result.get("transcription")
    language = result.get("language")
    if not isinstance(text, str):
        raise TypeError("Qwen3-ASR parsed result must contain string transcription")
    if requested_language == "es":
        return text, "es"
    if not isinstance(language, str) or not language.strip():
        raise TypeError("Qwen3-ASR auto-detected result must contain a language")
    normalized = {"Spanish": "es", "English": "en"}.get(language, language)
    return text, normalized
