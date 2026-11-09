"""Lazy, injection-friendly adapter for IBM Granite Speech 4.1."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import math
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

from dominican_eaters.speech.asr import BackendDescriptor, Transcript

DEFAULT_MODEL = "ibm-granite/granite-speech-4.1-2b"
PROMPT_TEMPLATE_ID = "granite-speech-4.1-asr-punctuated-v1"
PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."
ENVIRONMENT_LOCK_ID = "dominican-eaters-granite-transformers-v2"
DEFAULT_CHUNK_LENGTH_SECONDS = 30.0
MINIMUM_FINAL_CHUNK_SECONDS = 1.0


class GraniteDependencyError(ImportError):
    """Raised when the isolated environment lacks Granite runtime dependencies."""


class _CudaRuntime(Protocol):
    def is_available(self) -> bool: ...
    def empty_cache(self) -> None: ...
    def reset_peak_memory_stats(self) -> None: ...
    def synchronize(self) -> None: ...
    def max_memory_allocated(self) -> int: ...
    def max_memory_reserved(self) -> int: ...


class _Tensor(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    def to(self, device: str) -> _Tensor: ...
    def __getitem__(self, key: object) -> _Tensor: ...
    def unsqueeze(self, dimension: int) -> _Tensor: ...


class _AudioSamples(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, key: slice) -> _AudioSamples: ...


class _TorchRuntime(Protocol):
    cuda: _CudaRuntime
    float16: object
    float32: object

    def zeros(self, size: int) -> _Tensor: ...
    def from_numpy(self, samples: _AudioSamples) -> _Tensor: ...


class _AudioRuntime(Protocol):
    def read(self, path: str, *, dtype: str, always_2d: bool) -> tuple[_AudioSamples, int]: ...


class _Batch(Protocol):
    def to(self, device: str) -> _Batch: ...
    def __getitem__(self, key: str) -> _Tensor: ...


class _Tokenizer(Protocol):
    def apply_chat_template(
        self, chat: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str: ...

    def batch_decode(
        self, tokens: _Tensor, *, add_special_tokens: bool, skip_special_tokens: bool
    ) -> list[str]: ...


class _Processor(Protocol):
    tokenizer: _Tokenizer

    def __call__(
        self, prompt: str, waveform: _Tensor, *, device: str, return_tensors: str
    ) -> _Batch: ...


class _ProcessorFactory(Protocol):
    def from_pretrained(self, model: str, **kwargs: object) -> _Processor: ...


class _Model(Protocol):
    def to(self, device: str) -> _Model: ...
    def eval(self) -> object: ...
    def generate(self, **kwargs: object) -> _Tensor: ...


class _ModelFactory(Protocol):
    def from_pretrained(self, model: str, **kwargs: object) -> _Model: ...


@dataclass(frozen=True, slots=True)
class GraniteRuntimeBundle:
    torch: _TorchRuntime
    soundfile: _AudioRuntime
    processor_factory: _ProcessorFactory
    model_factory: _ModelFactory


RuntimeLoader = Callable[[], GraniteRuntimeBundle]


def _fixed_window_spans(
    sample_count: int,
    chunk_samples: int,
    *,
    minimum_final_samples: int,
) -> list[tuple[int, int]]:
    """Return contiguous bounded windows without leaving an undersized tail."""

    boundaries = list(range(0, sample_count, chunk_samples))
    boundaries.append(sample_count)
    if len(boundaries) > 2:
        final_samples = boundaries[-1] - boundaries[-2]
        combined_final_samples = boundaries[-1] - boundaries[-3]
        target_final_samples = min(minimum_final_samples, combined_final_samples // 2)
        if final_samples < target_final_samples:
            boundaries[-2] = sample_count - target_final_samples
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def load_granite_runtime() -> GraniteRuntimeBundle:
    """Import heavyweight dependencies only for a real model load."""

    try:
        torch = importlib.import_module("torch")
        soundfile = importlib.import_module("soundfile")
        transformers = importlib.import_module("transformers")
        processor_factory = transformers.AutoProcessor
        model_factory = transformers.AutoModelForSpeechSeq2Seq
    except (ImportError, AttributeError) as error:
        raise GraniteDependencyError(
            "Granite requires the isolated workers/granite Transformers environment"
        ) from error
    return GraniteRuntimeBundle(
        torch=cast(_TorchRuntime, torch),
        soundfile=cast(_AudioRuntime, soundfile),
        processor_factory=cast(_ProcessorFactory, processor_factory),
        model_factory=cast(_ModelFactory, model_factory),
    )


@dataclass(frozen=True, slots=True)
class GraniteSettings:
    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    language: str = "es"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    max_new_tokens: int = 200
    chunk_length_seconds: float = DEFAULT_CHUNK_LENGTH_SECONDS
    preset_id: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Granite model must not be empty")
        if self.language != "es":
            raise ValueError("The Granite worker currently supports Spanish ('es') only")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Unsupported Granite device: {self.device}")
        if self.precision not in ("auto", "fp32", "fp16"):
            raise ValueError(f"Unsupported Granite precision: {self.precision}")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("Granite FP16 requires CUDA")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.chunk_length_seconds) or self.chunk_length_seconds <= 0:
            raise ValueError("chunk_length_seconds must be finite and positive")


@dataclass(slots=True)
class GraniteBackend:
    settings: GraniteSettings
    runtime_loader: RuntimeLoader = load_granite_runtime
    _runtime: GraniteRuntimeBundle | None = field(default=None, init=False)
    _processor: _Processor | None = field(default=None, init=False)
    _model: _Model | None = field(default=None, init=False)
    _resolved_device: Literal["cpu", "cuda"] | None = field(default=None, init=False)
    _resolved_precision: Literal["fp32", "fp16"] | None = field(default=None, init=False)
    _runtime_versions: dict[str, str] = field(
        default_factory=lambda: {"python": platform.python_version()}, init=False
    )

    @property
    def backend_id(self) -> str:
        return f"ibm-granite/{self.settings.model}"

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
                "max_new_tokens": self.settings.max_new_tokens,
                "batch_size": 1,
                "segmentation": "fixed",
                "chunk_length_seconds": self.settings.chunk_length_seconds,
            },
            preset_id=self.settings.preset_id,
            runtime_id="granite",
            model_revision_requested=self.settings.model_revision,
            quantization="none",
            prompt_template_id=PROMPT_TEMPLATE_ID,
            audio_preprocessing={"channels": 1, "sample_rate_hz": 16000, "normalize": True},
            environment_lock_id=ENVIRONMENT_LOCK_ID,
            telemetry_source="pytorch_allocator",
        )

    def load(self) -> None:
        if self._model is not None:
            return
        runtime = self.runtime_loader()
        available = runtime.torch.cuda.is_available()
        if self.settings.device == "cuda" and not available:
            raise RuntimeError("CUDA was requested for Granite but is unavailable")
        device: Literal["cpu", "cuda"] = (
            "cuda"
            if self.settings.device == "cuda" or (self.settings.device == "auto" and available)
            else "cpu"
        )
        precision: Literal["fp32", "fp16"] = (
            "fp16"
            if self.settings.precision == "auto" and device == "cuda"
            else "fp32"
            if self.settings.precision == "auto"
            else self.settings.precision
        )
        if precision == "fp16" and device != "cuda":
            raise RuntimeError("Granite FP16 requires CUDA")
        revision = {}
        if self.settings.model_revision is not None:
            revision["revision"] = self.settings.model_revision
        processor = runtime.processor_factory.from_pretrained(self.settings.model, **revision)
        dtype = runtime.torch.float16 if precision == "fp16" else runtime.torch.float32
        model = runtime.model_factory.from_pretrained(
            self.settings.model,
            torch_dtype=dtype,
            **revision,
        )
        model = model.to(device)
        model.eval()
        self._runtime = runtime
        self._processor = processor
        self._model = model
        self._resolved_device = device
        self._resolved_precision = precision
        self._capture_versions()

    def warmup(self) -> None:
        runtime = self._require_runtime()
        self._infer(runtime.torch.zeros(16_000))

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        if not audio_path.is_absolute():
            raise ValueError("audio_path must be absolute")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        runtime = self._require_runtime()
        samples, sample_rate = runtime.soundfile.read(
            str(audio_path), dtype="float32", always_2d=False
        )
        if len(samples.shape) != 1 or sample_rate != 16_000:
            raise ValueError("Granite input must be mono 16 kHz audio")
        sample_count = int(samples.shape[0])
        if sample_count == 0:
            raise ValueError("Granite input must contain audio samples")
        chunk_samples = max(1, int(self.settings.chunk_length_seconds * sample_rate))
        spans = _fixed_window_spans(
            sample_count,
            chunk_samples,
            minimum_final_samples=int(MINIMUM_FINAL_CHUNK_SECONDS * sample_rate),
        )
        chunk_count = len(spans)
        resolved_duration = (
            duration_seconds if duration_seconds is not None else sample_count / sample_rate
        )
        if self._resolved_device == "cuda":
            runtime.torch.cuda.reset_peak_memory_stats()
        allocated = reserved = None
        chunks: list[str] = []
        try:
            for start, stop in spans:
                chunk = samples[start:stop]
                waveform = runtime.torch.from_numpy(chunk).unsqueeze(0)
                text = self._infer(waveform)
                if text:
                    chunks.append(text)
            if self._resolved_device == "cuda":
                runtime.torch.cuda.synchronize()
                allocated = int(runtime.torch.cuda.max_memory_allocated())
                reserved = int(runtime.torch.cuda.max_memory_reserved())
        finally:
            # A failed long sample must not poison every later request in this worker.
            if self._resolved_device == "cuda":
                runtime.torch.cuda.empty_cache()
        return Transcript(
            text=" ".join(chunks),
            language=self.settings.language,
            audio_duration_seconds=resolved_duration,
            gpu_peak_allocated_bytes=allocated,
            gpu_peak_reserved_bytes=reserved,
            metadata={
                "model": self.settings.model,
                "device": self._resolved_device,
                "precision": self._resolved_precision,
                "prompt_template_id": PROMPT_TEMPLATE_ID,
                "segmentation": "fixed",
                "chunk_length_seconds": self.settings.chunk_length_seconds,
                "chunk_count": chunk_count,
            },
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if self._runtime is not None and self._resolved_device == "cuda":
            self._runtime.torch.cuda.empty_cache()
        self._runtime = None

    def _infer(self, waveform: _Tensor) -> str:
        processor = self._processor
        model = self._model
        device = self._resolved_device
        if processor is None or model is None or device is None:
            raise RuntimeError("Granite backend is not loaded")
        prompt = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(prompt, waveform, device=device, return_tensors="pt").to(device)
        outputs = model.generate(
            **cast(dict[str, object], inputs),
            max_new_tokens=self.settings.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        input_tokens = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, input_tokens:].unsqueeze(0)
        decoded = processor.tokenizer.batch_decode(
            new_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise TypeError("Granite tokenizer must decode exactly one string")
        return decoded[0].strip()

    def _require_runtime(self) -> GraniteRuntimeBundle:
        if self._runtime is None or self._model is None or self._processor is None:
            raise RuntimeError("Granite backend is not loaded")
        return self._runtime

    def _capture_versions(self) -> None:
        for package in ("torch", "transformers", "soundfile"):
            try:
                self._runtime_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._runtime_versions[package] = "unknown"
