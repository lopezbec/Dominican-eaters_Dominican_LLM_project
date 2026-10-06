"""Lazy Transformers adapter for the local Voxtral Mini 3B 2507 checkpoint."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import platform
import tempfile
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from dominican_eaters.speech.asr import BackendDescriptor, Transcript

DEFAULT_MODEL = "mistralai/Voxtral-Mini-3B-2507"
PROMPT_TEMPLATE_ID = "voxtral-transcription-request-v1"
DEFAULT_MAX_AUDIO_SECONDS = 600.0
DEFAULT_MAX_NEW_TOKENS = 500


class VoxtralDependencyError(ImportError):
    """Raised when the isolated environment lacks the Voxtral runtime."""


class AudioPolicyError(ValueError):
    """Raised when an input violates the bounded local-audio policy."""


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


class _BatchInputs(Protocol):
    input_ids: Any

    def to(self, device: str, *, dtype: object) -> _BatchInputs: ...

    def keys(self) -> object: ...

    def __getitem__(self, key: str) -> object: ...


class _Processor(Protocol):
    def apply_transcription_request(
        self, *, language: str, audio: str, model_id: str
    ) -> _BatchInputs: ...

    def batch_decode(self, outputs: object, *, skip_special_tokens: bool) -> Sequence[str]: ...


class _Model(Protocol):
    device: object

    def generate(self, **inputs: object) -> Any: ...

    def eval(self) -> object: ...


class _Factory(Protocol):
    def from_pretrained(self, model: str, **options: object) -> object: ...


@dataclass(frozen=True, slots=True)
class VoxtralRuntimeBundle:
    processor_factory: _Factory
    model_factory: _Factory
    torch: _TorchRuntime


RuntimeLoader = Callable[[], VoxtralRuntimeBundle]


def load_voxtral_runtime() -> VoxtralRuntimeBundle:
    """Import heavyweight runtime modules only when the worker receives load."""

    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        processor_factory = transformers.AutoProcessor
        model_factory = transformers.VoxtralForConditionalGeneration
    except (ImportError, AttributeError) as error:
        raise VoxtralDependencyError(
            "Voxtral requires its isolated Transformers >=4.54 environment"
        ) from error
    return VoxtralRuntimeBundle(
        processor_factory=cast(_Factory, processor_factory),
        model_factory=cast(_Factory, model_factory),
        torch=cast(_TorchRuntime, torch),
    )


@dataclass(frozen=True, slots=True)
class VoxtralSettings:
    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    language: str = "es"
    device: Literal["auto", "cuda"] = "auto"
    precision: Literal["auto", "fp16"] = "auto"
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    timestamps: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Voxtral model must not be empty")
        if self.language != "es":
            raise ValueError("The Voxtral worker currently supports Spanish ('es') only")
        if self.device not in ("auto", "cuda"):
            raise ValueError("Voxtral Mini 3B requires CUDA in this runtime profile")
        if self.precision not in ("auto", "fp16"):
            raise ValueError("Voxtral Mini 3B requires FP16 in this runtime profile")
        if self.timestamps:
            raise ValueError("Local Voxtral Mini 3B 2507 timestamps are not supported")
        if self.max_audio_seconds <= 0 or self.max_audio_seconds > DEFAULT_MAX_AUDIO_SECONDS:
            raise ValueError(f"max_audio_seconds must be in (0, {DEFAULT_MAX_AUDIO_SECONDS:g}]")
        if self.max_new_tokens <= 0 or self.max_new_tokens > DEFAULT_MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be in [1, {DEFAULT_MAX_NEW_TOKENS}]")


@dataclass(slots=True)
class VoxtralBackend:
    settings: VoxtralSettings
    runtime_loader: RuntimeLoader = load_voxtral_runtime
    _runtime: VoxtralRuntimeBundle | None = field(default=None, init=False)
    _processor: _Processor | None = field(default=None, init=False)
    _model: _Model | None = field(default=None, init=False)
    _runtime_versions: dict[str, str] = field(
        default_factory=lambda: {"python": platform.python_version()}, init=False
    )

    @property
    def backend_id(self) -> str:
        return f"mistral-voxtral/{self.settings.model}"

    @property
    def descriptor(self) -> BackendDescriptor:
        loaded = self._model is not None
        return BackendDescriptor(
            backend_id=self.backend_id,
            model=self.settings.model,
            model_revision=self.settings.model_revision,
            language=self.settings.language,
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device="cuda" if loaded else None,
            effective_precision="fp16" if loaded else None,
            runtime_versions=dict(self._runtime_versions),
            options={
                "batch_size": 1,
                "max_audio_seconds": self.settings.max_audio_seconds,
                "max_new_tokens": self.settings.max_new_tokens,
                "timestamps": False,
                "transcription_method": "apply_transcription_request",
            },
            runtime_id="voxtral-transformers",
            model_revision_requested=self.settings.model_revision,
            prompt_template_id=PROMPT_TEMPLATE_ID,
            audio_preprocessing={"source": "AutoProcessor", "batch_size": 1},
            telemetry_source="torch_cuda_allocator",
        )

    def load(self) -> None:
        if self._model is not None:
            return
        runtime = self.runtime_loader()
        self._runtime = runtime
        self._capture_versions()
        if not runtime.torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Voxtral FP16 runtime profile")
        common: dict[str, object] = {}
        if self.settings.model_revision is not None:
            common["revision"] = self.settings.model_revision
        processor = cast(
            _Processor,
            runtime.processor_factory.from_pretrained(self.settings.model, **common),
        )
        model = cast(
            _Model,
            runtime.model_factory.from_pretrained(
                self.settings.model,
                torch_dtype=runtime.torch.float16,
                device_map="cuda",
                **common,
            ),
        )
        model.eval()
        self._processor = processor
        self._model = model

    def warmup(self) -> None:
        self._require_loaded()
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\0\0" * 16_000)
            self._generate(Path(temporary.name))

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        self._require_loaded()
        if not audio_path.is_absolute():
            raise ValueError("audio_path must be absolute")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        duration = duration_seconds if duration_seconds is not None else _wav_duration(audio_path)
        if duration is None:
            raise AudioPolicyError(
                "Audio duration is required to enforce the Voxtral context policy"
            )
        if duration <= 0:
            raise AudioPolicyError("Audio duration must be positive")
        if duration > self.settings.max_audio_seconds:
            raise AudioPolicyError(
                f"Audio duration {duration:.6g}s exceeds the configured Voxtral maximum "
                f"of {self.settings.max_audio_seconds:.6g}s"
            )
        runtime = cast(VoxtralRuntimeBundle, self._runtime)
        runtime.torch.cuda.reset_peak_memory_stats()
        text = self._generate(audio_path)
        runtime.torch.cuda.synchronize()
        return Transcript(
            text=text,
            language=self.settings.language,
            audio_duration_seconds=duration,
            gpu_peak_allocated_bytes=int(runtime.torch.cuda.max_memory_allocated()),
            gpu_peak_reserved_bytes=int(runtime.torch.cuda.max_memory_reserved()),
            metadata={
                "batch_size": 1,
                "max_audio_seconds": self.settings.max_audio_seconds,
                "max_new_tokens": self.settings.max_new_tokens,
                "prompt_template_id": PROMPT_TEMPLATE_ID,
                "timestamps": False,
            },
        )

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        runtime = self._runtime
        if runtime is not None:
            runtime.torch.cuda.empty_cache()
        self._runtime = None

    def _generate(self, audio_path: Path) -> str:
        model, processor, runtime = self._require_loaded()
        inputs = processor.apply_transcription_request(
            language=self.settings.language,
            audio=str(audio_path),
            model_id=self.settings.model,
        )
        inputs = inputs.to("cuda", dtype=runtime.torch.float16)
        generate_inputs = {key: inputs[key] for key in cast(Sequence[str], inputs.keys())}
        outputs = model.generate(**generate_inputs, max_new_tokens=self.settings.max_new_tokens)
        prompt_length = inputs.input_ids.shape[1]
        generated = outputs[:, prompt_length:]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != 1 or not isinstance(decoded[0], str):
            raise RuntimeError("Voxtral batch-one inference must return exactly one string")
        return decoded[0].strip()

    def _require_loaded(self) -> tuple[_Model, _Processor, VoxtralRuntimeBundle]:
        if self._model is None or self._processor is None or self._runtime is None:
            raise RuntimeError("Voxtral backend is not loaded")
        return self._model, self._processor, self._runtime

    def _capture_versions(self) -> None:
        for package in ("transformers", "torch", "accelerate", "soundfile"):
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
