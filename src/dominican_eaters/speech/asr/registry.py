"""Lightweight source of truth for ASR runtimes, backends, and model presets.

This module intentionally contains data and validation only. Importing it must never import a
model runtime, inspect a GPU, or access the network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

BackendName = Literal[
    "whisper",
    "parakeet",
    "canary",
    "granite",
    "qwen3_asr",
    "voxtral",
    "qwen2_audio",
    "granite_3_3",
]
RuntimeExecution = Literal["inline", "worker"]
PresetStatus = Literal["current", "candidate", "planned", "experimental", "blocked"]
LongFormPolicy = Literal[
    "chunked",
    "runtime-managed",
    "native-30-minutes",
    "bounded-feasibility-clips",
    "unverified",
]


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_text_items(
    values: Iterable[str], field_name: str, *, allow_empty: bool = True
) -> None:
    items = tuple(values)
    if not allow_empty and not items:
        raise ValueError(f"{field_name} must not be empty")
    if any(not item.strip() for item in items):
        raise ValueError(f"{field_name} must contain non-empty strings")


def _validate_status_reason(status: PresetStatus, reason: str | None) -> None:
    if status == "current" and reason is not None:
        raise ValueError("current entries cannot define a blocking reason")
    if status != "current" and (reason is None or not reason.strip()):
        raise ValueError(f"{status} entries must explain why they are not current")


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    """Dependency and process boundary for one isolated ASR runtime."""

    runtime_id: str
    label: str
    execution: RuntimeExecution
    worker_module: str | None
    interpreter_env_var: str | None
    required_modules: tuple[str, ...]
    required_executables: tuple[str, ...] = ()
    supported_python: str = ">=3.11"

    def __post_init__(self) -> None:
        _require_text(self.runtime_id, "runtime ID")
        _require_text(self.label, "runtime label")
        _require_text(self.supported_python, "supported Python range")
        if self.execution == "inline" and self.worker_module is not None:
            raise ValueError("inline runtimes cannot define a worker module")
        if self.execution == "worker" and not self.worker_module:
            raise ValueError("worker runtimes must define a worker module")
        if self.execution == "inline" and self.interpreter_env_var is not None:
            raise ValueError("inline runtimes cannot define a worker interpreter variable")
        if self.execution == "worker" and not self.interpreter_env_var:
            raise ValueError("worker runtimes must define a worker interpreter variable")
        if self.interpreter_env_var is not None:
            _require_text(self.interpreter_env_var, "interpreter environment variable")
        _require_text_items(self.required_modules, "required modules")
        _require_text_items(self.required_executables, "required executables")


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Stable adapter-family metadata, independent of CLI presentation."""

    name: BackendName
    label: str
    default_model: str
    runtime_id: str
    status: PresetStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "backend name")
        _require_text(self.label, "backend label")
        _require_text(self.default_model, "backend default model")
        _require_text(self.runtime_id, "backend runtime ID")
        _validate_status_reason(self.status, self.reason)

    @property
    def runtime(self) -> RuntimeSpec:
        """Return the runtime without importing its implementation."""

        return RUNTIME_SPECS[self.runtime_id]

    # These properties preserve the current BackendSpec composition vocabulary while making the
    # runtime record authoritative for dependency and process details.
    @property
    def execution(self) -> RuntimeExecution:
        return self.runtime.execution

    @property
    def worker_module(self) -> str | None:
        return self.runtime.worker_module

    @property
    def required_modules(self) -> tuple[str, ...]:
        return self.runtime.required_modules

    @property
    def required_executables(self) -> tuple[str, ...]:
        return self.runtime.required_executables


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capabilities supported by the exact preset/runtime combination."""

    timestamps: bool
    streaming: bool
    language_detection: bool
    long_form_policy: LongFormPolicy
    supported_devices: tuple[str, ...]
    supported_precisions: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text_items(self.supported_devices, "supported devices", allow_empty=False)
        _require_text_items(self.supported_precisions, "supported precisions", allow_empty=False)


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """Publication identity for one exact model and runtime policy."""

    preset_id: str
    label: str
    backend: BackendName
    model: str
    model_revision: str | None
    runtime_id: str
    precision: str | None
    quantization: str | None
    language: str
    status: PresetStatus
    capabilities: ModelCapabilities
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.preset_id, "preset ID")
        _require_text(self.label, "preset label")
        _require_text(self.backend, "preset backend")
        _require_text(self.model, "preset model")
        _require_text(self.runtime_id, "preset runtime ID")
        _require_text(self.language, "preset language")
        if self.model_revision is not None:
            _require_text(self.model_revision, "model revision")
        if self.precision is not None:
            _require_text(self.precision, "preset precision")
            if self.precision not in self.capabilities.supported_precisions:
                raise ValueError(
                    f"preset precision {self.precision!r} is not supported by its capabilities"
                )
        if self.quantization is not None:
            _require_text(self.quantization, "preset quantization")
        _validate_status_reason(self.status, self.reason)

    @property
    def runnable(self) -> bool:
        """Whether this preset is currently supported by an integrated adapter."""

        return self.status in {"current", "candidate"}


RUNTIME_SPECS: Mapping[str, RuntimeSpec] = MappingProxyType(
    {
        "whisper-native": RuntimeSpec(
            runtime_id="whisper-native",
            label="OpenAI Whisper native",
            execution="inline",
            worker_module=None,
            interpreter_env_var=None,
            required_modules=("whisper", "torch"),
            required_executables=("ffmpeg",),
            supported_python=">=3.8",
        ),
        "nemo-worker": RuntimeSpec(
            runtime_id="nemo-worker",
            label="NVIDIA NeMo worker",
            execution="worker",
            worker_module="dominican_eaters_nemo",
            interpreter_env_var="DOMINICAN_EATERS_NEMO_PYTHON",
            required_modules=("dominican_eaters_nemo", "nemo.collections.asr", "torch"),
            supported_python=">=3.11,<3.13",
        ),
        "granite-4.1-transformers": RuntimeSpec(
            runtime_id="granite-4.1-transformers",
            label="IBM Granite Speech 4.1 Transformers worker",
            execution="worker",
            worker_module="dominican_eaters_granite",
            interpreter_env_var="DOMINICAN_EATERS_GRANITE_PYTHON",
            required_modules=(
                "dominican_eaters_granite",
                "transformers",
                "torch",
                "torchaudio",
                "soundfile",
            ),
            supported_python=">=3.10",
        ),
        "qwen3-asr-transformers": RuntimeSpec(
            runtime_id="qwen3-asr-transformers",
            label="Qwen3-ASR Transformers worker",
            execution="worker",
            worker_module="dominican_eaters_qwen3_asr",
            interpreter_env_var="DOMINICAN_EATERS_QWEN3_ASR_PYTHON",
            required_modules=(
                "dominican_eaters_qwen3_asr",
                "transformers",
                "torch",
                "soundfile",
            ),
            supported_python=">=3.12,<3.13",
        ),
        "voxtral-transformers": RuntimeSpec(
            runtime_id="voxtral-transformers",
            label="Mistral Voxtral Transformers worker",
            execution="worker",
            worker_module="dominican_eaters_voxtral",
            interpreter_env_var="DOMINICAN_EATERS_VOXTRAL_PYTHON",
            required_modules=(
                "dominican_eaters_voxtral",
                "transformers",
                "torch",
                "mistral_common",
                "soundfile",
            ),
            supported_python=">=3.10",
        ),
        "qwen2-audio-transformers": RuntimeSpec(
            runtime_id="qwen2-audio-transformers",
            label="Qwen2-Audio Transformers feasibility worker",
            execution="worker",
            worker_module="dominican_eaters_qwen2_audio",
            interpreter_env_var="DOMINICAN_EATERS_QWEN2_AUDIO_PYTHON",
            required_modules=(
                "dominican_eaters_qwen2_audio",
                "transformers",
                "torch",
                "librosa",
            ),
            supported_python=">=3.10",
        ),
        "granite-3.3-transformers": RuntimeSpec(
            runtime_id="granite-3.3-transformers",
            label="IBM Granite Speech 3.3 Transformers feasibility worker",
            execution="worker",
            worker_module="dominican_eaters_granite",
            interpreter_env_var="DOMINICAN_EATERS_GRANITE_PYTHON",
            required_modules=(
                "dominican_eaters_granite",
                "transformers",
                "torch",
                "torchaudio",
                "peft",
                "soundfile",
            ),
            supported_python=">=3.10",
        ),
    }
)


BACKEND_SPECS: Mapping[BackendName, BackendSpec] = MappingProxyType(
    {
        "whisper": BackendSpec("whisper", "Whisper", "base", "whisper-native", "current"),
        "parakeet": BackendSpec(
            "parakeet",
            "Parakeet",
            "nvidia/parakeet-tdt-0.6b-v3",
            "nemo-worker",
            "current",
        ),
        "canary": BackendSpec("canary", "Canary", "nvidia/canary-1b-v2", "nemo-worker", "current"),
        "granite": BackendSpec(
            "granite",
            "Granite Speech 4.1",
            "ibm-granite/granite-speech-4.1-2b",
            "granite-4.1-transformers",
            "candidate",
            "Dedicated worker is integrated; reviewed T4 smoke evidence is pending.",
        ),
        "qwen3_asr": BackendSpec(
            "qwen3_asr",
            "Qwen3-ASR",
            "Qwen/Qwen3-ASR-1.7B-hf",
            "qwen3-asr-transformers",
            "candidate",
            "Dedicated worker is integrated; reviewed T4 smoke evidence is pending.",
        ),
        "voxtral": BackendSpec(
            "voxtral",
            "Voxtral Mini",
            "mistralai/Voxtral-Mini-3B-2507",
            "voxtral-transformers",
            "candidate",
            "Dedicated worker is integrated; constrained-context T4 smoke evidence is pending.",
        ),
        "qwen2_audio": BackendSpec(
            "qwen2_audio",
            "Qwen2-Audio",
            "Qwen/Qwen2-Audio-7B-Instruct",
            "qwen2-audio-transformers",
            "experimental",
            "Spanish ASR and a reproducible T4-compatible quantization remain unproven.",
        ),
        "granite_3_3": BackendSpec(
            "granite_3_3",
            "Granite Speech 3.3",
            "ibm-granite/granite-speech-3.3-8b",
            "granite-3.3-transformers",
            "blocked",
            "Published BF16 weights exceed T4 VRAM before runtime overhead, and no validated "
            "quantized runtime is available.",
        ),
    }
)


_WHISPER_CAPABILITIES = ModelCapabilities(
    # The native checkpoint supports timestamps, but the current adapter does not yet expose the
    # requested timestamp mode. P1-03 promotes this only after the output contract is implemented.
    timestamps=False,
    streaming=False,
    language_detection=True,
    long_form_policy="chunked",
    supported_devices=("cpu", "cuda"),
    supported_precisions=("fp16", "fp32"),
)
_NEMO_CAPABILITIES = ModelCapabilities(
    timestamps=True,
    streaming=False,
    language_detection=False,
    long_form_policy="runtime-managed",
    supported_devices=("cpu", "cuda"),
    supported_precisions=("fp16", "fp32", "bf16"),
)

MODEL_PRESETS: Mapping[str, ModelPreset] = MappingProxyType(
    {
        "whisper-base": ModelPreset(
            "whisper-base",
            "Whisper Base",
            "whisper",
            "base",
            None,
            "whisper-native",
            "fp16",
            None,
            "es",
            "current",
            _WHISPER_CAPABILITIES,
        ),
        "whisper-large-v3": ModelPreset(
            "whisper-large-v3",
            "Whisper Large v3",
            "whisper",
            "large-v3",
            None,
            "whisper-native",
            "fp16",
            None,
            "es",
            "current",
            _WHISPER_CAPABILITIES,
        ),
        "whisper-turbo": ModelPreset(
            "whisper-turbo",
            "Whisper Turbo",
            "whisper",
            "turbo",
            None,
            "whisper-native",
            "fp16",
            None,
            "es",
            "current",
            _WHISPER_CAPABILITIES,
        ),
        "parakeet-tdt-0.6b-v3": ModelPreset(
            "parakeet-tdt-0.6b-v3",
            "Parakeet TDT 0.6B v3",
            "parakeet",
            "nvidia/parakeet-tdt-0.6b-v3",
            None,
            "nemo-worker",
            "fp16",
            None,
            "es",
            "current",
            _NEMO_CAPABILITIES,
        ),
        "canary-1b-v2": ModelPreset(
            "canary-1b-v2",
            "Canary 1B v2",
            "canary",
            "nvidia/canary-1b-v2",
            None,
            "nemo-worker",
            "fp16",
            None,
            "es",
            "current",
            _NEMO_CAPABILITIES,
        ),
        "granite-speech-4.1-2b": ModelPreset(
            "granite-speech-4.1-2b",
            "Granite Speech 4.1 2B",
            "granite",
            "ibm-granite/granite-speech-4.1-2b",
            None,
            "granite-4.1-transformers",
            "fp16",
            None,
            "es",
            "candidate",
            ModelCapabilities(
                timestamps=False,
                streaming=False,
                language_detection=False,
                long_form_policy="runtime-managed",
                supported_devices=("cuda",),
                supported_precisions=("fp16",),
            ),
            "Worker is integrated; clean T4 validation is pending.",
        ),
        "qwen3-asr-1.7b": ModelPreset(
            "qwen3-asr-1.7b",
            "Qwen3-ASR 1.7B",
            "qwen3_asr",
            "Qwen/Qwen3-ASR-1.7B-hf",
            None,
            "qwen3-asr-transformers",
            "fp16",
            None,
            "es",
            "candidate",
            ModelCapabilities(
                timestamps=False,
                streaming=False,
                language_detection=True,
                long_form_policy="runtime-managed",
                supported_devices=("cuda",),
                supported_precisions=("fp16",),
            ),
            "Offline worker is integrated; T4 validation and timestamps remain pending.",
        ),
        "voxtral-mini-3b-2507": ModelPreset(
            "voxtral-mini-3b-2507",
            "Voxtral Mini 3B 2507",
            "voxtral",
            "mistralai/Voxtral-Mini-3B-2507",
            None,
            "voxtral-transformers",
            "fp16",
            None,
            "es",
            "candidate",
            ModelCapabilities(
                timestamps=False,
                streaming=False,
                language_detection=True,
                long_form_policy="native-30-minutes",
                supported_devices=("cuda",),
                supported_precisions=("fp16",),
            ),
            "Worker is integrated; conservative-context T4 validation is pending.",
        ),
        "qwen2-audio-7b-instruct": ModelPreset(
            "qwen2-audio-7b-instruct",
            "Qwen2-Audio 7B Instruct",
            "qwen2_audio",
            "Qwen/Qwen2-Audio-7B-Instruct",
            None,
            "qwen2-audio-transformers",
            None,
            None,
            "es",
            "experimental",
            ModelCapabilities(
                timestamps=False,
                streaming=False,
                language_detection=False,
                long_form_policy="bounded-feasibility-clips",
                supported_devices=("cuda",),
                supported_precisions=("fp16",),
            ),
            "Spanish ASR and a reproducible T4-compatible quantization remain unproven.",
        ),
        "granite-speech-3.3-8b": ModelPreset(
            "granite-speech-3.3-8b",
            "Granite Speech 3.3 8B",
            "granite_3_3",
            "ibm-granite/granite-speech-3.3-8b",
            "3.3.2",
            "granite-3.3-transformers",
            None,
            None,
            "es",
            "blocked",
            ModelCapabilities(
                timestamps=False,
                streaming=False,
                language_detection=False,
                long_form_policy="unverified",
                supported_devices=("cuda",),
                supported_precisions=("fp16",),
            ),
            "Published BF16 weights exceed T4 VRAM before overhead; no validated quantization "
            "is available.",
        ),
    }
)


CURRENT_BACKEND_SPECS: Mapping[BackendName, BackendSpec] = MappingProxyType(
    {name: spec for name, spec in BACKEND_SPECS.items() if spec.status in {"current", "candidate"}}
)
DEFAULT_MODELS: Mapping[BackendName, str] = MappingProxyType(
    {name: spec.default_model for name, spec in CURRENT_BACKEND_SPECS.items()}
)
BACKEND_OPTIONS = tuple((spec.label, name) for name, spec in CURRENT_BACKEND_SPECS.items())


def presets_with_status(*statuses: PresetStatus) -> tuple[ModelPreset, ...]:
    """Return presets in stable registry order, optionally filtered by lifecycle status."""

    accepted = frozenset(statuses)
    return tuple(
        preset for preset in MODEL_PRESETS.values() if not accepted or preset.status in accepted
    )


def runnable_presets() -> tuple[ModelPreset, ...]:
    """Return only presets backed by currently integrated adapters."""

    return presets_with_status("current", "candidate")


def _validate_registry() -> None:
    for key, runtime in RUNTIME_SPECS.items():
        if key != runtime.runtime_id:
            raise ValueError(f"runtime key {key!r} does not match {runtime.runtime_id!r}")
    for key, backend in BACKEND_SPECS.items():
        if key != backend.name:
            raise ValueError(f"backend key {key!r} does not match {backend.name!r}")
        if backend.runtime_id not in RUNTIME_SPECS:
            raise ValueError(f"backend {key!r} references unknown runtime {backend.runtime_id!r}")
    for key, preset in MODEL_PRESETS.items():
        if key != preset.preset_id:
            raise ValueError(f"preset key {key!r} does not match {preset.preset_id!r}")
        if preset.backend not in BACKEND_SPECS:
            raise ValueError(f"preset {key!r} references unknown backend {preset.backend!r}")
        if preset.runtime_id not in RUNTIME_SPECS:
            raise ValueError(f"preset {key!r} references unknown runtime {preset.runtime_id!r}")
        if BACKEND_SPECS[preset.backend].runtime_id != preset.runtime_id:
            raise ValueError(f"preset {key!r} does not use its backend runtime")
        if preset.runnable and BACKEND_SPECS[preset.backend].status not in {
            "current",
            "candidate",
        }:
            raise ValueError(f"runnable preset {key!r} uses a non-runnable backend")


_validate_registry()


__all__ = [
    "BACKEND_OPTIONS",
    "BACKEND_SPECS",
    "CURRENT_BACKEND_SPECS",
    "DEFAULT_MODELS",
    "MODEL_PRESETS",
    "RUNTIME_SPECS",
    "BackendName",
    "BackendSpec",
    "LongFormPolicy",
    "ModelCapabilities",
    "ModelPreset",
    "PresetStatus",
    "RuntimeExecution",
    "RuntimeSpec",
    "presets_with_status",
    "runnable_presets",
]
