"""JSONL entry point for the isolated Granite Speech worker."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import logging
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO, cast

from dominican_eaters.speech.asr import BackendDescriptor, Transcript
from dominican_eaters.speech.asr.worker_protocol import (
    PROTOCOL_VERSION,
    JSONObject,
    JSONValue,
    PreflightCheck,
    make_preflight_result,
)
from dominican_eaters.speech.asr.worker_server import WorkerDispatcher, WorkerError
from dominican_eaters.speech.asr.worker_server import serve as serve_worker

from . import __version__
from .adapters import GraniteBackend, GraniteDependencyError, GraniteSettings

LOGGER = logging.getLogger("dominican_eaters_granite")


class WorkerBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor: ...
    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def transcribe(
        self, audio_path: Path, *, duration_seconds: float | None = None
    ) -> Transcript: ...
    def close(self) -> None: ...


BackendFactory = Callable[[GraniteSettings], WorkerBackend]


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    checks: tuple[PreflightCheck, ...]
    environment: JSONObject


RuntimeInspector = Callable[[str, str], RuntimeInspection]


class CudaInspector(Protocol):
    def is_available(self) -> bool: ...
    def device_count(self) -> int: ...


class WorkerService:
    def __init__(
        self,
        factories: Mapping[str, BackendFactory] | None = None,
        *,
        runtime_inspector: RuntimeInspector | None = None,
    ) -> None:
        self._factories = dict(factories or {"granite": GraniteBackend})
        self._runtime_inspector = runtime_inspector or inspect_granite_runtime
        self._backend: WorkerBackend | None = None

    def describe(self) -> JSONObject:
        result: JSONObject = {
            "worker": "dominican-eaters-granite",
            "worker_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "backends": cast(list[JSONValue], sorted(self._factories)),
        }
        if self._backend is not None:
            result["descriptor"] = _descriptor_payload(self._backend.descriptor)
        return result

    def preflight(self, params: JSONObject) -> JSONObject:
        try:
            backend_name, settings = self._settings_from_params(params)
        except (TypeError, ValueError, KeyError) as error:
            return make_preflight_result(
                [PreflightCheck("configuration", "failed", "error", str(error))],
                environment=_base_environment(),
            )
        inspection = self._runtime_inspector(settings.device, settings.precision)
        configuration = PreflightCheck(
            "configuration",
            "passed",
            "error",
            f"{backend_name} configuration is valid",
            details={
                "backend": backend_name,
                "model": settings.model,
                "device": settings.device,
                "precision": settings.precision,
                "batch_size": 1,
            },
        )
        environment = dict(inspection.environment)
        environment.update(
            {"backend": backend_name, "model": settings.model, "weights_loaded": False}
        )
        return make_preflight_result((configuration, *inspection.checks), environment=environment)

    def load(self, params: JSONObject) -> JSONObject:
        if self._backend is not None:
            raise RuntimeError("Granite worker is already loaded")
        backend_name, settings = self._settings_from_params(params)
        backend = self._factories[backend_name](settings)
        self._backend = backend
        try:
            backend.load()
        except Exception:
            self._backend = None
            backend.close()
            raise
        return {"descriptor": _descriptor_payload(backend.descriptor)}

    def warmup(self) -> JSONObject:
        backend = self._require_backend()
        backend.warmup()
        return {"descriptor": _descriptor_payload(backend.descriptor)}

    def transcribe(self, params: JSONObject) -> JSONObject:
        backend = self._require_backend()
        raw_duration = params.get("duration_seconds")
        duration = None if raw_duration is None else float(cast(int | float, raw_duration))
        transcript = backend.transcribe(
            Path(cast(str, params["audio_path"])), duration_seconds=duration
        )
        return {
            "text": transcript.text,
            "language": transcript.language,
            "audio_duration_seconds": transcript.audio_duration_seconds,
            "gpu_peak_allocated_bytes": transcript.gpu_peak_allocated_bytes,
            "gpu_peak_reserved_bytes": transcript.gpu_peak_reserved_bytes,
            "metadata": cast(dict[str, JSONValue], transcript.metadata),
        }

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()

    def _settings_from_params(self, params: JSONObject) -> tuple[str, GraniteSettings]:
        backend_name = cast(str, params["backend"])
        if backend_name not in self._factories:
            raise ValueError(f"Unsupported Granite backend {backend_name!r}; expected 'granite'")
        options = cast(dict[str, JSONValue], params["options"])
        unknown = sorted(set(options) - {"max_new_tokens"})
        if unknown:
            raise ValueError(f"Unknown Granite options: {', '.join(unknown)}")
        max_new_tokens = options.get("max_new_tokens", 200)
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        quantization = params.get("quantization")
        if quantization not in (None, "none"):
            raise ValueError("The Transformers Granite worker does not support quantization")
        prompt = params.get("prompt_template_id")
        if prompt not in (None, "granite-speech-4.1-asr-punctuated-v1"):
            raise ValueError("Unsupported Granite prompt_template_id")
        revision = params.get("model_revision")
        preset = params.get("preset")
        return backend_name, GraniteSettings(
            model=cast(str, params["model"]),
            model_revision=cast(str | None, revision),
            language=cast(str, params["language"]),
            device=cast(str, params["device"]),  # type: ignore[arg-type]
            precision=cast(str, params["precision"]),  # type: ignore[arg-type]
            max_new_tokens=max_new_tokens,
            preset_id=cast(str | None, preset),
        )

    def _require_backend(self) -> WorkerBackend:
        if self._backend is None:
            raise RuntimeError("Granite worker is not loaded")
        return self._backend


def inspect_granite_runtime(device: str, precision: str) -> RuntimeInspection:
    """Inspect installed dependencies and CUDA without loading model weights."""

    checks: list[PreflightCheck] = []
    environment = _base_environment()
    supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(
        PreflightCheck(
            "python",
            "passed" if supported else "failed",
            "error",
            (
                f"Python {platform.python_version()} is supported"
                if supported
                else "Granite worker requires Python >=3.11,<3.13"
            ),
            details={"supported": ">=3.11,<3.13"},
        )
    )
    modules: dict[str, object | None] = {}
    for module_name, distribution in (
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("soundfile", "soundfile"),
    ):
        modules[module_name] = _import_runtime_module(
            module_name, distribution, checks, environment
        )
    transformers = modules["transformers"]
    api_available = transformers is not None and all(
        getattr(transformers, name, None) is not None
        for name in ("AutoProcessor", "AutoModelForSpeechSeq2Seq")
    )
    checks.append(
        PreflightCheck(
            "transformers_api",
            "passed" if api_available else "failed",
            "error",
            (
                "Granite Transformers APIs are available"
                if api_available
                else "AutoProcessor or AutoModelForSpeechSeq2Seq is unavailable"
            ),
        )
    )
    checks.extend(_cuda_checks(modules["torch"], device, precision, environment))
    return RuntimeInspection(tuple(checks), environment)


def _import_runtime_module(
    module_name: str,
    distribution: str,
    checks: list[PreflightCheck],
    environment: JSONObject,
) -> object | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        checks.append(
            PreflightCheck(
                f"module:{module_name}",
                "failed",
                "error",
                f"Unable to import {module_name}: {type(error).__name__}: {error}",
            )
        )
        return None
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    environment[f"{distribution}_version"] = version
    checks.append(
        PreflightCheck(
            f"module:{module_name}",
            "passed",
            "error",
            f"Imported {module_name}",
            details={"version": version},
        )
    )
    return module


def _cuda_checks(
    torch: object | None, device: str, precision: str, environment: JSONObject
) -> list[PreflightCheck]:
    if torch is None:
        return [
            PreflightCheck(
                "cuda",
                "failed" if device == "cuda" or precision == "fp16" else "skipped",
                "error" if device == "cuda" or precision == "fp16" else "info",
                "CUDA check unavailable because Torch import failed",
            )
        ]
    cuda = cast(CudaInspector | None, getattr(torch, "cuda", None))
    available = bool(cuda is not None and cuda.is_available())
    count = int(cuda.device_count()) if available and cuda is not None else 0
    environment.update({"cuda_available": available, "cuda_device_count": count})
    requires_cuda = device == "cuda" or precision == "fp16"
    if requires_cuda and not available:
        return [
            PreflightCheck(
                "cuda",
                "failed",
                "error",
                "Granite FP16/CUDA configuration requires an available CUDA device",
            )
        ]
    if available:
        return [
            PreflightCheck(
                "cuda",
                "passed",
                "error",
                "Torch reports CUDA available",
                details={"device_count": count, "fp16_policy": True},
            )
        ]
    return [
        PreflightCheck(
            "cuda",
            "passed",
            "info",
            "CUDA is unavailable; automatic selection will use CPU FP32",
        )
    ]


def serve(
    service: WorkerService,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    error_stream: TextIO,
) -> int:
    return serve_worker(
        WorkerDispatcher(service),
        input_stream,
        output_stream,
        error_stream,
        error_mapper=_map_error,
        logger=LOGGER,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dominican Eaters Granite JSONL worker")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    raise SystemExit(serve(WorkerService(), sys.stdin.buffer, sys.stdout.buffer, sys.stderr))


def _base_environment() -> JSONObject:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _descriptor_payload(descriptor: BackendDescriptor) -> JSONObject:
    return cast(
        JSONObject,
        {
            "backend_id": descriptor.backend_id,
            "model": descriptor.model,
            "model_revision": descriptor.model_revision,
            "language": descriptor.language,
            "requested_device": descriptor.requested_device,
            "requested_precision": descriptor.requested_precision,
            "effective_device": descriptor.effective_device,
            "effective_precision": descriptor.effective_precision,
            "runtime_versions": descriptor.runtime_versions,
            "options": descriptor.options,
        },
    )


def _map_error(error: Exception) -> WorkerError:
    if isinstance(error, GraniteDependencyError | ImportError):
        return WorkerError("dependency_missing", str(error))
    if isinstance(error, (ValueError, TypeError, FileNotFoundError, KeyError)):
        return WorkerError("invalid_argument", str(error))
    message = str(error)
    if isinstance(error, RuntimeError) and "out of memory" in message.lower():
        return WorkerError("cuda_oom", message)
    if isinstance(error, RuntimeError):
        return WorkerError("invalid_state", message)
    return WorkerError("backend_error", message)
