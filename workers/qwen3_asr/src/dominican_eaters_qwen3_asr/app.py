"""JSONL process entry point for the isolated Qwen3-ASR worker."""

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
from .adapters import Qwen3ASRBackend, Qwen3ASRDependencyError, Qwen3Settings

LOGGER = logging.getLogger("dominican_eaters_qwen3_asr")


class WorkerBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def transcribe(
        self, audio_path: Path, *, duration_seconds: float | None = None
    ) -> Transcript: ...

    def close(self) -> None: ...


BackendFactory = Callable[[Qwen3Settings], WorkerBackend]


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    checks: tuple[PreflightCheck, ...]
    environment: JSONObject


RuntimeInspector = Callable[[str, str], RuntimeInspection]


class CudaInspector(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...


class WorkerService:
    """Qwen3-ASR hooks hosted by the shared worker dispatcher."""

    def __init__(
        self,
        factories: Mapping[str, BackendFactory] | None = None,
        *,
        runtime_inspector: RuntimeInspector | None = None,
    ) -> None:
        self._factories = dict(factories or {"qwen3_asr": Qwen3ASRBackend})
        self._runtime_inspector = runtime_inspector or inspect_qwen3_runtime
        self._backend: WorkerBackend | None = None

    def describe(self) -> JSONObject:
        result: JSONObject = {
            "worker": "dominican-eaters-qwen3-asr",
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
        except (TypeError, ValueError) as error:
            return make_preflight_result(
                [PreflightCheck("configuration", "failed", "error", str(error))],
                environment=_base_environment(),
            )
        inspection = self._runtime_inspector(settings.device, settings.precision)
        configuration = PreflightCheck(
            "configuration",
            "passed",
            "error",
            f"{backend_name} offline configuration is valid",
            details={
                "backend": backend_name,
                "model": settings.model,
                "language": settings.language,
                "device": settings.device,
                "precision": settings.precision,
                "batch_size": 1,
                "attention_implementation": "sdpa",
            },
        )
        environment = dict(inspection.environment)
        environment.update(
            {"backend": backend_name, "model": settings.model, "weights_loaded": False}
        )
        return make_preflight_result((configuration, *inspection.checks), environment=environment)

    def load(self, params: JSONObject) -> JSONObject:
        if self._backend is not None:
            raise RuntimeError("Qwen3-ASR worker is already loaded")
        backend_name, settings = self._settings_from_params(params)
        backend = self._factories[backend_name](settings)
        self._backend = backend
        try:
            backend.load()
        except Exception:
            self._backend = None
            try:
                backend.close()
            except Exception:
                LOGGER.exception("Qwen3-ASR cleanup failed after load error")
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

    def _settings_from_params(self, params: JSONObject) -> tuple[str, Qwen3Settings]:
        backend_name = cast(str, params["backend"])
        if backend_name not in self._factories:
            supported = ", ".join(sorted(self._factories))
            raise ValueError(
                f"Unsupported Qwen3-ASR backend {backend_name!r}; expected: {supported}"
            )
        options = cast(dict[str, JSONValue], params["options"])
        allowed_options = {"max_new_tokens", "timestamps"}
        unknown_options = sorted(set(options) - allowed_options)
        if unknown_options:
            raise ValueError(f"Unknown Qwen3-ASR options: {', '.join(unknown_options)}")
        timestamps = options.get("timestamps", False)
        if timestamps is not False:
            raise ValueError("Qwen3-ASR timestamps require a separate forced-aligner preset")
        max_new_tokens = options.get("max_new_tokens", 256)
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        quantization = params.get("quantization")
        if quantization is not None:
            raise ValueError("The initial Qwen3-ASR worker does not support quantization")
        return backend_name, Qwen3Settings(
            model=cast(str, params["model"]),
            model_revision=cast(str | None, params.get("model_revision")),
            language=cast(str, params["language"]),  # type: ignore[arg-type]
            device=cast(str, params["device"]),  # type: ignore[arg-type]
            precision=cast(str, params["precision"]),  # type: ignore[arg-type]
            max_new_tokens=max_new_tokens,
        )

    def _require_backend(self) -> WorkerBackend:
        if self._backend is None:
            raise RuntimeError("Qwen3-ASR worker is not loaded")
        return self._backend


def inspect_qwen3_runtime(device: str, precision: str) -> RuntimeInspection:
    """Inspect native Transformers support without loading model weights."""

    checks: list[PreflightCheck] = []
    environment = _base_environment()
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 14)
    checks.append(
        PreflightCheck(
            "python",
            "passed" if supported_python else "failed",
            "error",
            (
                f"Python {platform.python_version()} is supported"
                if supported_python
                else "Qwen3-ASR worker requires Python >=3.11,<3.14"
            ),
            details={"supported": ">=3.11,<3.14"},
        )
    )
    transformers = _import_runtime_module("transformers", checks, environment)
    torch = _import_runtime_module("torch", checks, environment)
    if transformers is None:
        checks.append(
            PreflightCheck(
                "qwen3_asr_api",
                "skipped",
                "info",
                "Qwen3-ASR API check skipped because Transformers import failed",
            )
        )
    else:
        processor = getattr(transformers, "AutoProcessor", None)
        model = getattr(transformers, "AutoModelForMultimodalLM", None)
        available = processor is not None and model is not None
        checks.append(
            PreflightCheck(
                "qwen3_asr_api",
                "passed" if available else "failed",
                "error",
                (
                    "Native Qwen3-ASR Transformers APIs are available"
                    if available
                    else "AutoProcessor or AutoModelForMultimodalLM is unavailable"
                ),
            )
        )
    checks.extend(_cuda_checks(torch, device=device, precision=precision, environment=environment))
    return RuntimeInspection(tuple(checks), environment)


def _import_runtime_module(
    module_name: str, checks: list[PreflightCheck], environment: JSONObject
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
        version = importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    environment[f"{module_name}_version"] = version
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
    torch: object | None,
    *,
    device: str,
    precision: str,
    environment: JSONObject,
) -> list[PreflightCheck]:
    if torch is None:
        required = device == "cuda" or precision == "fp16"
        return [
            PreflightCheck(
                "cuda",
                "failed" if required else "skipped",
                "error" if required else "info",
                "CUDA check unavailable because Torch import failed",
            )
        ]
    cuda = cast(CudaInspector | None, getattr(torch, "cuda", None))
    try:
        available = bool(cuda is not None and cuda.is_available())
        count = int(cuda.device_count()) if available and cuda is not None else 0
    except Exception as error:
        return [
            PreflightCheck(
                "cuda",
                "failed",
                "error" if device == "cuda" else "warning",
                f"CUDA inspection failed: {type(error).__name__}: {error}",
            )
        ]
    environment["cuda_available"] = available
    environment["cuda_device_count"] = count
    required = device == "cuda" or precision == "fp16"
    if not available:
        return [
            PreflightCheck(
                "cuda",
                "failed" if required else "passed",
                "error" if required else "info",
                (
                    "CUDA/FP16 was requested but Torch reports CUDA unavailable"
                    if required
                    else "CUDA is unavailable; automatic selection will use CPU FP32"
                ),
                details={"available": False},
            )
        ]
    return [
        PreflightCheck(
            "cuda",
            "passed",
            "error",
            "Torch reports CUDA available for the Qwen3-ASR FP16 profile",
            details={"available": True, "device_count": count},
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
    parser = argparse.ArgumentParser(description="Dominican Eaters Qwen3-ASR JSONL worker")
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
    if isinstance(error, Qwen3ASRDependencyError | ImportError):
        return WorkerError("dependency_missing", str(error))
    if isinstance(error, (ValueError, TypeError, FileNotFoundError)):
        return WorkerError("invalid_argument", str(error))
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return WorkerError("cuda_oom", str(error), retryable=True)
    if isinstance(error, RuntimeError):
        return WorkerError("invalid_state", str(error))
    return WorkerError("backend_error", str(error))
