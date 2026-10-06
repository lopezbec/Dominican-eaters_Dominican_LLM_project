"""JSONL process entry point for the isolated Voxtral worker."""

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
from .adapters import (
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MAX_NEW_TOKENS,
    AudioPolicyError,
    VoxtralBackend,
    VoxtralDependencyError,
    VoxtralSettings,
)

LOGGER = logging.getLogger("dominican_eaters_voxtral")


class WorkerBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def transcribe(
        self, audio_path: Path, *, duration_seconds: float | None = None
    ) -> Transcript: ...

    def close(self) -> None: ...


BackendFactory = Callable[[VoxtralSettings], WorkerBackend]


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    checks: tuple[PreflightCheck, ...]
    environment: JSONObject


RuntimeInspector = Callable[[str, str], RuntimeInspection]


class CudaInspector(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def get_device_capability(self, device: int = 0) -> tuple[int, int]: ...


class WorkerService:
    def __init__(
        self,
        factories: Mapping[str, BackendFactory] | None = None,
        *,
        runtime_inspector: RuntimeInspector | None = None,
    ) -> None:
        self._factories = dict(factories or {"voxtral": VoxtralBackend})
        self._runtime_inspector = runtime_inspector or inspect_voxtral_runtime
        self._backend: WorkerBackend | None = None

    def describe(self) -> JSONObject:
        result: JSONObject = {
            "worker": "dominican-eaters-voxtral",
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
            f"{backend_name} configuration is valid",
            details={
                "backend": backend_name,
                "model": settings.model,
                "language": settings.language,
                "device": "cuda",
                "precision": "fp16",
                "batch_size": 1,
                "max_audio_seconds": settings.max_audio_seconds,
                "max_new_tokens": settings.max_new_tokens,
                "timestamps": False,
            },
        )
        environment = dict(inspection.environment)
        environment.update(
            {
                "backend": backend_name,
                "model": settings.model,
                "weights_loaded": False,
            }
        )
        return make_preflight_result((configuration, *inspection.checks), environment=environment)

    def load(self, params: JSONObject) -> JSONObject:
        if self._backend is not None:
            raise RuntimeError("Voxtral worker is already loaded")
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
                LOGGER.exception("Voxtral cleanup failed after load error")
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

    def _settings_from_params(self, params: JSONObject) -> tuple[str, VoxtralSettings]:
        backend_name = cast(str, params["backend"])
        if backend_name not in self._factories:
            raise ValueError("Unsupported Voxtral backend; expected 'voxtral'")
        options = cast(dict[str, JSONValue], params["options"])
        allowed = {"max_audio_seconds", "max_new_tokens", "timestamps"}
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise ValueError(f"Unknown Voxtral options: {', '.join(unknown)}")
        max_audio = options.get("max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        max_tokens = options.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        timestamps = options.get("timestamps", False)
        if isinstance(max_audio, bool) or not isinstance(max_audio, int | float):
            raise TypeError("max_audio_seconds must be a number")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if not isinstance(timestamps, bool):
            raise TypeError("timestamps must be a boolean")
        return backend_name, VoxtralSettings(
            model=cast(str, params["model"]),
            model_revision=cast(str | None, params.get("model_revision")),
            language=cast(str, params["language"]),
            device=cast(str, params["device"]),  # type: ignore[arg-type]
            precision=cast(str, params["precision"]),  # type: ignore[arg-type]
            max_audio_seconds=float(max_audio),
            max_new_tokens=max_tokens,
            timestamps=timestamps,
        )

    def _require_backend(self) -> WorkerBackend:
        if self._backend is None:
            raise RuntimeError("Voxtral worker is not loaded")
        return self._backend


def inspect_voxtral_runtime(device: str, precision: str) -> RuntimeInspection:
    checks: list[PreflightCheck] = []
    environment = _base_environment()
    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(
        PreflightCheck(
            "python",
            "passed" if supported_python else "failed",
            "error",
            (
                f"Python {platform.python_version()} is supported"
                if supported_python
                else "Voxtral worker requires Python >=3.11,<3.13"
            ),
            details={"supported": ">=3.11,<3.13"},
        )
    )
    transformers = _import_runtime_module("transformers", checks, environment)
    torch = _import_runtime_module("torch", checks, environment)
    _import_runtime_module("soundfile", checks, environment)
    if transformers is None:
        checks.append(
            PreflightCheck(
                "voxtral_api",
                "skipped",
                "info",
                "Voxtral API check skipped because Transformers import failed",
            )
        )
    else:
        available = all(
            getattr(transformers, name, None) is not None
            for name in ("AutoProcessor", "VoxtralForConditionalGeneration")
        )
        checks.append(
            PreflightCheck(
                "voxtral_api",
                "passed" if available else "failed",
                "error",
                (
                    "Transformers Voxtral APIs are available"
                    if available
                    else "Transformers lacks the Voxtral APIs; version >=4.54 is required"
                ),
            )
        )
    checks.extend(_cuda_checks(torch, environment))
    return RuntimeInspection(tuple(checks), environment)


def _import_runtime_module(
    name: str, checks: list[PreflightCheck], environment: JSONObject
) -> object | None:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        checks.append(
            PreflightCheck(
                f"module:{name}",
                "failed",
                "error",
                f"Unable to import {name}: {type(error).__name__}: {error}",
            )
        )
        return None
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    environment[f"{name}_version"] = version
    checks.append(
        PreflightCheck(
            f"module:{name}", "passed", "error", f"Imported {name}", {"version": version}
        )
    )
    return module


def _cuda_checks(torch: object | None, environment: JSONObject) -> list[PreflightCheck]:
    if torch is None:
        return [
            PreflightCheck(
                "cuda", "failed", "error", "CUDA check unavailable because Torch import failed"
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
                "error",
                f"CUDA inspection failed: {type(error).__name__}: {error}",
            )
        ]
    environment["cuda_available"] = available
    environment["cuda_device_count"] = count
    if not available or cuda is None:
        return [PreflightCheck("cuda", "failed", "error", "Voxtral FP16 requires CUDA")]
    capability = cuda.get_device_capability(0)
    environment["cuda_compute_capability"] = f"{capability[0]}.{capability[1]}"
    return [
        PreflightCheck(
            "cuda",
            "passed",
            "error",
            "Torch reports CUDA available for the FP16 profile",
            {"device_count": count, "compute_capability": list(capability)},
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
    parser = argparse.ArgumentParser(description="Dominican Eaters isolated Voxtral JSONL worker")
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
    if isinstance(error, VoxtralDependencyError):
        return WorkerError("dependency_missing", str(error))
    if isinstance(error, (AudioPolicyError, ValueError, TypeError, FileNotFoundError)):
        return WorkerError("invalid_audio", str(error))
    if isinstance(error, RuntimeError):
        message = str(error)
        code = "cuda_oom" if "out of memory" in message.lower() else "invalid_state"
        return WorkerError(code, message)
    return WorkerError("backend_error", str(error))
