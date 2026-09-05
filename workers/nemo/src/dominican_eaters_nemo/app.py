"""JSONL process entry point for the isolated NeMo worker."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO, cast

from dominican_eaters.speech.asr import BackendDescriptor, Transcript
from dominican_eaters.speech.asr.worker_protocol import (
    PROTOCOL_VERSION,
    JSONObject,
    JSONValue,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResponse,
    decode_request,
    encode_message,
    error_response,
    success_response,
)

from . import __version__
from .adapters import (
    CanaryBackend,
    NeMoSettings,
    ParakeetBackend,
    ShortAudioPolicy,
)

LOGGER = logging.getLogger("dominican_eaters_nemo")


class WorkerBackend(Protocol):
    @property
    def descriptor(self) -> BackendDescriptor: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def transcribe(
        self, audio_path: Path, *, duration_seconds: float | None = None
    ) -> Transcript: ...

    def close(self) -> None: ...


BackendFactory = Callable[[NeMoSettings], WorkerBackend]


class WorkerService:
    """Stateful dispatcher for one model lifecycle."""

    def __init__(self, factories: Mapping[str, BackendFactory] | None = None) -> None:
        self._factories = dict(
            factories
            or {
                "parakeet": ParakeetBackend,
                "canary": CanaryBackend,
            }
        )
        self._backend: WorkerBackend | None = None

    def dispatch(self, request: WorkerRequest) -> JSONObject:
        if request.method == "describe":
            result: JSONObject = {
                "worker": "dominican-eaters-nemo",
                "worker_version": __version__,
                "protocol_version": PROTOCOL_VERSION,
                "backends": cast(list[JSONValue], sorted(self._factories)),
            }
            if self._backend is not None:
                result["descriptor"] = _descriptor_payload(self._backend.descriptor)
            return result
        if request.method == "load":
            return self._load(request.params)
        if request.method == "warmup":
            backend = self._require_backend()
            backend.warmup()
            return {"descriptor": _descriptor_payload(backend.descriptor)}
        if request.method == "transcribe":
            backend = self._require_backend()
            raw_duration = request.params.get("duration_seconds")
            duration = None if raw_duration is None else float(cast(int | float, raw_duration))
            transcript = backend.transcribe(
                Path(cast(str, request.params["audio_path"])), duration_seconds=duration
            )
            return {
                "text": transcript.text,
                "language": transcript.language,
                "audio_duration_seconds": transcript.audio_duration_seconds,
                "gpu_peak_allocated_bytes": transcript.gpu_peak_allocated_bytes,
                "gpu_peak_reserved_bytes": transcript.gpu_peak_reserved_bytes,
                "metadata": cast(dict[str, JSONValue], transcript.metadata),
            }
        if request.method == "close":
            self.close()
            return {}
        raise AssertionError(f"Decoder admitted unsupported method: {request.method}")

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()

    def _load(self, params: JSONObject) -> JSONObject:
        if self._backend is not None:
            raise RuntimeError("NeMo worker is already loaded")
        backend_name = cast(str, params["backend"])
        try:
            factory = self._factories[backend_name]
        except KeyError as error:
            supported = ", ".join(sorted(self._factories))
            raise ValueError(
                f"Unsupported NeMo backend {backend_name!r}; expected one of: {supported}"
            ) from error
        options = cast(dict[str, JSONValue], params["options"])
        allowed_options = {"short_audio_policy", "minimum_audio_seconds", "timestamps"}
        unknown_options = sorted(set(options) - allowed_options)
        if unknown_options:
            raise ValueError(f"Unknown NeMo options: {', '.join(unknown_options)}")
        policy_value = options.get("short_audio_policy", ShortAudioPolicy.REJECT.value)
        if not isinstance(policy_value, str):
            raise TypeError("short_audio_policy must be a string")
        try:
            policy = ShortAudioPolicy(policy_value)
        except ValueError as error:
            raise ValueError("short_audio_policy must be 'reject' or 'allow'") from error
        minimum = options.get("minimum_audio_seconds", 0.1)
        if isinstance(minimum, bool) or not isinstance(minimum, int | float):
            raise TypeError("minimum_audio_seconds must be a number")
        timestamps = options.get("timestamps", False)
        if not isinstance(timestamps, bool):
            raise TypeError("timestamps must be a boolean")
        settings = NeMoSettings(
            model=cast(str, params["model"]),
            language=cast(str, params["language"]),
            device=cast(str, params["device"]),  # type: ignore[arg-type]
            precision=cast(str, params["precision"]),  # type: ignore[arg-type]
            short_audio_policy=policy,
            minimum_audio_seconds=float(minimum),
            timestamps=timestamps,
        )
        backend = factory(settings)
        self._backend = backend
        try:
            backend.load()
        except Exception:
            self._backend = None
            try:
                backend.close()
            except Exception:
                LOGGER.exception("NeMo backend cleanup failed after load error")
            raise
        return {"descriptor": _descriptor_payload(backend.descriptor)}

    def _require_backend(self) -> WorkerBackend:
        if self._backend is None:
            raise RuntimeError("NeMo worker is not loaded")
        return self._backend


def serve(
    service: WorkerService,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    error_stream: TextIO,
) -> int:
    """Serve requests until EOF or a successful close request."""

    exit_code = 0
    for line in input_stream:
        request: WorkerRequest | None = None
        response: WorkerResponse
        try:
            request = decode_request(line)
            with _model_stdout_to_stderr(error_stream):
                result = service.dispatch(request)
            response = success_response(request, result)
        except WorkerProtocolError as error:
            LOGGER.warning("Rejected worker request: %s", error)
            synthetic = request or WorkerRequest(
                PROTOCOL_VERSION, "invalid-request", "describe", {}
            )
            response = error_response(
                synthetic,
                code="invalid_request",
                message=str(error),
                retryable=False,
            )
        except Exception as error:
            LOGGER.exception("NeMo worker request failed")
            synthetic = request or WorkerRequest(
                PROTOCOL_VERSION, "invalid-request", "describe", {}
            )
            response = error_response(
                synthetic,
                code=_error_code(error),
                message=str(error),
                retryable=False,
            )
        output_stream.write(encode_message(response))
        output_stream.flush()
        if request is not None and request.method == "close" and response.ok:
            return exit_code
    try:
        with _model_stdout_to_stderr(error_stream):
            service.close()
    except Exception:
        LOGGER.exception("NeMo worker cleanup failed after input closed")
        exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dominican Eaters isolated NeMo JSONL worker")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    raise SystemExit(serve(WorkerService(), sys.stdin.buffer, sys.stdout.buffer, sys.stderr))


def _descriptor_payload(descriptor: BackendDescriptor) -> JSONObject:
    return cast(JSONObject, asdict(descriptor))


def _error_code(error: Exception) -> str:
    if isinstance(error, (ValueError, TypeError, FileNotFoundError)):
        return "invalid_argument"
    if isinstance(error, RuntimeError):
        return "invalid_state"
    return "backend_error"


@contextlib.contextmanager
def _model_stdout_to_stderr(error_stream: TextIO) -> Iterator[None]:
    """Protect stdout from Python and native-library diagnostic writes."""

    with contextlib.redirect_stdout(error_stream):
        saved_stdout: int | None = None
        try:
            if error_stream is sys.stderr:
                sys.stderr.flush()
                sys.stdout.flush()
                saved_stdout = os.dup(1)
                os.dup2(2, 1)
            yield
        finally:
            if saved_stdout is not None:
                os.dup2(saved_stdout, 1)
                os.close(saved_stdout)
