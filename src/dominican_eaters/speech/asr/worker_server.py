"""Reusable dispatcher and JSONL server for isolated ASR workers."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import BinaryIO, Protocol, TextIO

from .worker_protocol import (
    PROTOCOL_VERSION,
    JSONObject,
    WorkerProtocolError,
    WorkerRequest,
    WorkerResponse,
    decode_request,
    encode_message,
    error_response,
    success_response,
)

LOGGER = logging.getLogger(__name__)


class WorkerApplication(Protocol):
    """Model-family hooks hosted by :class:`WorkerDispatcher`."""

    def describe(self) -> JSONObject: ...

    def preflight(self, params: JSONObject) -> JSONObject: ...

    def load(self, params: JSONObject) -> JSONObject: ...

    def warmup(self) -> JSONObject: ...

    def transcribe(self, params: JSONObject) -> JSONObject: ...

    def close(self) -> None: ...


class WorkerService(Protocol):
    """Minimal service boundary consumed by the JSONL loop."""

    def dispatch(self, request: WorkerRequest) -> JSONObject: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerError:
    """Wire-safe classification for an application exception."""

    code: str
    message: str
    retryable: bool = False


ErrorMapper = Callable[[Exception], WorkerError]


class WorkerDispatcher:
    """Dispatch protocol methods to model-family hooks without runtime imports."""

    def __init__(self, application: WorkerApplication) -> None:
        self._application = application

    def dispatch(self, request: WorkerRequest) -> JSONObject:
        if request.method == "describe":
            return self._application.describe()
        if request.method == "preflight":
            return self._application.preflight(request.params)
        if request.method == "load":
            return self._application.load(request.params)
        if request.method == "warmup":
            return self._application.warmup()
        if request.method == "transcribe":
            return self._application.transcribe(request.params)
        if request.method == "close":
            self.close()
            return {}
        raise AssertionError(f"Decoder admitted unsupported method: {request.method}")

    def close(self) -> None:
        self._application.close()


def serve(
    service: WorkerService,
    input_stream: BinaryIO,
    output_stream: BinaryIO,
    error_stream: TextIO,
    *,
    error_mapper: ErrorMapper | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """Serve strict JSONL requests until EOF or a successful ``close``."""

    map_error = error_mapper or default_error_mapper
    log = logger or LOGGER
    for line in input_stream:
        request: WorkerRequest | None = None
        response: WorkerResponse
        try:
            request = decode_request(line)
            with model_stdout_to_stderr(error_stream):
                result = service.dispatch(request)
            response = success_response(request, result)
        except WorkerProtocolError as error:
            log.warning("Rejected worker request: %s", error)
            response = error_response(
                _response_request(request),
                code="invalid_request",
                message=str(error),
                retryable=False,
            )
        except Exception as error:
            log.exception("Worker request failed")
            mapped = map_error(error)
            response = error_response(
                _response_request(request),
                code=mapped.code,
                message=mapped.message,
                retryable=mapped.retryable,
            )
        output_stream.write(encode_message(response))
        output_stream.flush()
        if request is not None and request.method == "close" and response.ok:
            return 0

    try:
        with model_stdout_to_stderr(error_stream):
            service.close()
    except Exception:
        log.exception("Worker cleanup failed after input closed")
        return 1
    return 0


def default_error_mapper(error: Exception) -> WorkerError:
    """Map common failures while allowing each worker to supply richer codes."""

    if isinstance(error, ImportError):
        return WorkerError("dependency_missing", str(error))
    if isinstance(error, (ValueError, TypeError, FileNotFoundError)):
        return WorkerError("invalid_argument", str(error))
    if isinstance(error, RuntimeError):
        return WorkerError("invalid_state", str(error))
    return WorkerError("backend_error", str(error))


def _response_request(request: WorkerRequest | None) -> WorkerRequest:
    return request or WorkerRequest(PROTOCOL_VERSION, "invalid-request", "describe", {})


@contextlib.contextmanager
def model_stdout_to_stderr(error_stream: TextIO) -> Iterator[None]:
    """Keep Python and native-library diagnostics away from protocol stdout."""

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
