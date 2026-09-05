"""Lifecycle-safe client for ASR workers running in isolated Python environments."""

from __future__ import annotations

import math
import queue
import subprocess
import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from .contracts import BackendDescriptor, Transcript
from .worker_protocol import (
    ErrorResponse,
    JSONValue,
    WorkerRequest,
    decode_response,
    encode_message,
    make_request,
)


class WorkerProcessError(RuntimeError):
    """The worker could not complete a protocol operation."""


class WorkerTimeoutError(WorkerProcessError):
    """The worker did not respond before its configured deadline."""


class WorkerRemoteError(WorkerProcessError):
    """The worker returned a structured error response."""

    def __init__(self, response: ErrorResponse) -> None:
        super().__init__(f"worker failed ({response.error.code}): {response.error.message}")
        self.error_type = response.error.code
        self.retryable = response.error.retryable


class WorkerTransport(Protocol):
    """Byte-oriented process transport; useful as an offline test seam."""

    def start(self, argv: Sequence[str]) -> None: ...

    def write(self, frame: bytes) -> None: ...

    def read(self, timeout_seconds: float) -> bytes: ...

    def close(self, timeout_seconds: float) -> None: ...

    @property
    def stderr_tail(self) -> str: ...


TransportFactory = Callable[[], WorkerTransport]


@dataclass(frozen=True, slots=True)
class SubprocessBackendSettings:
    """Configuration passed explicitly across the worker boundary."""

    interpreter: Path
    worker_module: str
    backend: str
    model: str
    language: str = "es"
    device: str = "auto"
    precision: str = "auto"
    options: Mapping[str, JSONValue] = field(default_factory=dict)
    worker_args: tuple[str, ...] = ()
    request_timeout_seconds: float = 300.0
    shutdown_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.interpreter.is_absolute():
            raise ValueError("worker interpreter must be an absolute path")
        for name in ("worker_module", "backend", "model", "language", "device", "precision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or not math.isfinite(self.shutdown_timeout_seconds)
            or self.request_timeout_seconds <= 0
            or self.shutdown_timeout_seconds <= 0
        ):
            raise ValueError("worker timeouts must be positive")

    @property
    def argv(self) -> tuple[str, ...]:
        return (str(self.interpreter), "-m", self.worker_module, *self.worker_args)


class JsonlSubprocessBackend:
    """Implement the ASR contract through one strict request/response stream."""

    def __init__(
        self,
        settings: SubprocessBackendSettings,
        *,
        transport_factory: TransportFactory | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._settings = settings
        self._transport_factory = transport_factory or _SubprocessLineTransport
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._transport: WorkerTransport | None = None
        self._loaded = False
        self._descriptor = BackendDescriptor(
            backend_id=f"{settings.backend}/{settings.model}",
            model=settings.model,
            model_revision=None,
            language=settings.language,
            requested_device=settings.device,
            requested_precision=settings.precision,
        )

    @property
    def backend_id(self) -> str:
        return self._descriptor.backend_id

    @property
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def load(self) -> None:
        if self._loaded:
            return
        if self._transport is not None:
            raise RuntimeError("worker process is already started but not loaded")
        transport = self._transport_factory()
        self._transport = transport
        try:
            transport.start(self._settings.argv)
            self._exchange(
                make_request(
                    "load",
                    {
                        "backend": self._settings.backend,
                        "model": self._settings.model,
                        "language": self._settings.language,
                        "device": self._settings.device,
                        "precision": self._settings.precision,
                        "options": dict(self._settings.options),
                    },
                    request_id=self._new_request_id(),
                )
            )
            payload = self._exchange(make_request("describe", request_id=self._new_request_id()))
            self._descriptor = _descriptor_from_payload(payload, fallback=self._descriptor)
            self._loaded = True
        except Exception:
            self._abort()
            raise

    def warmup(self) -> None:
        self._require_loaded()
        self._exchange(make_request("warmup", request_id=self._new_request_id()))

    def transcribe(self, audio_path: Path) -> Transcript:
        self._require_loaded()
        resolved = audio_path.resolve()
        payload = self._exchange(
            make_request(
                "transcribe",
                {"audio_path": str(resolved)},
                request_id=self._new_request_id(),
            )
        )
        text = payload.get("text")
        if not isinstance(text, str):
            raise WorkerProcessError("worker transcript payload requires a string text field")
        language = payload.get("language")
        if language is not None and not isinstance(language, str):
            raise WorkerProcessError("worker transcript language must be a string or null")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise WorkerProcessError("worker transcript metadata must be an object")
        duration = _optional_positive_number(payload, "audio_duration_seconds")
        allocated = _optional_nonnegative_integer(payload, "gpu_peak_allocated_bytes")
        reserved = _optional_nonnegative_integer(payload, "gpu_peak_reserved_bytes")
        return Transcript(
            text=text,
            language=language,
            audio_duration_seconds=duration,
            gpu_peak_allocated_bytes=allocated,
            gpu_peak_reserved_bytes=reserved,
            metadata=cast(dict[str, object], metadata),
        )

    def close(self) -> None:
        transport = self._transport
        if transport is None:
            self._loaded = False
            return
        error: Exception | None = None
        if self._loaded:
            try:
                self._exchange(make_request("close", request_id=self._new_request_id()))
            except Exception as exc:
                error = exc
        try:
            if self._transport is transport:
                try:
                    transport.close(self._settings.shutdown_timeout_seconds)
                except Exception as exc:
                    if error is None:
                        error = exc
        finally:
            self._transport = None
            self._loaded = False
        if error is not None:
            raise error

    def _exchange(self, request: WorkerRequest) -> dict[str, JSONValue]:
        transport = self._transport
        if transport is None:
            raise RuntimeError("worker process is not started")
        try:
            transport.write(encode_message(request))
            response = decode_response(transport.read(self._settings.request_timeout_seconds))
        except TimeoutError as exc:
            self._abort()
            raise WorkerTimeoutError(
                f"worker {request.method} timed out after "
                f"{self._settings.request_timeout_seconds:g} seconds"
            ) from exc
        except Exception:
            self._abort()
            raise
        if response.request_id != request.request_id:
            self._abort()
            raise WorkerProcessError(
                f"worker response request_id {response.request_id!r} does not match "
                f"{request.request_id!r}"
            )
        if isinstance(response, ErrorResponse):
            raise WorkerRemoteError(response)
        return response.result

    def _new_request_id(self) -> str:
        request_id = self._request_id_factory()
        if not request_id:
            raise ValueError("request ID factory returned an empty value")
        return request_id

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("worker backend is not loaded")

    def _abort(self) -> None:
        transport = self._transport
        self._transport = None
        self._loaded = False
        if transport is not None:
            try:
                transport.close(self._settings.shutdown_timeout_seconds)
            except Exception:
                pass


def _descriptor_from_payload(
    payload: Mapping[str, JSONValue], *, fallback: BackendDescriptor
) -> BackendDescriptor:
    raw = payload.get("descriptor", payload)
    if not isinstance(raw, dict) or not raw:
        return fallback

    def required_string(name: str, default: str) -> str:
        value = raw.get(name, default)
        if not isinstance(value, str) or not value:
            raise WorkerProcessError(f"worker descriptor {name} must be a non-empty string")
        return value

    def optional_string(name: str, default: str | None) -> str | None:
        value = raw.get(name, default)
        if value is not None and not isinstance(value, str):
            raise WorkerProcessError(f"worker descriptor {name} must be a string or null")
        return value

    versions = raw.get("runtime_versions", fallback.runtime_versions)
    options = raw.get("options", fallback.options)
    if not isinstance(versions, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in versions.items()
    ):
        raise WorkerProcessError("worker descriptor runtime_versions must contain strings")
    if not isinstance(options, dict):
        raise WorkerProcessError("worker descriptor options must be an object")
    return BackendDescriptor(
        backend_id=required_string("backend_id", fallback.backend_id),
        model=required_string("model", fallback.model),
        model_revision=optional_string("model_revision", fallback.model_revision),
        language=required_string("language", fallback.language),
        requested_device=required_string("requested_device", fallback.requested_device),
        requested_precision=required_string("requested_precision", fallback.requested_precision),
        effective_device=optional_string("effective_device", fallback.effective_device),
        effective_precision=optional_string("effective_precision", fallback.effective_precision),
        runtime_versions=cast(dict[str, str], versions),
        options=cast(dict[str, object], options),
    )


def _optional_positive_number(payload: Mapping[str, JSONValue], field: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkerProcessError(f"worker transcript {field} must be a number or null")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise WorkerProcessError(f"worker transcript {field} must be positive and finite")
    return result


def _optional_nonnegative_integer(payload: Mapping[str, JSONValue], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise WorkerProcessError(f"worker transcript {field} must be a nonnegative integer")
    return value


class _SubprocessLineTransport:
    """Drain both child streams continuously so model logging cannot deadlock it."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: queue.Queue[bytes | BaseException | None] = queue.Queue()
        self._stderr: deque[bytes] = deque(maxlen=200)

    @property
    def stderr_tail(self) -> str:
        return b"".join(self._stderr).decode("utf-8", errors="replace").strip()

    def start(self, argv: Sequence[str]) -> None:
        if self._process is not None:
            raise RuntimeError("worker transport has already started")
        if not argv or not Path(argv[0]).is_absolute():
            raise ValueError("worker argv must start with an absolute interpreter path")
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise WorkerProcessError("worker process pipes were not created")
        self._process = process
        threading.Thread(target=self._drain_stdout, args=(process.stdout,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(process.stderr,), daemon=True).start()

    def write(self, frame: bytes) -> None:
        process = self._require_process()
        if process.poll() is not None:
            raise self._exited_error(process.returncode)
        stdin = process.stdin
        if stdin is None:
            raise WorkerProcessError("worker stdin is unavailable")
        try:
            stdin.write(frame)
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._exited_error(process.poll()) from exc

    def read(self, timeout_seconds: float) -> bytes:
        process = self._require_process()
        try:
            item = self._stdout.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError from exc
        if isinstance(item, BaseException):
            raise WorkerProcessError(f"worker stdout reader failed: {item}") from item
        if item is None:
            raise self._exited_error(process.poll())
        return item

    def close(self, timeout_seconds: float) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout_seconds)

    def _drain_stdout(self, stream: BinaryIO) -> None:
        try:
            while line := stream.readline():
                self._stdout.put(line)
        except BaseException as exc:
            self._stdout.put(exc)
        finally:
            self._stdout.put(None)

    def _drain_stderr(self, stream: BinaryIO) -> None:
        try:
            while chunk := stream.read(4096):
                self._stderr.append(chunk)
        except OSError:
            return

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("worker transport is not started")
        return self._process

    def _exited_error(self, return_code: int | None) -> WorkerProcessError:
        detail = f"; stderr: {self.stderr_tail}" if self.stderr_tail else ""
        return WorkerProcessError(f"worker exited with code {return_code}{detail}")
