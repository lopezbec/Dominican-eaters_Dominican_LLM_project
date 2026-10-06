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
from typing import BinaryIO, Literal, Protocol, cast

from .contracts import BackendDescriptor, Transcript
from .worker_protocol import (
    ErrorResponse,
    JSONObject,
    JSONValue,
    PreflightCheck,
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
        self.error = response.error
        self.code = response.error.code
        self.error_type = self.code
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
WorkerStderrSink = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class WorkerPreflightReport:
    """Normalized result of a no-weight worker runtime inspection."""

    ready: bool
    checks: tuple[PreflightCheck, ...]
    environment: JSONObject


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
    startup_timeout_seconds: float | None = None
    preflight_timeout_seconds: float | None = None
    load_timeout_seconds: float | None = None
    inference_timeout_seconds: float | None = None
    shutdown_timeout_seconds: float = 10.0
    preset: str | None = None
    model_revision: str | None = None
    quantization: str | None = None
    prompt_template_id: str | None = None

    def __post_init__(self) -> None:
        if not self.interpreter.is_absolute():
            raise ValueError("worker interpreter must be an absolute path")
        for name in ("worker_module", "backend", "model", "language", "device", "precision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("preset", "model_revision", "quantization", "prompt_template_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        timeout_values = {
            "request_timeout_seconds": self.request_timeout_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "preflight_timeout_seconds": self.preflight_timeout_seconds,
            "load_timeout_seconds": self.load_timeout_seconds,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
        }
        invalid = [
            name
            for name, value in timeout_values.items()
            if value is not None and (not math.isfinite(value) or value <= 0)
        ]
        if invalid:
            raise ValueError(f"worker timeouts must be positive: {', '.join(invalid)}")

    @property
    def argv(self) -> tuple[str, ...]:
        return (str(self.interpreter), "-m", self.worker_module, *self.worker_args)

    def timeout_for(self, phase: Literal["startup", "preflight", "load", "inference"]) -> float:
        """Resolve a phase deadline, retaining the legacy request timeout as fallback."""

        configured = getattr(self, f"{phase}_timeout_seconds")
        return self.request_timeout_seconds if configured is None else cast(float, configured)


class JsonlSubprocessBackend:
    """Implement the ASR contract through one strict request/response stream."""

    def __init__(
        self,
        settings: SubprocessBackendSettings,
        *,
        transport_factory: TransportFactory | None = None,
        request_id_factory: Callable[[], str] | None = None,
        stderr_sink: WorkerStderrSink | None = None,
    ) -> None:
        self._settings = settings
        self._transport_factory = transport_factory
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._stderr_sink = stderr_sink
        self._transport: WorkerTransport | None = None
        self._loaded = False
        self._preflight_ready: bool | None = None
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

    def preflight(self, *, close_after: bool = False) -> WorkerPreflightReport:
        """Inspect the selected worker runtime without downloading or loading model weights.

        The worker stays alive by default so a successful report can be followed by ``load()``
        in the same process. ``close_after=True`` is the explicit preflight-only lifecycle.
        """

        if self._loaded:
            raise RuntimeError("worker backend is already loaded")
        try:
            self._ensure_started()
            payload = self._exchange(
                make_request(
                    "preflight",
                    self._runtime_params(),
                    request_id=self._new_request_id(),
                ),
                timeout_seconds=self._settings.timeout_for("preflight"),
            )
            report = _preflight_report_from_payload(payload)
            self._preflight_ready = report.ready
        except Exception:
            self._abort()
            raise
        if close_after:
            self.close()
        return report

    def load(self) -> None:
        if self._loaded:
            return
        if self._preflight_ready is False:
            raise RuntimeError("worker preflight did not pass")
        try:
            self._ensure_started()
            self._exchange(
                make_request(
                    "load",
                    self._runtime_params(),
                    request_id=self._new_request_id(),
                ),
                timeout_seconds=self._settings.timeout_for("load"),
            )
            payload = self._exchange(
                make_request("describe", request_id=self._new_request_id()),
                timeout_seconds=self._settings.timeout_for("load"),
            )
            self._descriptor = _descriptor_from_payload(payload)
            self._loaded = True
        except Exception:
            self._abort()
            raise

    def warmup(self) -> None:
        self._require_loaded()
        self._exchange(
            make_request("warmup", request_id=self._new_request_id()),
            timeout_seconds=self._settings.timeout_for("load"),
        )

    def transcribe(self, audio_path: Path) -> Transcript:
        self._require_loaded()
        resolved = audio_path.resolve()
        payload = self._exchange(
            make_request(
                "transcribe",
                {"audio_path": str(resolved)},
                request_id=self._new_request_id(),
            ),
            timeout_seconds=self._settings.timeout_for("inference"),
        )
        expected_fields = {
            "text",
            "language",
            "audio_duration_seconds",
            "gpu_peak_allocated_bytes",
            "gpu_peak_reserved_bytes",
            "metadata",
        }
        _require_exact_fields(payload, expected_fields, "worker transcript payload")
        text = payload["text"]
        if not isinstance(text, str):
            raise WorkerProcessError("worker transcript payload requires a string text field")
        language = payload["language"]
        if language is not None and not isinstance(language, str):
            raise WorkerProcessError("worker transcript language must be a string or null")
        metadata = payload["metadata"]
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
            self._preflight_ready = None
            return
        error: Exception | None = None
        try:
            self._exchange(
                make_request("close", request_id=self._new_request_id()),
                timeout_seconds=self._settings.shutdown_timeout_seconds,
            )
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
            self._preflight_ready = None
        if error is not None:
            raise error

    def _exchange(
        self,
        request: WorkerRequest,
        *,
        timeout_seconds: float,
        operation: str | None = None,
    ) -> dict[str, JSONValue]:
        transport = self._transport
        if transport is None:
            raise RuntimeError("worker process is not started")
        try:
            transport.write(encode_message(request))
            response = decode_response(transport.read(timeout_seconds))
        except TimeoutError as exc:
            self._abort()
            operation_name = operation or request.method
            raise WorkerTimeoutError(
                f"worker {operation_name} timed out after {timeout_seconds:g} seconds"
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
        if response.protocol_version != request.protocol_version:
            self._abort()
            raise WorkerProcessError(
                f"worker response protocol_version {response.protocol_version} does not match "
                f"request version {request.protocol_version}"
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

    def _ensure_started(self) -> None:
        if self._transport is not None:
            return
        transport = (
            self._transport_factory()
            if self._transport_factory is not None
            else _SubprocessLineTransport(stderr_sink=self._stderr_sink)
        )
        self._transport = transport
        try:
            transport.start(self._settings.argv)
            self._exchange(
                make_request("describe", request_id=self._new_request_id()),
                timeout_seconds=self._settings.timeout_for("startup"),
                operation="startup",
            )
        except Exception:
            self._abort()
            raise

    def _runtime_params(self) -> JSONObject:
        return {
            "backend": self._settings.backend,
            "preset": self._settings.preset,
            "model": self._settings.model,
            "model_revision": self._settings.model_revision,
            "language": self._settings.language,
            "device": self._settings.device,
            "precision": self._settings.precision,
            "quantization": self._settings.quantization,
            "prompt_template_id": self._settings.prompt_template_id,
            "options": dict(self._settings.options),
        }

    def _abort(self) -> None:
        transport = self._transport
        self._transport = None
        self._loaded = False
        self._preflight_ready = None
        if transport is not None:
            try:
                transport.close(self._settings.shutdown_timeout_seconds)
            except Exception:
                pass


def _preflight_report_from_payload(payload: Mapping[str, JSONValue]) -> WorkerPreflightReport:
    _require_exact_fields(payload, {"ready", "checks", "environment"}, "worker preflight payload")
    ready = payload["ready"]
    if not isinstance(ready, bool):
        raise WorkerProcessError("worker preflight ready must be a boolean")
    raw_checks = payload["checks"]
    if not isinstance(raw_checks, list):
        raise WorkerProcessError("worker preflight checks must be an array")
    checks: list[PreflightCheck] = []
    has_blocking_failure = False
    for index, raw_check in enumerate(raw_checks):
        context = f"worker preflight check {index}"
        if not isinstance(raw_check, dict):
            raise WorkerProcessError(f"{context} must be an object")
        _require_exact_fields(
            raw_check,
            {"name", "status", "severity", "message", "details"},
            context,
        )
        name = _required_string(raw_check, "name", context)
        status = _required_choice(
            raw_check,
            "status",
            ("passed", "failed", "skipped"),
            context,
        )
        severity = _required_choice(
            raw_check,
            "severity",
            ("info", "warning", "error"),
            context,
        )
        message = raw_check["message"]
        if not isinstance(message, str):
            raise WorkerProcessError(f"{context} message must be a string")
        details = raw_check["details"]
        if not isinstance(details, dict):
            raise WorkerProcessError(f"{context} details must be an object")
        has_blocking_failure |= status == "failed" and severity == "error"
        checks.append(
            PreflightCheck(
                name=name,
                status=cast(Literal["passed", "failed", "skipped"], status),
                severity=cast(Literal["info", "warning", "error"], severity),
                message=message,
                details=cast(JSONObject, dict(details)),
            )
        )
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise WorkerProcessError("worker preflight environment must be an object")
    derived_ready = not has_blocking_failure
    if ready != derived_ready:
        raise WorkerProcessError("worker preflight ready does not match its error-severity checks")
    return WorkerPreflightReport(ready, tuple(checks), cast(JSONObject, dict(environment)))


def _required_string(payload: Mapping[str, JSONValue], field: str, context: str) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value:
        raise WorkerProcessError(f"{context} {field} must be a non-empty string")
    return value


def _required_choice(
    payload: Mapping[str, JSONValue],
    field: str,
    choices: tuple[str, ...],
    context: str,
) -> str:
    value = _required_string(payload, field, context)
    if value not in choices:
        raise WorkerProcessError(f"{context} {field} is unsupported: {value!r}")
    return value


def _descriptor_from_payload(payload: Mapping[str, JSONValue]) -> BackendDescriptor:
    _require_exact_fields(payload, {"descriptor"}, "worker describe payload")
    raw = payload["descriptor"]
    if not isinstance(raw, dict):
        raise WorkerProcessError("worker descriptor must be an object")
    expected_fields = {
        "backend_id",
        "model",
        "model_revision",
        "language",
        "requested_device",
        "requested_precision",
        "effective_device",
        "effective_precision",
        "runtime_versions",
        "options",
    }
    _require_exact_fields(raw, expected_fields, "worker descriptor")

    def required_string(name: str) -> str:
        value = raw[name]
        if not isinstance(value, str) or not value:
            raise WorkerProcessError(f"worker descriptor {name} must be a non-empty string")
        return value

    def optional_string(name: str) -> str | None:
        value = raw[name]
        if value is not None and not isinstance(value, str):
            raise WorkerProcessError(f"worker descriptor {name} must be a string or null")
        return value

    versions = raw["runtime_versions"]
    options = raw["options"]
    if not isinstance(versions, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in versions.items()
    ):
        raise WorkerProcessError("worker descriptor runtime_versions must contain strings")
    if not isinstance(options, dict):
        raise WorkerProcessError("worker descriptor options must be an object")
    return BackendDescriptor(
        backend_id=required_string("backend_id"),
        model=required_string("model"),
        model_revision=optional_string("model_revision"),
        language=required_string("language"),
        requested_device=required_string("requested_device"),
        requested_precision=required_string("requested_precision"),
        effective_device=optional_string("effective_device"),
        effective_precision=optional_string("effective_precision"),
        runtime_versions=cast(dict[str, str], versions),
        options=cast(dict[str, object], options),
    )


def _require_exact_fields(
    payload: Mapping[str, JSONValue], expected: set[str], context: str
) -> None:
    missing = sorted(expected - payload.keys())
    unknown = sorted(payload.keys() - expected)
    if missing:
        raise WorkerProcessError(f"{context} missing fields: {', '.join(missing)}")
    if unknown:
        raise WorkerProcessError(f"{context} unknown fields: {', '.join(unknown)}")


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

    def __init__(self, *, stderr_sink: WorkerStderrSink | None = None) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout: queue.Queue[bytes | BaseException | None] = queue.Queue()
        self._stderr: deque[bytes] = deque(maxlen=200)
        self._stderr_sink = stderr_sink
        self._stderr_lock = threading.Lock()

    @property
    def stderr_tail(self) -> str:
        with self._stderr_lock:
            tail = b"".join(self._stderr)
        return tail.decode("utf-8", errors="replace").strip()

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
                with self._stderr_lock:
                    self._stderr.append(chunk)
                self._emit_stderr(chunk.decode("utf-8", errors="replace"))
        except OSError:
            return

    def _emit_stderr(self, message: str) -> None:
        if self._stderr_sink is None:
            return
        try:
            self._stderr_sink(message)
        except Exception:
            # Diagnostics must never interrupt the protocol transport.
            return

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("worker transport is not started")
        return self._process

    def _exited_error(self, return_code: int | None) -> WorkerProcessError:
        detail = f"; stderr: {self.stderr_tail}" if self.stderr_tail else ""
        return WorkerProcessError(f"worker exited with code {return_code}{detail}")
