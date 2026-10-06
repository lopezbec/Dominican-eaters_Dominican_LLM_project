"""Strict JSONL protocol for isolated ASR worker processes."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias, cast
from uuid import uuid4

PROTOCOL_VERSION = 2
SUPPORTED_PROTOCOL_VERSIONS = frozenset({1, PROTOCOL_VERSION})
MAX_LINE_BYTES = 1024 * 1024
MAX_REQUEST_ID_LENGTH = 128

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
Method: TypeAlias = Literal["describe", "preflight", "load", "warmup", "transcribe", "close"]
PreflightStatus: TypeAlias = Literal["passed", "failed", "skipped"]
PreflightSeverity: TypeAlias = Literal["info", "warning", "error"]


class WorkerProtocolError(ValueError):
    """Base error for malformed or unsupported protocol messages."""


class LineTooLargeError(WorkerProtocolError):
    """A JSONL frame exceeded :data:`MAX_LINE_BYTES`."""


class MessageValidationError(WorkerProtocolError):
    """A JSON value does not match the protocol schema."""


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One machine-readable runtime compatibility check."""

    name: str
    status: PreflightStatus
    severity: PreflightSeverity
    message: str
    details: JSONObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    protocol_version: int
    request_id: str
    method: Method
    params: JSONObject


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class SuccessResponse:
    protocol_version: int
    request_id: str
    ok: Literal[True]
    result: JSONObject


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    protocol_version: int
    request_id: str
    ok: Literal[False]
    error: ErrorDetail


WorkerResponse: TypeAlias = SuccessResponse | ErrorResponse
Message: TypeAlias = WorkerRequest | WorkerResponse

_REQUEST_FIELDS = frozenset({"protocol_version", "request_id", "method", "params"})
_SUCCESS_FIELDS = frozenset({"protocol_version", "request_id", "ok", "result"})
_ERROR_FIELDS = frozenset({"protocol_version", "request_id", "ok", "error"})
_ERROR_DETAIL_FIELDS = frozenset({"code", "message", "retryable"})
_PREFLIGHT_RESULT_FIELDS = frozenset({"ready", "checks", "environment"})
_PREFLIGHT_CHECK_FIELDS = frozenset({"name", "status", "severity", "message", "details"})
_METHODS_BY_VERSION = {
    1: frozenset({"describe", "load", "warmup", "transcribe", "close"}),
    2: frozenset({"describe", "preflight", "load", "warmup", "transcribe", "close"}),
}
_V1_PARAM_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "describe": (frozenset(), frozenset()),
    "load": (
        frozenset({"backend", "model", "language", "device", "precision", "options"}),
        frozenset({"backend", "model", "language", "device", "precision", "options"}),
    ),
    "warmup": (frozenset(), frozenset()),
    "transcribe": (
        frozenset({"audio_path"}),
        frozenset({"audio_path", "duration_seconds"}),
    ),
    "close": (frozenset(), frozenset()),
}
_RUNTIME_FIELDS = frozenset(
    {
        "backend",
        "preset",
        "model",
        "model_revision",
        "language",
        "device",
        "precision",
        "quantization",
        "prompt_template_id",
        "options",
    }
)
_V2_PARAM_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    **_V1_PARAM_FIELDS,
    "preflight": (_RUNTIME_FIELDS, _RUNTIME_FIELDS),
    # Extra provenance is optional on load while existing clients migrate from v1.
    "load": (_V1_PARAM_FIELDS["load"][0], _RUNTIME_FIELDS),
}


def make_request(
    method: Method,
    params: JSONObject | None = None,
    *,
    request_id: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
) -> WorkerRequest:
    """Construct a request, generating a correlation ID when omitted."""
    request = WorkerRequest(
        protocol_version,
        request_id or uuid4().hex,
        method,
        {} if params is None else params,
    )
    _validate_request(request)
    return request


def success_response(request: WorkerRequest, result: JSONObject | None = None) -> SuccessResponse:
    """Construct a successful response correlated with ``request``."""
    payload = {} if result is None else result
    if request.method == "preflight":
        _validate_preflight_result(payload)
    response = SuccessResponse(request.protocol_version, request.request_id, True, payload)
    _validate_response(response)
    return response


def error_response(
    request: WorkerRequest,
    *,
    code: str,
    message: str,
    retryable: bool = False,
) -> ErrorResponse:
    """Construct a structured error correlated with ``request``."""
    response = ErrorResponse(
        request.protocol_version,
        request.request_id,
        False,
        ErrorDetail(code=code, message=message, retryable=retryable),
    )
    _validate_response(response)
    return response


def encode_message(message: Message) -> bytes:
    """Encode one message as a newline-terminated UTF-8 JSONL frame."""
    if isinstance(message, WorkerRequest):
        _validate_request(message)
    elif isinstance(message, SuccessResponse | ErrorResponse):
        _validate_response(message)
    else:
        raise TypeError("message must be a worker protocol dataclass")
    raw = cast(JSONObject, asdict(message))
    _validate_json_value(raw, path="message")
    encoded = (
        json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_LINE_BYTES:
        raise LineTooLargeError(
            f"encoded message is {len(encoded)} bytes; maximum is {MAX_LINE_BYTES}"
        )
    return encoded


def decode_request(line: bytes | str) -> WorkerRequest:
    """Decode one request and reject unknown envelope or parameter fields."""
    raw = _decode_object(line)
    _require_exact_fields(raw, _REQUEST_FIELDS, path="request")
    version, request_id = _common_fields(raw)
    method_value = _required_string(raw, "method")
    if method_value not in _METHODS_BY_VERSION[version]:
        raise MessageValidationError(f"unsupported method: {method_value!r}")
    params = _required_object(raw, "params")
    _validate_method_params(version, method_value, params)
    return WorkerRequest(version, request_id, cast(Method, method_value), params)


def decode_response(line: bytes | str) -> WorkerResponse:
    """Decode one success or error response with exact field validation."""
    raw = _decode_object(line)
    ok = raw.get("ok")
    if not isinstance(ok, bool):
        raise MessageValidationError("ok must be a boolean")
    _require_exact_fields(raw, _SUCCESS_FIELDS if ok else _ERROR_FIELDS, path="response")
    version, request_id = _common_fields(raw)
    if ok:
        return SuccessResponse(version, request_id, True, _required_object(raw, "result"))
    error = _required_object(raw, "error")
    _require_exact_fields(error, _ERROR_DETAIL_FIELDS, path="error")
    retryable = error["retryable"]
    if not isinstance(retryable, bool):
        raise MessageValidationError("error.retryable must be a boolean")
    return ErrorResponse(
        version,
        request_id,
        False,
        ErrorDetail(
            _required_string(error, "code", prefix="error."),
            _required_string(error, "message", allow_empty=True, prefix="error."),
            retryable,
        ),
    )


def _validate_request(request: WorkerRequest) -> None:
    _validate_protocol_version(request.protocol_version)
    _validate_request_id(request.request_id)
    if request.method not in _METHODS_BY_VERSION[request.protocol_version]:
        raise MessageValidationError(f"unsupported method: {request.method!r}")
    _validate_json_value(request.params, path="params")
    _validate_method_params(request.protocol_version, request.method, request.params)


def _validate_response(response: WorkerResponse) -> None:
    _validate_protocol_version(response.protocol_version)
    _validate_request_id(response.request_id)
    if isinstance(response, SuccessResponse):
        if response.ok is not True:
            raise MessageValidationError("success response ok must be true")
        _validate_json_value(response.result, path="result")
        return
    if response.ok is not False:
        raise MessageValidationError("error response ok must be false")
    if not response.error.code:
        raise MessageValidationError("error.code must be a non-empty string")
    if not isinstance(response.error.message, str):
        raise MessageValidationError("error.message must be a string")
    if not isinstance(response.error.retryable, bool):
        raise MessageValidationError("error.retryable must be a boolean")


def _validate_method_params(version: int, method: str, params: JSONObject) -> None:
    schemas = _V1_PARAM_FIELDS if version == 1 else _V2_PARAM_FIELDS
    required, allowed = schemas[method]
    _require_fields(params, required, allowed, path=f"{method} params")
    if method in ("load", "preflight"):
        for field in ("backend", "model", "language", "device", "precision"):
            _required_string(params, field)
        _required_object(params, "options")
        for field in ("preset", "model_revision", "quantization", "prompt_template_id"):
            if field in params:
                _optional_string(params, field)
    elif method == "transcribe":
        _required_string(params, "audio_path")
        if "duration_seconds" in params:
            duration = params["duration_seconds"]
            if (
                isinstance(duration, bool)
                or not isinstance(duration, int | float)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise MessageValidationError("duration_seconds must be a finite positive number")


def make_preflight_result(
    checks: Sequence[PreflightCheck],
    *,
    environment: JSONObject | None = None,
) -> JSONObject:
    """Build the canonical v2 preflight result and derive its readiness."""

    ready = not any(check.status == "failed" and check.severity == "error" for check in checks)
    result: JSONObject = {
        "ready": ready,
        "checks": cast(list[JSONValue], [asdict(check) for check in checks]),
        "environment": {} if environment is None else environment,
    }
    _validate_preflight_result(result)
    return result


def _validate_preflight_result(result: JSONObject) -> None:
    _require_exact_fields(result, _PREFLIGHT_RESULT_FIELDS, path="preflight result")
    ready = result["ready"]
    if not isinstance(ready, bool):
        raise MessageValidationError("preflight result ready must be a boolean")
    raw_checks = result["checks"]
    if not isinstance(raw_checks, list):
        raise MessageValidationError("preflight result checks must be an array")
    has_blocking_failure = False
    for index, raw_check in enumerate(raw_checks):
        path = f"preflight result checks[{index}]"
        if not isinstance(raw_check, dict):
            raise MessageValidationError(f"{path} must be a JSON object")
        check = raw_check
        _require_exact_fields(check, _PREFLIGHT_CHECK_FIELDS, path=path)
        _required_string(check, "name", prefix=f"{path}.")
        status = _required_string(check, "status", prefix=f"{path}.")
        if status not in ("passed", "failed", "skipped"):
            raise MessageValidationError(f"{path}.status is unsupported: {status!r}")
        severity = _required_string(check, "severity", prefix=f"{path}.")
        if severity not in ("info", "warning", "error"):
            raise MessageValidationError(f"{path}.severity is unsupported: {severity!r}")
        _required_string(check, "message", allow_empty=True, prefix=f"{path}.")
        _required_object(check, "details")
        has_blocking_failure |= status == "failed" and severity == "error"
    _required_object(result, "environment")
    if ready and has_blocking_failure:
        raise MessageValidationError(
            "preflight result cannot be ready with a failed error-severity check"
        )


def _decode_object(line: bytes | str) -> JSONObject:
    if isinstance(line, str):
        encoded = line.encode("utf-8")
    elif isinstance(line, bytes):
        encoded = line
    else:
        raise TypeError("JSONL frame must be bytes or str")
    if len(encoded) > MAX_LINE_BYTES:
        raise LineTooLargeError(f"frame is {len(encoded)} bytes; maximum is {MAX_LINE_BYTES}")
    if b"\n" in encoded.rstrip(b"\r\n"):
        raise MessageValidationError("expected exactly one JSONL frame")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MessageValidationError("frame is not valid UTF-8") from exc
    if not text.strip():
        raise MessageValidationError("frame is empty")
    try:
        value = json.loads(text, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise MessageValidationError(f"invalid JSON: {exc}") from exc
    _validate_json_value(value, path="message")
    if not isinstance(value, dict):
        raise MessageValidationError("message must be a JSON object")
    return cast(JSONObject, value)


def _object_without_duplicates(pairs: list[tuple[str, JSONValue]]) -> JSONObject:
    result: JSONObject = {}
    for key, value in pairs:
        if key in result:
            raise MessageValidationError(f"duplicate field: {key}")
        result[key] = value
    return result


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MessageValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MessageValidationError(f"{path} contains a non-string object key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise MessageValidationError(f"{path} contains unsupported value {type(value).__name__}")


def _require_fields(
    raw: JSONObject,
    required: frozenset[str],
    allowed: frozenset[str],
    *,
    path: str,
) -> None:
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - allowed)
    if missing:
        raise MessageValidationError(f"{path} missing fields: {', '.join(missing)}")
    if unknown:
        raise MessageValidationError(f"{path} unknown fields: {', '.join(unknown)}")


def _require_exact_fields(raw: JSONObject, expected: frozenset[str], *, path: str) -> None:
    _require_fields(raw, expected, expected, path=path)


def _common_fields(raw: JSONObject) -> tuple[int, str]:
    version = raw["protocol_version"]
    _validate_protocol_version(version)
    request_id = raw["request_id"]
    _validate_request_id(request_id)
    return cast(int, version), cast(str, request_id)


def _validate_protocol_version(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MessageValidationError("protocol_version must be an integer")
    if value not in SUPPORTED_PROTOCOL_VERSIONS:
        raise MessageValidationError(
            f"unsupported protocol_version {value}; expected one of "
            f"{', '.join(str(version) for version in sorted(SUPPORTED_PROTOCOL_VERSIONS))}"
        )


def _validate_request_id(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise MessageValidationError("request_id must be a non-empty string")
    if len(value) > MAX_REQUEST_ID_LENGTH:
        raise MessageValidationError(f"request_id exceeds maximum length {MAX_REQUEST_ID_LENGTH}")


def _required_string(
    raw: JSONObject,
    field: str,
    *,
    allow_empty: bool = False,
    prefix: str = "",
) -> str:
    value = raw[field]
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise MessageValidationError(f"{prefix}{field} must be {qualifier}")
    return value


def _required_object(raw: JSONObject, field: str) -> JSONObject:
    value = raw[field]
    if not isinstance(value, dict):
        raise MessageValidationError(f"{field} must be a JSON object")
    return value


def _optional_string(raw: JSONObject, field: str) -> str | None:
    value = raw[field]
    if value is not None and (not isinstance(value, str) or not value):
        raise MessageValidationError(f"{field} must be a non-empty string or null")
    return value
