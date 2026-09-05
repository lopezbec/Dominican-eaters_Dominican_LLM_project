from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from dominican_eaters.speech.asr.worker_protocol import (
    MAX_LINE_BYTES,
    PROTOCOL_VERSION,
    ErrorResponse,
    JSONValue,
    LineTooLargeError,
    MessageValidationError,
    Method,
    SuccessResponse,
    WorkerRequest,
    decode_request,
    decode_response,
    encode_message,
    error_response,
    make_request,
    success_response,
)


def load_params() -> dict[str, JSONValue]:
    return {
        "backend": "nemo",
        "model": "nvidia/parakeet-tdt-0.6b-v2",
        "language": "es",
        "device": "cuda",
        "precision": "bf16",
        "options": {"batch_size": 4, "timestamps": False, "tags": ["eval", None]},
    }


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("describe", {}),
        ("load", load_params()),
        ("warmup", {}),
        ("transcribe", {"audio_path": "/data/a.wav", "duration_seconds": 2.5}),
        ("close", {}),
    ],
)
def test_all_requests_roundtrip_as_one_jsonl_frame(
    method: Method, params: dict[str, JSONValue]
) -> None:
    message = make_request(method, params, request_id="request-001")
    encoded = encode_message(message)

    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert decode_request(encoded) == message


def test_make_request_generates_distinct_nonempty_correlation_ids() -> None:
    first = make_request("describe")
    second = make_request("describe")

    assert first.request_id
    assert first.request_id != second.request_id


def test_success_and_error_responses_roundtrip_and_retain_request_id() -> None:
    request = make_request("transcribe", {"audio_path": "/data/a.wav"}, request_id="abc")
    success = success_response(request, {"text": "Qué lo qué", "language": "es", "metadata": {}})
    failure = error_response(
        request, code="cuda_oom", message="GPU memory exhausted", retryable=True
    )

    assert isinstance(decode_response(encode_message(success)), SuccessResponse)
    decoded_failure = decode_response(encode_message(failure))
    assert isinstance(decoded_failure, ErrorResponse)
    assert decoded_failure.request_id == "abc"
    assert decoded_failure.error.code == "cuda_oom"
    assert decoded_failure.error.retryable is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.update(extra="unexpected"), "unknown fields"),
        (lambda raw: raw.pop("params"), "missing fields"),
        (lambda raw: raw.update(protocol_version=2), "protocol_version"),
        (lambda raw: raw.update(protocol_version=True), "must be an integer"),
        (lambda raw: raw.update(request_id=""), "request_id"),
        (lambda raw: raw.update(params=[]), "params must be a JSON object"),
    ],
)
def test_request_rejects_envelope_schema_drift(
    mutation: Callable[[dict[str, object]], object], match: str
) -> None:
    raw: dict[str, object] = json.loads(encode_message(make_request("load", load_params())))
    mutation(raw)

    with pytest.raises(MessageValidationError, match=match):
        decode_request(json.dumps(raw))


def test_method_params_have_exact_schemas() -> None:
    with pytest.raises(MessageValidationError, match="warmup params unknown fields"):
        make_request("warmup", {"runs": 2})
    with pytest.raises(MessageValidationError, match="transcribe params missing fields"):
        make_request("transcribe")
    with pytest.raises(MessageValidationError, match="load params missing fields"):
        make_request("load", {"model": "x"})
    with pytest.raises(MessageValidationError, match="duration_seconds"):
        make_request("transcribe", {"audio_path": "a.wav", "duration_seconds": 0})


def test_response_requires_exact_shape_for_ok_discriminator() -> None:
    success = {
        "protocol_version": 1,
        "request_id": "id",
        "ok": True,
        "result": {},
        "error": {"code": "x", "message": "x", "retryable": False},
    }
    with pytest.raises(MessageValidationError, match="unknown fields: error"):
        decode_response(json.dumps(success))

    error = {
        "protocol_version": 1,
        "request_id": "id",
        "ok": False,
        "error": {"code": "x", "message": "x", "retryable": False, "traceback": "secret"},
    }
    with pytest.raises(MessageValidationError, match="error unknown fields"):
        decode_response(json.dumps(error))


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"not-json\n",
        b"[]\n",
        b'{"protocol_version":1,"request_id":"x","method":"warmup","method":"close","params":{}}\n',
        b'{"protocol_version":1,"request_id":"x","method":"warmup","params":{}}\n{}\n',
        b"\xff\n",
    ],
)
def test_invalid_jsonl_frames_are_rejected(frame: bytes) -> None:
    with pytest.raises(MessageValidationError):
        decode_request(frame)


def test_line_size_limit_applies_before_json_parsing() -> None:
    with pytest.raises(LineTooLargeError):
        decode_request(b"x" * (MAX_LINE_BYTES + 1))


def test_json_payload_rejects_runtime_objects_and_nonfinite_numbers() -> None:
    request = make_request("describe")
    invalid: dict[str, JSONValue] = {"path": Path("audio.wav")}  # type: ignore[dict-item]
    with pytest.raises(MessageValidationError, match="Path"):
        success_response(request, invalid)
    with pytest.raises(MessageValidationError, match="non-finite"):
        success_response(request, {"value": float("nan")})


def test_error_requires_boolean_retryability() -> None:
    raw = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "id",
        "ok": False,
        "error": {"code": "runtime", "message": "failed", "retryable": "yes"},
    }
    with pytest.raises(MessageValidationError, match="retryable"):
        decode_response(json.dumps(raw))


def test_wrong_python_message_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        encode_message(object())  # type: ignore[arg-type]


def test_direct_dataclass_construction_is_validated_at_encoding() -> None:
    request = WorkerRequest(PROTOCOL_VERSION, "id", "warmup", {"unexpected": True})
    with pytest.raises(MessageValidationError, match="unknown fields"):
        encode_message(request)
