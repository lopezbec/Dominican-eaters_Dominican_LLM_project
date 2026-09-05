from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

from dominican_eaters.speech.asr import BackendDescriptor, Transcript
from dominican_eaters.speech.asr.worker_protocol import (
    ErrorResponse,
    SuccessResponse,
    decode_response,
    encode_message,
    make_request,
)

from dominican_eaters_nemo.adapters import NeMoSettings
from dominican_eaters_nemo.app import WorkerService, serve


@dataclass
class FakeBackend:
    settings: NeMoSettings
    fail_load: bool = False

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=f"fake/{self.settings.model}",
            model=self.settings.model,
            model_revision="offline",
            language=self.settings.language,
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device="cpu" if "load" in self.calls else None,
            effective_precision="fp32" if "load" in self.calls else None,
            runtime_versions={"fake": "1"},
            options={"short_audio_policy": self.settings.short_audio_policy.value},
        )

    def load(self) -> None:
        self.calls.append("load")
        print("model diagnostic must go to stderr")
        if self.fail_load:
            raise RuntimeError("load failed")

    def warmup(self) -> None:
        self.calls.append("warmup")

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        self.calls.append(f"transcribe:{audio_path}:{duration_seconds}")
        return Transcript(
            text="Hola",
            language="es",
            audio_duration_seconds=duration_seconds,
            metadata={"segments": [{"start": 0.0, "end": 0.2}]},
        )

    def close(self) -> None:
        self.calls.append("close")


def request_lines(*requests: object) -> io.BytesIO:
    return io.BytesIO(b"".join(encode_message(request) for request in requests))  # type: ignore[arg-type]


def decoded_lines(output: io.BytesIO):
    return [decode_response(line) for line in output.getvalue().splitlines()]


def load_request(*, backend: str = "parakeet", options=None):
    return make_request(
        "load",
        {
            "backend": backend,
            "model": "fixture-model",
            "language": "es",
            "device": "cpu",
            "precision": "fp32",
            "options": options or {},
        },
        request_id="load-1",
    )


def test_complete_jsonl_lifecycle_has_protocol_only_stdout(tmp_path: Path) -> None:
    created: list[FakeBackend] = []

    def factory(settings: NeMoSettings) -> FakeBackend:
        backend = FakeBackend(settings)
        created.append(backend)
        return backend

    requests = request_lines(
        make_request("describe", request_id="describe-1"),
        load_request(options={"minimum_audio_seconds": 0.25, "timestamps": True}),
        make_request("describe", request_id="describe-2"),
        make_request("warmup", request_id="warmup-1"),
        make_request(
            "transcribe",
            {"audio_path": str(tmp_path / "audio.wav"), "duration_seconds": 0.5},
            request_id="transcribe-1",
        ),
        make_request("close", request_id="close-1"),
    )
    output = io.BytesIO()
    errors = io.StringIO()

    exit_code = serve(WorkerService({"parakeet": factory}), requests, output, errors)
    responses = decoded_lines(output)

    assert exit_code == 0
    assert len(responses) == 6
    assert all(isinstance(response, SuccessResponse) for response in responses)
    assert responses[0].result["backends"] == ["parakeet"]  # type: ignore[union-attr]
    descriptor = responses[1].result["descriptor"]  # type: ignore[union-attr]
    assert isinstance(descriptor, dict)
    assert descriptor["backend_id"] == "fake/fixture-model"
    described = responses[2].result["descriptor"]  # type: ignore[union-attr]
    assert isinstance(described, dict)
    assert described["effective_device"] == "cpu"
    assert responses[4].result["text"] == "Hola"  # type: ignore[union-attr]
    assert created[0].settings.minimum_audio_seconds == 0.25
    assert created[0].settings.timestamps is True
    assert created[0].calls[-1] == "close"
    assert b"model diagnostic" not in output.getvalue()
    assert "model diagnostic must go to stderr" in errors.getvalue()


def test_invalid_request_and_backend_errors_are_structured() -> None:
    raw_invalid = (
        json.dumps(
            {
                "protocol_version": 1,
                "request_id": "bad",
                "method": "warmup",
                "params": {"unknown": True},
            }
        ).encode()
        + b"\n"
    )
    requests = io.BytesIO(
        raw_invalid
        + encode_message(make_request("warmup", request_id="not-loaded"))
        + encode_message(make_request("close", request_id="close"))
    )
    output = io.BytesIO()

    serve(WorkerService({}), requests, output, io.StringIO())
    responses = decoded_lines(output)

    assert len(responses) == 3
    assert isinstance(responses[0], ErrorResponse)
    assert responses[0].request_id == "invalid-request"
    assert responses[0].error.code == "invalid_request"
    assert isinstance(responses[1], ErrorResponse)
    assert responses[1].request_id == "not-loaded"
    assert responses[1].error.code == "invalid_state"
    assert isinstance(responses[2], SuccessResponse)


def test_load_rejects_unknown_backend_and_options() -> None:
    output = io.BytesIO()
    serve(
        WorkerService({"parakeet": FakeBackend}),
        request_lines(
            load_request(backend="unknown"),
            make_request("close", request_id="close"),
        ),
        output,
        io.StringIO(),
    )
    response = decoded_lines(output)[0]
    assert isinstance(response, ErrorResponse)
    assert response.error.code == "invalid_argument"
    assert "Unsupported NeMo backend" in response.error.message

    output = io.BytesIO()
    serve(
        WorkerService({"parakeet": FakeBackend}),
        request_lines(
            load_request(options={"mystery": True}),
            make_request("close", request_id="close"),
        ),
        output,
        io.StringIO(),
    )
    response = decoded_lines(output)[0]
    assert isinstance(response, ErrorResponse)
    assert "Unknown NeMo options" in response.error.message


def test_failed_load_closes_partial_backend_and_allows_retry() -> None:
    attempts = 0
    created: list[FakeBackend] = []

    def factory(settings: NeMoSettings) -> FakeBackend:
        nonlocal attempts
        attempts += 1
        backend = FakeBackend(settings, fail_load=attempts == 1)
        created.append(backend)
        return backend

    output = io.BytesIO()
    second_load = load_request()
    second_load = type(second_load)(
        second_load.protocol_version,
        "load-2",
        second_load.method,
        second_load.params,
    )
    serve(
        WorkerService({"parakeet": factory}),
        request_lines(
            load_request(),
            second_load,
            make_request("close", request_id="close"),
        ),
        output,
        io.StringIO(),
    )
    responses = decoded_lines(output)

    assert isinstance(responses[0], ErrorResponse)
    assert isinstance(responses[1], SuccessResponse)
    assert created[0].calls == ["load", "close"]
    assert created[1].calls == ["load", "close"]
