from __future__ import annotations

import io
from dataclasses import dataclass, field

from dominican_eaters.speech.asr.worker_protocol import (
    ErrorResponse,
    JSONObject,
    PreflightCheck,
    SuccessResponse,
    decode_response,
    encode_message,
    make_preflight_result,
    make_request,
)
from dominican_eaters.speech.asr.worker_server import (
    WorkerApplication,
    WorkerDispatcher,
    WorkerError,
    serve,
)


def runtime_params() -> JSONObject:
    return {
        "backend": "fixture",
        "preset": "fixture-small",
        "model": "org/fixture-small",
        "model_revision": "revision-1",
        "language": "es",
        "device": "cpu",
        "precision": "fp32",
        "quantization": None,
        "prompt_template_id": "transcribe-es-v1",
        "options": {},
    }


@dataclass
class FakeApplication:
    calls: list[str] = field(default_factory=list)

    def describe(self) -> JSONObject:
        self.calls.append("describe")
        return {"worker": "fixture"}

    def preflight(self, params: JSONObject) -> JSONObject:
        self.calls.append(f"preflight:{params['model']}")
        print("runtime diagnostic")
        return make_preflight_result(
            [PreflightCheck("runtime", "passed", "error", "runtime import succeeded")],
            environment={"lock_id": "fixture-lock"},
        )

    def load(self, params: JSONObject) -> JSONObject:
        self.calls.append(f"load:{params['model']}")
        return {"loaded": True}

    def warmup(self) -> JSONObject:
        self.calls.append("warmup")
        return {}

    def transcribe(self, params: JSONObject) -> JSONObject:
        self.calls.append(f"transcribe:{params['audio_path']}")
        return {"text": "Hola"}

    def close(self) -> None:
        self.calls.append("close")


def decoded_lines(output: io.BytesIO) -> list[SuccessResponse | ErrorResponse]:
    return [decode_response(line) for line in output.getvalue().splitlines()]


def test_dispatcher_serves_v2_preflight_without_loading_a_model() -> None:
    application = FakeApplication()
    requests = io.BytesIO(
        encode_message(make_request("describe", request_id="describe"))
        + encode_message(make_request("preflight", runtime_params(), request_id="preflight"))
        + encode_message(make_request("close", request_id="close"))
    )
    output = io.BytesIO()
    errors = io.StringIO()

    exit_code = serve(WorkerDispatcher(application), requests, output, errors)
    responses = decoded_lines(output)

    assert exit_code == 0
    assert all(isinstance(response, SuccessResponse) for response in responses)
    assert responses[1].result["ready"] is True
    assert application.calls == [
        "describe",
        "preflight:org/fixture-small",
        "close",
    ]
    assert "runtime diagnostic" in errors.getvalue()
    assert b"runtime diagnostic" not in output.getvalue()


def test_server_preserves_v1_response_version() -> None:
    application = FakeApplication()
    request = make_request("describe", request_id="legacy", protocol_version=1)
    output = io.BytesIO()

    serve(
        WorkerDispatcher(application),
        io.BytesIO(encode_message(request) + encode_message(make_request("close"))),
        output,
        io.StringIO(),
    )

    response = decoded_lines(output)[0]
    assert response.protocol_version == 1


def test_server_uses_custom_structured_error_mapping() -> None:
    application = FakeApplication()

    def broken_preflight(_params: JSONObject) -> JSONObject:
        raise OSError("temporary driver failure")

    application.preflight = broken_preflight  # type: ignore[method-assign]
    output = io.BytesIO()
    requests = io.BytesIO(
        encode_message(make_request("preflight", runtime_params(), request_id="preflight"))
        + encode_message(make_request("close", request_id="close"))
    )

    serve(
        WorkerDispatcher(application),
        requests,
        output,
        io.StringIO(),
        error_mapper=lambda error: WorkerError("driver_unavailable", str(error), True),
    )

    response = decoded_lines(output)[0]
    assert isinstance(response, ErrorResponse)
    assert response.error.code == "driver_unavailable"
    assert response.error.retryable is True


def test_eof_closes_application_and_reports_cleanup_failure() -> None:
    class BrokenCloseApplication(FakeApplication):
        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    output = io.BytesIO()

    exit_code = serve(
        WorkerDispatcher(BrokenCloseApplication()),
        io.BytesIO(encode_message(make_request("describe"))),
        output,
        io.StringIO(),
    )

    assert exit_code == 1


def test_fake_application_satisfies_shared_contract() -> None:
    application: WorkerApplication = FakeApplication()

    assert application.describe()["worker"] == "fixture"
