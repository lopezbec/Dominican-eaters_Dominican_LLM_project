from __future__ import annotations

import io
import json
from pathlib import Path

from dominican_eaters.speech.asr import BackendDescriptor, Transcript
from dominican_eaters.speech.asr.worker_protocol import PreflightCheck, make_request

from dominican_eaters_granite.adapters import GraniteSettings
from dominican_eaters_granite.app import RuntimeInspection, WorkerService, serve


class FakeBackend:
    def __init__(self, settings: GraniteSettings) -> None:
        self.settings = settings
        self.loaded = False
        self.closed = False

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="ibm-granite/fake",
            model=self.settings.model,
            model_revision=self.settings.model_revision,
            language="es",
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device="cuda" if self.loaded else None,
            effective_precision="fp16" if self.loaded else None,
        )

    def load(self) -> None:
        self.loaded = True

    def warmup(self) -> None:
        assert self.loaded

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        return Transcript("Hola", "es", duration_seconds, metadata={"path": str(audio_path)})

    def close(self) -> None:
        self.closed = True


def params(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "backend": "granite",
        "preset": "granite-speech-4.1-2b",
        "model": "ibm-granite/granite-speech-4.1-2b",
        "model_revision": None,
        "language": "es",
        "device": "cuda",
        "precision": "fp16",
        "quantization": "none",
        "prompt_template_id": "granite-speech-4.1-asr-punctuated-v1",
        "options": {},
    }
    result.update(overrides)
    return result


def test_protocol_lifecycle_contract() -> None:
    created: list[FakeBackend] = []

    def factory(settings: GraniteSettings) -> FakeBackend:
        backend = FakeBackend(settings)
        created.append(backend)
        return backend

    requests = [
        make_request("describe", request_id="describe"),
        make_request("load", params(), request_id="load"),  # type: ignore[arg-type]
        make_request("warmup", request_id="warmup"),
        make_request(
            "transcribe",
            {"audio_path": "/tmp/sample.wav", "duration_seconds": 1.25},
            request_id="transcribe",
        ),
        make_request("close", request_id="close"),
    ]
    frames = b"".join(
        (
            json.dumps(
                {
                    "protocol_version": request.protocol_version,
                    "request_id": request.request_id,
                    "method": request.method,
                    "params": request.params,
                }
            )
            + "\n"
        ).encode()
        for request in requests
    )
    incoming = io.BytesIO(frames)
    outgoing = io.BytesIO()

    assert serve(WorkerService({"granite": factory}), incoming, outgoing, io.StringIO()) == 0
    responses = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert all(response["ok"] for response in responses)
    assert responses[0]["result"]["backends"] == ["granite"]
    assert responses[3]["result"]["text"] == "Hola"
    assert created[0].closed is True


def test_preflight_is_no_weight_and_reports_environment() -> None:
    factory_calls = 0

    def factory(settings: GraniteSettings) -> FakeBackend:
        nonlocal factory_calls
        factory_calls += 1
        return FakeBackend(settings)

    inspection = RuntimeInspection(
        (PreflightCheck("transformers_api", "passed", "error", "available"),),
        {"transformers_version": "4.52.1", "cuda_available": True},
    )
    service = WorkerService(
        {"granite": factory}, runtime_inspector=lambda _device, _precision: inspection
    )
    result = service.preflight(params())  # type: ignore[arg-type]

    assert result["ready"] is True
    assert result["environment"]["weights_loaded"] is False  # type: ignore[index]
    assert factory_calls == 0


def test_preflight_rejects_quantization_and_unknown_options() -> None:
    service = WorkerService(runtime_inspector=lambda _device, _precision: RuntimeInspection((), {}))
    result = service.preflight(params(quantization="q4"))  # type: ignore[arg-type]
    assert result["ready"] is False
    result = service.preflight(params(options={"timestamps": True}))  # type: ignore[arg-type]
    assert result["ready"] is False


def test_load_error_is_structured_and_does_not_corrupt_stdout() -> None:
    def failing_factory(settings: GraniteSettings) -> FakeBackend:
        class Failing(FakeBackend):
            def load(self) -> None:
                print("model diagnostic")
                raise RuntimeError("CUDA out of memory")

        return Failing(settings)

    request = make_request("load", params(), request_id="load")  # type: ignore[arg-type]
    frame = (
        json.dumps(
            {
                "protocol_version": request.protocol_version,
                "request_id": request.request_id,
                "method": request.method,
                "params": request.params,
            }
        )
        + "\n"
    ).encode()
    outgoing = io.BytesIO()
    diagnostics = io.StringIO()
    serve(
        WorkerService({"granite": failing_factory}),
        io.BytesIO(frame),
        outgoing,
        diagnostics,
    )
    response = json.loads(outgoing.getvalue())
    assert response["error"]["code"] == "cuda_oom"
    assert "model diagnostic" in diagnostics.getvalue()
