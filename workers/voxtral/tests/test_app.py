from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from dominican_eaters.speech.asr import BackendDescriptor, Transcript
from dominican_eaters.speech.asr.worker_protocol import (
    SuccessResponse,
    decode_response,
    encode_message,
    make_request,
)

from dominican_eaters_voxtral import app as app_module
from dominican_eaters_voxtral.adapters import VoxtralSettings
from dominican_eaters_voxtral.app import WorkerService, serve


@dataclass
class FakeBackend:
    settings: VoxtralSettings

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    @property
    def descriptor(self) -> BackendDescriptor:
        loaded = "load" in self.calls
        return BackendDescriptor(
            backend_id=f"fake/{self.settings.model}",
            model=self.settings.model,
            model_revision=self.settings.model_revision,
            language="es",
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device="cuda" if loaded else None,
            effective_precision="fp16" if loaded else None,
            options={"batch_size": 1, "timestamps": False},
        )

    def load(self) -> None:
        self.calls.append("load")

    def warmup(self) -> None:
        self.calls.append("warmup")

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        self.calls.append(f"transcribe:{audio_path}:{duration_seconds}")
        return Transcript(text="Hola", language="es", audio_duration_seconds=duration_seconds)

    def close(self) -> None:
        self.calls.append("close")


def request_lines(*requests: object) -> io.BytesIO:
    return io.BytesIO(b"".join(encode_message(request) for request in requests))  # type: ignore[arg-type]


def decoded_lines(output: io.BytesIO):
    return [decode_response(line) for line in output.getvalue().splitlines()]


def params(*, options=None):
    return {
        "backend": "voxtral",
        "preset": "voxtral-mini-3b-2507",
        "model": "mistralai/Voxtral-Mini-3B-2507",
        "model_revision": "revision-1",
        "language": "es",
        "device": "cuda",
        "precision": "fp16",
        "quantization": None,
        "prompt_template_id": "voxtral-transcription-request-v1",
        "options": options or {},
    }


class FakeCudaInspector:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_capability(self, device: int = 0) -> tuple[int, int]:
        return (7, 5)


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    weight_loads: list[str] = []

    class ModelFactory:
        @staticmethod
        def from_pretrained(model: str, **options: object) -> object:
            weight_loads.append(model)
            raise AssertionError("preflight must not resolve model weights")

    modules = {
        "transformers": SimpleNamespace(
            AutoProcessor=object(), VoxtralForConditionalGeneration=ModelFactory
        ),
        "torch": SimpleNamespace(cuda=FakeCudaInspector()),
        "soundfile": SimpleNamespace(),
    }
    monkeypatch.setattr(app_module.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(app_module.importlib.metadata, "version", lambda name: "fixture-1")
    monkeypatch.setattr(app_module.sys, "version_info", (3, 12, 9))
    monkeypatch.setattr(app_module.platform, "python_version", lambda: "3.12.9")
    return weight_loads


def test_preflight_imports_runtime_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight_loads = install_fake_runtime(monkeypatch)
    created: list[FakeBackend] = []

    def factory(settings: VoxtralSettings) -> FakeBackend:
        backend = FakeBackend(settings)
        created.append(backend)
        return backend

    output = io.BytesIO()
    serve(
        WorkerService({"voxtral": factory}),
        request_lines(
            make_request("preflight", params(), request_id="preflight"),
            make_request("close", request_id="close"),
        ),
        output,
        io.StringIO(),
    )
    response = decoded_lines(output)[0]

    assert isinstance(response, SuccessResponse)
    assert response.result["ready"] is True
    assert response.result["environment"] == {
        "python_version": "3.12.9",
        "python_executable": app_module.sys.executable,
        "platform": app_module.platform.platform(),
        "transformers_version": "fixture-1",
        "torch_version": "fixture-1",
        "soundfile_version": "fixture-1",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_compute_capability": "7.5",
        "backend": "voxtral",
        "model": "mistralai/Voxtral-Mini-3B-2507",
        "weights_loaded": False,
    }
    assert created == []
    assert weight_loads == []


def test_complete_protocol_lifecycle() -> None:
    created: list[FakeBackend] = []

    def factory(settings: VoxtralSettings) -> FakeBackend:
        backend = FakeBackend(settings)
        created.append(backend)
        return backend

    output = io.BytesIO()
    exit_code = serve(
        WorkerService({"voxtral": factory}),
        request_lines(
            make_request("describe", request_id="describe"),
            make_request("load", params(options={"max_audio_seconds": 30}), request_id="load"),
            make_request("warmup", request_id="warmup"),
            make_request(
                "transcribe",
                {"audio_path": "/tmp/fixture.wav", "duration_seconds": 0.5},
                request_id="transcribe",
            ),
            make_request("close", request_id="close"),
        ),
        output,
        io.StringIO(),
    )
    responses = decoded_lines(output)

    assert exit_code == 0
    assert all(isinstance(response, SuccessResponse) for response in responses)
    assert responses[0].result["worker"] == "dominican-eaters-voxtral"  # type: ignore[union-attr]
    assert responses[3].result["text"] == "Hola"  # type: ignore[union-attr]
    assert created[0].settings.max_audio_seconds == 30
    assert created[0].calls == [
        "load",
        "warmup",
        "transcribe:/tmp/fixture.wav:0.5",
        "close",
    ]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"timestamps": True}, "timestamps are not supported"),
        ({"max_audio_seconds": 900}, "must be in"),
        ({"batch_size": 2}, "Unknown Voxtral options"),
    ],
)
def test_preflight_reports_invalid_profile(options: dict[str, object], message: str) -> None:
    output = io.BytesIO()
    serve(
        WorkerService({"voxtral": FakeBackend}),
        request_lines(
            make_request("preflight", params(options=options), request_id="preflight"),
            make_request("close", request_id="close"),
        ),
        output,
        io.StringIO(),
    )
    response = decoded_lines(output)[0]
    assert isinstance(response, SuccessResponse)
    assert response.result["ready"] is False
    assert message in str(response.result["checks"])
