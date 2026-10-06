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

from dominican_eaters_qwen3_asr import app as app_module
from dominican_eaters_qwen3_asr.adapters import Qwen3Settings
from dominican_eaters_qwen3_asr.app import WorkerService, serve


@dataclass
class FakeBackend:
    settings: Qwen3Settings

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    @property
    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id="fake/qwen3",
            model=self.settings.model,
            model_revision=self.settings.model_revision,
            language=self.settings.language,
            requested_device=self.settings.device,
            requested_precision=self.settings.precision,
            effective_device="cpu" if "load" in self.calls else None,
            effective_precision="fp32" if "load" in self.calls else None,
        )

    def load(self) -> None:
        self.calls.append("load")
        print("model diagnostics")

    def warmup(self) -> None:
        self.calls.append("warmup")

    def transcribe(self, audio_path: Path, *, duration_seconds: float | None = None) -> Transcript:
        self.calls.append("transcribe")
        return Transcript("Hola", "es", duration_seconds)

    def close(self) -> None:
        self.calls.append("close")


def request_lines(*requests: object) -> io.BytesIO:
    return io.BytesIO(b"".join(encode_message(request) for request in requests))  # type: ignore[arg-type]


def decoded_lines(output: io.BytesIO):
    return [decode_response(line) for line in output.getvalue().splitlines()]


def params(**overrides: object):
    value = {
        "backend": "qwen3_asr",
        "preset": "qwen3-asr-1.7b",
        "model": "Qwen/Qwen3-ASR-1.7B-hf",
        "model_revision": None,
        "language": "es",
        "device": "cuda",
        "precision": "fp16",
        "quantization": None,
        "prompt_template_id": "qwen3-asr-transcription-v1",
        "options": {},
    }
    value.update(overrides)
    return value


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1


def test_preflight_imports_runtime_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_loads: list[str] = []

    class ModelFactory:
        @staticmethod
        def from_pretrained(model: str, **kwargs: object) -> object:
            model_loads.append(model)
            raise AssertionError("preflight must not load weights")

    modules = {
        "transformers": SimpleNamespace(
            AutoProcessor=object(), AutoModelForMultimodalLM=ModelFactory
        ),
        "torch": SimpleNamespace(cuda=FakeCuda()),
    }
    monkeypatch.setattr(app_module.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        app_module.importlib.metadata,
        "version",
        lambda package: {"transformers": "5.13.0", "torch": "2.7.0"}[package],
    )
    monkeypatch.setattr(app_module.sys, "version_info", (3, 12, 9))
    output = io.BytesIO()
    serve(
        WorkerService({"qwen3_asr": FakeBackend}),
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
    assert response.result["environment"]["weights_loaded"] is False  # type: ignore[index]
    assert model_loads == []


def test_protocol_lifecycle_and_constraints(tmp_path: Path) -> None:
    output = io.BytesIO()
    errors = io.StringIO()
    service = WorkerService({"qwen3_asr": FakeBackend})
    serve(
        service,
        request_lines(
            make_request("describe", request_id="describe"),
            make_request("load", params(), request_id="load"),
            make_request("warmup", request_id="warmup"),
            make_request(
                "transcribe",
                {"audio_path": str(tmp_path / "audio.wav"), "duration_seconds": 1.0},
                request_id="transcribe",
            ),
            make_request("close", request_id="close"),
        ),
        output,
        errors,
    )
    responses = decoded_lines(output)
    assert all(isinstance(response, SuccessResponse) for response in responses)
    assert responses[3].result["text"] == "Hola"  # type: ignore[union-attr]
    assert "model diagnostics" in errors.getvalue()

    invalid = WorkerService({"qwen3_asr": FakeBackend})
    preflight = invalid.preflight(params(quantization="int8"))
    assert preflight["ready"] is False
    assert "does not support quantization" in str(preflight["checks"])
    preflight = invalid.preflight(params(options={"timestamps": True}))
    assert preflight["ready"] is False
    assert "forced-aligner" in str(preflight["checks"])
