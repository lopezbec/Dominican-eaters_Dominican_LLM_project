from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

from dominican_eaters_voxtral.adapters import (
    AudioPolicyError,
    VoxtralBackend,
    VoxtralRuntimeBundle,
    VoxtralSettings,
)


class FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.reset_calls = 0
        self.sync_calls = 0
        self.empty_calls = 0

    def is_available(self) -> bool:
        return self.available

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def synchronize(self) -> None:
        self.sync_calls += 1

    def empty_cache(self) -> None:
        self.empty_calls += 1

    def max_memory_allocated(self) -> int:
        return 111

    def max_memory_reserved(self) -> int:
        return 222


class FakeTorch:
    float16 = "fake-fp16"

    def __init__(self, available: bool = True) -> None:
        self.cuda = FakeCuda(available)


class FakeIds:
    shape = (1, 12)


class FakeInputs:
    input_ids = FakeIds()

    def __init__(self) -> None:
        self.to_calls: list[tuple[str, object]] = []
        self.values = {"input_ids": "ids", "input_features": "features"}

    def to(self, device: str, *, dtype: object) -> FakeInputs:
        self.to_calls.append((device, dtype))
        return self

    def keys(self) -> object:
        return self.values.keys()

    def __getitem__(self, key: str) -> object:
        return self.values[key]


class FakeOutputs:
    def __getitem__(self, item: object) -> str:
        assert isinstance(item, tuple)
        assert item[1] == slice(12, None)
        return "generated-token-slice"


class FakeProcessor:
    def __init__(self) -> None:
        self.inputs = FakeInputs()
        self.requests: list[dict[str, str]] = []
        self.decodes: list[tuple[object, bool]] = []

    def apply_transcription_request(self, **request: str) -> FakeInputs:
        self.requests.append(request)
        return self.inputs

    def batch_decode(self, outputs: object, *, skip_special_tokens: bool) -> list[str]:
        self.decodes.append((outputs, skip_special_tokens))
        return ["  Hola, mi gente.  "]


class FakeModel:
    device = "cuda"

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.eval_calls = 0

    def generate(self, **inputs: object) -> FakeOutputs:
        self.generate_calls.append(inputs)
        return FakeOutputs()

    def eval(self) -> object:
        self.eval_calls += 1
        return self


class FakeFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model: str, **options: object) -> object:
        self.calls.append((model, options))
        return self.value


def runtime(*, cuda: bool = True):
    processor = FakeProcessor()
    model = FakeModel()
    processor_factory = FakeFactory(processor)
    model_factory = FakeFactory(model)
    torch = FakeTorch(cuda)
    bundle = VoxtralRuntimeBundle(processor_factory, model_factory, torch)
    return bundle, processor, model, processor_factory, model_factory, torch


def wav_file(path: Path, duration: float = 0.25) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * int(16_000 * duration))
    return path


def test_import_is_lazy() -> None:
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_official_transcription_request_flow_and_fp16_policy(tmp_path: Path) -> None:
    bundle, processor, model, processor_factory, model_factory, torch = runtime()
    backend = VoxtralBackend(
        VoxtralSettings(
            model="mistralai/Voxtral-Mini-3B-2507",
            model_revision="revision-1",
            max_audio_seconds=30,
            max_new_tokens=64,
        ),
        runtime_loader=lambda: bundle,
    )

    backend.load()
    result = backend.transcribe(wav_file(tmp_path / "sample.wav"), duration_seconds=0.25)

    assert processor_factory.calls == [
        ("mistralai/Voxtral-Mini-3B-2507", {"revision": "revision-1"})
    ]
    assert model_factory.calls == [
        (
            "mistralai/Voxtral-Mini-3B-2507",
            {"torch_dtype": "fake-fp16", "device_map": "cuda", "revision": "revision-1"},
        )
    ]
    assert processor.requests == [
        {
            "language": "es",
            "audio": str(tmp_path / "sample.wav"),
            "model_id": "mistralai/Voxtral-Mini-3B-2507",
        }
    ]
    assert processor.inputs.to_calls == [("cuda", "fake-fp16")]
    assert model.generate_calls == [
        {"input_ids": "ids", "input_features": "features", "max_new_tokens": 64}
    ]
    assert processor.decodes == [("generated-token-slice", True)]
    assert result.text == "Hola, mi gente."
    assert result.language == "es"
    assert result.gpu_peak_allocated_bytes == 111
    assert result.gpu_peak_reserved_bytes == 222
    assert result.metadata["timestamps"] is False
    assert torch.cuda.reset_calls == 1
    assert torch.cuda.sync_calls == 1
    assert backend.descriptor.effective_precision == "fp16"
    assert backend.descriptor.options["batch_size"] == 1


def test_audio_context_policy_rejects_unknown_and_long_inputs(tmp_path: Path) -> None:
    bundle, _processor, model, *_rest = runtime()
    backend = VoxtralBackend(VoxtralSettings(max_audio_seconds=10), runtime_loader=lambda: bundle)
    backend.load()
    non_wav = tmp_path / "sample.mp3"
    non_wav.write_bytes(b"fixture")

    with pytest.raises(AudioPolicyError, match="duration is required"):
        backend.transcribe(non_wav)
    with pytest.raises(AudioPolicyError, match="exceeds"):
        backend.transcribe(non_wav, duration_seconds=10.1)
    assert model.generate_calls == []


def test_explicit_cuda_never_falls_back_and_close_is_idempotent() -> None:
    bundle, _processor, _model, _pf, model_factory, torch = runtime(cuda=False)
    backend = VoxtralBackend(VoxtralSettings(), runtime_loader=lambda: bundle)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        backend.load()
    assert model_factory.calls == []
    backend.close()
    backend.close()
    assert torch.cuda.empty_calls == 1


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"device": "cpu"}, "requires CUDA"),
        ({"precision": "bf16"}, "requires FP16"),
        ({"timestamps": True}, "timestamps are not supported"),
        ({"max_audio_seconds": 601}, "must be in"),
        ({"max_new_tokens": 501}, "must be in"),
        ({"language": "en"}, "Spanish"),
    ],
)
def test_settings_enforce_initial_profile(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VoxtralSettings(**settings)  # type: ignore[arg-type]
