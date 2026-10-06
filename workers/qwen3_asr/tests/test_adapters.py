from __future__ import annotations

import contextlib
import wave
from pathlib import Path

import pytest

from dominican_eaters_qwen3_asr.adapters import (
    Qwen3ASRBackend,
    Qwen3RuntimeBundle,
    Qwen3Settings,
)


class FakeCuda:
    def __init__(self, available: bool) -> None:
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
        return 100

    def max_memory_reserved(self) -> int:
        return 200


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, available: bool) -> None:
        self.cuda = FakeCuda(available)

    def inference_mode(self):
        return contextlib.nullcontext()


class FakeTensor:
    def __init__(self, width: int = 4) -> None:
        self.shape = (1, width)
        self.slices: list[object] = []

    def __getitem__(self, item: object):
        self.slices.append(item)
        return self


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeTensor())
        self.moved_to: tuple[object, object] | None = None

    def to(self, device: object, dtype: object):
        self.moved_to = (device, dtype)
        return self


class FakeProcessor:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.requests: list[dict[str, object]] = []
        self.inputs = FakeInputs()

    def apply_transcription_request(self, **kwargs: object):
        self.requests.append(kwargs)
        return self.inputs

    def decode(self, token_ids: object, *, return_format: str) -> list[object]:
        assert return_format == "parsed"
        return [self.parsed]


class FakeModel:
    device = "cuda"
    dtype = "float16"

    def __init__(self) -> None:
        self.generated = FakeTensor(8)
        self.generate_calls: list[dict[str, object]] = []
        self.moves: list[str] = []
        self.evaluated = False

    def to(self, device: str):
        self.device = device
        self.moves.append(device)
        return self

    def eval(self) -> None:
        self.evaluated = True

    def generate(self, **inputs: object):
        self.generate_calls.append(inputs)
        return self.generated


class FakeFactory:
    def __init__(self, value: object) -> None:
        self.value = value
        self.loads: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model: str, **kwargs: object):
        self.loads.append((model, kwargs))
        return self.value


def wav_file(path: Path) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 8_000)
    return path


def runtime(*, cuda: bool, parsed: object):
    torch = FakeTorch(cuda)
    model = FakeModel()
    processor = FakeProcessor(parsed)
    model_factory = FakeFactory(model)
    processor_factory = FakeFactory(processor)
    bundle = Qwen3RuntimeBundle(torch, model_factory, processor_factory)  # type: ignore[arg-type]
    return bundle, torch, model, processor, model_factory, processor_factory


def test_official_flow_forces_spanish_and_uses_fp16_batch_one(tmp_path: Path) -> None:
    bundle, torch, model, processor, model_factory, processor_factory = runtime(
        cuda=True, parsed={"language": "Spanish", "transcription": "Hola mundo"}
    )
    backend = Qwen3ASRBackend(
        Qwen3Settings(
            model="Qwen/Qwen3-ASR-1.7B-hf",
            model_revision="revision-1",
            device="cuda",
            precision="fp16",
        ),
        runtime_loader=lambda: bundle,
    )

    backend.load()
    result = backend.transcribe(wav_file(tmp_path / "sample.wav"))

    assert processor_factory.loads == [("Qwen/Qwen3-ASR-1.7B-hf", {"revision": "revision-1"})]
    assert model_factory.loads == [
        (
            "Qwen/Qwen3-ASR-1.7B-hf",
            {
                "dtype": "float16",
                "attn_implementation": "sdpa",
                "revision": "revision-1",
            },
        )
    ]
    assert processor.requests == [
        {"audio": str((tmp_path / "sample.wav").resolve()), "language": "Spanish"}
    ]
    assert model.generate_calls[0]["max_new_tokens"] == 256
    assert model.generate_calls[0]["do_sample"] is False
    assert result.text == "Hola mundo"
    assert result.language == "es"
    assert result.audio_duration_seconds == 0.5
    assert result.gpu_peak_allocated_bytes == 100
    assert result.metadata["language_mode"] == "forced"
    assert torch.cuda.reset_calls == 1
    assert model.evaluated is True


def test_auto_language_omits_hint_and_normalizes_detected_language(tmp_path: Path) -> None:
    bundle, _torch, _model, processor, _model_factory, _processor_factory = runtime(
        cuda=False, parsed={"language": "English", "transcription": "Hello"}
    )
    backend = Qwen3ASRBackend(
        Qwen3Settings(model="model", language="auto", device="cpu", precision="fp32"),
        runtime_loader=lambda: bundle,
    )
    backend.load()

    result = backend.transcribe(wav_file(tmp_path / "sample.wav"), duration_seconds=1.25)

    assert processor.requests == [{"audio": str((tmp_path / "sample.wav").resolve())}]
    assert result.language == "en"
    assert result.audio_duration_seconds == 1.25


def test_settings_and_outputs_reject_unsupported_initial_profiles(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'es' or 'auto'"):
        Qwen3Settings(model="model", language="en")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires CUDA"):
        Qwen3Settings(model="model", device="cpu", precision="fp16")
    bundle, _torch, _model, _processor, _model_factory, _processor_factory = runtime(
        cuda=False, parsed={"transcription": "Hola"}
    )
    backend = Qwen3ASRBackend(
        Qwen3Settings(model="model", language="auto", device="cpu", precision="fp32"),
        runtime_loader=lambda: bundle,
    )
    backend.load()
    with pytest.raises(TypeError, match="contain a language"):
        backend.transcribe(wav_file(tmp_path / "sample.wav"))


def test_cuda_request_never_falls_back() -> None:
    bundle, _torch, _model, _processor, model_factory, _processor_factory = runtime(
        cuda=False, parsed={"language": "Spanish", "transcription": "Hola"}
    )
    backend = Qwen3ASRBackend(
        Qwen3Settings(model="model", device="cuda", precision="fp16"),
        runtime_loader=lambda: bundle,
    )
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        backend.load()
    assert model_factory.loads == []
