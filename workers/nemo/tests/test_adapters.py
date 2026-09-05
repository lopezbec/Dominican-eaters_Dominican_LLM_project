from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

from dominican_eaters_nemo.adapters import (
    CanaryBackend,
    NeMoRuntimeBundle,
    NeMoSettings,
    ParakeetBackend,
    ShortAudioError,
    ShortAudioPolicy,
)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.empty_cache_calls = 0
        self.reset_peak_calls = 0
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_peak_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def max_memory_allocated(self) -> int:
        return 100

    def max_memory_reserved(self) -> int:
        return 200


class FakeTorch:
    def __init__(self, cuda: bool) -> None:
        self.cuda = FakeCuda(cuda)


class Hypothesis:
    text = "Hola, mi gente"
    language = "es"
    timestamp = {"word": [{"word": "Hola", "start": 0.0, "end": 0.2}]}


class FakeModel:
    def __init__(self, output: list[object] | None = None) -> None:
        self.output = output if output is not None else [Hypothesis()]
        self.devices: list[str] = []
        self.precisions: list[str] = []
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.eval_calls = 0

    def transcribe(self, audio: list[str], **options: object) -> list[object]:
        self.calls.append((audio, options))
        return self.output

    def to(self, device: str) -> FakeModel:
        self.devices.append(device)
        return self

    def eval(self) -> object:
        self.eval_calls += 1
        return self

    def half(self) -> FakeModel:
        self.precisions.append("fp16")
        return self

    def bfloat16(self) -> FakeModel:
        self.precisions.append("bf16")
        return self

    def float(self) -> FakeModel:
        self.precisions.append("fp32")
        return self


class FakeFactory:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.loads: list[str] = []

    def from_pretrained(self, *, model_name: str) -> FakeModel:
        self.loads.append(model_name)
        return self.model


class FakeModels:
    def __init__(self, factory: FakeFactory) -> None:
        self.ASRModel = factory


class FakeNeMo:
    def __init__(self, factory: FakeFactory) -> None:
        self.models = FakeModels(factory)


def runtime(*, cuda: bool = False, output: list[object] | None = None):
    model = FakeModel(output)
    factory = FakeFactory(model)
    torch = FakeTorch(cuda)
    return NeMoRuntimeBundle(FakeNeMo(factory), torch), model, factory, torch


def wav_file(path: Path, duration: float = 0.2) -> Path:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * int(16_000 * duration))
    return path


def test_import_does_not_load_nemo_or_torch() -> None:
    assert "nemo.collections.asr" not in sys.modules
    assert "torch" not in sys.modules


def test_parakeet_lifecycle_and_primitive_result(tmp_path: Path) -> None:
    bundle, model, factory, torch = runtime()
    backend = ParakeetBackend(
        NeMoSettings(
            model="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
            precision="fp32",
            timestamps=True,
        ),
        runtime_loader=lambda: bundle,
    )

    backend.load()
    backend.warmup()
    result = backend.transcribe(wav_file(tmp_path / "sample.wav"))
    backend.close()

    assert factory.loads == ["nvidia/parakeet-tdt-0.6b-v3"]
    assert model.devices == ["cpu"]
    assert model.precisions == ["fp32"]
    assert model.eval_calls == 1
    assert model.calls[0][1] == {"timestamps": True}
    assert model.calls[1][1] == {"timestamps": True}
    assert result.text == "Hola, mi gente"
    assert result.language == "es"
    assert result.metadata["timestamps"] == {"word": [{"word": "Hola", "start": 0.0, "end": 0.2}]}
    assert torch.cuda.empty_cache_calls == 0


def test_canary_always_sets_explicit_spanish_source_and_target(tmp_path: Path) -> None:
    bundle, model, _factory, torch = runtime(cuda=True)
    backend = CanaryBackend(
        NeMoSettings(model="nvidia/canary-1b-flash", timestamps=False),
        runtime_loader=lambda: bundle,
    )

    backend.load()
    result = backend.transcribe(wav_file(tmp_path / "sample.wav"))

    assert model.devices == ["cuda"]
    assert model.precisions == ["fp16"]
    assert model.calls[0][1] == {
        "timestamps": False,
        "source_lang": "es",
        "target_lang": "es",
    }
    assert result.language == "es"
    assert result.gpu_peak_allocated_bytes == 100
    assert result.gpu_peak_reserved_bytes == 200
    assert torch.cuda.reset_peak_calls == 1
    assert torch.cuda.synchronize_calls == 1
    assert backend.descriptor.options["source_lang"] == "es"
    assert backend.descriptor.options["target_lang"] == "es"


def test_short_audio_reject_policy_is_strict_and_configurable(tmp_path: Path) -> None:
    bundle, model, _factory, _torch = runtime()
    rejecting = ParakeetBackend(
        NeMoSettings(model="model", minimum_audio_seconds=0.5),
        runtime_loader=lambda: bundle,
    )
    rejecting.load()
    non_wav = tmp_path / "sample.mp3"
    non_wav.write_bytes(b"not decoded by the offline fake")

    with pytest.raises(ShortAudioError, match="below the configured minimum"):
        rejecting.transcribe(wav_file(tmp_path / "short.wav", 0.1))
    with pytest.raises(ShortAudioError, match="duration is required"):
        rejecting.transcribe(non_wav)

    allowing = ParakeetBackend(
        NeMoSettings(model="model", short_audio_policy=ShortAudioPolicy.ALLOW),
        runtime_loader=lambda: bundle,
    )
    allowing.load()
    allowing.transcribe(non_wav)
    assert len(model.calls) == 1


def test_non_wav_duration_can_be_supplied_by_controller(tmp_path: Path) -> None:
    path = tmp_path / "sample.flac"
    path.write_bytes(b"offline fixture")
    bundle, model, _factory, _torch = runtime()
    backend = ParakeetBackend(
        NeMoSettings(model="model", minimum_audio_seconds=0.2),
        runtime_loader=lambda: bundle,
    )
    backend.load()

    transcript = backend.transcribe(path, duration_seconds=0.25)

    assert transcript.metadata["duration_seconds"] == 0.25
    assert len(model.calls) == 1


def test_explicit_cuda_does_not_fall_back() -> None:
    bundle, _model, factory, _torch = runtime(cuda=False)
    backend = ParakeetBackend(
        NeMoSettings(model="model", device="cuda"), runtime_loader=lambda: bundle
    )

    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        backend.load()
    assert factory.loads == []


def test_invalid_or_nonprimitive_model_results_are_rejected(tmp_path: Path) -> None:
    audio = wav_file(tmp_path / "sample.wav")
    bundle, _model, _factory, _torch = runtime(output=[{"text": 42}])
    invalid_text = ParakeetBackend(NeMoSettings(model="model"), runtime_loader=lambda: bundle)
    invalid_text.load()
    with pytest.raises(TypeError, match="string text"):
        invalid_text.transcribe(audio)

    bundle, _model, _factory, _torch = runtime(
        output=[{"text": "hola", "timestamp": {"unsupported": object()}}]
    )
    invalid_metadata = ParakeetBackend(NeMoSettings(model="model"), runtime_loader=lambda: bundle)
    invalid_metadata.load()
    with pytest.raises(TypeError, match="Unsupported NeMo metadata"):
        invalid_metadata.transcribe(audio)


def test_settings_reject_implicit_language_and_invalid_precision_policy() -> None:
    with pytest.raises(ValueError, match="Spanish"):
        NeMoSettings(model="model", language="en")
    with pytest.raises(ValueError, match="requires CUDA"):
        NeMoSettings(model="model", device="cpu", precision="bf16")
    with pytest.raises(ValueError, match="positive"):
        NeMoSettings(model="model", minimum_audio_seconds=0)
