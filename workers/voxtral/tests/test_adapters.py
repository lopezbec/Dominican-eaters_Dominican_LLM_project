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


class FakeMonoAudio:
    def __init__(self, frame_count: int = 4_000) -> None:
        self.shape = (frame_count,)
        self.slices: list[slice] = []

    def __getitem__(self, item: slice) -> FakeMonoAudio:
        self.slices.append(item)
        start = 0 if item.start is None else item.start
        stop = self.shape[0] if item.stop is None else item.stop
        return FakeMonoAudio(stop - start)


class FakeAudioFrames:
    def __init__(self, frame_count: int = 4_000) -> None:
        self.shape = (frame_count, 2)
        self.mono = FakeMonoAudio(frame_count)
        self.mean_axes: list[int] = []

    def mean(self, *, axis: int) -> FakeMonoAudio:
        self.mean_axes.append(axis)
        return self.mono


class FakeSoundFile:
    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.frames = FakeAudioFrames()
        self.calls: list[tuple[str, str, bool]] = []

    def read(self, file: str, *, dtype: str, always_2d: bool):
        self.calls.append((file, dtype, always_2d))
        return self.frames, self.sample_rate


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
        self.requests: list[dict[str, object]] = []
        self.decodes: list[tuple[object, bool]] = []

    def apply_transcription_request(self, **request: object) -> FakeInputs:
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
    soundfile = FakeSoundFile()
    bundle = VoxtralRuntimeBundle(processor_factory, model_factory, torch, soundfile)
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
            "audio": bundle.soundfile.frames.mono,
            "model_id": "mistralai/Voxtral-Mini-3B-2507",
            "sampling_rate": 16_000,
            "format": "WAV",
        }
    ]
    assert bundle.soundfile.calls == [(str(tmp_path / "sample.wav"), "float32", True)]
    assert bundle.soundfile.frames.mean_axes == [1]
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


def test_audio_context_policy_rejects_unknown_and_nonpositive_inputs(tmp_path: Path) -> None:
    bundle, _processor, model, *_rest = runtime()
    backend = VoxtralBackend(VoxtralSettings(max_audio_seconds=10), runtime_loader=lambda: bundle)
    backend.load()
    non_wav = tmp_path / "sample.mp3"
    non_wav.write_bytes(b"fixture")

    with pytest.raises(AudioPolicyError, match="duration is required"):
        backend.transcribe(non_wav)
    with pytest.raises(AudioPolicyError, match="positive"):
        backend.transcribe(non_wav, duration_seconds=0)
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
        ({"chunk_seconds": 0}, "positive and finite"),
        ({"language": "en"}, "Spanish"),
    ],
)
def test_settings_enforce_initial_profile(settings: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VoxtralSettings(**settings)  # type: ignore[arg-type]


def test_normalized_wav_contract_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    bundle, _processor, _model, *_rest = runtime()
    bundle.soundfile.sample_rate = 44_100
    backend = VoxtralBackend(VoxtralSettings(), runtime_loader=lambda: bundle)
    backend.load()

    with pytest.raises(AudioPolicyError, match="requires normalized 16000 Hz"):
        backend.transcribe(wav_file(tmp_path / "sample.wav"), duration_seconds=0.25)


def test_long_audio_is_joined_from_t4_bounded_segments(tmp_path: Path) -> None:
    bundle, processor, model, _pf, _mf, torch = runtime()
    bundle.soundfile.frames = FakeAudioFrames(40_000)
    backend = VoxtralBackend(
        VoxtralSettings(max_audio_seconds=10, chunk_seconds=1),
        runtime_loader=lambda: bundle,
    )
    backend.load()

    result = backend.transcribe(wav_file(tmp_path / "sample.wav"))

    assert len(processor.requests) == 3
    assert [request["audio"].shape[0] for request in processor.requests] == [
        16_000,
        16_000,
        8_000,
    ]
    assert len(model.generate_calls) == 3
    assert result.text == "Hola, mi gente.\nHola, mi gente.\nHola, mi gente."
    assert result.audio_duration_seconds == 2.5
    assert result.metadata["long_form_policy"] == "fixed-segment"
    assert result.metadata["segment_seconds"] == 1
    assert result.metadata["segment_count"] == 3
    assert torch.cuda.empty_calls == 3


def test_failed_generation_releases_cuda_cache(tmp_path: Path) -> None:
    bundle, _processor, model, _pf, _mf, torch = runtime()

    def fail(**_inputs: object) -> object:
        raise RuntimeError("CUDA out of memory")

    model.generate = fail  # type: ignore[method-assign]
    backend = VoxtralBackend(VoxtralSettings(), runtime_loader=lambda: bundle)
    backend.load()

    with pytest.raises(RuntimeError, match="out of memory"):
        backend.transcribe(wav_file(tmp_path / "sample.wav"))

    assert torch.cuda.empty_calls == 1
