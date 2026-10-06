from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dominican_eaters_granite.adapters import (
    PROMPT,
    GraniteBackend,
    GraniteRuntimeBundle,
    GraniteSettings,
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
        return 123

    def max_memory_reserved(self) -> int:
        return 456


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.indexes: list[object] = []
        self.devices: list[str] = []

    def to(self, device: str) -> FakeTensor:
        self.devices.append(device)
        return self

    def __getitem__(self, key: object) -> FakeTensor:
        self.indexes.append(key)
        return FakeTensor((5,))

    def unsqueeze(self, dimension: int) -> FakeTensor:
        return FakeTensor((1, *self.shape))


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, cuda: bool) -> None:
        self.cuda = FakeCuda(cuda)
        self.zero_calls: list[int] = []

    def zeros(self, size: int) -> FakeTensor:
        self.zero_calls.append(size)
        return FakeTensor((size,))


class FakeAudio:
    def __init__(self, waveform: FakeTensor | None = None, rate: int = 16_000) -> None:
        self.waveform = waveform or FakeTensor((1, 3200))
        self.rate = rate
        self.calls: list[tuple[str, bool]] = []

    def load(self, path: str, *, normalize: bool) -> tuple[FakeTensor, int]:
        self.calls.append((path, normalize))
        return self.waveform, self.rate


class FakeTokenizer:
    def __init__(self) -> None:
        self.chats: list[list[dict[str, str]]] = []
        self.decoded: list[FakeTensor] = []

    def apply_chat_template(
        self, chat: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.chats.append(chat)
        return "rendered prompt"

    def batch_decode(
        self, tokens: FakeTensor, *, add_special_tokens: bool, skip_special_tokens: bool
    ) -> list[str]:
        assert add_special_tokens is False
        assert skip_special_tokens is True
        self.decoded.append(tokens)
        return ["  Hola, mi gente.  "]


class FakeBatch(dict[str, FakeTensor]):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeTensor((1, 7)), input_features=FakeTensor((1, 10)))
        self.devices: list[str] = []

    def to(self, device: str) -> FakeBatch:
        self.devices.append(device)
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[str, FakeTensor, str, str]] = []
        self.batches: list[FakeBatch] = []

    def __call__(
        self, prompt: str, waveform: FakeTensor, *, device: str, return_tensors: str
    ) -> FakeBatch:
        self.calls.append((prompt, waveform, device, return_tensors))
        batch = FakeBatch()
        self.batches.append(batch)
        return batch


class FakeProcessorFactory:
    def __init__(self, processor: FakeProcessor) -> None:
        self.processor = processor
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model: str, **kwargs: object) -> FakeProcessor:
        self.calls.append((model, kwargs))
        return self.processor


class FakeModel:
    def __init__(self) -> None:
        self.devices: list[str] = []
        self.eval_calls = 0
        self.generate_calls: list[dict[str, object]] = []
        self.output = FakeTensor((1, 12))

    def to(self, device: str) -> FakeModel:
        self.devices.append(device)
        return self

    def eval(self) -> object:
        self.eval_calls += 1
        return self

    def generate(self, **kwargs: object) -> FakeTensor:
        self.generate_calls.append(kwargs)
        return self.output


class FakeModelFactory:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model: str, **kwargs: object) -> FakeModel:
        self.calls.append((model, kwargs))
        return self.model


def runtime(cuda: bool = True):
    torch = FakeTorch(cuda)
    audio = FakeAudio()
    processor = FakeProcessor()
    model = FakeModel()
    processor_factory = FakeProcessorFactory(processor)
    model_factory = FakeModelFactory(model)
    bundle = GraniteRuntimeBundle(torch, audio, processor_factory, model_factory)
    return bundle, torch, audio, processor, model, processor_factory, model_factory


def test_import_does_not_load_heavy_runtime() -> None:
    assert "transformers" not in sys.modules
    assert "torchaudio" not in sys.modules


def test_official_fp16_flow_is_deterministic_and_slices_input(tmp_path: Path) -> None:
    bundle, torch, audio, processor, model, processor_factory, model_factory = runtime()
    backend = GraniteBackend(
        GraniteSettings(
            model="ibm-granite/granite-speech-4.1-2b",
            model_revision="abc123",
            device="cuda",
            precision="fp16",
        ),
        runtime_loader=lambda: bundle,
    )
    path = tmp_path / "sample.wav"
    path.write_bytes(b"offline fixture")

    backend.load()
    transcript = backend.transcribe(path.resolve(), duration_seconds=0.2)

    assert processor_factory.calls == [
        ("ibm-granite/granite-speech-4.1-2b", {"revision": "abc123"})
    ]
    assert model_factory.calls == [
        (
            "ibm-granite/granite-speech-4.1-2b",
            {"torch_dtype": "float16", "revision": "abc123"},
        )
    ]
    assert model.devices == ["cuda"]
    assert model.eval_calls == 1
    assert processor.tokenizer.chats == [[{"role": "user", "content": PROMPT}]]
    assert processor.calls[0][0] == "rendered prompt"
    assert processor.calls[0][2:] == ("cuda", "pt")
    assert processor.batches[0].devices == ["cuda"]
    assert model.generate_calls[0]["max_new_tokens"] == 200
    assert model.generate_calls[0]["do_sample"] is False
    assert model.generate_calls[0]["num_beams"] == 1
    assert model.output.indexes == [(0, slice(7, None, None))]
    assert transcript.text == "Hola, mi gente."
    assert transcript.language == "es"
    assert transcript.gpu_peak_allocated_bytes == 123
    assert transcript.gpu_peak_reserved_bytes == 456
    assert audio.calls == [(str(path.resolve()), True)]
    assert torch.cuda.reset_calls == 1
    assert torch.cuda.sync_calls == 1


def test_warmup_uses_one_second_synthetic_audio() -> None:
    bundle, torch, _audio, processor, model, _processor_factory, _model_factory = runtime(False)
    backend = GraniteBackend(
        GraniteSettings(device="cpu", precision="fp32"), runtime_loader=lambda: bundle
    )
    backend.load()
    backend.warmup()

    assert torch.zero_calls == [16_000]
    assert len(processor.calls) == 1
    assert len(model.generate_calls) == 1


def test_audio_contract_rejects_non_mono_or_non_16khz(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"fixture")
    bundle, *_ = runtime(False)
    bundle.torchaudio.waveform = FakeTensor((2, 3200))  # type: ignore[attr-defined]
    backend = GraniteBackend(
        GraniteSettings(device="cpu", precision="fp32"), runtime_loader=lambda: bundle
    )
    backend.load()
    with pytest.raises(ValueError, match="mono 16 kHz"):
        backend.transcribe(path.resolve())


def test_cuda_and_precision_policies_fail_before_weight_load() -> None:
    bundle, *_rest, model_factory = runtime(False)
    backend = GraniteBackend(
        GraniteSettings(device="cuda", precision="fp16"), runtime_loader=lambda: bundle
    )
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        backend.load()
    assert model_factory.calls == []
    with pytest.raises(ValueError, match="FP16 requires CUDA"):
        GraniteSettings(device="cpu", precision="fp16")
    with pytest.raises(ValueError, match="Unsupported Granite precision"):
        GraniteSettings(precision="bf16")  # type: ignore[arg-type]


def test_close_releases_cuda_resources() -> None:
    bundle, torch, *_ = runtime(True)
    backend = GraniteBackend(GraniteSettings(), runtime_loader=lambda: bundle)
    backend.load()
    backend.close()
    assert torch.cuda.empty_calls == 1


def test_descriptor_records_effective_policy() -> None:
    bundle, *_ = runtime(True)
    backend = GraniteBackend(
        GraniteSettings(preset_id="granite-speech-4.1-2b"), runtime_loader=lambda: bundle
    )
    backend.load()
    descriptor = backend.descriptor
    assert descriptor.effective_device == "cuda"
    assert descriptor.effective_precision == "fp16"
    assert descriptor.options == {"max_new_tokens": 200, "batch_size": 1}
    assert descriptor.audio_preprocessing == {
        "channels": 1,
        "sample_rate_hz": 16000,
        "normalize": True,
    }
