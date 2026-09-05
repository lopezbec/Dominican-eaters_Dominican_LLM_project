from __future__ import annotations

from pathlib import Path

import pytest

from dominican_eaters.speech.asr.whisper import (
    WhisperBackend,
    WhisperDependencyError,
    WhisperRuntimeBundle,
    WhisperSettings,
    load_whisper_runtime,
)


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self.available = available
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.reset_calls = 0

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1

    def max_memory_allocated(self) -> int:
        return 123

    def max_memory_reserved(self) -> int:
        return 456


class FakeTorch:
    def __init__(self, available: bool) -> None:
        self.cuda = FakeCuda(available)


class FakeNumpy:
    def zeros(self, size: int, *, dtype: str) -> object:
        return ("zeros", size, dtype)


class FakeModel:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.result = result or {"text": "Hola", "language": "es"}
        self.calls: list[tuple[object, dict[str, object]]] = []

    def transcribe(self, audio: object, **options: object) -> dict[str, object]:
        self.calls.append((audio, options))
        return self.result


class FakeWhisper:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.loads: list[tuple[str, str]] = []
        self.audio_paths: list[str] = []

    def load_model(self, name: str, *, device: str) -> FakeModel:
        self.loads.append((name, device))
        return self.model

    def load_audio(self, path: str) -> object:
        self.audio_paths.append(path)
        return [0.0] * 16_000


def runtime(
    *, cuda: bool = False, result: dict[str, object] | None = None
) -> tuple[WhisperRuntimeBundle, FakeWhisper, FakeTorch, FakeModel]:
    model = FakeModel(result)
    whisper = FakeWhisper(model)
    torch = FakeTorch(cuda)
    bundle = WhisperRuntimeBundle(whisper=whisper, torch=torch, numpy=FakeNumpy())
    return bundle, whisper, torch, model


def test_cpu_lifecycle_uses_deterministic_decoding_options(tmp_path: Path) -> None:
    bundle, whisper, torch, model = runtime()
    backend = WhisperBackend(
        WhisperSettings(model="small", device="cpu", precision="fp32"),
        runtime_loader=lambda: bundle,
    )

    backend.load()
    backend.warmup()
    transcript = backend.transcribe(tmp_path / "sample.wav")
    backend.close()

    assert whisper.loads == [("small", "cpu")]
    assert model.calls[0][0] == ("zeros", 16_000, "float32")
    assert isinstance(model.calls[1][0], list)
    assert len(model.calls[1][0]) == 16_000
    assert all(call[1]["temperature"] == 0.0 for call in model.calls)
    assert all(call[1]["condition_on_previous_text"] is False for call in model.calls)
    assert all(call[1]["fp16"] is False for call in model.calls)
    assert transcript.text == "Hola"
    assert transcript.metadata["model"] == "small"
    assert transcript.metadata["device"] == "cpu"
    assert transcript.audio_duration_seconds == 1.0
    assert torch.cuda.empty_cache_calls == 0


def test_auto_selects_cuda_and_releases_cache() -> None:
    bundle, whisper, torch, model = runtime(cuda=True)
    backend = WhisperBackend(WhisperSettings(), runtime_loader=lambda: bundle)

    backend.load()
    backend.warmup()
    transcript = backend.transcribe(Path("sample.wav"))
    backend.close()

    descriptor = backend.descriptor
    assert whisper.loads == [("base", "cuda")]
    assert model.calls[0][1]["fp16"] is True
    assert torch.cuda.empty_cache_calls == 1
    assert torch.cuda.reset_calls == 1
    assert torch.cuda.synchronize_calls == 2
    assert transcript.gpu_peak_allocated_bytes == 123
    assert transcript.gpu_peak_reserved_bytes == 456
    assert descriptor.backend_id == "openai-whisper/base"
    assert descriptor.effective_device == "cuda"
    assert descriptor.effective_precision == "fp16"
    assert descriptor.runtime_versions["python"]


def test_explicit_cuda_never_silently_falls_back() -> None:
    bundle, whisper, _torch, _model = runtime(cuda=False)
    backend = WhisperBackend(WhisperSettings(device="cuda"), runtime_loader=lambda: bundle)

    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        backend.load()

    assert whisper.loads == []


def test_invalid_result_is_rejected() -> None:
    bundle, _whisper, _torch, _model = runtime(result={"language": "es"})
    backend = WhisperBackend(WhisperSettings(device="cpu"), runtime_loader=lambda: bundle)
    backend.load()

    with pytest.raises(TypeError, match="string text field"):
        backend.transcribe(Path("sample.wav"))


def test_settings_reject_cpu_fp16_before_runtime_loading() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        WhisperSettings(device="cpu", precision="fp16")


def test_missing_runtime_has_actionable_environment_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> object:
        raise ModuleNotFoundError("missing test dependency")

    monkeypatch.setattr("importlib.import_module", missing)

    with pytest.raises(WhisperDependencyError, match="environments/whisper/requirements.in"):
        load_whisper_runtime()
