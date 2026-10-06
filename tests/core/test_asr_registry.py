from __future__ import annotations

import pytest

from dominican_eaters.speech.asr.registry import (
    BACKEND_SPECS,
    CURRENT_BACKEND_SPECS,
    DEFAULT_MODELS,
    MODEL_PRESETS,
    RUNTIME_SPECS,
    BackendSpec,
    ModelCapabilities,
    ModelPreset,
    RuntimeSpec,
    runnable_presets,
)


def test_registry_contains_every_target_preset_with_exact_model_identity() -> None:
    assert {preset.preset_id: preset.model for preset in MODEL_PRESETS.values()} == {
        "whisper-base": "base",
        "whisper-large-v3": "large-v3",
        "whisper-turbo": "turbo",
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
        "canary-1b-v2": "nvidia/canary-1b-v2",
        "granite-speech-4.1-2b": "ibm-granite/granite-speech-4.1-2b",
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B-hf",
        "voxtral-mini-3b-2507": "mistralai/Voxtral-Mini-3B-2507",
        "qwen2-audio-7b-instruct": "Qwen/Qwen2-Audio-7B-Instruct",
        "granite-speech-3.3-8b": "ibm-granite/granite-speech-3.3-8b",
    }


def test_only_integrated_presets_and_backends_are_runnable() -> None:
    assert tuple(CURRENT_BACKEND_SPECS) == ("whisper", "parakeet", "canary")
    assert tuple(preset.preset_id for preset in runnable_presets()) == (
        "whisper-base",
        "parakeet-tdt-0.6b-v3",
        "canary-1b-v2",
    )
    assert DEFAULT_MODELS == {
        "whisper": "base",
        "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
        "canary": "nvidia/canary-1b-v2",
    }


def test_non_current_presets_have_an_honest_reason() -> None:
    non_current = [preset for preset in MODEL_PRESETS.values() if not preset.runnable]
    assert non_current
    assert all(preset.reason and preset.reason.strip() for preset in non_current)
    assert MODEL_PRESETS["qwen2-audio-7b-instruct"].status == "experimental"
    assert MODEL_PRESETS["granite-speech-3.3-8b"].status == "blocked"
    assert MODEL_PRESETS["granite-speech-3.3-8b"].precision is None
    assert MODEL_PRESETS["granite-speech-3.3-8b"].quantization is None


def test_capabilities_do_not_claim_unavailable_timestamps_or_streaming() -> None:
    whisper = MODEL_PRESETS["whisper-base"].capabilities
    qwen3 = MODEL_PRESETS["qwen3-asr-1.7b"].capabilities
    voxtral = MODEL_PRESETS["voxtral-mini-3b-2507"].capabilities

    assert not whisper.timestamps
    assert not qwen3.timestamps
    assert not qwen3.streaming
    assert qwen3.language_detection
    assert not voxtral.timestamps
    assert not voxtral.streaming
    assert voxtral.long_form_policy == "native-30-minutes"


def test_runtime_specs_capture_isolation_and_compatibility_fields() -> None:
    whisper = RUNTIME_SPECS["whisper-native"]
    nemo = RUNTIME_SPECS["nemo-worker"]
    granite = RUNTIME_SPECS["granite-4.1-transformers"]

    assert whisper.execution == "inline"
    assert whisper.worker_module is None
    assert whisper.required_executables == ("ffmpeg",)
    assert nemo.worker_module == "dominican_eaters_nemo"
    assert nemo.interpreter_env_var == "DOMINICAN_EATERS_NEMO_PYTHON"
    assert granite.interpreter_env_var == "DOMINICAN_EATERS_GRANITE_PYTHON"
    assert "transformers" in granite.required_modules


def test_backend_compatibility_properties_are_derived_from_runtime() -> None:
    parakeet = BACKEND_SPECS["parakeet"]

    assert parakeet.execution == "worker"
    assert parakeet.worker_module == "dominican_eaters_nemo"
    assert "nemo.collections.asr" in parakeet.required_modules
    assert parakeet.required_executables == ()


def test_registry_records_are_immutable() -> None:
    with pytest.raises(TypeError):
        MODEL_PRESETS["another"] = MODEL_PRESETS["whisper-base"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        MODEL_PRESETS["whisper-base"].status = "blocked"  # type: ignore[misc]


def test_specs_reject_incoherent_configuration() -> None:
    capabilities = ModelCapabilities(
        timestamps=False,
        streaming=False,
        language_detection=False,
        long_form_policy="unverified",
        supported_devices=("cuda",),
        supported_precisions=("fp16",),
    )
    with pytest.raises(ValueError, match="worker runtimes must define a worker module"):
        RuntimeSpec("bad", "Bad", "worker", None, "BAD_PYTHON", ("torch",))
    with pytest.raises(ValueError, match="must explain"):
        BackendSpec("granite", "Granite", "model", "runtime", "planned")
    with pytest.raises(ValueError, match="not supported"):
        ModelPreset(
            "bad",
            "Bad",
            "granite",
            "model",
            None,
            "granite-4.1-transformers",
            "bf16",
            None,
            "es",
            "planned",
            capabilities,
            "Not validated",
        )
