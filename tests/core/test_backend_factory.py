from __future__ import annotations

import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dominican_eaters.cli.backends import create_asr_backend
from dominican_eaters.speech.asr import JsonlSubprocessBackend, WhisperBackend
from dominican_eaters.speech.asr.environment import preflight_asr_environment
from dominican_eaters.speech.asr.registry import (
    CURRENT_BACKEND_SPECS,
    DEFAULT_MODELS,
    MODEL_PRESETS,
)


def test_backend_registry_contains_only_currently_runnable_backends() -> None:
    assert tuple(CURRENT_BACKEND_SPECS) == ("whisper", "parakeet", "canary")
    assert CURRENT_BACKEND_SPECS["whisper"].label == "Whisper"
    assert CURRENT_BACKEND_SPECS["whisper"].execution == "inline"
    assert CURRENT_BACKEND_SPECS["whisper"].worker_module is None
    assert CURRENT_BACKEND_SPECS["whisper"].required_modules == ("whisper", "torch")
    assert CURRENT_BACKEND_SPECS["whisper"].required_executables == ("ffmpeg",)
    assert CURRENT_BACKEND_SPECS["parakeet"].worker_module == "dominican_eaters_nemo"
    assert "nemo.collections.asr" in CURRENT_BACKEND_SPECS["parakeet"].required_modules
    assert CURRENT_BACKEND_SPECS["canary"].worker_module == "dominican_eaters_nemo"
    assert DEFAULT_MODELS == {
        "whisper": "base",
        "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
        "canary": "nvidia/canary-1b-v2",
    }


def test_worker_environment_preflight_uses_exact_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "worker-python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
        return CompletedProcess(
            argv,
            0,
            '{"python_version":"3.12.8","modules":'
            '{"dominican_eaters_nemo":true,"nemo.collections.asr":true,"torch":true},'
            '"cuda_available":true}',
            "",
        )

    monkeypatch.setattr("dominican_eaters.speech.asr.environment.subprocess.run", run)
    report = preflight_asr_environment(
        backend="parakeet",
        worker_python=interpreter,
        requested_device="cuda",
    )

    assert report.ready
    assert report.python_version == "3.12.8"
    assert calls[0][0] == str(interpreter.resolve())
    assert [(check.kind, check.name) for check in report.checks] == [
        ("interpreter", "python"),
        ("module", "dominican_eaters_nemo"),
        ("module", "nemo.collections.asr"),
        ("module", "torch"),
        ("cuda", "torch.cuda"),
    ]


def test_environment_preflight_preserves_virtualenv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "base-python"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    virtualenv_python = tmp_path / "venv-python"
    virtualenv_python.symlink_to(target)
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
        return CompletedProcess(
            argv,
            0,
            '{"python_version":"3.12.8","modules":'
            '{"dominican_eaters_nemo":true,"nemo.collections.asr":true,"torch":true}}',
            "",
        )

    monkeypatch.setattr("dominican_eaters.speech.asr.environment.subprocess.run", run)
    report = preflight_asr_environment(
        backend="parakeet",
        worker_python=virtualenv_python,
        requested_device="cpu",
    )

    assert report.ready
    assert report.interpreter == virtualenv_python
    assert calls[0][0] == str(virtualenv_python)


def test_environment_preflight_reports_missing_worker_interpreter() -> None:
    report = preflight_asr_environment(
        backend="canary", worker_python=None, requested_device="auto"
    )

    assert not report.ready
    assert report.interpreter is None
    assert report.checks[0].kind == "interpreter"
    assert "--worker-python is required" in report.checks[0].detail


def test_whisper_environment_preflight_reports_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dominican_eaters.speech.asr.environment.shutil.which", lambda _name: None)

    def run(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(
            argv,
            0,
            '{"python_version":"3.11.9","modules":{"whisper":false,"torch":true}}',
            "",
        )

    monkeypatch.setattr("dominican_eaters.speech.asr.environment.subprocess.run", run)
    report = preflight_asr_environment(
        backend="whisper", worker_python=None, requested_device="cpu"
    )

    assert not report.ready
    failures = {(check.kind, check.name) for check in report.checks if not check.available}
    assert failures == {("executable", "ffmpeg"), ("module", "whisper")}


def test_whisper_factory_uses_backend_specific_default_without_loading() -> None:
    backend = create_asr_backend(
        backend="whisper",
        model=None,
        language="es",
        device="cpu",
        precision="fp32",
        worker_python=None,
        request_timeout_seconds=10,
        timestamps=False,
    )

    assert isinstance(backend, WhisperBackend)
    assert backend.backend_id == "openai-whisper/base"
    assert backend.descriptor.effective_device is None


def test_factory_attaches_explicit_preset_identity_without_loading() -> None:
    preset = MODEL_PRESETS["whisper-turbo"]
    backend = create_asr_backend(
        backend="whisper",
        model=preset.model,
        language=preset.language,
        device="cuda",
        precision="fp16",
        worker_python=None,
        request_timeout_seconds=10,
        timestamps=False,
        preset=preset,
    )

    descriptor = backend.descriptor
    assert descriptor.preset_id == "whisper-turbo"
    assert descriptor.runtime_id == "whisper-native"
    assert descriptor.model_revision_requested == preset.model_revision
    assert descriptor.quantization == preset.quantization


@pytest.mark.parametrize(
    ("backend_name", "model"),
    [
        ("parakeet", "nvidia/parakeet-tdt-0.6b-v3"),
        ("canary", "nvidia/canary-1b-v2"),
    ],
)
def test_nemo_factories_use_isolated_worker_and_model_defaults(
    backend_name: str, model: str
) -> None:
    backend = create_asr_backend(
        backend=backend_name,  # type: ignore[arg-type]
        model=None,
        language="es",
        device="auto",
        precision="auto",
        worker_python=Path(sys.executable),
        request_timeout_seconds=12.5,
        timestamps=True,
    )

    assert isinstance(backend, JsonlSubprocessBackend)
    assert backend.descriptor.model == model
    assert backend.descriptor.effective_device is None


def test_nemo_factory_requires_worker_interpreter() -> None:
    with pytest.raises(ValueError, match="--worker-python is required"):
        create_asr_backend(
            backend="parakeet",
            model=None,
            language="es",
            device="auto",
            precision="auto",
            worker_python=None,
            request_timeout_seconds=10,
            timestamps=False,
        )


def test_whisper_rejects_bf16_at_configuration_boundary() -> None:
    with pytest.raises(ValueError, match="does not support"):
        create_asr_backend(
            backend="whisper",
            model=None,
            language="es",
            device="cuda",
            precision="bf16",
            worker_python=None,
            request_timeout_seconds=10,
            timestamps=False,
        )
