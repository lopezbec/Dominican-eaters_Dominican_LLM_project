from __future__ import annotations

import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dominican_eaters.speech.asr.environment import preflight_asr_environment


def test_worker_preflight_uses_exact_virtualenv_symlink(
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
            '{"dominican_eaters_nemo":true,"nemo.collections.asr":true,"torch":true},'
            '"cuda_available":true}',
            "",
        )

    monkeypatch.setattr("dominican_eaters.speech.asr.environment.subprocess.run", run)
    report = preflight_asr_environment(
        backend="parakeet",
        worker_python=virtualenv_python,
        requested_device="cuda",
    )

    assert report.ready
    assert report.interpreter == virtualenv_python
    assert calls[0][0] == str(virtualenv_python)
    assert [(check.kind, check.name) for check in report.checks] == [
        ("interpreter", "python"),
        ("module", "dominican_eaters_nemo"),
        ("module", "nemo.collections.asr"),
        ("module", "torch"),
        ("cuda", "torch.cuda"),
    ]


def test_inline_preflight_uses_current_python_and_checks_executables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dominican_eaters.speech.asr.environment.shutil.which", lambda _name: None)
    calls: list[list[str]] = []

    def run(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
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
    assert calls[0][0] == sys.executable
    failures = {(check.kind, check.name) for check in report.checks if not check.available}
    assert failures == {("executable", "ffmpeg"), ("module", "whisper")}


def test_planned_backend_reports_missing_dedicated_interpreter() -> None:
    report = preflight_asr_environment(
        backend="granite", worker_python=None, requested_device="cuda"
    )

    assert not report.ready
    assert report.interpreter is None
    assert report.checks[0].kind == "interpreter"
    assert "--worker-python is required" in report.checks[0].detail
    assert "DOMINICAN_EATERS_GRANITE_PYTHON" in report.checks[0].detail


def test_probe_failure_is_structured_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "worker-python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)

    def run(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(argv, 2, "", "runtime import failed")

    monkeypatch.setattr("dominican_eaters.speech.asr.environment.subprocess.run", run)
    report = preflight_asr_environment(
        backend="qwen3_asr", worker_python=interpreter, requested_device="cuda"
    )

    assert not report.ready
    assert report.checks[-1].kind == "probe"
    assert report.checks[-1].detail == "runtime import failed"


def test_preflight_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        preflight_asr_environment(
            backend="whisper",
            worker_python=None,
            requested_device="cpu",
            timeout_seconds=0,
        )
