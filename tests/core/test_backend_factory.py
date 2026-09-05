from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dominican_eaters.cli.backends import create_asr_backend
from dominican_eaters.speech.asr import JsonlSubprocessBackend, WhisperBackend


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
