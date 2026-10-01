from __future__ import annotations

from pathlib import Path

import pytest

from dominican_eaters.collection.books import load_book_manifest
from dominican_eaters.collection.lyrics import load_lyrics_manifest
from dominican_eaters.collection.poems import load_poem_manifest
from dominican_eaters.data import load_manifest
from dominican_eaters.tui.commands import (
    DEFAULT_SOURCE_PATHS,
    CommandValidationError,
    Workflow,
    WorkflowRequest,
    build_cli_args,
    discover_worker_python,
)


def test_default_manifests_are_loadable_and_stt_audio_exists() -> None:
    project_root = Path(__file__).parents[2]

    load_book_manifest(project_root / DEFAULT_SOURCE_PATHS[Workflow.BOOKS_RUN])
    load_lyrics_manifest(project_root / DEFAULT_SOURCE_PATHS[Workflow.LYRICS_RUN])
    load_poem_manifest(project_root / DEFAULT_SOURCE_PATHS[Workflow.POEMS_RUN])
    stt_manifest = load_manifest(project_root / DEFAULT_SOURCE_PATHS[Workflow.STT_BENCHMARK])
    stt_manifest.preflight(verify_hashes=True)


@pytest.mark.parametrize(
    ("workflow", "expected"),
    [
        (Workflow.BOOKS_PREFLIGHT, ("collect", "books", "preflight", "input.json")),
        (Workflow.LYRICS_PREFLIGHT, ("collect", "lyrics", "preflight", "input.json")),
        (Workflow.POEMS_PREFLIGHT, ("collect", "poems", "preflight", "input.json")),
    ],
)
def test_collection_preflight_commands(workflow: Workflow, expected: tuple[str, ...]) -> None:
    assert build_cli_args(WorkflowRequest(workflow, "input.json")) == expected


def test_collection_run_includes_output_and_force() -> None:
    request = WorkflowRequest(
        Workflow.LYRICS_RUN,
        "lyrics.json",
        output_dir="artifacts/lyrics",
        force=True,
    )

    assert build_cli_args(request) == (
        "collect",
        "lyrics",
        "run",
        "lyrics.json",
        "--output-dir",
        "artifacts/lyrics",
        "--force",
    )


def test_config_validation_includes_root_overrides() -> None:
    request = WorkflowRequest(
        Workflow.CONFIG_VALIDATE,
        "config.yaml",
        data_root="data",
        artifacts_root="artifacts",
    )

    assert build_cli_args(request) == (
        "config",
        "validate",
        "config.yaml",
        "--data-root",
        "data",
        "--artifacts-root",
        "artifacts",
    )


def test_stt_preflight_includes_optional_controls() -> None:
    request = WorkflowRequest(
        Workflow.STT_PREFLIGHT,
        "stt.json",
        data_root="/datasets/stt",
        verify_hashes=True,
    )

    assert build_cli_args(request) == (
        "stt",
        "preflight",
        "stt.json",
        "--dataset-root",
        "/datasets/stt",
        "--verify-hashes",
    )


def test_whisper_benchmark_uses_selected_runtime_options() -> None:
    request = WorkflowRequest(
        Workflow.STT_BENCHMARK,
        "stt.json",
        output_dir="artifacts/run",
        model="large-v3",
        device="cuda",
        precision="fp16",
        verify_hashes=True,
        timestamps=True,
    )

    assert build_cli_args(request) == (
        "stt",
        "benchmark",
        "stt.json",
        "--output-dir",
        "artifacts/run",
        "--backend",
        "whisper",
        "--device",
        "cuda",
        "--precision",
        "fp16",
        "--warmup-runs",
        "1",
        "--request-timeout",
        "300",
        "--short-audio-policy",
        "reject",
        "--minimum-audio-seconds",
        "0.1",
        "--model",
        "large-v3",
        "--verify-hashes",
        "--timestamps",
    )


def test_nemo_benchmark_requires_and_normalizes_worker_python(tmp_path: Path) -> None:
    request = WorkflowRequest(
        Workflow.STT_BENCHMARK,
        "stt.json",
        output_dir="artifacts/run",
        backend="parakeet",
    )
    with pytest.raises(CommandValidationError, match="worker Python"):
        build_cli_args(request)

    worker = tmp_path / "worker" / "bin" / "python"
    args = build_cli_args(
        WorkflowRequest(
            Workflow.STT_BENCHMARK,
            "stt.json",
            output_dir="artifacts/run",
            backend="parakeet",
            worker_python=str(worker),
        )
    )
    assert args[-2:] == ("--worker-python", str(worker.resolve()))


def test_discovers_worker_python_from_environment(tmp_path: Path) -> None:
    worker = tmp_path / "configured-nemo" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)

    assert discover_worker_python(
        cwd=tmp_path,
        environ={"DOMINICAN_EATERS_WORKER_PYTHON": str(worker)},
    ) == str(worker.resolve())


def test_discovers_project_nemo_environment(tmp_path: Path) -> None:
    worker = tmp_path / ".venv-nemo" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)

    assert discover_worker_python(cwd=tmp_path, environ={}) == str(worker.resolve())


@pytest.mark.parametrize(
    "workflow_request, message",
    [
        (WorkflowRequest(Workflow.CONFIG_VALIDATE, ""), "configuration"),
        (WorkflowRequest(Workflow.BOOKS_RUN, "books.json"), "output directory"),
        (
            WorkflowRequest(
                Workflow.STT_BENCHMARK,
                "stt.json",
                output_dir="output",
                backend="unsupported",
            ),
            "Unsupported backend",
        ),
        (
            WorkflowRequest(
                Workflow.STT_BENCHMARK,
                "stt.json",
                output_dir="output",
                warmup_runs="-1",
            ),
            "Warmup runs",
        ),
    ],
)
def test_invalid_forms_are_rejected(workflow_request: WorkflowRequest, message: str) -> None:
    with pytest.raises(CommandValidationError, match=message):
        build_cli_args(workflow_request)
