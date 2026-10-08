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


def test_lyrics_audio_download_uses_collection_ledger_and_output_directory() -> None:
    request = WorkflowRequest(
        Workflow.LYRICS_DOWNLOAD_AUDIO,
        "artifacts/lyrics/lyrics-collection.json",
        output_dir="data/audio/lyrics",
        force=True,
    )

    assert build_cli_args(request) == (
        "collect",
        "lyrics",
        "download-audio",
        "artifacts/lyrics/lyrics-collection.json",
        "--output-dir",
        "data/audio/lyrics",
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
        "--backend",
        "whisper",
        "--device",
        "auto",
        "--precision",
        "auto",
        "--verify-hashes",
    )


def test_stt_manifest_build_scans_audio_directory() -> None:
    request = WorkflowRequest(
        Workflow.STT_MANIFEST_BUILD,
        "data/audio/full",
        output_dir="data/manifests/stt-all.json",
    )

    assert build_cli_args(request) == (
        "stt",
        "manifest",
        "build",
        "data/audio/full",
        "--output-file",
        "data/manifests/stt-all.json",
    )


def test_stt_preflight_uses_candidate_preset() -> None:
    request = WorkflowRequest(
        Workflow.STT_PREFLIGHT,
        "stt.json",
        preset="granite-speech-4.1-2b",
        device="cuda",
        precision="fp16",
    )

    assert build_cli_args(request) == (
        "stt",
        "preflight",
        "stt.json",
        "--preset",
        "granite-speech-4.1-2b",
        "--device",
        "cuda",
        "--precision",
        "fp16",
    )


def test_stt_benchmark_candidate_preset_requires_worker_python() -> None:
    request = WorkflowRequest(
        Workflow.STT_BENCHMARK,
        "stt.json",
        output_dir="artifacts/run",
        preset="qwen3-asr-1.7b",
    )

    with pytest.raises(CommandValidationError, match="isolated worker Python"):
        build_cli_args(request)


def test_discovers_qwen_worker_from_common_underscore_environment_name(
    tmp_path: Path,
) -> None:
    worker = tmp_path / ".venvs" / "qwen3_asr" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)

    detected = discover_worker_python(runtime_id="qwen3-asr-transformers", cwd=tmp_path, environ={})

    assert detected == str(worker.resolve())


def test_discovers_worker_module_installed_in_generic_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / ".venv" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    monkeypatch.setattr(
        "dominican_eaters.tui.commands._python_provides_module",
        lambda python, module: python == worker and module == "dominican_eaters_qwen3_asr",
    )

    detected = discover_worker_python(runtime_id="qwen3-asr-transformers", cwd=tmp_path, environ={})

    assert detected == str(worker.resolve())


def test_stt_benchmark_candidate_preset_builds_worker_command(tmp_path: Path) -> None:
    worker = tmp_path / "python"
    request = WorkflowRequest(
        Workflow.STT_BENCHMARK,
        "stt.json",
        output_dir="artifacts/run",
        preset="qwen3-asr-1.7b",
        device="cuda",
        precision="fp16",
        worker_python=str(worker),
    )

    args = build_cli_args(request)

    assert args[:7] == (
        "stt",
        "benchmark",
        "stt.json",
        "--output-dir",
        "artifacts/run",
        "--preset",
        "qwen3-asr-1.7b",
    )
    assert args[-2:] == ("--worker-python", str(worker.resolve()))


def test_stt_benchmark_current_worker_preset_requires_python() -> None:
    request = WorkflowRequest(
        Workflow.STT_BENCHMARK,
        "stt.json",
        output_dir="artifacts/run",
        preset="canary-1b-v2",
    )

    with pytest.raises(CommandValidationError, match="worker Python"):
        build_cli_args(request)


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


def test_stt_preflight_preserves_worker_virtualenv_symlink(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    worker = tmp_path / "worker-python"
    worker.symlink_to(target)

    args = build_cli_args(
        WorkflowRequest(
            Workflow.STT_PREFLIGHT,
            "stt.json",
            backend="canary",
            worker_python=str(worker),
        )
    )

    assert args[-2:] == ("--worker-python", str(worker))


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


def test_discovers_project_runtime_environment_under_venvs(tmp_path: Path) -> None:
    worker = tmp_path / ".venvs" / "voxtral" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)

    assert discover_worker_python(
        runtime_id="voxtral-transformers", cwd=tmp_path, environ={}
    ) == str(worker.resolve())


def test_discovers_worker_python_from_runtime_specific_environment(tmp_path: Path) -> None:
    worker = tmp_path / "granite-runtime" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)

    assert discover_worker_python(
        runtime_id="granite-4.1-transformers",
        cwd=tmp_path,
        environ={"DOMINICAN_EATERS_GRANITE_PYTHON": str(worker)},
    ) == str(worker)


def test_inline_runtime_does_not_discover_worker_python(tmp_path: Path) -> None:
    worker = tmp_path / "configured" / "bin" / "python"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)

    assert (
        discover_worker_python(
            runtime_id="whisper-native",
            cwd=tmp_path,
            environ={"DOMINICAN_EATERS_WORKER_PYTHON": str(worker)},
        )
        == ""
    )


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
