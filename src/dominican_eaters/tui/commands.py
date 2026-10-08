"""Pure command construction for the terminal workflow launcher."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from dominican_eaters.speech.asr.registry import (
    CURRENT_BACKEND_SPECS,
    MODEL_PRESETS,
    RUNTIME_SPECS,
    BackendName,
    BackendSpec,
    ModelPreset,
)


class Workflow(StrEnum):
    CONFIG_VALIDATE = "config-validate"
    BOOKS_PREFLIGHT = "books-preflight"
    BOOKS_RUN = "books-run"
    LYRICS_PREFLIGHT = "lyrics-preflight"
    LYRICS_RUN = "lyrics-run"
    LYRICS_DOWNLOAD_AUDIO = "lyrics-download-audio"
    POEMS_PREFLIGHT = "poems-preflight"
    POEMS_RUN = "poems-run"
    STT_PREFLIGHT = "stt-preflight"
    STT_MANIFEST_BUILD = "stt-manifest-build"
    STT_BENCHMARK = "stt-benchmark"


WORKFLOW_OPTIONS: tuple[tuple[str, Workflow], ...] = (
    ("Validate configuration", Workflow.CONFIG_VALIDATE),
    ("Preflight books manifest", Workflow.BOOKS_PREFLIGHT),
    ("Collect books", Workflow.BOOKS_RUN),
    ("Preflight lyrics manifest", Workflow.LYRICS_PREFLIGHT),
    ("Collect lyrics", Workflow.LYRICS_RUN),
    ("Download collected lyrics audio", Workflow.LYRICS_DOWNLOAD_AUDIO),
    ("Preflight poems manifest", Workflow.POEMS_PREFLIGHT),
    ("Collect poems", Workflow.POEMS_RUN),
    ("Preflight speech-to-text manifest", Workflow.STT_PREFLIGHT),
    ("Build speech-to-text manifest", Workflow.STT_MANIFEST_BUILD),
    ("Run speech-to-text benchmark", Workflow.STT_BENCHMARK),
)

COLLECTION_RUNS = {Workflow.BOOKS_RUN, Workflow.LYRICS_RUN, Workflow.POEMS_RUN}
OUTPUT_WORKFLOWS = COLLECTION_RUNS | {Workflow.STT_MANIFEST_BUILD, Workflow.STT_BENCHMARK}
OUTPUT_WORKFLOWS.add(Workflow.LYRICS_DOWNLOAD_AUDIO)

DEFAULT_SOURCE_PATHS: dict[Workflow, str] = {
    Workflow.CONFIG_VALIDATE: "config/default.yaml",
    Workflow.BOOKS_PREFLIGHT: "data/manifests/books.json",
    Workflow.BOOKS_RUN: "data/manifests/books.json",
    Workflow.LYRICS_PREFLIGHT: "data/manifests/lyrics.json",
    Workflow.LYRICS_RUN: "data/manifests/lyrics.json",
    Workflow.LYRICS_DOWNLOAD_AUDIO: "artifacts/lyrics/lyrics-collection.json",
    Workflow.POEMS_PREFLIGHT: "data/manifests/poems.json",
    Workflow.POEMS_RUN: "data/manifests/poems.json",
    Workflow.STT_PREFLIGHT: "data/manifests/stt.json",
    Workflow.STT_MANIFEST_BUILD: "data/audio",
    Workflow.STT_BENCHMARK: "data/manifests/stt.json",
}

DEFAULT_OUTPUT_DIRS: dict[Workflow, str] = {
    Workflow.BOOKS_RUN: "artifacts/books",
    Workflow.LYRICS_RUN: "artifacts/lyrics",
    Workflow.LYRICS_DOWNLOAD_AUDIO: "data/audio/lyrics",
    Workflow.POEMS_RUN: "artifacts/poems",
    Workflow.STT_BENCHMARK: "artifacts/stt-run",
    Workflow.STT_MANIFEST_BUILD: "data/manifests/stt-all.json",
}

WORKER_PYTHON_ENV_VARS = (
    "DOMINICAN_EATERS_WORKER_PYTHON",
    "NEMO_WORKER_PYTHON",
)
RUNTIME_VENV_DIRS = {
    "nemo-worker": (".venvs/nemo", ".venv-nemo"),
    "granite-4.1-transformers": (".venvs/granite", ".venv-granite"),
    "granite-3.3-transformers": (".venvs/granite", ".venv-granite"),
    "qwen3-asr-transformers": (".venvs/qwen3-asr", ".venv-qwen3-asr"),
    "voxtral-transformers": (".venvs/voxtral", ".venv-voxtral"),
    "qwen2-audio-transformers": (".venvs/qwen2-audio", ".venv-qwen2-audio"),
}


def _runtime_environment_hints(runtime_id: str) -> tuple[str, ...]:
    base = runtime_id.removesuffix("-transformers").removesuffix("-worker")
    values = {base, base.replace("-asr", ""), base.replace("-", "_")}
    return tuple(value for value in values if value)


def _python_provides_module(python: Path, module: str) -> bool:
    probe = f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    try:
        return (
            subprocess.run(
                [python, "-c", probe],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def discover_worker_python(
    *,
    runtime_id: str = "nemo-worker",
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Find the interpreter configured for one isolated runtime."""

    environment = os.environ if environ is None else environ
    root = Path.cwd() if cwd is None else cwd
    try:
        runtime = RUNTIME_SPECS[runtime_id]
    except KeyError as error:
        raise ValueError(f"Unknown ASR runtime: {runtime_id}") from error
    if runtime.execution == "inline":
        return ""
    runtime_variables = (
        (runtime.interpreter_env_var,) if runtime.interpreter_env_var is not None else ()
    )
    candidates = [
        Path(environment[name]).expanduser()
        for name in (*runtime_variables, *WORKER_PYTHON_ENV_VARS)
        if environment.get(name)
    ]
    environment_dirs = RUNTIME_VENV_DIRS.get(runtime_id, ())
    candidates.extend(root / directory / "bin" / "python" for directory in environment_dirs)

    hints = _runtime_environment_hints(runtime_id)
    discovered = (*root.glob(".venv*/bin/python"), *root.glob(".venvs/*/bin/python"))
    candidates.extend(
        candidate
        for candidate in discovered
        if any(hint in str(candidate.parent.parent).lower() for hint in hints)
    )

    active_environment = environment.get("VIRTUAL_ENV", "")
    runtime_hint = runtime_id.split("-")[0]
    if runtime_hint in Path(active_environment).name.lower():
        candidates.append(Path(active_environment) / "bin" / "python")

    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        if absolute.is_file() and os.access(absolute, os.X_OK):
            return str(absolute)

    if runtime.worker_module is not None:
        generic_candidates = (
            root / ".venv" / "bin" / "python",
            *root.glob(".venv*/bin/python"),
            *root.glob(".venvs/*/bin/python"),
        )
        for candidate in generic_candidates:
            absolute = Path(os.path.abspath(candidate))
            if (
                absolute.is_file()
                and os.access(absolute, os.X_OK)
                and _python_provides_module(absolute, runtime.worker_module)
            ):
                return str(absolute)
    return ""


class CommandValidationError(ValueError):
    """A required workflow field is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    workflow: Workflow
    source_path: str
    output_dir: str = ""
    data_root: str = ""
    artifacts_root: str = ""
    preset: str = ""
    backend: str = "whisper"
    model: str = ""
    device: str = "auto"
    precision: str = "auto"
    worker_python: str = ""
    warmup_runs: str = "1"
    request_timeout: str = "300"
    short_audio_policy: str = "reject"
    minimum_audio_seconds: str = "0.1"
    force: bool = False
    verify_hashes: bool = False
    timestamps: bool = False


def build_cli_args(request: WorkflowRequest) -> tuple[str, ...]:
    """Return validated arguments for ``python -m dominican_eaters``."""

    source_path = request.source_path.strip()
    if not source_path:
        name = "configuration" if request.workflow is Workflow.CONFIG_VALIDATE else "manifest"
        raise CommandValidationError(f"Select a {name} file.")

    if request.workflow is Workflow.CONFIG_VALIDATE:
        args = ["config", "validate", source_path]
        if request.data_root.strip():
            args.extend(("--data-root", request.data_root.strip()))
        if request.artifacts_root.strip():
            args.extend(("--artifacts-root", request.artifacts_root.strip()))
        return tuple(args)

    domain_commands = {
        Workflow.BOOKS_PREFLIGHT: ("collect", "books", "preflight"),
        Workflow.BOOKS_RUN: ("collect", "books", "run"),
        Workflow.LYRICS_PREFLIGHT: ("collect", "lyrics", "preflight"),
        Workflow.LYRICS_RUN: ("collect", "lyrics", "run"),
        Workflow.POEMS_PREFLIGHT: ("collect", "poems", "preflight"),
        Workflow.POEMS_RUN: ("collect", "poems", "run"),
    }
    if request.workflow in domain_commands:
        args = [*domain_commands[request.workflow], source_path]
        if request.workflow in COLLECTION_RUNS:
            args.extend(("--output-dir", _required_output_dir(request)))
            if request.force:
                args.append("--force")
        return tuple(args)

    if request.workflow is Workflow.LYRICS_DOWNLOAD_AUDIO:
        args = [
            "collect",
            "lyrics",
            "download-audio",
            source_path,
            "--output-dir",
            _required_output_dir(request),
        ]
        if request.force:
            args.append("--force")
        return tuple(args)

    if request.workflow is Workflow.STT_MANIFEST_BUILD:
        args = [
            "stt",
            "manifest",
            "build",
            source_path,
            "--output-file",
            _required_output_dir(request),
        ]
        return tuple(args)

    if request.workflow is Workflow.STT_PREFLIGHT:
        preset = _request_preset(request)
        if preset is None:
            _request_backend_spec(request)
        args = ["stt", "preflight", source_path]
        if request.data_root.strip():
            args.extend(("--dataset-root", request.data_root.strip()))
        if preset is not None:
            args.extend(("--preset", preset.preset_id))
        else:
            args.extend(("--backend", request.backend))
        args.extend(("--device", request.device, "--precision", request.precision))
        if preset is None and request.model.strip():
            args.extend(("--model", request.model.strip()))
        if request.worker_python.strip():
            executable = os.path.abspath(Path(request.worker_python.strip()).expanduser())
            args.extend(("--worker-python", executable))
        if request.verify_hashes:
            args.append("--verify-hashes")
        return tuple(args)

    if request.workflow is Workflow.STT_BENCHMARK:
        preset = _request_preset(request)
        backend_spec = _request_backend_spec(request) if preset is None else None
        if preset is not None and not preset.runnable:
            reason = preset.reason or "the adapter has not been promoted"
            raise CommandValidationError(
                f"Preset {preset.preset_id} is {preset.status} and cannot be benchmarked: {reason}"
            )
        if preset is None:
            assert backend_spec is not None
            runtime = backend_spec.runtime
        else:
            runtime = RUNTIME_SPECS[preset.runtime_id]
        if runtime.execution == "worker" and not request.worker_python.strip():
            raise CommandValidationError("Select the isolated worker Python executable.")
        warmup_runs = _nonnegative_integer(request.warmup_runs, "Warmup runs")
        request_timeout = _positive_number(request.request_timeout, "Request timeout")
        minimum_audio = _positive_number(request.minimum_audio_seconds, "Minimum audio duration")
        if request.short_audio_policy not in {"reject", "allow"}:
            raise CommandValidationError(
                f"Unsupported short-audio policy: {request.short_audio_policy}"
            )
        args = [
            "stt",
            "benchmark",
            source_path,
            "--output-dir",
            _required_output_dir(request),
        ]
        if preset is not None:
            args.extend(("--preset", preset.preset_id))
        else:
            args.extend(("--backend", request.backend))
        args.extend(
            [
                "--device",
                request.device,
                "--precision",
                request.precision,
                "--warmup-runs",
                warmup_runs,
                "--request-timeout",
                request_timeout,
                "--short-audio-policy",
                request.short_audio_policy,
                "--minimum-audio-seconds",
                minimum_audio,
            ]
        )
        if preset is None and request.model.strip():
            args.extend(("--model", request.model.strip()))
        if request.worker_python.strip():
            executable = os.path.abspath(Path(request.worker_python.strip()).expanduser())
            args.extend(("--worker-python", executable))
        if request.verify_hashes:
            args.append("--verify-hashes")
        if request.timestamps:
            args.append("--timestamps")
        return tuple(args)

    raise CommandValidationError(f"Unsupported workflow: {request.workflow}")


def _request_preset(request: WorkflowRequest) -> ModelPreset | None:
    preset_id = request.preset.strip()
    if not preset_id:
        return None
    try:
        return MODEL_PRESETS[preset_id]
    except KeyError as error:
        raise CommandValidationError(f"Unsupported preset: {preset_id}") from error


def _request_backend_spec(request: WorkflowRequest) -> BackendSpec:
    try:
        return CURRENT_BACKEND_SPECS[cast(BackendName, request.backend)]
    except KeyError as error:
        raise CommandValidationError(f"Unsupported backend: {request.backend}") from error


def _required_output_dir(request: WorkflowRequest) -> str:
    output_dir = request.output_dir.strip()
    if not output_dir:
        raise CommandValidationError("Select an output directory.")
    return output_dir


def _nonnegative_integer(value: str, label: str) -> str:
    try:
        parsed = int(value)
    except ValueError as error:
        raise CommandValidationError(f"{label} must be a nonnegative integer.") from error
    if parsed < 0 or str(parsed) != value.strip():
        raise CommandValidationError(f"{label} must be a nonnegative integer.")
    return str(parsed)


def _positive_number(value: str, label: str) -> str:
    try:
        parsed = float(value)
    except ValueError as error:
        raise CommandValidationError(f"{label} must be a positive number.") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise CommandValidationError(f"{label} must be a positive number.")
    return value.strip()
