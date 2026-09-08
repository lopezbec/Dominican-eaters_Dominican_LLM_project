"""Pure command construction for the terminal workflow launcher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Workflow(StrEnum):
    CONFIG_VALIDATE = "config-validate"
    BOOKS_PREFLIGHT = "books-preflight"
    BOOKS_RUN = "books-run"
    LYRICS_PREFLIGHT = "lyrics-preflight"
    LYRICS_RUN = "lyrics-run"
    POEMS_PREFLIGHT = "poems-preflight"
    POEMS_RUN = "poems-run"
    STT_PREFLIGHT = "stt-preflight"
    STT_BENCHMARK = "stt-benchmark"


WORKFLOW_OPTIONS: tuple[tuple[str, Workflow], ...] = (
    ("Validate configuration", Workflow.CONFIG_VALIDATE),
    ("Preflight books manifest", Workflow.BOOKS_PREFLIGHT),
    ("Collect books", Workflow.BOOKS_RUN),
    ("Preflight lyrics manifest", Workflow.LYRICS_PREFLIGHT),
    ("Collect lyrics", Workflow.LYRICS_RUN),
    ("Preflight poems manifest", Workflow.POEMS_PREFLIGHT),
    ("Collect poems", Workflow.POEMS_RUN),
    ("Preflight speech-to-text manifest", Workflow.STT_PREFLIGHT),
    ("Run speech-to-text benchmark", Workflow.STT_BENCHMARK),
)

COLLECTION_RUNS = {Workflow.BOOKS_RUN, Workflow.LYRICS_RUN, Workflow.POEMS_RUN}
OUTPUT_WORKFLOWS = COLLECTION_RUNS | {Workflow.STT_BENCHMARK}


class CommandValidationError(ValueError):
    """A required workflow field is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    workflow: Workflow
    source_path: str
    output_dir: str = ""
    data_root: str = ""
    artifacts_root: str = ""
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

    if request.workflow is Workflow.STT_PREFLIGHT:
        args = ["stt", "preflight", source_path]
        if request.data_root.strip():
            args.extend(("--dataset-root", request.data_root.strip()))
        if request.verify_hashes:
            args.append("--verify-hashes")
        return tuple(args)

    if request.workflow is Workflow.STT_BENCHMARK:
        if request.backend not in {"whisper", "parakeet", "canary"}:
            raise CommandValidationError(f"Unsupported backend: {request.backend}")
        if request.backend in {"parakeet", "canary"} and not request.worker_python.strip():
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
            "--backend",
            request.backend,
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
        if request.model.strip():
            args.extend(("--model", request.model.strip()))
        if request.worker_python.strip():
            executable = str(Path(request.worker_python.strip()).expanduser().resolve())
            args.extend(("--worker-python", executable))
        if request.verify_hashes:
            args.append("--verify-hashes")
        if request.timestamps:
            args.append("--timestamps")
        return tuple(args)

    raise CommandValidationError(f"Unsupported workflow: {request.workflow}")


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
