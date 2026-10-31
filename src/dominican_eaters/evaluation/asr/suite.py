"""Durable orchestration for a matrix of STT datasets and model presets."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

from dominican_eaters.data import atomic_write_json, exclusive_file_lock, load_manifest

SUITE_SCHEMA_VERSION: Final = 1
SUITE_EVENT_SCHEMA_VERSION: Final = 1
SUITE_LEDGER_FILENAME: Final = "suite.json"
DEFAULT_T4_PRESETS: Final = (
    "whisper-large-v3",
    "parakeet-tdt-0.6b-v3",
    "canary-1b-v2",
    "granite-speech-4.1-2b",
    "qwen3-asr-1.7b",
    "voxtral-mini-3b-2507",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class SuiteConflictError(ValueError):
    """Raised when an existing suite belongs to a different immutable plan."""


class SuiteStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class SuiteTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class SuiteEventKind(StrEnum):
    SUITE_STARTED = "suite_started"
    TASK_STARTED = "task_started"
    TASK_SKIPPED = "task_skipped"
    TASK_COMPLETED = "task_completed"
    SUITE_FINISHED = "suite_finished"
    SUITE_INTERRUPTED = "suite_interrupted"


@dataclass(frozen=True, slots=True)
class SuiteDataset:
    dataset_id: str
    manifest_path: str
    manifest_sha256: str
    samples: int

    def __post_init__(self) -> None:
        _require_identifier(self.dataset_id, "dataset ID")
        if not self.manifest_path:
            raise ValueError("manifest_path must not be empty")
        if len(self.manifest_sha256) != 64:
            raise ValueError("manifest_sha256 must contain 64 hexadecimal characters")
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError("dataset samples must be a positive integer")

    @classmethod
    def from_manifest(cls, dataset_id: str, manifest_path: Path) -> SuiteDataset:
        """Resolve and fingerprint one canonical manifest without loading any model."""

        resolved = manifest_path.expanduser().resolve()
        manifest = load_manifest(resolved)
        return cls(dataset_id, str(resolved), _file_sha256(resolved), len(manifest))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SuiteDataset:
        return cls(
            dataset_id=_string(value, "dataset_id"),
            manifest_path=_string(value, "manifest_path"),
            manifest_sha256=_string(value, "manifest_sha256"),
            samples=_integer(value, "samples"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "samples": self.samples,
        }


@dataclass(frozen=True, slots=True)
class SuitePlan:
    datasets: tuple[SuiteDataset, ...]
    presets: tuple[str, ...]
    device: str = "cuda"
    precision: str = "fp16"
    warmup_runs: int = 1
    request_timeout_seconds: float = 1800.0
    short_audio_policy: str = "reject"
    minimum_audio_seconds: float = 0.1
    verify_hashes: bool = True
    timestamps: bool = False
    schema_version: int = SUITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SUITE_SCHEMA_VERSION:
            raise ValueError(f"suite schema_version must be {SUITE_SCHEMA_VERSION}")
        object.__setattr__(self, "datasets", tuple(self.datasets))
        object.__setattr__(self, "presets", tuple(self.presets))
        if not self.datasets:
            raise ValueError("suite datasets must not be empty")
        if not self.presets:
            raise ValueError("suite presets must not be empty")
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("suite dataset IDs must be unique")
        if len(set(self.presets)) != len(self.presets):
            raise ValueError("suite presets must be unique")
        for preset in self.presets:
            _require_identifier(preset, "preset ID")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"unsupported suite device: {self.device}")
        if self.precision not in {"auto", "fp16", "fp32", "bf16"}:
            raise ValueError(f"unsupported suite precision: {self.precision}")
        if type(self.warmup_runs) is not int or self.warmup_runs < 0:
            raise ValueError("warmup_runs must be a nonnegative integer")
        if self.request_timeout_seconds <= 0 or self.minimum_audio_seconds <= 0:
            raise ValueError("suite timeouts and minimum audio duration must be positive")
        if self.short_audio_policy not in {"reject", "allow"}:
            raise ValueError(f"unsupported short audio policy: {self.short_audio_policy}")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SuitePlan:
        datasets = value.get("datasets")
        presets = value.get("presets")
        if not isinstance(datasets, list) or not all(
            isinstance(item, Mapping) for item in datasets
        ):
            raise ValueError("suite datasets must be an array of objects")
        if not isinstance(presets, list) or not all(isinstance(item, str) for item in presets):
            raise ValueError("suite presets must be an array of strings")
        return cls(
            datasets=tuple(SuiteDataset.from_dict(item) for item in datasets),
            presets=tuple(presets),
            device=_string(value, "device"),
            precision=_string(value, "precision"),
            warmup_runs=_integer(value, "warmup_runs"),
            request_timeout_seconds=_number(value, "request_timeout_seconds"),
            short_audio_policy=_string(value, "short_audio_policy"),
            minimum_audio_seconds=_number(value, "minimum_audio_seconds"),
            verify_hashes=_boolean(value, "verify_hashes"),
            timestamps=_boolean(value, "timestamps"),
            schema_version=_integer(value, "schema_version"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "presets": list(self.presets),
            "device": self.device,
            "precision": self.precision,
            "warmup_runs": self.warmup_runs,
            "request_timeout_seconds": self.request_timeout_seconds,
            "short_audio_policy": self.short_audio_policy,
            "minimum_audio_seconds": self.minimum_audio_seconds,
            "verify_hashes": self.verify_hashes,
            "timestamps": self.timestamps,
        }


@dataclass(frozen=True, slots=True)
class SuiteTaskOutcome:
    status: SuiteTaskStatus
    expected_samples: int
    succeeded_samples: int
    failed_samples: int

    def __post_init__(self) -> None:
        for name in ("expected_samples", "succeeded_samples", "failed_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SuiteTask:
    dataset_id: str
    preset_id: str
    status: SuiteTaskStatus = SuiteTaskStatus.PENDING
    attempt: int = 0
    output_dir: str | None = None
    expected_samples: int = 0
    succeeded_samples: int = 0
    failed_samples: int = 0
    error_type: str | None = None
    error_message: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset_id, self.preset_id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SuiteTask:
        return cls(
            dataset_id=_string(value, "dataset_id"),
            preset_id=_string(value, "preset_id"),
            status=SuiteTaskStatus(_string(value, "status")),
            attempt=_integer(value, "attempt"),
            output_dir=_optional_string(value.get("output_dir")),
            expected_samples=_integer(value, "expected_samples"),
            succeeded_samples=_integer(value, "succeeded_samples"),
            failed_samples=_integer(value, "failed_samples"),
            error_type=_optional_string(value.get("error_type")),
            error_message=_optional_string(value.get("error_message")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "preset_id": self.preset_id,
            "status": self.status.value,
            "attempt": self.attempt,
            "output_dir": self.output_dir,
            "expected_samples": self.expected_samples,
            "succeeded_samples": self.succeeded_samples,
            "failed_samples": self.failed_samples,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class SuiteLedger:
    suite_id: str
    plan: SuitePlan
    plan_fingerprint: str
    status: SuiteStatus
    tasks: tuple[SuiteTask, ...]
    created_at: str
    updated_at: str
    schema_version: int = SUITE_SCHEMA_VERSION

    @property
    def by_key(self) -> dict[tuple[str, str], SuiteTask]:
        return {task.key: task for task in self.tasks}

    def with_task(self, task: SuiteTask) -> SuiteLedger:
        values = self.by_key
        values[task.key] = task
        ordered = tuple(values[current.key] for current in self.tasks)
        return replace(self, tasks=ordered, updated_at=_now())

    @classmethod
    def initialize(cls, plan: SuitePlan) -> SuiteLedger:
        now = _now()
        tasks = tuple(
            SuiteTask(dataset.dataset_id, preset)
            for dataset in plan.datasets
            for preset in plan.presets
        )
        return cls(
            suite_id=uuid.uuid4().hex,
            plan=plan,
            plan_fingerprint=plan.fingerprint,
            status=SuiteStatus.PENDING,
            tasks=tasks,
            created_at=now,
            updated_at=now,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SuiteLedger:
        plan_value = value.get("plan")
        tasks = value.get("tasks")
        if not isinstance(plan_value, Mapping):
            raise ValueError("suite plan must be an object")
        if not isinstance(tasks, list) or not all(isinstance(item, Mapping) for item in tasks):
            raise ValueError("suite tasks must be an array of objects")
        ledger = cls(
            suite_id=_string(value, "suite_id"),
            plan=SuitePlan.from_dict(plan_value),
            plan_fingerprint=_string(value, "plan_fingerprint"),
            status=SuiteStatus(_string(value, "status")),
            tasks=tuple(SuiteTask.from_dict(item) for item in tasks),
            created_at=_string(value, "created_at"),
            updated_at=_string(value, "updated_at"),
            schema_version=_integer(value, "schema_version"),
        )
        if ledger.schema_version != SUITE_SCHEMA_VERSION:
            raise ValueError(f"unsupported suite schema_version: {ledger.schema_version}")
        if ledger.plan_fingerprint != ledger.plan.fingerprint:
            raise ValueError("suite plan fingerprint does not match its contents")
        return ledger

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "plan": self.plan.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "status": self.status.value,
            "tasks": [task.to_dict() for task in self.tasks],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class SuiteEvent:
    kind: SuiteEventKind
    suite_id: str
    suite_status: SuiteStatus
    task_index: int
    task_total: int
    dataset_id: str | None = None
    preset_id: str | None = None
    task_status: SuiteTaskStatus | None = None
    attempt: int | None = None
    output_dir: str | None = None
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    total: int = 0
    error_type: str | None = None
    message: str | None = None
    schema_version: int = SUITE_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event": self.kind.value,
            "suite_id": self.suite_id,
            "suite_status": self.suite_status.value,
            "task_index": self.task_index,
            "task_total": self.task_total,
            "dataset_id": self.dataset_id,
            "preset": self.preset_id,
            "task_status": self.task_status.value if self.task_status is not None else None,
            "attempt": self.attempt,
            "output_dir": self.output_dir,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total": self.total,
            "error_type": self.error_type,
            "message": self.message,
        }


SuiteEventCallback = Callable[[SuiteEvent], None]
SuiteTaskExecutor = Callable[[SuiteDataset, str, Path], SuiteTaskOutcome]


class SuiteRunner:
    """Execute and persist a dataset-by-preset matrix serially on one accelerator."""

    def __init__(
        self,
        executor: SuiteTaskExecutor,
        *,
        on_event: SuiteEventCallback | None = None,
    ) -> None:
        self._executor = executor
        self._on_event = on_event

    def run(self, plan: SuitePlan, output_dir: Path, *, resume: bool = True) -> SuiteLedger:
        root = output_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        ledger_path = root / SUITE_LEDGER_FILENAME
        with exclusive_file_lock(ledger_path, label="STT suite ledger"):
            ledger = self._load_or_initialize(plan, ledger_path, resume=resume)
            ledger = replace(ledger, status=SuiteStatus.RUNNING, updated_at=_now())
            self._save(ledger_path, ledger)
            self._emit(ledger, SuiteEventKind.SUITE_STARTED, 0)
            total = len(ledger.tasks)
            for task_index, task in enumerate(ledger.tasks, start=1):
                if task.status is SuiteTaskStatus.COMPLETE and _task_result_exists(task):
                    self._emit(
                        ledger,
                        SuiteEventKind.TASK_SKIPPED,
                        task_index,
                        task=task,
                        message="complete",
                    )
                    continue
                resumable_output = _resumable_task_output(task)
                if resumable_output is None:
                    attempt = task.attempt + 1
                    attempt_dir = (
                        root
                        / "datasets"
                        / task.dataset_id
                        / task.preset_id
                        / f"attempt-{attempt:03d}"
                    )
                else:
                    attempt = task.attempt
                    attempt_dir = resumable_output
                running = replace(
                    task,
                    status=SuiteTaskStatus.RUNNING,
                    attempt=attempt,
                    output_dir=str(attempt_dir),
                    expected_samples=0,
                    succeeded_samples=0,
                    failed_samples=0,
                    error_type=None,
                    error_message=None,
                )
                ledger = ledger.with_task(running)
                self._save(ledger_path, ledger)
                self._emit(ledger, SuiteEventKind.TASK_STARTED, task_index, task=running)
                dataset = next(
                    item for item in plan.datasets if item.dataset_id == running.dataset_id
                )
                try:
                    outcome = self._executor(dataset, running.preset_id, attempt_dir)
                except (KeyboardInterrupt, SystemExit) as error:
                    interrupted = replace(
                        running,
                        status=SuiteTaskStatus.INTERRUPTED,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    ledger = ledger.with_task(interrupted)
                    ledger = replace(ledger, status=SuiteStatus.INTERRUPTED, updated_at=_now())
                    self._save(ledger_path, ledger)
                    self._emit(
                        ledger,
                        SuiteEventKind.SUITE_INTERRUPTED,
                        task_index,
                        task=interrupted,
                        message=str(error),
                    )
                    raise
                except Exception as error:
                    finished = replace(
                        running,
                        status=SuiteTaskStatus.FAILED,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                else:
                    finished = replace(
                        running,
                        status=outcome.status,
                        expected_samples=outcome.expected_samples,
                        succeeded_samples=outcome.succeeded_samples,
                        failed_samples=outcome.failed_samples,
                    )
                ledger = ledger.with_task(finished)
                self._save(ledger_path, ledger)
                self._emit(
                    ledger,
                    SuiteEventKind.TASK_COMPLETED,
                    task_index,
                    task=finished,
                    message=finished.error_message,
                )
            final_status = _suite_status(ledger.tasks)
            ledger = replace(ledger, status=final_status, updated_at=_now())
            self._save(ledger_path, ledger)
            self._emit(ledger, SuiteEventKind.SUITE_FINISHED, total)
            return ledger

    @staticmethod
    def _load_or_initialize(
        plan: SuitePlan,
        ledger_path: Path,
        *,
        resume: bool,
    ) -> SuiteLedger:
        if ledger_path.exists():
            if not resume:
                raise SuiteConflictError(f"STT suite output already exists: {ledger_path.parent}")
            try:
                raw = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"could not read STT suite ledger {ledger_path}: {error}"
                ) from error
            if not isinstance(raw, Mapping):
                raise ValueError("STT suite ledger must be an object")
            ledger = SuiteLedger.from_dict(raw)
            if ledger.plan_fingerprint != plan.fingerprint:
                raise SuiteConflictError("existing STT suite uses a different dataset/model plan")
            return ledger
        lock_name = f"{ledger_path.name}.lock"
        unexpected = [path for path in ledger_path.parent.iterdir() if path.name != lock_name]
        if unexpected:
            raise SuiteConflictError(
                f"STT suite output is nonempty and has no ledger: {ledger_path.parent}"
            )
        ledger = SuiteLedger.initialize(plan)
        SuiteRunner._save(ledger_path, ledger)
        return ledger

    @staticmethod
    def _save(path: Path, ledger: SuiteLedger) -> None:
        atomic_write_json(path, ledger.to_dict())

    def _emit(
        self,
        ledger: SuiteLedger,
        kind: SuiteEventKind,
        task_index: int,
        *,
        task: SuiteTask | None = None,
        message: str | None = None,
    ) -> None:
        if self._on_event is None:
            return
        self._on_event(
            SuiteEvent(
                kind=kind,
                suite_id=ledger.suite_id,
                suite_status=ledger.status,
                task_index=task_index,
                task_total=len(ledger.tasks),
                dataset_id=None if task is None else task.dataset_id,
                preset_id=None if task is None else task.preset_id,
                task_status=None if task is None else task.status,
                attempt=None if task is None else task.attempt,
                output_dir=None if task is None else task.output_dir,
                completed=sum(item.status is SuiteTaskStatus.COMPLETE for item in ledger.tasks),
                succeeded=0 if task is None else task.succeeded_samples,
                failed=0 if task is None else task.failed_samples,
                total=0 if task is None else task.expected_samples,
                error_type=None if task is None else task.error_type,
                message=message,
            )
        )


def _task_result_exists(task: SuiteTask) -> bool:
    return task.output_dir is not None and (Path(task.output_dir) / "result.json").is_file()


def _resumable_task_output(task: SuiteTask) -> Path | None:
    """Return an interrupted attempt whose per-sample checkpoint can continue."""

    if task.status not in {SuiteTaskStatus.RUNNING, SuiteTaskStatus.INTERRUPTED}:
        return None
    if task.output_dir is None or task.attempt <= 0:
        return None
    output = Path(task.output_dir)
    if not (output / "checkpoint.json").is_file() or (output / "result.json").exists():
        return None
    return output


def _suite_status(tasks: Sequence[SuiteTask]) -> SuiteStatus:
    statuses = tuple(task.status for task in tasks)
    if statuses and all(status is SuiteTaskStatus.COMPLETE for status in statuses):
        return SuiteStatus.COMPLETE
    if any(status in {SuiteTaskStatus.COMPLETE, SuiteTaskStatus.PARTIAL} for status in statuses):
        return SuiteStatus.PARTIAL
    return SuiteStatus.FAILED


def _require_identifier(value: str, label: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must contain only letters, numbers, '.', '_' and '-': {value!r}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"suite field {key!r} must be a string")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int:
        raise ValueError(f"suite field {key!r} must be an integer")
    return item


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, int | float) or isinstance(item, bool):
        raise ValueError(f"suite field {key!r} must be a number")
    return float(item)


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise ValueError(f"suite field {key!r} must be a boolean")
    return item


def _optional_string(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("optional suite text fields must be strings or null")


__all__ = [
    "DEFAULT_T4_PRESETS",
    "SUITE_EVENT_SCHEMA_VERSION",
    "SUITE_LEDGER_FILENAME",
    "SUITE_SCHEMA_VERSION",
    "SuiteConflictError",
    "SuiteDataset",
    "SuiteEvent",
    "SuiteEventCallback",
    "SuiteEventKind",
    "SuiteLedger",
    "SuitePlan",
    "SuiteRunner",
    "SuiteStatus",
    "SuiteTask",
    "SuiteTaskExecutor",
    "SuiteTaskOutcome",
    "SuiteTaskStatus",
]
