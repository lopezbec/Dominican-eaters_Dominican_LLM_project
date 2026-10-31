from __future__ import annotations

import json
from pathlib import Path

import pytest

from dominican_eaters.evaluation.asr.suite import (
    SuiteConflictError,
    SuiteDataset,
    SuiteEvent,
    SuiteEventKind,
    SuitePlan,
    SuiteRunner,
    SuiteStatus,
    SuiteTaskOutcome,
    SuiteTaskStatus,
)


def dataset(dataset_id: str = "music", *, digest: str = "a" * 64) -> SuiteDataset:
    return SuiteDataset(dataset_id, f"/datasets/{dataset_id}.json", digest, 3)


def plan(*, digest: str = "a" * 64) -> SuitePlan:
    return SuitePlan(
        datasets=(dataset(digest=digest),),
        presets=("whisper-large-v3", "qwen3-asr-1.7b"),
    )


def test_suite_persists_matrix_and_continues_after_task_failure(tmp_path: Path) -> None:
    calls: list[tuple[str, str, Path]] = []
    events: list[SuiteEvent] = []

    def execute(item: SuiteDataset, preset: str, output: Path) -> SuiteTaskOutcome:
        calls.append((item.dataset_id, preset, output))
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}\n", encoding="utf-8")
        if preset == "qwen3-asr-1.7b":
            raise RuntimeError("worker unavailable")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    ledger = SuiteRunner(execute, on_event=events.append).run(plan(), tmp_path / "suite")

    assert ledger.status is SuiteStatus.PARTIAL
    assert [task.status for task in ledger.tasks] == [
        SuiteTaskStatus.COMPLETE,
        SuiteTaskStatus.FAILED,
    ]
    assert [call[:2] for call in calls] == [
        ("music", "whisper-large-v3"),
        ("music", "qwen3-asr-1.7b"),
    ]
    assert calls[0][2].name == "attempt-001"
    assert events[0].kind is SuiteEventKind.SUITE_STARTED
    assert events[-1].kind is SuiteEventKind.SUITE_FINISHED
    payload = json.loads((tmp_path / "suite" / "suite.json").read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert len(payload["tasks"]) == 2


def test_resume_skips_valid_complete_tasks_and_retries_failed_tasks(tmp_path: Path) -> None:
    first_calls: list[str] = []

    def first(_dataset: SuiteDataset, preset: str, output: Path) -> SuiteTaskOutcome:
        first_calls.append(preset)
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}\n", encoding="utf-8")
        status = (
            SuiteTaskStatus.COMPLETE if preset == "whisper-large-v3" else SuiteTaskStatus.FAILED
        )
        return SuiteTaskOutcome(status, 3, 3 if status is SuiteTaskStatus.COMPLETE else 0, 0)

    output = tmp_path / "suite"
    SuiteRunner(first).run(plan(), output)
    resumed_calls: list[Path] = []
    events: list[SuiteEvent] = []

    def resumed(_dataset: SuiteDataset, _preset: str, task_output: Path) -> SuiteTaskOutcome:
        resumed_calls.append(task_output)
        task_output.mkdir(parents=True)
        (task_output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    ledger = SuiteRunner(resumed, on_event=events.append).run(plan(), output, resume=True)

    assert first_calls == ["whisper-large-v3", "qwen3-asr-1.7b"]
    assert [path.name for path in resumed_calls] == ["attempt-002"]
    assert ledger.status is SuiteStatus.COMPLETE
    assert [task.attempt for task in ledger.tasks] == [1, 2]
    assert any(event.kind is SuiteEventKind.TASK_SKIPPED for event in events)


def test_resume_reruns_complete_task_when_result_artifact_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "suite"

    def execute(_dataset: SuiteDataset, _preset: str, task_output: Path) -> SuiteTaskOutcome:
        task_output.mkdir(parents=True)
        (task_output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    first = SuiteRunner(execute).run(plan(), output)
    first_task = first.tasks[0]
    assert first_task.output_dir is not None
    (Path(first_task.output_dir) / "result.json").unlink()
    calls: list[str] = []

    def resumed(_dataset: SuiteDataset, preset: str, task_output: Path) -> SuiteTaskOutcome:
        calls.append(preset)
        task_output.mkdir(parents=True)
        (task_output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    ledger = SuiteRunner(resumed).run(plan(), output)

    assert calls == ["whisper-large-v3"]
    assert ledger.tasks[0].attempt == 2
    assert ledger.tasks[1].attempt == 1


def test_resume_reuses_interrupted_attempt_with_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    task_output: Path | None = None

    def interrupted(_dataset: SuiteDataset, _preset: str, current_output: Path) -> SuiteTaskOutcome:
        nonlocal task_output
        task_output = current_output
        current_output.mkdir(parents=True)
        (current_output / "checkpoint.json").write_text("{}\n", encoding="utf-8")
        raise KeyboardInterrupt

    single_plan = SuitePlan((dataset(),), ("whisper-large-v3",))
    with pytest.raises(KeyboardInterrupt):
        SuiteRunner(interrupted).run(single_plan, output)

    resumed_outputs: list[Path] = []

    def resumed(_dataset: SuiteDataset, _preset: str, current_output: Path) -> SuiteTaskOutcome:
        resumed_outputs.append(current_output)
        (current_output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    ledger = SuiteRunner(resumed).run(single_plan, output)

    assert task_output is not None
    assert resumed_outputs == [task_output]
    assert ledger.tasks[0].attempt == 1
    assert ledger.status is SuiteStatus.COMPLETE


def test_resume_rejects_changed_dataset_fingerprint(tmp_path: Path) -> None:
    def execute(_dataset: SuiteDataset, _preset: str, output: Path) -> SuiteTaskOutcome:
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    output = tmp_path / "suite"
    SuiteRunner(execute).run(plan(), output)

    with pytest.raises(SuiteConflictError, match="different dataset/model plan"):
        SuiteRunner(execute).run(plan(digest="b" * 64), output)


def test_no_resume_rejects_existing_suite(tmp_path: Path) -> None:
    def execute(_dataset: SuiteDataset, _preset: str, output: Path) -> SuiteTaskOutcome:
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 3, 0)

    output = tmp_path / "suite"
    SuiteRunner(execute).run(plan(), output)

    with pytest.raises(SuiteConflictError, match="already exists"):
        SuiteRunner(execute).run(plan(), output, resume=False)


def test_suite_event_payload_exposes_dataset_model_and_counts(tmp_path: Path) -> None:
    events: list[SuiteEvent] = []

    def execute(_dataset: SuiteDataset, _preset: str, output: Path) -> SuiteTaskOutcome:
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}\n", encoding="utf-8")
        return SuiteTaskOutcome(SuiteTaskStatus.COMPLETE, 3, 2, 1)

    SuiteRunner(execute, on_event=events.append).run(
        SuitePlan((dataset(),), ("whisper-large-v3",)), tmp_path / "suite"
    )

    completed = next(event for event in events if event.kind is SuiteEventKind.TASK_COMPLETED)
    payload = completed.to_dict()
    assert payload["dataset_id"] == "music"
    assert payload["preset"] == "whisper-large-v3"
    assert payload["succeeded"] == 2
    assert payload["failed"] == 1
    assert payload["total"] == 3
