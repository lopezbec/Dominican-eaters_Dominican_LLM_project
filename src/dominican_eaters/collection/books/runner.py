"""Atomic, resumable orchestration for audiobook collection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dominican_eaters.data import (
    ConcurrentWriteError,
    atomic_write_json,
    exclusive_file_lock,
    to_json_value,
)

from .models import (
    BOOK_CHECKPOINT_SCHEMA_VERSION,
    AudiobookHit,
    BookCollectionCheckpoint,
    BookCollectionStatus,
    BookRecord,
    BookSeed,
    CheckpointState,
    CollectionIssue,
    MatchKind,
)
from .service import AudiobookSearchProvider, collect_book


class BookCollectionRunner:
    def __init__(
        self,
        provider: AudiobookSearchProvider,
        *,
        force: bool = False,
        retry_not_found: bool = True,
        retry_nonretryable_errors: bool = False,
    ) -> None:
        self._provider = provider
        self._force = force
        self._retry_not_found = retry_not_found
        self._retry_nonretryable_errors = retry_nonretryable_errors

    def run(self, seeds: Sequence[BookSeed], checkpoint_path: Path) -> BookCollectionCheckpoint:
        try:
            with exclusive_file_lock(checkpoint_path, label="book collection checkpoint"):
                return self._run_locked(seeds, checkpoint_path)
        except ConcurrentWriteError as exc:
            raise ConcurrentCollectionError(str(exc)) from exc

    def _run_locked(
        self, seeds: Sequence[BookSeed], checkpoint_path: Path
    ) -> BookCollectionCheckpoint:
        current = tuple(seeds)
        if not current:
            raise ValueError("book collection requires at least one seed")
        if not all(isinstance(seed, BookSeed) for seed in current):
            raise ValueError("book collection seeds must contain only BookSeed values")
        _ensure_unique(current)
        prior = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
        prior_by_id = {record.seed.book_id: record for record in prior.records} if prior else {}
        records = [
            replace(prior_by_id[seed.book_id], seed=seed, reused_existing=False)
            if seed.book_id in prior_by_id
            else BookRecord(seed)
            for seed in current
        ]
        self._write(
            checkpoint_path, BookCollectionCheckpoint(CheckpointState.RUNNING, tuple(records))
        )

        active: str | None = None
        try:
            for index, seed in enumerate(current):
                existing = records[index]
                if not self._force and self._should_reuse(existing):
                    records[index] = replace(existing, seed=seed, reused_existing=True)
                    continue
                active = seed.book_id
                self._write(
                    checkpoint_path,
                    BookCollectionCheckpoint(
                        CheckpointState.RUNNING, tuple(records), active_book_id=active
                    ),
                )
                records[index] = collect_book(
                    seed,
                    self._provider,
                    existing=None if self._force else existing,
                )
                active = None
                self._write(
                    checkpoint_path,
                    BookCollectionCheckpoint(CheckpointState.RUNNING, tuple(records)),
                )
        except BaseException as exc:
            self._write(
                checkpoint_path,
                BookCollectionCheckpoint(
                    CheckpointState.INTERRUPTED,
                    tuple(records),
                    active_book_id=active,
                    interruption_type=type(exc).__name__,
                    interruption_message=str(exc),
                ),
            )
            raise

        result = BookCollectionCheckpoint(CheckpointState.FINISHED, tuple(records))
        self._write(checkpoint_path, result)
        return result

    def _should_reuse(self, record: BookRecord) -> bool:
        if record.status in {BookCollectionStatus.FOUND, BookCollectionStatus.PARTIAL}:
            return True
        if record.status is BookCollectionStatus.NOT_FOUND:
            return not self._retry_not_found
        return (
            record.status is BookCollectionStatus.ERROR
            and record.issue is not None
            and not record.issue.retryable
            and not self._retry_nonretryable_errors
        )

    @staticmethod
    def _write(path: Path, checkpoint: BookCollectionCheckpoint) -> None:
        payload = to_json_value(checkpoint)
        if not isinstance(payload, dict):
            raise TypeError("book checkpoint root must be an object")
        atomic_write_json(path, payload)


def load_checkpoint(path: Path) -> BookCollectionCheckpoint:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read book checkpoint: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "state",
        "records",
        "active_book_id",
        "interruption_type",
        "interruption_message",
        "schema_version",
    }:
        raise ValueError("book checkpoint has an invalid top-level schema")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != BOOK_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported book checkpoint schema_version")
    records_raw = value["records"]
    if not isinstance(records_raw, list):
        raise ValueError("book checkpoint records must be an array")
    if not all(isinstance(item, dict) for item in records_raw):
        raise ValueError("book checkpoint records must contain objects")
    records = tuple(_record_from_dict(cast(dict[str, Any], item)) for item in records_raw)
    for field in ("active_book_id", "interruption_type", "interruption_message"):
        if value[field] is not None and not isinstance(value[field], str):
            raise ValueError(f"{field} must be a string or null")
    return BookCollectionCheckpoint(
        state=CheckpointState(value["state"]),
        records=records,
        active_book_id=value["active_book_id"],
        interruption_type=value["interruption_type"],
        interruption_message=value["interruption_message"],
        schema_version=value["schema_version"],
    )


def _record_from_dict(value: dict[str, Any]) -> BookRecord:
    if set(value) != {"seed", "status", "hit", "issue", "reused_existing"}:
        raise ValueError("book record has an invalid schema")
    if not isinstance(value["seed"], dict):
        raise ValueError("book record seed must be an object")
    seed_raw = cast(dict[str, Any], value["seed"])
    seed = BookSeed(**seed_raw)
    hit_raw = value["hit"]
    if hit_raw is None:
        hit = None
    else:
        if not isinstance(hit_raw, dict):
            raise ValueError("book record hit must be an object or null")
        hit_fields = dict(cast(dict[str, Any], hit_raw))
        hit_fields["kind"] = MatchKind(hit_fields["kind"])
        hit = AudiobookHit(**hit_fields)
    issue_raw = value["issue"]
    if issue_raw is not None and not isinstance(issue_raw, dict):
        raise ValueError("book record issue must be an object or null")
    issue = None if issue_raw is None else CollectionIssue(**cast(dict[str, Any], issue_raw))
    if not isinstance(value["reused_existing"], bool):
        raise ValueError("reused_existing must be a boolean")
    return BookRecord(
        seed=seed,
        status=BookCollectionStatus(value["status"]),
        hit=hit,
        issue=issue,
        reused_existing=value["reused_existing"],
    )


def _ensure_unique(seeds: Sequence[BookSeed]) -> None:
    ids = [seed.book_id for seed in seeds]
    duplicates = sorted({book_id for book_id in ids if ids.count(book_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate book IDs: {', '.join(duplicates)}")


class ConcurrentCollectionError(RuntimeError):
    """Raised when another process owns a collection checkpoint."""
