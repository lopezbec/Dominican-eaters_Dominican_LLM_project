"""Versioned, atomic checkpoints for poem collection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from dominican_eaters.data import atomic_write_json

from .models import ContentKind, OutcomeStatus, PoemOutcome, PoemSource, VideoCandidate

CHECKPOINT_SCHEMA_VERSION: Final = 1


class CheckpointState(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class CollectionCheckpoint:
    sources: tuple[PoemSource, ...]
    outcomes: tuple[PoemOutcome, ...]
    state: CheckpointState
    updated_at: str
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if type(self.schema_version) is not int or self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CHECKPOINT_SCHEMA_VERSION}")
        if not isinstance(self.state, CheckpointState):
            raise ValueError("state must be a CheckpointState")
        if not all(isinstance(source, PoemSource) for source in self.sources):
            raise ValueError("sources must contain only PoemSource values")
        if not all(isinstance(outcome, PoemOutcome) for outcome in self.outcomes):
            raise ValueError("outcomes must contain only PoemOutcome values")
        if not isinstance(self.updated_at, str) or not self.updated_at.strip():
            raise ValueError("updated_at must be a non-empty string")
        source_ids = [source.source_id for source in self.sources]
        duplicates = sorted({value for value in source_ids if source_ids.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate source_id values: {', '.join(duplicates)}")
        outcome_ids = [outcome.source.source_id for outcome in self.outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("checkpoint outcomes must contain at most one result per source")
        unknown = sorted(set(outcome_ids) - set(source_ids))
        if unknown:
            raise ValueError(f"checkpoint outcomes refer to unknown sources: {', '.join(unknown)}")
        if self.state is CheckpointState.COMPLETE and set(outcome_ids) != set(source_ids):
            raise ValueError("complete checkpoint requires one outcome per source")

    def to_dict(self) -> dict[str, Any]:
        def outcome_dict(outcome: PoemOutcome) -> dict[str, Any]:
            value = asdict(outcome)
            value["status"] = outcome.status.value
            value["content_kind"] = outcome.content_kind.value if outcome.content_kind else None
            return value

        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "updated_at": self.updated_at,
            "sources": [asdict(source) for source in self.sources],
            "outcomes": [outcome_dict(outcome) for outcome in self.outcomes],
        }


def write_checkpoint(path: Path, checkpoint: CollectionCheckpoint) -> None:
    atomic_write_json(path, checkpoint.to_dict())


def _exact(value: Mapping[str, Any], fields: set[str], context: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{context} fields do not match schema")


def load_checkpoint(path: Path) -> CollectionCheckpoint:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read checkpoint: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint must be an object")
    _exact(raw, {"schema_version", "state", "updated_at", "sources", "outcomes"}, "checkpoint")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CHECKPOINT_SCHEMA_VERSION}")
    sources = tuple(_source_from_dict(value) for value in _array(raw["sources"], "sources"))
    outcomes = tuple(_outcome_from_dict(value) for value in _array(raw["outcomes"], "outcomes"))
    return CollectionCheckpoint(
        sources=sources,
        outcomes=outcomes,
        state=CheckpointState(raw["state"]),
        updated_at=raw["updated_at"],
        schema_version=raw["schema_version"],
    )


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _object(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _source_from_dict(value: object) -> PoemSource:
    raw = _object(value, "source")
    fields = {
        "source_id",
        "title",
        "author",
        "publication_year",
        "genre",
        "reference_text",
        "provenance",
    }
    _exact(raw, fields, "source")
    return PoemSource(
        source_id=raw["source_id"],
        title=raw["title"],
        author=raw["author"],
        publication_year=raw["publication_year"],
        genre=raw["genre"],
        reference_text=raw["reference_text"],
        provenance=raw["provenance"],
    )


def _candidate_from_dict(value: object) -> VideoCandidate | None:
    if value is None:
        return None
    raw = _object(value, "candidate")
    fields = {"video_id", "url", "title", "duration_seconds", "provider", "query"}
    _exact(raw, fields, "candidate")
    return VideoCandidate(
        video_id=raw["video_id"],
        url=raw["url"],
        title=raw["title"],
        duration_seconds=raw["duration_seconds"],
        provider=raw["provider"],
        query=raw["query"],
    )


def _outcome_from_dict(value: object) -> PoemOutcome:
    raw = _object(value, "outcome")
    fields = {
        "source",
        "status",
        "attempted_at",
        "candidate",
        "content_kind",
        "match_score",
        "match_reasons",
        "error_type",
        "error_message",
        "error_retryable",
    }
    _exact(raw, fields, "outcome")
    kind = raw["content_kind"]
    reasons = raw["match_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("match_reasons must be an array of strings")
    return PoemOutcome(
        source=_source_from_dict(raw["source"]),
        status=OutcomeStatus(raw["status"]),
        attempted_at=raw["attempted_at"],
        candidate=_candidate_from_dict(raw["candidate"]),
        content_kind=ContentKind(kind) if kind is not None else None,
        match_score=raw["match_score"],
        match_reasons=tuple(reasons),
        error_type=raw["error_type"],
        error_message=raw["error_message"],
        error_retryable=raw["error_retryable"],
    )
