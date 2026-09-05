"""Atomic, restartable run ledger for lyrics collection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dominican_eaters.data import atomic_write_json, exclusive_file_lock

from .contracts import (
    LYRICS_SCHEMA_VERSION,
    CollectionResult,
    CollectionStatus,
    LyricsManifest,
    LyricsValidationError,
    require_exact_fields,
)

LEDGER_FILENAME: Final = "lyrics-collection.json"
_LEDGER_FIELDS = frozenset({"schema_version", "manifest", "results"})


class LedgerConflictError(RuntimeError):
    """Raised when an output ledger belongs to a different request manifest."""


@dataclass(frozen=True, slots=True)
class CollectionLedger:
    manifest: LyricsManifest
    results: tuple[CollectionResult, ...] = ()
    schema_version: int = LYRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != LYRICS_SCHEMA_VERSION:
            raise LyricsValidationError(f"ledger schema_version must be {LYRICS_SCHEMA_VERSION}")
        object.__setattr__(self, "results", tuple(self.results))
        manifest_by_id = {request.request_id: request for request in self.manifest}
        seen: set[str] = set()
        for result in self.results:
            identifier = result.request.request_id
            if identifier in seen:
                raise LyricsValidationError(f"duplicate result request_id: {identifier}")
            seen.add(identifier)
            if manifest_by_id.get(identifier) != result.request:
                raise LyricsValidationError(
                    f"result request does not match manifest request: {identifier}"
                )

    @property
    def by_request_id(self) -> dict[str, CollectionResult]:
        return {result.request.request_id: result for result in self.results}

    @property
    def pending_request_ids(self) -> tuple[str, ...]:
        completed = self.by_request_id
        return tuple(
            request.request_id for request in self.manifest if request.request_id not in completed
        )

    def with_result(self, result: CollectionResult) -> CollectionLedger:
        by_id = self.by_request_id
        by_id[result.request.request_id] = result
        ordered = tuple(
            by_id[request.request_id] for request in self.manifest if request.request_id in by_id
        )
        return CollectionLedger(self.manifest, ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CollectionLedger:
        if not isinstance(value, Mapping):
            raise LyricsValidationError("ledger must be an object")
        require_exact_fields(
            value,
            expected=_LEDGER_FIELDS,
            required=_LEDGER_FIELDS,
            context="ledger",
        )
        raw_results = value["results"]
        if not isinstance(raw_results, list):
            raise LyricsValidationError("ledger results must be an array")
        return cls(
            schema_version=value["schema_version"],
            manifest=LyricsManifest.from_dict(value["manifest"]),
            results=tuple(CollectionResult.from_dict(result) for result in raw_results),
        )


class LyricsLedgerStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.path = self.output_dir / LEDGER_FILENAME

    def initialize(self, manifest: LyricsManifest) -> CollectionLedger:
        if self.path.exists():
            previous = self.load()
            previous_requests = {request.request_id: request for request in previous.manifest}
            for request in manifest:
                old = previous_requests.get(request.request_id)
                if old is not None and old != request:
                    raise LedgerConflictError(
                        f"request_id changed meaning in existing ledger: {request.request_id}"
                    )
            current_ids = {request.request_id for request in manifest}
            ledger = CollectionLedger(
                manifest,
                tuple(
                    result
                    for result in previous.results
                    if result.request.request_id in current_ids
                ),
            )
            self.save(ledger)
            return ledger
        ledger = CollectionLedger(manifest)
        self.save(ledger)
        return ledger

    def load(self) -> CollectionLedger:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LyricsValidationError(f"could not read ledger {self.path}: {error}") from error
        return CollectionLedger.from_dict(raw)

    def save(self, ledger: CollectionLedger) -> None:
        atomic_write_json(self.path, ledger.to_dict())


class LyricsCollectionRunner:
    """Collect a manifest while checkpointing every completed request atomically."""

    def __init__(
        self,
        service: object,
        *,
        retry_not_found: bool = True,
        retry_partial: bool = True,
        retry_nonretryable_errors: bool = False,
    ) -> None:
        from .service import LyricsCollectionService

        if not isinstance(service, LyricsCollectionService):
            raise TypeError("service must be a LyricsCollectionService")
        self._service = service
        self._retry_not_found = retry_not_found
        self._retry_partial = retry_partial
        self._retry_nonretryable_errors = retry_nonretryable_errors

    def run(
        self, manifest: LyricsManifest, output_dir: str | Path, *, force: bool = False
    ) -> CollectionLedger:
        store = LyricsLedgerStore(output_dir)
        with exclusive_file_lock(store.path, label="lyrics collection ledger"):
            return self._run_locked(manifest, store, force=force)

    def _run_locked(
        self, manifest: LyricsManifest, store: LyricsLedgerStore, *, force: bool
    ) -> CollectionLedger:
        ledger = store.initialize(manifest)
        previous = ledger.by_request_id

        if force:
            ledger = CollectionLedger(manifest)
            store.save(ledger)
            previous = {}

        for request in manifest:
            old = previous.get(request.request_id)
            if not force and old is not None and not self._should_retry(old):
                continue
            attempt = 1 if old is None else old.attempt + 1
            result = self._service.collect(request, attempt=attempt)
            ledger = ledger.with_result(result)
            store.save(ledger)
        return ledger

    def _should_retry(self, result: CollectionResult) -> bool:
        if result.status is CollectionStatus.NOT_FOUND:
            return self._retry_not_found
        if result.status is CollectionStatus.PARTIAL:
            return self._retry_partial
        if result.status is CollectionStatus.ERROR:
            return result.retryable or self._retry_nonretryable_errors
        return False
