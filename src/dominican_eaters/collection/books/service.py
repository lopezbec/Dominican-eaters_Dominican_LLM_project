"""Provider-independent audiobook collection decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .models import (
    AudiobookHit,
    BookCollectionStatus,
    BookRecord,
    BookSeed,
    CollectionIssue,
    MatchKind,
)


class AudiobookSearchProvider(Protocol):
    def search(self, seed: BookSeed) -> AudiobookHit | None: ...


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def collect_book(
    seed: BookSeed,
    provider: AudiobookSearchProvider,
    *,
    existing: BookRecord | None = None,
) -> BookRecord:
    """Collect one book without ever losing its source identity."""

    if existing is not None and existing.seed.book_id != seed.book_id:
        raise ValueError("existing record does not belong to this seed")
    if existing is not None and existing.status in {
        BookCollectionStatus.FOUND,
        BookCollectionStatus.PARTIAL,
    }:
        return replace(existing, seed=seed, reused_existing=True)
    try:
        hit = provider.search(seed)
    except ProviderError as exc:
        return BookRecord(
            seed=seed,
            status=BookCollectionStatus.ERROR,
            issue=CollectionIssue(
                stage="search",
                error_type=type(exc).__name__,
                message=str(exc) or type(exc).__name__,
                retryable=exc.retryable,
            ),
        )
    if hit is None:
        return BookRecord(seed=seed, status=BookCollectionStatus.NOT_FOUND)
    status = (
        BookCollectionStatus.PARTIAL
        if hit.kind is MatchKind.PARTIAL
        else BookCollectionStatus.FOUND
    )
    return BookRecord(seed=seed, status=status, hit=hit)
