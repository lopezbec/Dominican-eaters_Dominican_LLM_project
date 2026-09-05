"""Provider boundary for poem recitation discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import PoemSource, VideoCandidate


class ProviderError(RuntimeError):
    """Expected transport/provider failure safe to persist as an outcome."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean")
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RecitationQuery:
    source: PoemSource


class RecitationSearch(Protocol):
    """Return parsed candidates without deciding which one is a domain match."""

    def search(self, query: RecitationQuery) -> tuple[VideoCandidate, ...]: ...
