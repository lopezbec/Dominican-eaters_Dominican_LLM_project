"""Immutable domain records for audiobook discovery."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from urllib.parse import parse_qs, urlparse

BOOK_CHECKPOINT_SCHEMA_VERSION = 1
_ID_PATTERN = re.compile(r"book_[0-9a-f]{24}")


def _required(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def stable_book_id(title: str, author: str) -> str:
    identity = f"{_identity_text(title)}\0{_identity_text(author)}".encode()
    return f"book_{sha256(identity).hexdigest()[:24]}"


class MatchKind(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class BookCollectionStatus(StrEnum):
    PENDING = "pending"
    FOUND = "found"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    ERROR = "error"


class CheckpointState(StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class BookSeed:
    book_id: str
    title: str
    author: str
    publication_year: int | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.book_id, str) or _ID_PATTERN.fullmatch(self.book_id) is None:
            raise ValueError("book_id must use the canonical book_<24 hex> form")
        object.__setattr__(self, "title", _required(self.title, "title"))
        object.__setattr__(self, "author", _required(self.author, "author"))
        if self.book_id != stable_book_id(self.title, self.author):
            raise ValueError("book_id does not match the canonical title/author identity")
        if self.publication_year is not None:
            if type(self.publication_year) is not int:
                raise ValueError("publication_year must be an integer or null")
            if not 1000 <= self.publication_year <= 2100:
                raise ValueError("publication_year must be between 1000 and 2100")
        if self.source is not None:
            object.__setattr__(self, "source", _required(self.source, "source"))

    @classmethod
    def create(
        cls,
        title: str,
        author: str,
        *,
        publication_year: int | None = None,
        source: str | None = None,
    ) -> BookSeed:
        return cls(stable_book_id(title, author), title, author, publication_year, source)


@dataclass(frozen=True, slots=True)
class AudiobookHit:
    video_id: str
    url: str
    provider_title: str
    duration_seconds: float
    kind: MatchKind
    query: str

    def __post_init__(self) -> None:
        for field in ("video_id", "provider_title", "query"):
            object.__setattr__(self, field, _required(getattr(self, field), field))
        if not isinstance(self.url, str):
            raise ValueError("url must be a string")
        parsed = urlparse(self.url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or host not in {"youtube.com", "www.youtube.com", "youtu.be"}:
            raise ValueError("url must be an HTTPS YouTube URL")
        if parsed.username is not None or parsed.password is not None or parsed.port is not None:
            raise ValueError("url must not contain credentials or a port")
        if host == "youtu.be":
            url_video_id = parsed.path.removeprefix("/").split("/", 1)[0]
        elif parsed.path == "/watch":
            url_video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith("/shorts/"):
            url_video_id = parsed.path.removeprefix("/shorts/").split("/", 1)[0]
        else:
            url_video_id = ""
        if url_video_id != self.video_id:
            raise ValueError("url video ID must match video_id")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, int | float
        ):
            raise ValueError("duration_seconds must be a number")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not isfinite(self.duration_seconds):
            raise ValueError("duration_seconds must be finite")
        if not isinstance(self.kind, MatchKind):
            raise ValueError("kind must be a MatchKind")


@dataclass(frozen=True, slots=True)
class CollectionIssue:
    stage: str
    error_type: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        for field in ("stage", "error_type", "message"):
            object.__setattr__(self, field, _required(getattr(self, field), field))
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a boolean")


@dataclass(frozen=True, slots=True)
class BookRecord:
    seed: BookSeed
    status: BookCollectionStatus = BookCollectionStatus.PENDING
    hit: AudiobookHit | None = None
    issue: CollectionIssue | None = None
    reused_existing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.seed, BookSeed):
            raise ValueError("seed must be a BookSeed")
        if not isinstance(self.status, BookCollectionStatus):
            raise ValueError("status must be a BookCollectionStatus")
        if not isinstance(self.reused_existing, bool):
            raise ValueError("reused_existing must be a boolean")
        matched = self.status in {BookCollectionStatus.FOUND, BookCollectionStatus.PARTIAL}
        if matched != (self.hit is not None):
            raise ValueError("found/partial records require a hit and other statuses forbid it")
        if (self.status is BookCollectionStatus.ERROR) != (self.issue is not None):
            raise ValueError("only error records require an issue")
        if self.hit is not None:
            expected = (
                BookCollectionStatus.PARTIAL
                if self.hit.kind is MatchKind.PARTIAL
                else BookCollectionStatus.FOUND
            )
            if self.status is not expected:
                raise ValueError("record status must agree with hit kind")


@dataclass(frozen=True, slots=True)
class BookCollectionCheckpoint:
    state: CheckpointState
    records: tuple[BookRecord, ...]
    active_book_id: str | None = None
    interruption_type: str | None = None
    interruption_message: str | None = None
    schema_version: int = BOOK_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported book checkpoint schema_version")
        if not isinstance(self.state, CheckpointState):
            raise ValueError("state must be a CheckpointState")
        if not all(isinstance(record, BookRecord) for record in self.records):
            raise ValueError("records must contain only BookRecord values")
        ids = [record.seed.book_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("checkpoint contains duplicate book IDs")
        if self.active_book_id is not None and self.active_book_id not in ids:
            raise ValueError("active_book_id must identify a checkpoint record")
        if self.state is CheckpointState.FINISHED:
            if self.active_book_id is not None:
                raise ValueError("finished checkpoints cannot have an active book")
            if any(record.status is BookCollectionStatus.PENDING for record in self.records):
                raise ValueError("finished checkpoints cannot contain pending records")
        interrupted = self.state is CheckpointState.INTERRUPTED
        if interrupted != (self.interruption_type is not None):
            raise ValueError("interrupted checkpoints require interruption details")
        if not interrupted and self.interruption_message is not None:
            raise ValueError("non-interrupted checkpoints cannot have an interruption message")
