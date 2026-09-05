"""Strict domain contracts for lyrics collection."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Final, overload
from urllib.parse import urlparse

LYRICS_SCHEMA_VERSION: Final = 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_REQUEST_FIELDS = frozenset({"request_id", "query", "expected_title", "expected_artist"})
_REQUIRED_REQUEST_FIELDS = frozenset({"request_id", "query"})
_MANIFEST_FIELDS = frozenset({"schema_version", "requests"})


class LyricsValidationError(ValueError):
    """Raised when lyrics collection data violates its canonical schema."""


def require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    required: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(required - set(value))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise LyricsValidationError(f"{context}: {'; '.join(details)}")


def required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise LyricsValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise LyricsValidationError(f"{field} must not be empty")
    return normalized


def optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return required_string(value, field=field)


def http_url(value: Any, *, field: str) -> str:
    url = required_string(value, field=field)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LyricsValidationError(f"{field} must be an HTTP(S) URL")
    return url


def genius_url(value: Any, *, field: str) -> str:
    """Accept only canonical HTTPS Genius URLs without authority tricks."""

    url = required_string(value, field=field)
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    try:
        port = parsed.port
    except ValueError as error:
        raise LyricsValidationError(f"{field} has an invalid port") from error
    if (
        parsed.scheme != "https"
        or (host != "genius.com" and not host.endswith(".genius.com"))
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise LyricsValidationError(f"{field} must be a canonical HTTPS Genius URL")
    return url


@dataclass(frozen=True, slots=True)
class LyricsRequest:
    """One stable, provider-independent collection request."""

    request_id: str
    query: str
    expected_title: str | None = None
    expected_artist: str | None = None

    def __post_init__(self) -> None:
        request_id = required_string(self.request_id, field="request_id")
        if _IDENTIFIER.fullmatch(request_id) is None:
            raise LyricsValidationError(
                "request_id must contain only letters, digits, '.', '_' or '-' and must not "
                "exceed 128 characters"
            )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "query", required_string(self.query, field="query"))
        object.__setattr__(
            self,
            "expected_title",
            optional_string(self.expected_title, field="expected_title"),
        )
        object.__setattr__(
            self,
            "expected_artist",
            optional_string(self.expected_artist, field="expected_artist"),
        )

    @classmethod
    def from_dict(cls, value: Any, *, index: int | None = None) -> LyricsRequest:
        context = "request" if index is None else f"requests[{index}]"
        if not isinstance(value, Mapping):
            raise LyricsValidationError(f"{context} must be an object")
        require_exact_fields(
            value,
            expected=_REQUEST_FIELDS,
            required=_REQUIRED_REQUEST_FIELDS,
            context=context,
        )
        return cls(
            request_id=value["request_id"],
            query=value["query"],
            expected_title=value.get("expected_title"),
            expected_artist=value.get("expected_artist"),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "expected_title": self.expected_title,
            "expected_artist": self.expected_artist,
        }


@dataclass(frozen=True, slots=True)
class LyricsManifest(Sequence[LyricsRequest]):
    """The immutable set of requests for a restartable collection run."""

    requests: tuple[LyricsRequest, ...]
    schema_version: int = LYRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != LYRICS_SCHEMA_VERSION:
            raise LyricsValidationError(
                f"schema_version must be {LYRICS_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        object.__setattr__(self, "requests", tuple(self.requests))
        if not self.requests:
            raise LyricsValidationError("requests must not be empty")
        if not all(isinstance(request, LyricsRequest) for request in self.requests):
            raise LyricsValidationError("requests must contain only LyricsRequest values")
        identifiers = [request.request_id for request in self.requests]
        duplicates = sorted(
            {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
        )
        if duplicates:
            raise LyricsValidationError(f"duplicate request_id values: {', '.join(duplicates)}")

    @classmethod
    def from_dict(cls, value: Any) -> LyricsManifest:
        if not isinstance(value, Mapping):
            raise LyricsValidationError("manifest must be an object")
        require_exact_fields(
            value,
            expected=_MANIFEST_FIELDS,
            required=_MANIFEST_FIELDS,
            context="manifest",
        )
        raw_requests = value["requests"]
        if not isinstance(raw_requests, list):
            raise LyricsValidationError("requests must be an array")
        return cls(
            schema_version=value["schema_version"],
            requests=tuple(
                LyricsRequest.from_dict(request, index=index)
                for index, request in enumerate(raw_requests)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requests": [request.to_dict() for request in self.requests],
        }

    @overload
    def __getitem__(self, index: int) -> LyricsRequest: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[LyricsRequest, ...]: ...

    def __getitem__(self, index: int | slice) -> LyricsRequest | tuple[LyricsRequest, ...]:
        return self.requests[index]

    def __len__(self) -> int:
        return len(self.requests)

    def __iter__(self) -> Iterator[LyricsRequest]:
        return iter(self.requests)


class CollectionStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    ERROR = "error"


class CollectionStage(StrEnum):
    SEARCH = "search"
    DETAILS = "details"
    LYRICS = "lyrics"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class CollectionIssue:
    stage: CollectionStage
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CollectionStage):
            raise LyricsValidationError("issue stage must be a CollectionStage")
        object.__setattr__(self, "code", required_string(self.code, field="issue code"))
        object.__setattr__(self, "message", required_string(self.message, field="issue message"))
        if type(self.retryable) is not bool:
            raise LyricsValidationError("issue retryable must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CollectionIssue:
        if not isinstance(value, Mapping):
            raise LyricsValidationError("issue must be an object")
        fields = frozenset({"stage", "code", "message", "retryable"})
        require_exact_fields(value, expected=fields, required=fields, context="issue")
        try:
            stage = CollectionStage(value["stage"])
        except (TypeError, ValueError) as error:
            raise LyricsValidationError(f"invalid issue stage: {value['stage']!r}") from error
        return cls(stage, value["code"], value["message"], value["retryable"])


@dataclass(frozen=True, slots=True)
class VideoMatch:
    video_id: str
    url: str
    title: str
    duration_seconds: int | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_id", required_string(self.video_id, field="video_id"))
        object.__setattr__(self, "url", http_url(self.url, field="video url"))
        object.__setattr__(self, "title", required_string(self.title, field="video title"))
        if self.duration_seconds is not None and (
            type(self.duration_seconds) is not int or self.duration_seconds < 0
        ):
            raise LyricsValidationError("duration_seconds must be a nonnegative integer")
        if self.confidence is not None and (
            not isinstance(self.confidence, int | float)
            or isinstance(self.confidence, bool)
            or not 0 <= self.confidence <= 1
            or not isfinite(self.confidence)
        ):
            raise LyricsValidationError("confidence must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: Any) -> VideoMatch:
        if not isinstance(value, Mapping):
            raise LyricsValidationError("video must be an object")
        fields = frozenset({"video_id", "url", "title", "duration_seconds", "confidence"})
        require_exact_fields(value, expected=fields, required=fields, context="video")
        return cls(
            video_id=value["video_id"],
            url=value["url"],
            title=value["title"],
            duration_seconds=value["duration_seconds"],
            confidence=value["confidence"],
        )


@dataclass(frozen=True, slots=True)
class SongRecord:
    source_song_id: str
    title: str
    artist: str
    genius_url: str
    genres: tuple[str, ...] = ()
    label: str | None = None
    album: str | None = None
    release_date: str | None = None
    lyrics: str | None = None
    video: VideoMatch | None = None
    source: str = "genius"

    def __post_init__(self) -> None:
        for field in ("source_song_id", "title", "artist", "source"):
            object.__setattr__(self, field, required_string(getattr(self, field), field=field))
        object.__setattr__(self, "genius_url", genius_url(self.genius_url, field="genius_url"))
        object.__setattr__(self, "genres", tuple(self.genres))
        if not all(isinstance(genre, str) and genre.strip() for genre in self.genres):
            raise LyricsValidationError("genres must contain only non-empty strings")
        for field in ("label", "album", "release_date", "lyrics"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, required_string(value, field=field))
        if self.video is not None and not isinstance(self.video, VideoMatch):
            raise LyricsValidationError("video must be a VideoMatch")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "source_song_id": self.source_song_id,
            "title": self.title,
            "artist": self.artist,
            "genius_url": self.genius_url,
            "genres": list(self.genres),
            "label": self.label,
            "album": self.album,
            "release_date": self.release_date,
            "lyrics": self.lyrics,
            "video": None if self.video is None else self.video.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> SongRecord:
        if not isinstance(value, Mapping):
            raise LyricsValidationError("song must be an object")
        fields = frozenset(
            {
                "source",
                "source_song_id",
                "title",
                "artist",
                "genius_url",
                "genres",
                "label",
                "album",
                "release_date",
                "lyrics",
                "video",
            }
        )
        require_exact_fields(value, expected=fields, required=fields, context="song")
        genres = value["genres"]
        if not isinstance(genres, list):
            raise LyricsValidationError("song genres must be an array")
        return cls(
            source=value["source"],
            source_song_id=value["source_song_id"],
            title=value["title"],
            artist=value["artist"],
            genius_url=value["genius_url"],
            genres=tuple(genres),
            label=value["label"],
            album=value["album"],
            release_date=value["release_date"],
            lyrics=value["lyrics"],
            video=None if value["video"] is None else VideoMatch.from_dict(value["video"]),
        )


@dataclass(frozen=True, slots=True)
class CollectionResult:
    request: LyricsRequest
    status: CollectionStatus
    attempt: int
    completed_at: str
    song: SongRecord | None = None
    issues: tuple[CollectionIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.request, LyricsRequest):
            raise LyricsValidationError("result request must be a LyricsRequest")
        if not isinstance(self.status, CollectionStatus):
            raise LyricsValidationError("result status must be a CollectionStatus")
        if type(self.attempt) is not int or self.attempt < 1:
            raise LyricsValidationError("attempt must be a positive integer")
        object.__setattr__(
            self, "completed_at", required_string(self.completed_at, field="completed_at")
        )
        object.__setattr__(self, "issues", tuple(self.issues))
        if not all(isinstance(issue, CollectionIssue) for issue in self.issues):
            raise LyricsValidationError("issues must contain only CollectionIssue values")
        if self.status is CollectionStatus.COMPLETE and (
            self.song is None or self.song.lyrics is None or self.song.video is None or self.issues
        ):
            raise LyricsValidationError("complete result requires lyrics and video without issues")
        if self.status is CollectionStatus.NOT_FOUND and (self.song is not None or self.issues):
            raise LyricsValidationError("not_found result cannot contain a song or issues")
        if self.status is CollectionStatus.ERROR and (self.song is not None or not self.issues):
            raise LyricsValidationError("error result requires issues and cannot contain a song")
        if self.status is CollectionStatus.PARTIAL and (self.song is None or not self.issues):
            raise LyricsValidationError("partial result requires a song and issues")

    @property
    def retryable(self) -> bool:
        return any(issue.retryable for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "status": self.status.value,
            "attempt": self.attempt,
            "completed_at": self.completed_at,
            "song": None if self.song is None else self.song.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CollectionResult:
        if not isinstance(value, Mapping):
            raise LyricsValidationError("result must be an object")
        fields = frozenset({"request", "status", "attempt", "completed_at", "song", "issues"})
        require_exact_fields(value, expected=fields, required=fields, context="result")
        try:
            status = CollectionStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise LyricsValidationError(
                f"invalid collection status: {value['status']!r}"
            ) from error
        raw_issues = value["issues"]
        if not isinstance(raw_issues, list):
            raise LyricsValidationError("result issues must be an array")
        return cls(
            request=LyricsRequest.from_dict(value["request"]),
            status=status,
            attempt=value["attempt"],
            completed_at=value["completed_at"],
            song=None if value["song"] is None else SongRecord.from_dict(value["song"]),
            issues=tuple(CollectionIssue.from_dict(issue) for issue in raw_issues),
        )
