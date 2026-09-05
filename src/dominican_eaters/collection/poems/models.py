"""Validated domain records for poem recitation collection."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from urllib.parse import parse_qs, urlparse

_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")


class PoemValidationError(ValueError):
    """Raised when a poem collection value violates the domain contract."""


def required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoemValidationError(f"{field} must be a non-empty string")
    return value.strip()


def optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return required_text(value, field)


def normalized_identity(value: str) -> str:
    """Normalize display text only for identity and matching."""

    decomposed = unicodedata.normalize("NFKD", value)
    unaccented = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    words = re.findall(r"[a-z0-9]+", unaccented.casefold())
    return " ".join(words)


def derive_source_id(title: str, author: str) -> str:
    """Derive the stable identity used to merge current sources with prior state."""

    normalized_title = normalized_identity(required_text(title, "title"))
    normalized_author = normalized_identity(required_text(author, "author"))
    digest = sha256(f"{normalized_title}\0{normalized_author}".encode()).hexdigest()[:20]
    return f"poem-{digest}"


@dataclass(frozen=True, slots=True)
class PoemSource:
    source_id: str
    title: str
    author: str
    publication_year: int | None = None
    genre: str | None = None
    reference_text: str | None = None
    provenance: str | None = None

    def __post_init__(self) -> None:
        source_id = required_text(self.source_id, "source_id")
        if _SOURCE_ID.fullmatch(source_id) is None:
            raise PoemValidationError("source_id must be a lowercase portable identifier")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "title", required_text(self.title, "title"))
        object.__setattr__(self, "author", required_text(self.author, "author"))
        if self.publication_year is not None and (
            type(self.publication_year) is not int or not 1 <= self.publication_year <= 9999
        ):
            raise PoemValidationError("publication_year must be an integer from 1 to 9999")
        for field in ("genre", "reference_text", "provenance"):
            object.__setattr__(self, field, optional_text(getattr(self, field), field))

    @classmethod
    def create(
        cls,
        *,
        title: str,
        author: str,
        publication_year: int | None = None,
        genre: str | None = None,
        reference_text: str | None = None,
        provenance: str | None = None,
    ) -> PoemSource:
        return cls(
            source_id=derive_source_id(title, author),
            title=title,
            author=author,
            publication_year=publication_year,
            genre=genre,
            reference_text=reference_text,
            provenance=provenance,
        )


def _video_id_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https":
        return None
    if host == "youtu.be":
        return parsed.path.lstrip("/").split("/", 1)[0] or None
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    video_id: str
    url: str
    title: str
    duration_seconds: float | None
    provider: str
    query: str

    def __post_init__(self) -> None:
        video_id = required_text(self.video_id, "video_id")
        if _VIDEO_ID.fullmatch(video_id) is None:
            raise PoemValidationError("video_id must be an 11-character YouTube identifier")
        url = required_text(self.url, "url")
        if _video_id_from_url(url) != video_id:
            raise PoemValidationError("url must be an HTTPS YouTube URL for video_id")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "url", url)
        for field in ("title", "provider", "query"):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, int | float)
            or not isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise PoemValidationError("duration_seconds must be a positive finite number")


class ContentKind(StrEnum):
    FRAGMENT = "fragment"
    RECITATION = "recitation"
    DRAMATIZATION = "dramatization"
    READING = "reading"
    PERFORMANCE = "performance"
    AUDIO_POETRY = "audio_poetry"
    OTHER = "other"


class OutcomeStatus(StrEnum):
    FOUND = "found"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PoemOutcome:
    source: PoemSource
    status: OutcomeStatus
    attempted_at: str
    candidate: VideoCandidate | None = None
    content_kind: ContentKind | None = None
    match_score: float | None = None
    match_reasons: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    error_retryable: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, PoemSource):
            raise PoemValidationError("source must be a PoemSource")
        if not isinstance(self.status, OutcomeStatus):
            raise PoemValidationError("status must be an OutcomeStatus")
        object.__setattr__(self, "attempted_at", required_text(self.attempted_at, "attempted_at"))
        object.__setattr__(self, "match_reasons", tuple(self.match_reasons))
        matched = self.status in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}
        if matched:
            if self.candidate is None or self.content_kind is None or self.match_score is None:
                raise PoemValidationError("found outcomes require a candidate, kind, and score")
            if not isinstance(self.candidate, VideoCandidate) or not isinstance(
                self.content_kind, ContentKind
            ):
                raise PoemValidationError("found outcomes require typed candidate and kind values")
            if (
                self.error_type is not None
                or self.error_message is not None
                or self.error_retryable is not None
            ):
                raise PoemValidationError("found outcomes cannot contain an error")
            if (
                isinstance(self.match_score, bool)
                or not isinstance(self.match_score, int | float)
                or not isfinite(self.match_score)
                or not 0 <= self.match_score <= 1
            ):
                raise PoemValidationError("match_score must be between zero and one")
            if (self.status is OutcomeStatus.PARTIAL) != (
                self.content_kind is ContentKind.FRAGMENT
            ):
                raise PoemValidationError("partial status must correspond exactly to fragment kind")
        elif self.status is OutcomeStatus.ERROR:
            if (
                self.candidate is not None
                or self.content_kind is not None
                or self.match_score is not None
            ):
                raise PoemValidationError("error outcomes cannot contain a match")
            object.__setattr__(self, "error_type", required_text(self.error_type, "error_type"))
            object.__setattr__(
                self, "error_message", required_text(self.error_message, "error_message")
            )
            if not isinstance(self.error_retryable, bool):
                raise PoemValidationError("error outcomes require error_retryable")
        elif any(
            value is not None
            for value in (
                self.candidate,
                self.content_kind,
                self.match_score,
                self.error_type,
                self.error_message,
                self.error_retryable,
            )
        ):
            raise PoemValidationError("not_found outcomes cannot contain a match or error")

    @property
    def resumable(self) -> bool:
        return self.status in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL} or (
            self.status is OutcomeStatus.ERROR and self.error_retryable is False
        )
