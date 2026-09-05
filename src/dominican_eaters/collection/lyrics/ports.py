"""Provider boundaries for lyrics collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import VideoMatch, genius_url, required_string


class ProviderError(RuntimeError):
    """A classified provider failure safe to persist in run artifacts."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.code = required_string(code, field="provider error code")
        self.retryable = retryable
        if type(retryable) is not bool:
            raise TypeError("retryable must be a boolean")
        super().__init__(required_string(message, field="provider error message"))


@dataclass(frozen=True, slots=True)
class GeniusCandidate:
    source_song_id: str
    title: str
    artist: str
    url: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_song_id", required_string(self.source_song_id, field="source_song_id")
        )
        object.__setattr__(self, "title", required_string(self.title, field="title"))
        object.__setattr__(self, "artist", required_string(self.artist, field="artist"))
        object.__setattr__(self, "url", genius_url(self.url, field="candidate url"))


@dataclass(frozen=True, slots=True)
class GeniusSongDetails:
    source_song_id: str
    title: str
    artist: str
    url: str
    genres: tuple[str, ...] = ()
    label: str | None = None
    album: str | None = None
    release_date: str | None = None

    def __post_init__(self) -> None:
        candidate = GeniusCandidate(self.source_song_id, self.title, self.artist, self.url)
        object.__setattr__(self, "source_song_id", candidate.source_song_id)
        object.__setattr__(self, "title", candidate.title)
        object.__setattr__(self, "artist", candidate.artist)
        object.__setattr__(self, "url", candidate.url)
        object.__setattr__(self, "genres", tuple(self.genres))
        if not all(isinstance(genre, str) and genre.strip() for genre in self.genres):
            raise ValueError("genres must contain only non-empty strings")
        for field in ("label", "album", "release_date"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, required_string(value, field=field))


class GeniusProvider(Protocol):
    def search(self, query: str) -> tuple[GeniusCandidate, ...]: ...

    def get_details(self, source_song_id: str) -> GeniusSongDetails | None: ...

    def get_lyrics(self, url: str) -> str | None: ...


class MusicVideoProvider(Protocol):
    def search(self, title: str, artist: str) -> VideoMatch | None: ...


class RateLimiter(Protocol):
    def wait(self) -> None: ...


class NoopRateLimiter:
    def wait(self) -> None:
        """Return immediately; useful when the provider performs its own limiting."""
