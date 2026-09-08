"""Typed, API-key-free YouTube search adapter."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol, cast


class YouTubeSearchError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class YouTubeVideo:
    video_id: str
    title: str
    duration_seconds: float

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", self.video_id) is None:
            raise ValueError("video_id must be an 11-character YouTube identifier")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, int | float)
            or not isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be a positive finite number")

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class YouTubeSearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 10) -> tuple[YouTubeVideo, ...]: ...


class SearchFunction(Protocol):
    def __call__(
        self, query: str, *, limit: int | None = None, sleep: float = 1
    ) -> Iterable[Mapping[str, Any]]: ...


def parse_display_duration(value: object) -> float:
    """Parse the ``MM:SS`` or ``HH:MM:SS`` duration returned by Scrapetube."""

    if not isinstance(value, str):
        raise ValueError("YouTube duration must be a string")
    parts = value.strip().split(":")
    if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid YouTube duration: {value!r}")
    numbers = [int(part) for part in parts]
    if numbers[-1] >= 60 or (len(numbers) == 3 and numbers[-2] >= 60):
        raise ValueError(f"invalid YouTube duration: {value!r}")
    seconds = numbers[-1] + 60 * numbers[-2]
    if len(numbers) == 3:
        seconds += 3_600 * numbers[0]
    if seconds <= 0:
        raise ValueError("YouTube duration must be positive")
    return float(seconds)


class ScrapeTubeSearch:
    """Search public YouTube metadata through Scrapetube without credentials."""

    def __init__(
        self,
        *,
        search_function: SearchFunction | None = None,
        sleep_seconds: float = 1,
    ) -> None:
        if not isfinite(sleep_seconds) or sleep_seconds < 0:
            raise ValueError("sleep_seconds must be a nonnegative finite number")
        self._search_function = search_function
        self._sleep_seconds = sleep_seconds

    def search(self, query: str, *, max_results: int = 10) -> tuple[YouTubeVideo, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("YouTube query must not be empty")
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        try:
            rows = self._search()(query.strip(), limit=max_results, sleep=self._sleep_seconds)
            videos = tuple(self._parse_row(row) for row in rows)
        except YouTubeSearchError:
            raise
        except Exception as error:
            raise YouTubeSearchError(
                f"YouTube search failed: {type(error).__name__}", retryable=True
            ) from error
        return tuple(video for video in videos if video is not None)

    def _search(self) -> SearchFunction:
        if self._search_function is not None:
            return self._search_function
        try:
            scrapetube = importlib.import_module("scrapetube")
        except ImportError as error:
            raise YouTubeSearchError(
                "Install dominican-eaters[providers] to search YouTube", retryable=False
            ) from error
        self._search_function = cast(SearchFunction, scrapetube.get_search)
        return self._search_function

    @staticmethod
    def _parse_row(row: Mapping[str, Any]) -> YouTubeVideo | None:
        video_id = row.get("videoId")
        title = _renderer_text(row.get("title"))
        duration = _renderer_text(row.get("lengthText"))
        if not isinstance(video_id, str) or title is None or duration is None:
            return None
        try:
            return YouTubeVideo(video_id, title, parse_display_duration(duration))
        except ValueError:
            return None


def _renderer_text(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    simple_text = value.get("simpleText")
    if isinstance(simple_text, str) and simple_text.strip():
        return simple_text.strip()
    runs = value.get("runs")
    if not isinstance(runs, list):
        return None
    text = "".join(str(run.get("text", "")) for run in runs if isinstance(run, Mapping)).strip()
    return text or None
