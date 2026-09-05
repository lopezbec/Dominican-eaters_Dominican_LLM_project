"""Small typed adapter for the official YouTube Data API v3."""

from __future__ import annotations

import html
import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol, cast

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
_DURATION = re.compile(
    r"P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?"
)


class YouTubeAPIError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


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


class HTTPResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class HTTPClient(Protocol):
    def get(self, url: str, *, params: Mapping[str, object], timeout: float) -> HTTPResponse: ...

    def close(self) -> None: ...


def parse_youtube_duration(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("YouTube duration must be a string")
    match = _DURATION.fullmatch(value)
    if match is None or not any(match.groupdict().values()):
        raise ValueError(f"invalid YouTube duration: {value!r}")
    parts = {key: float(number or 0) for key, number in match.groupdict().items()}
    seconds = (
        parts["days"] * 86_400 + parts["hours"] * 3_600 + parts["minutes"] * 60 + parts["seconds"]
    )
    if seconds <= 0:
        raise ValueError("YouTube duration must be positive")
    return seconds


class YouTubeDataAPI:
    """Fetch public search metadata using an API key and bounded requests."""

    def __init__(
        self,
        api_key: str,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 20.0,
        region_code: str = "DO",
        relevance_language: str = "es",
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("YouTube API key must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not re.fullmatch(r"[A-Z]{2}", region_code):
            raise ValueError("region_code must be an ISO 3166-1 alpha-2 code")
        if not re.fullmatch(r"[a-z]{2}", relevance_language):
            raise ValueError("relevance_language must be a two-letter code")
        self._api_key = api_key.strip()
        self._client = client
        self._owns_client = client is None
        self._closed = False
        self._timeout = timeout_seconds
        self._region = region_code
        self._language = relevance_language

    def search(self, query: str, *, max_results: int = 10) -> tuple[YouTubeVideo, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("YouTube query must not be empty")
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        search = self._request(
            _SEARCH_URL,
            {
                "key": self._api_key,
                "part": "snippet",
                "q": query.strip(),
                "type": "video",
                "maxResults": max_results,
                "regionCode": self._region,
                "relevanceLanguage": self._language,
                "safeSearch": "none",
            },
        )
        candidates: list[tuple[str, str]] = []
        for index, item in enumerate(_items(search, "search")):
            try:
                entry = _mapping(item, f"search.items[{index}]")
                identifier = _mapping(entry["id"], f"search.items[{index}].id")["videoId"]
                title = _mapping(entry["snippet"], f"search.items[{index}].snippet")["title"]
            except KeyError as exc:
                raise YouTubeAPIError(
                    f"YouTube search response is missing {exc.args[0]!r}", retryable=False
                ) from exc
            if not isinstance(identifier, str) or not identifier:
                raise YouTubeAPIError("YouTube videoId must be a non-empty string", retryable=False)
            if not isinstance(title, str) or not title.strip():
                raise YouTubeAPIError("YouTube title must be a non-empty string", retryable=False)
            candidates.append((identifier, html.unescape(title).strip()))
        if not candidates:
            return ()
        details = self._request(
            _VIDEOS_URL,
            {
                "key": self._api_key,
                "part": "contentDetails",
                "id": ",".join(identifier for identifier, _title in candidates),
            },
        )
        durations: dict[str, float] = {}
        for index, item in enumerate(_items(details, "videos")):
            entry = _mapping(item, f"videos.items[{index}]")
            identifier = entry.get("id")
            duration = _mapping(
                entry.get("contentDetails"), f"videos.items[{index}].contentDetails"
            ).get("duration")
            if not isinstance(identifier, str):
                raise YouTubeAPIError("YouTube video detail ID must be a string", retryable=False)
            try:
                durations[identifier] = parse_youtube_duration(duration)
            except ValueError as exc:
                raise YouTubeAPIError(str(exc), retryable=False) from exc
        missing = [identifier for identifier, _title in candidates if identifier not in durations]
        if missing:
            raise YouTubeAPIError(
                "YouTube video details are missing IDs: " + ", ".join(missing),
                retryable=True,
            )
        try:
            return tuple(
                YouTubeVideo(identifier, title, durations[identifier])
                for identifier, title in candidates
            )
        except ValueError as exc:
            raise YouTubeAPIError(
                f"invalid YouTube video metadata: {exc}", retryable=False
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None
        self._closed = True

    def _request(self, url: str, params: Mapping[str, object]) -> Mapping[str, object]:
        client = self._client_instance()
        try:
            response = client.get(url, params=params, timeout=self._timeout)
        except Exception as exc:
            raise YouTubeAPIError(
                f"YouTube request failed: {type(exc).__name__}", retryable=True
            ) from exc
        status = response.status_code
        if status < 200 or status >= 300:
            raise YouTubeAPIError(
                f"YouTube API returned HTTP {status}",
                status_code=status,
                retryable=status == 429 or status >= 500,
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise YouTubeAPIError("YouTube API returned invalid JSON", retryable=False) from exc
        return _mapping(payload, "YouTube response")

    def _client_instance(self) -> HTTPClient:
        if self._closed:
            raise YouTubeAPIError("YouTube client is closed", retryable=False)
        if self._client is None:
            try:
                httpx = importlib.import_module("httpx")
            except ImportError as exc:
                raise YouTubeAPIError(
                    "Install dominican-eaters[providers] to use the YouTube adapter",
                    retryable=False,
                ) from exc
            self._client = cast(HTTPClient, httpx.Client())
        return self._client


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise YouTubeAPIError(f"{context} must be an object", retryable=False)
    return value


def _items(value: Mapping[str, object], context: str) -> list[object]:
    items = value.get("items")
    if not isinstance(items, list):
        raise YouTubeAPIError(f"{context}.items must be an array", retryable=False)
    return items
