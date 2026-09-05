"""Typed Genius API and lyrics-page adapter."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any, Protocol, cast

from .contracts import LyricsValidationError, genius_url
from .ports import GeniusCandidate, GeniusSongDetails, ProviderError

_API_ROOT = "https://api.genius.com"


class HTTPResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> object: ...


class HTTPClient(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float,
    ) -> HTTPResponse: ...

    def close(self) -> None: ...


class _LyricsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._ignored_depth = 0
        self._parts: list[str] = []

    @property
    def text(self) -> str | None:
        lines = [" ".join(line.split()) for line in "".join(self._parts).splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned or None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self._depth == 0 and attributes.get("data-lyrics-container") == "true":
            self._depth = 1
            return
        if self._depth:
            self._depth += 1
            if tag in {"script", "style"}:
                self._ignored_depth += 1
            elif tag == "br":
                self._parts.append("\n")
            elif tag in {"div", "p", "li"} and self._parts:
                self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._depth:
            return
        if self._ignored_depth and tag in {"script", "style"}:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in {"div", "p", "li"}:
            self._parts.append("\n")
        self._depth -= 1
        if self._depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._depth and not self._ignored_depth:
            self._parts.append(data)


class GeniusAPI:
    def __init__(
        self,
        access_token: str,
        *,
        client: HTTPClient | None = None,
        timeout_seconds: float = 20.0,
        results_per_search: int = 10,
    ) -> None:
        if not isinstance(access_token, str) or not access_token.strip():
            raise ValueError("Genius access token must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 1 <= results_per_search <= 50:
            raise ValueError("results_per_search must be between 1 and 50")
        self._token = access_token.strip()
        self._client = client
        self._owns_client = client is None
        self._closed = False
        self._timeout = timeout_seconds
        self._limit = results_per_search

    def search(self, query: str) -> tuple[GeniusCandidate, ...]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Genius query must not be empty")
        payload = self._api_request("/search", params={"q": query.strip(), "per_page": self._limit})
        response = _mapping(payload.get("response"), "Genius search response")
        hits = response.get("hits")
        if not isinstance(hits, list):
            raise ProviderError("invalid_response", "Genius hits must be an array", retryable=False)
        candidates: list[GeniusCandidate] = []
        for index, hit in enumerate(hits):
            result = _mapping(_mapping(hit, f"hits[{index}]").get("result"), "Genius result")
            artist = _mapping(result.get("primary_artist"), "Genius primary_artist")
            try:
                candidates.append(
                    GeniusCandidate(
                        source_song_id=_song_id(_required(result, "id")),
                        title=_string(_required(result, "title"), "title"),
                        artist=_string(_required(artist, "name"), "artist name"),
                        url=_string(_required(result, "url"), "song URL"),
                    )
                )
            except ValueError as exc:
                raise ProviderError(
                    "invalid_response", f"invalid Genius search result: {exc}", retryable=False
                ) from exc
        return tuple(candidates)

    def get_details(self, source_song_id: str) -> GeniusSongDetails | None:
        requested_id = _song_id(source_song_id)
        payload = self._api_request(f"/songs/{requested_id}")
        response = _mapping(payload.get("response"), "Genius song response")
        raw_song = response.get("song")
        if raw_song is None:
            return None
        song = _mapping(raw_song, "Genius song")
        artist = _mapping(song.get("primary_artist"), "Genius primary_artist")
        primary_tag = song.get("primary_tag")
        genres: tuple[str, ...] = ()
        if primary_tag is not None:
            tag = _mapping(primary_tag, "Genius primary_tag")
            name = tag.get("name")
            if isinstance(name, str) and name.strip():
                genres = (name.strip(),)
        album_value = song.get("album")
        album = None
        if album_value is not None:
            album_name = _mapping(album_value, "Genius album").get("name")
            album = (
                album_name.strip() if isinstance(album_name, str) and album_name.strip() else None
            )
        returned_id = _song_id(_required(song, "id"))
        if returned_id != requested_id:
            raise ProviderError(
                "identity_mismatch",
                "Genius song response ID does not match the requested ID",
                retryable=False,
            )
        try:
            return GeniusSongDetails(
                source_song_id=returned_id,
                title=_string(_required(song, "title"), "title"),
                artist=_string(_required(artist, "name"), "artist name"),
                url=_string(_required(song, "url"), "song URL"),
                genres=genres,
                label=_optional_string(song.get("label")),
                album=album,
                release_date=_optional_string(song.get("release_date_for_display")),
            )
        except ValueError as exc:
            raise ProviderError(
                "invalid_response", f"invalid Genius song details: {exc}", retryable=False
            ) from exc

    def get_lyrics(self, url: str) -> str | None:
        try:
            safe_url = genius_url(url, field="lyrics URL")
        except LyricsValidationError as exc:
            raise ProviderError("invalid_url", str(exc), retryable=False) from exc
        response = self._request(safe_url, headers=None, params=None)
        parser = _LyricsParser()
        try:
            parser.feed(response.text)
            parser.close()
        except Exception as exc:
            raise ProviderError(
                "invalid_html", "could not parse Genius lyrics page", retryable=False
            ) from exc
        return parser.text

    def close(self) -> None:
        if self._closed:
            return
        if self._client is not None and self._owns_client:
            self._client.close()
        self._client = None
        self._closed = True

    def _api_request(
        self, path: str, *, params: Mapping[str, object] | None = None
    ) -> Mapping[str, Any]:
        response = self._request(
            f"{_API_ROOT}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._token}"},
        )
        try:
            payload = response.json()
        except Exception as exc:
            raise ProviderError(
                "invalid_json", "Genius returned invalid JSON", retryable=False
            ) from exc
        return _mapping(payload, "Genius response")

    def _request(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None,
        headers: Mapping[str, str] | None,
    ) -> HTTPResponse:
        client = self._client_instance()
        try:
            response = client.get(url, params=params, headers=headers, timeout=self._timeout)
        except Exception as exc:
            raise ProviderError(
                "transport", f"Genius request failed: {type(exc).__name__}", retryable=True
            ) from exc
        status = response.status_code
        if not 200 <= status < 300:
            retryable = status == 429 or status >= 500
            code = "rate_limit" if status == 429 else "http_error"
            raise ProviderError(code, f"Genius returned HTTP {status}", retryable=retryable)
        return response

    def _client_instance(self) -> HTTPClient:
        if self._closed:
            raise ProviderError("closed", "Genius client is closed", retryable=False)
        if self._client is None:
            try:
                httpx = importlib.import_module("httpx")
            except ImportError as exc:
                raise ProviderError(
                    "missing_dependency",
                    "Install dominican-eaters[providers] to use the Genius adapter",
                    retryable=False,
                ) from exc
            self._client = cast(HTTPClient, httpx.Client())
        return self._client


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderError("invalid_response", f"{context} must be an object", retryable=False)
    return value


def _required(value: Mapping[str, Any], key: str) -> object:
    if key not in value:
        raise ProviderError(
            "invalid_response", f"Genius response is missing {key!r}", retryable=False
        )
    return value[key]


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            "invalid_response", f"Genius {field} must be a non-empty string", retryable=False
        )
    return value.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _song_id(value: object) -> str:
    if isinstance(value, bool):
        raise ProviderError("invalid_response", "Genius song ID must be positive", retryable=False)
    normalized = str(value).strip() if isinstance(value, int | str) else ""
    if re.fullmatch(r"[1-9][0-9]*", normalized) is None:
        raise ProviderError(
            "invalid_response", "Genius song ID must be a positive decimal", retryable=False
        )
    return normalized
