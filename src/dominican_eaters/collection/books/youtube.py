"""YouTube Data API adapter for the audiobook provider port."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from dominican_eaters.collection.providers import YouTubeAPIError, YouTubeDataAPI, YouTubeVideo

from .models import AudiobookHit, BookSeed, MatchKind
from .service import ProviderError

_PARTIAL_MARKERS = frozenset({"fragmento", "fragmentos", "extracto", "parcial"})


def _tokens(value: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return frozenset(re.findall(r"[a-z0-9]+", normalized.casefold()))


def _words(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return tuple(re.findall(r"[a-z0-9]+", normalized.casefold()))


@dataclass(frozen=True, slots=True)
class _RankedHit:
    video: YouTubeVideo
    query: str
    score: float


class YouTubeAudiobookSearch:
    def __init__(
        self,
        api: YouTubeDataAPI,
        *,
        results_per_query: int = 5,
        minimum_duration_seconds: float = 30.0,
        maximum_duration_seconds: float = 43_200.0,
        minimum_title_coverage: float = 0.5,
    ) -> None:
        if not 1 <= results_per_query <= 50:
            raise ValueError("results_per_query must be between 1 and 50")
        if minimum_duration_seconds <= 0 or maximum_duration_seconds <= minimum_duration_seconds:
            raise ValueError("audiobook duration bounds are invalid")
        if not 0 <= minimum_title_coverage <= 1:
            raise ValueError("minimum_title_coverage must be between zero and one")
        self._api = api
        self._limit = results_per_query
        self._minimum_duration = minimum_duration_seconds
        self._maximum_duration = maximum_duration_seconds
        self._minimum_title_coverage = minimum_title_coverage

    def search(self, seed: BookSeed) -> AudiobookHit | None:
        title_tokens = _tokens(seed.title)
        author_tokens = _tokens(seed.author)
        author_words = _words(seed.author)
        author_anchor = author_words[-1] if author_words else ""
        ranked: list[_RankedHit] = []
        seen: set[str] = set()
        queries = (
            f"{seed.title} {seed.author} audiolibro completo",
            f"{seed.title} {seed.author} libro completo",
            f"{seed.title} {seed.author} audiolibro",
        )
        try:
            for query in queries:
                for video in self._api.search(query, max_results=self._limit):
                    if video.video_id in seen:
                        continue
                    seen.add(video.video_id)
                    if (
                        not self._minimum_duration
                        <= video.duration_seconds
                        <= self._maximum_duration
                    ):
                        continue
                    candidate_tokens = _tokens(video.title)
                    title_coverage = (
                        len(title_tokens & candidate_tokens) / len(title_tokens)
                        if title_tokens
                        else 0.0
                    )
                    if title_coverage < self._minimum_title_coverage:
                        continue
                    author_coverage = (
                        len(author_tokens & candidate_tokens) / len(author_tokens)
                        if author_tokens
                        else 0.0
                    )
                    if (
                        author_anchor
                        and author_anchor not in candidate_tokens
                        and author_coverage < 0.75
                    ):
                        continue
                    ranked.append(
                        _RankedHit(video, query, 0.8 * title_coverage + 0.2 * author_coverage)
                    )
        except YouTubeAPIError as exc:
            raise ProviderError(str(exc), retryable=exc.retryable) from exc
        if not ranked:
            return None
        selected = sorted(ranked, key=lambda item: (-item.score, item.video.video_id))[0]
        kind = (
            MatchKind.PARTIAL
            if _tokens(selected.video.title) & _PARTIAL_MARKERS
            else MatchKind.COMPLETE
        )
        return AudiobookHit(
            video_id=selected.video.video_id,
            url=selected.video.url,
            provider_title=selected.video.title,
            duration_seconds=selected.video.duration_seconds,
            kind=kind,
            query=selected.query,
        )
