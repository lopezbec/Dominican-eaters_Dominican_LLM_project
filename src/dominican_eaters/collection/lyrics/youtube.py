"""YouTube Data API adapter for music-video discovery."""

from __future__ import annotations

import re
import unicodedata

from dominican_eaters.collection.providers import YouTubeAPIError, YouTubeDataAPI

from .contracts import VideoMatch
from .ports import ProviderError


def _tokens(value: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return frozenset(re.findall(r"[a-z0-9]+", normalized.casefold()))


class YouTubeMusicVideoSearch:
    def __init__(
        self, api: YouTubeDataAPI, *, max_results: int = 10, minimum_confidence: float = 0.5
    ) -> None:
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        if not 0 < minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be greater than zero and at most one")
        self._api = api
        self._limit = max_results
        self._minimum_confidence = minimum_confidence

    def search(self, title: str, artist: str) -> VideoMatch | None:
        expected_title = _tokens(title)
        expected_artist = _tokens(artist)
        query = f"{title} {artist} video oficial"
        try:
            videos = self._api.search(query, max_results=self._limit)
        except YouTubeAPIError as exc:
            raise ProviderError("youtube_api", str(exc), retryable=exc.retryable) from exc
        if not videos:
            return None

        def score(candidate_title: str) -> tuple[float, float]:
            actual = _tokens(candidate_title)
            dimensions = (
                len(expected_title & actual) / len(expected_title),
                len(expected_artist & actual) / len(expected_artist),
            )
            return min(dimensions), sum(dimensions) / len(dimensions)

        selected = sorted(
            videos,
            key=lambda video: (-score(video.title)[0], -score(video.title)[1], video.video_id),
        )[0]
        minimum_identity_score, confidence = score(selected.title)
        if minimum_identity_score < self._minimum_confidence:
            return None
        return VideoMatch(
            video_id=selected.video_id,
            url=selected.url,
            title=selected.title,
            duration_seconds=round(selected.duration_seconds),
            confidence=confidence,
        )
