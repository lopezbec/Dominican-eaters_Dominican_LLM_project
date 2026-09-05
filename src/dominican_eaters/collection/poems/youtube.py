"""YouTube Data API adapter for poem recitation candidate discovery."""

from __future__ import annotations

from dominican_eaters.collection.providers import YouTubeAPIError, YouTubeDataAPI

from .models import VideoCandidate
from .ports import ProviderError, RecitationQuery


class YouTubeRecitationSearch:
    def __init__(self, api: YouTubeDataAPI, *, max_results: int = 10) -> None:
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        self._api = api
        self._limit = max_results

    def search(self, query: RecitationQuery) -> tuple[VideoCandidate, ...]:
        source = query.source
        parts = [source.title, source.author, source.genre or "", "poema recitación"]
        text = " ".join(part for part in parts if part).strip()
        try:
            videos = self._api.search(text, max_results=self._limit)
        except YouTubeAPIError as exc:
            raise ProviderError(str(exc), retryable=exc.retryable) from exc
        return tuple(
            VideoCandidate(
                video_id=video.video_id,
                url=video.url,
                title=video.title,
                duration_seconds=video.duration_seconds,
                provider="youtube-data-api-v3",
                query=text,
            )
            for video in videos
        )
