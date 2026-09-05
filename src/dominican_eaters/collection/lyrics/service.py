"""Network-independent orchestration for one lyrics request."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from .contracts import (
    CollectionIssue,
    CollectionResult,
    CollectionStage,
    CollectionStatus,
    LyricsRequest,
    SongRecord,
    VideoMatch,
)
from .ports import (
    GeniusCandidate,
    GeniusProvider,
    GeniusSongDetails,
    MusicVideoProvider,
    NoopRateLimiter,
    ProviderError,
    RateLimiter,
)


def _normalized_tokens(value: str) -> frozenset[str]:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return frozenset(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def select_candidate(
    request: LyricsRequest,
    candidates: Sequence[GeniusCandidate],
    *,
    minimum_score: float = 0.5,
) -> GeniusCandidate | None:
    """Choose deterministically, using explicit expected identity when supplied."""

    if not 0 < minimum_score <= 1:
        raise ValueError("minimum_score must be greater than zero and at most one")
    if not candidates:
        return None
    expected_title = _normalized_tokens(request.expected_title or "")
    expected_artist = _normalized_tokens(request.expected_artist or "")
    if not expected_title and not expected_artist:
        expected_title = _normalized_tokens(request.query)

    def score(candidate: GeniusCandidate) -> tuple[float, float, int]:
        dimensions: list[float] = []
        for expected, actual in (
            (expected_title, _normalized_tokens(candidate.title)),
            (expected_artist, _normalized_tokens(candidate.artist)),
        ):
            if expected:
                dimensions.append(len(expected & actual) / len(expected))
        return (
            min(dimensions),
            sum(dimensions) / len(dimensions),
            -candidates.index(candidate),
        )

    selected = max(candidates, key=score)
    return selected if score(selected)[0] >= minimum_score else None


class LyricsCollectionService:
    def __init__(
        self,
        genius: GeniusProvider,
        videos: MusicVideoProvider,
        *,
        limiter: RateLimiter | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._genius = genius
        self._videos = videos
        self._limiter = limiter or NoopRateLimiter()
        self._clock = clock or (lambda: datetime.now(UTC).isoformat())

    def collect(self, request: LyricsRequest, *, attempt: int = 1) -> CollectionResult:
        try:
            self._limiter.wait()
            candidates = self._genius.search(request.query)
        except ProviderError as error:
            return self._error_result(request, attempt, CollectionStage.SEARCH, error)

        candidate = select_candidate(request, candidates)
        if candidate is None:
            return CollectionResult(
                request=request,
                status=CollectionStatus.NOT_FOUND,
                attempt=attempt,
                completed_at=self._clock(),
            )

        try:
            details = self._genius.get_details(candidate.source_song_id)
        except ProviderError as error:
            return self._error_result(request, attempt, CollectionStage.DETAILS, error)
        if details is None:
            return self._error_result(
                request,
                attempt,
                CollectionStage.DETAILS,
                ProviderError(
                    "invalid_response", "provider returned no song details", retryable=False
                ),
            )
        if details.source_song_id != candidate.source_song_id:
            return self._error_result(
                request,
                attempt,
                CollectionStage.DETAILS,
                ProviderError(
                    "identity_mismatch",
                    "song details ID does not match the selected candidate",
                    retryable=False,
                ),
            )

        issues: list[CollectionIssue] = []
        lyrics = self._fetch_lyrics(details, issues)
        video = self._fetch_video(details, issues)
        song = SongRecord(
            source_song_id=details.source_song_id,
            title=details.title,
            artist=details.artist,
            genius_url=details.url,
            genres=details.genres,
            label=details.label,
            album=details.album,
            release_date=details.release_date,
            lyrics=lyrics,
            video=video,
        )
        status = CollectionStatus.COMPLETE if not issues else CollectionStatus.PARTIAL
        return CollectionResult(
            request=request,
            status=status,
            attempt=attempt,
            completed_at=self._clock(),
            song=song,
            issues=tuple(issues),
        )

    def _fetch_lyrics(
        self, details: GeniusSongDetails, issues: list[CollectionIssue]
    ) -> str | None:
        try:
            lyrics = self._genius.get_lyrics(details.url)
        except ProviderError as error:
            issues.append(self._issue(CollectionStage.LYRICS, error))
            return None
        if lyrics is None or not lyrics.strip():
            issues.append(
                CollectionIssue(
                    CollectionStage.LYRICS,
                    "not_found",
                    "lyrics were not found",
                    False,
                )
            )
            return None
        return lyrics.strip()

    def _fetch_video(
        self, details: GeniusSongDetails, issues: list[CollectionIssue]
    ) -> VideoMatch | None:
        try:
            video = self._videos.search(details.title, details.artist)
        except ProviderError as error:
            issues.append(self._issue(CollectionStage.VIDEO, error))
            return None
        if video is None:
            issues.append(
                CollectionIssue(
                    CollectionStage.VIDEO,
                    "not_found",
                    "music video was not found",
                    False,
                )
            )
        return video

    def _error_result(
        self,
        request: LyricsRequest,
        attempt: int,
        stage: CollectionStage,
        error: Exception,
    ) -> CollectionResult:
        return CollectionResult(
            request=request,
            status=CollectionStatus.ERROR,
            attempt=attempt,
            completed_at=self._clock(),
            issues=(self._issue(stage, error),),
        )

    @staticmethod
    def _issue(stage: CollectionStage, error: Exception) -> CollectionIssue:
        if isinstance(error, ProviderError):
            return CollectionIssue(stage, error.code, str(error), error.retryable)
        return CollectionIssue(stage, "unexpected", str(error) or type(error).__name__, False)
