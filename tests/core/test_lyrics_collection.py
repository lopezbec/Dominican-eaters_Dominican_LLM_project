from __future__ import annotations

import json
from pathlib import Path

import pytest

from dominican_eaters.collection.lyrics import (
    CollectionStatus,
    GeniusCandidate,
    GeniusSongDetails,
    LyricsCollectionRunner,
    LyricsCollectionService,
    LyricsLedgerStore,
    LyricsManifest,
    LyricsRequest,
    LyricsValidationError,
    ProviderError,
    VideoMatch,
    load_lyrics_manifest,
    select_candidate,
    write_lyrics_manifest,
)


def candidate(identifier: str = "song-1", title: str = "Ojalá que llueva café") -> GeniusCandidate:
    return GeniusCandidate(
        identifier, title, "Juan Luis Guerra", f"https://genius.com/{identifier}"
    )


def details(identifier: str = "song-1") -> GeniusSongDetails:
    return GeniusSongDetails(
        identifier,
        "Ojalá que llueva café",
        "Juan Luis Guerra",
        f"https://genius.com/{identifier}",
        genres=("merengue",),
    )


def video() -> VideoMatch:
    return VideoMatch(
        "video-1",
        "https://www.youtube.com/watch?v=video-1",
        "Ojalá que llueva café - video oficial",
        duration_seconds=240,
        confidence=0.95,
    )


class FakeGenius:
    def __init__(
        self,
        searches: list[object],
        *,
        song_details: GeniusSongDetails | None = None,
        lyrics: str | None = "Ojalá que llueva café",
    ) -> None:
        self.searches = searches
        self.song_details = song_details or details()
        self.lyrics = lyrics
        self.search_calls: list[str] = []

    def search(self, query: str) -> tuple[GeniusCandidate, ...]:
        self.search_calls.append(query)
        value = self.searches.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value  # type: ignore[return-value]

    def get_details(self, source_song_id: str) -> GeniusSongDetails | None:
        return self.song_details

    def get_lyrics(self, url: str) -> str | None:
        return self.lyrics


class FakeVideos:
    def __init__(self, result: object = None) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def search(self, title: str, artist: str) -> VideoMatch | None:
        self.calls.append((title, artist))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result  # type: ignore[return-value]


def service(genius: FakeGenius, videos: FakeVideos) -> LyricsCollectionService:
    return LyricsCollectionService(genius, videos, clock=lambda: "2026-09-04T12:00:00+00:00")


def test_manifest_is_strict_versioned_and_round_trips(tmp_path: Path) -> None:
    manifest = LyricsManifest((LyricsRequest("req-1", "Juan Luis Guerra"),))
    path = tmp_path / "lyrics.json"
    write_lyrics_manifest(manifest, path)

    assert load_lyrics_manifest(path) == manifest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LyricsValidationError, match="unknown fields"):
        load_lyrics_manifest(path)


def test_manifest_rejects_duplicate_request_ids() -> None:
    request = LyricsRequest("same", "query")
    with pytest.raises(LyricsValidationError, match="duplicate"):
        LyricsManifest((request, request))


def test_candidate_selection_uses_expected_identity_not_first_position() -> None:
    request = LyricsRequest(
        "req-1",
        "song",
        expected_title="Ojalá que llueva café",
        expected_artist="Juan Luis Guerra",
    )
    wrong = GeniusCandidate("wrong", "Un verano sin ti", "Bad Bunny", "https://genius.com/wrong")
    right = candidate()

    assert select_candidate(request, (wrong, right)) == right


def test_unrelated_candidate_is_rejected() -> None:
    request = LyricsRequest(
        "req-1",
        "Ojalá que llueva café Juan Luis Guerra",
        expected_title="Ojalá que llueva café",
        expected_artist="Juan Luis Guerra",
    )
    wrong = GeniusCandidate("9", "Tití me preguntó", "Bad Bunny", "https://genius.com/wrong")

    assert select_candidate(request, (wrong,)) is None


def test_complete_and_partial_results_are_explicit() -> None:
    request = LyricsRequest(
        "req-1",
        "song",
        expected_title="Ojalá que llueva café",
        expected_artist="Juan Luis Guerra",
    )
    complete = service(FakeGenius([(candidate(),)]), FakeVideos(video())).collect(request)
    partial = service(FakeGenius([(candidate(),)], lyrics=None), FakeVideos(None)).collect(request)

    assert complete.status is CollectionStatus.COMPLETE
    assert complete.song is not None and complete.song.video is not None
    assert partial.status is CollectionStatus.PARTIAL
    assert partial.song is not None and partial.song.lyrics is None
    assert {issue.stage.value for issue in partial.issues} == {"lyrics", "video"}


def test_mismatched_detail_identity_is_a_permanent_error() -> None:
    request = LyricsRequest(
        "req-1",
        "song",
        expected_title="Ojalá que llueva café",
        expected_artist="Juan Luis Guerra",
    )
    result = service(
        FakeGenius([(candidate(),)], song_details=details("different")), FakeVideos(video())
    ).collect(request)

    assert result.status is CollectionStatus.ERROR
    assert result.issues[0].code == "identity_mismatch"
    assert result.retryable is False


def test_not_found_and_classified_provider_error_retain_request() -> None:
    request = LyricsRequest("req-1", "missing")
    missing = service(FakeGenius([()]), FakeVideos()).collect(request)
    failed = service(
        FakeGenius([ProviderError("rate_limit", "slow down", retryable=True)]), FakeVideos()
    ).collect(request)

    assert missing.status is CollectionStatus.NOT_FOUND
    assert missing.request == request
    assert failed.status is CollectionStatus.ERROR
    assert failed.retryable is True


def test_unexpected_provider_bug_propagates() -> None:
    request = LyricsRequest("req-1", "song")
    with pytest.raises(RuntimeError, match="bug"):
        service(FakeGenius([RuntimeError("bug")]), FakeVideos()).collect(request)


def test_runner_checkpoints_each_request_and_skips_complete_resume(tmp_path: Path) -> None:
    request = LyricsRequest(
        "req-1",
        "song",
        expected_title="Ojalá que llueva café",
        expected_artist="Juan Luis Guerra",
    )
    manifest = LyricsManifest((request,))
    initial_genius = FakeGenius([(candidate(),)])
    first = LyricsCollectionRunner(service(initial_genius, FakeVideos(video()))).run(
        manifest, tmp_path
    )
    resumed_genius = FakeGenius([])
    second = LyricsCollectionRunner(service(resumed_genius, FakeVideos(video()))).run(
        manifest, tmp_path
    )

    assert first == second
    assert resumed_genius.search_calls == []
    assert LyricsLedgerStore(tmp_path).load() == second


def test_resume_retries_not_found_and_merges_new_requests(tmp_path: Path) -> None:
    first_request = LyricsRequest("req-1", "missing")
    LyricsCollectionRunner(service(FakeGenius([()]), FakeVideos())).run(
        LyricsManifest((first_request,)), tmp_path
    )
    second_request = LyricsRequest("req-2", "also missing")
    genius = FakeGenius([(), ()])

    result = LyricsCollectionRunner(service(genius, FakeVideos())).run(
        LyricsManifest((first_request, second_request)), tmp_path
    )

    assert genius.search_calls == ["missing", "also missing"]
    assert [item.request.request_id for item in result.results] == ["req-1", "req-2"]
    assert result.results[0].attempt == 2


def test_keyboard_interrupt_leaves_completed_record_and_pending_identity(tmp_path: Path) -> None:
    first = LyricsRequest("req-1", "first")
    second = LyricsRequest("req-2", "second")
    genius = FakeGenius([(candidate(),), KeyboardInterrupt("stop")])

    with pytest.raises(KeyboardInterrupt, match="stop"):
        LyricsCollectionRunner(service(genius, FakeVideos(video()))).run(
            LyricsManifest((first, second)), tmp_path
        )

    ledger = LyricsLedgerStore(tmp_path).load()
    assert [result.request.request_id for result in ledger.results] == ["req-1"]
    assert ledger.pending_request_ids == ("req-2",)
