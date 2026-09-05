from __future__ import annotations

import pytest

from dominican_eaters.collection.books import (
    BookSeed,
    MatchKind,
    YouTubeAudiobookSearch,
)
from dominican_eaters.collection.books import (
    ProviderError as BookProviderError,
)
from dominican_eaters.collection.lyrics import YouTubeMusicVideoSearch
from dominican_eaters.collection.poems import (
    PoemSource,
    RecitationQuery,
    YouTubeRecitationSearch,
)
from dominican_eaters.collection.poems import (
    ProviderError as PoemProviderError,
)
from dominican_eaters.collection.providers import YouTubeAPIError, YouTubeVideo


class API:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int = 10) -> tuple[YouTubeVideo, ...]:
        self.calls.append((query, max_results))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response  # type: ignore[return-value]


def video(identifier: str, title: str, duration: float = 180) -> YouTubeVideo:
    return YouTubeVideo(identifier, title, duration)


def test_audiobook_adapter_ranks_matches_and_marks_fragment_first() -> None:
    wrong = video("aaaaaaaaaaa", "Otra novela desconocida")
    partial = video("bbbbbbbbbbb", "Over Ramón Marrero Aristy audiolibro fragmento", 600)
    complete = video("ccccccccccc", "Over audiolibro", 600)
    api = API([(wrong, partial, complete), (), ()])

    result = YouTubeAudiobookSearch(api).search(BookSeed.create("Over", "Ramón Marrero Aristy"))

    assert result is not None
    assert result.video_id == "bbbbbbbbbbb"
    assert result.kind is MatchKind.PARTIAL
    assert len(api.calls) == 3


def test_audiobook_adapter_rejects_duration_and_identity_mismatches() -> None:
    too_short = video("aaaaaaaaaaa", "Over Ramón Marrero Aristy", 5)
    wrong_author = video("bbbbbbbbbbb", "Over otro autor", 600)
    api = API([(too_short, wrong_author), (), ()])

    assert (
        YouTubeAudiobookSearch(api).search(BookSeed.create("Over", "Ramón Marrero Aristy")) is None
    )


def test_audiobook_adapter_rejects_common_first_name_without_author_anchor() -> None:
    wrong_author = video("bbbbbbbbbbb", "Over audiolibro por Ramón García", 600)
    api = API([(wrong_author,), (), ()])

    assert (
        YouTubeAudiobookSearch(api).search(BookSeed.create("Over", "Ramón Marrero Aristy")) is None
    )


def test_poem_adapter_builds_typed_candidates_with_query_evidence() -> None:
    api = API([(video("abcdefghijk", "Hay un país en el mundo recitación"),)])
    source = PoemSource.create(
        title="Hay un país en el mundo", author="Pedro Mir", genre="poesía social"
    )

    results = YouTubeRecitationSearch(api).search(RecitationQuery(source))

    assert results[0].video_id == "abcdefghijk"
    assert results[0].provider == "youtube-data-api-v3"
    assert "poesía social" in results[0].query


def test_music_video_adapter_selects_best_title_artist_overlap() -> None:
    api = API(
        [
            (
                video("aaaaaaaaaaa", "Video cualquiera"),
                video("bbbbbbbbbbb", "Juan Luis Guerra Ojalá que llueva café oficial"),
            )
        ]
    )

    result = YouTubeMusicVideoSearch(api).search("Ojalá que llueva café", "Juan Luis Guerra")

    assert result is not None
    assert result.video_id == "bbbbbbbbbbb"
    assert result.confidence == 1.0


def test_music_video_adapter_rejects_title_without_expected_artist() -> None:
    api = API([(video("aaaaaaaaaaa", "Ojalá que llueva café artista diferente"),)])

    result = YouTubeMusicVideoSearch(api).search("Ojalá que llueva café", "Juan Luis Guerra")

    assert result is None


def test_domain_adapters_translate_transport_errors() -> None:
    error = YouTubeAPIError("quota", retryable=True, status_code=429)
    with pytest.raises(BookProviderError) as book:
        YouTubeAudiobookSearch(API([error])).search(BookSeed.create("Over", "Author"))
    assert book.value.retryable is True

    with pytest.raises(PoemProviderError) as poem:
        YouTubeRecitationSearch(API([error])).search(
            RecitationQuery(PoemSource.create(title="Poem", author="Author"))
        )
    assert poem.value.retryable is True
