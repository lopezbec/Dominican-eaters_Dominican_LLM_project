from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest

from dominican_eaters.collection.providers import (
    ScrapeTubeSearch,
    YouTubeSearchError,
    parse_display_duration,
)


class SearchFixture:
    def __init__(self, rows: Iterable[Mapping[str, Any]] | Exception) -> None:
        self.rows = rows
        self.calls: list[tuple[str, int | None, float]] = []

    def __call__(
        self, query: str, *, limit: int | None = None, sleep: float = 1
    ) -> Iterable[Mapping[str, Any]]:
        self.calls.append((query, limit, sleep))
        if isinstance(self.rows, Exception):
            raise self.rows
        return self.rows


def test_duration_parser_supports_minute_and_hour_displays() -> None:
    assert parse_display_duration("03:05") == 185
    assert parse_display_duration("2:03:04") == 7_384
    with pytest.raises(ValueError, match="invalid"):
        parse_display_duration("PT3M5S")
    with pytest.raises(ValueError, match="invalid"):
        parse_display_duration("2:61:00")


def test_search_maps_scrapetube_results_without_credentials() -> None:
    fixture = SearchFixture(
        [
            {
                "videoId": "abcdefghijk",
                "title": {"runs": [{"text": "Hay un país"}]},
                "lengthText": {"simpleText": "3:05"},
            }
        ]
    )

    videos = ScrapeTubeSearch(search_function=fixture, sleep_seconds=0).search(
        "poema", max_results=5
    )

    assert videos[0].title == "Hay un país"
    assert videos[0].duration_seconds == 185
    assert videos[0].url.endswith("v=abcdefghijk")
    assert fixture.calls == [("poema", 5, 0)]


def test_search_skips_live_and_malformed_results() -> None:
    fixture = SearchFixture(
        [
            {"videoId": "abcdefghijk", "title": {"simpleText": "Live"}},
            {
                "videoId": "bad",
                "title": {"simpleText": "Malformed"},
                "lengthText": {"simpleText": "1:00"},
            },
        ]
    )

    assert ScrapeTubeSearch(search_function=fixture).search("query") == ()


def test_transport_failure_is_typed_and_retryable() -> None:
    with pytest.raises(YouTubeSearchError) as captured:
        ScrapeTubeSearch(search_function=SearchFixture(OSError("offline"))).search("query")

    assert captured.value.retryable is True
    assert "OSError" in str(captured.value)


def test_missing_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError

    monkeypatch.setattr(
        "dominican_eaters.collection.providers.youtube.importlib.import_module", missing
    )

    with pytest.raises(YouTubeSearchError, match=r"dominican-eaters\[providers\]") as captured:
        ScrapeTubeSearch().search("query")

    assert captured.value.retryable is False


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ScrapeTubeSearch(search_function=SearchFixture([])).search(query)
