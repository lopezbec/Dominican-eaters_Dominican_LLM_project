from __future__ import annotations

from dataclasses import dataclass

import pytest

from dominican_eaters.collection.lyrics import GeniusAPI, ProviderError


@dataclass
class Response:
    status_code: int
    payload: object
    text: str = ""

    def json(self) -> object:
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Client:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, object, object]] = []

    def get(self, url: str, *, params=None, headers=None, timeout: float):  # type: ignore[no-untyped-def]
        self.calls.append((url, params, headers))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        pass


def test_search_and_details_parse_typed_official_payloads() -> None:
    client = Client(
        [
            Response(
                200,
                {
                    "response": {
                        "hits": [
                            {
                                "result": {
                                    "id": 42,
                                    "title": "Ojalá que llueva café",
                                    "url": "https://genius.com/song",
                                    "primary_artist": {"name": "Juan Luis Guerra"},
                                }
                            }
                        ]
                    }
                },
            ),
            Response(
                200,
                {
                    "response": {
                        "song": {
                            "id": 42,
                            "title": "Ojalá que llueva café",
                            "url": "https://genius.com/song",
                            "primary_artist": {"name": "Juan Luis Guerra"},
                            "primary_tag": {"name": "Merengue"},
                            "album": {"name": "Ojalá que llueva café"},
                            "release_date_for_display": "1989",
                        }
                    }
                },
            ),
        ]
    )
    api = GeniusAPI("token", client=client)

    candidates = api.search("Juan Luis Guerra")
    song = api.get_details("42")

    assert candidates[0].source_song_id == "42"
    assert song is not None and song.genres == ("Merengue",)
    assert song.album == "Ojalá que llueva café"
    assert client.calls[0][2] == {"Authorization": "Bearer token"}


def test_lyrics_parser_preserves_sections_breaks_and_parenthetical_words() -> None:
    html = """
    <html><div data-lyrics-container="true">[Coro]<br>Ojalá (que llueva café)</div>
    <div data-lyrics-container="true">Conuco de monte</div><script>ignored</script></html>
    """
    api = GeniusAPI("token", client=Client([Response(200, {}, html)]))

    assert api.get_lyrics("https://genius.com/song") == (
        "[Coro]\nOjalá (que llueva café)\nConuco de monte"
    )


def test_lyrics_parser_preserves_nested_block_boundaries() -> None:
    html = """
    <div data-lyrics-container="true"><div>Línea uno</div><div>Línea dos</div>
    <p>Línea tres</p></div>
    """
    api = GeniusAPI("token", client=Client([Response(200, {}, html)]))

    assert api.get_lyrics("https://genius.com/song") == "Línea uno\nLínea dos\nLínea tres"


@pytest.mark.parametrize(
    "url",
    [
        "http://genius.com/song",
        "https://localhost/song",
        "https://169.254.169.254/latest/meta-data",
        "https://genius.com.evil.test/song",
        "https://user@genius.com/song",
        "https://genius.com:8443/song",
    ],
)
def test_lyrics_fetch_rejects_noncanonical_urls_without_network_access(url: str) -> None:
    client = Client([])

    with pytest.raises(ProviderError, match="canonical HTTPS Genius URL") as captured:
        GeniusAPI("token", client=client).get_lyrics(url)

    assert captured.value.retryable is False
    assert client.calls == []


@pytest.mark.parametrize(("status", "retryable"), [(401, False), (429, True), (503, True)])
def test_http_errors_are_classified(status: int, retryable: bool) -> None:
    api = GeniusAPI("token", client=Client([Response(status, {})]))
    with pytest.raises(ProviderError) as captured:
        api.search("query")
    assert captured.value.retryable is retryable


def test_transport_and_invalid_schema_fail_explicitly() -> None:
    with pytest.raises(ProviderError) as transport:
        GeniusAPI("token", client=Client([OSError("offline")])).search("query")
    assert transport.value.retryable is True

    with pytest.raises(ProviderError, match="hits"):
        GeniusAPI("token", client=Client([Response(200, {"response": {}})])).search("query")


def test_search_rejects_blank_queries_before_network_access() -> None:
    client = Client([])

    with pytest.raises(ValueError, match="query must not be empty"):
        GeniusAPI("token", client=client).search("  ")

    assert client.calls == []


def test_song_details_reject_path_like_ids_before_network_access() -> None:
    client = Client([])

    with pytest.raises(ProviderError, match="positive decimal"):
        GeniusAPI("token", client=client).get_details("../search")

    assert client.calls == []


def test_song_details_reject_mismatched_response_identity() -> None:
    client = Client(
        [
            Response(
                200,
                {
                    "response": {
                        "song": {
                            "id": 43,
                            "title": "Song",
                            "url": "https://genius.com/song",
                            "primary_artist": {"name": "Artist"},
                        }
                    }
                },
            )
        ]
    )

    with pytest.raises(ProviderError, match="does not match") as captured:
        GeniusAPI("token", client=client).get_details("42")

    assert captured.value.retryable is False


def test_closed_client_cannot_be_reused() -> None:
    api = GeniusAPI("token", client=Client([]))
    api.close()
    api.close()

    with pytest.raises(ProviderError, match="closed") as captured:
        api.search("query")

    assert captured.value.retryable is False


def test_missing_optional_dependency_remains_nonretryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_httpx(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(
        "dominican_eaters.collection.lyrics.genius.importlib.import_module", missing_httpx
    )

    with pytest.raises(ProviderError) as captured:
        GeniusAPI("token").search("query")

    assert captured.value.code == "missing_dependency"
    assert captured.value.retryable is False
