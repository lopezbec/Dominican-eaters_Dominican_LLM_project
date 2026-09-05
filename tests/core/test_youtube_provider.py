from __future__ import annotations

from dataclasses import dataclass

import pytest

from dominican_eaters.collection.providers import (
    YouTubeAPIError,
    YouTubeDataAPI,
    parse_youtube_duration,
)


@dataclass
class Response:
    status_code: int
    payload: object

    def json(self) -> object:
        return self.payload


class Client:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, object], float]] = []
        self.closed = False

    def get(self, url: str, *, params, timeout: float):  # type: ignore[no-untyped-def]
        self.calls.append((url, dict(params), timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


def test_duration_parser_supports_hours_minutes_seconds_and_days() -> None:
    assert parse_youtube_duration("PT2H3M4.5S") == 7_384.5
    assert parse_youtube_duration("P1DT1S") == 86_401
    with pytest.raises(ValueError, match="invalid"):
        parse_youtube_duration("03:45")


def test_search_joins_official_search_and_video_detail_responses() -> None:
    client = Client(
        [
            Response(
                200,
                {
                    "items": [
                        {
                            "id": {"videoId": "abcdefghijk"},
                            "snippet": {"title": "Hay un pa&iacute;s"},
                        }
                    ]
                },
            ),
            Response(
                200,
                {
                    "items": [
                        {
                            "id": "abcdefghijk",
                            "contentDetails": {"duration": "PT3M5S"},
                        }
                    ]
                },
            ),
        ]
    )

    videos = YouTubeDataAPI("secret", client=client).search("poema", max_results=5)

    assert videos[0].title == "Hay un país"
    assert videos[0].duration_seconds == 185
    assert videos[0].url.endswith("v=abcdefghijk")
    assert client.calls[0][1]["type"] == "video"
    assert client.calls[0][1]["regionCode"] == "DO"
    assert client.calls[1][1]["part"] == "contentDetails"


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (403, False), (429, True), (500, True)],
)
def test_http_failures_are_classified(status: int, retryable: bool) -> None:
    api = YouTubeDataAPI("secret", client=Client([Response(status, {})]))
    with pytest.raises(YouTubeAPIError) as captured:
        api.search("query")
    assert captured.value.retryable is retryable
    assert captured.value.status_code == status


def test_transport_and_schema_failures_are_explicit() -> None:
    with pytest.raises(YouTubeAPIError) as transport:
        YouTubeDataAPI("secret", client=Client([OSError("offline")])).search("query")
    assert transport.value.retryable is True

    with pytest.raises(YouTubeAPIError, match="items"):
        YouTubeDataAPI("secret", client=Client([Response(200, {})])).search("query")


def test_missing_video_details_fail_instead_of_silently_dropping_a_candidate() -> None:
    client = Client(
        [
            Response(
                200,
                {
                    "items": [
                        {
                            "id": {"videoId": "abcdefghijk"},
                            "snippet": {"title": "Poema"},
                        }
                    ]
                },
            ),
            Response(200, {"items": []}),
        ]
    )

    with pytest.raises(YouTubeAPIError, match="missing IDs: abcdefghijk") as captured:
        YouTubeDataAPI("secret", client=client).search("query")

    assert captured.value.retryable is True


def test_malformed_video_identifier_is_a_typed_provider_error() -> None:
    client = Client(
        [
            Response(
                200,
                {"items": [{"id": {"videoId": "bad"}, "snippet": {"title": "Poema"}}]},
            ),
            Response(
                200,
                {"items": [{"id": "bad", "contentDetails": {"duration": "PT1M"}}]},
            ),
        ]
    )

    with pytest.raises(YouTubeAPIError, match="invalid YouTube video metadata") as captured:
        YouTubeDataAPI("secret", client=client).search("query")

    assert captured.value.retryable is False


def test_injected_client_is_not_owned_or_closed() -> None:
    client = Client([Response(200, {"items": []})])
    api = YouTubeDataAPI("secret", client=client)
    api.search("query")
    api.close()
    assert client.closed is False
    with pytest.raises(YouTubeAPIError, match="closed"):
        api.search("query")
