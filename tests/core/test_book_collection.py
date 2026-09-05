from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path

import pytest

from dominican_eaters.collection.books import (
    AudiobookHit,
    BookCollectionRunner,
    BookCollectionStatus,
    BookManifest,
    BookSeed,
    CheckpointState,
    ConcurrentCollectionError,
    MatchKind,
    ProviderError,
    collect_book,
    load_book_manifest,
    load_checkpoint,
    write_book_manifest,
)


def hit(*, partial: bool = False) -> AudiobookHit:
    return AudiobookHit(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        provider_title="Audiolibro fragmento" if partial else "Audiolibro completo",
        duration_seconds=120.0,
        kind=MatchKind.PARTIAL if partial else MatchKind.COMPLETE,
        query="book author audiolibro",
    )


class FakeProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def search(self, seed: BookSeed) -> AudiobookHit | None:
        self.calls.append(seed.book_id)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response  # type: ignore[return-value]


def test_stable_id_normalizes_unicode_case_and_spacing() -> None:
    first = BookSeed.create("  ÁRBOL   de fuego ", "María Pérez")
    second = BookSeed.create("árbol de fuego", "mari\u0301a pérez")

    assert first.book_id == second.book_id


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"title": "", "author": "A"}, "title"),
        ({"title": "T", "author": ""}, "author"),
        ({"title": "T", "author": "A", "publication_year": 999}, "publication_year"),
    ],
)
def test_seed_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BookSeed.create(**kwargs)  # type: ignore[arg-type]


def test_hit_requires_valid_youtube_url_and_duration() -> None:
    with pytest.raises(ValueError, match="YouTube"):
        AudiobookHit("id", "https://example.com/x", "title", 1, MatchKind.COMPLETE, "q")
    with pytest.raises(ValueError, match="positive"):
        AudiobookHit("id", "https://youtu.be/id", "title", 0, MatchKind.COMPLETE, "q")
    with pytest.raises(ValueError, match="match"):
        AudiobookHit("other", "https://youtu.be/id", "title", 1, MatchKind.COMPLETE, "q")
    for nonfinite in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite|positive"):
            AudiobookHit("id", "https://youtu.be/id", "title", nonfinite, MatchKind.COMPLETE, "q")


def test_collect_one_retains_not_found_and_structured_error() -> None:
    seed = BookSeed.create("Title", "Author")
    missing = collect_book(seed, FakeProvider([None]))
    failed = collect_book(seed, FakeProvider([ProviderError("timeout", retryable=True)]))

    assert missing.status is BookCollectionStatus.NOT_FOUND
    assert missing.seed is seed
    assert failed.status is BookCollectionStatus.ERROR
    assert failed.issue is not None
    assert failed.issue.retryable is True
    assert failed.issue.error_type == "ProviderError"


def test_partial_hit_has_explicit_partial_status() -> None:
    result = collect_book(BookSeed.create("Title", "Author"), FakeProvider([hit(partial=True)]))
    assert result.status is BookCollectionStatus.PARTIAL


def test_finished_checkpoint_contains_every_seed_and_round_trips(tmp_path: Path) -> None:
    seeds = (BookSeed.create("One", "Author"), BookSeed.create("Two", "Author"))
    provider = FakeProvider([None, hit()])
    path = tmp_path / "books.json"

    result = BookCollectionRunner(provider).run(seeds, path)
    loaded = load_checkpoint(path)

    assert result.state is CheckpointState.FINISHED
    assert loaded == result
    assert [record.status for record in loaded.records] == [
        BookCollectionStatus.NOT_FOUND,
        BookCollectionStatus.FOUND,
    ]


def test_resume_skips_valid_match_retries_not_found_and_adds_new_seed(tmp_path: Path) -> None:
    first, second = BookSeed.create("One", "Author"), BookSeed.create("Two", "Author")
    path = tmp_path / "books.json"
    initial_provider = FakeProvider([hit(), None])
    BookCollectionRunner(initial_provider).run((first, second), path)
    third = BookSeed.create("Three", "Author")
    resumed_provider = FakeProvider([hit(partial=True), None])

    result = BookCollectionRunner(resumed_provider).run((first, second, third), path)

    assert resumed_provider.calls == [second.book_id, third.book_id]
    assert [record.status for record in result.records] == [
        BookCollectionStatus.FOUND,
        BookCollectionStatus.PARTIAL,
        BookCollectionStatus.NOT_FOUND,
    ]
    assert result.records[0].reused_existing is True


def test_nonretryable_error_is_reused_unless_forced(tmp_path: Path) -> None:
    seed = BookSeed.create("One", "Author")
    path = tmp_path / "books.json"
    BookCollectionRunner(FakeProvider([ProviderError("auth", retryable=False)])).run((seed,), path)
    skipped_provider = FakeProvider([])

    skipped = BookCollectionRunner(skipped_provider).run((seed,), path)
    forced_provider = FakeProvider([hit()])
    forced = BookCollectionRunner(forced_provider, force=True).run((seed,), path)

    assert skipped_provider.calls == []
    assert skipped.records[0].status is BookCollectionStatus.ERROR
    assert skipped.records[0].reused_existing is True
    assert forced_provider.calls == [seed.book_id]
    assert forced.records[0].status is BookCollectionStatus.FOUND


def test_unexpected_provider_bug_is_not_converted_to_domain_error(tmp_path: Path) -> None:
    seed = BookSeed.create("One", "Author")
    path = tmp_path / "books.json"

    with pytest.raises(RuntimeError, match="programming bug"):
        BookCollectionRunner(FakeProvider([RuntimeError("programming bug")])).run((seed,), path)

    assert load_checkpoint(path).state is CheckpointState.INTERRUPTED


def test_interrupt_preserves_all_identities_and_active_book(tmp_path: Path) -> None:
    seeds = (BookSeed.create("One", "Author"), BookSeed.create("Two", "Author"))
    path = tmp_path / "books.json"
    provider = FakeProvider([hit(), KeyboardInterrupt("stop")])

    with pytest.raises(KeyboardInterrupt, match="stop"):
        BookCollectionRunner(provider).run(seeds, path)

    checkpoint = load_checkpoint(path)
    assert checkpoint.state is CheckpointState.INTERRUPTED
    assert checkpoint.active_book_id == seeds[1].book_id
    assert checkpoint.interruption_type == "KeyboardInterrupt"
    assert [record.status for record in checkpoint.records] == [
        BookCollectionStatus.FOUND,
        BookCollectionStatus.PENDING,
    ]


def test_duplicate_ids_and_invalid_checkpoint_fail_closed(tmp_path: Path) -> None:
    seed = BookSeed.create("Same", "Author")
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="duplicate"):
        BookCollectionRunner(provider).run((seed, seed), tmp_path / "duplicate.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level schema"):
        load_checkpoint(invalid)


def test_book_manifest_round_trip_and_identity_validation(tmp_path: Path) -> None:
    seed = BookSeed.create("Over", "Ramón Marrero Aristy", publication_year=1939)
    path = tmp_path / "books-manifest.json"
    write_book_manifest(BookManifest((seed,)), path)

    assert load_book_manifest(path) == BookManifest((seed,))

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["books"][0]["title"] = "A different title"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_book_manifest(path)


def test_checkpoint_has_single_writer_lock(tmp_path: Path) -> None:
    seed = BookSeed.create("One", "Author")
    path = tmp_path / "books.json"
    lock_path = path.with_suffix(".json.lock")
    lock_path.touch()

    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ConcurrentCollectionError, match="another process owns"):
            BookCollectionRunner(FakeProvider([])).run((seed,), path)
