from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dominican_eaters.collection.poems import (
    CheckpointState,
    ContentKind,
    OutcomeStatus,
    PoemCollector,
    PoemManifest,
    PoemSource,
    ProviderError,
    RecitationQuery,
    VideoCandidate,
    classify_content,
    derive_source_id,
    load_checkpoint,
    load_poem_manifest,
    select_candidate,
    write_poem_manifest,
)

NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def candidate(
    video_id: str = "abcdefghijk",
    title: str = "Hay un país en el mundo recitación Pedro Mir",
) -> VideoCandidate:
    return VideoCandidate(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
        title=title,
        duration_seconds=180,
        provider="youtube",
        query="Hay un país en el mundo Pedro Mir recitación",
    )


class FakeProvider:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def search(self, query: RecitationQuery) -> tuple[VideoCandidate, ...]:
        self.calls.append(query.source.source_id)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response  # type: ignore[return-value]


def source(title: str = "Hay un país en el mundo") -> PoemSource:
    return PoemSource.create(title=title, author="Pedro Mir", genre="social")


def collector(provider: FakeProvider, path: Path) -> PoemCollector:
    return PoemCollector(provider=provider, checkpoint_path=path, clock=lambda: NOW)


def test_source_identity_normalizes_accents_case_and_spacing() -> None:
    assert derive_source_id("  AMÉN de mariposas", "Pedro Mir") == derive_source_id(
        "amen   DE mariposas", "pedro mir"
    )


def test_fragment_classification_takes_precedence() -> None:
    assert classify_content("Recitación - fragmento del poema") is ContentKind.FRAGMENT


def test_candidate_selection_is_deterministic_and_rejects_weak_match() -> None:
    poem = source()
    weak = candidate("zzzzzzzzzzz", "Poema desconocido")
    strong = candidate()

    assert select_candidate(poem, (weak, strong)) is not None
    assert select_candidate(poem, (weak,), minimum_score=0.5) is None


def test_found_partial_not_found_and_provider_error_are_explicit(tmp_path: Path) -> None:
    complete = collector(FakeProvider([(candidate(),)]), tmp_path / "complete.json").collect(
        (source(),)
    )
    fragment = candidate(title="Hay un país en el mundo fragmento Pedro Mir")
    partial = collector(FakeProvider([(fragment,)]), tmp_path / "partial.json").collect((source(),))
    missing = collector(FakeProvider([()]), tmp_path / "missing.json").collect((source(),))
    failed = collector(
        FakeProvider([ProviderError("timeout", retryable=True)]), tmp_path / "failed.json"
    ).collect((source(),))

    assert complete.outcomes[0].status is OutcomeStatus.FOUND
    assert partial.outcomes[0].status is OutcomeStatus.PARTIAL
    assert missing.outcomes[0].status is OutcomeStatus.NOT_FOUND
    assert failed.outcomes[0].status is OutcomeStatus.ERROR
    assert failed.outcomes[0].error_retryable is True


def test_unexpected_provider_bug_propagates_and_checkpoints_interruption(tmp_path: Path) -> None:
    path = tmp_path / "poems.json"
    with pytest.raises(RuntimeError, match="bug"):
        collector(FakeProvider([RuntimeError("bug")]), path).collect((source(),))

    checkpoint = load_checkpoint(path)
    assert checkpoint.state is CheckpointState.INTERRUPTED
    assert checkpoint.outcomes == ()


def test_resume_loads_checkpoint_skips_match_and_retries_missing(tmp_path: Path) -> None:
    first, second = source(), source("Poema ausente")
    path = tmp_path / "poems.json"
    collector(FakeProvider([(candidate(),), ()]), path).collect((first, second))
    retry_provider = FakeProvider([()])

    result = collector(retry_provider, path).collect((first, second))

    assert retry_provider.calls == [second.source_id]
    assert [outcome.source.source_id for outcome in result.outcomes] == [
        first.source_id,
        second.source_id,
    ]
    assert result.state is CheckpointState.COMPLETE


def test_resume_skips_permanent_error_and_retries_retryable_error(tmp_path: Path) -> None:
    permanent, transient = source(), source("Poema transitorio")
    path = tmp_path / "poems.json"
    collector(
        FakeProvider(
            [
                ProviderError("invalid", retryable=False),
                ProviderError("timeout", retryable=True),
            ]
        ),
        path,
    ).collect((permanent, transient))
    provider = FakeProvider([()])

    result = collector(provider, path).collect((permanent, transient))

    assert provider.calls == [transient.source_id]
    assert result.outcomes[0].status is OutcomeStatus.ERROR
    assert result.outcomes[1].status is OutcomeStatus.NOT_FOUND


def test_resume_rejects_changed_source_identity_for_existing_id(tmp_path: Path) -> None:
    original = source()
    path = tmp_path / "poems.json"
    collector(FakeProvider([(candidate(),)]), path).collect((original,))
    changed = replace(original, title="A different poem")

    with pytest.raises(ValueError, match="source_id changed meaning"):
        collector(FakeProvider([]), path).collect((changed,))


def test_new_source_is_merged_without_losing_prior_match(tmp_path: Path) -> None:
    first = source()
    path = tmp_path / "poems.json"
    collector(FakeProvider([(candidate(),)]), path).collect((first,))
    second = source("Poema nuevo")
    provider = FakeProvider([()])

    result = collector(provider, path).collect((first, second))

    assert provider.calls == [second.source_id]
    assert len(result.outcomes) == 2


def test_keyboard_interrupt_preserves_prior_and_pending_source_identity(tmp_path: Path) -> None:
    first, second = source(), source("Poema dos")
    path = tmp_path / "poems.json"
    provider = FakeProvider([(candidate(),), KeyboardInterrupt("stop")])

    with pytest.raises(KeyboardInterrupt, match="stop"):
        collector(provider, path).collect((first, second))

    checkpoint = load_checkpoint(path)
    assert checkpoint.state is CheckpointState.INTERRUPTED
    assert checkpoint.sources == (first, second)
    assert [outcome.source.source_id for outcome in checkpoint.outcomes] == [first.source_id]


def test_poem_manifest_round_trip(tmp_path: Path) -> None:
    manifest = PoemManifest((source(),))
    path = tmp_path / "poems-manifest.json"
    write_poem_manifest(manifest, path)

    assert load_poem_manifest(path) == manifest
