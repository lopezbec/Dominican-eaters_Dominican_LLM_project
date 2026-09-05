"""Network-independent orchestration for poem recitation collection."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from dominican_eaters.data import exclusive_file_lock

from .artifacts import CheckpointState, CollectionCheckpoint, load_checkpoint, write_checkpoint
from .matching import select_candidate
from .models import ContentKind, OutcomeStatus, PoemOutcome, PoemSource, VideoCandidate
from .ports import ProviderError, RecitationQuery, RecitationSearch

Clock = Callable[[], datetime]


class PoemCollector:
    def __init__(
        self,
        *,
        provider: RecitationSearch,
        checkpoint_path: Path,
        minimum_score: float = 0.5,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        if not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be between zero and one")
        self._provider = provider
        self._checkpoint_path = checkpoint_path
        self._minimum_score = minimum_score
        self._clock = clock

    def collect(
        self,
        sources: Sequence[PoemSource],
        *,
        previous: CollectionCheckpoint | None = None,
        force: bool = False,
    ) -> CollectionCheckpoint:
        with exclusive_file_lock(self._checkpoint_path, label="poem collection checkpoint"):
            return self._collect_locked(sources, previous=previous, force=force)

    def _collect_locked(
        self,
        sources: Sequence[PoemSource],
        *,
        previous: CollectionCheckpoint | None,
        force: bool,
    ) -> CollectionCheckpoint:
        current = tuple(sources)
        self._validate_sources(current)
        if previous is None and self._checkpoint_path.exists():
            previous = load_checkpoint(self._checkpoint_path)
        if previous is not None:
            previous_by_id = {source.source_id: source for source in previous.sources}
            changed = sorted(
                source.source_id
                for source in current
                if source.source_id in previous_by_id and previous_by_id[source.source_id] != source
            )
            if changed:
                raise ValueError(
                    "source_id changed meaning in existing checkpoint: " + ", ".join(changed)
                )
        current_by_id = {source.source_id: source for source in current}
        prior = (
            {
                outcome.source.source_id: replace(
                    outcome, source=current_by_id[outcome.source.source_id]
                )
                for outcome in previous.outcomes
                if outcome.source.source_id in current_by_id
            }
            if previous
            else {}
        )
        outcomes = dict(prior)
        self._persist(current, outcomes, CheckpointState.RUNNING)

        try:
            for source in current:
                old = prior.get(source.source_id)
                if not force and old is not None and old.resumable:
                    outcome = replace(old, source=source)
                else:
                    outcome = self._collect_one(source)
                outcomes[source.source_id] = outcome
                self._persist(current, outcomes, CheckpointState.RUNNING)
        except BaseException:
            self._persist(current, outcomes, CheckpointState.INTERRUPTED)
            raise

        return self._persist(current, outcomes, CheckpointState.COMPLETE)

    def _collect_one(self, source: PoemSource) -> PoemOutcome:
        attempted_at = self._timestamp()
        try:
            candidates = self._provider.search(RecitationQuery(source))
            if not isinstance(candidates, tuple) or not all(
                isinstance(candidate, VideoCandidate) for candidate in candidates
            ):
                raise TypeError("provider search must return a tuple of VideoCandidate values")
            selected = select_candidate(source, candidates, minimum_score=self._minimum_score)
            if selected is None:
                return PoemOutcome(source, OutcomeStatus.NOT_FOUND, attempted_at)
            status = (
                OutcomeStatus.PARTIAL
                if selected.content_kind is ContentKind.FRAGMENT
                else OutcomeStatus.FOUND
            )
            return PoemOutcome(
                source=source,
                status=status,
                attempted_at=attempted_at,
                candidate=selected.candidate,
                content_kind=selected.content_kind,
                match_score=selected.score,
                match_reasons=selected.reasons,
            )
        except ProviderError as exc:
            return PoemOutcome(
                source=source,
                status=OutcomeStatus.ERROR,
                attempted_at=attempted_at,
                error_type=type(exc).__name__,
                error_message=str(exc) or type(exc).__name__,
                error_retryable=exc.retryable,
            )

    def _persist(
        self,
        sources: tuple[PoemSource, ...],
        outcomes: dict[str, PoemOutcome],
        state: CheckpointState,
    ) -> CollectionCheckpoint:
        checkpoint = CollectionCheckpoint(
            sources=sources,
            outcomes=tuple(
                outcomes[source.source_id] for source in sources if source.source_id in outcomes
            ),
            state=state,
            updated_at=self._timestamp(),
        )
        write_checkpoint(self._checkpoint_path, checkpoint)
        return checkpoint

    def _timestamp(self) -> str:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _validate_sources(sources: tuple[PoemSource, ...]) -> None:
        if not sources:
            raise ValueError("sources must not be empty")
        if not all(isinstance(source, PoemSource) for source in sources):
            raise TypeError("sources must contain only PoemSource values")
        ids = [source.source_id for source in sources]
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        if duplicates:
            raise ValueError(f"duplicate source_id values: {', '.join(duplicates)}")
