"""Coverage-aware ASR evaluation using the project's canonical text policy."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dominican_eaters.text_metrics import (
    NORMALIZATION_POLICY,
    TextScore,
    corpus_score,
    score_text,
)


@dataclass(frozen=True, slots=True)
class ReferenceSample:
    sample_id: str
    group_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not self.group_id:
            raise ValueError("group_id cannot be empty")


@dataclass(frozen=True, slots=True)
class Recognition:
    sample_id: str
    text: str | None
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if self.status not in ("ok", "failed"):
            raise ValueError("status must be 'ok' or 'failed'")
        if self.status == "ok" and self.text is None:
            raise ValueError("A successful recognition must contain text; use '' for empty output")
        if self.status == "failed" and self.text is not None:
            raise ValueError("A failed recognition cannot contain transcript text")


@dataclass(frozen=True, slots=True)
class UtteranceResult:
    sample_id: str
    group_id: str
    reference: str
    hypothesis: str
    score: TextScore

    @property
    def is_silence_control(self) -> bool:
        return self.score.words.reference_length == 0


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    expected: int
    scored: int
    missing: tuple[str, ...]
    failed: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def unscored(self) -> int:
        return len(self.missing) + len(self.failed)

    @property
    def rate(self) -> float | None:
        return self.scored / self.expected if self.expected else None

    @property
    def complete(self) -> bool:
        return self.expected > 0 and self.unscored == 0


@dataclass(frozen=True, slots=True)
class PercentileInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    group_count: int
    requested_replicates: int
    valid_wer_replicates: int
    valid_cer_replicates: int
    seed: int
    confidence: float
    wer: PercentileInterval | None
    cer: PercentileInterval | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ASREvaluationReport:
    normalization_policy: str
    coverage: CoverageSummary
    utterances: tuple[UtteranceResult, ...]
    speech_corpus: TextScore
    silence_controls: tuple[UtteranceResult, ...]
    bootstrap: BootstrapReport


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def grouped_bootstrap_intervals(
    utterances: Sequence[UtteranceResult],
    *,
    replicates: int = 2_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapReport:
    """Resample whole groups and recompute corpus WER/CER from counts."""

    if replicates < 0:
        raise ValueError("replicates cannot be negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    grouped: dict[str, list[TextScore]] = {}
    for utterance in utterances:
        if not utterance.is_silence_control:
            grouped.setdefault(utterance.group_id, []).append(utterance.score)

    group_scores = tuple(corpus_score(grouped[group_id]) for group_id in sorted(grouped))
    reason: str | None = None
    if len(group_scores) < 2:
        reason = "At least two independent non-silence groups are required"
    elif replicates == 0:
        reason = "Bootstrap was disabled"
    if reason is not None:
        return BootstrapReport(
            group_count=len(group_scores),
            requested_replicates=replicates,
            valid_wer_replicates=0,
            valid_cer_replicates=0,
            seed=seed,
            confidence=confidence,
            wer=None,
            cer=None,
            reason=reason,
        )

    rng = random.Random(seed)
    wer_draws: list[float] = []
    cer_draws: list[float] = []
    for _ in range(replicates):
        draw = tuple(
            group_scores[rng.randrange(len(group_scores))] for _ in range(len(group_scores))
        )
        aggregate = corpus_score(draw)
        if aggregate.wer is not None:
            wer_draws.append(aggregate.wer)
        if aggregate.cer is not None:
            cer_draws.append(aggregate.cer)

    tail = (1 - confidence) / 2

    def interval(values: list[float]) -> PercentileInterval | None:
        if not values:
            return None
        return PercentileInterval(
            lower=_percentile(values, tail),
            upper=_percentile(values, 1 - tail),
        )

    return BootstrapReport(
        group_count=len(group_scores),
        requested_replicates=replicates,
        valid_wer_replicates=len(wer_draws),
        valid_cer_replicates=len(cer_draws),
        seed=seed,
        confidence=confidence,
        wer=interval(wer_draws),
        cer=interval(cer_draws),
    )


def evaluate_asr(
    references: Sequence[ReferenceSample],
    recognitions: Sequence[Recognition],
    *,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 42,
) -> ASREvaluationReport:
    """Score successful outputs and report missing, failed, and unexpected IDs."""

    reference_by_id: dict[str, ReferenceSample] = {}
    for reference in references:
        if reference.sample_id in reference_by_id:
            raise ValueError(f"Duplicate reference sample_id: {reference.sample_id}")
        reference_by_id[reference.sample_id] = reference

    recognition_by_id: dict[str, Recognition] = {}
    for recognition in recognitions:
        if recognition.sample_id in recognition_by_id:
            raise ValueError(f"Duplicate recognition sample_id: {recognition.sample_id}")
        recognition_by_id[recognition.sample_id] = recognition

    missing: list[str] = []
    failed: list[str] = []
    scored: list[UtteranceResult] = []
    for sample_id, reference in reference_by_id.items():
        matched_recognition = recognition_by_id.get(sample_id)
        if matched_recognition is None:
            missing.append(sample_id)
        elif matched_recognition.status == "failed":
            failed.append(sample_id)
        else:
            assert matched_recognition.text is not None
            scored.append(
                UtteranceResult(
                    sample_id=sample_id,
                    group_id=reference.group_id,
                    reference=reference.text,
                    hypothesis=matched_recognition.text,
                    score=score_text(reference.text, matched_recognition.text),
                )
            )

    expected_ids = set(reference_by_id)
    unexpected = sorted(set(recognition_by_id) - expected_ids)
    silence_controls = tuple(row for row in scored if row.is_silence_control)
    speech_rows = tuple(row for row in scored if not row.is_silence_control)
    coverage = CoverageSummary(
        expected=len(reference_by_id),
        scored=len(scored),
        missing=tuple(sorted(missing)),
        failed=tuple(sorted(failed)),
        unexpected=tuple(unexpected),
    )
    return ASREvaluationReport(
        normalization_policy=NORMALIZATION_POLICY,
        coverage=coverage,
        utterances=tuple(scored),
        speech_corpus=corpus_score(row.score for row in speech_rows),
        silence_controls=silence_controls,
        bootstrap=grouped_bootstrap_intervals(
            speech_rows,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
    )
