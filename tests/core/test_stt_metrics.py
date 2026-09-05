from __future__ import annotations

import math
from itertools import product

import pytest

from dominican_eaters.evaluation.asr import (
    Recognition,
    ReferenceSample,
    evaluate_asr,
    grouped_bootstrap_intervals,
)
from dominican_eaters.text_metrics import (
    AlignmentCounts,
    character_tokens,
    corpus_score,
    normalize_text,
    score_text,
    word_tokens,
)


def test_spanish_normalization_is_nfc_lowercase_and_punctuation_free() -> None:
    decomposed = "A\u0301RBOL, NIÑA... ¿QUÉ tal? pingüino 42"
    assert normalize_text(decomposed) == "árbol niña qué tal pingüino 42"
    assert word_tokens(decomposed) == ("árbol", "niña", "qué", "tal", "pingüino", "42")
    assert "".join(character_tokens("¡Qué, sí!")) == "quésí"
    assert normalize_text("ano año si sí") == "ano año si sí"


def test_normalization_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="string"):
        normalize_text(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        (("a", "b", "c"), ("a", "b", "c"), AlignmentCounts(0, 0, 0, 3)),
        (("a", "b", "c"), ("a", "x", "c"), AlignmentCounts(1, 0, 0, 3)),
        (("a", "b", "c"), ("a", "c"), AlignmentCounts(0, 1, 0, 3)),
        (("a", "c"), ("a", "b", "c"), AlignmentCounts(0, 0, 1, 2)),
        ((), ("extra",), AlignmentCounts(0, 0, 1, 0)),
        (("missing",), (), AlignmentCounts(0, 1, 0, 1)),
    ],
)
def test_alignment_reports_exact_sdi_counts(reference, hypothesis, expected) -> None:
    from dominican_eaters.text_metrics import alignment_counts

    assert alignment_counts(reference, hypothesis) == expected
    assert expected.errors == expected.substitutions + expected.deletions + expected.insertions


def test_alignment_distance_exhaustively_matches_independent_dynamic_program() -> None:
    from dominican_eaters.text_metrics import alignment_counts

    def reference_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
        matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        for row in range(len(left) + 1):
            matrix[row][0] = row
        for column in range(len(right) + 1):
            matrix[0][column] = column
        for row, left_token in enumerate(left, start=1):
            for column, right_token in enumerate(right, start=1):
                matrix[row][column] = min(
                    matrix[row - 1][column] + 1,
                    matrix[row][column - 1] + 1,
                    matrix[row - 1][column - 1] + (left_token != right_token),
                )
        return matrix[-1][-1]

    sequences = [sequence for length in range(4) for sequence in product(("a", "b"), repeat=length)]
    for reference in sequences:
        for hypothesis in sequences:
            counts = alignment_counts(reference, hypothesis)
            assert counts.errors == reference_distance(reference, hypothesis)
            assert counts.reference_length == len(reference)


def test_score_text_distinguishes_wer_and_cer() -> None:
    score = score_text("casa", "caso")
    assert score.words == AlignmentCounts(1, 0, 0, 1)
    assert score.characters == AlignmentCounts(1, 0, 0, 4)
    assert score.wer == 1.0
    assert score.cer == 0.25


def test_empty_reference_rate_is_undefined_but_insertions_are_preserved() -> None:
    empty = score_text("", "hola mundo")
    assert empty.words == AlignmentCounts(0, 0, 2, 0)
    assert empty.characters == AlignmentCounts(0, 0, 9, 0)
    assert empty.wer is None
    assert empty.cer is None


def test_empty_hypothesis_is_all_deletions() -> None:
    empty = score_text("hola mi gente", "")
    assert empty.words == AlignmentCounts(0, 3, 0, 3)
    assert empty.wer == 1.0


def test_corpus_rates_use_summed_counts_instead_of_mean_file_rates() -> None:
    short = score_text("uno", "dos")
    long = score_text("a b c d e f g h i", "a b c d e f g h i")
    aggregate = corpus_score((short, long))
    assert aggregate.words.reference_length == 10
    assert aggregate.words.errors == 1
    assert aggregate.wer == 0.1
    assert aggregate.wer != (short.wer + long.wer) / 2  # type: ignore[operator]


def _complete_report():
    references = [
        ReferenceSample("a", "speaker-1", "hola mundo"),
        ReferenceSample("b", "speaker-1", "buen día"),
        ReferenceSample("c", "speaker-2", "cómo estás"),
    ]
    outputs = [
        Recognition("a", "hola mundo"),
        Recognition("b", "buenos días"),
        Recognition("c", "cómo estás"),
    ]
    return evaluate_asr(references, outputs, bootstrap_replicates=250, bootstrap_seed=9)


def test_grouped_bootstrap_is_deterministic_and_group_based() -> None:
    first = _complete_report()
    second = _complete_report()
    assert first.bootstrap == second.bootstrap
    assert first.bootstrap.group_count == 2
    assert first.bootstrap.requested_replicates == 250
    assert first.bootstrap.valid_wer_replicates == 250
    assert first.bootstrap.valid_cer_replicates == 250
    assert first.bootstrap.wer is not None
    assert 0 <= first.bootstrap.wer.lower <= first.bootstrap.wer.upper

    # Ordering utterances cannot change seeded draws because groups are sorted.
    reversed_bootstrap = grouped_bootstrap_intervals(
        tuple(reversed(first.utterances)), replicates=250, seed=9
    )
    assert reversed_bootstrap == first.bootstrap


def test_bootstrap_requires_two_groups_and_valid_parameters() -> None:
    result = evaluate_asr(
        [ReferenceSample("a", "only", "hola")],
        [Recognition("a", "hola")],
        bootstrap_replicates=10,
    ).bootstrap
    assert result.wer is None
    assert result.reason == "At least two independent non-silence groups are required"

    with pytest.raises(ValueError, match="replicates"):
        grouped_bootstrap_intervals((), replicates=-1)
    with pytest.raises(ValueError, match="confidence"):
        grouped_bootstrap_intervals((), confidence=1.0)


def test_coverage_report_scores_successful_empty_output_and_separates_silence() -> None:
    references = [
        ReferenceSample("ok-empty", "g1", "hola"),
        ReferenceSample("failed", "g2", "adiós"),
        ReferenceSample("missing", "g3", "bien"),
        ReferenceSample("silence", "g4", ""),
    ]
    outputs = [
        Recognition("ok-empty", ""),
        Recognition("failed", None, status="failed", error="out of memory"),
        Recognition("silence", "ruido"),
        Recognition("unexpected", "texto"),
    ]
    report = evaluate_asr(references, outputs, bootstrap_replicates=0)

    assert report.coverage.expected == 4
    assert report.coverage.scored == 2
    assert report.coverage.unscored == 2
    assert report.coverage.rate == 0.5
    assert not report.coverage.complete
    assert report.coverage.missing == ("missing",)
    assert report.coverage.failed == ("failed",)
    assert report.coverage.unexpected == ("unexpected",)
    assert report.speech_corpus.words == AlignmentCounts(0, 1, 0, 1)
    assert report.speech_corpus.wer == 1.0
    assert len(report.silence_controls) == 1
    assert report.silence_controls[0].score.words.insertions == 1
    assert report.silence_controls[0].score.wer is None


def test_all_empty_reference_corpus_has_undefined_rates() -> None:
    report = evaluate_asr(
        [ReferenceSample("silence", "g", "")],
        [Recognition("silence", "")],
        bootstrap_replicates=0,
    )
    assert report.speech_corpus.wer is None
    assert report.speech_corpus.cer is None
    assert len(report.silence_controls) == 1


def test_duplicate_ids_and_invalid_recognition_states_are_rejected() -> None:
    reference = ReferenceSample("x", "g", "hola")
    with pytest.raises(ValueError, match="Duplicate reference"):
        evaluate_asr([reference, reference], [])
    with pytest.raises(ValueError, match="Duplicate recognition"):
        evaluate_asr([reference], [Recognition("x", "a"), Recognition("x", "b")])
    with pytest.raises(ValueError, match="must contain text"):
        Recognition("x", None)
    with pytest.raises(ValueError, match="cannot contain"):
        Recognition("x", "partial", status="failed")
    with pytest.raises(ValueError, match="status"):
        Recognition("x", "partial", status="invalid")  # type: ignore[arg-type]


def test_counts_are_finite_and_rates_can_exceed_one() -> None:
    score = score_text("hola", "hola uno dos")
    assert score.wer == 2.0
    assert math.isfinite(score.wer)
