"""Canonical, dependency-free text normalization and edit metrics.

The project intentionally has one scoring policy.  Normalized text is NFC,
lowercase, and contains only Unicode letters/numbers separated by single
spaces.  Accents and ``ñ`` are preserved.  Word scoring splits on spaces;
character scoring removes spaces.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

NORMALIZATION_POLICY = "nfc-lower-unicode-alnum-whitespace-v1;cer-excludes-spaces"


@dataclass(frozen=True, slots=True)
class AlignmentCounts:
    """The error counts for one minimum-edit alignment."""

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float | None:
        """Return the error rate, or ``None`` for an empty reference."""

        if self.reference_length == 0:
            return None
        return self.errors / self.reference_length


@dataclass(frozen=True, slots=True)
class TextScore:
    normalized_reference: str
    normalized_hypothesis: str
    words: AlignmentCounts
    characters: AlignmentCounts

    @property
    def wer(self) -> float | None:
        return self.words.rate

    @property
    def cer(self) -> float | None:
        return self.characters.rate


def normalize_text(text: str) -> str:
    """Normalize text without erasing Spanish orthographic distinctions."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).lower()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def word_tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_text(text).split())


def character_tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize_text(text).replace(" ", ""))


def alignment_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> AlignmentCounts:
    """Compute exact S/D/I counts for a minimum Levenshtein alignment.

    Several alignments can have the same edit distance.  To keep counts stable,
    ties prefer substitution, then deletion, then insertion.
    """

    # Each cell stores (distance, substitutions, deletions, insertions).
    previous = [(index, 0, 0, index) for index in range(len(hypothesis) + 1)]
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [(reference_index, 0, reference_index, 0)]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            if reference_token == hypothesis_token:
                current.append(previous[hypothesis_index - 1])
                continue

            diagonal = previous[hypothesis_index - 1]
            above = previous[hypothesis_index]
            left = current[hypothesis_index - 1]
            candidates = (
                (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                (above[0] + 1, above[1], above[2] + 1, above[3]),
                (left[0] + 1, left[1], left[2], left[3] + 1),
            )
            current.append(min(candidates, key=lambda item: item[0]))
        previous = current

    _, substitutions, deletions, insertions = previous[-1]
    return AlignmentCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_length=len(reference),
    )


def score_text(reference: str, hypothesis: str) -> TextScore:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    return TextScore(
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        words=alignment_counts(normalized_reference.split(), normalized_hypothesis.split()),
        characters=alignment_counts(
            tuple(normalized_reference.replace(" ", "")),
            tuple(normalized_hypothesis.replace(" ", "")),
        ),
    )


def sum_counts(counts: Iterable[AlignmentCounts]) -> AlignmentCounts:
    substitutions = deletions = insertions = reference_length = 0
    for item in counts:
        substitutions += item.substitutions
        deletions += item.deletions
        insertions += item.insertions
        reference_length += item.reference_length
    return AlignmentCounts(substitutions, deletions, insertions, reference_length)


def corpus_score(scores: Iterable[TextScore]) -> TextScore:
    """Aggregate corpus rates from summed edit counts and denominators."""

    materialized = tuple(scores)
    return TextScore(
        normalized_reference="",
        normalized_hypothesis="",
        words=sum_counts(score.words for score in materialized),
        characters=sum_counts(score.characters for score in materialized),
    )
