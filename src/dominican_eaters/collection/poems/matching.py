"""Pure, deterministic poem candidate classification and ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ContentKind, PoemSource, VideoCandidate, normalized_identity

_STOP_WORDS = frozenset({"a", "al", "de", "del", "el", "en", "la", "las", "los", "un", "una", "y"})


def classify_content(title: str) -> ContentKind:
    normalized = normalized_identity(title)
    # A fragment marker is outcome-critical, so it takes precedence over all presentation styles.
    if any(word in normalized for word in ("fragmento", "extracto", "parcial")):
        return ContentKind.FRAGMENT
    if any(word in normalized for word in ("recitacion", "recita", "recitando", "declamacion")):
        return ContentKind.RECITATION
    if any(word in normalized for word in ("dramatizacion", "drama", "teatral", "teatro")):
        return ContentKind.DRAMATIZATION
    if any(word in normalized for word in ("lectura", "leyendo")):
        return ContentKind.READING
    if any(word in normalized for word in ("performance", "presentacion", "actuacion")):
        return ContentKind.PERFORMANCE
    if any(word in normalized for word in ("audiopoesia", "audio poesia")):
        return ContentKind.AUDIO_POETRY
    return ContentKind.OTHER


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        word
        for word in normalized_identity(value).split()
        if word not in _STOP_WORDS and len(word) > 1
    )


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: VideoCandidate
    content_kind: ContentKind
    score: float
    reasons: tuple[str, ...]


def rank_candidate(source: PoemSource, candidate: VideoCandidate) -> RankedCandidate:
    candidate_tokens = _tokens(candidate.title)
    title_tokens = _tokens(source.title)
    author_tokens = _tokens(source.author)
    title_overlap = (
        len(title_tokens & candidate_tokens) / len(title_tokens) if title_tokens else 0.0
    )
    author_overlap = (
        len(author_tokens & candidate_tokens) / len(author_tokens) if author_tokens else 0.0
    )
    kind = classify_content(candidate.title)
    kind_signal = 0.0 if kind is ContentKind.OTHER else 1.0
    score = round(min(1.0, 0.70 * title_overlap + 0.25 * author_overlap + 0.05 * kind_signal), 6)
    reasons = (
        f"title_overlap={title_overlap:.3f}",
        f"author_overlap={author_overlap:.3f}",
        f"content_kind={kind.value}",
    )
    return RankedCandidate(candidate, kind, score, reasons)


def select_candidate(
    source: PoemSource,
    candidates: tuple[VideoCandidate, ...],
    *,
    minimum_score: float = 0.5,
) -> RankedCandidate | None:
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be between zero and one")
    ranked = sorted(
        (rank_candidate(source, candidate) for candidate in candidates),
        key=lambda item: (-item.score, item.candidate.video_id),
    )
    if not ranked or ranked[0].score < minimum_score:
        return None
    return ranked[0]
