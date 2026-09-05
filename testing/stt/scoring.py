"""Dependency-free ASR scoring and grouped percentile bootstrap."""

from __future__ import annotations

import random
import unicodedata
from typing import Dict, List, Optional, Sequence

NORMALIZATION = "nfc-lower-unicode-alnum-whitespace-v1; CER excludes spaces"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(
        "".join(c if c.isalnum() else " " for c in text).split()
    )


def alignment_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, int]:
    """Minimum edit alignment, O(hypothesis length) memory.

    Ties prefer substitution, deletion, then insertion. Store counts along
    the chosen path; no full alignment matrix is needed for corpus scoring.
    """
    previous = [(0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
                continue
            s, d, ins = previous[j - 1]
            deletion = previous[j]
            insertion = current[j - 1]
            current.append(min(
                ( (s + 1, d, ins),
                  (deletion[0], deletion[1] + 1, deletion[2]),
                  (insertion[0], insertion[1], insertion[2] + 1) ),
                key=sum,
            ))
        previous = current
    s, d, ins = previous[-1]
    return {"substitutions": s, "deletions": d, "insertions": ins,
            "errors": s + d + ins, "reference_length": len(reference)}


def score_text(reference: str, hypothesis: str) -> Dict[str, object]:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    result: Dict[str, object] = {"normalized_reference": ref, "normalized_transcript": hyp}
    for prefix, a, b in (("word", ref.split(), hyp.split()),
                         ("char", list(ref.replace(" ", "")), list(hyp.replace(" ", "")))):
        counts = alignment_counts(a, b)
        result.update({prefix + "_" + key: value for key, value in counts.items()})
        result["wer" if prefix == "word" else "cer"] = (
            counts["errors"] / counts["reference_length"] if counts["reference_length"] else None
        )
    return result


def corpus_rate(rows: List[dict], prefix: str) -> Optional[float]:
    denominator = sum(row[prefix + "_reference_length"] for row in rows)
    return sum(row[prefix + "_errors"] for row in rows) / denominator if denominator else None


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_intervals(rows: List[dict], replicates: int, seed: int) -> dict:
    if replicates < 0:
        raise ValueError("Bootstrap replicates cannot be negative")
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    result = {"groups": len(groups), "replicates": replicates, "seed": seed,
              "method": "grouped-percentile", "confidence": 0.95,
              "wer": None, "cer": None, "valid_replicates": {"wer": 0, "cer": 0}}
    if len(groups) < 2 or replicates == 0:
        result["reason"] = "Need at least two independent groups and positive replicates"
        return result
    # Preaggregate groups: runtime depends on groups, not utterances per group.
    totals = [{key: sum(row[key] for row in group)
               for key in ("word_errors", "word_reference_length", "char_errors", "char_reference_length")}
              for group in groups.values()]
    rng = random.Random(seed)
    draws: Dict[str, List[float]] = {"wer": [], "cer": []}
    for _ in range(replicates):
        sampled = rng.choices(totals, k=len(totals))
        for metric, prefix in (("wer", "word"), ("cer", "char")):
            rate = corpus_rate(sampled, prefix)
            if rate is not None:
                draws[metric].append(rate)
    for metric, values in draws.items():
        result["valid_replicates"][metric] = len(values)
        if values:
            result[metric] = [percentile(values, 0.025), percentile(values, 0.975)]
    return result
