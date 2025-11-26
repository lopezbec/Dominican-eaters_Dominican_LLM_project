from difflib import SequenceMatcher
from typing import Tuple
from ..models.content import AlignmentQuality


def calculate_similarity(text1: str, text2: str, normalize_func=None) -> float:
    if normalize_func:
        text1 = normalize_func(text1)
        text2 = normalize_func(text2)
    
    return SequenceMatcher(None, text1, text2).ratio()


def calculate_wer(reference: str, hypothesis: str, normalize_func=None) -> float:
    if normalize_func:
        reference = normalize_func(reference)
        hypothesis = normalize_func(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,
                    d[i][j-1] + 1,
                    d[i-1][j-1] + 1
                )
    
    if len(ref_words) == 0:
        return 1.0
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def classify_alignment_quality(similarity: float) -> AlignmentQuality:
    if similarity >= 0.8:
        return AlignmentQuality.EXCELLENT
    elif similarity >= 0.6:
        return AlignmentQuality.GOOD
    elif similarity >= 0.4:
        return AlignmentQuality.FAIR
    else:
        return AlignmentQuality.POOR


def calculate_metrics(reference: str, hypothesis: str, normalize_func=None) -> Tuple[float, float, AlignmentQuality]:
    similarity = calculate_similarity(reference, hypothesis, normalize_func)
    wer = calculate_wer(reference, hypothesis, normalize_func)
    quality = classify_alignment_quality(similarity)
    
    return similarity, wer, quality
