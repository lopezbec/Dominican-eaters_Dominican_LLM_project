from typing import Tuple


class TextSimilarityMetrics:
    
    def __init__(self, text_normalizer=None):
        self.text_normalizer = text_normalizer
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
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
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        
        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return wer
    
    def calculate_levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self.calculate_levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def calculate_char_similarity(self, reference: str, hypothesis: str) -> float:
        if self.text_normalizer:
            ref_normalized = self.text_normalizer.normalize_text(reference)
            hyp_normalized = self.text_normalizer.normalize_text(hypothesis)
        else:
            ref_normalized = reference
            hyp_normalized = hypothesis
        
        if not ref_normalized:
            return 0.0 if hyp_normalized else 1.0
        
        distance = self.calculate_levenshtein_distance(ref_normalized, hyp_normalized)
        max_len = max(len(ref_normalized), len(hyp_normalized))
        
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        return similarity
    
    def calculate_jaccard_similarity(self, reference: str, hypothesis: str) -> float:
        if self.text_normalizer:
            ref_normalized = self.text_normalizer.normalize_text(reference)
            hyp_normalized = self.text_normalizer.normalize_text(hypothesis)
        else:
            ref_normalized = reference
            hyp_normalized = hypothesis
        
        ref_words = set(ref_normalized.split())
        hyp_words = set(hyp_normalized.split())
        
        if not ref_words and not hyp_words:
            return 1.0
        
        if not ref_words or not hyp_words:
            return 0.0
        
        intersection = ref_words.intersection(hyp_words)
        union = ref_words.union(hyp_words)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard
    
    def calculate_cosine_similarity(self, reference: str, hypothesis: str) -> float:
        if self.text_normalizer:
            ref_normalized = self.text_normalizer.normalize_text(reference)
            hyp_normalized = self.text_normalizer.normalize_text(hypothesis)
        else:
            ref_normalized = reference
            hyp_normalized = hypothesis
        
        ref_words = ref_normalized.split()
        hyp_words = hyp_normalized.split()
        
        all_words = list(set(ref_words + hyp_words))
        
        if not all_words:
            return 1.0
        
        ref_vec = [ref_words.count(word) for word in all_words]
        hyp_vec = [hyp_words.count(word) for word in all_words]
        
        dot_product = sum(r * h for r, h in zip(ref_vec, hyp_vec))
        ref_magnitude = sum(r * r for r in ref_vec) ** 0.5
        hyp_magnitude = sum(h * h for h in hyp_vec) ** 0.5
        
        if ref_magnitude == 0 or hyp_magnitude == 0:
            return 0.0
        
        cosine = dot_product / (ref_magnitude * hyp_magnitude)
        return cosine
