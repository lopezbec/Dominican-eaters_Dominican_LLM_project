#!/usr/bin/env python3
"""Evaluate ASR outputs against Genius lyrics and rank models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.clients.genius_client import GeniusAPIClient


def normalize_text(text: str) -> str:
    lowered = text.lower()
    stripped = re.sub(r"\[[^\]]*\]|\([^\)]*\)", " ", lowered)
    alnum_only = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", stripped)
    return re.sub(r"\s+", " ", alnum_only).strip()


def words(text: str) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = words(reference)
    hyp = words(hypothesis)
    if not ref:
        return 1.0 if hyp else 0.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i, ref_word in enumerate(ref, start=1):
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
    return d[len(ref)][len(hyp)] / len(ref)


def char_similarity(reference: str, hypothesis: str) -> float:
    return SequenceMatcher(
        None, normalize_text(reference), normalize_text(hypothesis)
    ).ratio()


def jaccard_similarity(reference: str, hypothesis: str) -> float:
    ref = set(words(reference))
    hyp = set(words(hypothesis))
    union = ref | hyp
    return (len(ref & hyp) / len(union)) if union else 0.0


def cosine_similarity(reference: str, hypothesis: str) -> float:
    ref_counts: Dict[str, int] = {}
    hyp_counts: Dict[str, int] = {}
    for token in words(reference):
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in words(hypothesis):
        hyp_counts[token] = hyp_counts.get(token, 0) + 1

    keys = set(ref_counts) | set(hyp_counts)
    if not keys:
        return 0.0

    dot = sum(ref_counts.get(k, 0) * hyp_counts.get(k, 0) for k in keys)
    ref_norm = math.sqrt(sum(v * v for v in ref_counts.values()))
    hyp_norm = math.sqrt(sum(v * v for v in hyp_counts.values()))
    if ref_norm == 0 or hyp_norm == 0:
        return 0.0
    return dot / (ref_norm * hyp_norm)


def filename_to_query(audio_filename: str) -> str:
    stem = Path(audio_filename).stem
    stem = re.sub(r"^lyrics-eater_\d+_", "", stem)
    stem = stem.replace("_", " ")
    stem = re.sub(
        r"\b(official|video|audio|lyric|lyrics|remastered|visualizer|hd|playlist)\b",
        " ",
        stem,
        flags=re.IGNORECASE,
    )
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def overlap_ratio(query: str, title_artist: str) -> float:
    q = set(words(query))
    t = set(words(title_artist))
    if not q:
        return 0.0
    return len(q & t) / len(q)


@dataclass
class GeniusReference:
    query: str
    genius_url: str
    title: str
    artist: str
    lyrics: str


class GeniusReferenceStore:
    def __init__(
        self,
        access_token: str,
        cache_path: Path,
        refresh_cache: bool,
        min_overlap: float,
    ):
        self._client = GeniusAPIClient(access_token)
        self._cache_path = cache_path
        self._min_overlap = min_overlap
        self._cache = {} if refresh_cache else self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, str]]:
        if not self._cache_path.exists():
            return {}
        return json.loads(self._cache_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_reference(self, audio_filename: str) -> Optional[GeniusReference]:
        if audio_filename in self._cache:
            row = self._cache[audio_filename]
            return GeniusReference(**row)

        query = filename_to_query(audio_filename)
        hits = self._client.search(query, per_page=5)
        if not hits:
            return None

        best = None
        best_score = -1.0
        for hit in hits:
            title = hit.get("title") or ""
            artist = hit.get("artist") or ""
            score = overlap_ratio(query, f"{title} {artist}")
            if score > best_score:
                best = hit
                best_score = score

        if best is None or best_score < self._min_overlap:
            return None

        url = best.get("url")
        if not url:
            return None

        lyrics = self._client.scrape_lyrics(url)
        if not lyrics:
            return None

        reference = GeniusReference(
            query=query,
            genius_url=url,
            title=best.get("title") or "",
            artist=best.get("artist") or "",
            lyrics=lyrics,
        )
        self._cache[audio_filename] = {
            "query": reference.query,
            "genius_url": reference.genius_url,
            "title": reference.title,
            "artist": reference.artist,
            "lyrics": reference.lyrics,
        }
        return reference


def load_transcript(path: Path) -> Optional[Tuple[str, str]]:
    row = json.loads(path.read_text(encoding="utf-8"))
    transcript = (row.get("transcript") or "").strip()
    file_name = row.get("file") or f"{path.stem}.m4a"
    if not transcript:
        return None
    return file_name, transcript


def iter_model_jsons(
    transcriptions_dir: Path, model_names: Iterable[str]
) -> Iterable[Tuple[str, Path]]:
    for model_name in model_names:
        model_dir = transcriptions_dir / model_name
        if not model_dir.exists():
            continue
        for json_file in sorted(model_dir.glob("*.json")):
            yield model_name, json_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ASR transcriptions against Genius lyrics"
    )
    parser.add_argument(
        "--transcriptions-dir",
        default="transcriptions",
        help="Directory containing model subfolders with JSON outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper", "parakeet", "canary"],
        help="Model folders to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for leaderboard outputs",
    )
    parser.add_argument(
        "--cache-file",
        default="ground_truth/genius_lyrics_cache.json",
        help="Cache file path for Genius references",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing cache and refetch references",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.35,
        help="Minimum query/title overlap to accept a Genius hit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    access_token = os.getenv("GENIUS_ACCESS_TOKEN", "")
    if not access_token:
        raise SystemExit("GENIUS_ACCESS_TOKEN is required in the environment")

    transcriptions_dir = Path(args.transcriptions_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cache_file = Path(args.cache_file).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    store = GeniusReferenceStore(
        access_token=access_token,
        cache_path=cache_file,
        refresh_cache=args.refresh_cache,
        min_overlap=args.min_overlap,
    )

    rows = []
    skipped = 0
    missing_reference = 0

    for model_name, json_file in iter_model_jsons(transcriptions_dir, args.models):
        loaded = load_transcript(json_file)
        if loaded is None:
            skipped += 1
            continue
        audio_file, transcript = loaded

        reference = store.get_reference(audio_file)
        if reference is None:
            missing_reference += 1
            continue

        wer = word_error_rate(reference.lyrics, transcript)
        csim = char_similarity(reference.lyrics, transcript)
        jsim = jaccard_similarity(reference.lyrics, transcript)
        cosim = cosine_similarity(reference.lyrics, transcript)

        rows.append(
            {
                "model": model_name,
                "audio_file": audio_file,
                "query": reference.query,
                "genius_title": reference.title,
                "genius_artist": reference.artist,
                "genius_url": reference.genius_url,
                "wer": round(wer, 4),
                "char_similarity": round(csim, 4),
                "jaccard_similarity": round(jsim, 4),
                "cosine_similarity": round(cosim, 4),
            }
        )

    store.save()

    by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    leaderboard = []
    for model, model_rows in sorted(by_model.items()):
        wers = [float(r["wer"]) for r in model_rows]
        csims = [float(r["char_similarity"]) for r in model_rows]
        jsims = [float(r["jaccard_similarity"]) for r in model_rows]
        cosims = [float(r["cosine_similarity"]) for r in model_rows]
        leaderboard.append(
            {
                "model": model,
                "samples": len(model_rows),
                "wer_mean": round(statistics.fmean(wers), 4),
                "wer_median": round(statistics.median(wers), 4),
                "char_similarity_mean": round(statistics.fmean(csims), 4),
                "jaccard_similarity_mean": round(statistics.fmean(jsims), 4),
                "cosine_similarity_mean": round(statistics.fmean(cosims), 4),
            }
        )

    leaderboard.sort(key=lambda item: item["wer_median"])

    details_csv = output_dir / "asr_scores_by_file.csv"
    with details_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "audio_file",
                "query",
                "genius_title",
                "genius_artist",
                "genius_url",
                "wer",
                "char_similarity",
                "jaccard_similarity",
                "cosine_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    leaderboard_json = output_dir / "asr_leaderboard.json"
    leaderboard_json.write_text(
        json.dumps(
            {
                "models": leaderboard,
                "meta": {
                    "rows_scored": len(rows),
                    "rows_skipped_empty_transcript": skipped,
                    "rows_missing_reference": missing_reference,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    leaderboard_csv = output_dir / "asr_leaderboard.csv"
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "samples",
                "wer_mean",
                "wer_median",
                "char_similarity_mean",
                "jaccard_similarity_mean",
                "cosine_similarity_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(leaderboard)

    print(f"Scored rows: {len(rows)}")
    print(f"Skipped rows (empty transcript): {skipped}")
    print(f"Missing Genius references: {missing_reference}")
    if leaderboard:
        print(f"Best ASR model by median WER: {leaderboard[0]['model']}")
    print(f"Wrote: {details_csv}")
    print(f"Wrote: {leaderboard_json}")
    print(f"Wrote: {leaderboard_csv}")


if __name__ == "__main__":
    main()
