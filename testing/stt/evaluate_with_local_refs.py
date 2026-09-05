#!/usr/bin/env python3
"""Evaluate ASR outputs against local human-reviewed references."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:
    from .scoring import NORMALIZATION, alignment_counts, bootstrap_intervals, corpus_rate, normalize_text, score_text
except ImportError:
    from scoring import NORMALIZATION, alignment_counts, bootstrap_intervals, corpus_rate, normalize_text, score_text


def words(text: str) -> List[str]:
    return normalize_text(text).split()


def characters(text: str) -> List[str]:
    return list(normalize_text(text).replace(" ", ""))


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    return alignment_counts(reference, hypothesis)["errors"]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> Optional[float]:
    return edit_distance(reference, hypothesis) / len(reference) if reference else None


def word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    return error_rate(words(reference), words(hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    return error_rate(characters(reference), characters(hypothesis))


@dataclass
class ReferenceSample:
    audio_file: str
    reference_text: str
    reference_text_path: str
    split: str
    source: str
    alignment_match_second: Optional[float]
    alignment_matched_word: str
    group_id: str = ""


def resolve_repo_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def build_reference_sample(
    item: Dict[str, object], manifest_path: Path
) -> ReferenceSample:
    audio_file = str(item.get("audio_file") or "").strip()
    if not audio_file:
        raise ValueError("Each manifest row must include 'audio_file'.")

    if "reference_text" in item and not isinstance(item["reference_text"], str):
        raise ValueError("reference_text must be a string, including for silence controls")
    reference_text = str(item.get("reference_text") or "").strip()
    reference_text_path = str(item.get("reference_text_path") or "").strip()
    if "reference_text" not in item and not reference_text_path:
        raise ValueError(
            f"Manifest row for '{audio_file}' must include 'reference_text' or 'reference_text_path'."
        )

    resolved_reference_path = ""
    if reference_text_path:
        resolved_path = resolve_repo_path(reference_text_path, manifest_path)
        if not resolved_path.exists():
            raise ValueError(
                f"Reference text file for '{audio_file}' was not found: {resolved_path}"
            )
        resolved_reference_path = str(resolved_path)
        if "reference_text" not in item:
            reference_text = resolved_path.read_text(encoding="utf-8")

    alignment_label = item.get("alignment_label")
    if alignment_label is None and (
        item.get("match_second") is not None or item.get("matched_word") is not None
    ):
        alignment_label = {
            "match_second": item.get("match_second"),
            "matched_word": item.get("matched_word"),
        }

    alignment_match_second: Optional[float] = None
    alignment_matched_word = ""
    if isinstance(alignment_label, dict):
        raw_second = alignment_label.get("match_second")
        if raw_second is not None and raw_second != "":
            alignment_match_second = float(raw_second)
        alignment_matched_word = str(alignment_label.get("matched_word") or "").strip()

    return ReferenceSample(
        audio_file=audio_file,
        reference_text=reference_text,
        reference_text_path=resolved_reference_path,
        split=str(item.get("split") or "unspecified"),
        source=str(item.get("source") or "local-human-transcript"),
        alignment_match_second=alignment_match_second,
        alignment_matched_word=alignment_matched_word,
        group_id=str(item.get("group_id") or audio_file),
    )


def load_manifest(manifest_path: Path) -> Dict[str, ReferenceSample]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows")
    else:
        rows = payload

    if not isinstance(rows, list):
        raise ValueError(
            "Manifest must be a JSON list or an object with a 'rows' list."
        )

    samples: Dict[str, ReferenceSample] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Manifest rows must be JSON objects.")
        sample = build_reference_sample(item, manifest_path)
        if sample.audio_file in samples:
            raise ValueError(f"Duplicate audio_file: {sample.audio_file}")
        samples[sample.audio_file] = sample
    return samples


def load_transcript(path: Path) -> Optional[Tuple[str, str]]:
    row = json.loads(path.read_text(encoding="utf-8"))
    transcript = (row.get("transcript") or "").strip()
    file_name = row.get("file") or f"{path.stem}.m4a"
    if row.get("status", "ok") != "ok":
        return None
    if not isinstance(row.get("transcript"), str):
        raise ValueError(f"Missing or invalid transcript: {path}")
    return str(file_name), transcript


def iter_model_jsons(
    transcriptions_dir: Path, model_names: Iterable[str]
) -> Iterable[Tuple[str, Path]]:
    for model_name in model_names:
        model_dir = transcriptions_dir / model_name
        if not model_dir.exists():
            continue
        for json_file in sorted(model_dir.glob("*.json")):
            yield model_name, json_file


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ASR transcriptions against local human transcript pairs"
    )
    parser.add_argument(
        "--transcriptions-dir",
        default="transcriptions",
        help="Directory containing model subfolders with JSON outputs",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON manifest describing local reference pairs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["whisper", "parakeet", "canary"],
        help="Model folders to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="results_local",
        help="Directory for leaderboard outputs",
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap replicates; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    return args


def evaluate(transcriptions_dir: Path, manifest_path: Path, model_names: Sequence[str],
             output_dir: Path, replicates: int = 2000, seed: int = 42) -> dict:
    references = load_manifest(manifest_path)
    if not references:
        raise ValueError("Manifest contains no samples")
    rows, coverage, leaderboard = [], [], []
    missing_reference = 0
    for model_name in dict.fromkeys(model_names):
        outputs = {}
        for _, path in iter_model_jsons(transcriptions_dir, [model_name]):
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = str(payload.get("file") or f"{path.stem}.m4a")
            if name in outputs:
                raise ValueError(f"Duplicate output for {model_name}: {name}")
            outputs[name] = payload
            missing_reference += int(name not in references)
        model_rows = []
        for name, ref in references.items():
            payload = outputs.get(name)
            status = "missing" if payload is None else payload.get("status", "ok")
            if status == "ok" and not isinstance(payload.get("transcript"), str):
                status = "invalid"
            coverage.append({"model": model_name, "audio_file": name, "status": status,
                             "error": payload.get("error") if payload else None})
            if status != "ok":
                continue
            transcript = payload["transcript"]
            row = {"model": model_name, "audio_file": name, "group_id": ref.group_id,
                   "split": ref.split, "source": ref.source,
                   "reference_text_path": ref.reference_text_path,
                   "alignment_match_second": ref.alignment_match_second,
                   "alignment_matched_word": ref.alignment_matched_word,
                   "reference_text": ref.reference_text, "transcript": transcript,
                   **score_text(ref.reference_text, transcript)}
            model_rows.append(row)
        rows.extend(model_rows)
        summary = {"model": model_name, "samples": len(model_rows),
                   "expected_samples": len(references),
                   "unscored_samples": len(references) - len(model_rows),
                   "wer_corpus": corpus_rate(model_rows, "word"),
                   "cer_corpus": corpus_rate(model_rows, "char"),
                   "confidence_intervals": bootstrap_intervals(model_rows, replicates, seed)}
        for metric in ("wer", "cer"):
            values = [r[metric] for r in model_rows if r[metric] is not None]
            summary[metric + "_mean"] = statistics.fmean(values) if values else None
            summary[metric + "_median"] = statistics.median(values) if values else None
        for prefix in ("word", "char"):
            for field in ("substitutions", "deletions", "insertions", "errors", "reference_length"):
                key = prefix + "_" + field
                summary[key] = sum(row[key] for row in model_rows)
        leaderboard.append(summary)
    # Incomplete runs are not ranked ahead of complete evaluations.
    leaderboard.sort(key=lambda r: (r["unscored_samples"] != 0,
                                   r["wer_corpus"] if r["wer_corpus"] is not None else float("inf"),
                                   r["model"]))
    report = {"models": leaderboard, "coverage": coverage,
              "meta": {"manifest": str(manifest_path), "rows_scored": len(rows),
                       "rows_skipped_empty_transcript": 0,
                       "rows_missing_reference": missing_reference,
                       "normalization": NORMALIZATION,
                       "comparison_warning": "Scores use available outputs; compare only matching coverage. No paired significance test is performed."}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "asr_leaderboard.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "asr_scores_by_file.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for filename, records, fallback in (
        ("asr_scores_by_file.csv", rows, ["model", "audio_file", "wer", "cer"]),
        ("asr_leaderboard.csv", [{k: v for k, v in r.items() if k != "confidence_intervals"}
                                 for r in leaderboard], ["model", "samples"]),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else fallback)
            writer.writeheader()
            writer.writerows(records)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = evaluate(Path(args.transcriptions_dir).resolve(), Path(args.manifest).resolve(),
                      args.models, Path(args.output_dir).resolve(), args.bootstrap, args.seed)
    print(f"Scored rows: {report['meta']['rows_scored']}")
    print(f"Wrote evaluation to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
