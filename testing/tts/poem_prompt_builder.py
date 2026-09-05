#!/usr/bin/env python3
"""Build short Spanish TTS prompts from all-poems markdown files."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import List


SKIP_PATTERNS = [
    r"^#",
    r"^---$",
    r"^Fuente original:",
    r"^Regresar a",
    r"^Volver a",
    r"^lista de poemas",
    r"^de Rhina",
    r"^Traducción de",
    r"^Poema\s*#?\d+",
    r"^Por\s+",
    r"\(\d{4}\s*[–-]\s*\d{4}\)",
]


def clean_markdown_content(text: str) -> str:
    lines = text.splitlines()
    kept: List[str] = []
    stop_after_hr = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line == "---":
            stop_after_hr = True
            continue

        if stop_after_hr:
            continue

        if any(re.search(pattern, line, re.IGNORECASE) for pattern in SKIP_PATTERNS):
            continue

        # Remove simple markdown marks but keep text content
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"`(.*?)`", r"\1", line)
        line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", line)
        kept.append(line)

    merged = " ".join(kept)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def sentence_candidates(text: str, min_chars: int, max_chars: int) -> List[str]:
    # Split by punctuation boundaries, preserving enough textual variety.
    rough = re.split(r"(?<=[\.!\?;:])\s+", text)

    out: List[str] = []
    for part in rough:
        s = re.sub(r"\s+", " ", part).strip(" \"'“”«»")
        if len(s) < min_chars or len(s) > max_chars:
            continue
        if len(s.split()) < 5:
            continue
        out.append(s)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a poem-based prompt set for TTS benchmarks"
    )
    parser.add_argument(
        "--poems-dir",
        default="../../all-poems",
        help="Path to all-poems directory (default: ../../all-poems from testing/tts)",
    )
    parser.add_argument(
        "--output",
        default="outputs/poems_prompts.jsonl",
        help="Output JSONL with selected prompts",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=24,
        help="Maximum prompt count",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=70,
        help="Minimum characters per prompt",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=180,
        help="Maximum characters per prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    poems_dir = Path(args.poems_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not poems_dir.exists() or not poems_dir.is_dir():
        raise FileNotFoundError(f"Poems directory not found: {poems_dir}")

    poem_files = sorted(poems_dir.glob("*.md"))
    if not poem_files:
        raise FileNotFoundError(f"No .md files found in: {poems_dir}")

    all_prompts = []
    for poem_file in poem_files:
        content = poem_file.read_text(encoding="utf-8", errors="ignore")
        clean = clean_markdown_content(content)
        if not clean:
            continue

        candidates = sentence_candidates(clean, args.min_chars, args.max_chars)
        for text in candidates:
            all_prompts.append(
                {
                    "source_file": poem_file.name,
                    "text": text,
                    "char_count": len(text),
                }
            )

    if not all_prompts:
        raise RuntimeError("No prompt candidates found after cleaning/filtering")

    rng = random.Random(args.seed)
    rng.shuffle(all_prompts)
    selected = all_prompts[: args.max_prompts]

    with output_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(selected, start=1):
            row = {
                "id": f"prompt_{i:03d}",
                "source_file": item["source_file"],
                "char_count": item["char_count"],
                "text": item["text"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} prompts to: {output_path}")


if __name__ == "__main__":
    main()
