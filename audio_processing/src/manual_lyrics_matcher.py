import argparse
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_track_index(stem: str) -> int:
    match = re.match(r"^[^_]+_(\d+)_", stem)
    if not match:
        return 10**9
    return int(match.group(1))


def collect_pairs(audio_dir: Path, reference_dir: Path) -> List[Dict[str, str]]:
    references_by_stem = {p.stem: p for p in reference_dir.glob("*.txt")}
    pairs: List[Dict[str, str]] = []

    for audio_file in audio_dir.glob("*.m4a"):
        stem = audio_file.stem
        reference_file = references_by_stem.get(stem)
        if not reference_file:
            continue
        pairs.append(
            {
                "stem": stem,
                "audio_path": str(audio_file),
                "reference_path": str(reference_file),
            }
        )

    pairs.sort(key=lambda x: (parse_track_index(x["stem"]), x["stem"]))
    return pairs


def load_processed_stems(output_file: Path) -> set:
    processed = set()
    if not output_file.exists():
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                stem = record.get("stem")
                if isinstance(stem, str) and stem:
                    processed.add(stem)
            except json.JSONDecodeError:
                continue

    return processed


def append_record(output_file: Path, record: Dict) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def run_session(
    pairs: List[Dict[str, str]],
    ffplay_bin: str,
    output_file: Path,
    resume: bool,
    limit: Optional[int],
) -> None:
    processed = load_processed_stems(output_file) if resume else set()
    remaining = [p for p in pairs if p["stem"] not in processed]

    if limit is not None:
        remaining = remaining[:limit]

    if not remaining:
        print("No tracks available for annotation with current filters.")
        return

    total = len(remaining)
    print(f"Tracks queued: {total}")
    print(f"Output file: {output_file}")

    for i, pair in enumerate(remaining, start=1):
        stem = pair["stem"]
        audio_path = Path(pair["audio_path"])
        reference_path = Path(pair["reference_path"])

        print("\n" + "=" * 80)
        print(f"[{i}/{total}] {stem}")
        print(f"Audio: {audio_path}")
        print(f"Lyrics: {reference_path}")
        print("-" * 80)

        lyrics_text = reference_path.read_text(encoding="utf-8", errors="replace")
        print(lyrics_text)
        print("-" * 80)
        print("Press Enter to start playback, or type q to quit.")
        start_choice = input("> ").strip().lower()
        if start_choice == "q":
            print("Exiting session.")
            return

        process = subprocess.Popen(
            [
                ffplay_bin,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                str(audio_path),
            ]
        )

        start_time = time.monotonic()
        print("Playback started. Press Enter at the first matching second.")
        input()
        elapsed_seconds = time.monotonic() - start_time
        stop_process(process)

        matched_word = input("Type the matched word: ").strip()

        record = {
            "stem": stem,
            "audio_file": audio_path.name,
            "reference_text_file": reference_path.name,
            "match_second": round(elapsed_seconds, 3),
            "matched_word": matched_word,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        append_record(output_file, record)

        print(f"Saved: second={record['match_second']} word='{record['matched_word']}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Play audio and manually annotate the second and word where lyrics match."
        )
    )
    parser.add_argument(
        "--module",
        default="lyrics-eater",
        help="Module name from config (default: lyrics-eater)",
    )
    parser.add_argument(
        "--config",
        default="audio_processing/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Start from pair index after sorting (0-based)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tracks to annotate in this run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stems already present in output jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path (default: <module>/reports/manual_word_matches.jsonl)",
    )
    parser.add_argument(
        "--ffplay-bin",
        default="ffplay",
        help="ffplay binary name or absolute path",
    )

    args = parser.parse_args()

    ffplay_path = (
        shutil.which(args.ffplay_bin)
        if not Path(args.ffplay_bin).exists()
        else args.ffplay_bin
    )
    if not ffplay_path:
        raise FileNotFoundError(
            "ffplay not found. Install ffmpeg or pass --ffplay-bin /path/to/ffplay"
        )

    config = load_config(args.config)
    module_config = config.get("modules", {}).get(args.module)
    if not module_config:
        raise ValueError(f"Module not found in config: {args.module}")

    audio_dir = Path(module_config["audio_dir"])
    reference_dir = Path(module_config["reference_texts_dir"])
    reports_dir = Path(module_config.get("reports_dir", f"{args.module}/reports"))

    output_path = (
        Path(args.output) if args.output else reports_dir / "manual_word_matches.jsonl"
    )

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference texts directory not found: {reference_dir}")

    pairs = collect_pairs(audio_dir=audio_dir, reference_dir=reference_dir)
    if not pairs:
        print("No audio/reference pairs found with matching stems.")
        return

    if args.index < 0 or args.index >= len(pairs):
        raise ValueError(
            f"--index out of range. Received {args.index}, available 0..{len(pairs) - 1}"
        )

    selected_pairs = pairs[args.index :]
    run_session(
        pairs=selected_pairs,
        ffplay_bin=ffplay_path,
        output_file=output_path,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
