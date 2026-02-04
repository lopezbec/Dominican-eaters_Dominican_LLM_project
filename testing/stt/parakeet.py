#!/usr/bin/env python3
# parakeet.py — lightweight wrapper moved to testing/stt
from pathlib import Path
import argparse
import logging
import json
from typing import List, Optional

from tqdm import tqdm
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("parakeet")


def find_audio_files(audio_dir: Path, max_files: int) -> List[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    files = sorted([p for p in audio_dir.glob("*.m4a")])
    return files[:max_files]


def transcribe_and_save(model, audio_path: Path, out_dir: Path) -> Optional[Path]:
    try:
        output = model.transcribe([str(audio_path)], timestamps=True)
        if not output:
            logger.error("No transcription returned for %s", audio_path)
            return None
        item = output[0]
        transcript_text = getattr(item, "text", "")
        timestamps = getattr(item, "timestamp", {})

        payload = {
            "file": audio_path.name,
            "transcript": transcript_text,
            "timestamp": timestamps,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{audio_path.stem}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return json_path
    except Exception:
        logger.exception("Failed to transcribe %s", audio_path)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe files with Parakeet and save timestamped JSONs"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        help="Maximum number of audio files to transcribe (default: 10)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Path to audio directory (defaults to lyrics-eater/audio)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for JSONs (default: testing/stt/transcriptions/parakeet/)",
    )
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    default_audio_dir = repo_root / "lyrics-eater" / "audio"
    default_out_dir = this_dir / "transcriptions" / "parakeet"

    audio_dir = Path(args.audio_dir) if args.audio_dir else default_audio_dir
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir

    logger.info("Audio dir: %s", audio_dir)
    logger.info("Output dir: %s", out_dir)
    logger.info("Max files: %d", args.max)

    files = find_audio_files(audio_dir, args.max)
    if not files:
        logger.error("No .m4a files found in %s", audio_dir)
        return

    logger.info("Loading Parakeet model (may take a moment)...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )

    saved = []
    for audio_path in tqdm(files, desc="Transcribing", unit="file"):
        json_path = transcribe_and_save(model, audio_path, out_dir)
        if json_path:
            saved.append(json_path)

    logger.info("Saved %d/%d transcripts to %s", len(saved), len(files), out_dir)
    if saved:
        logger.info("Example saved file: %s", saved[0])


if __name__ == "__main__":
    main()
