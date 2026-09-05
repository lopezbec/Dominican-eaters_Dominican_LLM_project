"""Batch XTTS-v2 voice cloning for a directory of reference texts.

Generates one WAV per text file using a single speaker reference clip.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List

from xtts_v2 import XTTSv2Model

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving original characters."""
    return re.sub(r"\s+", " ", text).strip()


def discover_text_files(reference_texts_dir: Path, pattern: str) -> List[Path]:
    """Discover and sort input text files."""
    files = sorted(reference_texts_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def ensure_coqui_license(agree_license: bool) -> None:
    """Ensure Coqui license agreement is set before loading XTTS."""
    if agree_license:
        os.environ["COQUI_TOS_AGREED"] = "1"
        return

    if os.environ.get("COQUI_TOS_AGREED") != "1":
        raise RuntimeError(
            "XTTS-v2 requires CPML agreement. Set COQUI_TOS_AGREED=1 "
            "or pass --agree-license. Terms: https://coqui.ai/cpml"
        )


def build_manifest(
    args: argparse.Namespace,
    model_name: str,
    selected_files: List[Path],
    results: List[Dict],
    started_at: float,
) -> Dict:
    """Create manifest payload for the run."""
    ended_at = time.time()
    ok_count = sum(1 for item in results if item["status"] == "ok")
    skipped_count = sum(1 for item in results if item["status"] == "skipped")
    error_count = sum(1 for item in results if item["status"] == "error")

    return {
        "tool": "batch_xtts_clone",
        "model": model_name,
        "language": args.language,
        "speaker_wav": str(Path(args.speaker_wav).resolve()),
        "reference_texts_dir": str(Path(args.reference_texts_dir).resolve()),
        "pattern": args.pattern,
        "skip_existing": args.skip_existing,
        "limit": args.limit,
        "max_chars": args.max_chars,
        "dry_run": args.dry_run,
        "started_at_epoch": started_at,
        "ended_at_epoch": ended_at,
        "duration_seconds": ended_at - started_at,
        "total_selected": len(selected_files),
        "ok": ok_count,
        "skipped": skipped_count,
        "errors": error_count,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic audio with XTTS-v2 for all text files in a directory"
        )
    )
    parser.add_argument(
        "--reference-texts-dir",
        required=True,
        help="Directory containing .txt reference files",
    )
    parser.add_argument(
        "--speaker-wav",
        required=True,
        help="Reference speaker WAV used for voice cloning",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where generated WAV files are written",
    )
    parser.add_argument(
        "--language",
        default="es",
        choices=XTTSv2Model.SUPPORTED_LANGUAGES,
        help="Language code for synthesis (default: es)",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for text files inside reference directory (default: *.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of text files to process",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional max chars per text before synthesis",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose output WAV already exists",
    )
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="XTTS model id/path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run model on: cpu or cuda (auto-detect if omitted)",
    )
    parser.add_argument(
        "--agree-license",
        action="store_true",
        help="Set COQUI_TOS_AGREED=1 for this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the run and write manifest without generating audio",
    )
    parser.add_argument(
        "--manifest-name",
        default="run_manifest.json",
        help="Manifest filename inside output directory",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    started_at = time.time()

    reference_dir = Path(args.reference_texts_dir)
    speaker_wav = Path(args.speaker_wav)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    if not reference_dir.exists() or not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference texts directory not found: {reference_dir}")

    if not speaker_wav.exists() or not speaker_wav.is_file():
        raise FileNotFoundError(f"Speaker wav not found: {speaker_wav}")

    ensure_coqui_license(args.agree_license)

    text_files = discover_text_files(reference_dir, args.pattern)
    if not text_files:
        raise FileNotFoundError(
            f"No text files found in {reference_dir} with pattern '{args.pattern}'"
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be > 0")
        text_files = text_files[: args.limit]

    if args.max_chars is not None and args.max_chars <= 0:
        raise ValueError("--max-chars must be > 0")

    logger.info("Selected %s text files", len(text_files))
    logger.info("Speaker reference: %s", speaker_wav)
    logger.info("Output dir: %s", output_dir)

    results: List[Dict] = []
    model = None

    try:
        if not args.dry_run:
            model = XTTSv2Model(model_name=args.model, device=args.device)
            model.load_model()

        for idx, text_path in enumerate(text_files, start=1):
            output_path = output_dir / f"{text_path.stem}.wav"
            result: Dict = {
                "index": idx,
                "text_path": str(text_path),
                "output_wav": str(output_path),
                "status": "error",
                "error": None,
                "chars": 0,
                "processing_time": 0.0,
            }

            try:
                raw_text = text_path.read_text(encoding="utf-8")
                normalized = normalize_text(raw_text)
                if args.max_chars is not None:
                    normalized = normalized[: args.max_chars]

                result["chars"] = len(normalized)

                if not normalized:
                    result["status"] = "skipped"
                    result["error"] = "empty_text_after_normalization"
                    results.append(result)
                    logger.warning(
                        "[%s/%s] Skipped empty text: %s",
                        idx,
                        len(text_files),
                        text_path.name,
                    )
                    continue

                if args.skip_existing and output_path.exists():
                    result["status"] = "skipped"
                    results.append(result)
                    logger.info(
                        "[%s/%s] Skipped existing: %s",
                        idx,
                        len(text_files),
                        output_path.name,
                    )
                    continue

                if args.dry_run:
                    result["status"] = "planned"
                    results.append(result)
                    logger.info(
                        "[%s/%s] Planned: %s", idx, len(text_files), text_path.name
                    )
                    continue

                start_item = time.time()
                model.tts_to_file(
                    text=normalized,
                    file_path=str(output_path),
                    speaker_wav=str(speaker_wav),
                    language=args.language,
                )
                result["processing_time"] = time.time() - start_item
                result["status"] = "ok"
                results.append(result)
                logger.info(
                    "[%s/%s] Generated: %s", idx, len(text_files), output_path.name
                )

            except Exception as item_error:
                result["error"] = str(item_error)
                results.append(result)
                logger.error(
                    "[%s/%s] Failed for %s: %s",
                    idx,
                    len(text_files),
                    text_path.name,
                    item_error,
                )

    finally:
        if model is not None:
            model.cleanup()

    manifest = build_manifest(args, args.model, text_files, results, started_at)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Manifest written to: %s", manifest_path)
    logger.info(
        "Done. total=%s ok=%s skipped=%s errors=%s",
        manifest["total_selected"],
        manifest["ok"],
        manifest["skipped"],
        manifest["errors"],
    )


if __name__ == "__main__":
    main()
