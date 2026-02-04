#!/usr/bin/env python3
"""
canary.py - NVIDIA Canary 1B Flash Batch ASR/AST

Transcribes and translates audio using NVIDIA's Canary 1B Flash model.

Model: nvidia/canary-1b-flash
- 883M parameters
- 4 languages: English, German, French, Spanish
- FastConformer Encoder + Transformer Decoder
- WER: 1.48% (LibriSpeech test-clean), 3.62% (Spanish MCV)
- Supports ASR + AST (Speech Translation)
- Word & segment-level timestamps
- License: CC-BY-4.0 (commercial use allowed)

Performance:
- RTFx: 1045 (A100), 1669 (H100)
- Excellent for Spanish (including Dominican Spanish)
- Multi-task: ASR + Translation

Tasks:
1. ASR: Transcribe in same language (en→en, es→es, etc.)
2. AST: Translate speech (es→en, en→es, etc.)

Documentation: https://huggingface.co/nvidia/canary-1b-flash
"""

from pathlib import Path
import argparse
import logging
import json
from typing import List, Optional
from tqdm import tqdm
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("canary")


def find_audio_files(audio_dir: Path, max_files: int) -> List[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    exts = {".wav", ".m4a", ".mp3", ".flac"}
    files = sorted([p for p in audio_dir.iterdir() if p.suffix.lower() in exts])
    return files[:max_files]


def transcribe_and_save(
    model,
    audio_path: Path,
    out_dir: Path,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    timestamps: bool = False,
) -> Optional[Path]:
    try:
        kwargs = {}
        if source_lang:
            kwargs["source_lang"] = source_lang
        if target_lang:
            kwargs["target_lang"] = target_lang
        if timestamps:
            kwargs["timestamps"] = True
        output = model.transcribe([str(audio_path)], **kwargs)
        if not output:
            logger.error("No transcription returned for %s", audio_path)
            return None
        item = output[0]
        transcript_text = getattr(item, "text", "")
        timestamp_obj = getattr(item, "timestamp", None)
        payload = {
            "file": audio_path.name,
            "transcript": transcript_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        if timestamp_obj is not None:
            payload["timestamp"] = timestamp_obj
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{audio_path.stem}.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return json_path
    except Exception:
        logger.exception("Failed to process %s", audio_path)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcribe/translate with canary-1b-v2"
    )
    parser.add_argument(
        "--max", type=int, default=10, help="Maximum number of audio files to process"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Source audio directory (defaults to lyrics-eater/audio)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for JSONs (default: testing/transcriptions/canary)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=None,
        help="Optional source language code (e.g. 'en')",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default=None,
        help="Optional target language code for translation (e.g. 'fr', 'spa')",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Request timestamps from model if supported",
    )
    args = parser.parse_args()
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent
    default_audio_dir = repo_root / "lyrics-eater" / "audio"
    default_out_dir = this_dir / "transcriptions" / "canary"
    audio_dir = Path(args.audio_dir) if args.audio_dir else default_audio_dir
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    logger.info("Audio dir: %s", audio_dir)
    logger.info("Output dir: %s", out_dir)
    logger.info("Max files: %d", args.max)
    logger.info("Source lang: %s", args.source_lang)
    logger.info("Target lang: %s", args.target_lang)
    logger.info("Timestamps: %s", args.timestamps)
    files = find_audio_files(audio_dir, args.max)
    if not files:
        logger.error("No audio files found in %s", audio_dir)
        return
    logger.info("Loading Canary 1B Flash model (this may take a while)...")
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/canary-1b-flash"
    )
    saved = []
    for audio_path in tqdm(files, desc="Processing", unit="file"):
        json_path = transcribe_and_save(
            model,
            audio_path,
            out_dir,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            timestamps=args.timestamps,
        )
        if json_path:
            saved.append(json_path)
    logger.info("Saved %d/%d outputs to %s", len(saved), len(files), out_dir)


if __name__ == "__main__":
    main()
