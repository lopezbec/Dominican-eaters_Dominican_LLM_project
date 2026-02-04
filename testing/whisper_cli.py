from pathlib import Path
import argparse
import logging
import json
from typing import List
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("whisper_test")


def find_audio_files(audio_dir: Path, max_files: int) -> List[Path]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    exts = {".wav", ".m4a", ".mp3", ".flac"}
    files = sorted([p for p in audio_dir.iterdir() if p.suffix.lower() in exts])
    return files[:max_files]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch test Whisper and save timestamped JSONs"
    )
    parser.add_argument(
        "--max", type=int, default=10, help="Maximum number of audio files to process"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Source audio dir (defaults to lyrics-eater/audio)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir (default: testing/transcriptions/whisper)",
    )
    parser.add_argument(
        "--model", type=str, default="small", help="Whisper model name (default: small)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code for transcription/translation (default: es)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper task: 'transcribe' (default) or 'translate')",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Also save plain .txt alongside the JSON",
    )
    args = parser.parse_args()
    this_dir = Path(__file__).resolve().parent  # testing/
    repo_root = this_dir.parent
    default_audio_dir = repo_root / "lyrics-eater" / "audio"
    default_out_dir = this_dir / "transcriptions" / "whisper"
    audio_dir = Path(args.audio_dir) if args.audio_dir else default_audio_dir
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    logger.info("Audio dir: %s", audio_dir)
    logger.info("Output dir: %s", out_dir)
    logger.info("Max files: %d", args.max)
    logger.info("Whisper model: %s", args.model)
    logger.info("Language: %s, Task: %s", args.language, args.task)
    files = find_audio_files(audio_dir, args.max)
    if not files:
        logger.error("No audio files found in %s", audio_dir)
        return
    # Use the test-friendly WhisperModelManager which mirrors the project's
    # audio_processing/src/models/whisper_model.py behavior.
    # Note: whisper_cli.py is deprecated — prefer testing/stt/whisper_model.py directly.
    try:
        # when running as script with sys.path[0] == testing/, this will import testing/whisper_model.py
        from whisper_model import WhisperModelManager
    except Exception:
        # when running from repo root or as a package, import via package path
        from testing.whisper_model import WhisperModelManager
    try:
        manager = WhisperModelManager(args.model, fp16=True)
    except Exception as e:
        logger.exception("openai-whisper package not available: %s", e)
        logger.error(
            "Install it in your venv: pip install openai-whisper and ensure ffmpeg is available"
        )
        return
    try:
        manager.load_model()
        logger.info("Loaded whisper model: %s on %s", args.model, manager.get_device())
    except Exception as e:
        logger.exception("Failed to load whisper model %s: %s", args.model, e)
        return
    saved = []
    for audio_path in tqdm(files, desc="Transcribing", unit="file"):
        try:
            result = manager.transcribe(
                str(audio_path), language=args.language, task=args.task
            )
            text = result.get("text", "")
            segments = result.get("segments", None)
            payload = {
                "file": audio_path.name,
                "model": args.model,
                "task": args.task,
                "language": args.language,
                "transcript": text,
            }
            if segments is not None:
                payload["segments"] = segments
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / f"{audio_path.stem}.json"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if args.save_txt:
                txt_path = out_dir / f"{audio_path.stem}.txt"
                txt_path.write_text(text, encoding="utf-8")
            saved.append(json_path)
        except Exception as e:
            logger.exception("Failed processing %s: %s", audio_path, e)
    logger.info("Saved %d/%d transcripts to %s", len(saved), len(files), out_dir)
    try:
        manager.cleanup()
    except Exception:
        pass


if __name__ == "__main__":
    main()
