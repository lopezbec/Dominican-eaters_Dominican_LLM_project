import logging
from typing import Dict

try:
    import whisper
    import torch
except Exception:
    whisper = None
    torch = None

logger = logging.getLogger(__name__)


class WhisperModelManager:
    """Lightweight mirror of audio_processing's WhisperModelManager for tests.

    - Detects CUDA via torch.cuda.is_available()
    - Loads model on cuda when available, falls back to cpu
    - Enables fp16 only on CUDA
    """

    def __init__(self, model_name: str, fp16: bool = True):
        if whisper is None:
            raise ImportError(
                "openai-whisper is required. Install with: pip install openai-whisper"
            )

        self.model_name = model_name
        self.fp16 = fp16
        self.model = None
        self.device = "cpu"

    def load_model(self) -> None:
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")

            if torch and getattr(torch, "cuda", None) and torch.cuda.is_available():
                try:
                    self.model = whisper.load_model(self.model_name, device="cuda")
                    self.device = "cuda"
                    logger.info("Model loaded on CUDA (GPU)")
                except Exception as e:
                    logger.warning(f"Failed to load on CUDA, using CPU: {e}")
                    try:
                        if torch and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self.model = whisper.load_model(self.model_name, device="cpu")
                    self.device = "cpu"
                    logger.info("Model loaded on CPU")
            else:
                self.model = whisper.load_model(self.model_name, device="cpu")
                self.device = "cpu"
                logger.info("Model loaded on CPU")

    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        fp16 = kwargs.pop("fp16", self.fp16) and self.device == "cuda"

        transcribe_kwargs = {
            "fp16": fp16,
            "verbose": False,
            **kwargs,
        }

        return self.model.transcribe(audio_path, **transcribe_kwargs)

    def cleanup(self) -> None:
        if torch and getattr(torch, "cuda", None) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def get_device(self) -> str:
        return self.device


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("--audio-dir", type=str, default=None, help="Audio directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max", type=int, default=3, help="Max files to process")
    parser.add_argument("--model", type=str, default="base", help="Whisper model")
    args = parser.parse_args()

    # Resolve paths
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    audio_dir = (
        Path(args.audio_dir) if args.audio_dir else repo_root / "lyrics-eater" / "audio"
    )
    out_dir = (
        Path(args.out_dir) if args.out_dir else this_dir / "transcriptions" / "whisper"
    )

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        exit(1)

    # Find audio files
    files = sorted(audio_dir.glob("*.m4a"))[: args.max]
    if not files:
        logger.error(f"No .m4a files found in {audio_dir}")
        exit(1)

    logger.info(f"Processing {len(files)} files from {audio_dir}")

    # Load model once
    model = WhisperModelManager(model_name=args.model, fp16=False)
    model.load_model()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    for audio_path in tqdm(files, desc="Transcribing"):
        try:
            result = model.transcribe(str(audio_path), language="es")

            output_file = out_dir / f"{audio_path.stem}.json"
            output_file.write_text(
                json.dumps(
                    {
                        "file": audio_path.name,
                        "transcript": result["text"],
                        "language": result.get("language", "es"),
                        "segments": result.get("segments", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.info(f"Saved: {output_file}")
        except Exception as e:
            logger.error(f"Failed to process {audio_path.name}: {e}")

    model.cleanup()
    logger.info("Done")
