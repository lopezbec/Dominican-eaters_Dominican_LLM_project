"""CPU/GPU-compatible wrapper for F5-TTS focused on Spanish zero-shot tests."""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class F5SpanishTTSModel:
    """Wrapper for F5-TTS API-based inference."""

    def __init__(
        self,
        model_name: str = "F5TTS_v1_Base",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._has_f5 = self._check_dependency()

    def _check_dependency(self) -> bool:
        try:
            from f5_tts.api import F5TTS  # noqa: F401

            return True
        except ImportError:
            logger.error("f5-tts package not found. Install with: pip install f5-tts")
            return False

    def load_model(self) -> None:
        if not self._has_f5:
            raise ImportError("f5-tts not installed. Run: pip install f5-tts")

        if self.model is not None:
            logger.info("Model already loaded")
            return

        try:
            from f5_tts.api import F5TTS

            logger.info(
                "Loading F5-TTS model=%s device=%s", self.model_name, self.device
            )
            self.model = F5TTS(model=self.model_name, device=self.device)
            logger.info("F5-TTS loaded successfully")
        except Exception as e:
            logger.error("Failed to load F5-TTS: %s", e)
            raise

    def tts_to_file(
        self,
        text: str,
        file_path: str,
        ref_audio: str,
        ref_text: str = "",
        speed: float = 1.0,
        nfe_step: int = 32,
        seed: Optional[int] = None,
        remove_silence: bool = False,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        ref_audio_path = Path(ref_audio)
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info("Generating F5-TTS speech for text: '%s...'", text[:50])
            if not ref_text:
                logger.warning(
                    "ref_text is empty. F5-TTS may trigger auto-transcription for reference."
                )

            self.model.infer(
                ref_file=str(ref_audio_path),
                ref_text=ref_text,
                gen_text=text,
                speed=speed,
                nfe_step=nfe_step,
                seed=seed,
                remove_silence=remove_silence,
                file_wave=str(output_path),
                show_info=logger.info,
            )

            logger.info("Audio saved to: %s", output_path.absolute())
            return str(output_path)
        except Exception as e:
            logger.error("F5-TTS generation failed: %s", e)
            raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test F5-TTS for Spanish zero-shot synthesis"
    )
    parser.add_argument(
        "--model",
        default="F5TTS_v1_Base",
        help="F5-TTS model config name (default: F5TTS_v1_Base)",
    )
    parser.add_argument(
        "--text",
        default="Hola, esto es una prueba de F5 para español.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Reference audio path for zero-shot voice conditioning",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript of the reference audio (recommended for fair comparisons)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (default: 1.0)",
    )
    parser.add_argument(
        "--nfe-step",
        type=int,
        default=32,
        help="Diffusion/flow denoising steps (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--remove-silence",
        action="store_true",
        help="Remove long silences in output WAV",
    )
    parser.add_argument(
        "--output",
        default="output_f5_spanish_tts.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (cpu/cuda/mps/xpu). Auto-detect if omitted",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        model = F5SpanishTTSModel(model_name=args.model, device=args.device)
        model.load_model()
        model.tts_to_file(
            text=args.text,
            file_path=args.output,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            speed=args.speed,
            nfe_step=args.nfe_step,
            seed=args.seed,
            remove_silence=args.remove_silence,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
