"""CPU-compatible wrapper for Coqui XTTS-v2 TTS.

XTTS-v2 is a Voice generation model that lets you clone voices into different
languages by using just a quick 6-second audio clip. No need for extensive training.

Installation:
    pip install TTS

Features:
    - Supports 17 languages including Spanish
    - Voice cloning with just 6-second audio clip
    - Emotion and style transfer by cloning
    - Cross-language voice cloning
    - Multi-lingual speech generation
    - 24kHz sampling rate

Supported Languages:
    en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

Example:
    >>> model = XTTSv2Model(device="cpu")
    >>> model.load_model()
    >>> model.tts_to_file(
    ...     text="Hola, esto es español dominicano.",
    ...     file_path="output.wav",
    ...     language="es"
    ... )

    >>> # With voice cloning
    >>> model.tts_to_file(
    ...     text="Hola, clonando esta voz.",
    ...     file_path="cloned.wav",
    ...     speaker_wav="reference.wav",
    ...     language="es"
    ... )

Reference:
    https://huggingface.co/coqui/XTTS-v2
"""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class XTTSv2Model:
    """CPU-compatible wrapper for Coqui XTTS-v2 TTS."""

    SUPPORTED_LANGUAGES = [
        "en",  # English
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "it",  # Italian
        "pt",  # Portuguese
        "pl",  # Polish
        "tr",  # Turkish
        "ru",  # Russian
        "nl",  # Dutch
        "cs",  # Czech
        "ar",  # Arabic
        "zh-cn",  # Chinese (Simplified)
        "ja",  # Japanese
        "hu",  # Hungarian
        "ko",  # Korean
        "hi",  # Hindi
    ]

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
    ):
        """Initialize XTTS-v2 TTS model.

        Args:
            model_name: Coqui TTS model name (default: xtts_v2)
            device: "cpu", "cuda", or None for auto-detect
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.tts = None
        self._has_tts = self._check_dependency()

    def _check_dependency(self) -> bool:
        """Check if TTS package is installed."""
        try:
            from TTS.api import TTS

            return True
        except ImportError:
            logger.error("TTS package not found. Install with: pip install TTS")
            return False

    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA available, using GPU")
                return "cuda"
        except ImportError:
            pass
        logger.info("Using CPU for inference")
        return "cpu"

    def load_model(self) -> None:
        """Load the XTTS-v2 model."""
        if not self._has_tts:
            raise ImportError("TTS package not installed. Run: pip install TTS")

        if self.tts is not None:
            logger.info("Model already loaded")
            return

        try:
            from TTS.api import TTS

            logger.info(f"Loading {self.model_name} on {self.device}")

            # Initialize TTS with model name and GPU flag
            gpu = self.device == "cuda"
            self.tts = TTS(self.model_name, gpu=gpu)

            logger.info("Model loaded successfully")
            logger.info(f"Using device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def tts_to_file(
        self,
        text: str,
        file_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "es",
    ) -> str:
        """Generate speech and save to file.

        Args:
            text: Text to synthesize
            file_path: Output file path (.wav recommended)
            speaker_wav: Path to reference audio for voice cloning (optional)
            language: Language code ("es" for Spanish)

        Returns:
            str: Path to generated audio file

        Example:
            >>> model = XTTSv2Model()
            >>> model.load_model()
            >>> output = model.tts_to_file(
            ...     text="Hola, esto es español.",
            ...     file_path="output.wav",
            ...     language="es"
            ... )

            >>> # Clone voice
            >>> model.tts_to_file(
            ...     text="Texto clonado.",
            ...     file_path="cloned.wav",
            ...     speaker_wav="mi_voz.wav",
            ...     language="es"
            ... )
        """
        if self.tts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' not in known supported list. "
                f"Attempting anyway. Known: {self.SUPPORTED_LANGUAGES}"
            )

        try:
            logger.info(f"Generating speech: '{text[:50]}...' (lang: {language})")

            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate with or without voice cloning
            if speaker_wav and Path(speaker_wav).exists():
                logger.info(f"Cloning voice from: {speaker_wav}")
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    speaker_wav=speaker_wav,
                    language=language,
                )
            else:
                if speaker_wav:
                    logger.warning(f"Speaker wav not found: {speaker_wav}")
                self.tts.tts_to_file(
                    text=text,
                    file_path=str(output_path),
                    language=language,
                )

            logger.info(f"Audio saved to: {output_path.absolute()}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "es",
        gpt_cond_len: int = 3,
    ):
        """Generate speech and return waveform directly.

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio for voice cloning (optional)
            language: Language code ("es" for Spanish)
            gpt_cond_len: GPT conditioning length in seconds (default: 3)

        Returns:
            dict: Contains 'wav' (audio array) and other metadata

        Note:
            This uses the lower-level API for more control.
        """
        if self.tts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            logger.info(f"Synthesizing: '{text[:50]}...' (lang: {language})")

            outputs = self.tts.synthesizer.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                gpt_cond_len=gpt_cond_len,
            )

            return outputs

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise

    def get_supported_languages(self):
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES.copy()

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.tts is not None:
            try:
                import torch

                del self.tts
                self.tts = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model resources cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


def main():
    """CLI demo for testing XTTS-v2 TTS."""
    import argparse
    import os

    # Fix PyTorch 2.6+ weights_only issue for XTTS-v2 checkpoints
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig

    torch.serialization.add_safe_globals(
        [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
    )

    parser = argparse.ArgumentParser(
        description="Test XTTS-v2 TTS on Dominican Spanish text"
    )
    parser.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Model name (default: xtts_v2)",
    )
    parser.add_argument(
        "--text",
        default="Hola, esto es una prueba de voz en español dominicano.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        default="es",
        choices=XTTSv2Model.SUPPORTED_LANGUAGES,
        help="Language code (default: es for Spanish)",
    )
    parser.add_argument(
        "--output",
        default="output_xtts_v2.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu or cuda (auto-detect if not specified)",
    )
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Path to reference audio for voice cloning (REQUIRED for XTTS-v2). "
        "Provide a 6-30 second audio clip of the voice to clone.",
    )
    parser.add_argument(
        "--agree-license",
        action="store_true",
        help="Agree to Coqui Public Model License (CPML). Required for first use. "
        "See: https://coqui.ai/cpml",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle license agreement
    if args.agree_license:
        import os

        os.environ["COQUI_TOS_AGREED"] = "1"
        logger.info("License agreement accepted (COQUI_TOS_AGREED=1)")
    else:
        logger.warning(
            "XTTS-v2 requires accepting the Coqui Public Model License (CPML). "
            "Use --agree-license to accept the terms of service. "
            "License: https://coqui.ai/cpml"
        )
        print(
            "\nYou must agree to the Coqui Public Model License (CPML) to use this model."
        )
        print("Terms: https://coqui.ai/cpml")
        print("\nCommercial licensing available: licensing@coqui.ai")
        print("\nTo accept, run with --agree-license flag")
        sys.exit(1)

    # XTTS-v2 requires a speaker reference for voice cloning
    if not args.speaker_wav:
        logger.error("XTTS-v2 requires a speaker reference audio file.")
        print("\n❌ ERROR: --speaker-wav is required for XTTS-v2")
        print("\nXTTS-v2 is a voice cloning model that requires a reference audio.")
        print("Usage examples:")
        print(
            '  python xtts_v2.py --text "Hola" --language es --speaker-wav ref.wav --agree-license'
        )
        print("\nThe reference audio should be:")
        print("  - 6-30 seconds of clear speech")
        print("  - Same language as the target text (for best results)")
        print("  - Clean, noise-free audio")
        sys.exit(1)

    if not Path(args.speaker_wav).exists():
        logger.error(f"Speaker reference not found: {args.speaker_wav}")
        sys.exit(1)

    try:
        # Initialize and load model
        model = XTTSv2Model(model_name=args.model, device=args.device)
        model.load_model()

        # Log capabilities
        logger.info(f"Supported languages: {model.get_supported_languages()}")

        # Generate speech
        logger.info(f"Generating speech for: '{args.text}'")
        logger.info(f"Language: {args.language}")

        output_path = model.tts_to_file(
            text=args.text,
            file_path=args.output,
            speaker_wav=args.speaker_wav,
            language=args.language,
        )

        logger.info(f"Audio saved to: {output_path}")

        # Cleanup
        model.cleanup()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install TTS")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
