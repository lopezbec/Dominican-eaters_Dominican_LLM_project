"""CPU-compatible wrapper for Chatterbox TTS.

Chatterbox is ResembleAI's multilingual TTS supporting 23 languages including Spanish.
0.5B parameter model with emotion exaggeration control and zero-shot voice cloning.

Installation:
    pip install chatterbox-tts

Features:
    - Multilingual support (23 languages including Spanish)
    - Zero-shot voice cloning from reference audio
    - Emotion exaggeration control (0.0-1.0)
    - Ultra-stable with alignment-informed inference
    - Built-in watermarking for responsible AI

Supported Languages:
    en, es, fr, de, it, pt, ar, zh, ja, ko, nl, pl, ru, sv, tr, hi, and more

Example:
    >>> model = ChatterboxModel(device="cpu")
    >>> model.load_model()
    >>> wav = model.generate(text="Hola, esto es español.", language_id="es")
    >>> ta.save("output.wav", wav, model.sample_rate)

    >>> # With voice cloning
    >>> wav = model.generate(
    ...     text="Hola, clonando esta voz.",
    ...     language_id="es",
    ...     audio_prompt_path="reference.wav"
    ... )

Reference:
    https://huggingface.co/ResembleAI/chatterbox
"""

# Monkey-patch torch.load to force CPU map_location before importing chatterbox
# This fixes "Attempting to deserialize object on a CUDA device" on CPU-only machines
import importlib
import sys
from unittest.mock import patch

_original_torch_load = None


def _patched_torch_load(*args, **kwargs):
    """Wrap torch.load to always use map_location='cpu' when CUDA unavailable."""
    import torch

    if not torch.cuda.is_available() and "map_location" not in kwargs:
        kwargs["map_location"] = "cpu"
    return _original_torch_load(*args, **kwargs)


# Apply patch before any imports that use torch.load
import torch

_original_torch_load = torch.load
torch.load = _patched_torch_load

import logging
import sys
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class ChatterboxModel:
    """CPU-compatible wrapper for Chatterbox TTS."""

    # Supported languages for multilingual model
    SUPPORTED_LANGUAGES = [
        "en",  # English
        "es",  # Spanish
        "fr",  # French
        "de",  # German
        "it",  # Italian
        "pt",  # Portuguese
        "ar",  # Arabic
        "zh",  # Chinese
        "ja",  # Japanese
        "ko",  # Korean
        "nl",  # Dutch
        "pl",  # Polish
        "ru",  # Russian
        "sv",  # Swedish
        "tr",  # Turkish
        "hi",  # Hindi
        "da",  # Danish
        "el",  # Greek
        "fi",  # Finnish
        "he",  # Hebrew
        "ms",  # Malay
        "no",  # Norwegian
        "sw",  # Swahili
    ]

    def __init__(
        self,
        model_name: str = "ResembleAI/chatterbox",
        device: Optional[str] = None,
    ):
        """Initialize Chatterbox TTS model.

        Args:
            model_name: HuggingFace model ID (default: ResembleAI/chatterbox)
            device: "cpu", "cuda", or None for auto-detect
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.model = None
        self.sample_rate = 24000  # Chatterbox outputs 24kHz
        self._has_chatterbox = self._check_dependency()

    def _check_dependency(self) -> bool:
        """Check if chatterbox-tts is installed."""
        try:
            import chatterbox

            return True
        except ImportError:
            logger.error(
                "chatterbox-tts package not found. Install with: pip install chatterbox-tts"
            )
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
        """Load the Chatterbox model."""
        if not self._has_chatterbox:
            raise ImportError(
                "chatterbox-tts not installed. Run: pip install chatterbox-tts"
            )

        if self.model is not None:
            logger.info("Model already loaded")
            return

        try:
            # Import here to avoid loading if not needed
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            logger.info(f"Loading {self.model_name} on {self.device}")

            # Load multilingual model (supports Spanish)
            # API: from_pretrained only takes device parameter
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

            logger.info("Model loaded successfully")
            logger.info(f"Sample rate: {self.sample_rate}Hz")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        text: str,
        language_id: str = "es",
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
    ):
        """Generate speech from text.

        Args:
            text: Text to synthesize
            language_id: Language code ("es" for Spanish, "en" for English, etc.)
            audio_prompt_path: Path to reference audio for voice cloning (optional)
            exaggeration: Emotion intensity (0.0-1.0). Higher = more expressive

        Returns:
            torch.Tensor: Generated audio waveform

        Example:
            >>> model = ChatterboxModel()
            >>> model.load_model()
            >>> wav = model.generate(
            ...     text="Hola, esto es español.",
            ...     language_id="es",
            ...     exaggeration=0.5
            ... )
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if language_id not in self.SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language_id}' not in known supported list. "
                f"Attempting anyway. Known: {self.SUPPORTED_LANGUAGES}"
            )

        try:
            logger.info(f"Generating speech: '{text[:50]}...' (lang: {language_id})")
            logger.info(f"Settings: exaggeration={exaggeration}")

            # Generate with optional voice cloning
            if audio_prompt_path and Path(audio_prompt_path).exists():
                logger.info(f"Using voice prompt: {audio_prompt_path}")
                wav = self.model.generate(
                    text=text,
                    language_id=language_id,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                )
            else:
                if audio_prompt_path:
                    logger.warning(f"Audio prompt not found: {audio_prompt_path}")
                wav = self.model.generate(
                    text=text,
                    language_id=language_id,
                    exaggeration=exaggeration,
                )

            logger.info(
                f"Generated audio: {wav.shape} samples, {len(wav) / self.sample_rate:.2f}s"
            )
            return wav

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_supported_languages(self):
        """Return list of supported language codes."""
        return self.SUPPORTED_LANGUAGES.copy()

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            try:
                import torch

                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model resources cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


def main():
    """CLI demo for testing Chatterbox TTS."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Chatterbox TTS on Dominican Spanish text"
    )
    parser.add_argument(
        "--model",
        default="ResembleAI/chatterbox",
        help="Model name or path",
    )
    parser.add_argument(
        "--text",
        default="Hola, esto es una prueba de voz en español dominicano.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        default="es",
        choices=ChatterboxModel.SUPPORTED_LANGUAGES,
        help="Language code (es=Spanish, en=English, etc.)",
    )
    parser.add_argument(
        "--output",
        default="output_chatterbox.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu or cuda (auto-detect if not specified)",
    )
    parser.add_argument(
        "--voice-prompt",
        default=None,
        help="Path to reference audio for voice cloning",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration (0.0-1.0, default: 0.5)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Initialize and load model
        model = ChatterboxModel(model_name=args.model, device=args.device)
        model.load_model()

        # Log capabilities
        logger.info(f"Supported languages: {model.get_supported_languages()}")

        # Generate speech
        logger.info(f"Generating speech for: '{args.text}'")
        logger.info(f"Language: {args.language}")

        wav = model.generate(
            text=args.text,
            language_id=args.language,
            audio_prompt_path=args.voice_prompt,
            exaggeration=args.exaggeration,
        )

        # Save output
        import torchaudio as ta

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Audio is already 2D [channels, samples], no need for unsqueeze
        ta.save(str(output_path), wav, model.sample_rate)

        duration = len(wav) / model.sample_rate
        logger.info(f"Saved to: {output_path.absolute()}")
        logger.info(f"Duration: {duration:.2f}s, Sample rate: {model.sample_rate}Hz")

        # Cleanup
        model.cleanup()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install chatterbox-tts torchaudio")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
