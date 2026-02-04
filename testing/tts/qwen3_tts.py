import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class Qwen3TTSModel:
    """CPU-compatible wrapper for Qwen3-TTS models.

    This is a standalone test wrapper for Qwen3-TTS-12Hz-1.7B-CustomVoice
    and related models. Optimized for CPU inference.

    Features:
        - Custom Voice generation with 9 premium speakers
        - Voice design from natural language descriptions
        - Voice cloning from reference audio (3-second rapid clone)
        - Multi-language support including Spanish

    Installation:
        pip install -U qwen-tts

    For CPU usage (recommended for this project):
        - Uses torch.float32 instead of bfloat16
        - Falls back to standard attention (no FlashAttention)
        - Automatically detects device (CPU/CUDA)

    Model variations:
        - Qwen3-TTS-12Hz-1.7B-CustomVoice: 9 premium voices, instruction control
        - Qwen3-TTS-12Hz-1.7B-VoiceDesign: Design voices from descriptions
        - Qwen3-TTS-12Hz-1.7B-Base: Base model for voice cloning
        - Qwen3-TTS-12Hz-0.6B-*: Smaller/faster variants

    Example:
        >>> model = Qwen3TTSModel(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        >>> model.load_model()
        >>> wavs, sr = model.generate_custom_voice(
        ...     text="Hola, esto es una prueba en español.",
        ...     language="Spanish",
        ...     speaker="Ryan"
        ... )
        >>> sf.write("output.wav", wavs[0], sr)

    Reference:
        https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    """

    SUPPORTED_SPEAKERS = [
        "Ryan",  # Dynamic male voice, strong rhythmic drive, English native
        "Aiden",  # Sunny American male voice, clear midrange, English native
        "Vivian",  # Bright young female voice, slightly edgy, Chinese native
        "Serena",  # Warm gentle young female voice, Chinese native
        "Uncle_Fu",  # Seasoned male voice, low mellow timbre, Chinese native
        "Dylan",  # Youthful Beijing male voice, clear natural, Chinese (Beijing Dialect)
        "Eric",  # Lively Chengdu male voice, husky brightness, Chinese (Sichuan Dialect)
        "Ono_Anna",  # Playful Japanese female voice, light nimble timbre, Japanese native
        "Sohee",  # Warm Korean female voice, rich emotion, Korean native
    ]

    SUPPORTED_LANGUAGES = [
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
        "Auto",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        """Initialize Qwen3 TTS model manager.

        Args:
            model_name: HuggingFace model ID or local path
            device: "cpu", "cuda", or None for auto-detect
            dtype: "float32", "bfloat16", "float16", or None for auto
        """
        self.model_name = model_name
        self.model = None
        self.device = device or self._detect_device()
        self.dtype_str = dtype or self._auto_dtype()
        self._has_qwen_tts = self._check_dependency()

    def _check_dependency(self) -> bool:
        """Check if qwen-tts is installed."""
        try:
            import qwen_tts

            return True
        except ImportError:
            logger.error(
                "qwen-tts package not found. Install with: pip install -U qwen-tts"
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

    def _auto_dtype(self) -> str:
        """Select appropriate dtype for device."""
        if self.device == "cpu":
            return "float32"  # CPU works best with float32
        else:
            return "bfloat16"  # GPU can use bfloat16 for efficiency

    def _get_torch_dtype(self):
        """Convert dtype string to torch dtype."""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype_str, torch.float32)

    def load_model(self) -> None:
        """Load the Qwen3 TTS model."""
        if not self._has_qwen_tts:
            raise ImportError("qwen-tts not installed. Run: pip install -U qwen-tts")

        if self.model is not None:
            logger.info("Model already loaded")
            return

        try:
            import torch
            from qwen_tts import Qwen3TTSModel as QwenModel

            logger.info(
                f"Loading {self.model_name} on {self.device} with {self.dtype_str}"
            )

            # Prepare kwargs based on device
            load_kwargs = {
                "device_map": self.device,
                "dtype": self._get_torch_dtype(),
            }

            # Only use FlashAttention on GPU with compatible dtype
            if self.device == "cuda" and self.dtype_str in ["float16", "bfloat16"]:
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using FlashAttention 2")
                except Exception:
                    logger.warning(
                        "FlashAttention not available, using standard attention"
                    )

            self.model = QwenModel.from_pretrained(self.model_name, **load_kwargs)
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = "Spanish",
        speaker: Union[str, List[str]] = "Ryan",
        instruct: Optional[Union[str, List[str]]] = None,
    ) -> tuple:
        """Generate speech using predefined custom voices.

        Args:
            text: Text(s) to synthesize
            language: Language(s) - "Spanish", "English", "Chinese", etc.
                     Use "Auto" for automatic language detection
            speaker: Speaker voice(s) - "Ryan", "Aiden", "Vivian", etc.
            instruct: Optional instruction for style control (e.g., "Speak softly")

        Returns:
            Tuple of (wavs: list[np.ndarray], sample_rate: int)

        Example:
            >>> model = Qwen3TTSModel()
            >>> model.load_model()
            >>>
            >>> # Single inference
            >>> wavs, sr = model.generate_custom_voice(
            ...     text="Hola, ¿cómo estás?",
            ...     language="Spanish",
            ...     speaker="Ryan"
            ... )
            >>> sf.write("spanish_test.wav", wavs[0], sr)
            >>>
            >>> # Batch inference
            >>> wavs, sr = model.generate_custom_voice(
            ...     text=["Hello world", "Hola mundo"],
            ...     language=["English", "Spanish"],
            ...     speaker=["Ryan", "Aiden"]
            ... )
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            kwargs = {
                "text": text,
                "language": language,
                "speaker": speaker,
            }
            if instruct is not None:
                kwargs["instruct"] = instruct

            wavs, sr = self.model.generate_custom_voice(**kwargs)
            return wavs, sr

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def get_supported_speakers(self) -> List[str]:
        """Return list of supported speaker names."""
        return self.SUPPORTED_SPEAKERS.copy()

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
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
    """Simple CLI demo for testing Qwen3 TTS."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Qwen3 TTS on Dominican Spanish text"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        help="Model name or path",
    )
    parser.add_argument(
        "--text",
        default="Hola, esto es una prueba de voz en español dominicano.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--speaker",
        default="Ryan",
        choices=Qwen3TTSModel.SUPPORTED_SPEAKERS,
        help="Speaker voice",
    )
    parser.add_argument(
        "--language",
        default="Spanish",
        choices=Qwen3TTSModel.SUPPORTED_LANGUAGES,
        help="Language",
    )
    parser.add_argument(
        "--output", default="output_qwen3_tts.wav", help="Output WAV file path"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu or cuda (auto-detect if not specified)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Initialize and load model
        model = Qwen3TTSModel(model_name=args.model, device=args.device)
        model.load_model()

        # Log capabilities
        logger.info(f"Supported speakers: {model.get_supported_speakers()}")
        logger.info(f"Supported languages: {model.get_supported_languages()}")

        # Generate speech
        logger.info(f"Generating speech for: '{args.text}'")
        logger.info(f"Speaker: {args.speaker}, Language: {args.language}")

        wavs, sr = model.generate_custom_voice(
            text=args.text, language=args.language, speaker=args.speaker
        )

        # Save output
        import soundfile as sf

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wavs[0], sr)

        logger.info(f"Saved to: {output_path.absolute()}")
        logger.info(f"Duration: {len(wavs[0]) / sr:.2f}s, Sample rate: {sr}Hz")

        # Cleanup
        model.cleanup()

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install -U qwen-tts soundfile")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
