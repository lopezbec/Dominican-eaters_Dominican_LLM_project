"""CPU/GPU-compatible wrapper for Kokoro TTS.

Kokoro is a lightweight open-weight TTS model. This wrapper keeps the same
CLI shape used by other scripts in testing/tts.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class KokoroTTSModel:
    """Wrapper for Kokoro TTS inference."""

    # From kokoro package docs (KPipeline lang_code)
    SUPPORTED_LANGUAGE_CODES = {
        "en-us": "a",
        "en-gb": "b",
        "es": "e",
        "fr": "f",
        "hi": "h",
        "it": "i",
        "ja": "j",
        "pt-br": "p",
        "zh": "z",
    }

    def __init__(
        self,
        lang_code: str = "e",
        default_voice: str = "ef_dora",
    ):
        self.lang_code = lang_code
        self.default_voice = default_voice
        self.pipeline = None
        self.sample_rate = 24000
        self._has_kokoro = self._check_dependency()

    def _check_dependency(self) -> bool:
        try:
            from kokoro import KPipeline  # noqa: F401
            import soundfile  # noqa: F401

            return True
        except ImportError:
            logger.error(
                "kokoro/soundfile package not found. Install with: "
                "pip install kokoro>=0.9.4 soundfile"
            )
            return False

    def load_model(self) -> None:
        if not self._has_kokoro:
            raise ImportError(
                "Missing dependencies. Run: pip install kokoro>=0.9.4 soundfile"
            )

        if self.pipeline is not None:
            logger.info("Model already loaded")
            return

        try:
            from kokoro import KPipeline

            logger.info("Loading Kokoro pipeline with lang_code=%s", self.lang_code)
            self.pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Kokoro loaded successfully")
        except Exception as e:
            logger.error("Failed to load Kokoro: %s", e)
            raise

    def synthesize(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0
    ) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        chosen_voice = voice or self.default_voice
        chunks = []

        try:
            logger.info("Generating Kokoro speech with voice=%s", chosen_voice)
            generator = self.pipeline(text, voice=chosen_voice, speed=speed)
            for _, _, audio in generator:
                if audio is None:
                    continue
                chunks.append(np.asarray(audio, dtype=np.float32).flatten())

            if not chunks:
                raise RuntimeError("Kokoro generated no audio chunks")

            return np.concatenate(chunks)
        except Exception as e:
            logger.error("Kokoro generation failed: %s", e)
            raise

    def tts_to_file(
        self,
        text: str,
        file_path: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> str:
        audio = self.synthesize(text=text, voice=voice, speed=speed)
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import soundfile as sf

        sf.write(str(output_path), audio, self.sample_rate)
        logger.info("Audio saved to: %s", output_path.absolute())
        return str(output_path)


def _resolve_lang_code(language: str) -> str:
    lang = language.lower()
    if len(lang) == 1:
        return lang
    if lang in KokoroTTSModel.SUPPORTED_LANGUAGE_CODES:
        return KokoroTTSModel.SUPPORTED_LANGUAGE_CODES[lang]
    if lang == "espanol":
        return "e"
    raise ValueError(
        "Unsupported language for Kokoro wrapper. "
        "Use one of: "
        f"{list(KokoroTTSModel.SUPPORTED_LANGUAGE_CODES.keys())} or a lang code letter."
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Test Kokoro TTS on Spanish text")
    parser.add_argument(
        "--text",
        default="Hola, esto es una prueba de Kokoro en español.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Language alias (es, en-us, fr...) or lang letter code (e, a, ...)",
    )
    parser.add_argument(
        "--voice",
        default="ef_dora",
        help="Kokoro voice id (default: ef_dora)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        default="output_kokoro_tts.wav",
        help="Output WAV file path",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        lang_code = _resolve_lang_code(args.language)
        model = KokoroTTSModel(lang_code=lang_code, default_voice=args.voice)
        model.load_model()
        model.tts_to_file(
            text=args.text,
            file_path=args.output,
            voice=args.voice,
            speed=args.speed,
        )
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
