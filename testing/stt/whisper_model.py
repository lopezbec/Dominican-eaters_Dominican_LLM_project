import logging
from typing import Dict
from pathlib import Path

try:
    import whisper
    import torch
except Exception:
    whisper = None
    torch = None

logger = logging.getLogger(__name__)


class WhisperModelManager:
    """Drop-in lightweight wrapper for testing openai-whisper models on CPU.

    Usage:
        mgr = WhisperModelManager('small', fp16=False)
        mgr.load_model()
        out = mgr.transcribe('path/to/file.m4a')
    """

    def __init__(self, model_name: str, fp16: bool = False):
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
            logger.info(f"Loading Whisper model: {self.model_name} (CPU preferred)")
            # default to CPU for reproducibility in testing
            self.model = whisper.load_model(self.model_name, device="cpu")
            self.device = "cpu"
            logger.info("Model loaded on CPU")

    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        transcribe_kwargs = {
            "fp16": False,  # CPU: do not use fp16
            "verbose": False,
            **kwargs,
        }

        return self.model.transcribe(audio_path, **transcribe_kwargs)

    def cleanup(self) -> None:
        # nothing to do for CPU tests
        return

    def get_device(self) -> str:
        return self.device
