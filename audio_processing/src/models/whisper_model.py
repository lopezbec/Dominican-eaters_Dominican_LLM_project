import logging
from typing import Dict, Optional

try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None

logger = logging.getLogger(__name__)


class WhisperModelManager:
    
    def __init__(self, model_name: str, fp16: bool = True):
        if whisper is None:
            raise ImportError(
                "openai-whisper is required. Install with: pip install openai-whisper"
            )
        
        self.model_name = model_name
        self.fp16 = fp16
        self.model = None
        self.device = 'cpu'
    
    def load_model(self):
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            
            if torch and torch.cuda.is_available():
                try:
                    self.model = whisper.load_model(self.model_name, device='cuda')
                    self.device = 'cuda'
                    logger.info("Model loaded on CUDA (GPU)")
                except Exception as e:
                    logger.warning(f"Failed to load on CUDA, using CPU: {e}")
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.model = whisper.load_model(self.model_name, device='cpu')
                    self.device = 'cpu'
                    logger.info("Model loaded on CPU")
            else:
                self.model = whisper.load_model(self.model_name, device='cpu')
                self.device = 'cpu'
                logger.info("Model loaded on CPU")
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        fp16 = kwargs.pop('fp16', self.fp16) and self.device == 'cuda'
        
        transcribe_kwargs = {
            'fp16': fp16,
            'verbose': False,
            **kwargs
        }
        
        return self.model.transcribe(audio_path, **transcribe_kwargs)
    
    def cleanup(self):
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_device(self) -> str:
        return self.device
