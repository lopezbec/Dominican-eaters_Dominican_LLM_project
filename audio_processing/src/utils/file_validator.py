import os
import logging
from typing import Tuple
from ..config.constants import (
    MIN_VALID_AUDIO_FILE_SIZE_BYTES,
    MIN_VALID_TRANSCRIPTION_FILE_SIZE_BYTES
)

logger = logging.getLogger(__name__)


class AudioFileValidator:
    
    def validate_exists(
        self, 
        path: str, 
        min_size: int = MIN_VALID_AUDIO_FILE_SIZE_BYTES
    ) -> Tuple[bool, str]:
        if not os.path.exists(path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(path)
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes, minimum {min_size})"
        
        return True, f"Valid audio file ({file_size} bytes)"


class TranscriptionFileValidator:
    
    def validate_exists(
        self, 
        path: str, 
        min_size: int = MIN_VALID_TRANSCRIPTION_FILE_SIZE_BYTES
    ) -> Tuple[bool, str]:
        if not os.path.exists(path):
            return False, "File does not exist"
        
        file_size = os.path.getsize(path)
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes, minimum {min_size})"
        
        return True, f"Valid transcription file ({file_size} bytes)"
