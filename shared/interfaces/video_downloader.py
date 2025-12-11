from abc import ABC, abstractmethod
from typing import Dict, Optional


class IVideoDownloader(ABC):
    
    @abstractmethod
    def extract_video_info(self, url: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def download(
        self, 
        url: str, 
        output_path: str,
        audio_format: str,
        audio_quality: str,
        sample_rate: int,
        channels: int,
        timeout: int
    ) -> None:
        pass
