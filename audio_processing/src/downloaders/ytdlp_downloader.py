import json
import subprocess
from typing import Dict, Optional
import logging

from shared.interfaces.video_downloader import IVideoDownloader

logger = logging.getLogger(__name__)


class YtDlpDownloader(IVideoDownloader):
    
    def extract_video_info(self, url: str) -> Optional[Dict]:
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-warnings',
                '--skip-download',
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
            return None
        except Exception as e:
            logger.error(f"Error extracting info from {url}: {e}")
            return None
    
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
        cmd = [
            'yt-dlp',
            '-f', 'bestaudio',
            '-x',
            '--audio-format', audio_format,
            '--audio-quality', audio_quality,
            '--postprocessor-args', f"ffmpeg:-ar {sample_rate} -ac {channels}",
            '-o', output_path,
            '--no-warnings',
            '--quiet',
            url
        ]
        
        subprocess.run(cmd, check=True, timeout=timeout)
