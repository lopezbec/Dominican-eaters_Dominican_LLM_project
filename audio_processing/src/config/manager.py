from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
import yaml


@dataclass
class AudioConfig:
    format: str = "m4a"
    sample_rate: int = 16000
    channels: int = 1
    codec: str = "aac"
    quality: str = "192k"


@dataclass
class DownloadConfig:
    max_retries: int = 3
    retry_delay: int = 5
    timeout: int = 300


@dataclass
class WhisperConfig:
    model: str = "medium"
    language: str = "es"
    word_timestamps: bool = True
    task: str = "transcribe"
    fp16: bool = False
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    device: str = "cpu"


@dataclass
class AlignmentConfig:
    min_similarity: float = 0.7
    aeneas_task_language: str = "spa"
    aeneas_is_text_type: str = "plain"


@dataclass
class ModuleConfig:
    name: str
    csv_path: Optional[str] = None
    excel_path: Optional[str] = None
    url_column: str = "URL YouTube"
    lyrics_column: Optional[str] = None
    audio_dir: str = "audio"
    transcriptions_dir: str = "transcriptions"
    alignments_dir: Optional[str] = None
    reports_dir: str = "reports"
    reference_texts_dir: Optional[str] = None


class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._load_config()
        
        self.audio = self._parse_audio_config()
        self.download = self._parse_download_config()
        self.whisper = self._parse_whisper_config()
        self.alignment = self._parse_alignment_config()
        self.modules = self._parse_modules_config()
    
    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
    
    def _parse_audio_config(self) -> AudioConfig:
        audio_data = self._raw_config.get('audio', {})
        return AudioConfig(**audio_data)
    
    def _parse_download_config(self) -> DownloadConfig:
        download_data = self._raw_config.get('download', {})
        return DownloadConfig(**download_data)
    
    def _parse_whisper_config(self) -> WhisperConfig:
        whisper_data = self._raw_config.get('whisper', {})
        return WhisperConfig(**whisper_data)
    
    def _parse_alignment_config(self) -> AlignmentConfig:
        alignment_data = self._raw_config.get('alignment', {})
        return AlignmentConfig(**alignment_data)
    
    def _parse_modules_config(self) -> Dict[str, ModuleConfig]:
        modules_data = self._raw_config.get('modules', {})
        modules = {}
        
        for name, data in modules_data.items():
            modules[name] = ModuleConfig(name=name, **data)
        
        return modules
    
    def get_module_config(self, module_name: str) -> Optional[ModuleConfig]:
        return self.modules.get(module_name)
    
    def get_raw_value(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self._raw_config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
