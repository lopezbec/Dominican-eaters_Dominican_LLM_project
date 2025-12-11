"""Base content model for all content types (books, songs, poems)."""

from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod


@dataclass
class BaseContent(ABC):
    """
    Abstract base class for all content types.
    
    Provides common interface for books, songs, and poems.
    Enforces Single Responsibility: defines the contract for content representation.
    """
    
    @abstractmethod
    def to_dict(self) -> Dict[str, str]:
        """
        Convert content to dictionary for export.
        
        Returns:
            Dictionary with localized column names
        """
        pass
    
    def get_display_name(self) -> str:
        """
        Get display name for the content.
        
        Returns:
            Human-readable content identifier
        """
        if hasattr(self, 'titulo'):
            return self.titulo
        elif hasattr(self, 'title'):
            return self.title
        return "Unknown"
    
    def has_youtube_url(self) -> bool:
        """
        Check if content has a valid YouTube URL.
        
        Returns:
            True if YouTube URL exists and is not default value
        """
        if hasattr(self, 'url_youtube'):
            return self.url_youtube not in ["NO ENCONTRADO", "N/A", ""]
        elif hasattr(self, 'youtube_url'):
            return self.youtube_url not in ["N/A", "", None]
        return False
    
    def has_transcription(self) -> bool:
        """
        Check if content has transcription data.
        
        Returns:
            True if transcription exists and is not empty
        """
        if hasattr(self, 'transcripcion'):
            return bool(self.transcripcion and self.transcripcion.strip())
        return False
