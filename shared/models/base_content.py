"""Base content model for all content types (books, songs, poems)."""

from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod

from .enums import ContentAvailability


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

    @classmethod
    @abstractmethod
    def create_from_text(cls, index: int, text: str):
        """
        Create a content object from a line of text.

        Args:
            index (int): The index of the line.
            text (str): The line of text.

        Returns:
            A content object.
        """
        pass

    
    def mark_as_found(self, url: str, duration: str, **kwargs):
        """
        Mark the content as found with details.
        
        Args:
            url: YouTube video URL
            duration: Video duration
            **kwargs: Additional fields specific to content type
        """
        if hasattr(self, 'url_youtube'):
            self.url_youtube = url
        elif hasattr(self, 'youtube_url'):
            self.youtube_url = url
        
        if hasattr(self, 'duracion'):
            self.duracion = duration
        elif hasattr(self, 'duration'):
            self.duration = duration
        
        partial = kwargs.get('partial', False)
        if hasattr(self, 'disponibilidad'):
            self.disponibilidad = ContentAvailability.PARTIAL if partial else ContentAvailability.FOUND
        
        self._set_additional_fields(**kwargs)
    
    def _set_additional_fields(self, **kwargs):
        """
        Override in subclasses to set content-specific fields.
        
        Args:
            **kwargs: Additional fields specific to content type
        """
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Get display name for the content.
        
        Returns:
            Human-readable content identifier
        """
        pass
    
    def has_youtube_url(self) -> bool:
        """
        Check if content has a valid YouTube URL.
        
        Returns:
            True if YouTube URL exists and is not default value
        """
        if hasattr(self, 'url_youtube'):
            return self.url_youtube not in [ContentAvailability.NOT_FOUND, "N/A", ""]
        elif hasattr(self, 'youtube_url'):
            return self.youtube_url not in ["N/A", "", None]
        return False
