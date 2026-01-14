"""Data model for Dominican books and audiobooks."""

from dataclasses import dataclass
from typing import Dict, Optional

from shared.models.base_content import BaseContent
from shared.models.enums import ContentAvailability
from shared.utils.text_parser import TextParser
from shared.constants.column_names import BookColumns


@dataclass
class Book(BaseContent):
    """
    Represents a Dominican book with audiobook information.
    """
    numero: int
    titulo: str
    autor: str
    año: str
    url_youtube: str = ContentAvailability.NOT_FOUND
    duracion: str = "N/A"
    tipo_contenido: str = "N/A"
    disponibilidad: str = ContentAvailability.NOT_FOUND
    transcripcion: str = ""

    def has_transcription(self) -> bool:
        """
        Check if content has transcription data.
        
        Returns:
            True if transcription exists and is not empty
        """
        return bool(self.transcripcion and self.transcripcion.strip())

    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert book to dictionary for export.
        
        Returns:
            Dictionary with column names
        """
        return {
            BookColumns.NUMBER: self.numero,
            BookColumns.TITLE: self.titulo,
            BookColumns.AUTHOR: self.autor,
            BookColumns.YEAR: self.año,
            BookColumns.URL_YOUTUBE: self.url_youtube,
            BookColumns.DURATION: self.duracion,
            BookColumns.CONTENT_TYPE: self.tipo_contenido,
            BookColumns.AVAILABILITY: self.disponibilidad,
            BookColumns.TRANSCRIPTION: self.transcripcion
        }
    
    def mark_as_found(self, url: str, duration: str, content_type: str, partial: bool = False):
        """
        Mark the book as found with details.
        
        Args:
            url: YouTube video URL
            duration: Video duration
            content_type: Type of content (e.g., "Lectura Completa", "Dramatización")
            partial: Whether it's a partial/fragment version
        """
        super().mark_as_found(url, duration, content_type=content_type, partial=partial)
    
    def _set_additional_fields(self, **kwargs):
        """Set book-specific fields."""
        if 'content_type' in kwargs:
            self.tipo_contenido = kwargs['content_type']
    
    def mark_as_partial(self, url: str, duration: str, content_type: str = "Fragmentos"):
        """
        Mark the book as partially found.
        """
        self.mark_as_found(url, duration, content_type, partial=True)
    
    @property
    def display_name(self) -> str:
        """Get display name for the book."""
        return f"{self.titulo} - {self.autor}"
    
    @staticmethod
    def create_from_text(numero: int, text: str) -> Optional['Book']:
        """
        Create a Book from a text line in format: "Título | Autor | Año"
        
        Args:
            numero: Sequential number
            text: Text line with book info
            
        Returns:
            Book object or None if parsing fails
        """
        field_names = ['titulo', 'autor', 'año']
        parsed = TextParser.parse_delimited(text, field_names)
        
        if parsed:
            return Book(
                numero=numero,
                titulo=parsed['titulo'],
                autor=parsed['autor'],
                año=parsed.get('año', 'N/A')
            )
        return None
