"""Data model for Dominican books and audiobooks."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Book:
    """
    Represents a Dominican book with audiobook information.
    """
    numero: int
    titulo: str
    autor: str
    año: str
    url_youtube: str = "NO ENCONTRADO"
    duracion: str = "N/A"
    tipo_contenido: str = "N/A"
    disponibilidad: str = "NO ENCONTRADO"
    
    def to_dict(self) -> dict:
        """
        Convert book to dictionary for export.
        
        Returns:
            Dictionary with column names
        """
        return {
            'Número': self.numero,
            'Título Libro': self.titulo,
            'Autor': self.autor,
            'Año': self.año,
            'URL YouTube': self.url_youtube,
            'Duración': self.duracion,
            'Tipo Contenido': self.tipo_contenido,
            'Disponibilidad': self.disponibilidad
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
        self.url_youtube = url
        self.duracion = duration
        self.tipo_contenido = content_type
        self.disponibilidad = "PARCIAL" if partial else "ENCONTRADO"
    
    def mark_as_partial(self, url: str, duration: str, content_type: str = "Fragmentos"):
        """
        Mark the book as partially found.
        """
        self.mark_as_found(url, duration, content_type, partial=True)
    
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
        try:
            parts = [p.strip() for p in text.split('|')]
            if len(parts) >= 2:
                titulo = parts[0]
                autor = parts[1]
                año = parts[2] if len(parts) > 2 else "N/A"
                return Book(numero=numero, titulo=titulo, autor=autor, año=año)
            return None
        except Exception:
            return None
