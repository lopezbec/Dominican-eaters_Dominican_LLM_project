"""Data model for Dominican poems and recitations."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from shared.models.base_content import BaseContent
from shared.models.enums import ContentAvailability
from shared.utils.text_parser import TextParser
from shared.constants.column_names import PoemColumns


@dataclass
class Poem(BaseContent):
    """
    Represents a Dominican poem with YouTube recitation information.
    """
    numero: int
    titulo: str
    autor: str
    año: str
    genero: str
    url_youtube: str = ContentAvailability.NOT_FOUND
    duracion: str = "N/A"
    recitador: str = "N/A"
    tipo_contenido: str = "N/A"
    calidad: str = "N/A"
    notas: str = ""
    disponibilidad: str = ContentAvailability.NOT_FOUND
    transcripcion: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert poem to dictionary for export.
        
        Returns:
            Dictionary with column names
        """
        return {
            PoemColumns.NUMBER: self.numero,
            PoemColumns.TITLE: self.titulo,
            PoemColumns.AUTHOR: self.autor,
            PoemColumns.YEAR: self.año,
            PoemColumns.GENRE: self.genero,
            PoemColumns.URL_YOUTUBE: self.url_youtube,
            PoemColumns.DURATION: self.duracion,
            PoemColumns.RECITER: self.recitador,
            PoemColumns.CONTENT_TYPE: self.tipo_contenido,
            PoemColumns.QUALITY: self.calidad,
            PoemColumns.NOTES: self.notas,
            PoemColumns.AVAILABILITY: self.disponibilidad,
            PoemColumns.TRANSCRIPTION: self.transcripcion
        }
    
    def mark_as_found(
        self, 
        url: str, 
        duration: str, 
        content_type: str,
        recitador: str = "N/A",
        calidad: str = "Buena",
        notas: str = "",
        partial: bool = False
    ):
        """
        Mark the poem as found with details.
        
        Args:
            url: YouTube video URL
            duration: Video duration
            content_type: Type of content (e.g., "Recitación", "Dramatización")
            recitador: Name of the person reciting
            calidad: Quality assessment (Excelente, Buena, Aceptable, Baja)
            notas: Additional notes
            partial: Whether it's a partial/fragment version
        """
        super().mark_as_found(
            url, 
            duration, 
            content_type=content_type,
            recitador=recitador,
            calidad=calidad,
            notas=notas,
            partial=partial
        )
    
    def _set_additional_fields(self, **kwargs):
        """Set poem-specific fields."""
        if 'content_type' in kwargs:
            self.tipo_contenido = kwargs['content_type']
        if 'recitador' in kwargs:
            self.recitador = kwargs['recitador']
        if 'calidad' in kwargs:
            self.calidad = kwargs['calidad']
        if 'notas' in kwargs:
            self.notas = kwargs['notas']
    
    def mark_as_partial(
        self, 
        url: str, 
        duration: str, 
        content_type: str = "Fragmentos",
        recitador: str = "N/A",
        calidad: str = "Aceptable",
        notas: str = "Solo fragmentos disponibles"
    ):
        """
        Mark the poem as partially found.
        """
        self.mark_as_found(url, duration, content_type, recitador, calidad, notas, partial=True)
    
    @property
    def display_name(self) -> str:
        """Get display name for the poem."""
        return f"{self.titulo} - {self.autor}"
    
    @staticmethod
    def create_from_text(numero: int, text: str) -> Optional['Poem']:
        """
        Create a Poem from a text line in format: "Título | Autor | Año | Género"
        
        Args:
            numero: Sequential number
            text: Text line with poem info
            
        Returns:
            Poem object or None if parsing fails
        """
        field_names = ['titulo', 'autor', 'año', 'genero']
        parsed = TextParser.parse_delimited(text, field_names)
        
        if parsed:
            return Poem(
                numero=numero,
                titulo=parsed['titulo'],
                autor=parsed['autor'],
                año=parsed.get('año', 'N/A'),
                genero=parsed.get('genero', 'N/A')
            )
        return None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Poem':
        """
        Create Poem from dictionary (for Excel loading).
        
        Maps Spanish column names to model attributes.
        
        Args:
            data: Dictionary with poem data (from pandas DataFrame)
            
        Returns:
            Poem object
            
        Example:
            >>> data = {
            ...     'Número': 1,
            ...     'Título': 'Quisqueya',
            ...     'Autor': 'Salomé Ureña',
            ...     'Año': '1874',
            ...     'Género': 'Patriótico',
            ...     'URL YouTube': 'https://youtube.com/...',
            ...     'Duración': '3:45'
            ... }
            >>> poem = Poem.from_dict(data)
        """
        return cls(
            numero=int(data.get(PoemColumns.NUMBER, 0)),
            titulo=str(data.get(PoemColumns.TITLE, '')),
            autor=str(data.get(PoemColumns.AUTHOR, '')),
            año=str(data.get(PoemColumns.YEAR, 'N/A')),
            genero=str(data.get(PoemColumns.GENRE, 'N/A')),
            url_youtube=str(data.get(PoemColumns.URL_YOUTUBE, ContentAvailability.NOT_FOUND)),
            duracion=str(data.get(PoemColumns.DURATION, 'N/A')),
            recitador=str(data.get(PoemColumns.RECITER, 'N/A')),
            tipo_contenido=str(data.get(PoemColumns.CONTENT_TYPE, 'N/A')),
            calidad=str(data.get(PoemColumns.QUALITY, 'N/A')),
            notas=str(data.get(PoemColumns.NOTES, '')),
            disponibilidad=str(data.get(PoemColumns.AVAILABILITY, ContentAvailability.NOT_FOUND)),
            transcripcion=str(data.get(PoemColumns.TRANSCRIPTION, ''))
        )
