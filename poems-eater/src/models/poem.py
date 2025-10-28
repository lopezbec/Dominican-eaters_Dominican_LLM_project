"""Data model for Dominican poems and recitations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Poem:
    """
    Represents a Dominican poem with YouTube recitation information.
    """
    numero: int
    titulo: str
    autor: str
    año: str
    genero: str
    url_youtube: str = "NO ENCONTRADO"
    duracion: str = "N/A"
    recitador: str = "N/A"
    tipo_contenido: str = "N/A"
    calidad: str = "N/A"
    notas: str = ""
    disponibilidad: str = "NO ENCONTRADO"
    
    def to_dict(self) -> dict:
        """
        Convert poem to dictionary for export.
        
        Returns:
            Dictionary with column names
        """
        return {
            '#': self.numero,
            'Poema': self.titulo,
            'Autor': self.autor,
            'Año': self.año,
            'Género': self.genero,
            'URL YouTube': self.url_youtube,
            'Duración': self.duracion,
            'Recitador': self.recitador,
            'Tipo Contenido': self.tipo_contenido,
            'Calidad': self.calidad,
            'Notas': self.notas,
            'Disponibilidad': self.disponibilidad
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
        self.url_youtube = url
        self.duracion = duration
        self.tipo_contenido = content_type
        self.recitador = recitador
        self.calidad = calidad
        self.notas = notas
        self.disponibilidad = "PARCIAL" if partial else "ENCONTRADO"
    
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
        try:
            parts = [p.strip() for p in text.split('|')]
            if len(parts) >= 2:
                titulo = parts[0]
                autor = parts[1]
                año = parts[2] if len(parts) > 2 else "N/A"
                genero = parts[3] if len(parts) > 3 else "N/A"
                return Poem(
                    numero=numero, 
                    titulo=titulo, 
                    autor=autor, 
                    año=año,
                    genero=genero
                )
            return None
        except Exception:
            return None
