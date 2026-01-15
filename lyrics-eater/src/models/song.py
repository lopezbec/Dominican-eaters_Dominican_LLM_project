"""Data models for song information."""

from dataclasses import dataclass
from typing import Any, Dict, List

from shared.models.base_content import BaseContent


@dataclass
class Song(BaseContent):
    """
    Represents a song with all its metadata.
    """
    song_id: str
    title: str
    artist: str
    url: str
    genres: str
    label: str
    album: str
    release_date: str
    lyrics: str
    youtube_url: str = "N/A"
    
    def to_dict(self) -> dict:
        """
        Convert song to dictionary for export.
        
        Returns:
            Dictionary with Spanish column names
        """
        return {
            'genero': self.genres,
            'artista': self.artist,
            'cancion': self.title,
            'letras': self.lyrics,
            'enlace_genius': self.url,
            'enlace_youtube': self.youtube_url,
            'discografica': self.label
        }
    
    @property
    def display_name(self) -> str:
        """Get display name for the song."""
        return f"{self.title} - {self.artist}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Song':
        """
        Create Song from dictionary (for Excel loading).
        
        Maps Spanish column names to model attributes.
        
        Args:
            data: Dictionary with song data (from pandas DataFrame)
            
        Returns:
            Song object
            
        Example:
            >>> data = {
            ...     'artista': 'Juan Luis Guerra',
            ...     'cancion': 'Ojalá Que Llueva Café',
            ...     'letras': 'Ojalá que llueva café...',
            ...     'enlace_genius': 'https://genius.com/...',
            ...     'enlace_youtube': 'https://youtube.com/...'
            ... }
            >>> song = Song.from_dict(data)
        """
        # Generate song_id from artist and title
        artist = str(data.get('artista', ''))
        title = str(data.get('cancion', ''))
        song_id = f"{artist}_{title}".replace(' ', '_').lower()
        
        return cls(
            song_id=song_id,
            title=title,
            artist=artist,
            url=str(data.get('enlace_genius', '')),
            genres=str(data.get('genero', 'N/A')),
            label=str(data.get('discografica', 'N/A')),
            album=str(data.get('album', 'N/A')),
            release_date=str(data.get('fecha_lanzamiento', 'N/A')),
            lyrics=str(data.get('letras', '')),
            youtube_url=str(data.get('enlace_youtube', 'N/A'))
        )
