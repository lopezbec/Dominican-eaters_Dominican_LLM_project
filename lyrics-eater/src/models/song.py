"""Data models for song information."""

from dataclasses import dataclass
from typing import List


@dataclass
class Song:
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
