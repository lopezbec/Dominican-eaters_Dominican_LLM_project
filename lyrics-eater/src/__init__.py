"""Lyrics Eater - Main package for extracting song lyrics from Genius."""

__version__ = '1.0.0'
__author__ = 'manuujrodcruz'

from .services import LyricsService
from .models import Song
from .utils import config, FileHandler

__all__ = [
    'LyricsService',
    'Song',
    'config',
    'FileHandler'
]
