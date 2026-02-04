"""Business logic for processing song lyrics requests."""

import logging
import time
from typing import Tuple, Optional

from shared.services.base_service import BaseService
from shared.services.stats_formatter import ProcessingStats
from shared.interfaces.searchers import GeniusSearcher, YouTubeSearcher
from src.models.song import Song

logger = logging.getLogger(__name__)


class LyricsService(BaseService):
    
    def __init__(self, genius_client: GeniusSearcher, youtube_client: YouTubeSearcher):
        super().__init__()
        self.genius_client = genius_client
        self.youtube_client = youtube_client
    
    def process_search_query(self, query: str) -> Tuple[Optional[Song], bool]:
        """
        Process a single search query and return song with lyrics.
        
        Args:
            query: Search query (e.g., "Obsesion Aventura")
            
        Returns:
            Tuple of (Song object or None, success boolean)
        """
        # Throttle requests to avoid hitting Genius rate limits (simple fixed sleep).
        # This sleeps 2 seconds before each search request.
        time.sleep(2)

        results = self.genius_client.search(query)
        
        if not results:
            logger.info("No results found for query: %s", query)
            return None, False
        
        first_result = results[0]
        logger.info("Found: %s - %s", first_result['title'], first_result['artist'])

        song_data = self.genius_client.get_song_details(first_result['id'])
        if not song_data:
            logger.warning("Could not fetch details for song ID: %s", first_result['id'])
            return None, False
        if isinstance(song_data, dict):
            song_obj = Song(
                song_id=str(song_data.get('id', '')),
                title=str(song_data.get('title', '')),
                artist=str(song_data.get('artist', '')),
                url=str(song_data.get('url', '')),
                genres=str(song_data.get('genres', 'N/A')),
                label=str(song_data.get('label', 'N/A')),
                album=str(song_data.get('album', 'N/A')),
                release_date=str(song_data.get('release_date', 'N/A')),
                lyrics="",
            )
        else:
            song_obj = song_data
        logger.info("Song details - Genre: %s, Label: %s", song_obj.genres, song_obj.label)
        logger.debug("Fetching lyrics...")
        lyrics = self.genius_client.scrape_lyrics(song_obj.url)
        if lyrics:
            logger.info("Lyrics obtained (%d characters)", len(lyrics))
            song_obj.lyrics = lyrics
        else:
            logger.warning("Could not obtain lyrics for: %s", song_obj.title)
            song_obj.lyrics = "N/A"
        logger.debug("Searching YouTube for music video...")
        youtube_url = self.youtube_client.search_music_video(song_obj.title, song_obj.artist)
        if youtube_url:
            logger.info("YouTube video found: %s", youtube_url)
            song_obj.youtube_url = youtube_url
        else:
            logger.info("No YouTube video found for: %s", song_obj.title)
        return song_obj, True
    
    def _process_single(self, item: str) -> Tuple[Optional[Song], bool]:
        return self.process_search_query(item)
    
    def _update_stats(self, stats: ProcessingStats, item: Optional[Song], success: bool):
        if success and item:
            stats.found += 1
        else:
            stats.not_found += 1
