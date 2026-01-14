"""Adapter classes to map concrete clients to Protocol interfaces."""

from typing import Dict, List, Optional
from ..interfaces.searchers import YouTubeSearcher, GeniusSearcher
from .youtube_client import YouTubeClient, create_audiobook_client, create_music_video_client, create_poem_recitation_client
from .genius_client import GeniusAPIClient


class YouTubeSearcherAdapter:
    """
    Adapter that makes YouTubeClient compatible with YouTubeSearcher protocol.
    
    Single Responsibility: Protocol compliance adapter.
    """
    
    def __init__(self, client: Optional[YouTubeClient] = None):
        self._audiobook_client = client or create_audiobook_client()
        self._music_client = create_music_video_client()
        self._poem_client = create_poem_recitation_client()
    
    def search_audiobook(self, title: str, author: str) -> Optional[Dict]:
        """Search for audiobook content."""
        result = self._audiobook_client.search_video(title, author)
        if result:
            return {
                'url': result['url'],
                'duration': result['duration'], 
                'title': result['title'],
                'type': result['type']
            }
        return None
    
    def search_poem_recitation(
        self,
        title: str,
        author: str,
        genre: str
    ) -> Optional[Dict]:
        """Search for poem recitation content."""
        extra_queries = [f"{title} {author} {genre}"] if genre and genre.lower() != 'n/a' else None
        result = self._poem_client.search_video(title, author, extra_queries)
        if result:
            return {
                'url': result['url'],
                'duration': result['duration'],
                'title': result['title'], 
                'type': result['type']
            }
        return None
    
    def search_music_video(self, title: str, artist: str) -> Optional[str]:
        """Search for music video content."""
        result = self._music_client.search_video(title, artist)
        return result['url'] if result else None


class GeniusSearcherAdapter:
    """
    Adapter that makes GeniusAPIClient compatible with GeniusSearcher protocol.
    
    Single Responsibility: Protocol compliance adapter.
    """
    
    def __init__(self, client: GeniusAPIClient):
        self._client = client
    
    def search(self, query: str) -> List[Dict]:
        """Search for songs using Genius API."""
        return self._client.search(query)
    
    def get_song_details(self, song_id: int):
        """Get detailed information about a song."""
        return self._client.get_song_details(song_id)
    
    def scrape_lyrics(self, url: str) -> Optional[str]:
        """Scrape lyrics from a Genius URL."""
        lyrics = self._client.scrape_lyrics(url)
        return lyrics if lyrics else None