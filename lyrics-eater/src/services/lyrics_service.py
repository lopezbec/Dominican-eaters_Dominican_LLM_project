"""Business logic for processing song lyrics requests."""

from typing import List, Tuple

from src.clients.genius_client import GeniusAPIClient
from src.clients.youtube_client import YouTubeAPIClient
from src.models.song import Song


class LyricsService:
    """
    Service for processing song search queries and fetching lyrics.
    """
    
    def __init__(self, genius_client: GeniusAPIClient, youtube_client: YouTubeAPIClient):
        """
        Initialize the service.
        
        Args:
            genius_client: Genius API client instance
            youtube_client: YouTube API client instance
        """
        self.genius_client = genius_client
        self.youtube_client = youtube_client
    
    def process_search_query(self, query: str) -> Tuple[Song, bool]:
        """
        Process a single search query and return song with lyrics.
        
        Args:
            query: Search query (e.g., "Obsesion Aventura")
            
        Returns:
            Tuple of (Song object or None, success boolean)
        """
        results = self.genius_client.search(query)
        
        if not results:
            print(f"   No results found for '{query}'")
            return None, False
        
        first_result = results[0]
        print(f"   Found: {first_result['title']} - {first_result['artist']}")
        
        song = self.genius_client.get_song_details(first_result['id'])
        
        if not song:
            print(f"   Could not fetch details")
            return None, False
        
        print(f"    Album: {song.album}")
        print(f"    Genre(s): {song.genres}")
        print(f"    Label: {song.label}")
        
        print(f"     Fetching lyrics...")
        lyrics = self.genius_client.scrape_lyrics(song.url)
        
        if lyrics:
            print(f"     Lyrics obtained ({len(lyrics)} chars)")
            song.lyrics = lyrics
        else:
            print(f"      Could not obtain lyrics")
            song.lyrics = "N/A"
        
        # Fetch YouTube link
        print(f"     Searching YouTube...")
        youtube_url = self.youtube_client.search_music_video(song.title, song.artist)
        if youtube_url:
            print(f"     ✓ YouTube link found")
            song.youtube_url = youtube_url
        
        return song, True
    
    def process_multiple_queries(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> Tuple[List[Song], int, int]:
        """
        Process multiple search queries.
        
        Args:
            queries: List of search queries
            show_progress: Whether to show progress messages
            
        Returns:
            Tuple of (list of Songs, successful count, failed count)
        """
        songs = []
        successful = 0
        failed = 0
        total = len(queries)
        
        for idx, query in enumerate(queries, 1):
            try:
                if show_progress:
                    print(f"\n[{idx}/{total}] Searching: '{query}'...")
                
                song, success = self.process_search_query(query)
                
                if success and song:
                    songs.append(song)
                    successful += 1
                else:
                    failed += 1
                    
            except KeyboardInterrupt:
                print("\n\n  Process interrupted by user")
                print(f"Songs processed so far: {successful}")
                break
            except Exception as e:
                print(f"   Unexpected error: {e}")
                failed += 1
                continue
        
        return songs, successful, failed
