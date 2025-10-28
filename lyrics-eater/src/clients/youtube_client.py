"""YouTube scraper client for fetching music video links."""

from typing import Optional
import scrapetube


class YouTubeAPIClient:
    """
    Client for searching YouTube videos using scrapetube (no API key needed).
    """
    
    def __init__(self):
        """
        Initialize YouTube scraper client.
        """
        pass
    
    def search_music_video(self, title: str, artist: str) -> Optional[str]:
        """
        Search for a music video on YouTube using scrapetube.
        
        Args:
            title: Song title
            artist: Artist name
            
        Returns:
            YouTube video URL if found, None otherwise
        """
        try:
            query = f"{title} {artist}"
            
            videos = scrapetube.get_search(query, limit=1, sleep=0)
            
            for video in videos:
                video_id = video.get('videoId')
                if video_id:
                    return f"https://www.youtube.com/watch?v={video_id}"
            
            return None
            
        except Exception as e:
            return None
