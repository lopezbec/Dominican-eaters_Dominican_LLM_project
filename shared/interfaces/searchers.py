from typing import Protocol, Dict, List, Optional


class YouTubeSearcher(Protocol):
    
    def search_audiobook(self, title: str, author: str) -> Optional[Dict]:
        ...
    
    def search_poem_recitation(
        self,
        title: str,
        author: str,
        genre: str
    ) -> Optional[Dict]:
        ...
    
    def search_music_video(self, title: str, artist: str) -> Optional[str]:
        ...


class GeniusSearcher(Protocol):
    
    def search(self, query: str) -> List[Dict]:
        ...
    
    def get_song_details(self, song_id: int):
        ...
    
    def scrape_lyrics(self, url: str) -> Optional[str]:
        ...
