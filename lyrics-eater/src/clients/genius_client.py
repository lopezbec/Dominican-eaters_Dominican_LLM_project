"""Genius API client for fetching song data."""

import re
import os
from typing import List, Optional, Dict
import requests
from bs4 import BeautifulSoup

from ..models.song import Song
from ..utils.config import config


class GeniusAPIClient:
    """
    Client for interacting with Genius API.
    """
    
    def __init__(self, access_token: str):
        """
        Initialize the Genius API client.
        
        Args:
            access_token: Genius API access token
        """
        self.access_token = access_token
        self.base_url = config.GENIUS_BASE_URL
        self._session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with default headers.
        """
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "LyricsEater/1.0"
        })
        return session
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: int = None
    ) -> Optional[Dict]:
        """
        Make a GET request to Genius API.
        
        Args:
            endpoint: API endpoint (e.g., '/search')
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            API response data or None on error
        """
        timeout = timeout or config.API_TIMEOUT
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self._session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            if result.get("meta", {}).get("status") == 200:
                return result.get("response")
            else:
                print(f" API Error: {result}")
                return None
                
        except requests.exceptions.Timeout:
            print(f" Timeout: Request exceeded {timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            print(f" Request Error: {e}")
            return None
    
    def search(self, query: str, per_page: int = None) -> List[Dict]:
        """
        Search for songs on Genius.
        
        Args:
            query: Search query (e.g., "Obsesion Aventura")
            per_page: Number of results per page
            
        Returns:
            List of song dictionaries
        """
        per_page = per_page or config.RESULTS_PER_PAGE
        params = {"q": query, "per_page": per_page}
        response = self._make_request("/search", params=params)
        
        if not response:
            return []
        
        hits = response.get("hits", [])
        return [
            {
                "id": hit.get("result", {}).get("id"),
                "title": hit.get("result", {}).get("title"),
                "artist": hit.get("result", {}).get("primary_artist", {}).get("name"),
                "url": hit.get("result", {}).get("url")
            }
            for hit in hits
        ]
    
    def get_song_details(self, song_id: int) -> Optional[Song]:
        """
        Get detailed information about a song.
        
        Args:
            song_id: Genius song ID
            
        Returns:
            Song object or None on error
        """
        response = self._make_request(f"/songs/{song_id}")
        
        if not response:
            return None
        
        song_data = response.get("song", {})
        album = song_data.get("album") or {}
        tags = song_data.get("tags", [])
        
        return Song(
            song_id=song_data.get("id"),
            title=song_data.get("title"),
            artist=song_data.get("primary_artist", {}).get("name"),
            url=song_data.get("url"),
            genres=", ".join([tag.get("name", "") for tag in tags]) if tags else "N/A",
            label=album.get("label", "N/A") if album else "N/A",
            album=album.get("name", "N/A") if album else "N/A",
            release_date=song_data.get("release_date_for_display", "N/A"),
            lyrics="" 
        )
    
    def scrape_lyrics(self, url: str, timeout: int = None) -> str:
        """
        Scrape lyrics from a Genius song page.
        
        Args:
            url: Genius song URL
            timeout: Request timeout in seconds
            
        Returns:
            Cleaned lyrics text or empty string on error
        """
        timeout = timeout or config.SCRAPING_TIMEOUT
        
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            lyrics_divs = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
            
            if not lyrics_divs:
                print(f"  No lyrics found at {url}")
                return ""
            
            # Extract and clean lyrics
            lyrics = '\n'.join([div.get_text(separator="\n") for div in lyrics_divs])
            lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)  # Remove [Verse], [Chorus], etc.
            lyrics = os.linesep.join([line for line in lyrics.splitlines() if line.strip()])
            
            return lyrics
            
        except requests.exceptions.Timeout:
            print(f"  Timeout: Scraping exceeded {timeout}s")
            return ""
        except requests.exceptions.RequestException as e:
            print(f" Scraping Error: {e}")
            return ""
        except Exception as e:
            print(f" Unexpected Error: {e}")
            return ""
