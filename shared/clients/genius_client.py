"""Genius API client for fetching song data and lyrics."""

import re
import requests
from typing import Optional, Dict, List
from bs4 import BeautifulSoup

from .base_client import BaseAPIClient, logger


class GeniusAPIClient(BaseAPIClient):
    
    def __init__(
        self,
        access_token: str,
        base_url: str = "https://api.genius.com",
        api_timeout: int = 10,
        scraping_timeout: int = 15,
        results_per_page: int = 5
    ):
        super().__init__(
            base_url=base_url,
            api_key=access_token,
            timeout=api_timeout,
            max_retries=3,
            backoff_factor=0.5
        )
        self.scraping_timeout = scraping_timeout
        self.results_per_page = results_per_page
    
    def _get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}
    
    def _make_api_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> Optional[Dict]:
        response = self.get(endpoint, params=params, timeout=timeout)
        
        if not response:
            return None
        
        try:
            result = response.json()
            
            if result.get("meta", {}).get("status") == 200:
                return result.get("response")
            else:
                logger.error("API Error: %s", result)
                return None
        except ValueError:
            logger.error("JSON Decode Error while parsing Genius response")
            return None
    
    def search(self, query: str, per_page: Optional[int] = None) -> List[Dict]:
        per_page = per_page or self.results_per_page
        params = {"q": query, "per_page": per_page}
        response = self._make_api_request("/search", params=params)
        
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
    
    def get_song_details(self, song_id: int) -> Optional[Dict]:
        response = self._make_api_request(f"/songs/{song_id}")
        
        if not response:
            return None
        
        song_data = response.get("song", {})
        album = song_data.get("album") or {}
        tags = song_data.get("tags", [])
        
        return {
            "id": song_data.get("id"),
            "title": song_data.get("title"),
            "artist": song_data.get("primary_artist", {}).get("name"),
            "url": song_data.get("url"),
            "genres": ", ".join([tag.get("name", "") for tag in tags]) if tags else "N/A",
            "label": album.get("label", "N/A") if album else "N/A",
            "album": album.get("name", "N/A") if album else "N/A",
            "release_date": song_data.get("release_date_for_display", "N/A"),
        }
    
    def scrape_lyrics(self, url: str, timeout: Optional[int] = None) -> str:
        timeout = timeout or self.scraping_timeout
        
        try:
            # Reuse session from BaseHTTPClient to get retries, headers and consistent timeouts
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            lyrics_divs = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
            
            if not lyrics_divs:
                logger.info("No lyrics found at %s", url)
                return ""
            
            lyrics = '\n'.join([div.get_text(separator="\n") for div in lyrics_divs])
            lyrics = self._clean_lyrics(lyrics)
            
            return lyrics
            
        except requests.exceptions.Timeout:
            logger.warning("Timeout: Scraping exceeded %ss for %s", timeout, url)
            return ""
        except requests.exceptions.RequestException as e:
            logger.error("Scraping Error: %s", e)
            return ""
        except Exception:
            logger.exception("Unexpected error while scraping %s", url)
            return ""
    
    def _clean_lyrics(self, lyrics: str) -> str:
        import os
        lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
        lyrics = os.linesep.join([line for line in lyrics.splitlines() if line.strip()])
        return lyrics
