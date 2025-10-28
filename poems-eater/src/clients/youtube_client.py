"""YouTube scraper client for fetching poem recitation videos."""

from typing import Optional, Dict, List
import scrapetube
import re


class YouTubeClient:
    """
    Client for searching YouTube videos of Dominican poem recitations using scrapetube.
    No API key needed!
    """
    
    def __init__(self, videos_per_search: int = 3):
        """
        Initialize YouTube scraper client.
        
        Args:
            videos_per_search: Number of videos to fetch per search query
        """
        self.videos_per_search = videos_per_search
    
    def _parse_duration(self, length_seconds: int) -> str:
        """
        Convert seconds to MM:SS format.
        
        Args:
            length_seconds: Duration in seconds
            
        Returns:
            Duration in MM:SS format
        """
        try:
            minutes = length_seconds // 60
            seconds = length_seconds % 60
            return f"{minutes}:{seconds:02d}"
        except:
            return "N/A"
    
    def _determine_content_type(self, title: str, description: str = "") -> str:
        """
        Determine type of content based on video title and description.
        
        Args:
            title: Video title
            description: Video description
            
        Returns:
            Content type classification
        """
        title_lower = title.lower()
        desc_lower = description.lower()
        combined = f"{title_lower} {desc_lower}"
        
        if any(word in combined for word in ['recitación', 'recita', 'recitando']):
            return "Recitación"
        elif any(word in combined for word in ['dramatización', 'drama', 'teatral', 'teatro']):
            return "Dramatización"
        elif any(word in combined for word in ['lectura', 'leyendo', 'lee']):
            return "Lectura"
        elif any(word in combined for word in ['performance', 'presentación', 'actuación']):
            return "Performance"
        elif any(word in combined for word in ['compilación', 'antología', 'colección']):
            return "Compilación"
        elif any(word in combined for word in ['documental', 'educativo', 'análisis']):
            return "Documental"
        elif any(word in combined for word in ['audio', 'audiopoesía']):
            return "Audiopoesía"
        elif any(word in combined for word in ['fragmento', 'extracto', 'parcial']):
            return "Fragmentos"
        else:
            return "Video Poético"
    
    def _estimate_quality(self, title: str, view_count: int = 0) -> str:
        """
        Estimate video quality based on title and metrics.
        
        Args:
            title: Video title
            view_count: Number of views
            
        Returns:
            Quality estimation (Excelente, Buena, Aceptable, Baja)
        """
        title_lower = title.lower()
        
        # Check for quality indicators
        professional_keywords = ['profesional', 'oficial', 'hd', 'alta calidad', 'audio limpio']
        amateur_keywords = ['celular', 'phone', 'baja calidad', 'audio malo']
        
        is_professional = any(kw in title_lower for kw in professional_keywords)
        is_amateur = any(kw in title_lower for kw in amateur_keywords)
        
        if is_professional or view_count > 10000:
            return "Excelente"
        elif is_amateur or view_count < 100:
            return "Aceptable"
        else:
            return "Buena"
    
    def search_poem_recitation(
        self, 
        titulo: str, 
        autor: str,
        genero: str = ""
    ) -> Optional[Dict[str, str]]:
        """
        Search for a poem recitation on YouTube using multiple strategies.
        
        Args:
            titulo: Poem title
            autor: Author name
            genero: Poem genre (optional)
            
        Returns:
            Dictionary with video info if found, None otherwise
        """
        # Try multiple search patterns for better results
        search_patterns = [
            f"{titulo} {autor} recitación",
            f"{titulo} poema dominicano",
            f"{autor} {titulo} completo",
            f"{titulo} {autor} dramatización",
            f"poema {titulo} recitado",
            f"{autor} poesía dominicana",
        ]
        
        # If genre is provided, add genre-specific searches
        if genero and genero != "N/A":
            search_patterns.insert(2, f"{titulo} {autor} poema {genero}")
        
        for query in search_patterns:
            try:
                videos = scrapetube.get_search(
                    query, 
                    limit=self.videos_per_search,
                    sleep=1  # Be respectful with requests
                )
                
                for video in videos:
                    video_id = video.get('videoId')
                    if not video_id:
                        continue
                    
                    title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                    length_seconds = int(video.get('lengthText', {}).get('simpleText', '0').replace(':', ''))
                    
                    # Skip very short videos (likely not full recitations)
                    if length_seconds and length_seconds < 30:
                        continue
                    
                    # Skip very long videos (likely compilations or unrelated)
                    if length_seconds and length_seconds > 1200:  # 20 minutes
                        continue
                    
                    duration = self._parse_duration(length_seconds) if length_seconds else "N/A"
                    content_type = self._determine_content_type(title)
                    
                    # Get view count for quality estimation
                    view_count_text = video.get('viewCountText', {}).get('simpleText', '0')
                    view_count = self._extract_view_count(view_count_text)
                    quality = self._estimate_quality(title, view_count)
                    
                    # Try to extract recitador from title
                    recitador = self._extract_recitador(title)
                    
                    return {
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'duration': duration,
                        'type': content_type,
                        'recitador': recitador,
                        'quality': quality,
                        'title': title
                    }
                    
            except Exception as e:
                # Continue to next search pattern if this one fails
                continue
        
        return None
    
    def _extract_view_count(self, view_text: str) -> int:
        """
        Extract numeric view count from text like "1.2K views" or "1,234 visualizaciones".
        
        Args:
            view_text: View count text
            
        Returns:
            Numeric view count
        """
        try:
            # Remove non-numeric characters except K, M, and digits
            cleaned = re.sub(r'[^\d.KM]', '', view_text.upper())
            
            if 'K' in cleaned:
                return int(float(cleaned.replace('K', '')) * 1000)
            elif 'M' in cleaned:
                return int(float(cleaned.replace('M', '')) * 1000000)
            else:
                return int(cleaned) if cleaned else 0
        except:
            return 0
    
    def _extract_recitador(self, title: str) -> str:
        """
        Try to extract recitador name from video title.
        
        Args:
            title: Video title
            
        Returns:
            Recitador name or "N/A"
        """
        # Common patterns in Spanish
        patterns = [
            r'recitado por (.+?)(?:\||$)',
            r'por (.+?)(?:\||$)',
            r'voz de (.+?)(?:\||$)',
            r'narrado por (.+?)(?:\||$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up common suffixes
                name = re.sub(r'\s*[-–—]\s*.*$', '', name)
                return name[:50]  # Limit length
        
        return "N/A"
