"""YouTube scraper client for finding Dominican audiobooks."""

from typing import Optional, List, Dict
import scrapetube
import re


class YouTubeClient:
    """
    Client for searching YouTube audiobooks using scrapetube (no API key needed).
    Specialized in finding Dominican literature audiobooks.
    """
    
    def __init__(self, videos_per_search: int = 3):
        """
        Initialize YouTube scraper client.
        
        Args:
            videos_per_search: Number of videos to analyze per search
        """
        self.videos_per_search = videos_per_search
    
    def search_audiobook(self, title: str, author: str) -> Optional[Dict[str, str]]:
        """
        Search for an audiobook on YouTube.
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            Dictionary with video info if found, None otherwise
            Format: {
                'url': str,
                'duration': str,
                'type': str (content type),
                'title': str (video title)
            }
        """
        try:
            # Try different search strategies
            search_queries = [
                f"{title} {author} audiolibro completo",
                f"{title} {author} audiobook",
                f"{title} {author} libro completo",
                f"{title} audiolibro dominicano",
                f"{author} {title} lectura"
            ]
            
            for query in search_queries:
                result = self._search_with_query(query, title, author)
                if result:
                    return result
            
            return None
            
        except Exception as e:
            return None
    
    def _search_with_query(self, query: str, book_title: str, author: str) -> Optional[Dict[str, str]]:
        """
        Execute a single search query on YouTube.
        
        Args:
            query: Search query
            book_title: Original book title to match
            author: Original author name to match
            
        Returns:
            Video info dictionary or None
        """
        try:
            videos = scrapetube.get_search(query, limit=self.videos_per_search, sleep=1)
            
            for video in videos:
                video_id = video.get('videoId')
                if not video_id:
                    continue
                
                # Get video metadata
                title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                duration = self._parse_duration(video)
                
                # CRITICAL: First verify that the video matches the book and author
                if not self._matches_book(title, book_title, author):
                    continue
                
                # Then check if it's an audiobook
                if not self._is_likely_audiobook(title):
                    continue
                
                # Classify content type
                content_type = self._classify_content(title, duration)
                
                return {
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'duration': duration,
                    'type': content_type,
                    'title': title
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _parse_duration(self, video: dict) -> str:
        """
        Extract and format video duration.
        
        Args:
            video: Video metadata from scrapetube
            
        Returns:
            Formatted duration string (e.g., "1:23:45")
        """
        try:
            length_text = video.get('lengthText', {}).get('simpleText', 'N/A')
            return length_text if length_text else 'N/A'
        except Exception:
            return 'N/A'
    
    def _classify_content(self, title: str, duration: str) -> str:
        """
        Classify the type of audiobook content based on title and duration.
        
        Args:
            title: Video title
            duration: Video duration
            
        Returns:
            Content type classification
        """
        title_lower = title.lower()
        
        # Check for complete audiobook indicators
        if 'completo' in title_lower or 'complete' in title_lower:
            if 'dramatización' in title_lower or 'dramatizado' in title_lower:
                return "Dramatización Completa"
            return "Lectura Completa"
        
        # Check for professional narration
        if 'audiolibro' in title_lower or 'audiobook' in title_lower:
            return "Narración Profesional"
        
        # Check for dramatizations
        if 'dramatización' in title_lower or 'teatro' in title_lower:
            return "Dramatización"
        
        # Check for fragments
        if any(word in title_lower for word in ['fragmento', 'capítulo', 'parte', 'extracto']):
            return "Fragmentos"
        
        # Check for analysis/review
        if any(word in title_lower for word in ['análisis', 'reseña', 'resumen', 'comentario']):
            return "Análisis/Reseña"
        
        # Default classification based on duration
        if duration != 'N/A':
            try:
                # Parse duration to minutes
                parts = duration.split(':')
                if len(parts) == 3:  # H:M:S
                    minutes = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 2:  # M:S
                    minutes = int(parts[0])
                else:
                    minutes = 0
                
                if minutes > 60:  # More than 1 hour
                    return "Lectura Completa"
                elif minutes > 15:
                    return "Lectura Parcial"
                else:
                    return "Fragmentos"
            except Exception:
                pass
        
        return "Lectura Amateur"
    
    def _is_likely_audiobook(self, title: str) -> bool:
        """
        Determine if a video is likely to be an audiobook.
        
        Args:
            title: Video title
            
        Returns:
            True if likely an audiobook, False otherwise
        """
        title_lower = title.lower()
        
        # Positive indicators
        positive_keywords = [
            'audiolibro', 'audiobook', 'libro completo', 'lectura',
            'narración', 'narrado', 'leído', 'dramatización',
            'audio libro', 'voz humana', 'leer'
        ]
        
        # Negative indicators (filter these out)
        negative_keywords = [
            'resumen', 'summary', 'trailer', 'preview',
            'música', 'music', 'instrumental', 'karaoke',
            'video oficial', 'official video', 'lyrics',
            'tutorial', 'how to', 'como'
        ]
        
        # Check for positive keywords
        has_positive = any(keyword in title_lower for keyword in positive_keywords)
        
        # Check for negative keywords
        has_negative = any(keyword in title_lower for keyword in negative_keywords)
        
        return has_positive and not has_negative
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing accents and special characters.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        import unicodedata
        # Remove accents
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        # Convert to lowercase and remove extra spaces
        text = ' '.join(text.lower().split())
        return text
    
    def _matches_book(self, video_title: str, book_title: str, author: str) -> bool:
        """
        Verify that the video title matches the book title and author.
        
        Args:
            video_title: Title of the YouTube video
            book_title: Title of the book we're searching for
            author: Author of the book
            
        Returns:
            True if the video matches the book, False otherwise
        """
        # Normalize all texts for comparison
        video_normalized = self._normalize_text(video_title)
        book_normalized = self._normalize_text(book_title)
        author_normalized = self._normalize_text(author)
        
        # Extract author's last name (usually the most distinctive part)
        author_parts = author_normalized.split()
        author_lastname = author_parts[-1] if author_parts else author_normalized
        
        # Extract main words from book title (ignore common words)
        common_words = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'y', 'o', 'en', 'a', 'para'}
        book_words = [word for word in book_normalized.split() if word not in common_words and len(word) > 2]
        
        # Check if video contains author's last name
        has_author = author_lastname in video_normalized
        
        # Check if video contains significant words from book title
        # At least 50% of the significant words should be present
        if book_words:
            matching_words = sum(1 for word in book_words if word in video_normalized)
            has_title = matching_words >= len(book_words) * 0.5
        else:
            # If no significant words, check for exact book title
            has_title = book_normalized in video_normalized
        
        # Both author and title must be present
        return has_author and has_title
    
    def search_multiple_strategies(self, title: str, author: str) -> List[Dict[str, str]]:
        """
        Search using multiple strategies and return all results.
        
        Args:
            title: Book title
            author: Author name
            
        Returns:
            List of video info dictionaries
        """
        results = []
        
        search_queries = [
            f"{title} {author} audiolibro",
            f"{title} {author} completo",
            f"{author} {title}",
            f"libro {title} audio",
        ]
        
        for query in search_queries:
            try:
                videos = scrapetube.get_search(query, limit=2, sleep=1)
                
                for video in videos:
                    video_id = video.get('videoId')
                    if video_id:
                        title_video = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                        duration = self._parse_duration(video)
                        
                        # CRITICAL: Verify that the video matches the book and author
                        if not self._matches_book(title_video, title, author):
                            continue
                        
                        # Then check if it's an audiobook
                        if not self._is_likely_audiobook(title_video):
                            continue
                        
                        content_type = self._classify_content(title_video, duration)
                        
                        results.append({
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'duration': duration,
                            'type': content_type,
                            'title': title_video
                        })
            except Exception:
                continue
        
        return results
