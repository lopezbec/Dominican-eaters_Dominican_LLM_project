"""Unified YouTube client for scraping videos across all content types."""

from typing import Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import scrapetube
import logging

from ..utils.text_utils import normalize_text, extract_last_name, extract_significant_words

logger = logging.getLogger(__name__)


@dataclass
class ContentSearchConfig:
    """
    Configuration for content-specific YouTube searches.
    
    Single Responsibility: Define search behavior for different content types.
    """
    search_queries_template: List[str]
    positive_keywords: List[str]
    negative_keywords: List[str]
    min_duration_seconds: int = 0
    max_duration_seconds: int = 7200


class ContentMatcher(ABC):
    """
    Abstract strategy for matching video titles to content.
    
    Single Responsibility: Define matching logic interface.
    """
    
    @abstractmethod
    def matches_content(self, video_title: str, content_title: str, content_author: str) -> bool:
        pass
    
    @abstractmethod
    def classify_content(self, title: str, duration: str) -> str:
        pass


class AudiobookMatcher(ContentMatcher):
    """
    Matcher strategy for audiobook content.
    
    Single Responsibility: Audiobook-specific matching and classification.
    """
    
    def matches_content(self, video_title: str, content_title: str, content_author: str) -> bool:
        video_normalized = normalize_text(video_title)
        book_normalized = normalize_text(content_title)
        author_normalized = normalize_text(content_author)
        
        author_lastname = extract_last_name(author_normalized)
        book_words = extract_significant_words(book_normalized)
        
        has_author = author_lastname in video_normalized
        
        if book_words:
            matching_words = sum(1 for word in book_words if word in video_normalized)
            has_title = matching_words >= len(book_words) * 0.5
        else:
            has_title = book_normalized in video_normalized
        
        return has_author and has_title
    
    def classify_content(self, title: str, duration: str) -> str:
        title_lower = title.lower()
        
        if 'completo' in title_lower or 'complete' in title_lower:
            if 'dramatización' in title_lower or 'dramatizado' in title_lower:
                return "Dramatización Completa"
            return "Lectura Completa"
        
        if 'audiolibro' in title_lower or 'audiobook' in title_lower:
            return "Narración Profesional"
        
        if 'dramatización' in title_lower or 'teatro' in title_lower:
            return "Dramatización"
        
        if any(word in title_lower for word in ['fragmento', 'capítulo', 'parte', 'extracto']):
            return "Fragmentos"
        
        if any(word in title_lower for word in ['análisis', 'reseña', 'resumen', 'comentario']):
            return "Análisis/Reseña"
        
        if duration != 'N/A':
            try:
                parts = duration.split(':')
                if len(parts) == 3:
                    minutes = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 2:
                    minutes = int(parts[0])
                else:
                    minutes = 0
                
                if minutes > 60:
                    return "Lectura Completa"
                elif minutes > 15:
                    return "Lectura Parcial"
                else:
                    return "Fragmentos"
            except (ValueError, TypeError):
                pass
        
        return "Lectura Amateur"


class MusicVideoMatcher(ContentMatcher):
    """
    Matcher strategy for music video content.
    
    Single Responsibility: Music video-specific matching (simple).
    """
    
    def matches_content(self, video_title: str, content_title: str, content_author: str) -> bool:
        return True
    
    def classify_content(self, title: str, duration: str) -> str:
        return "Music Video"


class PoemRecitationMatcher(ContentMatcher):
    """
    Matcher strategy for poem recitation content.
    
    Single Responsibility: Poem-specific matching and classification.
    """
    
    def matches_content(self, video_title: str, content_title: str, content_author: str) -> bool:
        video_normalized = normalize_text(video_title)
        poem_normalized = normalize_text(content_title)
        author_normalized = normalize_text(content_author)
        
        author_lastname = extract_last_name(author_normalized)
        poem_words = extract_significant_words(poem_normalized)
        
        has_author = author_lastname in video_normalized
        
        if poem_words:
            matching_words = sum(1 for word in poem_words if word in video_normalized)
            has_title = matching_words >= len(poem_words) * 0.3
        else:
            has_title = poem_normalized in video_normalized
        
        return has_author or has_title
    
    def classify_content(self, title: str, duration: str) -> str:
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['recitación', 'recita', 'recitando']):
            return "Recitación"
        elif any(word in title_lower for word in ['dramatización', 'drama', 'teatral', 'teatro']):
            return "Dramatización"
        elif any(word in title_lower for word in ['lectura', 'leyendo', 'lee']):
            return "Lectura"
        elif any(word in title_lower for word in ['performance', 'presentación', 'actuación']):
            return "Performance"
        elif any(word in title_lower for word in ['compilación', 'antología', 'colección']):
            return "Compilación"
        elif any(word in title_lower for word in ['documental', 'educativo', 'análisis']):
            return "Documental"
        elif any(word in title_lower for word in ['audio', 'audiopoesía']):
            return "Audiopoesía"
        elif any(word in title_lower for word in ['fragmento', 'extracto', 'parcial']):
            return "Fragmentos"
        else:
            return "Video Poético"


class YouTubeClient:
    """
    Unified YouTube scraper client with strategy pattern for different content types.
    
    Single Responsibility: YouTube video search and parsing.
    """
    
    def __init__(
        self,
        videos_per_search: int = 3,
        matcher: Optional[ContentMatcher] = None,
        config: Optional[ContentSearchConfig] = None
    ):
        self.videos_per_search = videos_per_search
        self.matcher = matcher or AudiobookMatcher()
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> ContentSearchConfig:
        return ContentSearchConfig(
            search_queries_template=[
                "{title} {author} audiolibro completo",
                "{title} {author} audiobook",
                "{title} {author} libro completo",
            ],
            positive_keywords=[
                'audiolibro', 'audiobook', 'libro completo', 'lectura',
                'narración', 'narrado', 'leído', 'dramatización',
            ],
            negative_keywords=[
                'resumen', 'summary', 'trailer', 'preview',
                'música', 'music', 'instrumental', 'karaoke',
            ],
            min_duration_seconds=30,
            max_duration_seconds=7200
        )
    
    def search_video(
        self,
        title: str,
        author: str,
        extra_queries: Optional[List[str]] = None
    ) -> Optional[Dict[str, str]]:
        """
        Search for a video on YouTube.
        
        Args:
            title: Content title
            author: Author/artist name
            extra_queries: Additional search queries to try
            
        Returns:
            Dictionary with video info if found, None otherwise
        """
        try:
            search_queries = [
                query.format(title=title, author=author)
                for query in self.config.search_queries_template
            ]
            
            if extra_queries:
                search_queries.extend(extra_queries)
            
            for query in search_queries:
                result = self._search_with_query(query, title, author)
                if result:
                    return result
            
            return None
            
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError) as e:
            logger.error("search_video failed for title=%s author=%s: %s", title, author, e)
            return None
    
    def _search_with_query(
        self,
        query: str,
        content_title: str,
        author: str
    ) -> Optional[Dict[str, str]]:
        """
        Execute a single search query on YouTube.
        
        Single Responsibility: Execute search and return first match.
        """
        try:
            videos = scrapetube.get_search(query, limit=self.videos_per_search, sleep=1)
            
            for video in videos:
                video_id = video.get('videoId')
                if not video_id:
                    continue
                
                title = video.get('title', {}).get('runs', [{}])[0].get('text', '')
                duration = self._parse_duration(video)
                
                if not self._is_valid_duration(duration):
                    continue
                
                if not self.matcher.matches_content(title, content_title, author):
                    continue
                
                if not self._is_likely_target_content(title):
                    continue
                
                content_type = self.matcher.classify_content(title, duration)
                
                return {
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'duration': duration,
                    'type': content_type,
                    'title': title
                }
            
            return None
            
        except (RuntimeError, ValueError, OSError) as e:
            logger.exception("Error during YouTube search for query=%s: %s", query, e)
            return None
        except (AttributeError, KeyError, IndexError, TypeError) as e:
            logger.error("Data parsing error during YouTube search for query=%s: %s", query, e)
            return None
        except Exception as e:
            logger.error("Unexpected error during YouTube search for query=%s: %s", query, e)
            return None
    
    def _parse_duration(self, video: dict) -> str:
        """
        Extract and format video duration.
        
        Single Responsibility: Duration parsing.
        """
        try:
            length_text = video.get('lengthText', {}).get('simpleText', 'N/A')
            return length_text if length_text else 'N/A'
        except (KeyError, AttributeError, TypeError, ValueError):
            return 'N/A'
    
    def _is_valid_duration(self, duration: str) -> bool:
        """
        Check if duration is within acceptable range.
        
        Single Responsibility: Duration validation.
        """
        if duration == 'N/A':
            return True
        
        try:
            parts = duration.split(':')
            if len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            else:
                return True
            
            return (self.config.min_duration_seconds <= seconds <= 
                    self.config.max_duration_seconds)
        except (ValueError, TypeError, IndexError):
            return True
    
    def _is_likely_target_content(self, title: str) -> bool:
        """
        Determine if a video is likely target content based on keywords.
        
        Single Responsibility: Keyword filtering.
        """
        title_lower = title.lower()
        
        has_positive = any(
            keyword in title_lower 
            for keyword in self.config.positive_keywords
        )
        
        has_negative = any(
            keyword in title_lower 
            for keyword in self.config.negative_keywords
        )
        
        return has_positive and not has_negative


def create_audiobook_client(videos_per_search: int = 3) -> YouTubeClient:
    """Factory function for audiobook YouTube client."""
    config = ContentSearchConfig(
        search_queries_template=[
            "{title} {author} audiolibro completo",
            "{title} {author} audiobook",
            "{title} {author} libro completo",
            "{title} audiolibro dominicano",
            "{author} {title} lectura"
        ],
        positive_keywords=[
            'audiolibro', 'audiobook', 'libro completo', 'lectura',
            'narración', 'narrado', 'leído', 'dramatización',
        ],
        negative_keywords=[
            'resumen', 'summary', 'trailer', 'preview',
            'música', 'music', 'instrumental',
        ],
        min_duration_seconds=60,
        max_duration_seconds=7200
    )
    return YouTubeClient(videos_per_search, AudiobookMatcher(), config)


def create_music_video_client() -> YouTubeClient:
    """Factory function for music video YouTube client."""
    config = ContentSearchConfig(
        search_queries_template=["{title} {author}"],
        positive_keywords=['video', 'official', 'music'],
        negative_keywords=[],
        min_duration_seconds=0,
        max_duration_seconds=600
    )
    return YouTubeClient(1, MusicVideoMatcher(), config)


def create_poem_recitation_client(videos_per_search: int = 3) -> YouTubeClient:
    """Factory function for poem recitation YouTube client."""
    config = ContentSearchConfig(
        search_queries_template=[
            "{title} {author} recitación",
            "{title} poema dominicano",
            "{author} {title} completo",
            "{title} {author} dramatización",
            "poema {title} recitado",
        ],
        positive_keywords=[
            'recitación', 'poema', 'recitado', 'dramatización',
            'lectura', 'performance', 'audiopoesía',
        ],
        negative_keywords=[
            'resumen', 'análisis crítico', 'tutorial',
        ],
        min_duration_seconds=30,
        max_duration_seconds=1200
    )
    return YouTubeClient(videos_per_search, PoemRecitationMatcher(), config)
