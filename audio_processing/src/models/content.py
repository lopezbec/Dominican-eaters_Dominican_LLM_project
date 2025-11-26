"""Data models for Dominican content (songs, poems, books)."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


class ContentType(Enum):
    """Type of Dominican content."""
    SONG = "song"
    POEM = "poem"
    BOOK = "book"


class VerificationStatus(Enum):
    """Status of content verification."""
    PENDING = "pending"
    SUCCESS = "success"
    NO_URL = "no_url"
    NO_TEXT = "no_reference_text"
    DOWNLOAD_FAILED = "download_failed"
    TRANSCRIPTION_FAILED = "transcription_failed"


class AlignmentQuality(Enum):
    """Quality level of audio-text alignment."""
    EXCELLENT = "excellent"  # >= 80% similarity
    GOOD = "good"            # >= 60% similarity
    FAIR = "fair"            # >= 40% similarity
    POOR = "poor"            # < 40% similarity


@dataclass
class Content:
    """Base class for Dominican content."""
    index: int
    title: str
    author: str
    youtube_url: Optional[str] = None
    content_type: ContentType = ContentType.SONG
    
    def __post_init__(self):
        if isinstance(self.content_type, str):
            self.content_type = ContentType(self.content_type)


@dataclass
class Song(Content):
    """Dominican song with lyrics."""
    lyrics: Optional[str] = None
    clean_lyrics: Optional[str] = None
    artist: Optional[str] = None
    content_type: ContentType = field(default=ContentType.SONG, init=False)
    
    def __post_init__(self):
        super().__post_init__()
        if self.artist and not self.author:
            self.author = self.artist


@dataclass
class Poem(Content):
    """Dominican poem."""
    text: Optional[str] = None
    content_type: ContentType = field(default=ContentType.POEM, init=False)


@dataclass
class Book(Content):
    """Dominican audiobook."""
    text: Optional[str] = None
    duration: Optional[float] = None
    content_type: ContentType = field(default=ContentType.BOOK, init=False)


@dataclass
class TranscriptionResult:
    text: str
    language: Optional[str] = None
    segments: List[Dict] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result from audio-text alignment verification."""
    index: int
    content: Content
    status: VerificationStatus
    similarity: float = 0.0
    wer: float = 1.0
    whisper_text_length: Optional[int] = None
    reference_text_length: Optional[int] = None
    alignment_quality: Optional[AlignmentQuality] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = VerificationStatus(self.status)
        if isinstance(self.alignment_quality, str):
            self.alignment_quality = AlignmentQuality(self.alignment_quality)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'index': self.index,
            'author': self.content.author,
            'title': self.content.title,
            'youtube_url': self.content.youtube_url,
            'content_type': self.content.content_type.value,
            'status': self.status.value,
            'similarity': round(self.similarity, 3),
            'wer': round(self.wer, 3),
            'whisper_text_length': self.whisper_text_length,
            'reference_text_length': self.reference_text_length,
            'alignment_quality': self.alignment_quality.value if self.alignment_quality else None,
            'error': self.error
        }
