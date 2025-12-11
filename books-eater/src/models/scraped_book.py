"""Data model for scraped books from librosdominicanos.com."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScrapedBook:
    """
    Represents a book scraped from librosdominicanos.com.
    """
    book_number: int
    title: str
    author: str
    year: str
    book_url: str
    pdf_url: Optional[str] = None
    pdf_downloaded: str = "No"
    content: str = ""
    youtube_url: str = "NO ENCONTRADO"
    youtube_duration: str = "N/A"
    youtube_content_type: str = "N/A"
    
    def to_dict(self) -> dict:
        """
        Convert book to dictionary for export.
        
        Returns:
            Dictionary with column names matching requirements
        """
        return {
            'Book Number': self.book_number,
            'Title': self.title,
            'Author': self.author,
            'Year': self.year,
            'PDF URL': self.pdf_url if self.pdf_url else "N/A",
            'PDF Downloaded': self.pdf_downloaded,
            'Content': self.content,
            'YouTube URL': self.youtube_url,
            'YouTube Duration': self.youtube_duration,
            'YouTube Content Type': self.youtube_content_type
        }
    
    def mark_pdf_downloaded(self):
        """Mark the PDF as successfully downloaded."""
        self.pdf_downloaded = "Yes"
    
    def set_youtube_info(self, url: str, duration: str, content_type: str):
        """
        Set YouTube video information.
        
        Args:
            url: YouTube video URL
            duration: Video duration
            content_type: Type of content
        """
        self.youtube_url = url
        self.youtube_duration = duration
        self.youtube_content_type = content_type
