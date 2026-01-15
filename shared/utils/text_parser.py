"""Text parsing utilities for content creation."""

from typing import Dict, List, Optional


class TextParser:
    """
    Utility class for parsing delimited text into structured data.
    
    Provides DRY solution for repeated parsing logic across models.
    """
    
    @staticmethod
    def parse_delimited(
        text: str, 
        field_names: List[str], 
        delimiter: str = '|'
    ) -> Optional[Dict[str, str]]:
        """
        Parse delimited text into a dictionary.
        
        Args:
            text: Text string with delimited values
            field_names: List of field names in order
            delimiter: Delimiter character (default: '|')
            
        Returns:
            Dictionary mapping field names to values, or None if parsing fails
            
        Example:
            >>> TextParser.parse_delimited("Title | Author | 2020", ["title", "author", "year"])
            {'title': 'Title', 'author': 'Author', 'year': '2020'}
        """
        try:
            parts = [p.strip() for p in text.split(delimiter)]
            if len(parts) >= len(field_names):
                return dict(zip(field_names, parts))
            return None
        except Exception:
            return None
