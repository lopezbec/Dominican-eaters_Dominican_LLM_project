from typing import List, Optional, TypeVar, Callable, Type
import os
import pandas as pd
from pathlib import Path
import logging

from shared.models.base_content import BaseContent
from shared.services.excel_formatter import save_books_to_excel, save_songs_to_excel, save_poems_to_excel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseContent)


class FileHandler:
    
    @staticmethod
    def load_items_from_file(
        filename: str,
        content_class: Type[T]
    ) -> Optional[List[T]]:
        try:
            if not os.path.exists(filename):
                return None
            
            items = []
            with open(filename, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f, 1):
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    item = content_class.create_from_text(idx, line)
                    if item:
                        items.append(item)
            
            return items if items else None
            
        except Exception as e:
            logger.error("Error reading %s: %s", filename, e)
            return None
    
    @staticmethod
    def load_searches(filename: str) -> Optional[List[str]]:
        try:
            if not os.path.exists(filename):
                return None
            
            searches = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    searches.append(line)
            
            return searches if searches else None
            
        except Exception as e:
            logger.error("Error reading %s: %s", filename, e)
            return None
    
    @staticmethod
    def save_to_excel(items: List[BaseContent], filename: str) -> bool:
        if not items:
            return False
        
        # Ensure parent directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        first_item = items[0]
        type_name = type(first_item).__name__.lower()
        if 'book' in type_name:
            return save_books_to_excel(items, filename)
        elif 'song' in type_name:
            return save_songs_to_excel(items, filename)
        elif 'poem' in type_name:
            return save_poems_to_excel(items, filename)
        else:
            data = [item.to_dict() for item in items]
            df = pd.DataFrame(data)
            df.to_excel(filename, index=False)
            return True
    
    @staticmethod
    def save_to_csv(items: List[BaseContent], filename: str) -> bool:
        try:
            if not items:
                return False
            
            # Ensure parent directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            data = [item.to_dict() for item in items]
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            return True
            
        except Exception as e:
            logger.error("Error saving CSV %s: %s", filename, e)
            return False
