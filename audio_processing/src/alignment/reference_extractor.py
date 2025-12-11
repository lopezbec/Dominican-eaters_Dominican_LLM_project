import os
from typing import Dict
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class ReferenceTextExtractor:
    
    def __init__(self, config: Dict):
        self.config = config
    
    def extract_text_from_excel(self, module_name: str) -> Dict:
        if module_name == 'lyrics-eater':
            return self._extract_lyrics()
        elif module_name == 'poems-eater':
            return self._extract_poems()
        else:
            raise ValueError(f"Unsupported module: {module_name}")
    
    def _extract_lyrics(self) -> Dict:
        module_config = self.config['modules']['lyrics-eater']
        excel_path = module_config['excel_path']
        lyrics_column = module_config['lyrics_column']
        url_column = module_config['url_column']
        reference_texts_dir = module_config['reference_texts_dir']
        
        os.makedirs(reference_texts_dir, exist_ok=True)
        
        if pd is None:
            raise ImportError("pandas and openpyxl required")
        
        df = pd.read_excel(excel_path)
        
        extracted = 0
        for idx, row in df.iterrows():
            url = row.get(url_column, '')
            lyrics = row.get(lyrics_column, '')
            
            if not isinstance(url, str) or not url.startswith('http'):
                continue
            
            if not isinstance(lyrics, str) or not lyrics.strip():
                continue
            
            filename = f"lyrics-eater_{idx:03d}.txt"
            output_path = os.path.join(reference_texts_dir, filename)
            
            from shared.utils.text_utils import clean_reference_text
            cleaned_lyrics = clean_reference_text(lyrics)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_lyrics)
            
            extracted += 1
        
        logger.info(f"Extracted {extracted} reference texts for lyrics-eater")
        return {'extracted': extracted}
    
    def _extract_poems(self) -> Dict:
        module_config = self.config['modules']['poems-eater']
        reference_texts_dir = module_config['reference_texts_dir']
        
        os.makedirs(reference_texts_dir, exist_ok=True)
        
        logger.warning("Poems reference texts extraction requires manual implementation")
        logger.info("Reference: poems-eater/src/utils/dominican_poems.py")
        
        return {'extracted': 0, 'note': 'Manual extraction required'}
    
    def extract_text_from_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
