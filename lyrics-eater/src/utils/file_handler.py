"""File handling utilities for reading searches and writing results."""

import os
from typing import List, Optional
import pandas as pd
from openpyxl.styles import Alignment

from ..models.song import Song


class FileHandler:
    """
    Handles file I/O operations.
    """
    
    @staticmethod
    def load_searches(filename: str) -> Optional[List[str]]:
        """
        Load search queries from a text file.
        
        Args:
            filename: Path to the searches file
            
        Returns:
            List of search queries or None if file not found
        """
        try:
            if not os.path.exists(filename):
                return None
            
            with open(filename, 'r', encoding='utf-8') as f:
                searches = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]
            
            return searches if searches else None
            
        except Exception as e:
            print(f" Error reading {filename}: {e}")
            return None
    
    @staticmethod
    def save_to_excel(songs: List[Song], filename: str) -> bool:
        """
        Save songs to Excel file with formatting.
        
        Args:
            songs: List of Song objects
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = [song.to_dict() for song in songs]
            
            df = pd.DataFrame(data)
            
            df = df[['genero', 'artista', 'cancion', 'letras', 'enlace_genius', 'enlace_youtube', 'discografica']]
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Canciones')
                
                worksheet = writer.sheets['Canciones']
                
                column_widths = {
                    'A': 20,  # genero
                    'B': 25,  # artista
                    'C': 30,  # cancion
                    'D': 80,  # letras
                    'E': 50,  # enlace_genius
                    'F': 50,  # enlace_youtube
                    'G': 25,  # discografica
                }
                
                for col, width in column_widths.items():
                    worksheet.column_dimensions[col].width = width
                
                for row in range(2, len(songs) + 2):
                    cell = worksheet.cell(row=row, column=4)  # Column D (lyrics)
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            print(f" Excel saved successfully: {filename}")
            return True
            
        except Exception as e:
            print(f" Error saving Excel: {e}")
            return False
    
    @staticmethod
    def save_to_csv(songs: List[Song], filename: str) -> bool:
        """
        Save songs to CSV file (alternative format).
        
        Args:
            songs: List of Song objects
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = [song.to_dict() for song in songs]
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f" CSV saved successfully: {filename}")
            return True
            
        except Exception as e:
            print(f" Error saving CSV: {e}")
            return False
