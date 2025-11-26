from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd

from src.models.book import Book


class TranscriptionExporter:
    
    def __init__(self, transcriptions_dir: str = "../books-eater/transcriptions"):
        self.transcriptions_dir = Path(transcriptions_dir)
    
    def load_transcription(self, audio_filename: str) -> Optional[str]:
        json_filename = Path(audio_filename).stem + '.json'
        json_path = self.transcriptions_dir / json_filename
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('transcription', '')
        except Exception as e:
            print(f"      Error loading transcription from {json_path}: {e}")
            return None
    
    def export_to_excel(
        self, 
        books: List[Book], 
        excel_path: str,
        csv_path: Optional[str] = None
    ) -> Dict[str, int]:
        stats = {
            'total': len(books),
            'with_transcription': 0,
            'without_transcription': 0,
            'no_audio': 0
        }
        
        print(f"\n{'='*60}")
        print("Exportando transcripciones a Excel/CSV")
        print(f"{'='*60}\n")
        
        for idx, book in enumerate(books, 1):
            if book.disponibilidad in ["ENCONTRADO", "PARCIAL"]:
                audio_filename = f"books-eater_{book.numero:03d}_{book.titulo.replace(' ', '_')[:50]}.m4a"
                
                transcription = self.load_transcription(audio_filename)
                
                if transcription:
                    book.transcripcion = transcription
                    stats['with_transcription'] += 1
                    print(f"[{idx}/{stats['total']}] {book.titulo[:50]} - Transcripción cargada ({len(transcription)} chars)")
                else:
                    stats['without_transcription'] += 1
                    print(f"[{idx}/{stats['total']}] {book.titulo[:50]} - No se encontró transcripción")
            else:
                stats['no_audio'] += 1
        
        df = pd.DataFrame([book.to_dict() for book in books])
        
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\nExcel guardado: {excel_path}")
        
        if csv_path:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"CSV guardado: {csv_path}")
        
        return stats
    
    def print_statistics(self, stats: Dict[str, int]):
        print(f"\n{'='*60}")
        print("Estadísticas de Exportación:")
        print(f"{'='*60}")
        print(f"   Total de libros: {stats['total']}")
        print(f"   Con transcripción: {stats['with_transcription']}")
        print(f"   Sin transcripción: {stats['without_transcription']}")
        print(f"   Sin audio: {stats['no_audio']}")
        
        if stats['total'] > 0:
            success_rate = (stats['with_transcription'] / stats['total']) * 100
            print(f"\n   Tasa de éxito: {success_rate:.1f}%")
        print(f"{'='*60}\n")
