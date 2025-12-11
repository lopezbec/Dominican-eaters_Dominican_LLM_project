from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import pandas as pd

from shared.models.base_content import BaseContent


class TranscriptionExporter:
    
    def __init__(self, transcriptions_dir: str, module_prefix: str = ""):
        self.transcriptions_dir = Path(transcriptions_dir)
        self.module_prefix = module_prefix
    
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
        content_items: List[BaseContent], 
        excel_path: str,
        csv_path: Optional[str] = None,
        content_type_name: str = "items",
        audio_filename_generator: Optional[callable] = None
    ) -> Dict[str, int]:
        stats = {
            'total': len(content_items),
            'with_transcription': 0,
            'without_transcription': 0,
            'no_audio': 0
        }
        
        print(f"\n{'='*60}")
        print("Exportando transcripciones a Excel/CSV")
        print(f"{'='*60}\n")
        
        for idx, item in enumerate(content_items, 1):
            if hasattr(item, 'disponibilidad') and item.disponibilidad in ["ENCONTRADO", "PARCIAL"]:
                if audio_filename_generator:
                    audio_filename = audio_filename_generator(item)
                else:
                    audio_filename = self._default_filename_generator(item)
                
                transcription = self.load_transcription(audio_filename)
                
                if transcription:
                    item.transcripcion = transcription
                    stats['with_transcription'] += 1
                    display_name = item.get_display_name()[:50]
                    print(f"[{idx}/{stats['total']}] {display_name} - Transcripción cargada ({len(transcription)} chars)")
                else:
                    stats['without_transcription'] += 1
                    display_name = item.get_display_name()[:50]
                    print(f"[{idx}/{stats['total']}] {display_name} - No se encontró transcripción")
            else:
                stats['no_audio'] += 1
        
        df = pd.DataFrame([item.to_dict() for item in content_items])
        
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"\nExcel guardado: {excel_path}")
        
        if csv_path:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"CSV guardado: {csv_path}")
        
        return stats
    
    def _default_filename_generator(self, item: BaseContent) -> str:
        title = item.get_display_name().replace(' ', '_')[:50]
        if hasattr(item, 'numero'):
            return f"{self.module_prefix}_{item.numero:03d}_{title}.m4a"
        return f"{self.module_prefix}_{title}.m4a"
    
    def print_statistics(self, stats: Dict[str, int], content_type_name: str = "items"):
        print(f"\n{'='*60}")
        print("Estadísticas de Exportación:")
        print(f"{'='*60}")
        print(f"   Total de {content_type_name}: {stats['total']}")
        print(f"   Con transcripción: {stats['with_transcription']}")
        print(f"   Sin transcripción: {stats['without_transcription']}")
        print(f"   Sin audio: {stats['no_audio']}")
        
        if stats['total'] > 0:
            success_rate = (stats['with_transcription'] / stats['total']) * 100
            print(f"\n   Tasa de éxito: {success_rate:.1f}%")
        print(f"{'='*60}\n")
