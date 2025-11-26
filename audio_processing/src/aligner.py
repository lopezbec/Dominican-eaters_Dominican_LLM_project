import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm
import logging
import subprocess

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForcedAligner:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.alignment_config = self.config['alignment']
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        
        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return wer
    
    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_best_alignment_start(
        self,
        reference: str,
        hypothesis: str,
        window_size: int = 50,
        min_match_length: int = 30
    ) -> Tuple[int, float]:
        ref_normalized = self.normalize_text(reference)
        hyp_normalized = self.normalize_text(hypothesis)
        
        ref_words = ref_normalized.split()
        hyp_words = hyp_normalized.split()
        
        if len(ref_words) < min_match_length:
            min_match_length = max(10, len(ref_words) // 2)
        
        search_window = min(window_size, len(ref_words))
        ref_prefix_words = ref_words[:search_window]
        
        best_start = 0
        best_similarity = 0.0
        
        max_offset = len(hyp_words)
        
        for offset in range(max_offset):
            remaining_words = len(hyp_words) - offset
            if remaining_words < search_window:
                break
            
            hyp_window_words = hyp_words[offset:offset + search_window]
            
            matches = 0
            for ref_word in ref_prefix_words:
                if ref_word in hyp_window_words:
                    matches += 1
            
            similarity = matches / len(ref_prefix_words) if ref_prefix_words else 0.0
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_start = offset
                
                if similarity >= 0.8:
                    break
        
        logger.info(f"Best alignment start: offset={best_start}, similarity={best_similarity:.3f}")
        
        return best_start, best_similarity
    
    def _calculate_sequence_similarity(self, text1: str, text2: str) -> float:
        words1 = text1.split()
        words2 = text2.split()
        
        if not words1 or not words2:
            return 0.0
        
        matches = 0
        for w1 in words1:
            if w1 in words2:
                matches += 1
        
        return matches / len(words1)
    
    def align_with_aeneas(
        self,
        audio_path: str,
        text_path: str,
        output_path: str
    ) -> bool:
        try:
            cmd = [
                'python3',
                '-m',
                'aeneas.tools.execute_task',
                audio_path,
                text_path,
                f"task_language={self.alignment_config['aeneas_task_language']}|"
                f"is_text_type={self.alignment_config['aeneas_is_text_type']}|"
                f"os_task_file_format=json",
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Aeneas alignment failed: {e}")
            return False
    
    def extract_reference_texts_lyrics(self) -> Dict:
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
            
            from .utils.text import clean_reference_text
            cleaned_lyrics = clean_reference_text(lyrics)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_lyrics)
            
            extracted += 1
        
        logger.info(f"Extracted {extracted} reference texts for lyrics-eater")
        return {'extracted': extracted}
    
    def extract_reference_texts_poems(self) -> Dict:
        module_config = self.config['modules']['poems-eater']
        reference_texts_dir = module_config['reference_texts_dir']
        
        os.makedirs(reference_texts_dir, exist_ok=True)
        
        logger.warning("Poems reference texts extraction requires manual implementation")
        logger.info("Reference: poems-eater/src/utils/dominican_poems.py")
        
        return {'extracted': 0, 'note': 'Manual extraction required'}
    
    def align_module(self, module_name: str = 'lyrics-eater') -> Dict:
        logger.info(f"Starting alignment for module: {module_name}")
        
        if module_name != 'lyrics-eater':
            logger.warning(f"Alignment only implemented for lyrics-eater, not {module_name}")
            return {'error': 'Module not supported for alignment'}
        
        module_config = self.config['modules'][module_name]
        transcriptions_dir = module_config['transcriptions_dir']
        reference_texts_dir = module_config['reference_texts_dir']
        alignments_dir = module_config['alignments_dir']
        audio_dir = module_config['audio_dir']
        
        os.makedirs(alignments_dir, exist_ok=True)
        
        transcription_files = list(Path(transcriptions_dir).glob('*.json'))
        
        if not transcription_files:
            logger.warning(f"No transcription files found in {transcriptions_dir}")
            return {'error': 'No transcription files found'}
        
        results = []
        successful = 0
        failed = 0
        total_wer = 0.0
        
        for trans_file in tqdm(transcription_files, desc=f"Aligning {module_name}"):
            result = {
                'file': trans_file.name,
                'success': False,
                'error': None,
                'wer': None,
                'has_reference': False,
                'alignment_start_offset': 0,
                'alignment_similarity': 0.0
            }
            
            try:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
                
                if not transcription_data.get('success'):
                    result['error'] = 'Transcription failed'
                    failed += 1
                    results.append(result)
                    continue
                
                hypothesis = transcription_data.get('transcription', '')
                
                match = re.match(r'lyrics-eater_(\d{3})_', trans_file.name)
                if not match:
                    result['error'] = 'Could not extract index from filename'
                    failed += 1
                    results.append(result)
                    continue
                
                index = match.group(1)
                reference_files = list(Path(reference_texts_dir).glob(f"{index}_*.txt"))
                
                if not reference_files:
                    result['error'] = 'No reference text found'
                    result['has_reference'] = False
                    failed += 1
                    results.append(result)
                    continue
                
                reference_file = reference_files[0]
                
                with open(reference_file, 'r', encoding='utf-8') as f:
                    reference = f.read()
                
                result['has_reference'] = True
                
                ref_normalized = self.normalize_text(reference)
                hyp_normalized = self.normalize_text(hypothesis)
                
                alignment_start, alignment_similarity = self.find_best_alignment_start(
                    reference, hypothesis
                )
                
                if alignment_start > 0:
                    hyp_words = hyp_normalized.split()
                    hyp_normalized = ' '.join(hyp_words[alignment_start:])
                    logger.info(f"Trimmed hypothesis from word {alignment_start} (similarity: {alignment_similarity:.3f})")
                
                wer = self.calculate_wer(ref_normalized, hyp_normalized)
                result['wer'] = wer
                result['alignment_start_offset'] = alignment_start
                result['alignment_similarity'] = alignment_similarity
                total_wer += wer
                
                audio_file = Path(audio_dir) / (trans_file.stem + '.m4a')
                if audio_file.exists():
                    alignment_output = Path(alignments_dir) / (trans_file.stem + '_alignment.json')
                    
                    aeneas_success = self.align_with_aeneas(
                        str(audio_file),
                        str(reference_file),
                        str(alignment_output)
                    )
                    
                    if aeneas_success:
                        result['alignment_file'] = str(alignment_output)
                
                result['success'] = True
                successful += 1
                logger.info(f"Aligned: {trans_file.name} - WER: {wer:.3f}")
                
            except Exception as e:
                result['error'] = str(e)
                failed += 1
                logger.error(f"Failed: {trans_file.name} - {e}")
            
            results.append(result)
        
        avg_wer = total_wer / successful if successful > 0 else 0.0
        
        report = {
            'module': module_name,
            'total_files': len(transcription_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(transcription_files) if transcription_files else 0,
            'average_wer': avg_wer,
            'results': results
        }
        
        report_path = os.path.join(module_config['reports_dir'], 'alignment_report.json')
        os.makedirs(module_config['reports_dir'], exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Alignment report saved to: {report_path}")
        logger.info(f"Summary: {successful}/{len(transcription_files)} successful, Average WER: {avg_wer:.3f}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Forced alignment with reference texts')
    parser.add_argument(
        '--action',
        type=str,
        choices=['extract-lyrics', 'extract-poems', 'align'],
        default='align',
        help='Action to perform'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    aligner = ForcedAligner(args.config)
    
    if args.action == 'extract-lyrics':
        aligner.extract_reference_texts_lyrics()
    elif args.action == 'extract-poems':
        aligner.extract_reference_texts_poems()
    elif args.action == 'align':
        aligner.align_module('lyrics-eater')


if __name__ == '__main__':
    main()
