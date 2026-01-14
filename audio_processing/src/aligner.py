import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm
import logging
import subprocess

from .metrics.text_similarity import TextSimilarityMetrics
from .utils.text_normalizer import TextNormalizer
from .alignment.reference_extractor import ReferenceTextExtractor
from .config.constants import (
    ALIGNMENT_WINDOW_SIZE_WORDS,
    ALIGNMENT_MIN_MATCH_LENGTH,
    ALIGNMENT_MAX_SEARCH_WORDS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForcedAligner:
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        text_normalizer: Optional[TextNormalizer] = None,
        similarity_metrics: Optional[TextSimilarityMetrics] = None,
        reference_extractor: Optional[ReferenceTextExtractor] = None
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.alignment_config = self.config['alignment']
        
        self.text_normalizer = text_normalizer or TextNormalizer()
        self.similarity_metrics = similarity_metrics or TextSimilarityMetrics(
            self.text_normalizer
        )
        self.reference_extractor = reference_extractor or ReferenceTextExtractor(
            self.config
        )
    
    def find_best_alignment_start(
        self,
        reference: str,
        hypothesis: str,
        window_size: int = ALIGNMENT_WINDOW_SIZE_WORDS,
        min_match_length: int = ALIGNMENT_MIN_MATCH_LENGTH,
        max_search_words: int = ALIGNMENT_MAX_SEARCH_WORDS
    ) -> Tuple[int, float]:
        ref_normalized = self.text_normalizer.normalize_text(reference)
        hyp_normalized = self.text_normalizer.normalize_text(hypothesis)
        
        ref_words = ref_normalized.split()
        hyp_words = hyp_normalized.split()
        
        if len(ref_words) < min_match_length:
            min_match_length = max(10, len(ref_words) // 2)
        
        search_window = min(window_size, len(ref_words))
        ref_prefix_words = ref_words[:search_window]
        
        best_start = 0
        best_similarity = 0.0
        
        max_offset = min(len(hyp_words), max_search_words)
        
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
                    logger.info(f"Early stop: High confidence match found at offset {offset} (similarity: {similarity:.3f})")
                    break
        
        logger.info(f"Best alignment start: offset={best_start}, similarity={best_similarity:.3f}")
        
        return best_start, best_similarity
    
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
        return self.reference_extractor.extract_text_from_excel('lyrics-eater')
    
    def extract_reference_texts_poems(self) -> Dict:
        return self.reference_extractor.extract_text_from_excel('poems-eater')
    
    def _find_reference_file(
        self,
        trans_file: Path,
        reference_texts_dir: str,
        module_name: str
    ) -> Optional[Path]:
        if module_name == 'lyrics-eater':
            match = re.match(r'lyrics-eater_(\d{3})_', trans_file.name)
            if match:
                index = match.group(1)
                reference_files = list(Path(reference_texts_dir).glob(f"{index}_*.txt"))
                if reference_files:
                    return reference_files[0]
        else:
            base_name = trans_file.stem
            txt_file = Path(reference_texts_dir) / f"{base_name}.txt"
            if txt_file.exists():
                return txt_file
        return None
    
    def _process_alignment(
        self,
        trans_file: Path,
        reference_texts_dir: Optional[str],
        alignments_dir: Optional[str],
        audio_dir: str,
        module_name: str
    ) -> Dict:
        result = {
            'file': trans_file.name,
            'success': False,
            'error': None,
            'wer': None,
            'char_similarity': None,
            'jaccard_similarity': None,
            'cosine_similarity': None,
            'has_reference': False,
            'alignment_start_offset': 0,
            'alignment_similarity': 0.0
        }
        
        with open(trans_file, 'r', encoding='utf-8') as f:
            transcription_data = json.load(f)
        
        if not transcription_data.get('success'):
            result['error'] = 'Transcription failed'
            return result
        
        hypothesis = transcription_data.get('transcription', '')
        
        if not reference_texts_dir:
            result['error'] = 'No reference texts directory configured'
            return result
        
        reference_file = self._find_reference_file(
            trans_file, reference_texts_dir, module_name
        )
        
        if not reference_file:
            reference = transcription_data.get('transcription', '')
            if not reference:
                result['error'] = 'No reference text found'
                return result
            logger.info(f"Using transcription as reference for {trans_file.name}")
        else:
            reference = self.reference_extractor.extract_text_from_txt(str(reference_file))
        
        result['has_reference'] = True
        
        ref_normalized = self.text_normalizer.normalize_text(reference)
        hyp_normalized = self.text_normalizer.normalize_text(hypothesis)
        
        alignment_start, alignment_similarity = self.find_best_alignment_start(
            reference, hypothesis
        )
        
        if alignment_start > 0:
            hyp_words = hyp_normalized.split()
            hyp_normalized = ' '.join(hyp_words[alignment_start:])
            logger.info(f"Trimmed hypothesis from word {alignment_start} (similarity: {alignment_similarity:.3f})")
        
        wer = self.similarity_metrics.calculate_wer(ref_normalized, hyp_normalized)
        char_sim = self.similarity_metrics.calculate_char_similarity(ref_normalized, hyp_normalized)
        jaccard_sim = self.similarity_metrics.calculate_jaccard_similarity(ref_normalized, hyp_normalized)
        cosine_sim = self.similarity_metrics.calculate_cosine_similarity(ref_normalized, hyp_normalized)
        
        result['wer'] = wer
        result['char_similarity'] = char_sim
        result['jaccard_similarity'] = jaccard_sim
        result['cosine_similarity'] = cosine_sim
        result['alignment_start_offset'] = alignment_start
        result['alignment_similarity'] = alignment_similarity
        
        if alignments_dir:
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
        logger.info(f"Aligned: {trans_file.name} - WER: {wer:.3f}, CharSim: {char_sim:.3f}, Jaccard: {jaccard_sim:.3f}, Cosine: {cosine_sim:.3f}")
        
        return result
    
    def align_module(self, module_name: str) -> Dict:
        logger.info(f"Starting alignment verification for module: {module_name}")
        
        module_config = self.config['modules'][module_name]
        transcriptions_dir = module_config['transcriptions_dir']
        reference_texts_dir = module_config.get('reference_texts_dir')
        alignments_dir = module_config.get('alignments_dir')
        audio_dir = module_config['audio_dir']
        
        if alignments_dir:
            os.makedirs(alignments_dir, exist_ok=True)
        
        transcription_files = list(Path(transcriptions_dir).glob('*.json'))
        
        if not transcription_files:
            logger.warning(f"No transcription files found in {transcriptions_dir}")
            return {'error': 'No transcription files found'}
        
        results = []
        successful = 0
        failed = 0
        total_wer = 0.0
        total_char_sim = 0.0
        total_jaccard = 0.0
        total_cosine = 0.0
        
        for trans_file in tqdm(transcription_files, desc=f"Aligning {module_name}"):
            try:
                result = self._process_alignment(
                    trans_file,
                    reference_texts_dir,
                    alignments_dir,
                    audio_dir,
                    module_name
                )
                
                if result['success']:
                    successful += 1
                    total_wer += result['wer']
                    total_char_sim += result['char_similarity']
                    total_jaccard += result['jaccard_similarity']
                    total_cosine += result['cosine_similarity']
                else:
                    failed += 1
                
            except Exception as e:
                result = {
                    'file': trans_file.name,
                    'success': False,
                    'error': str(e),
                    'has_reference': False
                }
                failed += 1
                logger.error(f"Failed: {trans_file.name} - {e}")
            
            results.append(result)
        
        avg_wer = total_wer / successful if successful > 0 else 0.0
        avg_char_sim = total_char_sim / successful if successful > 0 else 0.0
        avg_jaccard = total_jaccard / successful if successful > 0 else 0.0
        avg_cosine = total_cosine / successful if successful > 0 else 0.0
        
        report = {
            'module': module_name,
            'total_files': len(transcription_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(transcription_files) if transcription_files else 0,
            'average_wer': avg_wer,
            'average_char_similarity': avg_char_sim,
            'average_jaccard_similarity': avg_jaccard,
            'average_cosine_similarity': avg_cosine,
            'results': results
        }
        
        report_path = os.path.join(module_config['reports_dir'], 'alignment_report.json')
        os.makedirs(module_config['reports_dir'], exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Alignment report saved to: {report_path}")
        logger.info(f"Summary: {successful}/{len(transcription_files)} successful")
        logger.info(f"Metrics - WER: {avg_wer:.3f}, CharSim: {avg_char_sim:.3f}, Jaccard: {avg_jaccard:.3f}, Cosine: {avg_cosine:.3f}")
        
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
        '--module',
        type=str,
        choices=['books-eater', 'poems-eater', 'lyrics-eater'],
        default='lyrics-eater',
        help='Module to align'
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
        aligner.align_module(args.module)


if __name__ == '__main__':
    main()
