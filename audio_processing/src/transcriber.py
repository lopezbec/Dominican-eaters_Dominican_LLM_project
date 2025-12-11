import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm
import logging

try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.whisper_config = self.config['whisper']
        self.model = None
        self.device = 'cpu'
        
    def load_model(self):
        if whisper is None:
            raise ImportError("openai-whisper is required. Install with: pip install openai-whisper")
        
        if self.model is None:
            model_name = self.whisper_config['model']
            logger.info(f"Loading Whisper model: {model_name}")
            
            if torch and torch.cuda.is_available():
                try:
                    self.model = whisper.load_model(model_name, device='cuda')
                    self.device = 'cuda'
                    logger.info(f"Model loaded on CUDA (GPU)")
                except Exception as e:
                    logger.warning(f"Failed to load on CUDA, using CPU: {e}")
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.model = whisper.load_model(model_name, device='cpu')
                    self.device = 'cpu'
                    logger.info("Model loaded on CPU")
            else:
                self.model = whisper.load_model(model_name, device='cpu')
                self.device = 'cpu'
                logger.info("Model loaded on CPU")
    
    def transcribe_audio(self, audio_path: str, skip_existing: bool = True, partial: bool = False) -> Dict:
        result = {
            'audio_path': audio_path,
            'success': False,
            'error': None,
            'transcription': None,
            'duration': None,
            'language': None,
            'segments': [],
            'words': [],
            'skipped': False,
            'partial': partial
        }
        
        try:
            start_time = time.time()
            
            fp16 = self.whisper_config.get('fp16', True) and self.device == 'cuda'
            
            transcribe_kwargs = {
                'language': self.whisper_config['language'],
                'task': self.whisper_config['task'],
                'word_timestamps': self.whisper_config['word_timestamps'],
                'fp16': fp16,
                'beam_size': self.whisper_config.get('beam_size', 5),
                'best_of': self.whisper_config.get('best_of', 5),
                'temperature': self.whisper_config.get('temperature', 0.0),
                'verbose': False
            }
            
            if partial:
                partial_duration = self.whisper_config.get('partial_duration', 60)
                transcribe_kwargs['duration'] = partial_duration
                logger.info(f"Partial transcription: first {partial_duration} seconds")
            
            transcription = self.model.transcribe(audio_path, **transcribe_kwargs)
            
            processing_time = time.time() - start_time
            
            result['success'] = True
            result['transcription'] = transcription['text']
            result['language'] = transcription.get('language', 'es')
            result['duration'] = transcription.get('duration')
            result['processing_time'] = processing_time
            
            segments_data = []
            all_words = []
            
            for segment in transcription.get('segments', []):
                segment_data = {
                    'id': segment['id'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'confidence': segment.get('avg_logprob', 0.0)
                }
                
                if 'words' in segment:
                    words = []
                    for word in segment['words']:
                        word_data = {
                            'word': word['word'],
                            'start': word['start'],
                            'end': word['end'],
                            'confidence': word.get('probability', 0.0)
                        }
                        words.append(word_data)
                        all_words.append(word_data)
                    segment_data['words'] = words
                
                segments_data.append(segment_data)
            
            result['segments'] = segments_data
            result['words'] = all_words
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Transcription failed for {audio_path}: {e}")
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
    
    def transcribe_module(self, module_name: str, skip_existing: bool = True, partial: bool = False) -> Dict:
        logger.info(f"Starting transcription for module: {module_name}")
        if partial:
            logger.info("Partial mode: transcribing only first portion of audio for alignment verification")
        
        self.load_model()
        
        module_config = self.config['modules'][module_name]
        audio_dir = module_config['audio_dir']
        transcriptions_dir = module_config['transcriptions_dir']
        
        os.makedirs(transcriptions_dir, exist_ok=True)
        
        audio_files = list(Path(audio_dir).glob('*.m4a'))
        
        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return {'error': 'No audio files found'}
        
        results = []
        successful = 0
        failed = 0
        skipped = 0
        
        for audio_file in tqdm(audio_files, desc=f"Transcribing {module_name}"):
            output_filename = audio_file.stem + '.json'
            output_path = os.path.join(transcriptions_dir, output_filename)
            
            if skip_existing and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 100:
                    logger.info(f"Skipping existing transcription: {output_filename}")
                    skipped += 1
                    successful += 1
                    results.append({
                        'file': audio_file.name,
                        'success': True,
                        'skipped': True,
                        'error': None,
                        'duration': None,
                        'processing_time': 0
                    })
                    continue
                else:
                    logger.warning(f"Transcription too small ({file_size} bytes), re-transcribing: {output_filename}")
            
            result = self.transcribe_audio(str(audio_file), skip_existing, partial=partial)
            
            if result['success']:
                successful += 1
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Transcribed: {audio_file.name}")
            else:
                failed += 1
                logger.error(f"Failed: {audio_file.name} - {result['error']}")
            
            results.append({
                'file': audio_file.name,
                'success': result['success'],
                'skipped': result.get('skipped', False),
                'error': result.get('error'),
                'duration': result.get('duration'),
                'processing_time': result.get('processing_time')
            })
        
        report = {
            'module': module_name,
            'total_files': len(audio_files),
            'successful': successful,
            'skipped': skipped,
            'failed': failed,
            'success_rate': successful / len(audio_files) if audio_files else 0,
            'model': self.whisper_config['model'],
            'device': self.device,
            'results': results
        }
        
        report_path = os.path.join(module_config['reports_dir'], 'transcription_report.json')
        os.makedirs(module_config['reports_dir'], exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Transcription report saved to: {report_path}")
        logger.info(f"Summary: {successful}/{len(audio_files)} successful ({skipped} skipped), {failed} failed")
        
        return report
    
    def transcribe_all_modules(self, skip_existing: bool = True, partial: bool = False) -> Dict[str, Dict]:
        results = {}
        for module_name in self.config['modules'].keys():
            try:
                results[module_name] = self.transcribe_module(module_name, skip_existing, partial=partial)
                
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared CUDA cache after module")
                    
            except Exception as e:
                logger.error(f"Error processing module {module_name}: {e}")
                results[module_name] = {'error': str(e)}
                
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe audio files with Whisper')
    parser.add_argument(
        '--module',
        type=str,
        choices=['books-eater', 'poems-eater', 'lyrics-eater', 'all'],
        default='all',
        help='Module to transcribe'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-transcription even if files exist'
    )
    parser.add_argument(
        '--partial',
        action='store_true',
        help='Transcribe only first portion of audio for alignment verification'
    )
    
    args = parser.parse_args()
    
    transcriber = AudioTranscriber(args.config)
    
    skip_existing = not args.force
    
    if args.module == 'all':
        transcriber.transcribe_all_modules(skip_existing, partial=args.partial)
    else:
        transcriber.transcribe_module(args.module, skip_existing, partial=args.partial)


if __name__ == '__main__':
    main()
