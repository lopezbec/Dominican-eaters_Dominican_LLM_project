import os
import time
from pathlib import Path
from typing import Dict
import logging

from .utils.file_validator import TranscriptionFileValidator
from .utils.progress_reporter import ProgressReporter
from .utils.report_exporter import ReportExporter
from .models.whisper_model import WhisperModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    
    def __init__(
        self,
        config: Dict,
        model_manager: WhisperModelManager,
        file_validator: TranscriptionFileValidator,
        progress_reporter: ProgressReporter,
        report_exporter: ReportExporter
    ):
        self.config = config
        self.whisper_config = config['whisper']
        self.model_manager = model_manager
        self.file_validator = file_validator
        self.progress_reporter = progress_reporter
        self.report_exporter = report_exporter
        
    def load_model(self):
        self.model_manager.load_model()
    
    def transcribe_audio(
        self, 
        audio_path: str, 
        skip_existing: bool = True, 
        partial: bool = False
    ) -> Dict:
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
            
            transcribe_kwargs = {
                'language': self.whisper_config['language'],
                'task': self.whisper_config['task'],
                'word_timestamps': self.whisper_config['word_timestamps'],
                'beam_size': self.whisper_config.get('beam_size', 5),
                'best_of': self.whisper_config.get('best_of', 5),
                'temperature': self.whisper_config.get('temperature', 0.0),
            }
            
            if partial:
                partial_duration = self.whisper_config.get('partial_duration', 60)
                transcribe_kwargs['duration'] = partial_duration
                self.progress_reporter.report_success(
                    f"Partial transcription: first {partial_duration} seconds"
                )
            
            transcription = self.model_manager.transcribe(audio_path, **transcribe_kwargs)
            
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
            
            self.model_manager.cleanup()
            
        except Exception as e:
            result['error'] = str(e)
            self.progress_reporter.report_error(f"Transcription failed for {audio_path}: {e}")
            self.model_manager.cleanup()
        
        return result
    
    def transcribe_module(
        self, 
        module_name: str, 
        skip_existing: bool = True, 
        partial: bool = False
    ) -> Dict:
        self.progress_reporter.report_success(f"Starting transcription for module: {module_name}")
        if partial:
            self.progress_reporter.report_success(
                "Partial mode: transcribing only first portion of audio for alignment verification"
            )
        
        self.load_model()
        
        module_config = self.config['modules'][module_name]
        audio_dir = module_config['audio_dir']
        transcriptions_dir = module_config['transcriptions_dir']
        
        os.makedirs(transcriptions_dir, exist_ok=True)
        
        audio_files = list(Path(audio_dir).glob('*.m4a'))
        
        if not audio_files:
            self.progress_reporter.report_warning(f"No audio files found in {audio_dir}")
            return {'error': 'No audio files found'}
        
        results = []
        successful = 0
        failed = 0
        skipped = 0
        
        for audio_file in self.progress_reporter.report_batch(
            audio_files, f"Transcribing {module_name}"
        ):
            output_filename = audio_file.stem + '.json'
            output_path = os.path.join(transcriptions_dir, output_filename)
            
            if skip_existing:
                is_valid, validation_msg = self.file_validator.validate_exists(output_path)
                if is_valid:
                    self.progress_reporter.report_success(
                        f"Skipping existing transcription: {output_filename}"
                    )
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
                elif os.path.exists(output_path):
                    self.progress_reporter.report_warning(
                        f"Re-transcribing {output_filename}: {validation_msg}"
                    )
            
            result = self.transcribe_audio(str(audio_file), skip_existing, partial=partial)
            
            if result['success']:
                successful += 1
                
                import json
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                self.progress_reporter.report_success(f"Transcribed: {audio_file.name}")
            else:
                failed += 1
                self.progress_reporter.report_error(
                    f"Failed: {audio_file.name} - {result['error']}"
                )
            
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
            'device': self.model_manager.get_device(),
            'results': results
        }
        
        report_path = os.path.join(module_config['reports_dir'], 'transcription_report.json')
        self.report_exporter.export_json(report, report_path)
        
        self.progress_reporter.report_success(
            f"Summary: {successful}/{len(audio_files)} successful ({skipped} skipped), "
            f"{failed} failed"
        )
        
        return report
    
    def transcribe_all_modules(
        self, 
        skip_existing: bool = True, 
        partial: bool = False
    ) -> Dict[str, Dict]:
        results = {}
        for module_name in self.config['modules'].keys():
            try:
                results[module_name] = self.transcribe_module(
                    module_name, skip_existing, partial=partial
                )
                self.model_manager.cleanup()
                self.progress_reporter.report_success("Cleared CUDA cache after module")
            except Exception as e:
                self.progress_reporter.report_error(
                    f"Error processing module {module_name}: {e}"
                )
                results[module_name] = {'error': str(e)}
                self.model_manager.cleanup()
        return results


def main():
    import argparse
    from .utils.config_loader import ConfigLoader
    
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
    
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)
    
    whisper_config = config['whisper']
    model_manager = WhisperModelManager(
        model_name=whisper_config['model'],
        fp16=whisper_config.get('fp16', True)
    )
    file_validator = TranscriptionFileValidator()
    progress_reporter = ProgressReporter(logger)
    report_exporter = ReportExporter()
    
    transcriber = AudioTranscriber(
        config=config,
        model_manager=model_manager,
        file_validator=file_validator,
        progress_reporter=progress_reporter,
        report_exporter=report_exporter
    )
    
    skip_existing = not args.force
    
    if args.module == 'all':
        transcriber.transcribe_all_modules(skip_existing, partial=args.partial)
    else:
        transcriber.transcribe_module(args.module, skip_existing, partial=args.partial)


if __name__ == '__main__':
    main()
