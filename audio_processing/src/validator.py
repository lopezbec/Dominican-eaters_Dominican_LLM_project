import os
import json
from pathlib import Path
from typing import Dict, List
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineValidator:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def validate_downloads(self, module_name: str) -> Dict:
        module_config = self.config['modules'][module_name]
        audio_dir = module_config['audio_dir']
        reports_dir = module_config['reports_dir']
        
        audio_files = list(Path(audio_dir).glob('*.m4a'))
        
        report_path = Path(reports_dir) / 'download_report.json'
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
        else:
            report = None
        
        validation = {
            'module': module_name,
            'audio_files_count': len(audio_files),
            'has_download_report': report is not None,
            'report_successful': report.get('successful', 0) if report else 0,
            'status': 'ok' if audio_files else 'no_files'
        }
        
        return validation
    
    def validate_transcriptions(self, module_name: str) -> Dict:
        module_config = self.config['modules'][module_name]
        transcriptions_dir = module_config['transcriptions_dir']
        reports_dir = module_config['reports_dir']
        
        transcription_files = list(Path(transcriptions_dir).glob('*.json'))
        
        report_path = Path(reports_dir) / 'transcription_report.json'
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
        else:
            report = None
        
        validation = {
            'module': module_name,
            'transcription_files_count': len(transcription_files),
            'has_transcription_report': report is not None,
            'report_successful': report.get('successful', 0) if report else 0,
            'status': 'ok' if transcription_files else 'no_files'
        }
        
        return validation
    
    def validate_alignments(self, module_name: str = 'lyrics-eater') -> Dict:
        if module_name != 'lyrics-eater':
            return {'module': module_name, 'status': 'not_applicable'}
        
        module_config = self.config['modules'][module_name]
        alignments_dir = module_config['alignments_dir']
        reports_dir = module_config['reports_dir']
        
        alignment_files = list(Path(alignments_dir).glob('*.json'))
        
        report_path = Path(reports_dir) / 'alignment_report.json'
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
        else:
            report = None
        
        validation = {
            'module': module_name,
            'alignment_files_count': len(alignment_files),
            'has_alignment_report': report is not None,
            'report_successful': report.get('successful', 0) if report else 0,
            'average_wer': report.get('average_wer', None) if report else None,
            'status': 'ok' if alignment_files else 'no_files'
        }
        
        return validation
    
    def validate_module(self, module_name: str) -> Dict:
        logger.info(f"Validating module: {module_name}")
        
        validation = {
            'module': module_name,
            'downloads': self.validate_downloads(module_name),
            'transcriptions': self.validate_transcriptions(module_name),
        }
        
        if module_name == 'lyrics-eater':
            validation['alignments'] = self.validate_alignments(module_name)
        
        return validation
    
    def validate_all(self) -> Dict:
        results = {}
        for module_name in self.config['modules'].keys():
            results[module_name] = self.validate_module(module_name)
        
        return results
    
    def print_validation_report(self, validation: Dict):
        print("\n" + "="*60)
        print("PIPELINE VALIDATION REPORT")
        print("="*60)
        
        for module_name, module_data in validation.items():
            print(f"\n[{module_name.upper()}]")
            
            if 'downloads' in module_data:
                dl = module_data['downloads']
                print(f"  Downloads: {dl['audio_files_count']} files | Status: {dl['status']}")
            
            if 'transcriptions' in module_data:
                tr = module_data['transcriptions']
                print(f"  Transcriptions: {tr['transcription_files_count']} files | Status: {tr['status']}")
            
            if 'alignments' in module_data:
                al = module_data['alignments']
                wer_str = f"WER: {al['average_wer']:.3f}" if al['average_wer'] is not None else "WER: N/A"
                print(f"  Alignments: {al['alignment_files_count']} files | {wer_str} | Status: {al['status']}")
        
        print("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate audio processing pipeline')
    parser.add_argument(
        '--module',
        type=str,
        choices=['books-eater', 'poems-eater', 'lyrics-eater', 'all'],
        default='all',
        help='Module to validate'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    validator = PipelineValidator(args.config)
    
    if args.module == 'all':
        validation = validator.validate_all()
    else:
        validation = {args.module: validator.validate_module(args.module)}
    
    validator.print_validation_report(validation)
    
    output_path = 'validation_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Validation report saved to: {output_path}")


if __name__ == '__main__':
    main()
