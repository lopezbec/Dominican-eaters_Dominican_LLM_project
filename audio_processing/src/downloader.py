import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.interfaces.video_downloader import IVideoDownloader
from shared.utils.file_utils import get_safe_filename
from .utils.file_validator import AudioFileValidator
from .utils.data_loader import DataLoader
from .utils.progress_reporter import ProgressReporter
from .utils.report_exporter import ReportExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioDownloader:
    
    def __init__(
        self,
        config: Dict,
        video_downloader: IVideoDownloader,
        file_validator: AudioFileValidator,
        data_loader: DataLoader,
        progress_reporter: ProgressReporter,
        report_exporter: ReportExporter
    ):
        self.config = config
        self.audio_config = config['audio']
        self.download_config = config['download']
        self.video_downloader = video_downloader
        self.file_validator = file_validator
        self.data_loader = data_loader
        self.progress_reporter = progress_reporter
        self.report_exporter = report_exporter
    
    def extract_video_info(self, url: str) -> Optional[Dict]:
        return self.video_downloader.extract_video_info(url)
    
    def download_audio(
        self, 
        url: str, 
        output_path: str, 
        submodule: str,
        index: int,
        force: bool = False
    ) -> Dict:
        result = {
            'url': url,
            'index': index,
            'success': False,
            'error': None,
            'file_path': None,
            'duration': None,
            'title': None,
            'skipped': False
        }
        
        try:
            info = self.extract_video_info(url)
            if not info:
                result['error'] = 'Failed to extract video info'
                return result
            
            title = info.get('title', f'unknown_{index}')
            duration = info.get('duration', None)
            sanitized_title = get_safe_filename(title, max_length=100)
            
            filename = f"{submodule}_{index:03d}_{sanitized_title}.{self.audio_config['format']}"
            full_path = os.path.join(output_path, filename)
            
            if not force:
                is_valid, validation_msg = self.file_validator.validate_exists(full_path)
                if is_valid:
                    self.progress_reporter.report_success(f"Skipping existing file: {filename}")
                    result['success'] = True
                    result['skipped'] = True
                    result['file_path'] = full_path
                    result['title'] = title
                    result['duration'] = duration
                    return result
                elif os.path.exists(full_path):
                    self.progress_reporter.report_warning(
                        f"Re-downloading {filename}: {validation_msg}"
                    )
            
            self.video_downloader.download(
                url=url,
                output_path=full_path,
                audio_format=self.audio_config['format'],
                audio_quality=self.audio_config['quality'],
                sample_rate=self.audio_config['sample_rate'],
                channels=self.audio_config['channels'],
                timeout=self.download_config['timeout']
            )
            
            result['success'] = True
            result['file_path'] = full_path
            result['title'] = title
            result['duration'] = duration
            
        except subprocess.TimeoutExpired:
            result['error'] = 'Download timeout'
        except subprocess.CalledProcessError as e:
            result['error'] = f'Download failed: {e}'
        except Exception as e:
            result['error'] = f'Unexpected error: {e}'
        
        return result
    
    def _extract_reference_texts(self, module_name: str) -> None:
        """
        Extract reference texts after downloads complete.
        
        For lyrics-eater: Extract lyrics from Excel
        For poems-eater: Extract poems from data source
        For books-eater: No extraction needed (long-form content)
        """
        try:
            if module_name == 'lyrics-eater':
                self.progress_reporter.report_success("Extracting reference texts from lyrics...")
                from .alignment.reference_extractor import ReferenceTextExtractor
                extractor = ReferenceTextExtractor(self.config)
                result = extractor.extract_text_from_excel(module_name)
                self.progress_reporter.report_success(
                    f"Extracted {result.get('extracted', 0)} reference texts for {module_name}"
                )
            elif module_name == 'poems-eater':
                self.progress_reporter.report_success("Extracting reference texts from poems...")
                from .alignment.reference_extractor import ReferenceTextExtractor
                extractor = ReferenceTextExtractor(self.config)
                result = extractor.extract_text_from_excel(module_name)
                if result.get('note'):
                    self.progress_reporter.report_warning(result['note'])
            elif module_name == 'books-eater':
                # Books don't need reference extraction (full audiobooks)
                logger.info("Skipping reference extraction for books-eater (full audiobooks)")
        except Exception as e:
            self.progress_reporter.report_warning(
                f"Failed to extract reference texts for {module_name}: {e}"
            )
            logger.exception(f"Reference extraction error for {module_name}")
    
    def download_module(self, module_name: str, force: bool = False) -> Dict:
        self.progress_reporter.report_success(f"Starting download for module: {module_name}")
        
        module_config = self.config['modules'][module_name]
        
        if 'csv_path' in module_config:
            urls = self.data_loader.load_urls_from_csv(
                module_config['csv_path'],
                module_config['url_column']
            )
        elif 'excel_path' in module_config:
            urls = self.data_loader.load_urls_from_excel(
                module_config['excel_path'],
                module_config['url_column']
            )
        else:
            raise ValueError(f"No data source configured for {module_name}")
        
        audio_dir = module_config['audio_dir']
        os.makedirs(audio_dir, exist_ok=True)
        
        results = []
        successful = 0
        failed = 0
        
        for idx, url in enumerate(
            self.progress_reporter.report_batch(urls, f"Downloading {module_name}")
        ):
            result = self.download_audio(url, audio_dir, module_name, idx, force)
            results.append(result)
            
            if result['success']:
                successful += 1
                self.progress_reporter.report_success(f"Downloaded: {result['title']}")
            else:
                failed += 1
                self.progress_reporter.report_error(f"Failed: {url} - {result['error']}")
        
        report = {
            'module': module_name,
            'total_urls': len(urls),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(urls) if urls else 0,
            'results': results
        }
        
        report_path = os.path.join(module_config['reports_dir'], 'download_report.json')
        self.report_exporter.export_json(report, report_path)
        
        self.progress_reporter.report_success(
            f"Summary: {successful}/{len(urls)} successful, {failed} failed"
        )
        
        # Extract reference texts after successful downloads
        self._extract_reference_texts(module_name)
        
        return report
    
    def download_all_modules(self, force: bool = False) -> Dict[str, Dict]:
        results = {}
        for module_name in self.config['modules'].keys():
            try:
                results[module_name] = self.download_module(module_name, force)
            except Exception as e:
                self.progress_reporter.report_error(f"Error processing module {module_name}: {e}")
                results[module_name] = {'error': str(e)}
        return results


def main():
    import argparse
    from .utils.config_loader import ConfigLoader
    from .downloaders.ytdlp_downloader import YtDlpDownloader
    
    parser = argparse.ArgumentParser(description='Download audio from YouTube URLs')
    parser.add_argument(
        '--module',
        type=str,
        choices=['books-eater', 'poems-eater', 'lyrics-eater', 'all'],
        default='all',
        help='Module to download'
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
        help='Force re-download even if files exist'
    )
    
    args = parser.parse_args()
    
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)
    
    video_downloader = YtDlpDownloader()
    file_validator = AudioFileValidator()
    data_loader = DataLoader()
    progress_reporter = ProgressReporter(logger)
    report_exporter = ReportExporter()
    
    downloader = AudioDownloader(
        config=config,
        video_downloader=video_downloader,
        file_validator=file_validator,
        data_loader=data_loader,
        progress_reporter=progress_reporter,
        report_exporter=report_exporter
    )
    
    if args.module == 'all':
        downloader.download_all_modules(args.force)
    else:
        downloader.download_module(args.module, args.force)


if __name__ == '__main__':
    main()
