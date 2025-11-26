import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioDownloader:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.audio_config = self.config['audio']
        self.download_config = self.config['download']
        
    def sanitize_filename(self, filename: str) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', '_', filename)
        filename = filename[:100]
        return filename
    
    def extract_video_info(self, url: str) -> Optional[Dict]:
        try:
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-warnings',
                '--skip-download',
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
            return None
        except Exception as e:
            logger.error(f"Error extracting info from {url}: {e}")
            return None
    
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
            sanitized_title = self.sanitize_filename(title)
            
            filename = f"{submodule}_{index:03d}_{sanitized_title}.{self.audio_config['format']}"
            full_path = os.path.join(output_path, filename)
            
            if not force and os.path.exists(full_path):
                file_size = os.path.getsize(full_path)
                if file_size > 1024:
                    logger.info(f"Skipping existing file: {filename} ({file_size} bytes)")
                    result['success'] = True
                    result['skipped'] = True
                    result['file_path'] = full_path
                    result['title'] = title
                    result['duration'] = duration
                    return result
                else:
                    logger.warning(f"File too small ({file_size} bytes), re-downloading: {filename}")
            
            cmd = [
                'yt-dlp',
                '-f', 'bestaudio',
                '-x',
                '--audio-format', self.audio_config['format'],
                '--audio-quality', self.audio_config['quality'],
                '--postprocessor-args', f"ffmpeg:-ar {self.audio_config['sample_rate']} -ac {self.audio_config['channels']}",
                '-o', full_path,
                '--no-warnings',
                '--quiet',
                url
            ]
            
            subprocess.run(cmd, check=True, timeout=self.download_config['timeout'])
            
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
    
    def load_urls_from_csv(self, csv_path: str, url_column: str) -> List[str]:
        if pd is None:
            raise ImportError("pandas is required. Install with: pip install pandas")
        
        df = pd.read_csv(csv_path)
        urls = []
        for idx, row in df.iterrows():
            url = row.get(url_column, '')
            if isinstance(url, str) and url.startswith('http') and 'youtube.com' in url:
                urls.append(url)
        return urls
    
    def load_urls_from_excel(self, excel_path: str, url_column: str) -> List[str]:
        if pd is None:
            raise ImportError("pandas and openpyxl are required. Install with: pip install pandas openpyxl")
        
        df = pd.read_excel(excel_path)
        urls = []
        for idx, row in df.iterrows():
            url = row.get(url_column, '')
            if isinstance(url, str) and url.startswith('http') and 'youtube.com' in url:
                urls.append(url)
        return urls
    
    def download_module(self, module_name: str, force: bool = False) -> Dict:
        logger.info(f"Starting download for module: {module_name}")
        
        module_config = self.config['modules'][module_name]
        
        if 'csv_path' in module_config:
            urls = self.load_urls_from_csv(
                module_config['csv_path'],
                module_config['url_column']
            )
        elif 'excel_path' in module_config:
            urls = self.load_urls_from_excel(
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
        
        for idx, url in enumerate(tqdm(urls, desc=f"Downloading {module_name}")):
            result = self.download_audio(url, audio_dir, module_name, idx, force)
            results.append(result)
            
            if result['success']:
                successful += 1
                logger.info(f"Downloaded: {result['title']}")
            else:
                failed += 1
                logger.error(f"Failed: {url} - {result['error']}")
        
        report = {
            'module': module_name,
            'total_urls': len(urls),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(urls) if urls else 0,
            'results': results
        }
        
        report_path = os.path.join(module_config['reports_dir'], 'download_report.json')
        os.makedirs(module_config['reports_dir'], exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Download report saved to: {report_path}")
        logger.info(f"Summary: {successful}/{len(urls)} successful, {failed} failed")
        
        return report
    
    def download_all_modules(self, force: bool = False) -> Dict[str, Dict]:
        results = {}
        for module_name in self.config['modules'].keys():
            try:
                results[module_name] = self.download_module(module_name, force)
            except Exception as e:
                logger.error(f"Error processing module {module_name}: {e}")
                results[module_name] = {'error': str(e)}
        return results


def main():
    import argparse
    
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
    
    downloader = AudioDownloader(args.config)
    
    if args.module == 'all':
        downloader.download_all_modules(args.force)
    else:
        downloader.download_module(args.module, args.force)


if __name__ == '__main__':
    main()
