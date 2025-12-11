#!/usr/bin/env python3
"""
Unified CLI for Dominican Eaters LLM Project
Wraps existing audio_processing scripts with a clean interface
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent

def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}\n")
    working_dir: Path = cwd if cwd else PROJECT_ROOT
    result: subprocess.CompletedProcess = subprocess.run(cmd, cwd=working_dir)
    if result.returncode != 0:
        print(f"Failed with exit code {result.returncode}")
        return False
    return True

def scrape(module_type: str) -> None:
    print(f"Scraping {module_type} data...")
    
    modules: dict[str, str] = {
        'books': 'books-eater',
        'lyrics': 'lyrics-eater', 
        'poems': 'poems-eater'
    }
    
    if module_type == 'all':
        for name, path in modules.items():
            print(f"\nScraping {name}...")
            run_command(['python', 'main.py'], f"Scraping {name}", cwd=PROJECT_ROOT / path)
    else:
        module_path: Path = PROJECT_ROOT / modules[module_type]
        run_command(['python', 'main.py'], f"Scraping {module_type}", cwd=module_path)

def download(module_type: str, force: bool = False) -> None:
    cmd: List[str] = ['python', 'audio_processing/src/downloader.py', '--module', module_type, '--config', 'audio_processing/config.yaml']
    if force:
        cmd.append('--force')
    
    run_command(cmd, f"Downloading audio for {module_type}")

def transcribe(module_type: str, model: str = 'base', force: bool = False, partial: bool = False) -> None:
    cmd: List[str] = ['python', 'audio_processing/src/transcriber.py', '--module', module_type, '--config', 'audio_processing/config.yaml']
    if force:
        cmd.append('--force')
    if partial:
        cmd.append('--partial')
    
    run_command(cmd, f"Transcribing audio for {module_type}")

def align(module_type: str = 'lyrics-eater') -> None:
    cmd: List[str] = ['python', 'audio_processing/src/aligner.py', '--action', 'align', '--module', module_type, '--config', 'audio_processing/config.yaml']
    run_command(cmd, f"Aligning text for {module_type}")

def validate(module_type: str) -> None:
    cmd: List[str] = ['python', 'audio_processing/src/validator.py', '--module', module_type, '--config', 'audio_processing/config.yaml']
    run_command(cmd, f"Validating {module_type}")

def pipeline(module_type: str, skip_scrape: bool = False, skip_download: bool = False, 
             skip_transcribe: bool = False, force: bool = False, model: str = 'base', 
             partial: bool = False) -> None:
    print(f"\nRUNNING FULL PIPELINE FOR {module_type.upper()}")
    
    modules: List[str] = ['books-eater', 'lyrics-eater', 'poems-eater'] if module_type == 'all' else [f'{module_type}-eater']
    
    for module in modules:
        print(f"\nProcessing {module}")
        
        module_short: str = module.replace('-eater', '')
        
        if not skip_scrape:
            print(f"\nStep 1: Scraping {module_short}...")
            scrape(module_short)
        
        if not skip_download:
            print(f"\nStep 2: Downloading audio for {module}...")
            download(module, force=force)
        
        if not skip_transcribe:
            print(f"\nStep 3: Transcribing audio for {module}...")
            use_partial: bool = partial or (module == 'lyrics-eater')
            transcribe(module, model=model, force=force, partial=use_partial)
        
        if module == 'lyrics-eater':
            print(f"\nStep 4: Aligning text for {module}...")
            align(module)
        
        print(f"\nStep 5: Validating {module}...")
        validate(module)
    
    print(f"\nPIPELINE COMPLETE!")

def setup() -> None:
    print("Setting up Dominican Eaters environment...")
    
    dirs: List[str] = [
        'books-eater/audio',
        'books-eater/transcriptions',
        'books-eater/reports',
        'lyrics-eater/audio',
        'lyrics-eater/transcriptions',
        'lyrics-eater/alignments',
        'lyrics-eater/reports',
        'lyrics-eater/reference_texts',
        'poems-eater/audio',
        'poems-eater/transcriptions',
        'poems-eater/reports',
        'poems-eater/reference_texts',
        'logs'
    ]
    
    for dir_path in dirs:
        full_path: Path = PROJECT_ROOT / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("  1. Activate your virtual environment")
    print("  2. Install dependencies:")
    print("     pip install -r requirements.txt")
    print("  3. Run: python dominican-eater.py pipeline --type all")

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dominican Eaters LLM Project - Unified Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Setup environment:
    python dominican-eater.py setup
  
  Scrape all content:
    python dominican-eater.py scrape --type all
  
  Download audio (with smart skip):
    python dominican-eater.py download --type lyrics
  
  Transcribe with Whisper:
    python dominican-eater.py transcribe --type books --model large
    python dominican-eater.py transcribe --type books --partial
  
  Align lyrics with audio:
    python dominican-eater.py align --type lyrics-eater
    python dominican-eater.py align --type books-eater
  
  Run complete pipeline:
    python dominican-eater.py pipeline --type all
    python dominican-eater.py pipeline --type lyrics --model large --partial
  
  Validate data:
    python dominican-eater.py validate --type all
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    setup_parser = subparsers.add_parser('setup', help='Setup project directories')
    
    scrape_parser = subparsers.add_parser('scrape', help='Scrape content metadata')
    scrape_parser.add_argument('--type', choices=['books', 'lyrics', 'poems', 'all'], default='all', help='Content type to scrape')
    
    download_parser = subparsers.add_parser('download', help='Download audio files')
    download_parser.add_argument('--type', choices=['books-eater', 'lyrics-eater', 'poems-eater', 'all'], default='all', help='Module to download')
    download_parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    
    transcribe_parser = subparsers.add_parser('transcribe', help='Transcribe audio with Whisper')
    transcribe_parser.add_argument('--type', choices=['books-eater', 'lyrics-eater', 'poems-eater', 'all'], default='all', help='Module to transcribe')
    transcribe_parser.add_argument('--model', default='base', help='Whisper model (tiny, base, small, medium, large)')
    transcribe_parser.add_argument('--force', action='store_true', help='Force re-transcription even if files exist')
    transcribe_parser.add_argument('--partial', action='store_true', help='Transcribe only first 60 seconds for quick testing')
    
    align_parser = subparsers.add_parser('align', help='Align transcriptions with reference text')
    align_parser.add_argument('--type', choices=['books-eater', 'lyrics-eater', 'poems-eater'], default='lyrics-eater', help='Module to align')
    
    validate_parser = subparsers.add_parser('validate', help='Validate pipeline outputs')
    validate_parser.add_argument('--type', choices=['books-eater', 'lyrics-eater', 'poems-eater', 'all'], default='all', help='Module to validate')
    
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete pipeline')
    pipeline_parser.add_argument('--type', choices=['books', 'lyrics', 'poems', 'all'], default='all', help='Content type to process')
    pipeline_parser.add_argument('--skip-scrape', action='store_true', help='Skip scraping step')
    pipeline_parser.add_argument('--skip-download', action='store_true', help='Skip download step')
    pipeline_parser.add_argument('--skip-transcribe', action='store_true', help='Skip transcription step')
    pipeline_parser.add_argument('--model', default='base', help='Whisper model to use')
    pipeline_parser.add_argument('--force', action='store_true', help='Force re-processing')
    pipeline_parser.add_argument('--partial', action='store_true', help='Transcribe only first 60 seconds for quick testing')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'setup':
        setup()
    elif args.command == 'scrape':
        scrape(args.type)
    elif args.command == 'download':
        download(args.type, force=args.force)
    elif args.command == 'transcribe':
        transcribe(args.type, model=args.model, force=args.force, partial=args.partial)
    elif args.command == 'align':
        align(args.type)
    elif args.command == 'validate':
        validate(args.type)
    elif args.command == 'pipeline':
        pipeline(
            args.type,
            skip_scrape=args.skip_scrape,
            skip_download=args.skip_download,
            skip_transcribe=args.skip_transcribe,
            force=args.force,
            model=args.model,
            partial=args.partial
        )

if __name__ == '__main__':
    main()
