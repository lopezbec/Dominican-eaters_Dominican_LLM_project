#!/usr/bin/env python3
"""
Dominican Eaters - Unified CLI Entry Point

A command-line interface for managing the entire Dominican Spanish content
collection and processing pipeline for LLM training.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import subprocess

import click

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='2.0', prog_name='Dominican Eaters')
def cli():
    """Dominican Eaters - Dominican Spanish LLM Training Dataset Pipeline
    
    A comprehensive toolkit for collecting, processing, and aligning
    Dominican Spanish audio content from YouTube.
    """
    pass


@cli.command()
@click.option(
    '--module',
    type=click.Choice(['books', 'lyrics', 'poems', 'all'], case_sensitive=False),
    default='all',
    help='Content module to scrape'
)
def scrape(module: str):
    """Scrape content metadata from web sources.
    
    Examples:
        cli.py scrape --module books
        cli.py scrape --module all
    """
    click.echo(f"Starting scrape for module: {module}")
    
    modules_to_run = []
    if module == 'all':
        modules_to_run = ['books', 'lyrics', 'poems']
    else:
        modules_to_run = [module]
    
    for mod in modules_to_run:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Running {mod}-eater scraper...")
            click.echo(f"{'='*60}")
            
            main_script = Path(f"{mod}-eater/main.py")
            if not main_script.exists():
                click.echo(f"Warning: {main_script} not found", err=True)
                continue
            
            result = subprocess.run(
                [sys.executable, str(main_script)],
                cwd=str(Path.cwd()),
                capture_output=False
            )
            
            if result.returncode != 0:
                click.echo(f"Error: {mod}-eater failed with code {result.returncode}", err=True)
            else:
                click.secho(f"✓ {mod}-eater completed successfully", fg='green')
                
        except Exception as e:
            click.echo(f"Error running {mod}-eater: {e}", err=True)
            logger.error(f"Failed to run {mod}-eater", exc_info=True)


@cli.command()
@click.option(
    '--module',
    type=click.Choice(['books-eater', 'lyrics-eater', 'poems-eater', 'all'], case_sensitive=False),
    required=True,
    help='Module to download audio from'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force re-download existing files'
)
def download(module: str, force: bool):
    """Download audio files from YouTube.
    
    Examples:
        cli.py download --module lyrics-eater
        cli.py download --module all --force
    """
    click.echo(f"Starting audio download for module: {module}")
    
    modules_to_run = []
    if module == 'all':
        modules_to_run = ['books-eater', 'lyrics-eater', 'poems-eater']
    else:
        modules_to_run = [module]
    
    for mod in modules_to_run:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Downloading audio for {mod}...")
            click.echo(f"{'='*60}")
            
            cmd = [
                sys.executable, '-m', 
                'audio_processing.src.downloader',
                '--module', mod
            ]
            if force:
                cmd.append('--force')
            
            result = subprocess.run(
                cmd,
                cwd=str(Path.cwd())
            )
            
            if result.returncode != 0:
                click.echo(f"Error: {mod} download failed", err=True)
            else:
                click.secho(f"✓ {mod} download completed", fg='green')
            
        except Exception as e:
            click.echo(f"Error downloading {mod}: {e}", err=True)
            logger.error(f"Failed to download {mod}", exc_info=True)


@cli.command()
@click.option(
    '--module',
    type=click.Choice(['books-eater', 'lyrics-eater', 'poems-eater', 'all'], case_sensitive=False),
    required=True,
    help='Module to transcribe'
)
@click.option(
    '--model',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']),
    default='base',
    help='Whisper model to use'
)
@click.option(
    '--partial',
    is_flag=True,
    help='Use partial transcription (first 45 seconds only, for lyrics validation)'
)
def transcribe(module: str, model: str, partial: bool):
    """Transcribe audio files using Whisper.
    
    Examples:
        cli.py transcribe --module lyrics-eater --model base
        cli.py transcribe --module books-eater --model large
        cli.py transcribe --module lyrics-eater --partial
    """
    click.echo(f"Starting transcription for module: {module} with model: {model}")
    
    modules_to_run = []
    if module == 'all':
        modules_to_run = ['books-eater', 'lyrics-eater', 'poems-eater']
    else:
        modules_to_run = [module]
    
    for mod in modules_to_run:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Transcribing {mod} audio files...")
            click.echo(f"{'='*60}")
            
            cmd = [
                sys.executable, '-m', 
                'audio_processing.src.transcriber',
                '--module', mod,
                '--model', model
            ]
            if partial:
                cmd.append('--partial')
            
            result = subprocess.run(
                cmd,
                cwd=str(Path.cwd())
            )
            
            if result.returncode != 0:
                click.echo(f"Error: {mod} transcription failed", err=True)
            else:
                click.secho(f"✓ {mod} transcription completed", fg='green')
            
        except Exception as e:
            click.echo(f"Error transcribing {mod}: {e}", err=True)
            logger.error(f"Failed to transcribe {mod}", exc_info=True)


@cli.command()
@click.option(
    '--module',
    type=click.Choice(['books-eater', 'lyrics-eater', 'poems-eater', 'all'], case_sensitive=False),
    required=True,
    help='Module to align'
)
def align(module: str):
    """Align and verify transcriptions against reference texts.
    
    Uses multi-metric evaluation (WER, character similarity, Jaccard, cosine)
    to assess transcription quality and find optimal alignment start positions.
    
    Examples:
        cli.py align --module lyrics-eater
        cli.py align --module all
    """
    click.echo(f"Starting alignment verification for module: {module}")
    
    modules_to_run = []
    if module == 'all':
        modules_to_run = ['books-eater', 'lyrics-eater', 'poems-eater']
    else:
        modules_to_run = [module]
    
    from audio_processing.src.aligner import ForcedAligner
    
    for mod in modules_to_run:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Aligning {mod} transcriptions...")
            click.echo(f"{'='*60}")
            
            aligner = ForcedAligner(config_path='audio_processing/config.yaml')
            report = aligner.align_module(mod)
            
            if 'error' not in report:
                click.secho(f"✓ {mod} alignment completed", fg='green')
                click.echo(f"  Results: {report['successful']}/{report['total_files']} successful")
                click.echo(f"  Average WER: {report['average_wer']:.3f}")
                click.echo(f"  Average Char Similarity: {report['average_char_similarity']:.3f}")
            else:
                click.echo(f"Warning: {report['error']}", err=True)
            
        except Exception as e:
            click.echo(f"Error aligning {mod}: {e}", err=True)
            logger.error(f"Failed to align {mod}", exc_info=True)


@cli.command()
@click.option(
    '--module',
    type=click.Choice(['books-eater', 'lyrics-eater', 'poems-eater', 'all'], case_sensitive=False),
    default='all',
    help='Module to validate'
)
def validate(module: str):
    """Validate outputs and generate quality reports.
    
    Checks for missing files, corrupted data, and quality metrics.
    
    Examples:
        cli.py validate --module lyrics-eater
        cli.py validate --module all
    """
    click.echo(f"Starting validation for module: {module}")
    
    modules_to_run = []
    if module == 'all':
        modules_to_run = ['books-eater', 'lyrics-eater', 'poems-eater']
    else:
        modules_to_run = [module]
    
    for mod in modules_to_run:
        try:
            click.echo(f"\n{'='*60}")
            click.echo(f"Validating {mod}...")
            click.echo(f"{'='*60}")
            
            cmd = [
                sys.executable, '-m', 
                'audio_processing.src.validator',
                '--module', mod
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(Path.cwd())
            )
            
            if result.returncode != 0:
                click.echo(f"Warning: {mod} validation had issues", err=True)
            else:
                click.secho(f"✓ {mod} validation completed", fg='green')
            
        except Exception as e:
            click.echo(f"Error validating {mod}: {e}", err=True)
            logger.error(f"Failed to validate {mod}", exc_info=True)


@cli.command()
@click.option(
    '--module',
    type=click.Choice(['books', 'lyrics', 'poems', 'all'], case_sensitive=False),
    default='all',
    help='Module to process'
)
@click.option(
    '--skip-scrape',
    is_flag=True,
    help='Skip scraping step'
)
@click.option(
    '--skip-download',
    is_flag=True,
    help='Skip download step'
)
@click.option(
    '--skip-transcribe',
    is_flag=True,
    help='Skip transcription step'
)
@click.option(
    '--skip-align',
    is_flag=True,
    help='Skip alignment step'
)
@click.option(
    '--model',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']),
    default='base',
    help='Whisper model to use for transcription'
)
@click.option(
    '--force',
    is_flag=True,
    help='Force re-processing of existing files'
)
def pipeline(module: str, skip_scrape: bool, skip_download: bool, 
             skip_transcribe: bool, skip_align: bool, model: str, force: bool):
    """Run the complete pipeline for a module.
    
    Executes: scrape → download → transcribe → align → validate
    
    Examples:
        cli.py pipeline --module lyrics
        cli.py pipeline --module all --skip-scrape --model large
        cli.py pipeline --module books --force
    """
    click.echo(f"Starting pipeline for module: {module}")
    click.echo(f"Model: {model}, Force: {force}")
    click.echo(f"{'='*60}\n")
    
    ctx = click.get_current_context()
    
    # Step 1: Scrape
    if not skip_scrape:
        click.echo("Step 1/5: Scraping...")
        ctx.invoke(scrape, module=module)
    else:
        click.echo("Step 1/5: Scraping [SKIPPED]")
    
    # Convert module names for audio processing
    module_suffix = '-eater'
    if module == 'all':
        process_module = 'all'
    else:
        process_module = f"{module}{module_suffix}"
    
    # Step 2: Download
    if not skip_download:
        click.echo("\nStep 2/5: Downloading audio...")
        ctx.invoke(download, module=process_module, force=force)
    else:
        click.echo("\nStep 2/5: Downloading audio [SKIPPED]")
    
    # Step 3: Transcribe
    if not skip_transcribe:
        click.echo("\nStep 3/5: Transcribing...")
        # Use partial for lyrics by default
        use_partial = (module == 'lyrics')
        ctx.invoke(transcribe, module=process_module, model=model, partial=use_partial)
    else:
        click.echo("\nStep 3/5: Transcribing [SKIPPED]")
    
    # Step 4: Align
    if not skip_align:
        click.echo("\nStep 4/5: Aligning...")
        ctx.invoke(align, module=process_module)
    else:
        click.echo("\nStep 4/5: Aligning [SKIPPED]")
    
    # Step 5: Validate
    click.echo("\nStep 5/5: Validating...")
    ctx.invoke(validate, module=process_module)
    
    click.echo(f"\n{'='*60}")
    click.secho("✓ Pipeline completed successfully!", fg='green', bold=True)
    click.echo(f"{'='*60}")


@cli.command()
def setup():
    """Set up project directories and dependencies.
    
    Creates all required directories for data storage and processing.
    """
    click.echo("Setting up Dominican Eaters project structure...")
    
    directories = [
        'audio_processing/audio/books',
        'audio_processing/audio/lyrics',
        'audio_processing/audio/poems',
        'audio_processing/transcriptions/books',
        'audio_processing/transcriptions/lyrics',
        'audio_processing/transcriptions/poems',
        'audio_processing/reference_texts/books',
        'audio_processing/reference_texts/lyrics',
        'audio_processing/reference_texts/poems',
        'audio_processing/alignments/books',
        'audio_processing/alignments/lyrics',
        'audio_processing/alignments/poems',
        'audio_processing/reports/books',
        'audio_processing/reports/lyrics',
        'audio_processing/reports/poems',
        'audio_processing/data',
        'books-eater/audio',
        'books-eater/transcriptions',
        'books-eater/reports',
        'lyrics-eater/audio',
        'lyrics-eater/transcriptions',
        'lyrics-eater/reference_texts',
        'lyrics-eater/alignments',
        'lyrics-eater/reports',
        'poems-eater/audio',
        'poems-eater/transcriptions',
        'poems-eater/reference_texts',
        'poems-eater/alignments',
        'poems-eater/reports',
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"  ✓ Created: {directory}")
    
    click.secho("\n✓ Setup completed successfully!", fg='green', bold=True)
    click.echo("\nNext steps:")
    click.echo("  1. Configure .env file (for lyrics-eater GENIUS_TOKEN)")
    click.echo("  2. Run: cli.py scrape --module all")
    click.echo("  3. Run: cli.py pipeline --module all")


if __name__ == '__main__':
    cli()
