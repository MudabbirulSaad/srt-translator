#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced CLI interface for SRT Translator using Typer and Rich.
"""

import os
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich import box
from pydantic import BaseModel, Field, validator

from ..utils.terminal_compat import (
    is_unicode_supported, get_safe_arrow, get_safe_bullet, 
    get_safe_box_style, enable_unicode_output
)

from ..translators import get_translator, TRANSLATOR_REGISTRY
from ..utils import setup_logging, read_srt_file, write_srt_file, get_subtitle_statistics
from ..utils.text import detect_language, normalize_language_code, extract_text_for_language_detection
from ..core import translate_subtitles, validate_translation_direction

# Try to enable Unicode output
enable_unicode_output()

# Create Typer app
app = typer.Typer(
    name="SRT Translator",
    help="A professional tool for translating subtitle files",
    add_completion=False,
)

# Create Rich console with appropriate encoding
console = Console(safe_box=not is_unicode_supported())

class TranslationConfig(BaseModel):
    """Configuration model for translation settings"""
    input_file: str = Field(..., description="Input SRT file to translate")
    output: Optional[str] = Field(None, description="Output SRT file")
    source_lang: str = Field("en", description="Source language code (ISO 639-1)")
    target_lang: str = Field("es", description="Target language code (ISO 639-1)")
    provider: str = Field("google", description="Translation provider to use")
    workers: int = Field(0, description="Number of worker threads for translation")
    batch_size: int = Field(5, description="Batch size for translation requests")
    verbose: bool = Field(False, description="Enable verbose logging")
    log_file: Optional[str] = Field(None, description="Log file path")
    libre_url: str = Field(
        "https://translate.argosopentech.com/translate",
        description="LibreTranslate API URL"
    )
    
    @validator('input_file')
    def validate_input_file(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"Input file not found: {v}")
        return v
    
    @validator('provider')
    def validate_provider(cls, v):
        if v not in TRANSLATOR_REGISTRY:
            raise ValueError(f"Unknown provider: {v}. Available providers: {', '.join(TRANSLATOR_REGISTRY.keys())}")
        return v

def setup_rich_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging with Rich handler"""
    logger = logging.getLogger("srt_translator")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create Rich handler
    rich_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=False,
        omit_repeated_times=False,
    )
    rich_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Add handler to logger
    logger.addHandler(rich_handler)
    
    return logger

def create_rich_progress() -> Progress:
    """Create a Rich progress bar"""
    # Use compatible separator character
    separator = get_safe_bullet()
    
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn(separator),
        TimeElapsedColumn(),
        TextColumn(separator),
        TimeRemainingColumn(),
        console=console,
        expand=True
    )

def display_subtitle_preview(subtitles, count=3):
    """Display a preview of the subtitles in a Rich table"""
    if not subtitles:
        return
    
    # Use terminal-compatible box style and arrow
    box_style = get_safe_box_style()
    arrow = get_safe_arrow()
    
    table = Table(title="Subtitle Preview", box=box_style)
    table.add_column("Index", style="cyan")
    table.add_column("Timing", style="green")
    table.add_column("Content", style="white")
    
    # Show first few subtitles
    for i, subtitle in enumerate(subtitles[:count]):
        table.add_row(
            str(subtitle.index),
            f"{subtitle.start} {arrow} {subtitle.end}",
            subtitle.content
        )
    
    # If there are more subtitles, add an ellipsis row
    if len(subtitles) > count:
        table.add_row("...", "...", "...")
    
    console.print(table)

def display_translation_summary(stats: Dict[str, Any], config: TranslationConfig, 
                               source_lang: str, target_lang: str, 
                               translator_stats: Dict[str, Any], 
                               total_time: float):
    """Display a summary of the translation job"""
    table = Table(title="Translation Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    # Add rows with statistics
    table.add_row("Input File", config.input_file)
    table.add_row("Output File", config.output)
    table.add_row("Translation Direction", f"{source_lang} → {target_lang}")
    table.add_row("Provider", config.provider)
    table.add_row("Subtitles Processed", str(stats["count"]))
    table.add_row("Total Characters", str(stats["total_chars"]))
    table.add_row("Average Characters/Subtitle", f"{stats['avg_chars']:.2f}")
    table.add_row("Worker Threads", str(translator_stats.get("worker_count", "N/A")))
    table.add_row("Processing Time", f"{total_time:.2f} seconds")
    table.add_row("Processing Speed", f"{stats['count'] / total_time:.2f} subtitles/second")
    table.add_row("Cache Hits", f"{translator_stats['cache_hit_count']} ({translator_stats['cache_hit_count']/stats['count']*100:.1f}%)")
    table.add_row("Successful Translations", str(translator_stats['success_count']))
    table.add_row("Failed Translations", str(translator_stats['error_count']))
    
    console.print(table)

@app.command()
def translate(
    input_file: str = typer.Argument(..., help="Input SRT file to translate"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output SRT file (default: input_file.target_lang.srt)"),
    source_lang: str = typer.Option("en", "--source-lang", "-s", help="Source language code (ISO 639-1)"),
    target_lang: str = typer.Option("es", "--target-lang", "-t", help="Target language code (ISO 639-1)"),
    provider: str = typer.Option("google", "--provider", "-p", help="Translation provider to use", 
                                show_choices=True, 
                                autocompletion=lambda: list(TRANSLATOR_REGISTRY.keys())),
    workers: int = typer.Option(0, "--workers", "-w", help="Number of worker threads for translation. Use 0 for auto, negative values for percentage of CPU cores"),
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Batch size for translation requests"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (default: auto-generated)"),
    libre_url: str = typer.Option("https://translate.argosopentech.com/translate", "--libre-url", help="LibreTranslate API URL (only used with --provider=libre)"),
    preview: bool = typer.Option(False, "--preview", help="Preview subtitles without translating"),
):
    """
    Translate SRT subtitle files using free translation APIs.
    
    Examples:
    
    - Translate from English to Spanish using Google Translate:
      $ srt-translator input.srt -t es
    
    - Translate from Japanese to English using MyMemory with 8 worker threads:
      $ srt-translator japanese.srt -s ja -t en -p mymemory -w 8
    
    - Use automatic thread count selection (based on CPU cores and subtitle count):
      $ srt-translator large_movie.srt -t de -w 0
    """
    # Record start time
    start_time = time.time()
    
    try:
        # Create configuration
        config = TranslationConfig(
            input_file=input_file,
            output=output,
            source_lang=source_lang,
            target_lang=target_lang,
            provider=provider,
            workers=workers,
            batch_size=batch_size,
            verbose=verbose,
            log_file=log_file,
            libre_url=libre_url
        )
        
        # Set up logging
        logger = setup_logging(config.log_file, config.verbose)
        
        # Generate output file name if not specified
        if not config.output:
            base_name, ext = os.path.splitext(config.input_file)
            config.output = f"{base_name}.{config.target_lang}{ext}"
        
        # Display job information
        console.print(Panel.fit(
            f"[bold cyan]SRT Translator[/bold cyan] - [bold]Starting translation job[/bold]",
            title="Job Info",
            subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue"
        ))
        
        # Read input SRT file
        with console.status("[bold green]Reading SRT file...[/bold green]"):
            subtitles = read_srt_file(config.input_file)
        
        # Get subtitle statistics
        stats = get_subtitle_statistics(subtitles)
        subtitle_count = stats["count"]
        
        # Display subtitle statistics
        console.print(Panel(
            f"Found [bold]{subtitle_count}[/bold] subtitles\n"
            f"Total characters: [bold]{stats['total_chars']}[/bold]\n"
            f"Average characters per subtitle: [bold]{stats['avg_chars']:.2f}[/bold]",
            title="Subtitle Statistics",
            border_style="green"
        ))
        
        # Display subtitle preview
        display_subtitle_preview(subtitles)
        
        # If preview mode, exit here
        if preview:
            console.print("[bold green]Preview completed. Exiting without translation.[/bold green]")
            return 0
        
        # Detect and validate translation direction
        with console.status("[bold green]Detecting language and validating translation direction...[/bold green]"):
            source_lang, target_lang = validate_translation_direction(
                config.source_lang,
                config.target_lang,
                subtitles
            )
        
        # Initialize translator
        with console.status(f"[bold green]Initializing {provider} translator...[/bold green]"):
            translator = get_translator(
                config.provider,
                source_lang,
                target_lang
            )
            logger.info(f"Translator initialized: {translator.__class__.__name__}")
        
        # Use terminal-compatible arrow
        arrow = get_safe_arrow()
        
        # Display translation direction
        console.print(f"Translation direction: [bold cyan]{source_lang}[/bold cyan] {arrow} [bold cyan]{target_lang}[/bold cyan]")
        
        # Determine optimal worker count if auto mode
        if config.workers <= 0:
            cpu_count = os.cpu_count() or 4
            if config.workers == 0:  # Auto mode
                if subtitle_count < 20:
                    effective_workers = 1
                elif subtitle_count < 100:
                    effective_workers = min(4, cpu_count)
                else:
                    effective_workers = min(8, cpu_count)
            else:  # Negative value = percentage of CPU cores
                percentage = abs(config.workers)
                effective_workers = max(1, int(cpu_count * percentage / 100))
            
            console.print(f"Auto-selected [bold]{effective_workers}[/bold] worker threads based on system resources and subtitle count")
        else:
            effective_workers = config.workers
        
        # Store worker count in translator for statistics
        translator.worker_count = effective_workers
        
        # Translate subtitles with rich progress bar
        console.print("[bold]Starting translation process...[/bold]")
        
        with create_rich_progress() as progress:
            # Create a task for the progress bar
            task_id = progress.add_task(f"[cyan]Translating[/cyan]", total=subtitle_count)
            
            # Define a progress callback
            def progress_callback(current, total):
                progress.update(task_id, completed=current)
            
            # Translate subtitles
            translated_subtitles = translate_subtitles(
                subtitles, 
                translator, 
                max_workers=effective_workers,
                batch_size=config.batch_size,
                progress_callback=progress_callback
            )
        
        # Write output SRT file
        with console.status(f"[bold green]Writing translated subtitles to {config.output}...[/bold green]"):
            write_srt_file(translated_subtitles, config.output)
        
        # Get translation statistics
        translator_stats = translator.get_stats()
        translator_stats["worker_count"] = effective_workers
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Display translation summary
        display_translation_summary(stats, config, source_lang, target_lang, translator_stats, total_time)
        
        # Display completion message
        console.print(Panel.fit(
            f"[bold green]Translation completed successfully![/bold green]\n"
            f"Output file: [bold]{config.output}[/bold]",
            title="Job Complete",
            subtitle=f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="green"
        ))
        
        return 0
    
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="Translation Failed",
            border_style="red"
        ))
        if verbose:
            console.print_exception()
        return 1

def main():
    """Main entry point for the CLI"""
    app()