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
import srt

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich import box
from rich.layout import Layout
from pydantic import BaseModel, Field, validator

from ..utils.terminal_compat import (
    is_unicode_supported, get_safe_arrow, get_safe_bullet, 
    get_safe_box_style, enable_unicode_output
)

from ..translators import get_translator, TRANSLATOR_REGISTRY
from ..enhancers import get_enhancer
from ..utils import setup_logging, read_srt_file, write_srt_file, get_subtitle_statistics
from ..utils.text import detect_language, normalize_language_code, extract_text_for_language_detection
from ..core import translate_subtitles, validate_translation_direction
from ..transcribe.whisper_transcriber import transcribe_to_srt, WhisperTranscriber

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

class TranscribeConfig(BaseModel):
    """Configuration model for transcription settings"""
    input_file: str = Field(..., description="Input audio/video file to transcribe")
    output: Optional[str] = Field(None, description="Output SRT file")
    model_size: str = Field("base", description="Whisper model size")
    language: Optional[str] = Field(None, description="Language code for transcription (auto-detect if None)")
    device: Optional[str] = Field(None, description="Device to use for inference (cpu, cuda, auto)")
    task: str = Field("transcribe", description="Task to perform (transcribe or translate)")
    verbose: bool = Field(False, description="Enable verbose logging")
    log_file: Optional[str] = Field(None, description="Log file path")
    word_timestamps: bool = Field(True, description="Include word-level timestamps")
    highlight_words: bool = Field(False, description="Highlight words in the output")
    translate: bool = Field(False, description="Translate the generated subtitles after transcription")
    target_lang: Optional[str] = Field(None, description="Target language for translation (if translate=True)")
    translation_provider: str = Field("google", description="Translation provider to use (if translate=True)")
    enhance: bool = Field(False, description="Enhance translations using AI (if translate=True)")
    enhance_model: str = Field("llama3", description="Ollama model to use for enhancement (if translate=True)")
    
    @validator('input_file')
    def validate_input_file(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"Input file not found: {v}")
        return v
    
    @validator('model_size')
    def validate_model_size(cls, v):
        valid_models = ["tiny", "base", "small", "medium", "large", "turbo"]
        if v not in valid_models:
            raise ValueError(f"Invalid model size: {v}. Valid options: {', '.join(valid_models)}")
        return v
    
    @validator('task')
    def validate_task(cls, v):
        valid_tasks = ["transcribe", "translate"]
        if v not in valid_tasks:
            raise ValueError(f"Invalid task: {v}. Valid options: {', '.join(valid_tasks)}")
        return v
    
    @validator('translation_provider')
    def validate_provider(cls, v):
        if v not in TRANSLATOR_REGISTRY:
            raise ValueError(f"Unknown provider: {v}. Available providers: {', '.join(TRANSLATOR_REGISTRY.keys())}")
        return v


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
    preview: bool = Field(False, description="Preview subtitles without translating")
    enhance: bool = Field(False, description="Enhance translations using AI")
    enhance_model: str = Field("llama3", description="Ollama model to use for enhancement")
    enhance_host: Optional[str] = Field(None, description="Custom Ollama API endpoint")
    enhance_temperature: float = Field(0.7, description="Temperature for the Ollama model (0.0 to 1.0)")
    
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
    """Create an enhanced Rich progress bar with professional styling"""
    return Progress(
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(
            bar_width=None,
            complete_style="green",
            finished_style="bold green",
            pulse_style="cyan"
        ),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True
    )

def display_subtitle_preview(subtitles: List[srt.Subtitle], max_display: int = 5):
    """Display an enhanced preview of subtitles with professional styling"""
    # Create header text
    header_text = Text()
    header_text.append("🎬 ", style="bold yellow")
    header_text.append(f"Showing {min(max_display, len(subtitles))} of {len(subtitles)} subtitles", style="cyan")
    
    # Create enhanced subtitle preview table
    table = Table(
        title="[bold yellow]Subtitle Preview[/bold yellow]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        padding=(0, 1),
        expand=True
    )
    
    table.add_column("#", style="cyan", no_wrap=True, justify="center", width=5)
    table.add_column("Timing", style="magenta", width=20)
    table.add_column("Content", style="green", ratio=3)
    
    # Add rows for subtitles (limited to max_display)
    for i, subtitle in enumerate(subtitles[:max_display]):
        start_time = str(subtitle.start).split('.')[0]
        end_time = str(subtitle.end).split('.')[0]
        timing = f"{start_time} {get_safe_arrow()} {end_time}"
        
        # Format content with syntax highlighting if it contains markup
        content = subtitle.content
        if "<" in content and ">" in content:
            content = f"[dim italic]{content}[/dim italic]"
        
        # Add row with alternating background for better readability
        row_style = "on grey7" if i % 2 == 0 else ""
        table.add_row(
            f"[bold]{subtitle.index}[/bold]", 
            timing, 
            content,
            style=row_style
        )
    
    # Add ellipsis row if there are more subtitles
    if len(subtitles) > max_display:
        remaining = len(subtitles) - max_display
        table.add_row(
            "...", 
            "...", 
            f"[dim italic]+ {remaining} more subtitle{'' if remaining == 1 else 's'} not shown[/dim italic]"
        )
    
    # Print the header and table
    console.print(header_text)
    console.print(table)
    
    # Print tip
    tip_text = Text()
    tip_text.append("💡 ", style="yellow")
    tip_text.append("TIP: ", style="bold yellow")
    tip_text.append("Use ", style="dim")
    tip_text.append("--preview", style="bold cyan")
    tip_text.append(" flag to view subtitles without translating", style="dim")
    console.print(tip_text)

def display_job_info(config: TranslationConfig):
    """Display job information with enhanced visual styling"""
    # Create a rich layout for better visual organization
    job_details = []
    
    # Add file information with icons
    job_details.append(f"[bold cyan]📂 Input:[/bold cyan] [white]{config.input_file}[/white]")
    job_details.append(f"[bold cyan]📤 Output:[/bold cyan] [white]{config.output}[/white]")
    
    # Add translation details with icons
    job_details.append(f"[bold cyan]🌐 Translation:[/bold cyan] [white]{config.source_lang} → {config.target_lang}[/white]")
    job_details.append(f"[bold cyan]🔄 Provider:[/bold cyan] [white]{config.provider}[/white]")
    
    # Add enhancement details if enabled
    if config.enhance:
        job_details.append(f"[bold cyan]🧠 AI Enhancement:[/bold cyan] [green]Enabled[/green] ([white]{config.enhance_model}[/white])")
        job_details.append(f"[bold cyan]🌡️ Temperature:[/bold cyan] [white]{config.enhance_temperature}[/white]")
    
    # Create the panel with the details
    console.print(Panel(
        "\n".join(job_details),
        title="[bold blue]SRT Translator Pro[/bold blue]",
        subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        border_style="blue",
        padding=(1, 2),
        box=box.ROUNDED
    ))

def display_subtitle_stats(stats: Dict[str, Any]):
    """Display subtitle statistics with enhanced styling"""
    # Create a simple but visually appealing panel
    stats_text = Text()
    stats_text.append("\n📊 ", style="bold green")
    stats_text.append("Found ", style="white")
    stats_text.append(f"{stats['count']}", style="bold green")
    stats_text.append(" subtitles\n\n", style="white")
    
    stats_text.append("📝 ", style="bold yellow")
    stats_text.append("Total characters: ", style="white")
    stats_text.append(f"{stats['total_chars']:,}", style="bold yellow")
    stats_text.append("\n\n", style="white")
    
    stats_text.append("📏 ", style="bold magenta")
    stats_text.append("Average characters per subtitle: ", style="white")
    stats_text.append(f"{stats['avg_chars']:.2f}", style="bold magenta")
    stats_text.append("\n", style="white")
    
    # Create a simple bar visualization
    max_bar_width = 20
    normalized_count = min(max_bar_width, stats['count'])
    bar = "█" * normalized_count
    stats_text.append(f"\nSubtitles:  {bar} {stats['count']}", style="green")
    
    # Print the panel
    console.print(Panel(
        stats_text,
        title="[bold green]Subtitle Statistics[/bold green]",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2)
    ))

def display_translation_summary(stats: Dict[str, Any], config: TranslationConfig, 
                               source_lang: str, target_lang: str, 
                               translator_stats: Dict[str, Any], 
                               total_time: float):
    """Display a professional summary of the translation job with enhanced visuals"""
    # Create header with logo and title
    header_text = Text()
    header_text.append("🌐 ", style="bold blue")
    header_text.append("SRT TRANSLATOR PRO", style="bold blue")
    header_text.append(" | ", style="dim")
    header_text.append("Translation Summary", style="bold cyan")
    
    console.print(header_text)
    
    # Create main table with all information
    table = Table(
        title="Translation Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        expand=True
    )
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    # Add rows with statistics
    table.add_row("Input File", config.input_file)
    table.add_row("Output File", config.output)
    table.add_row("Translation Direction", f"[bold]{source_lang}[/bold] → [bold]{target_lang}[/bold]")
    table.add_row("Provider", config.provider)
    
    # Add AI enhancement info if enabled
    if config.enhance:
        table.add_row("AI Enhancement", f"[green]Enabled[/green] (Ollama: {config.enhance_model})")
        table.add_row("Enhancement Temperature", f"{config.enhance_temperature}")
    
    table.add_row("Subtitles Processed", str(stats["count"]))
    table.add_row("Total Characters", f"{stats['total_chars']:,}")
    table.add_row("Average Characters/Subtitle", f"{stats['avg_chars']:.2f}")
    table.add_row("Worker Threads", str(translator_stats.get("worker_count", "N/A")))
    table.add_row("Processing Time", f"{total_time:.2f} seconds")
    table.add_row("Processing Speed", f"{stats['count'] / total_time:.2f} subtitles/second")
    
    # Add cache and success metrics with color coding
    cache_hit_percent = (translator_stats['cache_hit_count']/stats['count']*100) if stats['count'] > 0 else 0
    table.add_row("Cache Hits", f"{translator_stats['cache_hit_count']} ([yellow]{cache_hit_percent:.1f}%[/yellow])")
    
    success_percent = (translator_stats['success_count']/stats['count']*100) if stats['count'] > 0 else 0
    success_style = "green" if success_percent == 100 else "yellow" if success_percent > 90 else "red"
    table.add_row("Successful Translations", f"{translator_stats['success_count']} ([{success_style}]{success_percent:.1f}%[/{success_style}])")
    
    if translator_stats['error_count'] > 0:
        table.add_row("Failed Translations", f"[bold red]{translator_stats['error_count']}[/bold red]")
    else:
        table.add_row("Failed Translations", f"[green]0[/green]")
    
    # Print the table
    console.print(table)
    
    # Print completion message
    footer_text = Text()
    footer_text.append("✅ ", style="green")
    footer_text.append("Translation completed at ", style="dim")
    footer_text.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="bold white")
    console.print(footer_text)

@app.command()
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
    enhance: bool = typer.Option(False, "--enhance", help="Enhance translations using AI"),
    enhance_model: str = typer.Option("llama3", "--enhance-model", help="Ollama model to use for enhancement"),
    enhance_host: Optional[str] = typer.Option(None, "--enhance-host", help="Custom Ollama API endpoint (e.g., http://localhost:11434)"),
    enhance_temperature: float = typer.Option(0.7, "--enhance-temperature", help="Temperature for the Ollama model (0.0 to 1.0)"),
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
      
    - Enhance translations using Ollama AI model:
      $ srt-translator input.srt -t fr --enhance --enhance-model llama3
      
    - Use a custom Ollama endpoint with specific temperature:
      $ srt-translator input.srt -t es --enhance --enhance-host http://192.168.1.100:11434 --enhance-temperature 0.5
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Set up logging
        logger = setup_logging(log_file=log_file, verbose=verbose)
        
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
            libre_url=libre_url,
            preview=preview,
            enhance=enhance,
            enhance_model=enhance_model,
            enhance_host=enhance_host,
            enhance_temperature=enhance_temperature
        )
        
        # Set default output file if not specified
        if not config.output:
            input_base = os.path.splitext(config.input_file)[0]
            config.output = f"{input_base}.{config.target_lang}.srt"
        
        # Display job information
        display_job_info(config)
        
        # Read SRT file
        with console.status("[bold green]Reading SRT file...[/bold green]"):
            subtitles = read_srt_file(config.input_file)
            logger.info(f"Found {len(subtitles)} subtitles")
        
        # Get subtitle statistics
        stats = get_subtitle_statistics(subtitles)
        display_subtitle_stats(stats)
        
        # Preview subtitles if requested
        if config.preview:
            display_subtitle_preview(subtitles)
            console.print("Preview completed. Exiting without translation.")
            return 0
        
        # Detect and validate language
        with console.status("[bold green]Detecting language and validating translation direction...[/bold green]"):
            source_lang, target_lang = validate_translation_direction(
                config.source_lang, 
                config.target_lang, 
                subtitles
            )
        
        # Initialize translator
        with console.status("[bold green]Initializing translator...[/bold green]"):
            # Initialize translator with appropriate parameters
            translator_kwargs = {
                "source_lang": source_lang,
                "target_lang": target_lang,
            }
            
            # Add provider-specific parameters
            if config.provider == "libre":
                translator_kwargs["api_url"] = config.libre_url
            
            translator = get_translator(
                provider=config.provider,
                **translator_kwargs
            )
            logger.info(f"Translator initialized: {translator.__class__.__name__}")
        
        console.print(f"Translation direction: {source_lang} → {target_lang}")
        
        # Determine number of worker threads
        subtitle_count = len(subtitles)
        if config.workers == 0:  # Auto mode
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            
            # Heuristic: 1 worker per 20 subtitles, but no more than CPU count
            if subtitle_count < 20:
                effective_workers = 1
            elif subtitle_count < 100:
                effective_workers = min(subtitle_count // 20 + 1, cpu_count)
            else:
                effective_workers = min(8, cpu_count)
            
            console.print(f"Auto-selected [bold]{effective_workers}[/bold] worker threads based on system resources and subtitle count")
        elif config.workers < 0:  # Percentage of CPU cores
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
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
            
            # Create enhancer if enabled
            enhancer = None
            if config.enhance:
                console.print(f"[bold]AI enhancement enabled using Ollama model: [cyan]{config.enhance_model}[/cyan][/bold]")
                
                # Check if Ollama is installed
                try:
                    import ollama
                    console.print("[green]✓[/green] Ollama Python package is installed")
                    
                    # Try to list models to verify connection
                    try:
                        client = ollama.Client(host=config.enhance_host)
                        response = client.list()
                        
                        # Handle different response types
                        try:
                            # Convert response to dictionary if it's not already
                            response_dict = response
                            if not isinstance(response, dict):
                                # Try to access as attribute
                                if hasattr(response, 'models'):
                                    model_names = [model["name"] for model in response.models]
                                    logger.info(f"Found models via attribute access: {model_names}")
                                # Try to convert to dict if it has a __dict__ attribute
                                elif hasattr(response, '__dict__'):
                                    response_dict = response.__dict__
                                    logger.info(f"Converted response to dict via __dict__: {list(response_dict.keys())}")
                                else:
                                    # Try to serialize to dict
                                    import json
                                    try:
                                        response_dict = json.loads(json.dumps(response))
                                        logger.info(f"Converted response to dict via json: {list(response_dict.keys())}")
                                    except:
                                        # Last resort: convert to string and log
                                        logger.info(f"Ollama API Response: {str(response)}")
                                        # Assume model is available
                                        model_names = [config.enhance_model]
                                        logger.info(f"Using fallback model: {model_names}")
                            
                            # Extract model names with flexible handling
                            model_names = []
                            
                            # If we have a dictionary, try to extract model names
                            if isinstance(response_dict, dict):
                                logger.info(f"Ollama API Response keys: {list(response_dict.keys())}")
                                
                                # Try different known response formats
                                if "models" in response_dict:
                                    model_names = [model["name"] for model in response_dict["models"]]
                                elif "models" in response_dict.get("data", {}):
                                    model_names = [model["name"] for model in response_dict["data"]["models"]]
                                else:
                                    # Try to extract from any list of dictionaries in the response
                                    for key, value in response_dict.items():
                                        if isinstance(value, list):
                                            for item in value:
                                                if isinstance(item, dict) and "name" in item:
                                                    model_names.append(item["name"])
                        except Exception as e:
                            logger.warning(f"Error parsing Ollama API response: {e}")
                            # Fallback to assuming the model is available
                            model_names = [config.enhance_model]
                            logger.info(f"Using fallback model after error: {model_names}")
                        
                        console.print(f"[green]✓[/green] Connected to Ollama server")
                        
                        if model_names:
                            console.print(f"[bold]Available models:[/bold] {', '.join(model_names)}")
                            
                            # Check if the specified model is available
                            if config.enhance_model not in model_names:
                                console.print(f"[bold yellow]Warning:[/bold yellow] Model [cyan]{config.enhance_model}[/cyan] not found in available models")
                                console.print(f"You may need to pull the model first with: [bold]ollama pull {config.enhance_model}[/bold]")
                                console.print(f"[bold]Proceeding with enhancement anyway[/bold] - Ollama will attempt to use the model")
                            else:
                                console.print(f"[green]✓[/green] Model [cyan]{config.enhance_model}[/cyan] is available")
                        else:
                            console.print(f"[bold yellow]Warning:[/bold yellow] Could not retrieve model list from Ollama API")
                            console.print(f"[bold]Proceeding with enhancement anyway[/bold] - Ollama will attempt to use model: [cyan]{config.enhance_model}[/cyan]")
                        
                        # Initialize the enhancer
                        try:
                            enhancer = get_enhancer(
                                provider="ollama",
                                source_lang=source_lang,
                                target_lang=target_lang,
                                model=config.enhance_model,
                                host=config.enhance_host,
                                temperature=config.enhance_temperature
                            )
                            console.print(f"[green]✓[/green] Enhancement initialized successfully")
                        except Exception as e:
                            console.print(f"[bold red]Error:[/bold red] Failed to initialize enhancer: {str(e)}")
                            if not typer.confirm("Continue without enhancement?", default=True):
                                return 1
                    
                    except Exception as e:
                        console.print(f"[bold red]Error:[/bold red] Failed to connect to Ollama server: {str(e)}")
                        console.print("Make sure Ollama is running with: [bold]ollama serve[/bold]")
                        if not typer.confirm("Continue without enhancement?", default=True):
                            return 1
                
                except ImportError:
                    console.print("[bold red]Error:[/bold red] Ollama Python package is not installed")
                    console.print("Install it with: [bold]pip install ollama[/bold]")
                    if not typer.confirm("Continue without enhancement?", default=True):
                        return 1
            
            # Translate subtitles
            translated_subtitles = translate_subtitles(
                subtitles, 
                translator, 
                max_workers=effective_workers,
                batch_size=config.batch_size,
                progress_callback=progress_callback,
                enhancer=enhancer
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
        
        # Display enhanced completion message
        # Create a visually appealing panel
        completion_text = Text()
        completion_text.append("\n✅ ", style="bold green")
        completion_text.append("Translation completed successfully!\n\n", style="bold green")
        
        completion_text.append("📄 ", style="bold blue")
        completion_text.append("Output file: ", style="bold cyan")
        completion_text.append(f"{config.output}\n\n", style="white")
        
        # Add stats
        completion_text.append("📊 ", style="bold yellow")
        completion_text.append("Statistics: ", style="bold cyan")
        completion_text.append(f"{stats['count']} subtitles processed in ", style="white")
        completion_text.append(f"{total_time:.2f} seconds", style="bold yellow")
        
        # Add enhancement info if enabled
        if config.enhance:
            completion_text.append("\n\n🧠 ", style="bold magenta")
            completion_text.append("AI Enhancement: ", style="bold cyan")
            completion_text.append(f"Applied using ", style="white")
            completion_text.append(f"{config.enhance_model}", style="bold magenta")
            completion_text.append(" model", style="white")
        
        # Print the panel
        console.print(Panel(
            completion_text,
            title="[bold green]Job Complete[/bold green]",
            subtitle=f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2)
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

@app.command()
def translate_srt_file(
    srt_file: str,
    source_lang: str,
    target_lang: str,
    provider: str = "google",
    enhance: bool = False,
    enhance_model: str = "llama3.2-vision:11b",
    verbose: bool = False,
    log_file: Optional[str] = None
) -> int:
    """
    Translate an SRT file using the specified provider and options.
    
    Args:
        srt_file: Path to the SRT file to translate
        source_lang: Source language code
        target_lang: Target language code
        provider: Translation provider to use
        enhance: Whether to enhance translations using AI
        enhance_model: Ollama model to use for enhancement
        verbose: Whether to enable verbose logging
        log_file: Path to the log file
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Generate output path
        base_name, ext = os.path.splitext(srt_file)
        output_file = f"{base_name}.{target_lang}{ext}"
        
        # Display translation info
        console.print("\n")
        console.print(Panel(
            f"🌐 [bold cyan]Translating:[/bold cyan] [white]{srt_file}[/white]\n"
            f"📄 [bold cyan]Output:[/bold cyan] [white]{output_file}[/white]\n"
            f"🔤 [bold cyan]Languages:[/bold cyan] [white]{source_lang} → {target_lang}[/white]\n"
            f"🔄 [bold cyan]Provider:[/bold cyan] [white]{provider}[/white]"
            + (f"\n🧠 [bold cyan]Enhancement:[/bold cyan] [green]Enabled[/green] ([white]{enhance_model}[/white])" if enhance else ""),
            title="[bold blue]SRT Translator Pro[/bold blue]",
            subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
            padding=(1, 2),
            box=box.ROUNDED
        ))
        
        logger.info(f"Starting translation with provider: {provider}")
        logger.info(f"Source language: {source_lang}, Target language: {target_lang}")
        logger.info(f"Enhancement: {enhance}, Model: {enhance_model}")
        
        # Read the subtitles
        subtitles = read_srt_file(srt_file)
        subtitle_count = len(subtitles)
        logger.info(f"Read {subtitle_count} subtitles from {srt_file}")
        
        # Get translator
        translator = get_translator(
            provider=provider,
            source_lang=source_lang,
            target_lang=target_lang,
            libre_url="https://translate.argosopentech.com/translate"  # Default LibreTranslate URL
        )
        
        # Get enhancer if enabled
        enhancer = None
        if enhance:
            from ..enhancers import get_enhancer
            enhancer = get_enhancer(
                provider="ollama",
                source_lang=source_lang,
                target_lang=target_lang,
                model=enhance_model,
                temperature=0.7,
                host=None  # Use default Ollama host
            )
            logger.info(f"Created enhancer with model: {enhance_model}")
        
        # Determine worker count (auto-select)
        effective_workers = 0
        
        # Create progress display
        with create_rich_progress() as progress:
            # Create a task for the progress bar
            task_id = progress.add_task(f"[cyan]Translating[/cyan]", total=subtitle_count)
            
            # Define a progress callback
            def progress_callback(current, total):
                progress.update(task_id, completed=current)
            
            # Translate the subtitles
            start_time = time.time()
            result = translate_subtitles(
                subtitles=subtitles,
                translator=translator,
                enhancer=enhancer,
                workers=effective_workers,
                batch_size=5,
                progress_callback=progress_callback
            )
            
            # Get translated subtitles and statistics
            translated_subtitles = result["subtitles"]
            translator_stats = result["stats"]
            
            # Write the translated subtitles to file
            write_srt_file(translated_subtitles, output_file)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Display translation summary
            stats = get_subtitle_statistics(translated_subtitles)
            
            # Create a simplified TranslationConfig for the summary display
            config = TranslationConfig(
                input_file=srt_file,
                output=output_file,
                source_lang=source_lang,
                target_lang=target_lang,
                provider=provider,
                enhance=enhance,
                enhance_model=enhance_model
            )
            
            display_translation_summary(
                stats=stats,
                config=config,
                source_lang=source_lang,
                target_lang=target_lang,
                translator_stats=translator_stats,
                total_time=total_time
            )
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        console.print(Panel(
            f"[bold red]Error during translation:[/bold red] {str(e)}",
            title="Translation Failed",
            border_style="red"
        ))
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
            console.print_exception()
        return 1


@app.command()
def transcribe(
    input_file: str = typer.Argument(..., help="Input audio/video file to transcribe"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output SRT file (default: input_file.srt)"),
    model_size: str = typer.Option("base", "--model-size", "-m", help="Whisper model size (tiny, base, small, medium, large, turbo)"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code for transcription (auto-detect if not specified)"),
    device: Optional[str] = typer.Option(None, "--device", "-d", help="Device to use for inference (cpu, cuda, auto)"),
    task: str = typer.Option("transcribe", "--task", help="Task to perform (transcribe or translate to English)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (default: auto-generated)"),
    word_timestamps: bool = typer.Option(True, "--word-timestamps", help="Include word-level timestamps"),
    highlight_words: bool = typer.Option(False, "--highlight-words", help="Highlight words in the output"),
    translate: bool = typer.Option(False, "--translate", "-t", help="Translate the generated subtitles after transcription"),
    target_lang: Optional[str] = typer.Option(None, "--target-lang", help="Target language for translation (required if --translate is used)"),
    translation_provider: str = typer.Option("google", "--provider", "-p", help="Translation provider to use for translation",
                                           show_choices=True,
                                           autocompletion=lambda: list(TRANSLATOR_REGISTRY.keys())),
    enhance: bool = typer.Option(False, "--enhance", help="Enhance translations using AI (only if --translate is used)"),
    enhance_model: str = typer.Option("llama3.2-vision:11b", "--enhance-model", help="Ollama model to use for enhancement"),
):
    """
    Transcribe audio/video files to SRT subtitles using Whisper.
    
    Examples:
    
    - Transcribe a video file with auto-detected language:
      $ srt-translator transcribe video.mp4
    
    - Transcribe an audio file with a specific language and model:
      $ srt-translator transcribe audio.mp3 --language ja --model-size medium
    
    - Transcribe and then translate to Spanish:
      $ srt-translator transcribe video.mp4 --translate --target-lang es
      
    - Transcribe, translate, and enhance with Ollama:
      $ srt-translator transcribe video.mp4 --translate --target-lang fr --enhance
    """
    try:
        # Set up logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_file is None:
            log_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"srt_translator_{timestamp}.log")
        
        logger = setup_rich_logging(verbose)
        setup_logging(log_file, verbose)
        logger.info(f"Log file: {log_file}")
        
        # Create configuration
        config = TranscribeConfig(
            input_file=input_file,
            output=output,
            model_size=model_size,
            language=language,
            device=device,
            task=task,
            verbose=verbose,
            log_file=log_file,
            word_timestamps=word_timestamps,
            highlight_words=highlight_words,
            translate=translate,
            target_lang=target_lang,
            translation_provider=translation_provider,
            enhance=enhance,
            enhance_model=enhance_model,
        )
        
        # Generate output path if not provided
        if config.output is None:
            base_name, _ = os.path.splitext(config.input_file)
            config.output = f"{base_name}.srt"
        
        # Display job information
        console.print(Panel(
            f"🎬 [bold cyan]Transcribing:[/bold cyan] [white]{config.input_file}[/white]\n"
            f"📄 [bold cyan]Output:[/bold cyan] [white]{config.output}[/white]\n"
            f"🔊 [bold cyan]Model:[/bold cyan] [white]Whisper {config.model_size}[/white]\n"
            f"🌐 [bold cyan]Language:[/bold cyan] [white]{config.language or 'auto-detect'}[/white]\n"
            f"🖥️ [bold cyan]Device:[/bold cyan] [white]{config.device or 'auto'}[/white]",
            title="[bold blue]SRT Transcriber Pro[/bold blue]",
            subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue",
            padding=(1, 2),
            box=box.ROUNDED
        ))
        
        # Start timer
        start_time = time.time()
        
        # Transcribe the file
        logger.info(f"Starting transcription with Whisper {config.model_size}...")
        
        # Create progress display
        with Progress(
            TextColumn("[bold blue]Transcribing[/bold blue]"),
            BarColumn(bar_width=None, complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            expand=True
        ) as progress:
            task = progress.add_task("Processing audio/video...", total=100)
            
            # Update progress to show activity (actual progress not available from Whisper)
            for i in range(1, 90):
                progress.update(task, completed=i)
                time.sleep(0.05)
            
            # Perform transcription
            srt_file = transcribe_to_srt(
                file_path=config.input_file,
                output_path=config.output,
                model_size=config.model_size,
                language=config.language,
                device=config.device,
                task=config.task,
                verbose=config.verbose,
                word_timestamps=config.word_timestamps,
                highlight_words=config.highlight_words
            )
            
            # Complete progress
            progress.update(task, completed=100)
        
        # Calculate time taken
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
        
        # Read the generated SRT file to get statistics
        subtitles = read_srt_file(srt_file)
        stats = get_subtitle_statistics(subtitles)
        
        # Display subtitle statistics
        display_subtitle_stats(stats)
        
        # Display subtitle preview
        display_subtitle_preview(subtitles)
        
        # Translate if requested
        if config.translate:
            if not config.target_lang:
                raise ValueError("Target language (--target-lang) is required when using --translate")
            
            # Detect source language if not specified
            source_lang = config.language
            if not source_lang:
                # Try to detect the language from the subtitles
                sample_text = extract_text_for_language_detection(subtitles)
                detected_lang = detect_language(sample_text)
                source_lang = normalize_language_code(detected_lang)
                logger.info(f"Detected source language: {source_lang}")
            
            # Generate output path for translation
            base_name, ext = os.path.splitext(srt_file)
            output_file = f"{base_name}.{config.target_lang}{ext}"
            
            # Display translation info
            console.print("\n")
            console.print(Panel(
                f"🌐 [bold cyan]Translating:[/bold cyan] [white]{srt_file}[/white]\n"
                f"📄 [bold cyan]Output:[/bold cyan] [white]{output_file}[/white]\n"
                f"🔤 [bold cyan]Languages:[/bold cyan] [white]{source_lang} → {config.target_lang}[/white]\n"
                f"🔄 [bold cyan]Provider:[/bold cyan] [white]{config.translation_provider}[/white]"
                + (f"\n🧠 [bold cyan]Enhancement:[/bold cyan] [green]Enabled[/green] ([white]{config.enhance_model}[/white])" if config.enhance else ""),
                title="[bold blue]SRT Translator Pro[/bold blue]",
                subtitle=f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                border_style="blue",
                padding=(1, 2),
                box=box.ROUNDED
            ))
            
            try:
                logger.info(f"Starting translation with provider: {config.translation_provider}")
                logger.info(f"Source language: {source_lang}, Target language: {config.target_lang}")
                logger.info(f"Enhancement: {config.enhance}, Model: {config.enhance_model}")
                
                # Read the subtitles
                subtitles = read_srt_file(srt_file)
                subtitle_count = len(subtitles)
                logger.info(f"Read {subtitle_count} subtitles from {srt_file}")
                
                # Get translator
                translator = get_translator(
                    provider=config.translation_provider,
                    source_lang=source_lang,
                    target_lang=config.target_lang,
                    libre_url="https://translate.argosopentech.com/translate" if config.translation_provider == "libre" else None
                )
                
                # Get enhancer if enabled
                enhancer = None
                if config.enhance:
                    from ..enhancers import get_enhancer
                    try:
                        enhancer = get_enhancer(
                            provider="ollama",
                            source_lang=source_lang,
                            target_lang=config.target_lang,
                            model=config.enhance_model,
                            temperature=0.7,
                            host=None  # Use default Ollama host
                        )
                        logger.info(f"Created enhancer with model: {config.enhance_model}")
                    except Exception as e:
                        logger.error(f"Error creating enhancer: {str(e)}")
                        console.print(f"[bold yellow]Warning:[/bold yellow] Could not create enhancer: {str(e)}")
                        console.print("[yellow]Continuing without enhancement...[/yellow]")
                
                # Create progress display
                with create_rich_progress() as progress:
                    # Create a task for the progress bar
                    task_id = progress.add_task(f"[cyan]Translating[/cyan]", total=subtitle_count)
                    
                    # Define a progress callback
                    def progress_callback(current, total):
                        progress.update(task_id, completed=current)
                    
                    # Translate the subtitles
                    start_time = time.time()
                    # Determine appropriate number of workers (use CPU count)
                    import multiprocessing
                    effective_workers = max(1, multiprocessing.cpu_count() - 1)
                    logger.info(f"Using {effective_workers} worker threads for translation")
                    
                    translated_subtitles = translate_subtitles(
                        subtitles=subtitles,
                        translator=translator,
                        enhancer=enhancer,
                        max_workers=effective_workers,
                        batch_size=5,
                        progress_callback=progress_callback
                    )
                    
                    # Calculate translation statistics
                    total_time = time.time() - start_time
                    translator_stats = {
                        "total_time": total_time,
                        "subtitles_per_second": len(subtitles) / total_time if total_time > 0 else 0,
                        "total_subtitles": len(subtitles),
                        "cache_hit_count": 0,  # No cache hits for direct translation
                        "api_char_count": sum(len(s.content) for s in subtitles),
                        "request_count": len(subtitles),
                        "worker_count": effective_workers,
                        "success_count": len(subtitles),  # Assume all translations succeeded
                        "error_count": 0  # Assume no errors
                    }
                    
                    # Write the translated subtitles to file
                    logger.info(f"Writing translated subtitles to: {output_file}")
                    
                    # Ensure subtitles are properly formatted before writing
                    for i, subtitle in enumerate(translated_subtitles):
                        # Ensure index is sequential
                        subtitle.index = i + 1
                        
                        # Fix any encoding issues in content
                        if isinstance(subtitle.content, str):
                            # Replace common encoding issues
                            subtitle.content = subtitle.content.replace('Ã¡', 'á')
                            subtitle.content = subtitle.content.replace('Ã©', 'é')
                            subtitle.content = subtitle.content.replace('Ã­', 'í')
                            subtitle.content = subtitle.content.replace('Ã³', 'ó')
                            subtitle.content = subtitle.content.replace('Ãº', 'ú')
                            subtitle.content = subtitle.content.replace('Ã±', 'ñ')
                    
                    # Write to file
                    write_srt_file(translated_subtitles, output_file)
                    logger.info(f"Translated subtitles written to {output_file}")
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    
                    # Display translation summary
                    stats = get_subtitle_statistics(translated_subtitles)
                    
                    # Create a simplified TranslationConfig for the summary display
                    config_for_display = TranslationConfig(
                        input_file=srt_file,
                        output=output_file,
                        source_lang=source_lang,
                        target_lang=config.target_lang,
                        provider=config.translation_provider,
                        enhance=config.enhance,
                        enhance_model=config.enhance_model
                    )
                    
                    display_translation_summary(
                        stats=stats,
                        config=config_for_display,
                        source_lang=source_lang,
                        target_lang=config.target_lang,
                        translator_stats=translator_stats,
                        total_time=total_time
                    )
                
                # Return success
                return 0
                
            except Exception as e:
                logger.error(f"Error during translation: {str(e)}")
                console.print(Panel(
                    f"[bold red]Error during translation:[/bold red] {str(e)}",
                    title="Translation Failed",
                    border_style="red"
                ))
                if config.verbose:
                    import traceback
                    logger.error(traceback.format_exc())
                    console.print_exception()
                return 1
            
            # Return the translation result
            return translate_result
        
        # Display completion message
        total_time = time.time() - start_time
        completion_text = Text()
        completion_text.append("\n✅ ", style="bold green")
        completion_text.append("Transcription completed successfully!\n\n", style="bold green")
        
        completion_text.append("📄 ", style="bold blue")
        completion_text.append("Output file: ", style="bold cyan")
        completion_text.append(f"{config.output}\n\n", style="white")
        
        # Add stats
        completion_text.append("📊 ", style="bold yellow")
        completion_text.append("Statistics: ", style="bold cyan")
        completion_text.append(f"{stats['count']} subtitles generated in ", style="white")
        completion_text.append(f"{total_time:.2f} seconds", style="bold yellow")
        
        # Print the panel
        console.print(Panel(
            completion_text,
            title="[bold green]Transcription Complete[/bold green]",
            subtitle=f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2)
        ))
        
        return 0
    
    except Exception as e:
        console.print(Panel(
            f"[bold red]Error:[/bold red] {str(e)}",
            title="Transcription Failed",
            border_style="red"
        ))
        if verbose:
            console.print_exception()
        return 1


def main():
    """Main entry point for the CLI"""
    import sys
    
    # Check if we're using the old command format (first arg is a file)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and not sys.argv[1] in ['translate', 'transcribe']:
        # This looks like the old format: srt-translator input.srt [options]
        # Convert to the new format: srt-translator translate input.srt [options]
        input_file = sys.argv[1]
        new_args = ['translate', input_file] + sys.argv[2:]
        sys.argv = [sys.argv[0]] + new_args
        
    # Run the Typer app
    app()