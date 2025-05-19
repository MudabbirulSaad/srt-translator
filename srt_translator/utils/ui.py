#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI utilities for SRT Translator.
"""

from typing import Dict, List, Optional, Any
import os
import time
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.box import ROUNDED, HEAVY_EDGE, ASCII
from rich.syntax import Syntax
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from .terminal_compat import (
    is_unicode_supported, get_safe_arrow, get_safe_bullet, 
    get_safe_box_style, enable_unicode_output
)

# Language names mapping for better display
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "fa": "Persian",
    "he": "Hebrew",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "sk": "Slovak",
    "uk": "Ukrainian",
    "el": "Greek",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "sl": "Slovenian",
    "sr": "Serbian",
}

class TranslationUI:
    """Rich UI for SRT Translator"""
    
    def __init__(self):
        """Initialize the UI"""
        # Try to enable Unicode output
        enable_unicode_output()
        
        # Create console with appropriate settings for terminal compatibility
        self.console = Console(safe_box=not is_unicode_supported())
        self.start_time = time.time()
        
        # Set box styles based on terminal compatibility
        self.box_style = get_safe_box_style()
        self.heavy_box_style = ASCII if not is_unicode_supported() else HEAVY_EDGE
        self.arrow = get_safe_arrow()
        self.bullet = get_safe_bullet()
    
    def get_language_name(self, lang_code: str) -> str:
        """Get the full language name from a language code"""
        return LANGUAGE_NAMES.get(lang_code.lower(), lang_code)
    
    def display_header(self, title: str = "SRT Translator"):
        """Display a header with the application title"""
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="blue",
            box=self.heavy_box_style
        ))
        self.console.print()
    
    def display_job_info(self, input_file: str, output_file: str, 
                        source_lang: str, target_lang: str,
                        provider: str):
        """Display job information in a panel"""
        source_lang_name = self.get_language_name(source_lang)
        target_lang_name = self.get_language_name(target_lang)
        
        job_info = (
            f"[bold]Input:[/bold] {input_file}\n"
            f"[bold]Output:[/bold] {output_file}\n"
            f"[bold]Translation:[/bold] {source_lang_name} ({source_lang}) {self.arrow} {target_lang_name} ({target_lang})\n"
            f"[bold]Provider:[/bold] {provider.capitalize()}"
        )
        
        self.console.print(Panel(
            job_info,
            title="[bold]Job Information[/bold]",
            border_style="blue",
            box=self.box_style
        ))
    
    def display_subtitle_stats(self, stats: Dict[str, Any]):
        """Display subtitle statistics in a panel"""
        duration_str = str(stats["duration"]).split(".")[0] if "duration" in stats else "N/A"
        
        stats_info = (
            f"[bold]Count:[/bold] {stats['count']} subtitles\n"
            f"[bold]Characters:[/bold] {stats['total_chars']} total ({stats['avg_chars']:.1f} avg per subtitle)\n"
            f"[bold]Duration:[/bold] {duration_str}"
        )
        
        self.console.print(Panel(
            stats_info,
            title="[bold]Subtitle Statistics[/bold]",
            border_style="green",
            box=self.box_style
        ))
    
    def display_subtitle_preview(self, subtitles, count=3):
        """Display a preview of the subtitles in a table"""
        if not subtitles:
            return
        
        table = Table(title="Subtitle Preview", box=self.box_style)
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("Timing", style="green")
        table.add_column("Content", style="white")
        
        # Show first few subtitles
        for i, subtitle in enumerate(subtitles[:count]):
            table.add_row(
                str(subtitle.index),
                f"{subtitle.start} {self.arrow} {subtitle.end}",
                subtitle.content
            )
        
        # If there are more subtitles, add an ellipsis row
        if len(subtitles) > count:
            table.add_row("...", "...", "...")
        
        self.console.print(table)
    
    def create_progress_bar(self) -> Progress:
        """Create a Rich progress bar"""
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TextColumn(self.bullet),
            TimeElapsedColumn(),
            TextColumn(self.bullet),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
    
    def display_translation_summary(self, stats: Dict[str, Any], 
                                   source_lang: str, target_lang: str, 
                                   translator_stats: Dict[str, Any]):
        """Display a summary of the translation job"""
        source_lang_name = self.get_language_name(source_lang)
        target_lang_name = self.get_language_name(target_lang)
        
        # Calculate total time
        total_time = time.time() - self.start_time
        
        table = Table(title="Translation Summary", box=self.box_style)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Add rows with statistics
        table.add_row("Translation Direction", f"{source_lang_name} ({source_lang}) {self.arrow} {target_lang_name} ({target_lang})")
        table.add_row("Subtitles Processed", str(stats["count"]))
        table.add_row("Worker Threads", str(translator_stats.get("worker_count", "N/A")))
        table.add_row("Processing Time", f"{total_time:.2f} seconds")
        table.add_row("Processing Speed", f"{stats['count'] / total_time:.2f} subtitles/second")
        table.add_row("Cache Hits", f"{translator_stats['cache_hit_count']} ({translator_stats['cache_hit_count']/stats['count']*100:.1f}%)")
        table.add_row("Successful Translations", str(translator_stats['success_count']))
        table.add_row("Failed Translations", str(translator_stats['error_count']))
        table.add_row("Final Rate Limit Delay", f"{translator_stats['current_delay']}s")
        
        self.console.print(table)
    
    def display_completion(self, output_file: str):
        """Display completion message"""
        elapsed_time = time.time() - self.start_time
        elapsed_str = self.format_time(elapsed_time)
        
        self.console.print()
        self.console.print(Panel.fit(
            f"[bold green]Translation completed successfully![/bold green]\n"
            f"[bold]Output file:[/bold] {output_file}\n"
            f"[bold]Total time:[/bold] {elapsed_str}",
            title="Job Complete",
            subtitle=f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="green",
            box=self.heavy_box_style
        ))
        self.console.print()
    
    def display_error(self, error_message: str, verbose: bool = False):
        """Display an error message"""
        self.console.print()
        self.console.print(Panel(
            f"[bold red]Error:[/bold red] {error_message}",
            title="Translation Failed",
            border_style="red",
            box=self.heavy_box_style
        ))
        
        if verbose:
            self.console.print_exception()
        
        self.console.print()
    
    def format_time(self, seconds: float) -> str:
        """Format time in seconds to a readable string"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes} minutes {seconds:.1f} seconds"
        else:
            hours = int(seconds // 3600)
            seconds = seconds % 3600
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{hours} hours {minutes} minutes {seconds:.1f} seconds"