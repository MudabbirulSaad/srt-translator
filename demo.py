#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for the enhanced SRT Translator UI.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from srt_translator.utils import TranslationUI, read_srt_file, get_subtitle_statistics
from srt_translator.utils.terminal_compat import enable_unicode_output
from srt_translator.translators import get_translator
from srt_translator.core import translate_subtitles, validate_translation_direction

# Try to enable Unicode output for better terminal compatibility
enable_unicode_output()

def main():
    """Run a demo of the enhanced SRT Translator UI"""
    # Initialize the UI
    ui = TranslationUI()
    
    # Display header
    ui.display_header("SRT Translator Pro")
    
    # Get input file from command line or use a default
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Use a specific file from the workspace
        default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "RM_Japanese_Mom_Eri_Takigawa_1080.srt")
        if os.path.exists(default_file):
            input_file = default_file
        else:
            # Find an SRT file in the current directory
            srt_files = list(Path('.').glob('*.srt'))
            if not srt_files:
                ui.display_error("No SRT files found. Please specify an input file.")
                return 1
            input_file = str(srt_files[0])
    
    # Set translation parameters
    source_lang = "en"
    target_lang = "fr"
    provider = "google"
    output_file = input_file.replace('.srt', f'.{target_lang}.srt')
    
    # Display job information
    ui.display_job_info(
        input_file=input_file,
        output_file=output_file,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=provider
    )
    
    try:
        # Read input SRT file
        with ui.console.status("[bold green]Reading SRT file...[/bold green]"):
            subtitles = read_srt_file(input_file)
        
        # Get subtitle statistics
        stats = get_subtitle_statistics(subtitles)
        
        # Display subtitle statistics
        ui.display_subtitle_stats(stats)
        
        # Display subtitle preview
        ui.display_subtitle_preview(subtitles)
        
        # Detect and validate translation direction
        with ui.console.status("[bold green]Detecting language and validating translation direction...[/bold green]"):
            source_lang, target_lang = validate_translation_direction(
                source_lang,
                target_lang,
                subtitles
            )
        
        # Initialize translator
        with ui.console.status(f"[bold green]Initializing {provider} translator...[/bold green]"):
            translator = get_translator(
                provider,
                source_lang,
                target_lang
            )
        
        # Determine worker count
        worker_count = min(8, os.cpu_count() or 4)
        translator.worker_count = worker_count
        
        ui.console.print(f"Using [bold]{worker_count}[/bold] worker threads for translation")
        ui.console.print(f"Translation direction: [bold cyan]{source_lang}[/bold cyan] → [bold cyan]{target_lang}[/bold cyan]")
        
        # Translate subtitles with progress bar
        ui.console.print("[bold]Starting translation process...[/bold]")
        
        with ui.create_progress_bar() as progress:
            # Create a task for the progress bar
            task_id = progress.add_task(f"[cyan]Translating[/cyan]", total=len(subtitles))
            
            # Define a progress callback
            def progress_callback(current, total):
                progress.update(task_id, completed=current)
            
            # Translate subtitles (only translate a few for demo purposes)
            demo_subtitles = subtitles[:min(10, len(subtitles))]
            translated_subtitles = translate_subtitles(
                demo_subtitles, 
                translator, 
                max_workers=worker_count,
                batch_size=5,
                progress_callback=progress_callback
            )
        
        # Write output SRT file
        with ui.console.status(f"[bold green]Writing translated subtitles to {output_file}...[/bold green]"):
            # For demo purposes, we'll just simulate writing
            time.sleep(1)
            # In a real scenario, you would use:
            # write_srt_file(translated_subtitles, output_file)
        
        # Display translation summary
        ui.display_translation_summary(
            stats=stats,
            source_lang=source_lang,
            target_lang=target_lang,
            translator_stats=translator.get_stats()
        )
        
        # Display completion message
        ui.display_completion(output_file)
        
        return 0
    
    except Exception as e:
        ui.display_error(str(e), verbose=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())