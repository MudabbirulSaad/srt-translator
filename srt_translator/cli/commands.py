#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command implementations for SRT Translator CLI.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..utils import setup_logging, read_srt_file, write_srt_file, get_subtitle_statistics
from ..translators import get_translator
from ..core import translate_subtitles, validate_translation_direction

# Get logger
logger = logging.getLogger("srt_translator")

def main(args: Optional[Dict[str, Any]] = None) -> int:
    """
    Main entry point for SRT Translator CLI.
    
    Args:
        args: Dictionary with parsed arguments (if None, args will be parsed from command line)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments if not provided
    if args is None:
        from .parser import parse_args
        args = parse_args()
    
    # Set up logging
    setup_logging(args.get('log_file'), args.get('verbose', False))
    
    # Record start time
    start_time = time.time()
    
    try:
        # Log job start
        logger.info("=" * 50)
        logger.info("SRT Translator - Starting translation job")
        logger.info("=" * 50)
        logger.info(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Input file: {args['input_file']}")
        logger.info(f"Output file: {args['output']}")
        logger.info(f"Translation provider: {args['provider']}")
        logger.info(f"Source language: {args['source_lang']}")
        logger.info(f"Target language: {args['target_lang']}")
        logger.info(f"Worker threads: {args['workers']}")
        logger.info(f"Batch size: {args['batch_size']}")
        
        # Initialize translator
        logger.info("Initializing translator...")
        translator = get_translator(
            args['provider'],
            args['source_lang'],
            args['target_lang']
        )
        logger.info(f"Translator initialized: {translator.__class__.__name__}")
        
        # Read input SRT file
        subtitles = read_srt_file(args['input_file'])
        
        # Get subtitle statistics
        stats = get_subtitle_statistics(subtitles)
        subtitle_count = stats["count"]
        logger.info(f"Found {subtitle_count} subtitles")
        logger.info(f"Total characters: {stats['total_chars']}")
        logger.info(f"Average characters per subtitle: {stats['avg_chars']:.2f}")
        
        # Validate translation direction
        source_lang, target_lang = validate_translation_direction(
            args['source_lang'],
            args['target_lang'],
            subtitles
        )
        
        # Update translator if language direction changed
        if source_lang != args['source_lang'] or target_lang != args['target_lang']:
            logger.info(f"Updating translation direction: {source_lang} -> {target_lang}")
            translator = get_translator(args['provider'], source_lang, target_lang)
        
        logger.info(f"Translation direction: {source_lang} -> {target_lang}")
        
        # Determine optimal worker count if auto mode
        if args['workers'] <= 0:
            cpu_count = os.cpu_count() or 4
            if args['workers'] == 0:  # Auto mode
                if subtitle_count < 20:
                    effective_workers = 1
                elif subtitle_count < 100:
                    effective_workers = min(4, cpu_count)
                else:
                    effective_workers = min(8, cpu_count)
            else:  # Negative value = percentage of CPU cores
                percentage = abs(args['workers'])
                effective_workers = max(1, int(cpu_count * percentage / 100))
            
            logger.info(f"Auto-selected {effective_workers} worker threads based on system resources and subtitle count")
        else:
            effective_workers = args['workers']
        
        # Translate subtitles
        translated_subtitles = translate_subtitles(
            subtitles, 
            translator, 
            max_workers=effective_workers,
            batch_size=args['batch_size']
        )
        
        # Write output SRT file
        write_srt_file(translated_subtitles, args['output'])
        
        # Get translation statistics
        translator_stats = translator.get_stats()
        
        # Log completion statistics
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"Translation job completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average processing speed: {subtitle_count / total_time:.2f} subtitles/second")
        logger.info(f"Translation direction: {source_lang} -> {target_lang}")
        logger.info(f"Worker threads: {effective_workers}")
        logger.info(f"Cache hits: {translator_stats['cache_hit_count']} ({translator_stats['cache_hit_count']/subtitle_count*100:.1f}%)")
        logger.info(f"Successful translations: {translator_stats['success_count']}")
        logger.info(f"Failed translations: {translator_stats['error_count']}")
        logger.info(f"Final rate limit delay: {translator_stats['current_delay']}s")
        logger.info(f"Translation completed: {args['input_file']} -> {args['output']}")
        logger.info("=" * 50)
        
        return 0
    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        logger.info("=" * 50)
        return 1