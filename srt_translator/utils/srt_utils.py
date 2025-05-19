#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRT file processing utilities for SRT Translator.
"""

import os
import srt
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

# Get logger
logger = logging.getLogger("srt_translator")

def read_srt_file(file_path: str) -> List[srt.Subtitle]:
    """
    Read SRT file and parse subtitles.
    
    Args:
        file_path: Path to SRT file
        
    Returns:
        List of SRT subtitle objects
    """
    logger.info(f"Reading SRT file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encodings if UTF-8 fails
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                logger.info(f"Trying to read file with {encoding} encoding")
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"Failed to read SRT file with any encoding: {file_path}")
            raise
    
    # Parse subtitles
    try:
        subtitles = list(srt.parse(content))
        logger.info(f"Found {len(subtitles)} subtitles")
        return subtitles
    except Exception as e:
        logger.error(f"Failed to parse SRT file: {e}")
        raise

def write_srt_file(subtitles: List[srt.Subtitle], file_path: str) -> None:
    """
    Write subtitles to SRT file with proper encoding for special characters.
    
    Args:
        subtitles: List of SRT subtitle objects
        file_path: Path to output SRT file
    """
    logger.info(f"Writing translated subtitles to: {file_path}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Fix encoding issues in subtitle content
        fixed_subtitles = []
        for subtitle in subtitles:
            # Create a copy of the subtitle to avoid modifying the original
            fixed_subtitle = srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content=subtitle.content,
                proprietary=subtitle.proprietary
            )
            
            # Fix common encoding issues for Spanish characters
            if isinstance(fixed_subtitle.content, str):
                # Map of problematic characters to their correct form
                char_map = {
                    'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
                    'Ã±': 'ñ', 'Ã¼': 'ü', 'Ã': 'í', 'Â': '',
                    'Ã\x81': 'Á', 'Ã\x89': 'É', 'Ã\x8d': 'Í', 'Ã\x93': 'Ó', 'Ã\x9a': 'Ú',
                    'Ã\x91': 'Ñ'
                }
                
                # Apply all character replacements
                for bad_char, good_char in char_map.items():
                    fixed_subtitle.content = fixed_subtitle.content.replace(bad_char, good_char)
            
            fixed_subtitles.append(fixed_subtitle)
        
        # Write subtitles to file with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(srt.compose(fixed_subtitles))
        
        logger.info(f"Translated subtitles written to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write SRT file: {e}")
        raise

def get_subtitle_statistics(subtitles: List[srt.Subtitle]) -> Dict[str, Any]:
    """
    Get statistics about subtitles.
    
    Args:
        subtitles: List of SRT subtitle objects
        
    Returns:
        Dictionary with subtitle statistics
    """
    if not subtitles:
        return {
            "count": 0,
            "total_chars": 0,
            "avg_chars": 0,
            "duration": timedelta(0)
        }
    
    # Calculate statistics
    subtitle_count = len(subtitles)
    total_chars = sum(len(subtitle.content) for subtitle in subtitles)
    avg_chars = total_chars / subtitle_count if subtitle_count > 0 else 0
    
    # Calculate total duration
    duration = subtitles[-1].end - subtitles[0].start
    
    return {
        "count": subtitle_count,
        "total_chars": total_chars,
        "avg_chars": avg_chars,
        "duration": duration
    }

def generate_output_filename(input_file: str, target_lang: str) -> str:
    """
    Generate output filename based on input file and target language.
    
    Args:
        input_file: Input SRT file path
        target_lang: Target language code
        
    Returns:
        Output SRT file path
    """
    base_name, ext = os.path.splitext(input_file)
    return f"{base_name}.{target_lang}{ext}"