#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text processing utilities for SRT Translator.
"""

import re
import logging
from typing import List, Optional, Tuple
from langdetect import detect, LangDetectException

# Get logger
logger = logging.getLogger("srt_translator")

def safe_str(text: str) -> str:
    """Convert text to a safe string representation for logging"""
    if not text:
        return ""
    
    # Replace problematic characters with their Unicode escape sequences
    result = ""
    for char in text:
        if ord(char) < 32 or ord(char) > 126:
            result += f"\\u{ord(char):04x}"
        else:
            result += char
    return result

def detect_language(text: str, min_text_length: int = 100) -> Optional[str]:
    """
    Detect the language of a text using langdetect.
    
    Args:
        text: The text to detect language from
        min_text_length: Minimum text length for reliable detection
        
    Returns:
        ISO 639-1 language code or None if detection failed
    """
    if not text or len(text) < min_text_length:
        logger.warning(f"Text too short for reliable language detection: {len(text) if text else 0} chars")
        return None
    
    try:
        return detect(text)
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        return None

def normalize_language_code(lang_code: str) -> str:
    """
    Normalize language code to ISO 639-1 format.
    
    Args:
        lang_code: Language code to normalize
        
    Returns:
        Normalized ISO 639-1 language code
    """
    # Common language code mappings
    mappings = {
        # ISO 639-2/T to ISO 639-1
        "eng": "en",
        "fra": "fr",
        "deu": "de",
        "spa": "es",
        "ita": "it",
        "jpn": "ja",
        "kor": "ko",
        "zho": "zh",
        "rus": "ru",
        # Common variations
        "zh-cn": "zh",
        "zh-tw": "zh",
        "en-us": "en",
        "en-gb": "en",
        "pt-br": "pt",
        "pt-pt": "pt",
    }
    
    # Convert to lowercase for consistent matching
    lang_code = lang_code.lower()
    
    # Return mapped code if available, otherwise return the original code
    return mappings.get(lang_code, lang_code)

def extract_text_for_language_detection(subtitles: List) -> str:
    """
    Extract text from subtitles for language detection.
    
    Args:
        subtitles: List of SRT subtitle objects
        
    Returns:
        Concatenated text from subtitles
    """
    # Extract content from first 20 subtitles or all if less than 20
    sample_size = min(20, len(subtitles))
    sample_text = " ".join(subtitle.content for subtitle in subtitles[:sample_size])
    
    # Clean the text to improve detection accuracy
    # Remove HTML/XML tags
    sample_text = re.sub(r'<[^>]+>', ' ', sample_text)
    # Remove special characters and numbers
    sample_text = re.sub(r'[^a-zA-Z\u00C0-\u00FF\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\s]', ' ', sample_text)
    # Normalize whitespace
    sample_text = re.sub(r'\s+', ' ', sample_text).strip()
    
    return sample_text