#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core translation functionality for SRT Translator.
"""

import time
import logging
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import srt

from .translators.base import TranslationProvider
from .enhancers.base import EnhancementProvider
from .utils.progress import ProgressBar
from .utils.text import detect_language, normalize_language_code, extract_text_for_language_detection

# Get logger
logger = logging.getLogger("srt_translator")

def translate_subtitles(
    subtitles: List[srt.Subtitle], 
    translator: TranslationProvider, 
    max_workers: int = 5,
    batch_size: int = 5,
    progress_callback=None,
    enhancer: Optional[EnhancementProvider] = None
) -> List[srt.Subtitle]:
    """
    Translate subtitles using multithreading.
    
    Args:
        subtitles: List of SRT subtitle objects
        translator: Translation provider
        max_workers: Maximum number of worker threads
        batch_size: Batch size for translation
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of translated SRT subtitle objects
    """
    if not subtitles:
        logger.warning("No subtitles to translate")
        return []
    
    # Log translation parameters
    logger.info(f"Translating {len(subtitles)} subtitles...")
    logger.info(f"Using {max_workers} worker threads for translation")
    
    # Create a copy of the subtitles to avoid modifying the original
    translated_subtitles = [srt.Subtitle(
        index=s.index,
        start=s.start,
        end=s.end,
        content=s.content,
        proprietary=s.proprietary
    ) for s in subtitles]
    
    # Initialize progress tracking
    total_subtitles = len(subtitles)
    completed_count = 0
    progress_lock = threading.RLock()
    
    # Initialize progress bar if no callback is provided
    progress_bar = None
    if progress_callback is None:
        from .utils.progress import ProgressBar
        progress_bar = ProgressBar(total_subtitles, prefix="Translating")
    
    # Start timing
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel translation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list to store futures
        future_to_index = {}
        
        # Submit translation tasks
        for i, subtitle in enumerate(subtitles):
            future = executor.submit(translator.translate, subtitle.content)
            future_to_index[future] = i
        
        # Process completed translations
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                translated_text = future.result()
                translated_subtitles[index].content = translated_text
            except Exception as e:
                logger.error(f"Translation failed for subtitle {index+1}: {e}")
                # Keep original content if translation fails
            
            # Update progress
            with progress_lock:
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_subtitles)
                elif progress_bar:
                    progress_bar.update(completed_count)
    
    # Calculate and log translation statistics
    elapsed_time = time.time() - start_time
    logger.info(f"Translation completed in {elapsed_time:.1f}s")
    logger.info(f"Average speed: {len(subtitles) / elapsed_time:.2f} subtitles/second")
    
    # Apply AI enhancement if enhancer is provided
    if enhancer:
        logger.info(f"Enhancing translations using {enhancer.get_provider_name()} provider...")
        
        # Reset progress tracking for enhancement
        completed_count = 0
        
        # Create a new progress callback specifically for enhancement
        def enhancement_progress_callback(current, total):
            if progress_callback:
                # Make sure we don't exceed 100%
                current_value = min(current, total)
                progress_callback(current_value, total)
        
        # Start timing for enhancement
        enhance_start_time = time.time()
        
        # Enhance the translated subtitles
        enhanced_subtitles = enhancer.enhance_batch(
            subtitles=translated_subtitles,
            batch_size=batch_size,
            progress_callback=enhancement_progress_callback
        )
        
        # Replace translated subtitles with enhanced ones
        translated_subtitles = enhanced_subtitles
        
        # Calculate and log enhancement statistics
        enhance_elapsed_time = time.time() - enhance_start_time
        logger.info(f"Enhancement completed in {enhance_elapsed_time:.1f}s")
        logger.info(f"Average enhancement speed: {len(subtitles) / enhance_elapsed_time:.2f} subtitles/second")
    
    return translated_subtitles

def detect_subtitle_language(subtitles: List[srt.Subtitle]) -> Optional[str]:
    """
    Detect the language of subtitles.
    
    Args:
        subtitles: List of SRT subtitle objects
        
    Returns:
        Detected language code or None if detection failed
    """
    if not subtitles:
        logger.warning("No subtitles to detect language from")
        return None
    
    # Extract text for language detection
    sample_text = extract_text_for_language_detection(subtitles)
    
    # Detect language
    detected_lang = detect_language(sample_text)
    
    if detected_lang:
        logger.info(f"Detected subtitle language: {detected_lang}")
        return normalize_language_code(detected_lang)
    else:
        logger.warning("Failed to detect subtitle language")
        return None

def validate_translation_direction(
    source_lang: str, 
    target_lang: str, 
    subtitles: List[srt.Subtitle]
) -> Tuple[str, str]:
    """
    Validate and potentially correct the translation direction.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        subtitles: List of SRT subtitle objects
        
    Returns:
        Tuple of (validated_source_lang, validated_target_lang)
    """
    # Normalize language codes
    source_lang = normalize_language_code(source_lang)
    target_lang = normalize_language_code(target_lang)
    
    # Detect subtitle language
    detected_lang = detect_subtitle_language(subtitles)
    
    if detected_lang:
        # If detected language matches target language, swap direction
        if detected_lang == target_lang:
            logger.warning(f"Detected language ({detected_lang}) matches target language. Swapping translation direction.")
            return target_lang, source_lang
        
        # If detected language doesn't match source language, update source
        if detected_lang != source_lang:
            logger.warning(f"Detected language ({detected_lang}) doesn't match specified source language ({source_lang}). Using detected language.")
            return detected_lang, target_lang
    
    # Return original direction if no changes needed
    return source_lang, target_lang