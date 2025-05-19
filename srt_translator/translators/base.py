#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base translator class that all specific translator implementations inherit from.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Tuple

# Get logger
logger = logging.getLogger("srt_translator")

class TranslationProvider:
    """Base class for translation providers"""
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        # Thread-safe cache to avoid duplicate translations
        self.cache = {}
        self.cache_lock = threading.RLock()
        # Rate limiting parameters
        self.min_delay = 0.2  # Minimum delay between requests (seconds)
        self.max_delay = 2.0  # Maximum delay between requests (seconds)
        self.current_delay = 0.5  # Current adaptive delay
        self.last_request_time = 0  # Last request timestamp
        self.rate_limit_lock = threading.RLock()
        # Stats
        self.success_count = 0
        self.error_count = 0
        self.cache_hit_count = 0
    
    def wait_for_rate_limit(self):
        """Wait appropriate time to respect rate limits"""
        with self.rate_limit_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            wait_time = max(0, self.current_delay - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_request_time = time.time()
    
    def adjust_rate_limit(self, success: bool):
        """Adjust rate limiting based on success/failure"""
        with self.rate_limit_lock:
            if success:
                self.success_count += 1
                # After several successful requests, try reducing delay slightly
                if self.success_count % 10 == 0 and self.error_count == 0:
                    self.current_delay = max(self.min_delay, self.current_delay * 0.9)
            else:
                self.error_count += 1
                # Increase delay on error
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)
    
    def translate(self, text: str) -> str:
        """Translate text from source language to target language with caching and rate limiting"""
        from srt_translator.utils.text import safe_str
        
        if not text.strip():
            return text
        
        # Check cache first (thread-safe)
        with self.cache_lock:
            if text in self.cache:
                self.cache_hit_count += 1
                # Use safe string representation for logging
                preview = safe_str(text[:30]) if len(text) > 30 else safe_str(text)
                logger.debug(f"Using cached translation for: {preview}...")
                return self.cache[text]
        
        # Wait for rate limiting
        self.wait_for_rate_limit()
        
        try:
            # Perform the actual translation (implemented by subclasses)
            translated = self._translate_implementation(text)
            
            # Record success for rate limiting
            self.adjust_rate_limit(True)
            
            # Cache the result (thread-safe)
            with self.cache_lock:
                self.cache[text] = translated
            
            return translated
        except Exception as e:
            # Record failure for rate limiting
            self.adjust_rate_limit(False)
            
            # Use safe string representation for logging
            preview = safe_str(text[:30]) if len(text) > 30 else safe_str(text)
            logger.warning(f"Translation failed for '{preview}': {e}. Returning original text.")
            return text
    
    def _translate_implementation(self, text: str) -> str:
        """Actual translation implementation to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _translate_implementation method")
    
    def translate_batch(self, texts: List[str], batch_size: int = 5) -> List[str]:
        """Translate a batch of texts with rate limiting"""
        import random
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = []
            for text in batch:
                batch_results.append(self.translate(text))
            results.extend(batch_results)
            if i + batch_size < len(texts):
                # Sleep to avoid rate limiting (random delay between 1-2 seconds)
                time.sleep(1 + random.random())
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get translation statistics"""
        return {
            "success_count": self.success_count,
            "error_count": self.error_count,
            "cache_hit_count": self.cache_hit_count,
            "current_delay": round(self.current_delay, 2)
        }