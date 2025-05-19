#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Translate API provider using deep-translator library.
"""

import logging
from .base import TranslationProvider

# Get logger
logger = logging.getLogger("srt_translator")

class GoogleFreeTranslator(TranslationProvider):
    """Google Translate API provider using deep-translator library"""
    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang)
        try:
            from deep_translator import GoogleTranslator
            # Initialize the translator with specified languages
            self.translator = GoogleTranslator(source=source_lang, target=target_lang)
            # Set appropriate rate limits for Google Translate
            self.min_delay = 0.2  # Google can handle faster requests
            self.current_delay = 0.3  # Start with a conservative delay
        except ImportError:
            logger.error("deep-translator library not found. Install with: pip install deep-translator")
            raise
    
    def _translate_implementation(self, text: str) -> str:
        """Translate text using Google Translate (implementation)"""
        from srt_translator.utils.text import safe_str
        
        # Use the translator to translate the text
        translated = self.translator.translate(text)
        
        if translated:
            return translated
        else:
            # Use safe string representation for logging
            preview = safe_str(text[:50]) if len(text) > 50 else safe_str(text)
            logger.warning(f"Empty translation result for: {preview}...")
            return text