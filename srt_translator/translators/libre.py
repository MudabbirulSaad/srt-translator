#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LibreTranslate API provider - completely free and open source.
"""

import os
import logging
import requests
from .base import TranslationProvider

# Get logger
logger = logging.getLogger("srt_translator")

class LibreTranslator(TranslationProvider):
    """LibreTranslate API provider - completely free and open source"""
    def __init__(self, source_lang: str, target_lang: str, api_url: str = "https://translate.argosopentech.com/translate"):
        super().__init__(source_lang, target_lang)
        self.api_url = api_url
        # Set appropriate rate limits for LibreTranslate
        self.min_delay = 0.5  # LibreTranslate needs more conservative rate limiting
        self.current_delay = 0.8  # Start with a conservative delay
    
    def _translate_implementation(self, text: str) -> str:
        """Translate text using LibreTranslate API (implementation)"""
        from srt_translator.utils.text import safe_str
        
        # Prepare the API request
        payload = {
            "q": text,
            "source": self.source_lang,
            "target": self.target_lang,
            "format": "text",
            "api_key": os.environ.get('LIBRETRANSLATE_API_KEY', '')
        }
        
        # Make the API request
        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        translated = data.get('translatedText', '')
        
        if translated:
            return translated
        else:
            # Use safe string representation for logging
            preview = safe_str(text[:30]) if len(text) > 30 else safe_str(text)
            logger.warning(f"LibreTranslate returned empty result for '{preview}'. Returning original text.")
            raise Exception("LibreTranslate returned empty result")