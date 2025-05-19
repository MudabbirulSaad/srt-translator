#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MyMemory free translation API provider.
"""

import os
import logging
import requests
from .base import TranslationProvider

# Get logger
logger = logging.getLogger("srt_translator")

class MyMemoryTranslator(TranslationProvider):
    """MyMemory free translation API provider"""
    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang)
        # MyMemory API endpoint
        self.api_url = "https://api.mymemory.translated.net/get"
        # Email for MyMemory API (optional, increases daily limit)
        self.email = os.environ.get('MYMEMORY_EMAIL', '')
        # Set appropriate rate limits for MyMemory
        self.min_delay = 0.5  # MyMemory needs more conservative rate limiting
        self.current_delay = 0.8  # Start with a conservative delay
    
    def _translate_implementation(self, text: str) -> str:
        """Translate text using MyMemory API (implementation)"""
        from srt_translator.utils.text import safe_str
        
        # Prepare the API request
        params = {
            'q': text,
            'langpair': f"{self.source_lang}|{self.target_lang}"
        }
        
        # Add email if available
        if self.email:
            params['de'] = self.email
        
        # Make the API request
        response = requests.get(self.api_url, params=params)
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        if data['responseStatus'] == 200:
            translated = data['responseData']['translatedText']
            return translated
        else:
            # Use safe string representation for logging
            preview = safe_str(text[:30]) if len(text) > 30 else safe_str(text)
            logger.warning(f"MyMemory translation failed for '{preview}': {data['responseStatus']}. Returning original text.")
            raise Exception(f"MyMemory API error: {data['responseStatus']}")