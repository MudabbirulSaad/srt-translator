#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Translation providers module for SRT Translator.
"""

import logging
from typing import Dict, Type
from .base import TranslationProvider
from .google import GoogleFreeTranslator
from .mymemory import MyMemoryTranslator
from .libre import LibreTranslator

# Get logger
logger = logging.getLogger("srt_translator")

# Registry of available translator providers
TRANSLATOR_REGISTRY: Dict[str, Type[TranslationProvider]] = {
    "google": GoogleFreeTranslator,
    "mymemory": MyMemoryTranslator,
    "libre": LibreTranslator
}

def get_translator(provider: str, source_lang: str, target_lang: str) -> TranslationProvider:
    """Factory function to get the appropriate translator"""
    if provider not in TRANSLATOR_REGISTRY:
        logger.error(f"Unknown provider: {provider}. Available providers: {', '.join(TRANSLATOR_REGISTRY.keys())}")
        raise ValueError(f"Unknown provider: {provider}")
    
    try:
        return TRANSLATOR_REGISTRY[provider](source_lang, target_lang)
    except ValueError as e:
        logger.error(str(e))
        logger.info("Falling back to Google Translator which doesn't require an API key")
        return GoogleFreeTranslator(source_lang, target_lang)
    except Exception as e:
        logger.error(f"Error initializing {provider} translator: {e}")
        logger.info("Falling back to Google Translator")
        return GoogleFreeTranslator(source_lang, target_lang)