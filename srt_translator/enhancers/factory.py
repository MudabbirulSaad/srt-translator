"""
Enhancement Provider Factory

This module provides a factory function to get the appropriate enhancement provider.
"""

import logging
from typing import Dict, Any, Optional

from .base import EnhancementProvider
from .ollama import OllamaEnhancer, OLLAMA_AVAILABLE


logger = logging.getLogger("srt_translator")


def get_enhancer(
    provider: str,
    source_lang: str,
    target_lang: str,
    **kwargs
) -> EnhancementProvider:
    """
    Get an enhancement provider instance.
    
    Args:
        provider: Enhancement provider name ('ollama')
        source_lang: Source language code (ISO 639-1)
        target_lang: Target language code (ISO 639-1)
        **kwargs: Additional provider-specific parameters
        
    Returns:
        EnhancementProvider instance
        
    Raises:
        ValueError: If the provider is not supported or required dependencies are missing
    """
    providers = {
        "ollama": (OllamaEnhancer, OLLAMA_AVAILABLE),
    }
    
    if provider not in providers:
        supported = ", ".join(providers.keys())
        raise ValueError(f"Unsupported enhancement provider: {provider}. Supported providers: {supported}")
    
    enhancer_class, is_available = providers[provider]
    
    if not is_available:
        raise ValueError(
            f"Enhancement provider '{provider}' is not available. "
            f"Please install the required dependencies."
        )
    
    try:
        return enhancer_class(source_lang, target_lang, **kwargs)
    except Exception as e:
        logger.error(f"Error creating enhancement provider '{provider}': {e}")
        raise ValueError(f"Error creating enhancement provider '{provider}': {e}")


def list_available_enhancers() -> Dict[str, bool]:
    """
    List all available enhancement providers.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    return {
        "ollama": OLLAMA_AVAILABLE,
    }