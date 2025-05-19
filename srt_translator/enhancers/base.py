"""
Base Enhancement Provider

This module defines the base class for subtitle enhancement providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import srt


class EnhancementProvider(ABC):
    """
    Base class for subtitle enhancement providers.
    
    Enhancement providers take translated subtitles and improve them using AI models
    to ensure better context, natural language flow, and cultural appropriateness.
    """
    
    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        """
        Initialize the enhancement provider.
        
        Args:
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)
            **kwargs: Additional provider-specific parameters
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.options = kwargs
    
    @abstractmethod
    def enhance_subtitle(self, subtitle: srt.Subtitle, context: Optional[List[srt.Subtitle]] = None) -> srt.Subtitle:
        """
        Enhance a single subtitle using AI.
        
        Args:
            subtitle: The subtitle to enhance
            context: Optional list of surrounding subtitles for context
            
        Returns:
            Enhanced subtitle
        """
        pass
    
    def enhance_batch(self, subtitles: List[srt.Subtitle], batch_size: int = 5, progress_callback=None) -> List[srt.Subtitle]:
        """
        Enhance a batch of subtitles.
        
        This method can be overridden by providers that support batch processing.
        The default implementation processes subtitles one by one with context.
        
        Args:
            subtitles: List of subtitles to enhance
            batch_size: Number of subtitles to process at once
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of enhanced subtitles
        """
        enhanced_subtitles = []
        total = len(subtitles)
        
        for i, subtitle in enumerate(subtitles):
            # Get context (previous and next subtitles)
            start_idx = max(0, i - 2)
            end_idx = min(len(subtitles), i + 3)
            context = subtitles[start_idx:i] + subtitles[i+1:end_idx]
            
            # Enhance subtitle with context
            enhanced_subtitle = self.enhance_subtitle(subtitle, context)
            enhanced_subtitles.append(enhanced_subtitle)
            
            # Update progress if callback is provided
            if progress_callback:
                progress_callback(i + 1, total)
        
        return enhanced_subtitles
    
    def get_provider_name(self) -> str:
        """
        Get the name of the enhancement provider.
        
        Returns:
            Provider name
        """
        return self.__class__.__name__.replace("Enhancer", "")
    
    @abstractmethod
    def get_options(self) -> Dict[str, Any]:
        """
        Get the provider-specific options.
        
        Returns:
            Dictionary of provider options
        """
        pass