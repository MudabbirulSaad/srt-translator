#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for SRT Translator.
"""

from .text import safe_str, detect_language, normalize_language_code, extract_text_for_language_detection
from .progress import ProgressBar
from .logging_utils import setup_logging, UnicodeHandler
from .srt_utils import read_srt_file, write_srt_file, get_subtitle_statistics, generate_output_filename
from .ui import TranslationUI

__all__ = [
    'safe_str',
    'detect_language',
    'normalize_language_code',
    'extract_text_for_language_detection',
    'ProgressBar',
    'setup_logging',
    'UnicodeHandler',
    'read_srt_file',
    'write_srt_file',
    'get_subtitle_statistics',
    'generate_output_filename',
    'TranslationUI'
]