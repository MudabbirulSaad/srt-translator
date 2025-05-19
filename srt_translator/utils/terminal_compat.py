#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Terminal compatibility utilities for SRT Translator.
Ensures proper display across different terminal environments.
"""

import os
import sys
import platform

def is_windows():
    """Check if running on Windows"""
    return os.name == 'nt' or platform.system() == 'Windows'

def is_unicode_supported():
    """
    Check if the terminal supports Unicode characters.
    This is a best-effort detection that may not be 100% accurate.
    """
    if is_windows():
        # Check if running in Windows Terminal which has better Unicode support
        if os.environ.get('WT_SESSION'):
            return True
        
        # Check if PYTHONIOENCODING is set to UTF-8
        if os.environ.get('PYTHONIOENCODING', '').lower() == 'utf-8':
            return True
        
        # Check console encoding
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            return kernel32.GetConsoleOutputCP() == 65001  # 65001 is the code page for UTF-8
        except (AttributeError, ImportError):
            pass
        
        # Default to False on Windows as many terminals have issues with Unicode
        return False
    
    # Most Unix terminals support Unicode
    return True

def get_safe_arrow():
    """Get a terminal-safe arrow symbol"""
    return "->" if not is_unicode_supported() else "→"

def get_safe_bullet():
    """Get a terminal-safe bullet point symbol"""
    return "*" if not is_unicode_supported() else "•"

def get_safe_box_style():
    """Get a terminal-safe box style for Rich"""
    from rich import box
    return box.ASCII if not is_unicode_supported() else box.ROUNDED

def enable_unicode_output():
    """
    Try to enable Unicode output in the terminal.
    This is a best-effort attempt that may not work in all environments.
    """
    if is_windows():
        # Try to set console code page to UTF-8
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)  # 65001 is the code page for UTF-8
        except (AttributeError, ImportError):
            pass
        
        # Set PYTHONIOENCODING environment variable
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Set stdout and stderr encoding to UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')