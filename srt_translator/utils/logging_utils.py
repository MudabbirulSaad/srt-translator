#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for SRT Translator.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Optional

class UnicodeHandler(logging.StreamHandler):
    """Custom logging handler that properly handles Unicode characters"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Replace problematic characters with their Unicode escape sequences
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, a default name will be generated)
        verbose: Whether to enable verbose logging
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("srt_translator")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create console handler with Unicode support
    console_handler = UnicodeHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified or generate a default log file
    if log_file is None:
        # Get the project root directory
        project_root = get_project_root()
        
        # Create logs directory path
        logs_dir = os.path.join(project_root, "logs")
        
        # Generate timestamp for log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"srt_translator_{timestamp}.log")
    
    try:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")
    except Exception as e:
        logger.error(f"Failed to set up file logging: {e}")
    
    return logger

def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Absolute path to the project root directory
    """
    # Start with the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to get to the project root (from utils/ to srt_translator/)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    return project_root