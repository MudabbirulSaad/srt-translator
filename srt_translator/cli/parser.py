#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line argument parser for SRT Translator.
"""

import os
import argparse
import logging
from typing import Dict, List, Any, Optional

# Get logger
logger = logging.getLogger("srt_translator")

def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for SRT Translator.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="SRT Translator - Translate SRT subtitle files using free translation APIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_file",
        help="Input SRT file to translate"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        help="Output SRT file (default: input_file.target_lang.srt)"
    )
    parser.add_argument(
        "-s", "--source-lang",
        default="en",
        help="Source language code (ISO 639-1)"
    )
    parser.add_argument(
        "-t", "--target-lang",
        default="es",
        help="Target language code (ISO 639-1)"
    )
    parser.add_argument(
        "-p", "--provider",
        default="google",
        choices=["google", "mymemory", "libre"],
        help="Translation provider to use"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=0,
        help="Number of worker threads for translation. Use 0 for auto, negative values for percentage of CPU cores"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=5,
        help="Batch size for translation requests"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path (default: auto-generated)"
    )
    parser.add_argument(
        "--libre-url",
        default="https://translate.argosopentech.com/translate",
        help="LibreTranslate API URL (only used with --provider=libre)"
    )
    
    return parser

def parse_args(args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments (if None, sys.argv will be used)
        
    Returns:
        Dictionary with parsed arguments
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Validate input file
    if not os.path.isfile(parsed_args.input_file):
        parser.error(f"Input file not found: {parsed_args.input_file}")
    
    # Generate output file name if not specified
    if not parsed_args.output:
        base_name, ext = os.path.splitext(parsed_args.input_file)
        parsed_args.output = f"{base_name}.{parsed_args.target_lang}{ext}"
    
    return vars(parsed_args)