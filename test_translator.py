#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the refactored SRT Translator package.
"""

import os
import sys
import time
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_package_import():
    """Test importing the refactored package"""
    logger.info("Testing package import...")
    try:
        import srt_translator
        logger.info(f"Successfully imported srt_translator package (version: {srt_translator.__version__})")
        return True
    except ImportError as e:
        logger.error(f"Failed to import srt_translator package: {e}")
        return False

def test_translator_modules():
    """Test importing individual modules"""
    logger.info("Testing module imports...")
    modules = [
        "srt_translator.core",
        "srt_translator.cli",
        "srt_translator.translators",
        "srt_translator.utils"
    ]
    
    success = True
    for module in modules:
        try:
            __import__(module)
            logger.info(f"Successfully imported {module}")
        except ImportError as e:
            logger.error(f"Failed to import {module}: {e}")
            success = False
    
    return success

def test_translation(input_file=None):
    """Test the translation functionality"""
    from srt_translator.utils import read_srt_file
    from srt_translator.translators import get_translator
    from srt_translator.core import translate_subtitles
    
    # Use sample.srt if input_file is not provided
    if input_file is None:
        input_file = "sample.srt"
        # Create a sample SRT file if it doesn't exist
        if not os.path.exists(input_file):
            create_sample_srt(input_file)
    
    logger.info(f"Testing translation with {input_file}...")
    
    try:
        # Read SRT file
        subtitles = read_srt_file(input_file)
        logger.info(f"Read {len(subtitles)} subtitles from {input_file}")
        
        # Initialize translator
        translator = get_translator("google", "en", "fr")
        logger.info(f"Initialized translator: {translator.__class__.__name__}")
        
        # Translate first subtitle only for testing
        start_time = time.time()
        logger.info(f"Translating first subtitle: '{subtitles[0].content}'")
        translated = translator.translate(subtitles[0].content)
        logger.info(f"Translation result: '{translated}'")
        logger.info(f"Translation took {time.time() - start_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        return False

def create_sample_srt(filename):
    """Create a sample SRT file for testing"""
    logger.info(f"Creating sample SRT file: {filename}")
    
    sample_content = """1
00:00:01,000 --> 00:00:04,000
Hello, this is a test subtitle.

2
00:00:05,000 --> 00:00:09,000
This is the second subtitle for testing.

3
00:00:10,000 --> 00:00:14,000
The SRT Translator can translate this to any language.

4
00:00:15,000 --> 00:00:19,000
This is the fourth subtitle line.

5
00:00:20,000 --> 00:00:25,000
And this is the final subtitle for our test.
"""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        logger.info(f"Successfully created sample SRT file: {filename}")
    except Exception as e:
        logger.error(f"Failed to create sample SRT file: {e}")

def main():
    """Run all tests"""
    logger.info("=" * 50)
    logger.info("SRT Translator Package Test")
    logger.info("=" * 50)
    
    tests = [
        ("Package Import", test_package_import),
        ("Module Imports", test_translator_modules),
        ("Translation", test_translation)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\nRunning test: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Test '{name}' raised an exception: {e}")
            results.append((name, False))
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Results Summary")
    logger.info("=" * 50)
    
    all_passed = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{name}: {status}")
        all_passed = all_passed and success
    
    logger.info("=" * 50)
    logger.info(f"Overall Status: {'PASSED' if all_passed else 'FAILED'}")
    logger.info("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())