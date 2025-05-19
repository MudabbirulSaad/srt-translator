#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRT Translator - A CLI tool to translate SRT subtitle files using free translation APIs.
"""

import argparse
import os
import sys
import time
import random
import threading
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import srt
from datetime import timedelta, datetime

# Configure logging with proper encoding handling
log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Create logger
logger = logging.getLogger("srt_translator")
logger.setLevel(logging.INFO)

# Function to safely encode log messages for console output
def safe_str(obj):
    """Convert any object to a string that can be safely printed to console"""
    if isinstance(obj, str):
        # Replace non-ASCII characters with their closest ASCII equivalents or '?'
        return obj.encode('ascii', 'replace').decode('ascii')
    return str(obj)

# Custom StreamHandler that handles Unicode characters safely
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Safely encode the message
            safe_msg = safe_str(msg)
            self.stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Console handler with safe encoding
console_handler = SafeStreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# We'll add file handler later when we know the output path

# Progress bar class
class ProgressBar:
    """A simple progress bar for console output with ETA and elapsed time"""
    
    def __init__(self, total, width=50, title="Progress"):
        self.total = total
        self.width = width
        self.title = title
        self.start_time = time.time()
        self.count = 0
        
    def update(self, count=None):
        """Update the progress bar"""
        if count is not None:
            self.count = count
        else:
            self.count += 1
            
        # Calculate percentage
        percent = self.count / self.total
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        # Calculate ETA
        if percent > 0:
            eta = elapsed / percent - elapsed
            eta_str = self._format_time(eta)
        else:
            eta_str = "N/A"
            
        # Create the progress bar (using ASCII characters for compatibility)
        filled_width = int(self.width * percent)
        bar = '#' * filled_width + '-' * (self.width - filled_width)
        
        # Print the progress bar
        sys.stdout.write(f"\r{self.title}: [{bar}] {percent:.1%} | Elapsed: {elapsed_str} | ETA: {eta_str}")
        sys.stdout.flush()
        
        # Return True if completed
        return self.count >= self.total
    
    def finish(self):
        """Mark the progress as complete"""
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    def _format_time(self, seconds):
        """Format seconds into a human-readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds %= 60
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# Check for required packages and install if missing
def check_and_install_package(package_name, install_spec=None):
    """Check if a package is installed and install it if missing"""
    if install_spec is None:
        install_spec = package_name
    
    try:
        __import__(package_name)
    except ImportError:
        logger.info(f"Installing {package_name} library...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_spec])
            logger.info(f"{package_name} installed successfully")
        except Exception as e:
            logger.error(f"Failed to install {package_name}: {e}")
            logger.error(f"Please install it manually with: pip install {install_spec}")
            sys.exit(1)

# Check and install required packages
check_and_install_package("srt")
check_and_install_package("requests")
check_and_install_package("deep_translator", "deep-translator")
check_and_install_package("langdetect")

# Now import the required modules
import requests
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# Translation providers
class TranslationProvider:
    """Base class for translation providers"""
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        # Thread-safe cache to avoid duplicate translations
        self.cache = {}
        self.cache_lock = threading.RLock()
        # Rate limiting parameters
        self.min_delay = 0.2  # Minimum delay between requests (seconds)
        self.max_delay = 2.0  # Maximum delay between requests (seconds)
        self.current_delay = 0.5  # Current adaptive delay
        self.last_request_time = 0  # Last request timestamp
        self.rate_limit_lock = threading.RLock()
        # Stats
        self.success_count = 0
        self.error_count = 0
        self.cache_hit_count = 0
    
    def wait_for_rate_limit(self):
        """Wait appropriate time to respect rate limits"""
        with self.rate_limit_lock:
            now = time.time()
            elapsed = now - self.last_request_time
            wait_time = max(0, self.current_delay - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_request_time = time.time()
    
    def adjust_rate_limit(self, success: bool):
        """Adjust rate limiting based on success/failure"""
        with self.rate_limit_lock:
            if success:
                self.success_count += 1
                # After several successful requests, try reducing delay slightly
                if self.success_count % 10 == 0 and self.error_count == 0:
                    self.current_delay = max(self.min_delay, self.current_delay * 0.9)
            else:
                self.error_count += 1
                # Increase delay on error
                self.current_delay = min(self.max_delay, self.current_delay * 1.5)
    
    def translate(self, text: str) -> str:
        """Translate text from source language to target language with caching and rate limiting"""
        if not text.strip():
            return text
        
        # Check cache first (thread-safe)
        with self.cache_lock:
            if text in self.cache:
                self.cache_hit_count += 1
                # Use safe string representation for logging
                preview = safe_str(text[:30]) if len(text) > 30 else safe_str(text)
                logger.debug(f"Using cached translation for: {preview}...")
                return self.cache[text]
        
        # Wait for rate limiting
        self.wait_for_rate_limit()
        
        try:
            # Perform the actual translation (implemented by subclasses)
            translated = self._translate_implementation(text)
            
            # Record success for rate limiting
            self.adjust_rate_limit(True)
            
            # Cache the result (thread-safe)
            with self.cache_lock:
                self.cache[text] = translated
            
            return translated
        except Exception as e:
            # Record failure for rate limiting
            self.adjust_rate_limit(False)
            
            # Use safe string representation for logging
            preview = safe_str(text[:30]) if len(text) > 30 else safe_str(text)
            logger.warning(f"Translation failed for '{preview}': {e}. Returning original text.")
            return text
    
    def _translate_implementation(self, text: str) -> str:
        """Actual translation implementation to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _translate_implementation method")
    
    def translate_batch(self, texts: List[str], batch_size: int = 5) -> List[str]:
        """Translate a batch of texts with rate limiting"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = []
            for text in batch:
                batch_results.append(self.translate(text))
            results.extend(batch_results)
            if i + batch_size < len(texts):
                # Sleep to avoid rate limiting (random delay between 1-2 seconds)
                time.sleep(1 + random.random())
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get translation statistics"""
        return {
            "success_count": self.success_count,
            "error_count": self.error_count,
            "cache_hit_count": self.cache_hit_count,
            "current_delay": round(self.current_delay, 2)
        }


class GoogleFreeTranslator(TranslationProvider):
    """Google Translate API provider using direct HTTP requests"""
    def __init__(self, source_lang: str, target_lang: str):
        super().__init__(source_lang, target_lang)
        # Initialize the translator with specified languages
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)
        # Set appropriate rate limits for Google Translate
        self.min_delay = 0.2  # Google can handle faster requests
        self.current_delay = 0.3  # Start with a conservative delay
    
    def _translate_implementation(self, text: str) -> str:
        """Translate text using Google Translate (implementation)"""
        # Use the translator to translate the text
        translated = self.translator.translate(text)
        
        if translated:
            return translated
        else:
            # Use safe string representation for logging
            preview = safe_str(text[:50]) if len(text) > 50 else safe_str(text)
            logger.warning(f"Empty translation result for: {preview}...")
            return text


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


def get_translator(provider: str, source_lang: str, target_lang: str) -> TranslationProvider:
    """Factory function to get the appropriate translator"""
    providers = {
        "google": GoogleFreeTranslator,
        "mymemory": MyMemoryTranslator,
        "libre": LibreTranslator
    }
    
    if provider not in providers:
        logger.error(f"Unknown provider: {provider}. Available providers: {', '.join(providers.keys())}")
        sys.exit(1)
    
    try:
        return providers[provider](source_lang, target_lang)
    except Exception as e:
        logger.error(f"Failed to initialize {provider} translator: {e}")
        logger.info("Falling back to Google Translator")
        return GoogleFreeTranslator(source_lang, target_lang)


def detect_language(text: str, default_lang: str = 'en') -> str:
    """Detect the language of the given text"""
    if not text or len(text.strip()) < 10:
        return default_lang
    
    try:
        # Get a sample of the text (first 1000 characters)
        sample = text[:1000]
        # Detect language
        lang = detect(sample)
        return lang
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}. Using default language: {default_lang}")
        return default_lang

def read_srt_file(file_path: str) -> Tuple[List[srt.Subtitle], str]:
    """Read SRT file and return list of subtitle objects and detected language"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        subtitles = list(srt.parse(content))
        
        # Extract text content for language detection
        text_content = ' '.join([sub.content for sub in subtitles[:20]])  # Use first 20 subtitles
        detected_lang = detect_language(text_content)
        
        return subtitles, detected_lang
    except UnicodeDecodeError:
        # Try with different encodings
        encodings = ['latin-1', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                subtitles = list(srt.parse(content))
                
                # Extract text content for language detection
                text_content = ' '.join([sub.content for sub in subtitles[:20]])  # Use first 20 subtitles
                detected_lang = detect_language(text_content)
                
                return subtitles, detected_lang
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to read {file_path} with any encoding")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading SRT file: {e}")
        sys.exit(1)


def write_srt_file(subtitles: List[srt.Subtitle], output_path: str) -> None:
    """Write subtitles to SRT file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))
        logger.info(f"Translated subtitles written to {output_path}")
    except Exception as e:
        logger.error(f"Error writing SRT file: {e}")
        sys.exit(1)


def translate_subtitles(
    subtitles: List[srt.Subtitle], 
    translator: TranslationProvider,
    max_workers: int = 1,
    batch_size: int = 5
) -> List[srt.Subtitle]:
    """Translate subtitles using the provided translator with adaptive rate limiting and parallel processing"""
    total_subtitles = len(subtitles)
    logger.info(f"Translating {total_subtitles} subtitles...")
    
    # Extract content to translate
    texts = [sub.content for sub in subtitles]
    
    # Create progress bar
    progress = ProgressBar(total_subtitles, title="Translating")
    
    # Translate in parallel with adaptive rate limiting
    translated_texts = [None] * total_subtitles  # Pre-allocate list with correct size
    
    # Start time for statistics
    start_time = time.time()
    
    # Adaptive rate limiting parameters
    class RateLimiter:
        def __init__(self, initial_delay=0.5):
            self.delay = initial_delay
            self.success_count = 0
            self.error_count = 0
            self.last_request_time = 0
            self.lock = threading.Lock()
        
        def wait(self):
            """Wait appropriate time before next request"""
            with self.lock:
                # Calculate time to wait based on last request
                now = time.time()
                elapsed = now - self.last_request_time
                wait_time = max(0, self.delay - elapsed)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Update last request time
                self.last_request_time = time.time()
        
        def success(self):
            """Record successful request and potentially adjust delay"""
            with self.lock:
                self.success_count += 1
                # After 10 successful requests, try reducing delay slightly
                if self.success_count % 10 == 0 and self.error_count == 0:
                    self.delay = max(0.2, self.delay * 0.9)  # Don't go below 0.2s
        
        def error(self):
            """Record error and increase delay"""
            with self.lock:
                self.error_count += 1
                # Increase delay on error
                self.delay = min(2.0, self.delay * 1.5)  # Don't go above 2s
    
    # Create rate limiter
    rate_limiter = RateLimiter()
    
    # Shared progress counter with thread safety
    progress_lock = threading.Lock()
    completed_count = 0
    
    def translate_item(index, text):
        """Translate a single subtitle with rate limiting"""
        nonlocal completed_count
        
        if not text.strip():
            result = text
        else:
            # Check cache first (translator handles this)
            try:
                # Wait appropriate time before request
                rate_limiter.wait()
                
                # Translate the text
                result = translator.translate(text)
                
                # Record successful translation
                rate_limiter.success()
            except Exception as e:
                # Record error and adjust rate limiting
                rate_limiter.error()
                logger.warning(f"Translation error (will retry): {safe_str(str(e))}")
                
                # Wait longer and retry once
                time.sleep(1.0)
                try:
                    rate_limiter.wait()
                    result = translator.translate(text)
                    rate_limiter.success()
                except Exception as e2:
                    logger.error(f"Translation failed after retry: {safe_str(str(e2))}")
                    result = text  # Use original text on failure
        
        # Store result at correct position
        translated_texts[index] = result
        
        # Update progress
        with progress_lock:
            nonlocal completed_count
            completed_count += 1
            progress.update(completed_count)
            
        return result
    
    # Determine optimal number of workers based on subtitle count
    if max_workers <= 0:
        # Auto-determine based on CPU count and subtitle count
        cpu_count = os.cpu_count() or 4
        if total_subtitles < 20:
            actual_workers = 1  # Sequential for very small files
        elif total_subtitles < 100:
            actual_workers = min(4, cpu_count)  # Moderate parallelism for small files
        else:
            actual_workers = min(8, cpu_count)  # Higher parallelism for large files
    else:
        actual_workers = max_workers
    
    logger.info(f"Using {actual_workers} worker threads for translation")
    
    if actual_workers > 1 and total_subtitles > 10:
        # Process in parallel with adaptive chunking
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all translation tasks
            futures = [
                executor.submit(translate_item, i, text)
                for i, text in enumerate(texts)
            ]
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                # Just ensure all futures complete (results already stored in translated_texts)
                try:
                    future.result()  # This will re-raise any exception from the thread
                except Exception as e:
                    logger.error(f"Unexpected error in translation thread: {e}")
    else:
        # Sequential processing for small files or when max_workers=1
        for i, text in enumerate(texts):
            translate_item(i, text)
    
    # Complete the progress bar
    progress.finish()
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    translations_per_second = total_subtitles / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Translation completed in {progress._format_time(elapsed_time)}")
    logger.info(f"Average speed: {translations_per_second:.2f} subtitles/second")
    
    # Create new subtitles with translated content
    translated_subtitles = []
    for i, sub in enumerate(subtitles):
        translated_sub = srt.Subtitle(
            index=sub.index,
            start=sub.start,
            end=sub.end,
            content=translated_texts[i],
            proprietary=sub.proprietary
        )
        translated_subtitles.append(translated_sub)
    
    return translated_subtitles


def main():
    """Main entry point for the SRT translator CLI"""
    parser = argparse.ArgumentParser(
        description="Translate SRT subtitle files using free translation APIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to the input SRT file"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Path to the output SRT file. If not specified, will use input filename with target language code appended"
    )
    
    parser.add_argument(
        "-p", "--provider", 
        choices=["google", "mymemory", "libre"],
        default="google",
        help="Translation provider to use"
    )
    
    parser.add_argument(
        "-s", "--source-lang", 
        default="en",
        help="Source language code (e.g., en, fr, es)"
    )
    
    parser.add_argument(
        "-t", "--target-lang", 
        default="es",
        help="Target language code (e.g., en, fr, es)"
    )
    
    parser.add_argument(
        "-w", "--workers", 
        type=int,
        default=0,
        help="Number of worker threads for parallel translation (0=auto, negative=percentage of CPU cores)"
    )
    
    parser.add_argument(
        "-b", "--batch-size", 
        type=int,
        default=5,
        help="Batch size for translation requests to avoid rate limiting"
    )
    
    parser.add_argument(
        "--adaptive-rate",
        action="store_true",
        help="Enable adaptive rate limiting that adjusts based on API response"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "-l", "--log-file",
        help="Path to log file. If not specified, will create a log file in the same directory as the output file"
    )
    
    args = parser.parse_args()
    
    # Set up log file
    if not args.log_file:
        log_dir = os.path.dirname(os.path.abspath(args.input_file))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(log_dir, f"srt_translator_{timestamp}.log")
    
    # Add file handler to logger with UTF-8 encoding to properly handle all characters
    try:
        file_handler = logging.FileHandler(args.log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
        
        # Set logging level
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
        else:
            file_handler.setLevel(logging.INFO)
    except Exception as e:
        logger.warning(f"Failed to create log file: {e}. Continuing without file logging.")
    
    logger.info("=" * 50)
    logger.info("SRT Translator - Starting translation job")
    logger.info("=" * 50)
    logger.info(f"Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {args.log_file}")
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Set output file if not specified
    if not args.output:
        base, ext = os.path.splitext(args.input_file)
        args.output = f"{base}.{args.target_lang}{ext}"
    
    # Log job parameters
    logger.info(f"Input file: {os.path.abspath(args.input_file)}")
    logger.info(f"Output file: {os.path.abspath(args.output)}")
    logger.info(f"Translation provider: {args.provider}")
    logger.info(f"Source language: {args.source_lang}")
    logger.info(f"Target language: {args.target_lang}")
    logger.info(f"Worker threads: {args.workers}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Get translator
    start_time = time.time()
    logger.info("Initializing translator...")
    translator = get_translator(args.provider, args.source_lang, args.target_lang)
    logger.info(f"Translator initialized: {translator.__class__.__name__}")
    
    # Read input SRT file and detect language
    logger.info(f"Reading SRT file: {args.input_file}")
    subtitles, detected_lang = read_srt_file(args.input_file)
    subtitle_count = len(subtitles)
    logger.info(f"Found {subtitle_count} subtitles")
    
    # Check if detected language matches the source language
    if detected_lang != args.source_lang:
        logger.info(f"Detected language: {detected_lang}, different from specified source language: {args.source_lang}")
        
        # If detected language matches target language, we need to swap
        if detected_lang == args.target_lang:
            logger.warning(f"Detected language ({detected_lang}) is the same as target language. This would result in no translation.")
            logger.info(f"Swapping source and target languages: {args.target_lang} -> {args.source_lang}")
            args.source_lang, args.target_lang = args.target_lang, args.source_lang
            
            # Reinitialize translator with swapped languages
            translator = get_translator(args.provider, args.source_lang, args.target_lang)
        else:
            # Use detected language as source
            logger.info(f"Using detected language ({detected_lang}) as source language instead of {args.source_lang}")
            args.source_lang = detected_lang
            
            # Reinitialize translator with detected source language
            translator = get_translator(args.provider, args.source_lang, args.target_lang)
    
    # Log some statistics about the subtitles
    total_chars = sum(len(sub.content) for sub in subtitles)
    avg_chars = total_chars / subtitle_count if subtitle_count > 0 else 0
    logger.info(f"Total characters: {total_chars}")
    logger.info(f"Average characters per subtitle: {avg_chars:.2f}")
    logger.info(f"Translation direction: {args.source_lang} -> {args.target_lang}")
    
    # Determine optimal worker count if auto mode
    if args.workers <= 0:
        cpu_count = os.cpu_count() or 4
        if args.workers == 0:  # Auto mode
            if subtitle_count < 20:
                effective_workers = 1
            elif subtitle_count < 100:
                effective_workers = min(4, cpu_count)
            else:
                effective_workers = min(8, cpu_count)
        else:  # Negative value = percentage of CPU cores
            percentage = abs(args.workers)
            effective_workers = max(1, int(cpu_count * percentage / 100))
        
        logger.info(f"Auto-selected {effective_workers} worker threads based on system resources and subtitle count")
    else:
        effective_workers = args.workers
    
    # Translate subtitles
    logger.info("Starting translation process...")
    translated_subtitles = translate_subtitles(
        subtitles, 
        translator, 
        max_workers=effective_workers,
        batch_size=args.batch_size
    )
    
    # Write output SRT file
    logger.info(f"Writing translated subtitles to: {args.output}")
    write_srt_file(translated_subtitles, args.output)
    
    # Get translation statistics
    translator_stats = translator.get_stats()
    
    # Log completion statistics
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Translation job completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average processing speed: {subtitle_count / total_time:.2f} subtitles/second")
    logger.info(f"Translation direction: {args.source_lang} -> {args.target_lang}")
    logger.info(f"Worker threads: {effective_workers}")
    logger.info(f"Cache hits: {translator_stats['cache_hit_count']} ({translator_stats['cache_hit_count']/subtitle_count*100:.1f}%)")
    logger.info(f"Successful translations: {translator_stats['success_count']}")
    logger.info(f"Failed translations: {translator_stats['error_count']}")
    logger.info(f"Final rate limit delay: {translator_stats['current_delay']}s")
    logger.info(f"Translation completed: {args.input_file} -> {args.output}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()