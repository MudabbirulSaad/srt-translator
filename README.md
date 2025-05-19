# SRT Translator

A powerful and flexible command-line tool for translating SRT subtitle files using various free translation APIs, featuring a beautiful modern terminal UI.

## Features

- **Multiple Translation Providers**: Support for Google Translate, MyMemory, and LibreTranslate APIs
- **Automatic Language Detection**: Detects the source language of subtitles and adjusts translation direction if needed
- **Multithreaded Translation**: Parallel processing for faster translation with adaptive rate limiting
- **Beautiful Terminal UI**: Rich, colorful interface with tables, panels, and progress indicators
- **Progress Tracking**: Real-time progress bar with ETA and elapsed time
- **Comprehensive Logging**: Detailed logs with translation statistics and error handling
- **Caching**: Avoids redundant translations of identical subtitle content
- **Unicode Support**: Properly handles international characters and encodings
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux with automatic terminal capability detection

## Installation

### Option 1: Install from source

```bash
git clone https://github.com/mudabbirulsaad/srt-translator.git
cd srt-translator
pip install -e .
```

### Option 2: Run without installation

```bash
git clone https://github.com/mudabbirulsaad/srt-translator.git
cd srt-translator
python run_translator.py [options]
```

## Usage

### Basic Usage

```bash
srt-translator input.srt -t fr
```

This will translate `input.srt` from English (default source language) to French and save the result as `input.fr.srt`.

### Enhanced UI Demo

To see the enhanced UI in action, you can run the demo script:

```bash
python demo.py your_subtitle_file.srt
```

### Command-line Options

```
Usage: srt-translator [OPTIONS] INPUT_FILE

  Translate SRT subtitle files using free translation APIs.

  Examples:

  - Translate from English to Spanish using Google Translate:
    $ srt-translator input.srt -t es

  - Translate from Japanese to English using MyMemory with 8 worker threads:
    $ srt-translator japanese.srt -s ja -t en -p mymemory -w 8

  - Use automatic thread count selection (based on CPU cores and subtitle count):
    $ srt-translator large_movie.srt -t de -w 0

Arguments:
  INPUT_FILE  Input SRT file to translate  [required]

Options:
  -o, --output TEXT                 Output SRT file (default:
                                    input_file.target_lang.srt)
  -s, --source-lang TEXT            Source language code (ISO 639-1)  [default:
                                    en]
  -t, --target-lang TEXT            Target language code (ISO 639-1)  [default:
                                    es]
  -p, --provider [google|mymemory|libre]
                                    Translation provider to use  [default:
                                    google]
  -w, --workers INTEGER             Number of worker threads for translation.
                                    Use 0 for auto, negative values for
                                    percentage of CPU cores  [default: 0]
  -b, --batch-size INTEGER          Batch size for translation requests
                                    [default: 5]
  -v, --verbose                     Enable verbose logging  [default: False]
  --log-file TEXT                   Log file path (default: auto-generated)
  --libre-url TEXT                  LibreTranslate API URL (only used with
                                    --provider=libre)  [default:
                                    https://translate.argosopentech.com/translate]
  --preview                         Preview subtitles without translating
  -h, --help                        Show this message and exit.
```

### Examples

#### Translate from Japanese to English using Google Translate

```bash
srt-translator japanese_movie.srt -s ja -t en
```

#### Translate using MyMemory with 8 worker threads

```bash
srt-translator movie.srt -t fr -p mymemory -w 8
```

#### Translate using LibreTranslate with a custom API URL

```bash
srt-translator movie.srt -t es -p libre --libre-url http://localhost:5000/translate
```

#### Use automatic thread count selection (based on CPU cores and subtitle count)

```bash
srt-translator large_movie.srt -t de -w 0
```

#### Use 50% of available CPU cores

```bash
srt-translator large_movie.srt -t it -w -50
```

## Environment Variables

The following environment variables can be used to configure API keys:

- `MYMEMORY_EMAIL`: Email for MyMemory API (increases daily limit)
- `LIBRETRANSLATE_API_KEY`: API key for LibreTranslate (if required)

## Project Structure

```
srt-translator/
├── srt_translator/              # Main package
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point when run as a module
│   ├── core.py                  # Core translation functionality
│   ├── cli/                     # Command-line interface
│   │   ├── __init__.py
│   │   ├── commands.py          # Legacy CLI command implementations
│   │   ├── parser.py            # Command-line argument parsing
│   │   └── typer_cli.py         # Enhanced CLI using Typer and Rich
│   ├── translators/             # Translation providers
│   │   ├── __init__.py          # Provider factory
│   │   ├── base.py              # Base translator class
│   │   ├── google.py            # Google Translate implementation
│   │   ├── libre.py             # LibreTranslate implementation
│   │   └── mymemory.py          # MyMemory implementation
│   ├── transcribe/              # Audio/video transcription (future)
│   │   ├── __init__.py
│   │   └── whisper_transcriber.py # Whisper integration placeholder
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── logging_utils.py     # Logging setup
│       ├── progress.py          # Progress tracking
│       ├── srt_utils.py         # SRT file operations
│       ├── terminal_compat.py   # Terminal compatibility utilities
│       ├── text.py              # Text processing utilities
│       └── ui.py                # Rich UI components
├── demo.py                      # Demo script for enhanced UI
├── run_translator.py            # Runner script with dependency checking
├── setup.py                     # Package installation script
├── test_translator.py           # Test script for the package
└── README.md                    # This file
```

## Future Extensions

The modular architecture makes it easy to add new features:

1. **Audio/Video Transcription**: The groundwork is already laid for integration with Whisper or similar APIs to transcribe audio/video files directly into SRT format. The `transcribe` module contains placeholder code that can be completed with a full implementation.

2. **New Translation Providers**: Add new provider classes in the `translators` directory by implementing the base `TranslationProvider` class. The system is designed to make it easy to add new translation services.

3. **GUI Interface**: The Rich UI components can be adapted to create a full graphical interface using frameworks like PyQt, Tkinter, or a web interface with Flask/FastAPI.

4. **Batch Processing**: Add support for translating multiple SRT files in batch with parallel processing for maximum efficiency.

5. **Translation Memory**: Implement a persistent translation memory database for improved reuse across translation jobs.

6. **Custom Fine-tuned Models**: Add support for using custom fine-tuned translation models for specific domains or language pairs.

7. **Cloud Integration**: Add support for cloud storage services like Google Drive, Dropbox, or S3 for input/output files.

## Dependencies

- **Core**: srt, deep-translator, requests, langdetect
- **Enhanced UI**: rich, typer, colorama, pydantic, tqdm

## License

MIT License