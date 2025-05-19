#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whisper-based transcription for audio/video files.
Provides functionality to generate subtitles from audio/video using OpenAI's Whisper model.
"""

import os
import logging
import tempfile
import subprocess
from typing import Optional, Dict, Any, List, Tuple, Union
import srt
from datetime import timedelta
from pathlib import Path
import json

try:
    import whisper
    import ffmpeg
    import numpy as np
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Get logger
logger = logging.getLogger("srt_translator")

class WhisperTranscriber:
    """
    Transcriber class using OpenAI's Whisper model for audio/video transcription.
    
    This class provides functionality to transcribe audio/video files to SRT format
    using OpenAI's Whisper model. It supports various model sizes and can auto-detect
    the language or use a specified language.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, 
                 compute_type: str = "float16", download_root: Optional[str] = None):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, turbo)
            device: Device to use for inference (cpu, cuda). If None, automatically selects the best available device.
            compute_type: Compute type for model inference (float16, float32, int8)
            download_root: Directory to download and cache models
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper package is not installed. Please install it with 'pip install openai-whisper ffmpeg-python'"
            )
        
        self.model_size = model_size
        
        # Auto-select device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.compute_type = compute_type
        self.download_root = download_root
        self.model = None
        
        logger.info(f"Initializing Whisper transcriber (model: {model_size}, device: {self.device}, compute_type: {compute_type})")
        
        # Load the model immediately
        self.load_model()
    
    def load_model(self):
        """
        Load the Whisper model.
        """
        logger.info(f"Loading Whisper model: {self.model_size}")
        try:
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device,
                download_root=self.download_root,
                in_memory=True
            )
            logger.info(f"Whisper model loaded successfully: {self.model_size} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _extract_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio from a video/audio file.
        
        Args:
            file_path: Path to the audio/video file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        logger.info(f"Extracting audio from: {file_path}")
        try:
            # First try using whisper's load_audio
            audio = whisper.load_audio(file_path)
            sample_rate = 16000  # Whisper uses 16kHz
            logger.info(f"Audio extracted successfully using whisper.load_audio: {len(audio)/sample_rate:.2f}s")
            return audio, sample_rate
        except Exception as e:
            logger.warning(f"Failed to extract audio using whisper.load_audio: {e}")
            
            # Fallback to ffmpeg-python
            try:
                logger.info("Falling back to ffmpeg-python for audio extraction")
                out, _ = (
                    ffmpeg.input(file_path)
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=16000)
                    .run(capture_stdout=True, capture_stderr=True)
                )
                audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
                sample_rate = 16000
                logger.info(f"Audio extracted successfully using ffmpeg-python: {len(audio)/sample_rate:.2f}s")
                return audio, sample_rate
            except Exception as e2:
                logger.error(f"Failed to extract audio using ffmpeg-python: {e2}")
                
                # Last resort: use subprocess to call ffmpeg directly
                logger.info("Falling back to subprocess ffmpeg for audio extraction")
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.close()
                
                try:
                    subprocess.check_call([
                        'ffmpeg', '-y', '-i', file_path, 
                        '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', 
                        temp_file.name
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Read the audio file
                    audio = whisper.load_audio(temp_file.name)
                    sample_rate = 16000
                    
                    # Clean up
                    os.unlink(temp_file.name)
                    
                    logger.info(f"Audio extracted successfully using subprocess ffmpeg: {len(audio)/sample_rate:.2f}s")
                    return audio, sample_rate
                except Exception as e3:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
                    logger.error(f"All audio extraction methods failed. Last error: {e3}")
                    raise RuntimeError(f"Failed to extract audio from {file_path}: {e3}")
    
    def transcribe_file(self, file_path: str, output_path: Optional[str] = None, 
                        language: Optional[str] = None, task: str = "transcribe",
                        verbose: bool = False, word_timestamps: bool = True,
                        highlight_words: bool = False) -> str:
        """
        Transcribe an audio/video file to SRT format.
        
        Args:
            file_path: Path to the audio/video file
            output_path: Path to save the SRT file (if None, uses input filename with .srt extension)
            language: Language code for transcription (if None, auto-detect)
            task: Task to perform ('transcribe' or 'translate' to English)
            verbose: Whether to print detailed progress information
            word_timestamps: Whether to include word-level timestamps
            highlight_words: Whether to highlight words in the output
            
        Returns:
            Path to the generated SRT file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate output path if not provided
        if output_path is None:
            base_name, _ = os.path.splitext(file_path)
            output_path = f"{base_name}.srt"
        
        logger.info(f"Transcribing file: {file_path}")
        logger.info(f"Output SRT file: {output_path}")
        logger.info(f"Task: {task}, Language: {language or 'auto-detect'}")
        
        # Extract audio from the file
        audio, _ = self._extract_audio(file_path)
        
        # Transcribe the audio
        logger.info("Starting transcription with Whisper...")
        transcription_options = {
            "task": task,
            "verbose": verbose,
            "word_timestamps": word_timestamps,
        }
        
        if language:
            transcription_options["language"] = language
            
        # Perform the transcription
        result = self.model.transcribe(audio, **transcription_options)
        
        # Convert the result to SRT format
        logger.info("Converting transcription to SRT format...")
        subtitles = self._result_to_srt(result, highlight_words)
        
        # Write the SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))
        
        logger.info(f"Transcription completed: {output_path}")
        
        # Return information about the transcription
        detected_language = result.get("language", "unknown")
        logger.info(f"Detected language: {detected_language}")
        
        return output_path
    
    def _result_to_srt(self, result: Dict[str, Any], highlight_words: bool = False) -> List[srt.Subtitle]:
        """
        Convert Whisper transcription result to SRT subtitles.
        
        Args:
            result: Whisper transcription result
            highlight_words: Whether to highlight words in the output
            
        Returns:
            List of SRT subtitles
        """
        subtitles = []
        
        # Check if we have segments in the result
        if "segments" not in result:
            logger.warning("No segments found in transcription result")
            return subtitles
        
        # Process each segment
        for i, segment in enumerate(result["segments"]):
            # Get start and end times
            start_time = segment["start"]
            end_time = segment["end"]
            
            # Get the text
            text = segment["text"].strip()
            
            # Create subtitle
            subtitle = srt.Subtitle(
                index=i+1,
                start=timedelta(seconds=start_time),
                end=timedelta(seconds=end_time),
                content=text
            )
            
            subtitles.append(subtitle)
        
        return subtitles
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available Whisper models.
        
        Returns:
            List of available model names
        """
        return ["tiny", "base", "small", "medium", "large", "turbo"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "dimensions": str(self.model.dims) if hasattr(self.model, "dims") else "unknown"
        }


def transcribe_to_srt(file_path: str, output_path: Optional[str] = None, 
                     model_size: str = "base", language: Optional[str] = None,
                     device: Optional[str] = None, task: str = "transcribe",
                     verbose: bool = False, word_timestamps: bool = True,
                     highlight_words: bool = False) -> str:
    """
    Convenience function to transcribe an audio/video file to SRT format.
    
    Args:
        file_path: Path to the audio/video file
        output_path: Path to save the SRT file (if None, uses input filename with .srt extension)
        model_size: Whisper model size (tiny, base, small, medium, large, turbo)
        language: Language code for transcription (if None, auto-detect)
        device: Device to use for inference (cpu, cuda, auto)
        task: Task to perform ('transcribe' or 'translate' to English)
        verbose: Whether to print detailed progress information
        word_timestamps: Whether to include word-level timestamps
        highlight_words: Whether to highlight words in the output
        
    Returns:
        Path to the generated SRT file
    """
    transcriber = WhisperTranscriber(model_size=model_size, device=device)
    return transcriber.transcribe_file(
        file_path, 
        output_path, 
        language=language,
        task=task,
        verbose=verbose,
        word_timestamps=word_timestamps,
        highlight_words=highlight_words
    )