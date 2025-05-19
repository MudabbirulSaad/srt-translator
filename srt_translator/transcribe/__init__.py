#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transcription module for SRT Translator.
This module provides functionality to transcribe audio/video files to SRT using Whisper.
"""

from .whisper_transcriber import WhisperTranscriber, transcribe_to_srt

__all__ = ["WhisperTranscriber", "transcribe_to_srt"]