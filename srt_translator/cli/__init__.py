#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for SRT Translator.
"""

from .parser import create_parser, parse_args
from .commands import main as legacy_main
from .typer_cli import main as enhanced_main

# Use the enhanced CLI by default
main = enhanced_main

__all__ = ['create_parser', 'parse_args', 'main', 'legacy_main', 'enhanced_main']