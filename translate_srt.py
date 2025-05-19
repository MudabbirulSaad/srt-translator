#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRT Translator CLI - A simple wrapper script to run the SRT translator
"""

import os
import sys
import argparse
from srt_translator import main as translator_main

def main():
    """Entry point for the CLI wrapper"""
    # Just call the main function from srt_translator
    translator_main()

if __name__ == "__main__":
    main()