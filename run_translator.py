#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script for SRT Translator.
This script checks for dependencies and runs the translator.
"""

import os
import sys
import importlib.util
import subprocess
from typing import Dict, List

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'srt': 'srt>=3.5.0',
        'deep-translator': 'deep-translator>=1.10.1',
        'requests': 'requests>=2.25.1',
        'langdetect': 'langdetect>=1.0.9'
    }
    
    missing_packages = []
    
    for package, install_spec in required_packages.items():
        if importlib.util.find_spec(package) is None:
            missing_packages.append(install_spec)
    
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
    else:
        print("All dependencies are already installed.")


def main():
    """Main entry point for the runner script"""
    # Check and install dependencies
    check_and_install_dependencies()
    
    try:
        # Import the translator module now that dependencies are installed
        from srt_translator.cli import main as translator_main
        
        # Run the translator
        sys.exit(translator_main())
    except ImportError:
        # If the package is not installed, try to run the original script
        print("SRT Translator package not found. Trying to run the script directly...")
        try:
            # Check if the original script exists
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "srt_translator.py")
            if os.path.exists(script_path):
                print(f"Running original script: {script_path}")
                # Run the original script
                from importlib.machinery import SourceFileLoader
                srt_translator = SourceFileLoader("srt_translator", script_path).load_module()
                srt_translator.main()
            else:
                print("Error: SRT Translator script not found.")
                sys.exit(1)
        except Exception as e:
            print(f"Error running SRT Translator: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()