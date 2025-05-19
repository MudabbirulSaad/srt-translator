#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for SRT Translator package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="srt-translator",
    version="1.0.0",
    author="SRT Translator Team",
    author_email="mudabbirulsaad@gmail.com",
    description="A CLI tool to translate SRT subtitle files using free translation APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mudabbirulsaad/srt-translator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "srt>=3.5.0",
        "deep-translator>=1.10.1",
        "requests>=2.25.1",
        "langdetect>=1.0.9",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "colorama>=0.4.6",
        "pydantic>=2.0.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "srt-translator=srt_translator.cli:main",
            "srt-translator-legacy=srt_translator.cli:legacy_main",
        ],
    },
)