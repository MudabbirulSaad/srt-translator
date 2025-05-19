#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Progress tracking utilities for SRT Translator.
"""

import time
import threading
import sys
from typing import Optional

class ProgressBar:
    """
    A thread-safe progress bar with ETA and elapsed time display.
    """
    def __init__(self, total: int, prefix: str = "", length: int = 50):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of items to process
            prefix: Prefix string for the progress bar
            length: Character length of the progress bar
        """
        self.total = total
        self.prefix = prefix
        self.length = length
        self.start_time = time.time()
        self.current = 0
        self.lock = threading.RLock()
        self.last_update_time = 0
        self.update_interval = 0.1  # Update at most every 0.1 seconds
    
    def update(self, current: Optional[int] = None):
        """
        Update progress bar.
        
        Args:
            current: Current progress value (if None, increment by 1)
        """
        with self.lock:
            # Update current value
            if current is not None:
                self.current = current
            else:
                self.current += 1
            
            # Throttle updates to avoid excessive printing
            now = time.time()
            if now - self.last_update_time < self.update_interval and self.current < self.total:
                return
            
            self.last_update_time = now
            
            # Calculate progress metrics
            percent = self.current / self.total * 100
            filled_length = int(self.length * self.current // self.total)
            bar = '#' * filled_length + '-' * (self.length - filled_length)
            
            # Calculate time metrics
            elapsed = now - self.start_time
            if self.current > 0:
                eta = elapsed * (self.total - self.current) / self.current
            else:
                eta = 0
            
            # Create progress string
            progress_str = f"{self.prefix}: [{bar}] {percent:.1f}% | Elapsed: {self._format_time(elapsed)} | ETA: {self._format_time(eta)}"
            
            # Print progress
            print(f"\r{progress_str}", end='', flush=True)
            
            # Print newline when complete
            if self.current >= self.total:
                print()
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            seconds = seconds % 3600
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{hours}h {minutes}m {seconds:.1f}s"