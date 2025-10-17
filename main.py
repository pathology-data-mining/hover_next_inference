#!/usr/bin/env python3
"""
Main entry point for HoVer-NeXt inference.
This file provides a convenient way to run the inference pipeline.
"""

import sys
import os

# Add the src directory to the path to allow importing the inference module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference.__main__ import main

if __name__ == "__main__":
    main()
