#!/bin/bash
# HowdyVox Launcher for M3 Mac (Apple Silicon)
# This script sets the necessary environment variables for Opus library support

# Set library paths for Opus (required for wireless audio)
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH

# Activate conda environment and run the launcher
/opt/anaconda3/bin/conda run -n howdy310 python launch_howdy_shell.py
