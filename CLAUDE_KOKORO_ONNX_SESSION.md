# Kokoro ONNX Implementation Session Log

## Overview
This document summarizes our work implementing Kokoro ONNX support for HowdyTTS. We made progress setting up the infrastructure but encountered issues with speech quality.

## What We Accomplished

1. Created and enhanced the download script for Kokoro ONNX models:
   - Implemented robust URL handling with fallbacks
   - Added retry logic for failed downloads
   - Created voice file management with fallbacks

2. Updated the core implementation:
   - Improved voice file loading with format detection
   - Created a default phoneme dictionary
   - Added better error handling throughout

3. Added configuration and integration:
   - Updated config.py to support Kokoro ONNX
   - Modified run_voice_assistant.py to handle ONNX model
   - Created test instructions

## Issues Encountered

The main issue was audio quality - the generated speech was gibberish despite successful model loading. We attempted several approaches:

1. First tried using sophisticated phoneme mappings and token conversions
2. Then simplified to basic character-by-character tokenization
3. Further simplified token-to-id mapping

None of these approaches produced clear speech with the ONNX model.

## Next Steps

For our next session, we could:

1. Try using different models from Hugging Face
2. Adapt the original Kokoro code for tokenization directly
3. Debug the model itself to see if there's a version mismatch
4. Consider building a custom ONNX model from a working Kokoro model

## Resources

- Hugging Face model: `onnx-community/Kokoro-82M-v1.0-ONNX`
- Script: `download_kokoro_onnx.py`
- Integration: `voice_assistant/kokoro_onnx/`
- Test script: `test_kokoro_onnx.py`

## Current Status

We reverted to using the standard Kokoro CLI for now, but the infrastructure for the ONNX implementation remains in place for future improvement.