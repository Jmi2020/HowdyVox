#!/usr/bin/env python3
"""
Test script for KokoroOnnx TTS implementation
"""

import os
import argparse
import sys
from pathlib import Path

try:
    from voice_assistant.kokoro_onnx.integration import create_onnx_tts
    from voice_assistant.kokoro_onnx.converter import convert_kokoro_to_onnx
except ImportError as e:
    print(f"Error importing KokoroOnnx: {e}")
    print("Make sure you have installed all required dependencies:")
    print("  pip install onnxruntime numpy soundfile")
    sys.exit(1)

def test_conversion(source_model_path, output_dir=None, voice_name="am_michael"):
    """Test converting a KokoroTTS model to ONNX format"""
    print(f"Converting KokoroTTS model from {source_model_path} to ONNX format...")
    
    onnx_model_path = convert_kokoro_to_onnx(
        source_model_path=source_model_path,
        output_dir=output_dir,
        voice_name=voice_name
    )
    
    if onnx_model_path:
        print(f"Conversion successful! ONNX model saved to: {onnx_model_path}")
        return onnx_model_path
    else:
        print("Conversion failed.")
        return None

def test_tts(model_path=None, text=None, output_path=None, voice_name="am_michael"):
    """Test the KokoroOnnx TTS functionality"""
    if text is None:
        text = "Howdy partner! This is a test of the KokoroTTS ONNX implementation."
    
    if output_path is None:
        output_path = "test_output.wav"
    
    print(f"Testing KokoroOnnx TTS with voice: {voice_name}")
    print(f"Text: {text}")
    print(f"Output path: {output_path}")
    
    # Create the TTS integration
    tts_engine = create_onnx_tts(voice=voice_name, model_path=model_path)
    
    # Test the TTS functionality
    tts_engine.test_tts(text)
    
    print("Test complete.")

def main():
    parser = argparse.ArgumentParser(description="Test KokoroOnnx TTS implementation")
    parser.add_argument("--convert", help="Path to source KokoroTTS model for conversion to ONNX")
    parser.add_argument("--output", help="Output directory for converted model")
    parser.add_argument("--model", help="Path to ONNX model directory")
    parser.add_argument("--voice", default="am_michael", help="Voice name (default: am_michael)")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--tts-output", help="Output path for TTS audio")
    
    args = parser.parse_args()
    
    # Check for required ONNX dependencies
    try:
        import onnxruntime
        print(f"ONNXRuntime version: {onnxruntime.__version__}")
    except ImportError:
        print("ONNXRuntime not installed. Please install it with: pip install onnxruntime")
        sys.exit(1)
    
    # If convert option is specified, run the conversion
    if args.convert:
        onnx_model_path = test_conversion(
            source_model_path=args.convert,
            output_dir=args.output,
            voice_name=args.voice
        )
        
        # If conversion was successful and no explicit model path was provided for TTS,
        # use the newly converted model
        if onnx_model_path and not args.model:
            args.model = onnx_model_path
    
    # Run TTS test with specified or default parameters
    test_tts(
        model_path=args.model,
        text=args.text,
        output_path=args.tts_output,
        voice_name=args.voice
    )

if __name__ == "__main__":
    main()