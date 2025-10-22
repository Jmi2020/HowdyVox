#!/usr/bin/env python3
"""
Voice Blending Script for Kokoro ONNX

This script demonstrates blending different voice styles in Kokoro ONNX TTS.
It creates audio with a blend of multiple voices with adjustable ratios.
"""

import os
import argparse
import soundfile as sf
import numpy as np
from kokoro_onnx import Kokoro

# Define all supported voices based on the SupportedVoices.txt file
SUPPORTED_VOICES = {
    # Female English US voices
    'af_alloy': 'English US Female',
    'af_aoede': 'English US Female',
    'af_bella': 'English US Female',
    'af_heart': 'English US Female',
    'af_jessica': 'English US Female',
    'af_kore': 'English US Female',
    'af_nicole': 'English US Female',
    'af_nova': 'English US Female',
    'af_river': 'English US Female',
    'af_sarah': 'English US Female',
    'af_sky': 'English US Female',
    
    # Male English US voices
    'am_adam': 'English US Male',
    'am_echo': 'English US Male',
    'am_eric': 'English US Male',
    'am_fenrir': 'English US Male',
    'am_liam': 'English US Male',
    'am_michael': 'English US Male (Cowboy)',
    'am_onyx': 'English US Male',
    'am_puck': 'English US Male',
    
    # Other voices
    'bf_alice': 'Other Female',
    'bf_emma': 'Other Female',
    'bf_isabella': 'Other Female',
    'bf_lily': 'Other Female',
    'bm_daniel': 'Other Male',
    'bm_fable': 'Other Male',
    'bm_george': 'Other Male',
    'bm_lewis': 'Other Male'
}

# Group voices by category for easier selection
VOICE_CATEGORIES = {
    'female': [v for v in SUPPORTED_VOICES.keys() if v.startswith('af_')],
    'male': [v for v in SUPPORTED_VOICES.keys() if v.startswith('am_')],
    'other_female': [v for v in SUPPORTED_VOICES.keys() if v.startswith('bf_')],
    'other_male': [v for v in SUPPORTED_VOICES.keys() if v.startswith('bm_')]
}

def blend_voices(text, voice_ratios, output_file="blended_audio.wav", speed=1.0):
    """
    Blend multiple voices and generate audio from text.
    
    Args:
        text (str): Text to convert to speech
        voice_ratios (dict): Dictionary mapping voice IDs to percentages
        output_file (str): Output audio file path
        speed (float): Speed factor for speech
    
    Returns:
        str: Path to the generated audio file
    """
    # Validate voice_ratios
    if not voice_ratios:
        raise ValueError("No voices specified for blending")
    
    # Check if all specified voices are supported
    unsupported = [v for v in voice_ratios.keys() if v not in SUPPORTED_VOICES]
    if unsupported:
        raise ValueError(f"Unsupported voices: {', '.join(unsupported)}")
    
    # Ensure percentages sum to 100
    total = sum(voice_ratios.values())
    if total != 100:
        print(f"Warning: Voice ratios sum to {total}%. Normalizing to 100%.")
        voice_ratios = {k: (v / total) * 100 for k, v in voice_ratios.items()}
    
    # Set correct paths to model files
    model_path = os.path.join("models", "kokoro-v1.0.onnx")
    voices_path = os.path.join("models", "voices-v1.0.bin")
    
    # Get Kokoro instance directly
    print(f"Loading Kokoro model from {model_path} and voices from {voices_path}")
    kokoro = Kokoro(model_path, voices_path)
    
    # Create voice blend
    blend = None
    for voice_id, percentage in voice_ratios.items():
        print(f"Adding {voice_id} ({SUPPORTED_VOICES[voice_id]}) at {percentage:.1f}%")
        voice_vector = kokoro.get_voice_style(voice_id)
        
        if blend is None:
            blend = voice_vector * (percentage / 100)
        else:
            blend = np.add(blend, voice_vector * (percentage / 100))
    
    # Generate audio with the blended voice
    print(f"Generating audio with {len(voice_ratios)} voice blend")
    samples, sample_rate = kokoro.create(
        text,
        voice=blend,
        speed=speed,
        lang="en-us",
    )
    
    # Save the audio file
    sf.write(output_file, samples, sample_rate)
    print(f"Created {output_file}")
    
    return output_file

def list_available_voices():
    """List all available voice styles in the Kokoro model"""
    # Use direct model paths
    model_path = os.path.join("models", "kokoro-v1.0.onnx")
    voices_path = os.path.join("models", "voices-v1.0.bin")
    
    try:
        # Create Kokoro instance directly with correct paths
        print(f"Loading Kokoro model from {model_path} and voices from {voices_path}")
        kokoro = Kokoro(model_path, voices_path)
        
        # Get available voices from the model
        model_voices = []
        if hasattr(kokoro, 'get_available_voices'):
            model_voices = kokoro.get_available_voices()
        elif hasattr(kokoro, '_voices') and isinstance(kokoro._voices, dict):
            model_voices = list(kokoro._voices.keys())
        
        if model_voices:
            print(f"Available voices in model: {', '.join(model_voices)}")
        
        # Print supported voices by category
        print("\nSupported Voices by Category:")
        print("\nEnglish US Female Voices:")
        for voice in VOICE_CATEGORIES['female']:
            print(f"  {voice} - {SUPPORTED_VOICES[voice]}")
        
        print("\nEnglish US Male Voices:")
        for voice in VOICE_CATEGORIES['male']:
            print(f"  {voice} - {SUPPORTED_VOICES[voice]}")
        
        print("\nOther Female Voices:")
        for voice in VOICE_CATEGORIES['other_female']:
            print(f"  {voice} - {SUPPORTED_VOICES[voice]}")
        
        print("\nOther Male Voices:")
        for voice in VOICE_CATEGORIES['other_male']:
            print(f"  {voice} - {SUPPORTED_VOICES[voice]}")
        
    except Exception as e:
        print(f"Error listing voices: {e}")

def parse_voice_ratios(voice_ratio_str):
    """
    Parse voice ratio string in the format "voice1:ratio1,voice2:ratio2,..."
    
    Args:
        voice_ratio_str (str): Voice ratio string, e.g., "af_nicole:30,am_michael:70"
        
    Returns:
        dict: Dictionary mapping voice IDs to percentages
    """
    voice_ratios = {}
    
    if not voice_ratio_str:
        return voice_ratios
        
    try:
        parts = voice_ratio_str.split(',')
        for part in parts:
            voice, ratio = part.split(':')
            voice_ratios[voice.strip()] = float(ratio.strip())
    except ValueError:
        raise ValueError("Invalid voice ratio format. Use 'voice1:ratio1,voice2:ratio2,...'")
    
    return voice_ratios

def main():
    parser = argparse.ArgumentParser(description="Blend voices using Kokoro ONNX TTS")
    parser.add_argument("--text", type=str, default="Howdy partner! This audio uses a custom blend of voices.", 
                        help="Text to convert to speech")
    parser.add_argument("--voices", type=str, 
                        help="Voice blend specification in format 'voice1:ratio1,voice2:ratio2,...' (e.g., 'af_nicole:30,am_michael:70')")
    parser.add_argument("--nicole", type=float, default=None, help="Percentage of Nicole's voice (0-100)")
    parser.add_argument("--michael", type=float, default=None, help="Percentage of Michael's voice (0-100)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed factor (default: 1.0)")
    parser.add_argument("--output", type=str, default="blended_audio.wav", help="Output audio file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    
    args = parser.parse_args()
    
    # Make sure the models directory exists
    if not os.path.exists("models"):
        print("Error: The 'models' directory doesn't exist.")
        print("Make sure you're running this script from the project root directory.")
        return
        
    # Check if model files exist
    if not os.path.exists(os.path.join("models", "kokoro-v1.0.onnx")):
        print("Error: Model file 'kokoro-v1.0.onnx' not found in models directory.")
        print("Download it with: wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
        return
        
    if not os.path.exists(os.path.join("models", "voices-v1.0.bin")):
        print("Error: Voices file 'voices-v1.0.bin' not found in models directory.")
        print("Download it with: wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
        return
    
    # List available voices if requested
    if args.list_voices:
        list_available_voices()
        return
    
    # Determine voice ratios
    voice_ratios = {}
    if args.voices:
        # Parse voice ratios from the --voices argument
        voice_ratios = parse_voice_ratios(args.voices)
    elif args.nicole is not None and args.michael is not None:
        # Use the original nicole/michael blend
        voice_ratios = {'af_nicole': args.nicole, 'am_michael': args.michael}
    else:
        # Default to the original 30/70 Nicole/Michael blend
        voice_ratios = {'af_nicole': 30, 'am_michael': 70}
    
    # Generate blended audio
    try:
        output_path = blend_voices(args.text, voice_ratios, args.output, args.speed)
        print(f"Done! Audio saved to: {output_path}")
    except Exception as e:
        print(f"Error generating blended audio: {e}")

if __name__ == "__main__":
    main()