#!/usr/bin/env python3
"""
Voice Blending Test Script

This script provides an easy way to test different voice blending combinations
and compare the results using all available voices.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from blend_voices import SUPPORTED_VOICES, VOICE_CATEGORIES

def test_basic_blend(text, output_dir="test_blends", voice1="af_nicole", voice2="am_michael", 
                   steps=5, speed=1.0):
    """
    Generate test audio files with different blending ratios between two voices.
    
    Args:
        text (str): Text to convert to speech
        output_dir (str): Directory to save output files
        voice1 (str): First voice ID
        voice2 (str): Second voice ID
        steps (int): Number of blend steps (excluding pure voices)
        speed (float): Speech speed factor
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Kokoro with direct paths to model files
    model_path = os.path.join("models", "kokoro-v1.0.onnx")
    voices_path = os.path.join("models", "voices-v1.0.bin")
    
    # Check if model files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at: {model_path}")
        return False
    
    if not os.path.exists(voices_path):
        print(f"Error: Voices file not found at: {voices_path}")
        return False
    
    # Make sure the requested voices are supported
    if voice1 not in SUPPORTED_VOICES:
        print(f"Error: Voice '{voice1}' not found in supported voices")
        return False
    
    if voice2 not in SUPPORTED_VOICES:
        print(f"Error: Voice '{voice2}' not found in supported voices")
        return False
    
    print(f"Loading Kokoro model from {model_path} and voices from {voices_path}")
    kokoro = Kokoro(model_path, voices_path)
    
    # Generate pure voice samples
    print(f"Generating pure voice samples for {voice1} and {voice2}...")
    
    # Pure voice1
    try:
        samples, sample_rate = kokoro.create(
            text,
            voice=voice1,
            speed=speed,
            lang="en-us",
        )
        voice1_file = os.path.join(output_dir, f"{voice1}_100.wav")
        sf.write(voice1_file, samples, sample_rate)
        print(f"Created {voice1_file}")
    except Exception as e:
        print(f"Error creating pure {voice1} sample: {e}")
        return False
    
    # Pure voice2
    try:
        samples, sample_rate = kokoro.create(
            text,
            voice=voice2,
            speed=speed,
            lang="en-us",
        )
        voice2_file = os.path.join(output_dir, f"{voice2}_100.wav")
        sf.write(voice2_file, samples, sample_rate)
        print(f"Created {voice2_file}")
    except Exception as e:
        print(f"Error creating pure {voice2} sample: {e}")
        return False
    
    # Get voice vectors
    voice1_vector = kokoro.get_voice_style(voice1)
    voice2_vector = kokoro.get_voice_style(voice2)
    
    # Generate blended voices
    print(f"\nGenerating {steps} blended voice samples between {voice1} and {voice2}...")
    
    # Create blend ratios (excluding 0% and 100% as we already did those)
    ratios = [i / steps for i in range(1, steps)]
    
    for ratio in ratios:
        voice1_pct = int(ratio * 100)
        voice2_pct = 100 - voice1_pct
        
        # Create blend
        blend = np.add(
            voice1_vector * (voice1_pct / 100),
            voice2_vector * (voice2_pct / 100)
        )
        
        # Generate audio with this blend
        output_file = os.path.join(output_dir, f"blend_{voice1}_{voice1_pct}_{voice2}_{voice2_pct}.wav")
        
        try:
            samples, sample_rate = kokoro.create(
                text,
                voice=blend,
                speed=speed,
                lang="en-us",
            )
            sf.write(output_file, samples, sample_rate)
            print(f"Created {output_file} ({voice1}: {voice1_pct}%, {voice2}: {voice2_pct}%)")
        except Exception as e:
            print(f"Error creating blend with {voice1}: {voice1_pct}%, {voice2}: {voice2_pct}%: {e}")
    
    print(f"\nAll test files generated in '{output_dir}' directory")
    print("Listen to the files to compare different blending ratios")
    
    return True

def test_multi_voice_blend(text, output_dir="test_blends", voice_groups=None, speed=1.0):
    """
    Generate interesting multi-voice blends that combine more than two voices.
    
    Args:
        text (str): Text to convert to speech
        output_dir (str): Directory to save output files
        voice_groups (list): List of dictionaries with voice combinations to test
        speed (float): Speech speed factor
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default interesting voice combinations to test if none provided
    if voice_groups is None:
        voice_groups = [
            {
                'name': 'multi_female',
                'blend': {'af_nicole': 40, 'af_bella': 30, 'af_jessica': 30}
            },
            {
                'name': 'multi_male',
                'blend': {'am_michael': 40, 'am_adam': 30, 'am_eric': 30}
            },
            {
                'name': 'balanced_quartet',
                'blend': {'af_nicole': 25, 'af_bella': 25, 'am_michael': 25, 'am_adam': 25}
            },
            {
                'name': 'mostly_michael',
                'blend': {'am_michael': 70, 'af_nicole': 15, 'am_adam': 15}
            },
            {
                'name': 'diverse_blend',
                'blend': {'am_michael': 40, 'af_nicole': 30, 'bf_alice': 15, 'bm_daniel': 15}
            }
        ]
    
    # Initialize Kokoro
    model_path = os.path.join("models", "kokoro-v1.0.onnx")
    voices_path = os.path.join("models", "voices-v1.0.bin")
    
    print(f"Loading Kokoro model from {model_path} and voices from {voices_path}")
    kokoro = Kokoro(model_path, voices_path)
    
    print(f"Testing {len(voice_groups)} multi-voice blend combinations...")
    
    # Generate each multi-voice blend
    for group in voice_groups:
        name = group['name']
        voice_ratios = group['blend']
        
        print(f"\nGenerating {name} blend with {len(voice_ratios)} voices:")
        for voice, ratio in voice_ratios.items():
            print(f"  {voice}: {ratio}%")
            
        # Create the blend
        blend = None
        for voice_id, percentage in voice_ratios.items():
            voice_vector = kokoro.get_voice_style(voice_id)
            
            if blend is None:
                blend = voice_vector * (percentage / 100)
            else:
                blend = np.add(blend, voice_vector * (percentage / 100))
        
        # Generate audio
        output_file = os.path.join(output_dir, f"blend_{name}.wav")
        
        try:
            samples, sample_rate = kokoro.create(
                text,
                voice=blend,
                speed=speed,
                lang="en-us",
            )
            sf.write(output_file, samples, sample_rate)
            print(f"Created {output_file}")
        except Exception as e:
            print(f"Error creating {name} blend: {e}")
    
    print(f"\nAll multi-voice blends generated in '{output_dir}' directory")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test different voice blending combinations with Kokoro ONNX TTS")
    parser.add_argument("--text", type=str, 
                        default="Howdy partner! I'm testing different voice blending combinations. What do you think of this voice?",
                        help="Text to convert to speech")
    parser.add_argument("--output-dir", type=str, default="test_blends",
                        help="Directory to save output files")
    parser.add_argument("--voice1", type=str, default="af_nicole",
                        help="First voice to blend")
    parser.add_argument("--voice2", type=str, default="am_michael",
                        help="Second voice to blend")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of blend steps between the two voices")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed factor (default: 1.0)")
    parser.add_argument("--multi-voice", action="store_true",
                        help="Test multi-voice blends with more than two voices")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        print("Supported Voices by Category:")
        
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
        
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test basic two-voice blending
    test_basic_blend(
        args.text, 
        args.output_dir, 
        args.voice1, 
        args.voice2, 
        args.steps,
        args.speed
    )
    
    # Test multi-voice blending if requested
    if args.multi_voice:
        test_multi_voice_blend(args.text, args.output_dir, speed=args.speed)

if __name__ == "__main__":
    main()