#!/usr/bin/env python3
"""
Blended Voice Configuration Tool

This tool allows you to configure HowdyTTS to use a blended voice
by creating a custom voice profile and patching the Kokoro TTS system.
"""

import os
import argparse
import logging
import numpy as np
from kokoro_onnx import Kokoro
from blend_voices import SUPPORTED_VOICES, VOICE_CATEGORIES, parse_voice_ratios

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def patch_kokoro_for_blended_voice(profile_name, voice_ratios):
    """
    Patch the Kokoro TTS system to use a blended voice as default.
    
    Args:
        profile_name (str): Name to give this voice profile
        voice_ratios (dict): Dictionary mapping voice IDs to percentages
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
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
            logging.warning(f"Voice ratios sum to {total}%. Normalizing to 100%.")
            voice_ratios = {k: (v / total) * 100 for k, v in voice_ratios.items()}
        
        # Set paths to model files
        model_path = os.path.join("models", "kokoro-v1.0.onnx")
        voices_path = os.path.join("models", "voices-v1.0.bin")
        
        # Check if model files exist
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            return False
            
        if not os.path.exists(voices_path):
            logging.error(f"Voices file not found at: {voices_path}")
            return False
        
        # Get Kokoro instance directly
        logging.info(f"Loading Kokoro model from {model_path} and voices from {voices_path}")
        kokoro = Kokoro(model_path, voices_path)
        
        # Create voice blend
        blend = None
        for voice_id, percentage in voice_ratios.items():
            logging.info(f"Adding {voice_id} ({SUPPORTED_VOICES[voice_id]}) at {percentage:.1f}%")
            voice_vector = kokoro.get_voice_style(voice_id)
            
            if blend is None:
                blend = voice_vector * (percentage / 100)
            else:
                blend = np.add(blend, voice_vector * (percentage / 100))
        
        # Store the blended voice directly in Kokoro model's voice store
        if hasattr(kokoro, '_voices'):
            # Check if _voices exists and is a dictionary
            if isinstance(getattr(kokoro, '_voices'), dict):
                kokoro._voices[profile_name] = blend
                logging.info(f"Successfully added blended voice '{profile_name}' to Kokoro voices")
            else:
                logging.warning("kokoro._voices exists but is not a dictionary")
                # Fall back to the patch method
                _patch_create_method(kokoro, blend)
        else:
            logging.warning("Could not access internal voice store, using patched create method")
            _patch_create_method(kokoro, blend)
        
        # Verify we can use the voice - carefully handle numpy arrays
        try:
            # Test with a very short text to minimize processing time
            samples, sample_rate = kokoro.create("Test.", voice=profile_name, speed=1.0, lang="en-us")
            if samples is not None and len(samples) > 0:
                logging.info(f"Successfully tested voice '{profile_name}'")
            else:
                raise ValueError("Generated samples is empty or None")
        except Exception as e1:
            logging.warning(f"Could not test voice by name: {e1}")
            try:
                # Try with the blend vector directly
                samples, sample_rate = kokoro.create("Test.", voice=blend, speed=1.0, lang="en-us")
                if samples is not None and len(samples) > 0:
                    logging.info("Voice blend works directly, but not by name")
                else:
                    raise ValueError("Generated samples is empty or None")
            except Exception as e2:
                logging.error(f"Failed to test blended voice directly: {e2}")
                return False
        
        # Create env file with the configuration
        with open(".env", "a") as f:
            f.write(f"\n# Blended voice configuration\n")
            f.write(f"KOKORO_VOICE=\"{profile_name}\"\n")
            
            # Write each voice component to the env file
            for voice_id, percentage in voice_ratios.items():
                voice_name = voice_id.upper()
                f.write(f"KOKORO_VOICE_{voice_name}_RATIO={percentage}\n")
        
        logging.info(f"Updated .env file with voice configuration: {profile_name}")
        logging.info(f"To use this voice in the voice assistant, set KOKORO_VOICE={profile_name} in your config")
        
        return True
    
    except Exception as e:
        logging.error(f"Error configuring blended voice: {e}")
        return False

def _patch_create_method(kokoro, blend):
    """Helper function to patch the create method of Kokoro"""
    original_create = kokoro.create
    
    def patched_create(text, voice=None, speed=1.0, lang="en-us"):
        # Check if voice is a string that matches am_michael
        if isinstance(voice, str) and voice == "am_michael":
            return original_create(text, voice=blend, speed=speed, lang=lang)
        return original_create(text, voice=voice, speed=speed, lang=lang)
    
    kokoro.create = patched_create
    logging.info("Patched Kokoro create method to use blended voice")

def list_available_voices():
    """List all available voice styles that can be used for blending"""
    print("Available Voices for Blending:")
    
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
    
    print("\nExample voice blend specification:")
    print("  af_nicole:30,am_michael:70")

def main():
    parser = argparse.ArgumentParser(description="Configure HowdyTTS to use a blended voice")
    parser.add_argument("--name", type=str, default="blended_cowboy",
                        help="Name for the blended voice profile")
    parser.add_argument("--voices", type=str,
                        help="Voice blend specification in format 'voice1:ratio1,voice2:ratio2,...' (e.g., 'af_nicole:30,am_michael:70')")
    parser.add_argument("--nicole", type=float, default=None,
                        help="Percentage of Nicole's voice (0-100) [legacy mode]")
    parser.add_argument("--michael", type=float, default=None,
                        help="Percentage of Michael's voice (0-100) [legacy mode]")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        list_available_voices()
        return
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("Error: The 'models' directory doesn't exist.")
        print("Make sure you're running this script from the project root directory.")
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
    
    # Configure the system
    success = patch_kokoro_for_blended_voice(args.name, voice_ratios)
    
    if success:
        print(f"\n✅ Voice assistant configured to use blended voice '{args.name}'")
        
        # Show the voice blend composition
        print("Voice blend composition:")
        for voice, ratio in voice_ratios.items():
            print(f"   {voice} ({SUPPORTED_VOICES[voice]}): {ratio:.1f}%")
        
        print("\nTo use this voice, run the voice assistant normally:")
        print("   python run_voice_assistant.py")
    else:
        print("\n❌ Failed to configure blended voice")
        print("   See log for details")

if __name__ == "__main__":
    main()