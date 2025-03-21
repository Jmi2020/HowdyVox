"""
Blended Voice TTS Module for HowdyTTS

This module extends the text-to-speech capabilities of HowdyTTS with voice blending.
It allows for creating custom voice profiles by blending different voice styles.
"""

import os
import soundfile as sf
import numpy as np
import logging
from voice_assistant.config import Config
from voice_assistant.kokoro_manager import KokoroManager
from voice_assistant.text_to_speech import split_text_into_chunks

# Store blended voice profiles
voice_profiles = {}

def create_voice_profile(name, voice_components):
    """
    Create a blended voice profile.
    
    Args:
        name (str): Name for this voice profile
        voice_components (dict): Dictionary mapping voice IDs to percentages
            Example: {"af_nicole": 30, "am_michael": 70}
    
    Returns:
        np.ndarray: The blended voice vector
    """
    # Validate total percentages add up to 100
    total = sum(voice_components.values())
    if total != 100:
        logging.warning(f"Voice percentages sum to {total}, normalizing to 100%")
        # Normalize percentages
        voice_components = {k: (v / total) * 100 for k, v in voice_components.items()}
    
    kokoro = KokoroManager.get_instance()
    
    # Get voice styles and blend them
    blend = None
    for voice_id, percentage in voice_components.items():
        try:
            voice_vector = kokoro.get_voice_style(voice_id)
            if blend is None:
                blend = voice_vector * (percentage / 100)
            else:
                blend = np.add(blend, voice_vector * (percentage / 100))
        except Exception as e:
            logging.error(f"Error getting voice style '{voice_id}': {e}")
            return None
    
    if blend is None:
        logging.error("Failed to create voice blend")
        return None
        
    # Store the profile
    voice_profiles[name] = blend
    logging.info(f"Created voice profile '{name}' with components: {voice_components}")
    
    return blend

def get_voice_profile(name):
    """
    Get a previously created voice profile.
    
    Args:
        name (str): Name of the voice profile
        
    Returns:
        np.ndarray or None: The voice profile vector or None if not found
    """
    return voice_profiles.get(name, None)

def tts_with_blended_voice(text, voice_profile_name, output_file="blended_output.wav"):
    """
    Generate text-to-speech using a blended voice profile.
    
    Args:
        text (str): Text to convert to speech
        voice_profile_name (str): Name of the voice profile to use
        output_file (str): Output file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if profile exists
    profile = get_voice_profile(voice_profile_name)
    if profile is None:
        logging.error(f"Voice profile '{voice_profile_name}' not found")
        return False
    
    try:
        # Get Kokoro instance
        kokoro = KokoroManager.get_instance()
        
        # Split text into manageable chunks for longer content
        chunks = split_text_into_chunks(text)
        
        # Process each chunk
        all_samples = []
        sample_rate = None
        
        for i, chunk in enumerate(chunks):
            # Generate audio with the blended voice
            samples, chunk_sample_rate = kokoro.create(
                chunk,
                voice=profile,
                speed=Config.KOKORO_SPEED,
                lang="en-us"
            )
            
            # Store the first sample rate (should be the same for all chunks)
            if sample_rate is None:
                sample_rate = chunk_sample_rate
                
            # Append samples
            all_samples.append(samples)
        
        # Concatenate all samples
        if all_samples:
            # Convert list of arrays to a single array
            combined_samples = np.concatenate(all_samples)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # Write the audio file
            sf.write(output_file, combined_samples, sample_rate)
            logging.info(f"Generated TTS audio with voice profile '{voice_profile_name}': {output_file}")
            return True
        else:
            logging.error("No audio samples generated")
            return False
            
    except Exception as e:
        logging.error(f"Error generating TTS with blended voice: {e}")
        return False

# Create some default blended voice profiles
def initialize_default_profiles():
    """Initialize default voice blend profiles"""
    try:
        # Mild Nicole blend - Slight female influence (80% Michael, 20% Nicole)
        create_voice_profile("mild_nicole", {"am_michael": 80, "af_nicole": 20})
        
        # Balanced blend - 50/50 mix
        create_voice_profile("balanced", {"am_michael": 50, "af_nicole": 50})
        
        # Gentle cowboy - 70% Michael, 30% Nicole (like in the original example)
        create_voice_profile("gentle_cowboy", {"am_michael": 70, "af_nicole": 30})
        
        logging.info("Default voice profiles initialized")
    except Exception as e:
        logging.error(f"Failed to initialize default voice profiles: {e}")

# Initialize profiles when module is imported
initialize_default_profiles()