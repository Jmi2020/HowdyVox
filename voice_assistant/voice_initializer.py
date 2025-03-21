"""
Voice Initializer Module

This module handles initialization of voice profiles, including blended voices.
It's used during startup to ensure custom voice profiles are properly registered.
"""

import os
import logging
import numpy as np
from dotenv import load_dotenv
from voice_assistant.kokoro_manager import KokoroManager
from voice_assistant.config import Config

# Load environment variables
load_dotenv()

def initialize_voices():
    """
    Initialize custom voice profiles based on .env configuration.
    This should be called during startup to ensure blended voices are available.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Reset Kokoro instance to ensure we start fresh
        KokoroManager.reset_instance()
        
        # Get a fresh Kokoro instance
        kokoro = KokoroManager.get_instance(local_model_path=Config.LOCAL_MODEL_PATH)
        
        # Look for voice blend configuration in environment variables
        voice_name = os.getenv("KOKORO_VOICE")
        if not voice_name:
            voice_name = Config.KOKORO_VOICE
            
        logging.info(f"Initializing voice configuration, target voice: {voice_name}")
        
        # Check if this is a blended voice by looking for ratio environment variables
        voice_ratios = {}
        for env_var, value in os.environ.items():
            if env_var.startswith("KOKORO_VOICE_") and env_var.endswith("_RATIO"):
                voice_id = env_var.replace("KOKORO_VOICE_", "").replace("_RATIO", "").lower()
                try:
                    ratio = float(value)
                    voice_ratios[voice_id] = ratio
                except ValueError:
                    logging.warning(f"Invalid voice ratio value for {voice_id}: {value}")
        
        # If we found voice ratios and there's more than one voice component, this is a blended voice
        if len(voice_ratios) > 1:
            logging.info(f"Found blended voice configuration with {len(voice_ratios)} components: {voice_ratios}")
            
            # Create the blended voice
            blend = None
            for voice_id, percentage in voice_ratios.items():
                try:
                    voice_vector = kokoro.get_voice_style(voice_id)
                    if blend is None:
                        blend = voice_vector * (percentage / 100)
                    else:
                        blend = np.add(blend, voice_vector * (percentage / 100))
                        
                    logging.info(f"Added {voice_id} at {percentage:.1f}%")
                except Exception as e:
                    logging.error(f"Failed to add voice {voice_id}: {e}")
            
            if blend is not None:
                # Register the blended voice directly in the kokoro instance
                if hasattr(kokoro, '_voices') and isinstance(getattr(kokoro, '_voices'), dict):
                    kokoro._voices[voice_name] = blend
                    logging.info(f"Successfully registered blended voice '{voice_name}'")
                else:
                    # If we can't access the internal voices dict, monkey patch the create method
                    original_create = kokoro.create
                    
                    def patched_create(text, voice=None, speed=1.0, lang="en-us"):
                        if isinstance(voice, str) and voice == voice_name:
                            # Use our blend for this specific voice name
                            logging.debug(f"Using blended voice for '{voice_name}'")
                            return original_create(text, voice=blend, speed=speed, lang=lang)
                        return original_create(text, voice=voice, speed=speed, lang=lang)
                    
                    kokoro.create = patched_create
                    logging.info(f"Patched Kokoro create method to support blended voice '{voice_name}'")
                
                # Verify the voice works
                try:
                    samples, _ = kokoro.create("Test", voice=voice_name, speed=1.0, lang="en-us")
                    if samples is not None and len(samples) > 0:
                        logging.info(f"Successfully verified blended voice '{voice_name}'")
                        return True
                except Exception as e:
                    logging.error(f"Could not verify voice by name: {e}")
                    logging.info("Will use direct voice vector instead of named profile")
        else:
            logging.info("No blended voice configuration found, using standard voice")
            
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize voice configuration: {e}")
        return False

# Auto-initialize when the module is imported
initialize_success = initialize_voices()