"""
Integration with the voice assistant framework for KokoroOnnxTTS
"""

import os
import time
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
import soundfile as sf
from pathlib import Path
from .tts import KokoroOnnxTTS

class KokoroOnnxIntegration:
    """
    Integration class for using KokoroOnnxTTS with the voice assistant framework
    """
    
    def __init__(self, voice: str = "am_michael", model_path: Optional[str] = None, dtype: str = "q8"):
        """
        Initialize the KokoroTTS ONNX integration
        
        Args:
            voice: Name of the voice to use
            model_path: Optional path to model directory
            dtype: Model precision - "fp32", "fp16", "q8", "q4", or "q4f16"
        """
        try:
            self.tts = KokoroOnnxTTS(model_path=model_path, voice=voice, dtype=dtype)
            self.available_voices = self.tts.list_voices()
            
            if self.available_voices:
                print(f"Available voices: {', '.join(self.available_voices)}")
                
                # Check if requested voice is available
                if voice not in self.available_voices:
                    print(f"Requested voice '{voice}' not found. Falling back to first available voice.")
                    self.voice = self.available_voices[0]
                else:
                    self.voice = voice
                    
                print(f"Initialized KokoroOnnxIntegration with voice: {self.voice}")
            else:
                print("No voices found. TTS may not work properly.")
                self.voice = voice  # Keep the original voice name even though it doesn't exist
        except Exception as e:
            print(f"Error initializing KokoroOnnxTTS: {e}")
            print("Setting up dummy TTS (will not produce audio)")
            # Create a dummy TTS as fallback
            self.tts = None
            self.available_voices = []
            self.voice = voice
            
        self.sample_rate = getattr(self.tts, "SAMPLE_RATE", 24000) if self.tts else 24000
        
    def generate_audio(self, text: str, output_path: str) -> str:
        """
        Generate audio from text and save to the output path
        
        Args:
            text: The text to convert to speech
            output_path: Path to save the audio file
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            if not self.tts:
                print("Warning: TTS engine not available")
                return ""
                
            start_time = time.time()
            print(f"Generating audio with KokoroOnnxTTS: {text[:50]}...")
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Get active voice - use the current voice first, 
            # but fallback to any available voice if current voice doesn't exist
            voice_to_use = self.voice
            if self.available_voices and voice_to_use not in self.available_voices:
                print(f"Voice '{voice_to_use}' not available. Available voices: {', '.join(self.available_voices)}")
                voice_to_use = self.available_voices[0]
                print(f"Falling back to voice: {voice_to_use}")
            
            # Generate speech with options
            options = {
                "voice": voice_to_use,
                "output_file": output_path
            }
            
            # Generate speech
            audio_array, _ = self.tts.generate(text, options)
            
            duration = time.time() - start_time
            print(f"Audio generated in {duration:.2f}s and saved to {output_path}")
            
            return output_path
        except Exception as e:
            print(f"Error generating audio: {e}")
            return ""
            
    def test_tts(self, text: str = "Howdy partner! This is a test of the new and improved KokoroTTS ONNX integration."):
        """
        Test the TTS functionality with a sample text
        
        Args:
            text: Sample text to test with
        """
        try:
            if not self.tts:
                print("Warning: TTS engine not available")
                return
                
            temp_file = "test_kokoro_onnx.wav"
            print(f"Testing KokoroOnnxTTS with voice '{self.voice}'")
            print(f"Text: '{text}'")
            
            self.generate_audio(text, temp_file)
            print(f"Test audio saved to {os.path.abspath(temp_file)}")
            
            # Print available voices
            voices = self.tts.list_voices()
            if voices:
                print(f"Available voices: {', '.join(voices)}")
        except Exception as e:
            print(f"Test failed: {e}")
    
    def list_voices(self):
        """List all available voices."""
        return self.available_voices if self.tts else []
    
    def change_voice(self, voice: str) -> bool:
        """
        Change the active voice.
        
        Args:
            voice: The voice ID to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.tts:
            return False
            
        if voice not in self.available_voices and self.available_voices:
            print(f"Voice '{voice}' not found. Available: {', '.join(self.available_voices)}")
            return False
            
        self.voice = voice
        print(f"Changed voice to: {voice}")
        return True

# Helper function to create the integration
def create_onnx_tts(voice: str = "am_michael", model_path: Optional[str] = None, dtype: str = "q8") -> KokoroOnnxIntegration:
    """
    Create a KokoroOnnxIntegration instance
    
    Args:
        voice: Voice name to use
        model_path: Optional path to model directory
        dtype: Model precision - "fp32", "fp16", "q8", "q4", or "q4f16"
        
    Returns:
        KokoroOnnxIntegration: The integration instance
    """
    return KokoroOnnxIntegration(voice=voice, model_path=model_path, dtype=dtype)