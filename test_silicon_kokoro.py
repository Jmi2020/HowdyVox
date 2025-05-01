#!/usr/bin/env python3
# test_silicon_kokoro.py
"""
Test script to verify that Kokoro TTS is correctly using onnxruntime-silicon on Apple Silicon Macs.
This will print diagnostic information and try to synthesize a short test phrase.
"""

import os
import sys
import platform
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def main():
    print("\n===== Testing Kokoro TTS with Silicon optimizations =====\n")
    
    # Check platform
    print(f"Platform: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {sys.version}")
    
    # Check if we're on Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        print("\n✅ Running on Apple Silicon Mac")
    else:
        print("\n⚠️ Not running on Apple Silicon Mac - optimizations won't apply")
    
    # Import ONNX Runtime and check version
    try:
        # Try to import the silicon-specific runtime first
        print("\nAttempting to import onnxruntime-silicon...")
        import onnxruntime_silicon as ort
        print(f"✅ Successfully imported onnxruntime-silicon")
        try:
            print(f"Version: {ort.__version__}")
        except AttributeError:
            print(f"Version info not available")
        print(f"Available providers: {ort.get_available_providers() if hasattr(ort, 'get_available_providers') else 'Unknown'}")
    except ImportError:
        print("❌ Could not import onnxruntime-silicon, falling back to standard onnxruntime")
        try:
            import onnxruntime as ort
            print(f"Using standard onnxruntime")
            try:
                print(f"Version: {ort.__version__}")
            except AttributeError:
                print(f"Version info not available")
            print(f"Available providers: {ort.get_available_providers() if hasattr(ort, 'get_available_providers') else 'Unknown'}")
        except ImportError:
            print("❌ No version of onnxruntime is installed!")
            return False
    
    # Import Kokoro manager
    print("\nInitializing Kokoro TTS...")
    try:
        from voice_assistant.kokoro_manager import KokoroManager
        
        start_time = time.time()
        # Get model path from environment
        local_model_path = os.getenv("LOCAL_MODEL_PATH", "models")
        kokoro = KokoroManager.get_instance(local_model_path=local_model_path)
        init_time = time.time() - start_time
        print(f"✅ Kokoro TTS model initialized in {init_time:.2f} seconds")
        
        # Test synthesis
        print("\nSynthesizing test audio...")
        test_text = "Howdy partner! This is a test of the Kokoro TTS engine with Silicon optimizations."
        
        # Get voice from environment or use default
        voice = os.getenv("KOKORO_VOICE", "am_michael")
        
        start_time = time.time()
        audio = kokoro.predict(test_text, voice=voice)
        synthesis_time = time.time() - start_time
        print(f"✅ Audio synthesized in {synthesis_time:.2f} seconds")
        
        # Save audio to a test file
        output_path = "test_silicon_synthesis.wav"
        import soundfile as sf
        sf.write(output_path, audio, 24000)
        print(f"✅ Audio saved to {output_path}")
        
        # Try to play the audio
        try:
            import pygame
            pygame.mixer.init(frequency=24000)
            pygame.mixer.music.load(output_path)
            pygame.mixer.music.play()
            print("✅ Playing audio... (if you hear it, everything is working!)")
            time.sleep(5)  # Let it play for a few seconds
            pygame.mixer.quit()
        except Exception as e:
            print(f"⚠️ Could not play audio: {e}")
        
        print("\n===== Test completed successfully =====")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n===== Test failed =====")
        return False

if __name__ == "__main__":
    main()
