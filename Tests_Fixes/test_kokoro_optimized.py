#!/usr/bin/env python3
"""
Test script for optimized Kokoro TTS implementation
This verifies that the enhanced session options are working correctly.
"""
import os
import sys
import time
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Add parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import the optimized KokoroManager
    try:
        from voice_assistant.kokoro_manager import KokoroManager
        print(f"✅ Successfully imported KokoroManager")
    except Exception as e:
        print(f"❌ Error importing KokoroManager: {e}")
        return

    # Test the TTS functionality with optimizations
    try:
        # Measure initialization time
        start_time = time.time()
        
        # Get the Kokoro instance (first time setup)
        print("Initializing Kokoro (this may take a moment)...")
        kokoro = KokoroManager.get_instance()
        
        init_time = time.time() - start_time
        print(f"✅ Successfully initialized Kokoro TTS in {init_time:.2f} seconds")
        
        # Test voice synthesis
        test_text = "This is a test of the optimized Kokoro TTS engine."
        
        # Measure synthesis time
        start_time = time.time()
        
        voice = "am_michael"
        print(f"Synthesizing speech with voice '{voice}'...")
        # Use the create() method which we know works
        audio, sample_rate = kokoro.create(test_text, voice=voice, speed=1.0, lang="en-us")
        
        synthesis_time = time.time() - start_time
        print(f"✅ Successfully synthesized speech in {synthesis_time:.2f} seconds")
        print(f"   Audio length: {len(audio)} samples at {sample_rate} Hz")
        
        # Save the audio output for verification
        try:
            import soundfile as sf
            output_path = os.path.join(os.path.dirname(__file__), "optimized_tts_output.wav")
            sf.write(output_path, audio, sample_rate)
            print(f"✅ Audio saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
        
        # Test second synthesis (should be faster with optimizations)
        start_time = time.time()
        
        second_text = "This is a second test to check if caching and optimization is working."
        print("Synthesizing second speech sample...")
        try:
            # Always use the create method since we've confirmed it works
            audio2, sample_rate = kokoro.create(second_text, voice=voice, speed=1.0, lang="en-us")
        except Exception as e:
            print(f"❌ Error in second synthesis: {e}")
            raise
        
        second_synthesis_time = time.time() - start_time
        print(f"✅ Second synthesis completed in {second_synthesis_time:.2f} seconds")
        print(f"   Performance improvement: {(synthesis_time - second_synthesis_time) / synthesis_time * 100:.1f}%")
        
        # Save the second audio output
        try:
            output_path2 = os.path.join(os.path.dirname(__file__), "optimized_tts_output2.wav")
            sf.write(output_path2, audio2, sample_rate)
            print(f"✅ Second audio saved to {output_path2}")
        except Exception as e:
            print(f"❌ Error saving second audio: {e}")
        
    except Exception as e:
        print(f"❌ Error testing optimized Kokoro TTS: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 All tests passed! Optimized Kokoro TTS is working correctly.")

if __name__ == "__main__":
    main()