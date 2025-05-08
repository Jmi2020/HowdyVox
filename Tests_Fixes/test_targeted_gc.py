#!/usr/bin/env python3
"""
Test script to verify the targeted garbage collection implementation
"""
import os
import sys
import time
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print(f"Testing Targeted Garbage Collection")
    
    # Add the parent directory to sys.path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import the targeted_gc function
    try:
        from voice_assistant.utils import targeted_gc
        print("✅ Successfully imported targeted_gc function")
    except Exception as e:
        print(f"❌ Failed to import targeted_gc function: {e}")
        return
    
    # Create some audio-related objects to test the targeted garbage collection
    try:
        import pyaudio
        import wave
        from pydub import AudioSegment
        import io
        
        print("Creating audio objects to test garbage collection...")
        
        # Create a PyAudio object
        p = pyaudio.PyAudio()
        print(f"✅ Created PyAudio object: {p}")
        
        # Create a wave object if a test file exists
        wave_obj = None
        test_file_path = os.path.join(parent_dir, "test_audio", "test.wav")
        if not os.path.exists(test_file_path):
            # Create a simple test file
            print("Creating test audio file...")
            sample_rate = 44100
            duration = 1  # seconds
            samples = b'\x00' * (sample_rate * 2 * duration)  # 1 second of silence, 16-bit
            with open(test_file_path, 'wb') as f:
                f.write(b'RIFF')
                f.write((36 + len(samples)).to_bytes(4, 'little'))  # File size
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write((16).to_bytes(4, 'little'))  # Subchunk1Size
                f.write((1).to_bytes(2, 'little'))  # AudioFormat (PCM)
                f.write((1).to_bytes(2, 'little'))  # NumChannels
                f.write((sample_rate).to_bytes(4, 'little'))  # SampleRate
                f.write((sample_rate * 2).to_bytes(4, 'little'))  # ByteRate
                f.write((2).to_bytes(2, 'little'))  # BlockAlign
                f.write((16).to_bytes(2, 'little'))  # BitsPerSample
                f.write(b'data')
                f.write(len(samples).to_bytes(4, 'little'))  # Subchunk2Size
                f.write(samples)
        
        try:
            wave_obj = wave.open(test_file_path, 'rb')
            print(f"✅ Created Wave_read object: {wave_obj}")
        except Exception as e:
            print(f"❌ Could not create Wave_read object: {e}")
        
        # Create an AudioSegment object
        audio_segment = None
        try:
            audio_segment = AudioSegment.silent(duration=1000)  # 1 second of silence
            print(f"✅ Created AudioSegment object: {audio_segment}")
        except Exception as e:
            print(f"❌ Could not create AudioSegment object: {e}")
        
        # Count objects before garbage collection
        pyaudio_count_before = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'PyAudio')
        wave_count_before = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'Wave_read')
        audio_segment_count_before = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'AudioSegment')
        
        print(f"Before targeted GC: PyAudio: {pyaudio_count_before}, Wave_read: {wave_count_before}, AudioSegment: {audio_segment_count_before}")
        
        # Run targeted garbage collection
        print("Running targeted garbage collection...")
        cleaned_count = targeted_gc()
        print(f"✅ Targeted GC cleaned up {cleaned_count} audio-related objects")
        
        # Count objects after garbage collection
        pyaudio_count_after = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'PyAudio')
        wave_count_after = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'Wave_read')
        audio_segment_count_after = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'AudioSegment')
        
        print(f"After targeted GC: PyAudio: {pyaudio_count_after}, Wave_read: {wave_count_after}, AudioSegment: {audio_segment_count_after}")
        
        # Calculate the difference
        pyaudio_diff = pyaudio_count_before - pyaudio_count_after
        wave_diff = wave_count_before - wave_count_after
        audio_segment_diff = audio_segment_count_before - audio_segment_count_after
        
        print(f"Objects cleaned up: PyAudio: {pyaudio_diff}, Wave_read: {wave_diff}, AudioSegment: {audio_segment_diff}")
        
        # Run standard garbage collection for comparison
        print("\nRunning standard garbage collection for comparison...")
        gc.collect()
        
        # Count objects after standard garbage collection
        pyaudio_count_after_std = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'PyAudio')
        wave_count_after_std = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'Wave_read')
        audio_segment_count_after_std = sum(1 for obj in gc.get_objects() if type(obj).__name__ == 'AudioSegment')
        
        print(f"After standard GC: PyAudio: {pyaudio_count_after_std}, Wave_read: {wave_count_after_std}, AudioSegment: {audio_segment_count_after_std}")
        
    except Exception as e:
        print(f"❌ Error testing targeted garbage collection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Targeted garbage collection tests completed!")

if __name__ == "__main__":
    main()