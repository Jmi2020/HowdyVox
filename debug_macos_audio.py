#!/usr/bin/env python3
"""Debug audio issues specific to macOS."""

import os
import sys
import pyaudio
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_assistant.intelligent_vad import IntelligentVAD

print("=== macOS Audio Diagnostic ===\n")

# Check microphone permissions
print("1. Checking microphone access...")
print("   If prompted, please allow microphone access in System Preferences")
print("   Go to: System Preferences > Security & Privacy > Privacy > Microphone\n")

p = pyaudio.PyAudio()

# List all audio devices
print("2. Available audio input devices:")
default_input = None
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        is_default = " (DEFAULT)" if i == p.get_default_input_device_info()['index'] else ""
        print(f"   [{i}] {info['name']} - {info['maxInputChannels']} ch, {info['defaultSampleRate']} Hz{is_default}")
        if is_default:
            default_input = i

# Test different audio configurations
print("\n3. Testing different audio configurations...")

configs = [
    {"device": None, "channels": 1, "name": "Default device, Mono"},
    {"device": None, "channels": 2, "name": "Default device, Stereo"},
    {"device": default_input, "channels": 1, "name": "Explicit default, Mono"},
]

# Add specific devices if found
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0 and "Opal" in info['name']:
        configs.append({"device": i, "channels": 1, "name": f"Opal Tadpole, Mono"})
    elif info['maxInputChannels'] > 0 and "MacBook" in info['name']:
        configs.append({"device": i, "channels": 1, "name": f"MacBook Mic, Mono"})

vad = IntelligentVAD()

for config in configs:
    print(f"\nTesting: {config['name']}")
    try:
        # Open stream with specific config
        stream = p.open(
            format=pyaudio.paInt16,
            channels=config['channels'],
            rate=16000,
            input=True,
            input_device_index=config['device'],
            frames_per_buffer=512
        )
        
        print("  Recording 2 seconds...")
        speech_detected = False
        max_level = 0
        
        for _ in range(int(2 * 16000 / 512)):  # 2 seconds
            try:
                # Read audio
                audio_data = stream.read(512, exception_on_overflow=False)
                
                # Handle stereo to mono conversion if needed
                if config['channels'] == 2:
                    # Convert stereo to mono by averaging channels
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                else:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Check audio level
                level = np.max(np.abs(audio_array))
                max_level = max(max_level, level)
                
                # Test with VAD
                audio_float = audio_array.astype(np.float32) / 32768.0
                is_speech, confidence = vad.process_chunk(audio_float)
                
                if is_speech:
                    speech_detected = True
                    print(f"  ✓ Speech detected! Level: {level}")
                    
            except Exception as e:
                print(f"  Error reading: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        print(f"  Max level: {max_level}")
        print(f"  Speech detected: {'Yes' if speech_detected else 'No'}")
        
        if max_level < 100:
            print("  ⚠️  WARNING: Very low audio levels!")
        
    except Exception as e:
        print(f"  ✗ Failed to open stream: {e}")

# Test with different buffer sizes
print("\n4. Testing different buffer sizes...")
buffer_sizes = [512, 1024, 2048, 4096]

for buffer_size in buffer_sizes:
    print(f"\nBuffer size: {buffer_size}")
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=buffer_size
        )
        
        # Test a few reads
        success_count = 0
        for i in range(5):
            try:
                audio_data = stream.read(512, exception_on_overflow=False)
                if len(audio_data) == 512 * 2:  # 2 bytes per sample
                    success_count += 1
            except Exception as e:
                print(f"  Read {i} failed: {e}")
        
        stream.stop_stream()
        stream.close()
        
        print(f"  Successful reads: {success_count}/5")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# macOS-specific audio format test
print("\n5. Testing macOS audio format handling...")
try:
    # Try Float32 format (sometimes works better on macOS)
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=512
    )
    
    print("  Float32 format: ✓ Supported")
    
    # Read and test
    audio_data = stream.read(512)
    audio_float = np.frombuffer(audio_data, dtype=np.float32)
    print(f"  Audio range: [{np.min(audio_float):.3f}, {np.max(audio_float):.3f}]")
    
    # Test with VAD
    vad.reset()
    is_speech, confidence = vad.process_chunk(audio_float)
    print(f"  VAD test: is_speech={is_speech}")
    
    stream.stop_stream()
    stream.close()
    
except Exception as e:
    print(f"  Float32 format: ✗ Not supported - {e}")

p.terminate()

print("\n=== Recommendations ===")
print("1. Check System Preferences > Security & Privacy > Privacy > Microphone")
print("2. Make sure Python/Terminal has microphone access")
print("3. Try using a different microphone if available")
print("4. Consider using input_device_index to specify exact device")
print("5. If using stereo device, convert to mono before VAD processing")