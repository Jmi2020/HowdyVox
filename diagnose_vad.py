#!/usr/bin/env python3
"""Diagnose why VAD is not detecting speech."""

import os
import sys
import time
import logging
import numpy as np
import pyaudio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_vad_model_loading():
    """Test if VAD model loads correctly."""
    print("\n=== Testing VAD Model Loading ===")
    try:
        from voice_assistant.intelligent_vad import IntelligentVAD
        print("✓ Imported IntelligentVAD successfully")
        
        vad = IntelligentVAD()
        print("✓ Created IntelligentVAD instance")
        
        print(f"✓ Model loaded: {vad.model is not None}")
        print(f"✓ Sample rate: {vad.sample_rate}")
        print(f"✓ Chunk size: {vad.chunk_size} samples")
        print(f"✓ Chunk duration: {vad.chunk_duration_ms}ms")
        
        return vad
    except Exception as e:
        print(f"✗ Failed to load VAD: {e}")
        logging.exception("VAD loading error:")
        return None

def test_microphone_input():
    """Test if microphone is working."""
    print("\n=== Testing Microphone Input ===")
    
    p = pyaudio.PyAudio()
    
    # List audio devices
    print("\nAvailable audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} - {info['maxInputChannels']} channels")
    
    # Test recording
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=480  # 30ms at 16kHz
        )
        
        print("\n✓ Microphone stream opened successfully")
        print("Recording 2 seconds of audio to test levels...")
        
        max_level = 0
        for i in range(int(2 * 16000 / 480)):  # 2 seconds
            data = stream.read(480, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            level = np.max(np.abs(audio_array))
            max_level = max(max_level, level)
            
            if i % 10 == 0:  # Print every 300ms
                print(f"  Audio level: {level} (max: {max_level})")
        
        stream.stop_stream()
        stream.close()
        
        print(f"\n✓ Microphone test complete. Max level: {max_level}")
        if max_level < 100:
            print("⚠️  WARNING: Audio levels are very low. Check microphone.")
        
        return True
    except Exception as e:
        print(f"✗ Microphone test failed: {e}")
        return False
    finally:
        p.terminate()

def test_vad_on_live_audio(vad):
    """Test VAD on live microphone input."""
    print("\n=== Testing VAD on Live Audio ===")
    print("Speak into the microphone for 5 seconds...")
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=480  # 30ms chunks
        )
        
        speech_detected_count = 0
        total_chunks = 0
        
        print("\nListening... (speak now)")
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            try:
                # Read audio
                audio_data = stream.read(480, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Process with VAD
                is_speech, confidence = vad.process_chunk(audio_float)
                total_chunks += 1
                
                if is_speech:
                    speech_detected_count += 1
                    print(f"🎤 SPEECH DETECTED! Confidence: {confidence:.2f}")
                
                # Print status every 0.5 seconds
                if total_chunks % 16 == 0:  # ~500ms
                    print(f"  Processed {total_chunks} chunks, speech in {speech_detected_count} chunks")
                    
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        stream.stop_stream()
        stream.close()
        
        print(f"\n=== Results ===")
        print(f"Total chunks: {total_chunks}")
        print(f"Speech detected: {speech_detected_count} chunks")
        print(f"Speech percentage: {speech_detected_count/total_chunks*100:.1f}%")
        
        if speech_detected_count == 0:
            print("\n⚠️  WARNING: No speech detected!")
            print("Possible issues:")
            print("  1. Microphone not working or too quiet")
            print("  2. VAD model not loaded correctly")
            print("  3. Audio format mismatch")
            
    except Exception as e:
        print(f"✗ Live audio test failed: {e}")
        logging.exception("Live audio error:")
    finally:
        p.terminate()

def test_vad_on_synthetic_speech():
    """Test VAD on synthetic speech-like signal."""
    print("\n=== Testing VAD on Synthetic Speech ===")
    
    vad = test_vad_model_loading()
    if not vad:
        return
    
    # Create a speech-like signal (modulated noise)
    t = np.linspace(0, 0.03, 480)  # 30ms
    
    # Test 1: Strong signal
    print("\nTest 1: Strong speech-like signal")
    carrier = np.random.normal(0, 0.3, 480)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4Hz modulation
    signal = (carrier * envelope).astype(np.float32)
    
    is_speech, confidence = vad.process_chunk(signal)
    print(f"Result: is_speech={is_speech}, confidence={confidence}")
    
    # Test 2: Weak signal
    print("\nTest 2: Weak speech-like signal")
    weak_signal = (signal * 0.1).astype(np.float32)
    is_speech, confidence = vad.process_chunk(weak_signal)
    print(f"Result: is_speech={is_speech}, confidence={confidence}")
    
    # Test 3: Pure noise
    print("\nTest 3: Pure noise")
    noise = np.random.normal(0, 0.1, 480).astype(np.float32)
    is_speech, confidence = vad.process_chunk(noise)
    print(f"Result: is_speech={is_speech}, confidence={confidence}")

def check_torch_and_model():
    """Check PyTorch and model details."""
    print("\n=== Checking PyTorch and Model ===")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        # Try loading model directly
        print("\nLoading Silero VAD model directly...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        print("✓ Model loaded successfully")
        
        # Test model on dummy input
        print("\nTesting model on dummy input...")
        dummy_input = torch.randn(480)  # 30ms at 16kHz
        model.reset_states()
        output = model(dummy_input, 16000)
        print(f"✓ Model output: {output.item():.4f}")
        
    except Exception as e:
        print(f"✗ PyTorch/Model check failed: {e}")
        logging.exception("PyTorch error:")

if __name__ == "__main__":
    print("=== VAD Diagnostic Tool ===\n")
    
    # Run all tests
    print("1. Checking PyTorch and model...")
    check_torch_and_model()
    
    print("\n2. Testing VAD model loading...")
    vad = test_vad_model_loading()
    
    print("\n3. Testing microphone input...")
    mic_ok = test_microphone_input()
    
    if vad and mic_ok:
        print("\n4. Testing VAD on live audio...")
        test_vad_on_live_audio(vad)
    
    print("\n5. Testing VAD on synthetic signals...")
    test_vad_on_synthetic_speech()
    
    print("\n=== Diagnostic Complete ===")