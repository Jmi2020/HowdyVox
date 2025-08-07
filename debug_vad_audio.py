#!/usr/bin/env python3
"""Debug VAD audio processing to see what's being sent to the model."""

import os
import sys
import time
import logging
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_assistant.intelligent_vad import IntelligentVAD

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("=== VAD Audio Debug ===\n")

# Initialize VAD
vad = IntelligentVAD()
print(f"VAD initialized with chunk size: {vad.chunk_size}")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Audio parameters matching VAD requirements
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requirement
FORMAT = pyaudio.paInt16

print(f"\nAudio settings:")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Chunk size: {CHUNK_SIZE} samples")
print(f"  Format: 16-bit signed integer")

# Open stream
stream = p.open(
    format=FORMAT,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE * 4  # Larger buffer to prevent underruns
)

print("\nRecording for 5 seconds. Please speak!\n")

# Collect data for analysis
all_audio = []
all_vad_results = []
chunk_count = 0
speech_chunks = 0

start_time = time.time()

while time.time() - start_time < 5.0:
    try:
        # Read audio
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        
        # Convert to numpy
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Store for analysis
        all_audio.extend(audio_array)
        
        # Convert to float32 for VAD
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Calculate some stats
        rms = np.sqrt(np.mean(audio_float**2))
        peak = np.max(np.abs(audio_float))
        
        # Process with VAD
        is_speech, confidence = vad.process_chunk(audio_float)
        all_vad_results.append((is_speech, confidence))
        
        chunk_count += 1
        if is_speech:
            speech_chunks += 1
            print(f"Chunk {chunk_count:3d}: SPEECH! RMS={rms:.4f}, Peak={peak:.4f}")
        else:
            print(f"Chunk {chunk_count:3d}: silence  RMS={rms:.4f}, Peak={peak:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        break

stream.stop_stream()
stream.close()
p.terminate()

print(f"\n=== Summary ===")
print(f"Total chunks processed: {chunk_count}")
print(f"Speech chunks detected: {speech_chunks}")
print(f"Speech percentage: {speech_chunks/chunk_count*100:.1f}%")

# Analyze the audio
all_audio = np.array(all_audio)
print(f"\nAudio statistics:")
print(f"  Total samples: {len(all_audio)}")
print(f"  Max amplitude: {np.max(np.abs(all_audio))}")
print(f"  RMS level: {np.sqrt(np.mean(all_audio.astype(np.float32)**2)):.1f}")
print(f"  Non-zero samples: {np.sum(all_audio != 0)}")

# Plot the audio and VAD results
plt.figure(figsize=(12, 8))

# Plot 1: Audio waveform
plt.subplot(3, 1, 1)
time_axis = np.arange(len(all_audio)) / SAMPLE_RATE
plt.plot(time_axis, all_audio)
plt.title('Audio Waveform')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot 2: Audio level (RMS in chunks)
plt.subplot(3, 1, 2)
chunk_times = []
chunk_rms = []
for i in range(0, len(all_audio) - CHUNK_SIZE, CHUNK_SIZE):
    chunk = all_audio[i:i+CHUNK_SIZE].astype(np.float32) / 32768.0
    chunk_times.append(i / SAMPLE_RATE)
    chunk_rms.append(np.sqrt(np.mean(chunk**2)))
plt.plot(chunk_times, chunk_rms, 'b-')
plt.title('Audio Level (RMS)')
plt.ylabel('RMS')
plt.grid(True)

# Plot 3: VAD results
plt.subplot(3, 1, 3)
vad_times = [i * CHUNK_SIZE / SAMPLE_RATE for i in range(len(all_vad_results))]
vad_values = [1 if result[0] else 0 for result in all_vad_results]
plt.plot(vad_times, vad_values, 'r-', linewidth=2)
plt.title('VAD Detection')
plt.ylabel('Speech Detected')
plt.xlabel('Time (seconds)')
plt.ylim(-0.1, 1.1)
plt.grid(True)

plt.tight_layout()
plt.savefig('test_output/vad_debug_plot.png')
print("\nPlot saved to: test_output/vad_debug_plot.png")

# Save a sample chunk for detailed analysis
print("\n=== Analyzing a sample chunk ===")
sample_chunk_idx = min(10, len(all_audio) // CHUNK_SIZE - 1)
sample_start = sample_chunk_idx * CHUNK_SIZE
sample_chunk = all_audio[sample_start:sample_start + CHUNK_SIZE]
sample_float = sample_chunk.astype(np.float32) / 32768.0

print(f"Sample chunk {sample_chunk_idx}:")
print(f"  Shape: {sample_chunk.shape}")
print(f"  Dtype: {sample_chunk.dtype}")
print(f"  Min: {np.min(sample_chunk)}")
print(f"  Max: {np.max(sample_chunk)}")
print(f"  Mean: {np.mean(sample_chunk):.2f}")
print(f"  Std: {np.std(sample_chunk):.2f}")
print(f"  First 10 values: {sample_chunk[:10]}")

# Test this chunk with VAD
vad.reset()
is_speech, confidence = vad.process_chunk(sample_float)
print(f"  VAD result: is_speech={is_speech}, confidence={confidence}")

# Try with louder audio
print("\n=== Testing with amplified audio ===")
amplified = sample_float * 10  # Amplify by 10x
amplified = np.clip(amplified, -1.0, 1.0)  # Clip to valid range
is_speech_amp, confidence_amp = vad.process_chunk(amplified)
print(f"Amplified result: is_speech={is_speech_amp}, confidence={confidence_amp}")

# Test with known speech-like pattern
print("\n=== Testing with synthetic speech pattern ===")
t = np.linspace(0, 0.032, 512)  # 32ms
# Create a complex waveform that mimics speech
fundamental = 0.2 * np.sin(2 * np.pi * 120 * t)  # 120 Hz fundamental
harmonic1 = 0.1 * np.sin(2 * np.pi * 240 * t)   # First harmonic
harmonic2 = 0.05 * np.sin(2 * np.pi * 360 * t)  # Second harmonic
noise = 0.05 * np.random.randn(512)              # Add some noise
envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) # 4Hz amplitude modulation
synthetic = (fundamental + harmonic1 + harmonic2 + noise) * envelope
synthetic = synthetic.astype(np.float32)

vad.reset()
is_speech_synth, confidence_synth = vad.process_chunk(synthetic)
print(f"Synthetic result: is_speech={is_speech_synth}, confidence={confidence_synth}")