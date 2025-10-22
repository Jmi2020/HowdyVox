#!/usr/bin/env python3
"""
Audio-reactive playback wrapper for HowdyVox
Integrates ReactiveMeter with audio playback for face animation
"""

import wave
import pyaudio
import os
import logging
import time
from typing import Optional

# Import the reactive meter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_modules'))
from tts_reactive_meter import ReactiveMeter

# Global meter instance
_meter_instance: Optional[ReactiveMeter] = None
_meter_enabled = False


def init_reactive_meter(enabled=True, udp_host="127.0.0.1", udp_port=31337):
    """
    Initialize the global reactive meter instance.

    Args:
        enabled: Whether to enable audio reactivity
        udp_host: Hostname/IP of face renderer
        udp_port: UDP port of face renderer
    """
    global _meter_instance, _meter_enabled

    _meter_enabled = enabled

    if enabled:
        _meter_instance = ReactiveMeter(
            samplerate=24000,  # Kokoro default
            sample_width=2,  # 16-bit PCM
            channels=1,  # Mono
            udp_host=udp_host,
            udp_port=udp_port,
            win_ms=20,  # 20ms analysis window
            update_hz=12,  # 12 updates per second
        )
        logging.info(f"Audio reactive meter initialized → {udp_host}:{udp_port}")
    else:
        logging.info("Audio reactive meter disabled")


def get_reactive_meter() -> Optional[ReactiveMeter]:
    """Get the global reactive meter instance"""
    return _meter_instance if _meter_enabled else None


def play_audio_reactive(file_path: str):
    """
    Play audio file with reactive analysis for face animation.

    This wraps the standard play_audio with reactive meter processing.

    Args:
        file_path: Path to WAV file to play
    """
    if not os.path.exists(file_path):
        logging.error(f"Audio file not found: {file_path}")
        return

    meter = get_reactive_meter()

    # Signal start of speech
    if meter:
        meter.send_state("speaking")

    try:
        # Open WAV file
        wf = wave.open(file_path, "rb")
        p = pyaudio.PyAudio()

        # Open audio stream
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        # Get audio parameters
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()

        # Log audio info
        logging.debug(f"Playing: {file_path} ({sample_rate}Hz, {channels}ch)")

        # Playback loop
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        frame_count = 0

        while data:
            # Play audio
            stream.write(data)

            # Feed to reactive meter
            if meter:
                meter.process(data)

                # Call tick periodically (~every 83ms at 12 Hz)
                if frame_count % 10 == 0:
                    meter.tick()

            # Read next chunk
            data = wf.readframes(chunk_size)
            frame_count += 1

        # Final tick to ensure last features are sent
        if meter:
            meter.tick()

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

    except Exception as e:
        logging.error(f"Error in reactive playback: {e}")
    finally:
        # Signal end of speech
        if meter and _meter_enabled:
            meter.send_state("idle")


def send_state(state: str):
    """
    Send a state message to the face renderer.

    Args:
        state: State name (idle, listening, thinking, speaking)
    """
    meter = get_reactive_meter()
    if meter:
        meter.send_state(state)


# Example usage
if __name__ == "__main__":
    import numpy as np
    from array import array
    import soundfile as sf

    print("🧪 Testing audio reactive playback")
    print("=" * 50)

    # Generate test audio
    print("Generating test audio...")
    duration = 3.0
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Sine sweep with amplitude modulation
    freq = 440 + 200 * np.sin(2 * np.pi * 0.5 * t)
    amplitude = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 0.3 * t))
    audio = amplitude * np.sin(2 * np.pi * freq * t)

    # Save as WAV
    test_file = "temp/test_reactive.wav"
    os.makedirs("temp", exist_ok=True)
    sf.write(test_file, audio.astype(np.float32), sample_rate)

    # Initialize meter
    init_reactive_meter(enabled=True, udp_host="127.0.0.1", udp_port=31337)

    # Play with reactive analysis
    print("Playing with reactive analysis...")
    play_audio_reactive(test_file)

    print("✓ Test complete!")
