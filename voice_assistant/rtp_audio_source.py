#!/usr/bin/env python3
"""
RTP Audio Source for HowdyTTS
Receives RTP/UDP audio from ESP32-P4 RTP audio device and provides it to STT pipeline

This module integrates rtp_receiver.py with HowdyTTS's existing audio pipeline
by providing the same interface as NetworkAudioSource/EnhancedAudioRecorder.
"""

import logging
import threading
import time
import wave
import numpy as np
from typing import Optional
from collections import deque

from .rtp_receiver import RTPReceiver
from .intelligent_vad import IntelligentVAD

logger = logging.getLogger(__name__)


class RTPAudioSource:
    """
    RTP-based audio source for ESP32-P4 RTP audio device.

    This class provides the same interface as NetworkAudioSource/EnhancedAudioRecorder
    but receives audio via RTP/UDP from the rtp-audio-device firmware.

    Features:
    - RFC 3550 RTP packet reception and decoding
    - G.711 μ-law decoding to PCM
    - Intelligent VAD for speech detection
    - Compatible with HowdyTTS STT pipeline
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5004):
        """
        Initialize RTP audio source.

        Args:
            host: IP address to bind to (0.0.0.0 for all interfaces)
            port: UDP port to receive RTP packets (default 5004)
        """
        self.host = host
        self.port = port

        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 320  # 20ms @ 16kHz (matches RTP frame size)

        # RTP receiver
        self.rtp_receiver: Optional[RTPReceiver] = None

        # VAD for speech detection
        self.vad = IntelligentVAD(
            sample_rate=self.sample_rate,
            chunk_duration_ms=20
        )

        # Audio buffering for recording
        self.audio_buffer = deque(maxlen=1000)  # ~20 seconds at 20ms chunks
        self.pre_speech_buffer = deque(maxlen=40)  # ~800ms pre-speech buffer

        # Recording state
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self._recording_success = False

        # VAD state
        self._vad_residual = np.array([], dtype=np.int16)

        # Statistics
        self.stats = {
            'packets_received': 0,
            'audio_processed': 0,
            'vad_detections': 0,
            'speech_sessions': 0
        }

        logger.info(f"RTPAudioSource initialized: {host}:{port}")

    def start(self) -> bool:
        """Start the RTP audio source."""
        try:
            # Create and start RTP receiver
            self.rtp_receiver = RTPReceiver(
                host=self.host,
                port=self.port,
                audio_callback=self._on_audio_received
            )

            self.rtp_receiver.start()

            logger.info("✓ RTP audio source started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start RTP audio source: {e}")
            return False

    def stop(self):
        """Stop the RTP audio source."""
        logger.info("Stopping RTP audio source...")

        # Stop recording if active
        self.is_recording = False

        # Stop RTP receiver
        if self.rtp_receiver:
            self.rtp_receiver.stop()
            self.rtp_receiver = None

        # Wait for recording thread
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        logger.info("✓ RTP audio source stopped")

    def record_audio(self,
                    file_path: str,
                    max_duration: float = 30.0,
                    silence_timeout: float = 2.0,
                    energy_threshold: Optional[float] = None,
                    is_wake_word_response: bool = False,
                    **kwargs) -> bool:
        """
        Record audio from RTP stream using intelligent VAD.

        This method provides the same interface as EnhancedAudioRecorder.record_audio()
        and NetworkAudioSource.record_audio().

        Args:
            file_path: Path to save the recorded audio
            max_duration: Maximum recording duration in seconds
            silence_timeout: Maximum silence before stopping
            energy_threshold: Minimum energy threshold (unused for RTP)
            is_wake_word_response: Whether this is recording after wake word
            **kwargs: Additional arguments (for compatibility)

        Returns:
            bool: True if recording was successful, False otherwise
        """
        if not self.rtp_receiver:
            logger.error("RTP receiver not started")
            return False

        if self.is_recording:
            logger.warning("Already recording audio")
            return False

        logger.info(f"📱 Starting RTP audio recording from ESP32-P4 device")

        # Clear buffers
        self.audio_buffer.clear()
        self.pre_speech_buffer.clear()

        # Reset VAD state
        self.vad.reset()
        self._vad_residual = np.array([], dtype=np.int16)

        # Start recording
        self.is_recording = True
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(file_path, max_duration, silence_timeout, is_wake_word_response),
            daemon=True
        )
        self.recording_thread.start()

        # Wait for recording to complete
        self.recording_thread.join()

        # Check success
        success = self._recording_success

        if success:
            logger.info(f"✅ RTP audio recording successful: {file_path}")
        else:
            logger.warning(f"⚠️ RTP audio recording failed or no speech detected")

        return success

    def get_device_info(self) -> str:
        """Get current device information."""
        if self.rtp_receiver:
            stats = self.rtp_receiver.get_stats()
            return f"RTP Device: {self.host}:{self.port} ({stats['packets_received']} packets received)"
        return "RTP receiver not started"

    def get_stats(self) -> dict:
        """Get RTP audio source statistics."""
        result = {
            'rtp_audio': self.stats,
            'is_recording': self.is_recording
        }

        if self.rtp_receiver:
            result['rtp_receiver'] = self.rtp_receiver.get_stats()

        return result

    def has_recent_audio(self, max_age: float = 2.0) -> bool:
        """Check if we have received audio recently."""
        if not self.rtp_receiver:
            return False

        # Check RTP receiver stats
        stats = self.rtp_receiver.get_stats()
        return stats['packets_received'] > 0

    def _on_audio_received(self, pcm_data: bytes):
        """
        Callback for receiving decoded PCM audio from RTP receiver.

        Args:
            pcm_data: 16-bit PCM audio data
        """
        try:
            # Convert bytes to numpy array
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

            # Track statistics
            self.stats['packets_received'] += 1

            # Add to pre-speech buffer (always, even when not recording)
            self.pre_speech_buffer.append(audio_int16)

            # Add to recording buffer if recording
            if self.is_recording:
                self.audio_buffer.append(audio_int16)

        except Exception as e:
            logger.error(f"Error processing RTP audio: {e}")

    def _recording_loop(self, file_path: str, max_duration: float,
                       silence_timeout: float, is_wake_word_response: bool = False):
        """
        Main recording loop with VAD and utterance detection.

        Args:
            file_path: Path to save recorded audio
            max_duration: Maximum recording duration
            silence_timeout: Silence timeout before stopping
            is_wake_word_response: Whether this is a wake word response
        """
        recording_started = False
        speech_detected = False
        silence_start = None
        recorded_chunks = []

        start_time = time.time()

        # Grace period for wake word responses
        wake_word_grace_period = 1.0 if is_wake_word_response else 0.0
        force_record_timeout = 0.8  # Force recording after this time

        logger.info(f"📊 Starting RTP recording loop (wake_word_response: {is_wake_word_response})")

        while self.is_recording and (time.time() - start_time) < max_duration:
            try:
                # Wait for audio data
                if not self.audio_buffer:
                    time.sleep(0.01)
                    continue

                # Get next audio chunk
                audio_chunk = self.audio_buffer.popleft()
                self.stats['audio_processed'] += 1

                # Convert to float for VAD
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                audio_level = float(np.sqrt(np.mean(np.square(audio_float))))

                # Build VAD-sized chunk (Silero expects 512 samples @16kHz)
                vad_chunk_ready = False
                vad_input_float = None

                if audio_chunk.size:
                    combined = np.concatenate([self._vad_residual, audio_chunk])
                    if combined.size >= self.vad.chunk_size:
                        vad_chunk = combined[:self.vad.chunk_size]
                        self._vad_residual = combined[self.vad.chunk_size:]
                        vad_input_float = vad_chunk.astype(np.float32) / 32768.0
                        vad_chunk_ready = True
                    else:
                        self._vad_residual = combined

                # Run VAD when we have enough samples
                is_speech = False
                if vad_chunk_ready:
                    is_speech, confidence = self.vad.process_chunk(vad_input_float)

                    if is_speech:
                        logger.debug(f"🎙️ VAD: Speech detected (confidence: {confidence:.3f})")

                # Energy fallback for low-confidence situations
                energy_threshold = 0.003
                if not is_speech and audio_level >= energy_threshold:
                    logger.debug(f"🎚️ Energy fallback triggered (level {audio_level:.4f} >= {energy_threshold:.4f})")
                    is_speech = True

                # For wake word responses, be more permissive during grace period
                if is_wake_word_response and (time.time() - start_time) < wake_word_grace_period:
                    if audio_level > 0.005:
                        is_speech = True

                # Force recording start after timeout
                elapsed = time.time() - start_time
                if not recording_started and elapsed >= force_record_timeout:
                    logger.info(f"🎙️ Forcing recording start after {elapsed:.1f}s timeout")
                    is_speech = True

                # Handle speech detection
                if is_speech:
                    self.stats['vad_detections'] += 1

                    if not recording_started:
                        # Start recording - include pre-speech buffer
                        recorded_chunks.extend(list(self.pre_speech_buffer))
                        recording_started = True
                        speech_detected = True
                        self.stats['speech_sessions'] += 1
                        logger.info("🎙️ RTP speech detected - started recording")

                    # Reset silence timer
                    silence_start = None
                else:
                    # No speech detected
                    if recording_started:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > silence_timeout:
                            # Silence timeout reached
                            logger.info("🔇 RTP silence timeout - stopping recording")
                            break

                # Add chunk to recording if we're recording
                if recording_started:
                    recorded_chunks.append(audio_chunk)

            except Exception as e:
                logger.error(f"❌ Error in RTP recording loop: {e}")
                break

        self.is_recording = False

        # Save recorded audio
        if recorded_chunks and speech_detected:
            self._save_audio(recorded_chunks, file_path)
            self._recording_success = True
            logger.info(f"💾 RTP saved {len(recorded_chunks)} audio chunks to {file_path}")
        else:
            self._recording_success = False
            logger.warning("⚠️ RTP no speech detected during recording")

    def _save_audio(self, chunks: list, file_path: str):
        """
        Save recorded audio chunks to WAV file.

        Args:
            chunks: List of numpy arrays containing audio data
            file_path: Path to save the audio file
        """
        try:
            # Concatenate all chunks
            audio_data = np.concatenate(chunks)

            # Save as WAV file
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())

            duration = len(audio_data) / self.sample_rate
            logger.info(f"💾 Audio saved: {file_path} ({len(audio_data)} samples, {duration:.2f}s)")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create RTP audio source
    audio_source = RTPAudioSource(host="0.0.0.0", port=5004)

    if audio_source.start():
        try:
            logger.info("RTP audio source started - waiting for device connection...")
            time.sleep(5)

            logger.info("Testing RTP audio recording...")
            success = audio_source.record_audio(
                "test_rtp_recording.wav",
                max_duration=10.0,
                silence_timeout=3.0
            )

            logger.info(f"Recording result: {success}")

            # Show stats
            stats = audio_source.get_stats()
            logger.info(f"Stats: {stats}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            audio_source.stop()

    logger.info("RTP audio source test complete")