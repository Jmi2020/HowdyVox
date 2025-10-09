# voice_assistant/mac_audio_recorder.py

import os
import time
import logging
import wave
import numpy as np
from collections import deque
from typing import Optional, Tuple
from pydub import AudioSegment
import threading

from .mac_voice_isolation import MacVoiceIsolation, VoiceIsolationConfig
from .mac_voice_isolation import kAUVoiceIOProperty_VoiceProcessingQuality_High
from .intelligent_vad import IntelligentVAD
from .utterance_detector import IntelligentUtteranceDetector, UtteranceContext
from .config import Config

class MacAudioRecorder:
    """
    Audio recorder with macOS native voice isolation.
    
    This recorder combines:
    - macOS Neural Engine voice isolation
    - Intelligent VAD for speech detection
    - Pre-speech buffering
    - Utterance boundary detection
    """
    
    def __init__(self):
        """Initialize Mac audio recorder."""
        # Audio parameters
        self.sample_rate = 48000  # macOS voice isolation prefers 48kHz
        self.channels = 1
        # Silero VAD requires 32ms chunks (512 samples at 16kHz)
        self.chunk_duration_ms = 32  # 32ms chunks to match Silero VAD
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Calculate buffer size for our chunk duration (32ms to match Silero VAD)
        # This will be recalculated by voice isolation for the actual sample rate
        initial_buffer_size = int(self.sample_rate * self.chunk_duration_ms / 1000)

        # Initialize voice isolation with proper config
        # It will detect and use the device's native format
        vi_config = VoiceIsolationConfig(
            sample_rate=self.sample_rate,  # Initial preference (will be overridden)
            channels=self.channels,
            quality=kAUVoiceIOProperty_VoiceProcessingQuality_High,
            enable_agc=True,
            enable_noise_suppression=True,
            buffer_size=initial_buffer_size  # Will be recalculated based on actual sample rate
        )
        self.voice_isolation = MacVoiceIsolation(vi_config)

        # Update sample rate to match what voice isolation actually uses
        # (it will use the device's native format, e.g., 44100 Hz)
        self.voice_isolation_sample_rate = self.voice_isolation.config.sample_rate

        # Use the buffer size calculated by voice isolation (maintains same duration)
        self.voice_isolation_chunk_size = self.voice_isolation.config.buffer_size

        logging.info(
            f"Voice isolation using {self.voice_isolation_sample_rate} Hz, "
            f"chunk size: {self.voice_isolation_chunk_size} samples"
        )

        # Silero VAD only supports 8000 or 16000 Hz
        # We'll use 16000 Hz and resample if needed
        self.vad_sample_rate = 16000
        self.vad = IntelligentVAD(
            sample_rate=self.vad_sample_rate,
            chunk_duration_ms=self.chunk_duration_ms
        )

        # Calculate VAD chunk size (Silero requires exactly 512 samples at 16kHz)
        self.vad_chunk_size = 512  # Fixed for Silero VAD at 16kHz

        # Buffer for accumulating resampled audio before feeding to VAD
        self.vad_buffer = np.array([], dtype=np.float32)
        
        # Initialize utterance detector
        self.utterance_detector = IntelligentUtteranceDetector()
        
        # Pre-speech buffer (500ms)
        self.pre_speech_buffer_size = int(0.5 * self.sample_rate / self.chunk_size)
        
        # Recording state
        self.recording_active = False
        self.recording_thread = None
        
        logging.info("Mac audio recorder initialized with voice isolation")
    
    def record_audio(self,
                    file_path: str,
                    timeout: float = 10.0,
                    phrase_time_limit: Optional[float] = None,
                    is_wake_word_response: bool = False) -> bool:
        """
        Record audio with voice isolation.
        
        Args:
            file_path: Path to save audio file
            timeout: Maximum time to wait for speech
            phrase_time_limit: Maximum recording duration
            is_wake_word_response: Whether this follows wake word
            
        Returns:
            bool: Success status
        """
        # Reset components
        self.vad.reset()
        self.vad_buffer = np.array([], dtype=np.float32)  # Clear VAD buffer
        context = UtteranceContext()
        
        # Audio buffers
        pre_speech_buffer = deque(maxlen=self.pre_speech_buffer_size)
        audio_buffer = []
        
        # State tracking
        recording_started = False
        speech_detected = False
        start_time = time.time()
        
        # Collected audio for this recording
        self.current_recording = []
        self.recording_active = True
        
        # Define callback for voice isolation
        def audio_callback(audio_chunk):
            """Callback from voice isolation with processed audio."""
            if not self.recording_active:
                return
            
            # Add to current recording buffer
            self.current_recording.append(audio_chunk)
        
        # Set the callback
        self.voice_isolation.set_process_callback(audio_callback)
        
        try:
            # Start voice isolation
            self.voice_isolation.start()
            
            logging.info("Mac voice isolation recording started...")
            
            # Recording loop
            while self.recording_active:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check timeout
                if not speech_detected and elapsed > timeout:
                    logging.warning(f"No speech detected within {timeout}s")
                    break
                
                # Get processed audio from voice isolation
                processed_audio = self.voice_isolation.read_processed_audio(timeout=0.1)
                
                if processed_audio is None:
                    continue
                
                # Resample for VAD if needed (Silero only supports 8k or 16k Hz)
                if self.voice_isolation_sample_rate != self.vad_sample_rate:
                    # Simple linear resampling
                    num_samples_out = int(len(processed_audio) * self.vad_sample_rate / self.voice_isolation_sample_rate)
                    resampled_audio = np.interp(
                        np.linspace(0, len(processed_audio), num_samples_out),
                        np.arange(len(processed_audio)),
                        processed_audio
                    ).astype(np.float32)
                else:
                    resampled_audio = processed_audio

                # Always add original (non-resampled) to pre-speech buffer
                pre_speech_buffer.append(processed_audio)

                # Add resampled audio to buffer
                self.vad_buffer = np.concatenate([self.vad_buffer, resampled_audio])

                # Process all complete 512-sample chunks in the buffer
                # Track if any chunk detected speech
                chunk_has_speech = False
                latest_confidence = 0.0

                while len(self.vad_buffer) >= self.vad_chunk_size:
                    # Extract exactly 512 samples
                    chunk = self.vad_buffer[:self.vad_chunk_size]
                    self.vad_buffer = self.vad_buffer[self.vad_chunk_size:]

                    # Detect speech using VAD
                    is_speech, confidence = self.vad.process_chunk(chunk)

                    if is_speech:
                        chunk_has_speech = True
                        latest_confidence = confidence

                # Handle speech detection (if any chunk had speech)
                if chunk_has_speech and not recording_started:
                    logging.info(f"Speech detected (confidence: {latest_confidence:.2f})")
                    recording_started = True
                    speech_detected = True
                    
                    # Add pre-speech buffer
                    audio_buffer.extend(pre_speech_buffer)
                    audio_buffer.append(processed_audio)
                    
                    context.update_speech_detected()
                    
                elif recording_started:
                    # Add to recording buffer
                    audio_buffer.append(processed_audio)

                    # Update context
                    context.total_speech_duration = (
                        len(audio_buffer) * self.chunk_duration_ms / 1000
                    )

                    # Check for utterance end using latest VAD results
                    should_end, reason = self.utterance_detector.should_end_utterance(
                        context, chunk_has_speech, latest_confidence
                    )
                    
                    if should_end:
                        logging.info(f"Utterance ended: {reason}")
                        break
                    
                    # Check time limit
                    if (phrase_time_limit and 
                        context.total_speech_duration > phrase_time_limit):
                        logging.info(f"Reached time limit of {phrase_time_limit}s")
                        break
            
            # Stop recording
            self.recording_active = False
            self.voice_isolation.stop()
            
            # Save recording
            if audio_buffer:
                return self._save_recording(audio_buffer, file_path, 
                                          is_wake_word_response)
            else:
                logging.warning("No audio recorded")
                return False
                
        except Exception as e:
            logging.error(f"Error during recording: {e}")
            self.recording_active = False
            return False
            
        finally:
            self.voice_isolation.stop()
    
    def _save_recording(self,
                       audio_buffer: list,
                       file_path: str,
                       is_wake_word_response: bool) -> bool:
        """Save recorded audio to file."""
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(audio_buffer)
            
            # Apply wake word trimming if needed
            if is_wake_word_response:
                # Trim first 500ms (using voice isolation sample rate)
                trim_samples = int(0.5 * self.voice_isolation_sample_rate)
                if len(audio_data) > trim_samples:
                    audio_data = audio_data[trim_samples:]
                    logging.info("Trimmed activation sound")

            # Convert float32 to int16
            audio_int16 = (audio_data * 32768).astype(np.int16)

            # Save as WAV first
            wav_path = file_path.replace('.mp3', '.wav')
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.voice_isolation_sample_rate)
                wf.writeframes(audio_int16.tobytes())

            # Convert to MP3 if requested
            if file_path.endswith('.mp3'):
                # Resample to 16kHz for compatibility if needed
                audio_segment = AudioSegment.from_wav(wav_path)
                if self.voice_isolation_sample_rate != 16000:
                    audio_segment = audio_segment.set_frame_rate(16000)
                audio_segment.export(file_path, format="mp3", bitrate="128k")
                os.remove(wav_path)

            duration = len(audio_data) / self.voice_isolation_sample_rate
            logging.info(f"Recording saved: {file_path} (duration: {duration:.2f}s)")
            
            # Log voice isolation statistics
            stats = self.voice_isolation.get_statistics()
            logging.debug(f"Voice isolation stats: {stats}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving recording: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.voice_isolation:
            self.voice_isolation.cleanup()

# Global recorder instance
_mac_recorder_instance = None

def get_mac_recorder() -> Optional[MacAudioRecorder]:
    """Get or create the global Mac audio recorder instance."""
    global _mac_recorder_instance
    
    if _mac_recorder_instance is None:
        try:
            _mac_recorder_instance = MacAudioRecorder()
            logging.info("Mac audio recorder with voice isolation initialized")
        except Exception as e:
            logging.error(f"Failed to initialize Mac audio recorder: {e}")
            return None
    
    return _mac_recorder_instance

def record_audio_mac(file_path: str,
                     timeout: float = 10,
                     phrase_time_limit: Optional[float] = None,
                     is_wake_word_response: bool = False) -> bool:
    """
    Drop-in replacement for record_audio using Mac voice isolation.
    
    Args:
        file_path: Path to save the audio file
        timeout: Maximum time to wait for speech
        phrase_time_limit: Maximum recording duration
        is_wake_word_response: Whether this follows wake word detection
        
    Returns:
        bool: True if recording was successful
    """
    recorder = get_mac_recorder()
    if not recorder:
        return False
    
    return recorder.record_audio(
        file_path=file_path,
        timeout=timeout,
        phrase_time_limit=phrase_time_limit,
        is_wake_word_response=is_wake_word_response
    )