# voice_assistant/enhanced_audio.py

import os
import time
import logging
import wave
import pyaudio
import numpy as np
from collections import deque
from typing import Optional, Tuple
from pydub import AudioSegment

from .intelligent_vad import IntelligentVAD
from .utterance_detector import IntelligentUtteranceDetector, UtteranceContext
from .config import Config

class EnhancedAudioRecorder:
    """
    Enhanced audio recorder with intelligent voice activity detection.
    
    This recorder solves the problems of missing speech beginnings and
    premature cutoffs by using neural network VAD and intelligent
    utterance boundary detection.
    """
    
    def __init__(self):
        """Initialize the enhanced audio recorder."""
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.format = pyaudio.paInt16
        # Silero VAD requires exactly 512 samples for 16kHz (32ms)
        self.chunk_size = 512  # Fixed size for Silero VAD
        self.chunk_duration_ms = 32  # 512 samples at 16kHz = 32ms
        
        # Initialize components
        self.vad = IntelligentVAD(
            sample_rate=self.sample_rate,
            chunk_duration_ms=32  # Silero VAD requires 32ms chunks
        )
        self.utterance_detector = IntelligentUtteranceDetector()
        
        # Pre-speech buffer (500ms) to capture speech beginnings
        self.pre_speech_buffer_size = int(0.5 * self.sample_rate / self.chunk_size)
        
        # Audio interface
        self.pyaudio = pyaudio.PyAudio()
        
    def record_audio(self, 
                    file_path: str,
                    timeout: float = 10.0,
                    phrase_time_limit: Optional[float] = None,
                    is_wake_word_response: bool = False) -> bool:
        """
        Record audio with intelligent VAD and save to file.
        
        Args:
            file_path: Path to save the audio file
            timeout: Maximum time to wait for speech to start
            phrase_time_limit: Maximum duration of recording
            is_wake_word_response: Whether this is right after wake word detection
            
        Returns:
            bool: True if audio was successfully recorded
        """
        # Reset VAD for new recording
        self.vad.reset()
        
        # Initialize context
        context = UtteranceContext()
        
        # Audio buffers
        pre_speech_buffer = deque(maxlen=self.pre_speech_buffer_size)
        audio_buffer = []
        chunk_accumulator = b''  # Buffer to accumulate partial chunks
        
        # State tracking
        recording_started = False
        speech_detected = False
        start_time = time.time()
        
        # Open audio stream
        stream = None
        try:
            # Use a larger buffer to reduce the chance of underruns
            buffer_size = max(self.chunk_size * 4, 2048)  # At least 4x chunk size or 2048 frames
            
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=buffer_size,
                stream_callback=None  # Use blocking mode
            )
            
            logging.info("Enhanced recording started - listening for speech...")
            
            # Give the audio stream a moment to stabilize
            time.sleep(0.1)
            
            # Prime the stream by reading and discarding a few chunks
            for _ in range(3):
                try:
                    stream.read(self.chunk_size, exception_on_overflow=False)
                except:
                    pass
            
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check initial timeout
                if not speech_detected and elapsed > timeout:
                    logging.warning(f"No speech detected within {timeout}s timeout")
                    break
                
                # Read audio chunk with accumulator for handling partial reads
                try:
                    # Read audio data - always read full chunk size to avoid partial reads
                    # Silero VAD needs exactly 512 samples
                    try:
                        audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    except Exception as e:
                        logging.debug(f"Stream read error: {e}")
                        continue
                    
                    # Accumulate data if we got a partial read
                    chunk_accumulator += audio_data
                    
                    # Check if we have enough data for a full chunk
                    bytes_needed = self.chunk_size * 2  # 2 bytes per int16 sample
                    
                    if len(chunk_accumulator) < bytes_needed:
                        # Not enough data yet, continue accumulating
                        continue
                    
                    # Extract a full chunk worth of data
                    audio_data = chunk_accumulator[:bytes_needed]
                    chunk_accumulator = chunk_accumulator[bytes_needed:]  # Keep any extra for next time
                    
                except Exception as e:
                    logging.error(f"Error reading audio: {e}")
                    continue
                
                # Convert to numpy array for processing
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Ensure we have exactly the right number of samples
                if len(audio_array) != self.chunk_size:
                    logging.warning(f"Unexpected chunk size: {len(audio_array)} samples (expected {self.chunk_size})")
                    if len(audio_array) < self.chunk_size:
                        # Pad with zeros if still too short
                        audio_array = np.pad(audio_array, (0, self.chunk_size - len(audio_array)), mode='constant')
                    else:
                        # Truncate if too long
                        audio_array = audio_array[:self.chunk_size]
                
                # Debug: Check audio levels periodically
                if self.vad.detection_count % 30 == 0:  # Every ~1 second
                    max_val = np.max(np.abs(audio_array))
                    rms_val = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
                    logging.debug(f"Audio levels - Max: {max_val}, RMS: {rms_val:.1f}")
                
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Always add to pre-speech buffer
                pre_speech_buffer.append(audio_data)
                
                # Detect speech using intelligent VAD
                try:
                    is_speech, confidence = self.vad.process_chunk(audio_float)
                except Exception as e:
                    logging.error(f"VAD processing error: {e}")
                    # On VAD error, assume no speech to continue recording
                    is_speech = False
                    confidence = 0.0
                
                # Handle speech detection
                if is_speech and not recording_started:
                    logging.info(f"Speech detected with confidence {confidence:.2f}")
                    recording_started = True
                    speech_detected = True
                    
                    # Add pre-speech buffer to capture beginning
                    audio_buffer.extend(pre_speech_buffer)
                    audio_buffer.append(audio_data)
                    
                    # Update context
                    context.update_speech_detected()
                    context.total_speech_duration = len(audio_buffer) * self.chunk_duration_ms / 1000
                    
                elif recording_started:
                    # Add to recording buffer
                    audio_buffer.append(audio_data)
                    
                    # Update total speech duration
                    context.total_speech_duration = len(audio_buffer) * self.chunk_duration_ms / 1000
                    
                    # Check for utterance end
                    should_end, reason = self.utterance_detector.should_end_utterance(
                        context, is_speech, confidence
                    )
                    
                    if should_end:
                        logging.info(f"Utterance ended: {reason}")
                        break
                    
                    # Check phrase time limit
                    if phrase_time_limit and context.total_speech_duration > phrase_time_limit:
                        logging.info(f"Reached phrase time limit of {phrase_time_limit}s")
                        break
            
            # Process and save recording
            if audio_buffer:
                return self._save_recording(audio_buffer, file_path, is_wake_word_response)
            else:
                logging.warning("No audio recorded")
                return False
                
        except Exception as e:
            logging.error(f"Error during recording: {e}")
            return False
            
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def _save_recording(self, 
                       audio_buffer: list,
                       file_path: str,
                       is_wake_word_response: bool) -> bool:
        """Save recorded audio to file with optional post-processing."""
        try:
            # Combine audio chunks
            audio_data = b''.join(audio_buffer)
            
            # Apply wake word trimming if needed
            if is_wake_word_response:
                # Trim first 500ms to remove activation sound
                trim_samples = int(0.5 * self.sample_rate)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                if len(audio_array) > trim_samples:
                    audio_array = audio_array[trim_samples:]
                    audio_data = audio_array.tobytes()
                    logging.info("Trimmed activation sound from recording")
            
            # Save as WAV first
            wav_path = file_path.replace('.mp3', '.wav')
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            # Convert to MP3 if requested
            if file_path.endswith('.mp3'):
                audio_segment = AudioSegment.from_wav(wav_path)
                audio_segment.export(file_path, format="mp3", bitrate="128k")
                os.remove(wav_path)
            
            # Log statistics
            duration = len(audio_data) / (self.sample_rate * 2)  # 2 bytes per sample
            logging.info(f"Recording saved: {file_path} (duration: {duration:.2f}s)")
            
            # Update detector statistics
            vad_stats = self.vad.get_performance_stats()
            logging.debug(f"VAD performance: {vad_stats}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving recording: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pyaudio'):
            self.pyaudio.terminate()

# Global recorder instance
_recorder_instance = None

def get_enhanced_recorder() -> EnhancedAudioRecorder:
    """Get or create the global enhanced recorder instance."""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = EnhancedAudioRecorder()
    return _recorder_instance

def record_audio_enhanced(file_path: str, 
                         timeout: float = 10,
                         phrase_time_limit: Optional[float] = None,
                         **kwargs) -> bool:
    """
    Drop-in replacement for the existing record_audio function.
    
    This function provides the same interface but uses intelligent VAD.
    """
    recorder = get_enhanced_recorder()
    return recorder.record_audio(
        file_path=file_path,
        timeout=timeout,
        phrase_time_limit=phrase_time_limit,
        is_wake_word_response=kwargs.get('is_wake_word_response', False)
    )