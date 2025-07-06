# voice_assistant/intelligent_vad.py

import torch
import numpy as np
import logging
from collections import deque
import time
from typing import Tuple, List, Optional

class IntelligentVAD:
    """
    Intelligent Voice Activity Detection using Silero VAD model.
    
    This class provides neural network-based voice activity detection that is
    significantly more accurate than energy-based methods. It can detect speech
    in various conditions including:
    - Low volume speech and whispers
    - High background noise environments
    - Multiple speakers
    - Various accents and speaking styles
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_duration_ms: int = 32):
        """
        Initialize the Intelligent VAD system.
        
        Args:
            sample_rate: Audio sampling rate in Hz (default: 16000)
            chunk_duration_ms: Duration of each audio chunk in milliseconds (default: 30)
        """
        logging.info("Initializing Intelligent VAD system...")
        
        # Load the Silero VAD model
        # This will download the model on first use (about 1.5MB)
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True  # Use ONNX for better performance
            )
            logging.info("Silero VAD model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load Silero VAD model: {e}")
            raise
        
        # Extract utility functions
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
        
        # Configuration
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        # Silero VAD requires exactly 512 samples for 16kHz (32ms) or 256 for 8kHz
        if sample_rate == 16000:
            self.chunk_size = 512  # Exactly 512 samples for 16kHz
            self.chunk_duration_ms = 32  # Override to exact duration
        elif sample_rate == 8000:
            self.chunk_size = 256  # Exactly 256 samples for 8kHz
            self.chunk_duration_ms = 32  # Override to exact duration
        else:
            raise ValueError(f"Silero VAD only supports 8000 or 16000 Hz, got {sample_rate}")
        
        # Create VAD iterator for streaming detection
        self.vad_iterator = self._create_vad_iterator()
        
        # Performance monitoring
        self.detection_count = 0
        self.total_processing_time = 0.0
        
        # Speech state tracking
        self._is_speaking = False
        
    def _create_vad_iterator(self):
        """Create a configured VAD iterator for streaming audio processing."""
        return self.VADIterator(
            self.model,
            threshold=0.3,  # Lower threshold for better sensitivity
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=300,  # Shorter silence duration
            speech_pad_ms=100  # Less padding for quicker response
        )
    
    def reset(self):
        """Reset the VAD iterator for a new audio stream."""
        self.vad_iterator = self._create_vad_iterator()
        self._is_speaking = False
        logging.debug("VAD iterator reset")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Process a single audio chunk and return speech detection result.
        
        Args:
            audio_chunk: Audio data as numpy array (float32, -1 to 1)
            
        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        start_time = time.time()
        
        try:
            # Ensure audio is in correct format
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize if needed (int16 to float32)
            if np.abs(audio_chunk).max() > 1.0:
                audio_chunk = audio_chunk / 32768.0
            
            # Ensure we have the right number of samples
            if len(audio_chunk) != self.chunk_size:
                logging.warning(f"Chunk size mismatch: got {len(audio_chunk)}, expected {self.chunk_size}")
                if len(audio_chunk) < self.chunk_size:
                    # Pad with zeros
                    audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)), mode='constant')
                else:
                    # Truncate
                    audio_chunk = audio_chunk[:self.chunk_size]
            
            # Convert to torch tensor for VAD iterator
            audio_tensor = torch.from_numpy(audio_chunk).float()
            
            # Get VAD prediction
            speech_dict = self.vad_iterator(audio_tensor)
            
            # Handle the iterator's output format
            # It returns {'start': sample} when speech starts, {'end': sample} when it ends
            if speech_dict:
                if 'start' in speech_dict:
                    self._is_speaking = True
                    logging.debug(f"Speech started at sample {speech_dict['start']}")
                elif 'end' in speech_dict:
                    self._is_speaking = False
                    logging.debug(f"Speech ended at sample {speech_dict['end']}")
            
            is_speech = getattr(self, '_is_speaking', False)
            
            # Get confidence from the model directly for logging
            self.model.reset_states()
            confidence = self.model(audio_tensor, self.sample_rate).item()
            
            # Update performance metrics
            self.detection_count += 1
            self.total_processing_time += (time.time() - start_time)
            
            return is_speech, confidence
            
        except Exception as e:
            logging.error(f"Error in VAD processing: {e}")
            # Return safe defaults on error
            return False, 0.0
    
    def get_speech_segments(self, audio_data: np.ndarray) -> List[dict]:
        """
        Detect all speech segments in a complete audio recording.
        
        Args:
            audio_data: Complete audio data as numpy array
            
        Returns:
            List of speech segments with start/end timestamps
        """
        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data)
        else:
            audio_tensor = audio_data
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_silence_duration_ms=600,
            speech_pad_ms=300
        )
        
        return speech_timestamps
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics for monitoring."""
        avg_time = (self.total_processing_time / self.detection_count 
                   if self.detection_count > 0 else 0)
        
        return {
            'detections': self.detection_count,
            'total_time': self.total_processing_time,
            'avg_time_per_chunk': avg_time,
            'real_time_factor': avg_time / (self.chunk_duration_ms / 1000)
        }