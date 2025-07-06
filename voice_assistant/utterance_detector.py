# voice_assistant/utterance_detector.py

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from collections import deque

@dataclass
class UtteranceContext:
    """
    Tracks context for intelligent utterance boundary detection.
    
    This context object maintains the state needed to make intelligent
    decisions about when a user has finished speaking.
    """
    energy_history: deque = field(default_factory=lambda: deque(maxlen=100))
    silence_start_time: Optional[float] = None
    last_speech_time: float = field(default_factory=time.time)
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)
    partial_transcript: str = ""
    consecutive_silence_chunks: int = 0
    total_speech_duration: float = 0.0
    
    def update_speech_detected(self):
        """Update context when speech is detected."""
        self.last_speech_time = time.time()
        self.silence_start_time = None
        self.consecutive_silence_chunks = 0
    
    def update_silence_detected(self):
        """Update context when silence is detected."""
        if self.silence_start_time is None:
            self.silence_start_time = time.time()
        self.consecutive_silence_chunks += 1
    
    @property
    def current_silence_duration(self) -> float:
        """Get current silence duration in seconds."""
        if self.silence_start_time is None:
            return 0.0
        return time.time() - self.silence_start_time

class IntelligentUtteranceDetector:
    """
    Detects utterance boundaries using multiple signals for natural conversation flow.
    
    This detector goes beyond simple pause detection by considering:
    - Speech patterns and rhythm
    - Linguistic completeness
    - Contextual cues
    - User-specific adaptation
    """
    
    def __init__(self, 
                 min_utterance_duration: float = 0.5,
                 max_initial_silence: float = 10.0,
                 min_final_silence: float = 0.8,
                 max_final_silence: float = 2.0):
        """
        Initialize the utterance detector.
        
        Args:
            min_utterance_duration: Minimum duration to consider valid speech
            max_initial_silence: Maximum silence before any speech detected
            min_final_silence: Minimum silence after speech to end utterance
            max_final_silence: Maximum silence to wait before forcing end
        """
        # Core parameters
        self.min_utterance_duration = min_utterance_duration
        self.max_initial_silence = max_initial_silence
        self.min_final_silence = min_final_silence
        self.max_final_silence = max_final_silence
        
        # Linguistic analysis
        self.sentence_endings = {'.', '!', '?'}
        self.weak_boundaries = {',', ';', ':', '—'}
        self.filler_words = {
            'um', 'uh', 'hmm', 'well', 'so', 'like',
            'you know', 'i mean', 'actually', 'basically'
        }
        
        # Adaptive parameters
        self.question_pause_factor = 0.7  # Shorter pause after questions
        self.incomplete_pause_factor = 1.5  # Longer pause for incomplete sentences
        self.filler_pause_factor = 1.8  # Even longer after fillers
        
        # Statistics for adaptation
        self.utterance_history = deque(maxlen=10)
        
    def should_end_utterance(self, 
                           context: UtteranceContext,
                           is_speech_detected: bool,
                           confidence: float = 1.0) -> Tuple[bool, str]:
        """
        Determine if the current utterance should end.
        
        Args:
            context: Current utterance context
            is_speech_detected: Whether speech is detected in current chunk
            confidence: VAD confidence score
            
        Returns:
            Tuple of (should_end: bool, reason: str)
        """
        # Update context based on current detection
        if is_speech_detected:
            context.update_speech_detected()
        else:
            context.update_silence_detected()
        
        # Case 1: No speech detected yet
        if not context.speech_segments and context.total_speech_duration == 0:
            if context.current_silence_duration > self.max_initial_silence:
                return True, "max_initial_silence_exceeded"
            return False, "waiting_for_speech"
        
        # Case 2: Currently speaking
        if is_speech_detected:
            return False, "speech_ongoing"
        
        # Case 3: In silence after speech
        silence_duration = context.current_silence_duration
        
        # Check for timeout
        if silence_duration > self.max_final_silence:
            return True, "max_final_silence_exceeded"
        
        # Calculate dynamic threshold based on context
        threshold = self._calculate_silence_threshold(context)
        
        if silence_duration >= threshold:
            reason = self._get_end_reason(context, threshold)
            return True, reason
        
        return False, f"silence_too_short_{silence_duration:.2f}s"
    
    def _calculate_silence_threshold(self, context: UtteranceContext) -> float:
        """
        Calculate dynamic silence threshold based on linguistic context.
        
        This is where the intelligence happens - we adjust the pause threshold
        based on what the user has said and how they said it.
        """
        base_threshold = self.min_final_silence
        
        # Analyze partial transcript if available
        if context.partial_transcript:
            text = context.partial_transcript.strip().lower()
            
            # Check for question
            if text.endswith('?'):
                return base_threshold * self.question_pause_factor
            
            # Check for incomplete sentence
            last_char = text[-1] if text else ''
            if last_char in self.weak_boundaries:
                return base_threshold * self.incomplete_pause_factor
            
            # Check for filler words at the end
            words = text.split()
            if words:
                last_phrase = ' '.join(words[-2:])  # Last two words
                for filler in self.filler_words:
                    if last_phrase.endswith(filler):
                        return base_threshold * self.filler_pause_factor
            
            # Check for complete sentence
            if last_char in self.sentence_endings:
                return base_threshold
        
        # Default: slightly longer than base to be safe
        return base_threshold * 1.2
    
    def _get_end_reason(self, context: UtteranceContext, threshold: float) -> str:
        """Generate a descriptive reason for ending the utterance."""
        if context.partial_transcript:
            if context.partial_transcript.strip().endswith('?'):
                return "question_completed"
            elif context.partial_transcript.strip()[-1:] in self.sentence_endings:
                return "sentence_completed"
            elif any(context.partial_transcript.lower().endswith(f) for f in self.filler_words):
                return "filler_timeout"
        
        return f"silence_threshold_met_{threshold:.2f}s"
    
    def update_statistics(self, utterance_duration: float, silence_duration: float):
        """Update statistics for adaptive behavior."""
        self.utterance_history.append({
            'duration': utterance_duration,
            'final_silence': silence_duration,
            'timestamp': time.time()
        })