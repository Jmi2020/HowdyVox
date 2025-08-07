#!/usr/bin/env python3

import logging
import time
import numpy as np
from typing import Optional, Dict, List, NamedTuple, Callable
from collections import deque
from enum import Enum
from dataclasses import dataclass
import threading

from .esp32_p4_protocol import ESP32P4PacketInfo, ESP32P4ProtocolParser, ESP32P4VADFlags, ESP32P4WakeWordFlags
from .intelligent_vad import IntelligentVAD

class VADDecision(Enum):
    """VAD decision types for coordination."""
    NO_SPEECH = "no_speech"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    WAKE_WORD_DETECTED = "wake_word_detected"
    WAKE_WORD_END = "wake_word_end"
    UNCERTAIN = "uncertain"

@dataclass
class VADResult:
    """Unified VAD result combining edge and server decisions."""
    decision: VADDecision
    confidence: float
    edge_confidence: Optional[float] = None
    server_confidence: Optional[float] = None
    edge_decision: Optional[bool] = None
    server_decision: Optional[bool] = None
    coordination_method: str = "unknown"
    quality_metrics: Optional[Dict] = None
    wake_word_info: Optional[Dict] = None  # Wake word detection details

class VADFusionStrategy(Enum):
    """VAD fusion strategies for combining edge and server decisions."""
    EDGE_PRIORITY = "edge_priority"        # Prefer edge VAD when available
    SERVER_PRIORITY = "server_priority"    # Prefer server VAD
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by confidence
    MAJORITY_VOTE = "majority_vote"       # Simple majority
    ADAPTIVE = "adaptive"                 # Adapt based on performance

@dataclass
class VADPerformanceMetrics:
    """Performance tracking for VAD decisions."""
    total_decisions: int = 0
    edge_decisions: int = 0
    server_decisions: int = 0
    coordinated_decisions: int = 0
    agreement_count: int = 0
    disagreement_count: int = 0
    false_positive_feedback: int = 0
    false_negative_feedback: int = 0
    wake_word_detections: int = 0
    wake_word_validations: int = 0
    wake_word_rejections: int = 0
    avg_edge_confidence: float = 0.0
    avg_server_confidence: float = 0.0
    avg_wake_word_confidence: float = 0.0
    avg_processing_time: float = 0.0

class ESP32P4VADCoordinator:
    """
    Advanced VAD coordinator that fuses ESP32-P4 edge VAD with server-side Silero VAD.
    
    This coordinator implements sophisticated fusion logic to combine:
    - Edge VAD: Fast, low-latency decisions from ESP32-P4 hardware
    - Server VAD: High-accuracy Silero neural network decisions
    
    Features:
    - Multiple fusion strategies (edge priority, confidence weighted, adaptive)
    - Performance tracking and feedback learning
    - Real-time adaptation based on accuracy metrics
    - Speech boundary detection enhancement
    - Noise robustness through dual VAD validation
    """
    
    def __init__(self, 
                 server_vad: IntelligentVAD,
                 fusion_strategy: VADFusionStrategy = VADFusionStrategy.ADAPTIVE,
                 edge_timeout_ms: float = 100.0,
                 wake_word_callback: Optional[Callable] = None):
        """
        Initialize VAD coordinator.
        
        Args:
            server_vad: Server-side Silero VAD instance
            fusion_strategy: Strategy for combining edge and server decisions
            edge_timeout_ms: Timeout for edge VAD decisions
            wake_word_callback: Callback function when wake word is detected
        """
        self.server_vad = server_vad
        self.fusion_strategy = fusion_strategy
        self.edge_timeout_ms = edge_timeout_ms
        self.wake_word_callback = wake_word_callback
        
        # Protocol parser for ESP32-P4 packets
        self.protocol_parser = ESP32P4ProtocolParser()
        
        # VAD state tracking
        self.current_state = VADDecision.NO_SPEECH
        self.last_decision_time = time.time()
        self.speech_start_time: Optional[float] = None
        
        # Edge VAD state tracking per device
        self.device_states: Dict[str, Dict] = {}
        
        # Performance tracking
        self.metrics = VADPerformanceMetrics()
        self.decision_history = deque(maxlen=1000)  # Last 1000 decisions
        
        # Adaptive strategy parameters
        self.edge_weight = 0.7  # Initial edge weight for adaptive strategy
        self.server_weight = 0.3  # Initial server weight
        self.adaptation_rate = 0.01  # Learning rate for weight updates
        
        # Speech boundary detection
        self.boundary_detection_window = deque(maxlen=5)  # 5-frame window
        self.min_speech_duration_ms = 200  # Minimum speech duration
        self.min_silence_duration_ms = 300  # Minimum silence duration
        
        # Threading for performance monitoring
        self.performance_lock = threading.RLock()
        
        logging.info(f"ESP32-P4 VAD Coordinator initialized with {fusion_strategy.value} strategy")
    
    def process_packet(self, 
                      packet_info: ESP32P4PacketInfo,
                      audio_chunk: np.ndarray) -> VADResult:
        """
        Process ESP32-P4 packet and coordinate VAD decisions.
        
        Args:
            packet_info: Parsed ESP32-P4 packet information
            audio_chunk: Audio data for server VAD processing
            
        Returns:
            VADResult with coordinated decision
        """
        start_time = time.time()
        
        # Extract edge VAD information if available
        edge_vad_info = self._extract_edge_vad(packet_info)
        
        # Extract wake word information if available
        wake_word_info = self._extract_wake_word_info(packet_info)
        
        # Run server VAD
        server_decision, server_confidence = self.server_vad.process_chunk(audio_chunk)
        
        # Coordinate decisions
        vad_result = self._coordinate_decisions(
            edge_vad_info, 
            server_decision, 
            server_confidence,
            packet_info.source_addr,
            wake_word_info
        )
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_metrics(vad_result, processing_time)
        
        # Update state tracking
        self._update_state(vad_result)
        
        # Store decision for analysis
        self.decision_history.append({
            'timestamp': time.time(),
            'result': vad_result,
            'processing_time': processing_time
        })
        
        return vad_result
    
    def _extract_edge_vad(self, packet_info: ESP32P4PacketInfo) -> Optional[Dict]:
        """Extract edge VAD information from ESP32-P4 packet."""
        if not self.protocol_parser.is_enhanced_packet(packet_info):
            return None
        
        vad_state = self.protocol_parser.get_vad_state(packet_info)
        device_id = f"{packet_info.source_addr[0]}:{packet_info.source_addr[1]}"
        
        # Update device state tracking
        if device_id not in self.device_states:
            self.device_states[device_id] = {
                'last_update': time.time(),
                'speech_start_time': None,
                'consecutive_speech': 0,
                'consecutive_silence': 0
            }
        
        device_state = self.device_states[device_id]
        device_state['last_update'] = time.time()
        
        # Track speech/silence runs for boundary detection
        if vad_state['voice_active']:
            device_state['consecutive_speech'] += 1
            device_state['consecutive_silence'] = 0
            if device_state['speech_start_time'] is None:
                device_state['speech_start_time'] = time.time()
        else:
            device_state['consecutive_silence'] += 1
            device_state['consecutive_speech'] = 0
            device_state['speech_start_time'] = None
        
        return {
            'has_vad': True,
            'voice_active': vad_state['voice_active'],
            'speech_start': vad_state['speech_start'],
            'speech_end': vad_state['speech_end'],
            'confidence': vad_state['confidence'],
            'quality': vad_state['quality'],
            'high_confidence': vad_state['high_confidence'],
            'device_id': device_id,
            'device_state': device_state,
            'audio_metrics': {
                'max_amplitude': vad_state['max_amplitude'],
                'noise_floor': vad_state['noise_floor'],
                'snr_db': vad_state['snr_db'],
                'zero_crossing_rate': vad_state['zero_crossing_rate']
            }
        }
    
    def _extract_wake_word_info(self, packet_info: ESP32P4PacketInfo) -> Optional[Dict]:
        """Extract wake word information from ESP32-P4 packet."""
        if not self.protocol_parser.is_wake_word_packet(packet_info):
            return None
        
        wake_word_state = self.protocol_parser.get_wake_word_state(packet_info)
        device_id = f"{packet_info.source_addr[0]}:{packet_info.source_addr[1]}"
        
        # Handle wake word detected event
        if wake_word_state['wake_detected'] and self.wake_word_callback:
            wake_word_details = {
                'device_id': device_id,
                'keyword_id': wake_word_state['keyword_id'],
                'confidence': wake_word_state['confidence'],
                'quality': wake_word_state['quality'],
                'detection_start_ms': wake_word_state['detection_start_ms'],
                'detection_duration_ms': wake_word_state['detection_duration_ms'],
                'high_confidence': wake_word_state['high_confidence_wake'],
                'source_addr': packet_info.source_addr
            }
            
            # Call wake word callback in a separate thread to avoid blocking
            threading.Thread(
                target=self.wake_word_callback,
                args=(wake_word_details,),
                daemon=True
            ).start()
            
            logging.info(f"Wake word detected from {device_id} with confidence {wake_word_state['confidence']:.2f}")
        
        return {
            'has_wake_word': True,
            'wake_detected': wake_word_state['wake_detected'],
            'wake_end': wake_word_state['wake_end'],
            'high_confidence_wake': wake_word_state['high_confidence_wake'],
            'confidence': wake_word_state['confidence'],
            'quality': wake_word_state['quality'],
            'keyword_id': wake_word_state['keyword_id'],
            'detection_duration_ms': wake_word_state['detection_duration_ms'],
            'validated': wake_word_state['wake_validated'],
            'rejected': wake_word_state['wake_rejected'],
            'device_id': device_id
        }
    
    def _coordinate_decisions(self, 
                            edge_vad_info: Optional[Dict],
                            server_decision: bool,
                            server_confidence: float,
                            source_addr: tuple,
                            wake_word_info: Optional[Dict] = None) -> VADResult:
        """Coordinate edge and server VAD decisions using fusion strategy."""
        
        # Check for wake word detection first (highest priority)
        if wake_word_info and wake_word_info['wake_detected']:
            return self._handle_wake_word_detection(wake_word_info, server_decision, server_confidence)
        
        # Handle server-only case
        if not edge_vad_info:
            result = self._server_only_decision(server_decision, server_confidence)
            if wake_word_info:
                result.wake_word_info = wake_word_info
            return result
        
        # Extract edge information
        edge_decision = edge_vad_info['voice_active']
        edge_confidence = edge_vad_info['confidence']
        quality = edge_vad_info['quality']
        
        # Apply fusion strategy
        if self.fusion_strategy == VADFusionStrategy.EDGE_PRIORITY:
            result = self._edge_priority_fusion(
                edge_decision, edge_confidence, 
                server_decision, server_confidence, quality
            )
        elif self.fusion_strategy == VADFusionStrategy.SERVER_PRIORITY:
            result = self._server_priority_fusion(
                edge_decision, edge_confidence,
                server_decision, server_confidence, quality
            )
        elif self.fusion_strategy == VADFusionStrategy.CONFIDENCE_WEIGHTED:
            result = self._confidence_weighted_fusion(
                edge_decision, edge_confidence,
                server_decision, server_confidence, quality
            )
        elif self.fusion_strategy == VADFusionStrategy.MAJORITY_VOTE:
            result = self._majority_vote_fusion(
                edge_decision, edge_confidence,
                server_decision, server_confidence, quality
            )
        elif self.fusion_strategy == VADFusionStrategy.ADAPTIVE:
            result = self._adaptive_fusion(
                edge_decision, edge_confidence,
                server_decision, server_confidence, quality
            )
        else:
            # Fallback to confidence weighted
            result = self._confidence_weighted_fusion(
                edge_decision, edge_confidence,
                server_decision, server_confidence, quality
            )
        
        # Add edge-specific information
        result.edge_decision = edge_decision
        result.edge_confidence = edge_confidence
        result.quality_metrics = edge_vad_info['audio_metrics']
        
        # Add wake word information if available
        if wake_word_info:
            result.wake_word_info = wake_word_info
            # Check for wake word end
            if wake_word_info['wake_end']:
                result.decision = VADDecision.WAKE_WORD_END
        
        # Detect speech boundaries using edge information
        if edge_vad_info['speech_start']:
            result.decision = VADDecision.SPEECH_START
        elif edge_vad_info['speech_end']:
            result.decision = VADDecision.SPEECH_END
        
        return result
    
    def _handle_wake_word_detection(self, 
                                   wake_word_info: Dict, 
                                   server_decision: bool, 
                                   server_confidence: float) -> VADResult:
        """Handle wake word detection with high priority."""
        # Wake word detection overrides normal VAD decisions
        confidence = wake_word_info['confidence']
        
        # Enhance confidence if server VAD also detects speech
        if server_decision:
            confidence = min(1.0, confidence + (server_confidence * 0.2))
        
        return VADResult(
            decision=VADDecision.WAKE_WORD_DETECTED,
            confidence=confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method="wake_word_priority",
            wake_word_info=wake_word_info
        )
    
    def _server_only_decision(self, 
                            server_decision: bool, 
                            server_confidence: float) -> VADResult:
        """Handle server-only VAD decision."""
        decision = VADDecision.SPEECH_DETECTED if server_decision else VADDecision.NO_SPEECH
        
        return VADResult(
            decision=decision,
            confidence=server_confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method="server_only"
        )
    
    def _edge_priority_fusion(self, 
                            edge_decision: bool, edge_confidence: float,
                            server_decision: bool, server_confidence: float,
                            quality: float) -> VADResult:
        """Edge-priority fusion strategy."""
        # Use edge decision if confidence and quality are sufficient
        confidence_threshold = 0.6
        quality_threshold = 0.5
        
        if edge_confidence >= confidence_threshold and quality >= quality_threshold:
            decision = VADDecision.SPEECH_DETECTED if edge_decision else VADDecision.NO_SPEECH
            final_confidence = edge_confidence
            method = "edge_priority_high_quality"
        else:
            # Fall back to server decision
            decision = VADDecision.SPEECH_DETECTED if server_decision else VADDecision.NO_SPEECH
            final_confidence = server_confidence
            method = "edge_priority_server_fallback"
        
        return VADResult(
            decision=decision,
            confidence=final_confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method=method
        )
    
    def _server_priority_fusion(self,
                              edge_decision: bool, edge_confidence: float,
                              server_decision: bool, server_confidence: float,
                              quality: float) -> VADResult:
        """Server-priority fusion strategy."""
        # Use server decision primarily, edge for enhancement
        decision = VADDecision.SPEECH_DETECTED if server_decision else VADDecision.NO_SPEECH
        
        # Enhance confidence using edge information
        if edge_decision == server_decision:
            # Agreement - boost confidence
            final_confidence = min(1.0, server_confidence + (edge_confidence * 0.2))
            method = "server_priority_agreement"
        else:
            # Disagreement - slightly reduce confidence
            final_confidence = max(0.0, server_confidence - 0.1)
            method = "server_priority_disagreement"
        
        return VADResult(
            decision=decision,
            confidence=final_confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method=method
        )
    
    def _confidence_weighted_fusion(self,
                                  edge_decision: bool, edge_confidence: float,
                                  server_decision: bool, server_confidence: float,
                                  quality: float) -> VADResult:
        """Confidence-weighted fusion strategy."""
        # Adjust edge confidence by quality
        adjusted_edge_confidence = edge_confidence * quality
        
        # Weight decisions by confidence
        edge_weight = adjusted_edge_confidence / (adjusted_edge_confidence + server_confidence)
        server_weight = 1.0 - edge_weight
        
        # Weighted vote
        edge_vote = 1.0 if edge_decision else 0.0
        server_vote = 1.0 if server_decision else 0.0
        
        weighted_vote = (edge_vote * edge_weight) + (server_vote * server_weight)
        final_decision = weighted_vote > 0.5
        
        decision = VADDecision.SPEECH_DETECTED if final_decision else VADDecision.NO_SPEECH
        final_confidence = min(1.0, weighted_vote)
        
        return VADResult(
            decision=decision,
            confidence=final_confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method="confidence_weighted"
        )
    
    def _majority_vote_fusion(self,
                            edge_decision: bool, edge_confidence: float,
                            server_decision: bool, server_confidence: float,
                            quality: float) -> VADResult:
        """Majority vote fusion strategy."""
        # Simple majority vote
        votes = [edge_decision, server_decision]
        speech_votes = sum(votes)
        
        if speech_votes > len(votes) / 2:
            decision = VADDecision.SPEECH_DETECTED
            # Average confidence when agreeing
            final_confidence = (edge_confidence + server_confidence) / 2.0
            method = "majority_vote_speech"
        else:
            decision = VADDecision.NO_SPEECH
            # Use inverse of average confidence
            final_confidence = 1.0 - ((edge_confidence + server_confidence) / 2.0)
            method = "majority_vote_silence"
        
        return VADResult(
            decision=decision,
            confidence=final_confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method=method
        )
    
    def _adaptive_fusion(self,
                       edge_decision: bool, edge_confidence: float,
                       server_decision: bool, server_confidence: float,
                       quality: float) -> VADResult:
        """Adaptive fusion strategy that learns from performance."""
        # Adjust edge confidence by quality
        adjusted_edge_confidence = edge_confidence * quality
        
        # Use learned weights
        edge_vote = 1.0 if edge_decision else 0.0
        server_vote = 1.0 if server_decision else 0.0
        
        weighted_vote = ((edge_vote * adjusted_edge_confidence * self.edge_weight) + 
                        (server_vote * server_confidence * self.server_weight))
        
        # Normalize
        total_weight = (adjusted_edge_confidence * self.edge_weight + 
                       server_confidence * self.server_weight)
        
        if total_weight > 0:
            final_confidence = weighted_vote / total_weight
        else:
            final_confidence = 0.5
        
        final_decision = final_confidence > 0.5
        decision = VADDecision.SPEECH_DETECTED if final_decision else VADDecision.NO_SPEECH
        
        return VADResult(
            decision=decision,
            confidence=final_confidence,
            server_decision=server_decision,
            server_confidence=server_confidence,
            coordination_method="adaptive"
        )
    
    def _update_metrics(self, vad_result: VADResult, processing_time: float):
        """Update performance metrics."""
        with self.performance_lock:
            self.metrics.total_decisions += 1
            
            if vad_result.edge_decision is not None:
                self.metrics.edge_decisions += 1
                if vad_result.server_decision is not None:
                    self.metrics.coordinated_decisions += 1
                    # Check agreement
                    if vad_result.edge_decision == vad_result.server_decision:
                        self.metrics.agreement_count += 1
                    else:
                        self.metrics.disagreement_count += 1
            else:
                self.metrics.server_decisions += 1
            
            # Update confidence averages
            if vad_result.edge_confidence is not None:
                self.metrics.avg_edge_confidence = (
                    (self.metrics.avg_edge_confidence * (self.metrics.edge_decisions - 1) + 
                     vad_result.edge_confidence) / self.metrics.edge_decisions
                )
            
            if vad_result.server_confidence is not None:
                server_count = self.metrics.server_decisions + self.metrics.coordinated_decisions
                self.metrics.avg_server_confidence = (
                    (self.metrics.avg_server_confidence * (server_count - 1) + 
                     vad_result.server_confidence) / server_count
                )
            
            # Update processing time
            self.metrics.avg_processing_time = (
                (self.metrics.avg_processing_time * (self.metrics.total_decisions - 1) + 
                 processing_time) / self.metrics.total_decisions
            )
            
            # Update wake word metrics
            if vad_result.wake_word_info:
                if vad_result.decision == VADDecision.WAKE_WORD_DETECTED:
                    self.metrics.wake_word_detections += 1
                
                wake_confidence = vad_result.wake_word_info.get('confidence', 0.0)
                self.metrics.avg_wake_word_confidence = (
                    (self.metrics.avg_wake_word_confidence * (self.metrics.wake_word_detections - 1) + 
                     wake_confidence) / max(1, self.metrics.wake_word_detections)
                )
                
                if vad_result.wake_word_info.get('validated', False):
                    self.metrics.wake_word_validations += 1
                elif vad_result.wake_word_info.get('rejected', False):
                    self.metrics.wake_word_rejections += 1
    
    def _update_state(self, vad_result: VADResult):
        """Update VAD state tracking."""
        current_time = time.time()
        
        # Update speech timing
        if vad_result.decision == VADDecision.SPEECH_START:
            self.speech_start_time = current_time
        elif vad_result.decision == VADDecision.SPEECH_END:
            if self.speech_start_time:
                speech_duration = current_time - self.speech_start_time
                logging.debug(f"Speech duration: {speech_duration:.2f}s")
            self.speech_start_time = None
        
        self.current_state = vad_result.decision
        self.last_decision_time = current_time
    
    def provide_feedback(self, is_correct: bool, decision_timestamp: float):
        """
        Provide feedback on VAD decision accuracy for adaptive learning.
        
        Args:
            is_correct: Whether the decision was correct
            decision_timestamp: Timestamp of the decision to update
        """
        with self.performance_lock:
            # Find the decision in history
            for decision_entry in reversed(self.decision_history):
                if abs(decision_entry['timestamp'] - decision_timestamp) < 0.1:  # 100ms tolerance
                    vad_result = decision_entry['result']
                    
                    if not is_correct:
                        # Update error counts
                        if vad_result.decision in [VADDecision.SPEECH_DETECTED, VADDecision.SPEECH_START]:
                            self.metrics.false_positive_feedback += 1
                        else:
                            self.metrics.false_negative_feedback += 1
                        
                        # Adapt weights for adaptive strategy
                        if self.fusion_strategy == VADFusionStrategy.ADAPTIVE:
                            self._adapt_weights(vad_result, is_correct)
                    
                    break
        
        logging.debug(f"Received feedback: {'correct' if is_correct else 'incorrect'} "
                     f"at {decision_timestamp}")
    
    def _adapt_weights(self, vad_result: VADResult, is_correct: bool):
        """Adapt fusion weights based on feedback."""
        if vad_result.edge_decision is None or vad_result.server_decision is None:
            return
        
        # Determine which VAD was more accurate
        edge_was_better = False
        if not is_correct:
            # If final decision was wrong, check if the alternative would have been right
            if vad_result.decision == VADDecision.SPEECH_DETECTED:
                # We said speech, but it was wrong
                if not vad_result.edge_decision and vad_result.server_decision:
                    edge_was_better = True  # Edge said no speech (correct)
                elif vad_result.edge_decision and not vad_result.server_decision:
                    edge_was_better = False  # Server said no speech (correct)
            else:
                # We said no speech, but it was wrong
                if vad_result.edge_decision and not vad_result.server_decision:
                    edge_was_better = True  # Edge said speech (correct)
                elif not vad_result.edge_decision and vad_result.server_decision:
                    edge_was_better = False  # Server said speech (correct)
        
        # Adjust weights
        if edge_was_better:
            self.edge_weight = min(0.9, self.edge_weight + self.adaptation_rate)
            self.server_weight = 1.0 - self.edge_weight
        else:
            self.server_weight = min(0.9, self.server_weight + self.adaptation_rate)
            self.edge_weight = 1.0 - self.server_weight
        
        logging.debug(f"Adapted weights - Edge: {self.edge_weight:.3f}, "
                     f"Server: {self.server_weight:.3f}")
    
    def get_performance_metrics(self) -> VADPerformanceMetrics:
        """Get current performance metrics."""
        with self.performance_lock:
            return self.metrics
    
    def get_device_states(self) -> Dict[str, Dict]:
        """Get current device VAD states."""
        return self.device_states.copy()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        with self.performance_lock:
            self.metrics = VADPerformanceMetrics()
            self.decision_history.clear()
        
        logging.info("VAD coordinator metrics reset")
    
    def set_fusion_strategy(self, strategy: VADFusionStrategy):
        """Change fusion strategy at runtime."""
        old_strategy = self.fusion_strategy
        self.fusion_strategy = strategy
        
        logging.info(f"VAD fusion strategy changed from {old_strategy.value} to {strategy.value}")
    
    def get_current_state(self) -> VADDecision:
        """Get current VAD state."""
        return self.current_state
    
    def is_speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self.current_state in [VADDecision.SPEECH_DETECTED, VADDecision.SPEECH_START]


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock Silero VAD for testing
    class MockIntelligentVAD:
        def process_chunk(self, audio_chunk):
            # Simple energy-based VAD for testing
            energy = np.abs(audio_chunk).mean()
            return energy > 0.01, energy
    
    # Create coordinator
    server_vad = MockIntelligentVAD()
    coordinator = ESP32P4VADCoordinator(server_vad, VADFusionStrategy.ADAPTIVE)
    
    # Test with mock packet (would normally come from ESP32-P4)
    import struct
    
    # Create enhanced test packet
    basic_data = struct.pack('<IHHBBB', 1, 512, 16000, 1, 16, 0)
    vad_data = struct.pack('<BBBBBHHBB', 0x02, 0x09, 180, 200, 25000, 1200, 400, 20, 0)
    audio_data = np.random.normal(0, 0.02, 512).astype(np.float32)
    audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
    
    test_packet_data = basic_data + vad_data + audio_bytes
    
    # Parse packet
    packet_info = coordinator.protocol_parser.parse_packet(
        test_packet_data, 
        ("192.168.1.100", 8000)
    )
    
    if packet_info:
        # Process with coordinator
        result = coordinator.process_packet(packet_info, audio_data)
        
        print(f"VAD Result: {result}")
        print(f"Performance: {coordinator.get_performance_metrics()}")
        
        # Test feedback
        coordinator.provide_feedback(True, time.time())
        
        print(f"After feedback: {coordinator.get_performance_metrics()}")
    
    print("ESP32-P4 VAD Coordinator test complete")