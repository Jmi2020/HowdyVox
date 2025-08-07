#!/usr/bin/env python3

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import uuid

from .esp32_p4_protocol import ESP32P4PacketInfo, ESP32P4ProtocolParser
from .esp32_p4_vad_coordinator import ESP32P4VADCoordinator, VADDecision
from .esp32_p4_websocket import ESP32P4WebSocketServer, DeviceInfo

class WakeWordSource(Enum):
    """Source of wake word detection."""
    ESP32_P4_EDGE = "esp32_p4_edge"
    SERVER_PORCUPINE = "server_porcupine"
    HYBRID_VALIDATED = "hybrid_validated"

@dataclass
class WakeWordEvent:
    """Wake word detection event."""
    event_id: str
    source: WakeWordSource
    keyword_id: int
    keyword_name: str
    confidence: float
    device_id: Optional[str] = None
    timestamp: float = 0.0
    validation_confidence: Optional[float] = None
    audio_snippet: Optional[bytes] = None
    validated: bool = False
    rejected: bool = False

class WakeWordValidationStrategy(Enum):
    """Strategies for validating edge wake word detections."""
    EDGE_ONLY = "edge_only"           # Trust edge detections
    SERVER_VALIDATION = "server_validation"  # Require server validation
    HYBRID_CONSENSUS = "hybrid_consensus"    # Combine edge and server
    ADAPTIVE_THRESHOLD = "adaptive_threshold"  # Dynamic thresholds

@dataclass
class WakeWordMetrics:
    """Wake word detection performance metrics."""
    total_detections: int = 0
    edge_detections: int = 0
    server_detections: int = 0
    hybrid_detections: int = 0
    validated_detections: int = 0
    rejected_detections: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    avg_confidence: float = 0.0
    avg_validation_time: float = 0.0

class ESP32P4WakeWordBridge:
    """
    Bridge between ESP32-P4 wake word detection and HowdyTTS Porcupine pipeline.
    
    This bridge:
    - Receives wake word events from ESP32-P4 devices
    - Validates edge detections using server-side Porcupine
    - Coordinates multiple device wake word detections
    - Routes validated wake word events to HowdyTTS pipeline
    - Provides feedback to ESP32-P4 devices for learning
    """
    
    # Standard keyword mappings (expandable)
    KEYWORD_MAPPING = {
        0: "unknown",
        1: "hey_howdy",
        2: "hey_google", 
        3: "alexa",
        4: "computer",
        5: "ok_google"
    }
    
    def __init__(self, 
                 vad_coordinator: ESP32P4VADCoordinator,
                 websocket_server: ESP32P4WebSocketServer,
                 porcupine_callback: Callable,
                 validation_strategy: WakeWordValidationStrategy = WakeWordValidationStrategy.HYBRID_CONSENSUS,
                 validation_timeout: float = 1.0,
                 confidence_threshold: float = 0.6):
        """
        Initialize wake word bridge.
        
        Args:
            vad_coordinator: ESP32-P4 VAD coordinator
            websocket_server: WebSocket server for device communication
            porcupine_callback: Existing HowdyTTS wake word callback
            validation_strategy: Strategy for validating wake word detections
            validation_timeout: Timeout for server validation (seconds)
            confidence_threshold: Minimum confidence for wake word acceptance
        """
        self.vad_coordinator = vad_coordinator
        self.websocket_server = websocket_server
        self.porcupine_callback = porcupine_callback
        self.validation_strategy = validation_strategy
        self.validation_timeout = validation_timeout
        self.confidence_threshold = confidence_threshold
        
        # Wake word event tracking
        self.pending_validations: Dict[str, WakeWordEvent] = {}
        self.recent_events: deque = deque(maxlen=100)  # Last 100 wake word events
        self.device_wake_word_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Multi-device coordination
        self.multi_device_events: Dict[str, List[WakeWordEvent]] = defaultdict(list)
        self.coordination_window: float = 0.5  # 500ms window for multi-device events
        
        # Performance metrics
        self.metrics = WakeWordMetrics()
        
        # Threading
        self.validation_lock = threading.RLock()
        self.cleanup_timer: Optional[threading.Timer] = None
        
        # Server Porcupine instance (if available for validation)
        self.server_porcupine = None
        self.server_porcupine_available = False
        
        # Set up callbacks
        self._setup_callbacks()
        
        logging.info(f"ESP32-P4 Wake Word Bridge initialized with {validation_strategy.value} strategy")
    
    def _setup_callbacks(self):
        """Set up callbacks for VAD coordinator and WebSocket server."""
        # Set wake word callback in VAD coordinator
        self.vad_coordinator.wake_word_callback = self._on_edge_wake_word_detected
        
        # Set WebSocket callbacks
        self.websocket_server.set_callbacks(
            device_connected=self._on_device_connected,
            device_disconnected=self._on_device_disconnected,
            wake_word_sync=self._on_wake_word_sync
        )
    
    def set_server_porcupine(self, porcupine_instance):
        """Set server Porcupine instance for validation."""
        self.server_porcupine = porcupine_instance
        self.server_porcupine_available = True
        logging.info("Server Porcupine instance set for wake word validation")
    
    def _on_edge_wake_word_detected(self, wake_word_details: Dict):
        """Handle wake word detection from ESP32-P4 device."""
        try:
            # Create wake word event
            event = WakeWordEvent(
                event_id=str(uuid.uuid4()),
                source=WakeWordSource.ESP32_P4_EDGE,
                keyword_id=wake_word_details.get('keyword_id', 0),
                keyword_name=self.KEYWORD_MAPPING.get(wake_word_details.get('keyword_id', 0), "unknown"),
                confidence=wake_word_details.get('confidence', 0.0),
                device_id=wake_word_details.get('device_id'),
                timestamp=time.time(),
                audio_snippet=wake_word_details.get('audio_snippet')
            )
            
            self.metrics.total_detections += 1
            self.metrics.edge_detections += 1
            
            logging.info(f"Edge wake word detected: {event.keyword_name} from {event.device_id} "
                        f"with confidence {event.confidence:.2f}")
            
            # Add to device history
            if event.device_id:
                self.device_wake_word_history[event.device_id].append(event)
            
            # Process based on validation strategy
            if self.validation_strategy == WakeWordValidationStrategy.EDGE_ONLY:
                self._handle_edge_only_validation(event)
            elif self.validation_strategy == WakeWordValidationStrategy.SERVER_VALIDATION:
                self._handle_server_validation(event)
            elif self.validation_strategy == WakeWordValidationStrategy.HYBRID_CONSENSUS:
                self._handle_hybrid_validation(event)
            elif self.validation_strategy == WakeWordValidationStrategy.ADAPTIVE_THRESHOLD:
                self._handle_adaptive_validation(event)
        
        except Exception as e:
            logging.error(f"Error handling edge wake word detection: {e}")
    
    def _handle_edge_only_validation(self, event: WakeWordEvent):
        """Handle edge-only validation strategy."""
        if event.confidence >= self.confidence_threshold:
            self._trigger_wake_word_callback(event)
            
            # Send validation feedback to device
            if event.device_id:
                self.websocket_server.send_wake_word_validation(
                    event.device_id, 
                    event.keyword_id, 
                    event.confidence,
                    event.event_id
                )
        else:
            # Reject low confidence detection
            self._reject_wake_word_event(event, "low_confidence")
    
    def _handle_server_validation(self, event: WakeWordEvent):
        """Handle server validation strategy."""
        if not self.server_porcupine_available:
            # Fallback to edge-only if no server Porcupine
            self._handle_edge_only_validation(event)
            return
        
        # Add to pending validations
        with self.validation_lock:
            self.pending_validations[event.event_id] = event
        
        # Start validation timer
        timer = threading.Timer(
            self.validation_timeout,
            self._validation_timeout,
            args=[event.event_id]
        )
        timer.start()
        
        # Trigger server Porcupine validation if audio available
        if event.audio_snippet and self.server_porcupine:
            self._validate_with_server_porcupine(event)
        else:
            # No audio available, use confidence-based validation
            if event.confidence >= self.confidence_threshold:
                self._validate_wake_word_event(event, event.confidence)
            else:
                self._reject_wake_word_event(event, "no_audio_low_confidence")
    
    def _handle_hybrid_validation(self, event: WakeWordEvent):
        """Handle hybrid consensus validation."""
        # Check for multi-device coordination
        coordination_key = f"{event.keyword_id}_{int(event.timestamp / self.coordination_window)}"
        self.multi_device_events[coordination_key].append(event)
        
        # Start coordination timer
        timer = threading.Timer(
            self.coordination_window,
            self._process_multi_device_coordination,
            args=[coordination_key]
        )
        timer.start()
    
    def _handle_adaptive_validation(self, event: WakeWordEvent):
        """Handle adaptive threshold validation."""
        # Adjust threshold based on device history
        device_accuracy = self._get_device_accuracy(event.device_id)
        adjusted_threshold = self.confidence_threshold * (2.0 - device_accuracy)
        
        if event.confidence >= adjusted_threshold:
            self._trigger_wake_word_callback(event)
            
            if event.device_id:
                self.websocket_server.send_wake_word_validation(
                    event.device_id,
                    event.keyword_id,
                    event.confidence,
                    event.event_id
                )
        else:
            self._reject_wake_word_event(event, "adaptive_threshold")
    
    def _validate_with_server_porcupine(self, event: WakeWordEvent):
        """Validate wake word using server Porcupine."""
        try:
            # This is a placeholder - actual implementation would process audio
            # through Porcupine and get validation confidence
            
            # For now, simulate validation based on confidence
            validation_confidence = min(1.0, event.confidence + 0.1)
            
            if validation_confidence >= self.confidence_threshold:
                self._validate_wake_word_event(event, validation_confidence)
            else:
                self._reject_wake_word_event(event, "server_validation_failed")
        
        except Exception as e:
            logging.error(f"Error in server Porcupine validation: {e}")
            self._reject_wake_word_event(event, "validation_error")
    
    def _validate_wake_word_event(self, event: WakeWordEvent, validation_confidence: float):
        """Validate and trigger wake word event."""
        event.validated = True
        event.validation_confidence = validation_confidence
        
        with self.validation_lock:
            if event.event_id in self.pending_validations:
                del self.pending_validations[event.event_id]
        
        self.metrics.validated_detections += 1
        
        # Send validation to device
        if event.device_id:
            self.websocket_server.send_wake_word_validation(
                event.device_id,
                event.keyword_id,
                validation_confidence,
                event.event_id
            )
        
        # Trigger wake word callback
        self._trigger_wake_word_callback(event)
        
        logging.info(f"Wake word validated: {event.keyword_name} with confidence {validation_confidence:.2f}")
    
    def _reject_wake_word_event(self, event: WakeWordEvent, reason: str):
        """Reject wake word event."""
        event.rejected = True
        
        with self.validation_lock:
            if event.event_id in self.pending_validations:
                del self.pending_validations[event.event_id]
        
        self.metrics.rejected_detections += 1
        
        # Send rejection to device
        if event.device_id:
            self.websocket_server.send_wake_word_rejection(
                event.device_id,
                event.keyword_id,
                reason,
                event.event_id
            )
        
        logging.debug(f"Wake word rejected: {event.keyword_name} - {reason}")
    
    def _trigger_wake_word_callback(self, event: WakeWordEvent):
        """Trigger the HowdyTTS wake word callback."""
        try:
            # Add event to recent history
            self.recent_events.append(event)
            
            # Update metrics
            final_confidence = event.validation_confidence or event.confidence
            self.metrics.avg_confidence = (
                (self.metrics.avg_confidence * (self.metrics.validated_detections - 1) + final_confidence) /
                max(1, self.metrics.validated_detections)
            )
            
            # Call the HowdyTTS callback
            # The callback expects no parameters, so we just call it
            if self.porcupine_callback:
                self.porcupine_callback()
            
            logging.info(f"Triggered HowdyTTS wake word callback for {event.keyword_name}")
        
        except Exception as e:
            logging.error(f"Error triggering wake word callback: {e}")
    
    def _validation_timeout(self, event_id: str):
        """Handle validation timeout."""
        with self.validation_lock:
            if event_id in self.pending_validations:
                event = self.pending_validations.pop(event_id)
                self._reject_wake_word_event(event, "validation_timeout")
    
    def _process_multi_device_coordination(self, coordination_key: str):
        """Process multi-device wake word coordination."""
        events = self.multi_device_events.get(coordination_key, [])
        if not events:
            return
        
        # Clean up coordination entry
        if coordination_key in self.multi_device_events:
            del self.multi_device_events[coordination_key]
        
        if len(events) == 1:
            # Single device detection - process normally
            event = events[0]
            if event.confidence >= self.confidence_threshold:
                self._trigger_wake_word_callback(event)
                if event.device_id:
                    self.websocket_server.send_wake_word_validation(
                        event.device_id, event.keyword_id, event.confidence, event.event_id
                    )
            else:
                self._reject_wake_word_event(event, "single_device_low_confidence")
        else:
            # Multiple device detections - use consensus
            avg_confidence = sum(e.confidence for e in events) / len(events)
            best_event = max(events, key=lambda e: e.confidence)
            
            if avg_confidence >= self.confidence_threshold:
                # Create hybrid event
                hybrid_event = WakeWordEvent(
                    event_id=str(uuid.uuid4()),
                    source=WakeWordSource.HYBRID_VALIDATED,
                    keyword_id=best_event.keyword_id,
                    keyword_name=best_event.keyword_name,
                    confidence=avg_confidence,
                    timestamp=best_event.timestamp,
                    validated=True
                )
                
                self.metrics.hybrid_detections += 1
                
                # Send validation to all contributing devices
                for event in events:
                    if event.device_id:
                        self.websocket_server.send_wake_word_validation(
                            event.device_id, event.keyword_id, avg_confidence, event.event_id
                        )
                
                # Broadcast sync to all devices
                self.websocket_server.broadcast_wake_word_sync(
                    best_event.device_id or "multiple", 
                    best_event.keyword_id, 
                    avg_confidence
                )
                
                self._trigger_wake_word_callback(hybrid_event)
                
                logging.info(f"Multi-device wake word consensus: {len(events)} devices, confidence {avg_confidence:.2f}")
            else:
                # Reject all events
                for event in events:
                    self._reject_wake_word_event(event, "multi_device_low_consensus")
    
    def _get_device_accuracy(self, device_id: str) -> float:
        """Get historical accuracy for device."""
        if not device_id:
            return 0.5  # Default accuracy
        
        history = self.device_wake_word_history.get(device_id, [])
        if not history:
            return 0.5
        
        validated = sum(1 for event in history if event.validated)
        total = len(history)
        
        return validated / total if total > 0 else 0.5
    
    def _on_device_connected(self, device_info: DeviceInfo):
        """Handle device connection."""
        logging.info(f"Wake word bridge: Device {device_info.device_id} connected")
    
    def _on_device_disconnected(self, device_info: DeviceInfo):
        """Handle device disconnection."""
        logging.info(f"Wake word bridge: Device {device_info.device_id} disconnected")
    
    def _on_wake_word_sync(self, device_id: str, data: Dict):
        """Handle wake word synchronization between devices."""
        logging.debug(f"Wake word sync from {device_id}: {data}")
    
    def provide_feedback(self, 
                        event_id: str, 
                        is_correct: bool, 
                        user_feedback: str = ""):
        """Provide feedback on wake word detection accuracy."""
        # Find the event in recent history
        for event in self.recent_events:
            if event.event_id == event_id:
                if is_correct:
                    logging.info(f"Positive feedback for wake word {event.keyword_name}")
                else:
                    self.metrics.false_positives += 1
                    logging.info(f"False positive feedback for wake word {event.keyword_name}")
                    
                    # Send correction to device
                    if event.device_id:
                        self.websocket_server.send_wake_word_rejection(
                            event.device_id,
                            event.keyword_id,
                            f"user_feedback: {user_feedback}",
                            event_id
                        )
                break
    
    def get_metrics(self) -> WakeWordMetrics:
        """Get wake word detection metrics."""
        return self.metrics
    
    def get_recent_events(self, count: int = 10) -> List[WakeWordEvent]:
        """Get recent wake word events."""
        return list(self.recent_events)[-count:]
    
    def get_device_history(self, device_id: str) -> List[WakeWordEvent]:
        """Get wake word history for specific device."""
        return list(self.device_wake_word_history.get(device_id, []))
    
    def set_validation_strategy(self, strategy: WakeWordValidationStrategy):
        """Change validation strategy at runtime."""
        old_strategy = self.validation_strategy
        self.validation_strategy = strategy
        logging.info(f"Wake word validation strategy changed from {old_strategy.value} to {strategy.value}")
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        old_threshold = self.confidence_threshold
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logging.info(f"Wake word confidence threshold changed from {old_threshold:.2f} to {self.confidence_threshold:.2f}")
    
    def cleanup(self):
        """Clean up resources."""
        with self.validation_lock:
            self.pending_validations.clear()
        
        self.multi_device_events.clear()
        
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        logging.info("Wake word bridge cleaned up")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock components for testing
    class MockVADCoordinator:
        def __init__(self):
            self.wake_word_callback = None
    
    class MockWebSocketServer:
        def set_callbacks(self, **kwargs):
            pass
        
        def send_wake_word_validation(self, device_id, keyword_id, confidence, correlation_id):
            print(f"Validation sent to {device_id}: keyword={keyword_id}, confidence={confidence}")
        
        def send_wake_word_rejection(self, device_id, keyword_id, reason, correlation_id):
            print(f"Rejection sent to {device_id}: keyword={keyword_id}, reason={reason}")
        
        def broadcast_wake_word_sync(self, source_device_id, keyword_id, confidence):
            print(f"Sync broadcast: source={source_device_id}, keyword={keyword_id}")
    
    def mock_porcupine_callback():
        print("HowdyTTS wake word callback triggered!")
    
    # Create bridge
    vad_coordinator = MockVADCoordinator()
    websocket_server = MockWebSocketServer()
    
    bridge = ESP32P4WakeWordBridge(
        vad_coordinator=vad_coordinator,
        websocket_server=websocket_server,
        porcupine_callback=mock_porcupine_callback,
        validation_strategy=WakeWordValidationStrategy.HYBRID_CONSENSUS
    )
    
    # Simulate wake word detection
    wake_word_details = {
        'device_id': 'esp32_p4_001',
        'keyword_id': 1,
        'confidence': 0.85,
        'quality': 0.9,
        'detection_start_ms': 1234,
        'detection_duration_ms': 500,
        'high_confidence': True,
        'source_addr': ('192.168.1.100', 8000)
    }
    
    bridge._on_edge_wake_word_detected(wake_word_details)
    
    # Show metrics
    print(f"Metrics: {bridge.get_metrics()}")
    
    print("ESP32-P4 Wake Word Bridge test complete")