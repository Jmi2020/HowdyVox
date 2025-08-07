# ESP32-P4 Wake Word Detection Integration Guide

This guide explains how to integrate the enhanced ESP32-P4 wake word detection system with HowdyTTS for Option C bidirectional communication.

## Overview

The ESP32-P4 Wake Word Detection System extends the existing ESP32-P4 integration to support:

- **Edge Wake Word Detection**: ESP32-P4 devices perform local Porcupine-style wake word detection
- **Server Validation**: HowdyTTS server validates edge detections using server-side Porcupine
- **Hybrid Consensus**: Multiple validation strategies including edge-only, server validation, and hybrid consensus
- **Real-time Feedback**: WebSocket channel provides immediate feedback to ESP32-P4 devices
- **Multi-device Coordination**: Synchronizes wake word events across multiple ESP32-P4 devices
- **Seamless Integration**: Works with existing HowdyTTS `handle_wake_word()` callback

## Architecture

```
┌─────────────────┐    UDP Audio + Wake Word    ┌──────────────────┐
│   ESP32-P4      │ ────────────────────────────> │   HowdyTTS       │
│   Edge Device   │                               │   Server         │
│                 │    WebSocket Feedback        │                  │
│ • Wake Word     │ <──────────────────────────── │ • Wake Word      │
│   Detection     │                               │   Validation     │
│ • VAD           │                               │ • Porcupine      │
│ • Audio Capture │                               │ • VAD Fusion     │
└─────────────────┘                               └──────────────────┘
```

## Integration Steps

### 1. Import Enhanced Components

Add these imports to your main HowdyTTS program:

```python
from voice_assistant.esp32_p4_wake_word import (
    ESP32P4WakeWordBridge,
    WakeWordValidationStrategy
)
from voice_assistant.esp32_p4_websocket import ESP32P4WebSocketServer
from voice_assistant.esp32_p4_vad_coordinator import ESP32P4VADCoordinator
```

### 2. Initialize WebSocket Feedback Server

```python
# Initialize WebSocket server for ESP32-P4 feedback
websocket_server = ESP32P4WebSocketServer(
    host="0.0.0.0",
    port=8001,  # Different from UDP audio port (8000)
    max_devices=10
)

if not websocket_server.start():
    logging.error("Failed to start ESP32-P4 WebSocket server")
```

### 3. Initialize Wake Word Bridge

```python
# Get existing handle_wake_word callback from your program
from run_voice_assistant import handle_wake_word

# Create wake word bridge
wake_word_bridge = ESP32P4WakeWordBridge(
    vad_coordinator=vad_coordinator,  # Your existing VAD coordinator
    websocket_server=websocket_server,
    porcupine_callback=handle_wake_word,
    validation_strategy=WakeWordValidationStrategy.HYBRID_CONSENSUS,
    confidence_threshold=0.7
)

# Optional: Set server Porcupine for validation
if hasattr(wake_word_detector, 'porcupine'):
    wake_word_bridge.set_server_porcupine(wake_word_detector.porcupine)
```

### 4. Update NetworkAudioSource Integration

If using NetworkAudioSource, update the initialization:

```python
# In voice_assistant/network_audio_source.py
def __init__(self, target_room=None, wake_word_callback=None):
    # ... existing initialization ...
    
    # Initialize WebSocket server
    self.websocket_server = ESP32P4WebSocketServer(
        host="0.0.0.0",
        port=8001
    )
    
    # Enhanced VAD coordinator with wake word support
    self.vad_coordinator = ESP32P4VADCoordinator(
        server_vad=self.vad,
        fusion_strategy=VADFusionStrategy.ADAPTIVE,
        wake_word_callback=wake_word_callback
    )
    
    # Wake word bridge
    if wake_word_callback:
        self.wake_word_bridge = ESP32P4WakeWordBridge(
            vad_coordinator=self.vad_coordinator,
            websocket_server=self.websocket_server,
            porcupine_callback=wake_word_callback,
            validation_strategy=WakeWordValidationStrategy.HYBRID_CONSENSUS
        )
```

### 5. Modify Main Program (run_voice_assistant.py)

Add wake word bridge initialization in the main program:

```python
# In the wireless audio initialization section
if args.wireless:
    # ... existing wireless setup ...
    
    # Initialize wake word bridge for ESP32-P4 devices
    if network_audio_source and hasattr(network_audio_source, 'wake_word_bridge'):
        logging.info("ESP32-P4 wake word detection enabled")
        
        # Set server Porcupine for validation if available
        if 'wake_word_detector' in locals() and hasattr(wake_word_detector, 'porcupine'):
            network_audio_source.wake_word_bridge.set_server_porcupine(
                wake_word_detector.porcupine
            )
```

## ESP32-P4 Firmware Requirements

### Enhanced UDP Packet Format

The ESP32-P4 firmware must support the new packet format:

```c
// Packet structure for wake word detection
typedef struct {
    // Basic header (12 bytes)
    uint32_t sequence;
    uint16_t sample_count;
    uint16_t sample_rate;
    uint8_t channels;
    uint8_t bits_per_sample;
    uint16_t flags;
    
    // VAD header (12 bytes) - version = 0x03 for wake word support
    uint8_t version;
    uint8_t vad_flags;
    uint8_t vad_confidence;
    uint8_t detection_quality;
    uint16_t max_amplitude;
    uint16_t noise_floor;
    uint16_t zero_crossing_rate;
    uint8_t snr_db_scaled;
    uint8_t reserved;
    
    // Wake word header (12 bytes) - when wake word detected
    uint8_t wake_version;         // 0x03
    uint8_t wake_word_flags;      // WAKE_WORD_DETECTED, etc.
    uint8_t wake_confidence;      // 0-255
    uint8_t keyword_id;          // 1=hey_howdy, 2=hey_google, etc.
    uint16_t detection_start_ms; // Start time in packet
    uint16_t detection_duration_ms; // Duration of wake word
    uint8_t wake_word_quality;   // Quality metric 0-255
    uint8_t validation_confidence; // Server validation (updated via WebSocket)
    uint8_t reserved2;
    
    // Audio data follows...
} esp32_p4_wake_word_packet_t;
```

### Wake Word Flags

```c
typedef enum {
    WAKE_WORD_DETECTED = 0x01,
    WAKE_WORD_END = 0x02,
    HIGH_CONFIDENCE_WAKE = 0x04,
    MULTIPLE_KEYWORDS = 0x08,
    WAKE_WORD_VALIDATED = 0x10,  // Set by server via WebSocket
    WAKE_WORD_REJECTED = 0x20,   // Set by server via WebSocket
} esp32_p4_wake_word_flags_t;
```

### WebSocket Client Integration

ESP32-P4 firmware should connect to WebSocket server for feedback:

```c
// WebSocket connection to server
void connect_websocket() {
    // Connect to ws://server_ip:8001
    // Send device registration
    send_device_registration();
}

// Handle feedback messages
void handle_websocket_message(const char* message) {
    // Parse JSON feedback
    // Update wake word sensitivity based on validation/rejection
    // Sync with other devices for multi-device coordination
}
```

## Configuration Options

### Validation Strategies

- **EDGE_ONLY**: Trust ESP32-P4 detections above threshold
- **SERVER_VALIDATION**: Require server Porcupine validation  
- **HYBRID_CONSENSUS**: Combine edge and server decisions
- **ADAPTIVE_THRESHOLD**: Dynamic thresholds based on device accuracy

### Confidence Thresholds

- Default: 0.6 (60% confidence)
- High accuracy mode: 0.8
- High sensitivity mode: 0.4

### Multi-device Coordination

- Coordination window: 500ms
- Consensus algorithm: Average confidence from multiple devices
- Sync broadcasting to prevent false wake words

## Monitoring and Metrics

### Wake Word Metrics

```python
# Get performance metrics
metrics = wake_word_bridge.get_metrics()
print(f"Total detections: {metrics.total_detections}")
print(f"Edge detections: {metrics.edge_detections}")
print(f"Validated detections: {metrics.validated_detections}")
print(f"Average confidence: {metrics.avg_confidence:.2f}")
```

### Device Status

```python
# Get connected devices
devices = websocket_server.get_connected_devices()
for device_id, device_info in devices.items():
    print(f"Device {device_id}: accuracy={device_info.wake_word_accuracy:.2f}")
```

### Recent Events

```python
# Get recent wake word events
recent_events = wake_word_bridge.get_recent_events(10)
for event in recent_events:
    print(f"{event.timestamp}: {event.keyword_name} from {event.device_id}")
```

## Troubleshooting

### Common Issues

1. **No wake word packets received**
   - Check ESP32-P4 firmware supports VERSION_WAKE_WORD (0x03)
   - Verify wake word detection is enabled in firmware
   - Check UDP packet format matches expected structure

2. **WebSocket connection fails**
   - Ensure port 8001 is open and available
   - Check device registration message format
   - Verify network connectivity between ESP32-P4 and server

3. **Wake words not triggering**
   - Check confidence thresholds are appropriate
   - Verify validation strategy is suitable for your setup
   - Monitor validation/rejection feedback

4. **Multiple false positives**
   - Increase confidence threshold
   - Enable server validation strategy
   - Check for acoustic interference between devices

### Debug Logging

Enable debug logging for detailed information:

```python
logging.getLogger('voice_assistant.esp32_p4_wake_word').setLevel(logging.DEBUG)
logging.getLogger('voice_assistant.esp32_p4_websocket').setLevel(logging.DEBUG)
```

## Performance Considerations

### Latency

- Edge detection: ~50-100ms (device processing)
- Server validation: +100-200ms (network + processing)  
- Hybrid consensus: +200-500ms (coordination window)

### Resource Usage

- WebSocket connections: ~1KB per device
- Wake word event history: ~10KB per device
- Processing overhead: <5% CPU for typical loads

### Scalability

- Supports up to 10 concurrent ESP32-P4 devices
- Linear scaling with number of devices
- Configurable validation strategies for performance vs accuracy

## Integration Examples

### Example 1: Edge-Only Mode (Lowest Latency)

```python
wake_word_bridge = ESP32P4WakeWordBridge(
    vad_coordinator=vad_coordinator,
    websocket_server=websocket_server,
    porcupine_callback=handle_wake_word,
    validation_strategy=WakeWordValidationStrategy.EDGE_ONLY,
    confidence_threshold=0.8  # Higher threshold for edge-only
)
```

### Example 2: High Accuracy Mode

```python
wake_word_bridge = ESP32P4WakeWordBridge(
    vad_coordinator=vad_coordinator,
    websocket_server=websocket_server,
    porcupine_callback=handle_wake_word,
    validation_strategy=WakeWordValidationStrategy.SERVER_VALIDATION,
    validation_timeout=2.0,  # Allow more time for validation
    confidence_threshold=0.6
)
```

### Example 3: Multi-Device Consensus

```python
wake_word_bridge = ESP32P4WakeWordBridge(
    vad_coordinator=vad_coordinator,
    websocket_server=websocket_server,
    porcupine_callback=handle_wake_word,
    validation_strategy=WakeWordValidationStrategy.HYBRID_CONSENSUS,
    confidence_threshold=0.7
)
# Automatically handles multiple device coordination
```

This integration provides a robust, scalable wake word detection system that enhances the ESP32-P4 capabilities while maintaining seamless integration with the existing HowdyTTS pipeline.