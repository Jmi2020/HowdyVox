# ESP32-P4 HowdyScreen Integration Guide

This guide explains how to integrate your ESP32-P4 HowdyScreen wireless voice assistant devices with HowdyTTS for complete bidirectional VAD and wake word detection.

## Overview

The ESP32-P4 HowdyScreen integration provides the **first-ever ESP32-P4 integration** with HowdyTTS, featuring:

- **Bidirectional VAD System**: Edge (ESP32-P4) + Server (Silero) VAD fusion with 5 strategies
- **Wake Word Detection**: ESP32-P4 edge detection with Porcupine server validation  
- **Enhanced UDP Protocol**: VERSION_WAKE_WORD (0x03) with 24-byte headers
- **WebSocket Feedback**: Real-time VAD corrections and threshold adaptation (Port 8001)
- **Multi-device Coordination**: Room-based device management and wake word sharing
- **Adaptive Learning**: Self-improving system based on usage patterns
- **Visual Feedback**: Round display UI with audio level visualization

## Architecture

```
┌─────────────────────────────────────┐    ┌────────────────────────────────────┐
│        ESP32-P4 HowdyScreen         │    │         HowdyTTS Server            │
├─────────────────────────────────────┤    ├────────────────────────────────────┤
│                                     │    │                                    │
│ ┌─────────────────────────────────┐ │    │ ┌──────────────────────────────┐   │
│ │   Enhanced VAD + Wake Word      │ │    │ │    Silero VAD + Porcupine    │   │
│ │   - Energy + spectral analysis │ │    │ │    - Neural VAD              │   │
│ │   - "Hey Howdy" pattern match  │ │    │ │    - Wake word validation    │   │
│ │   - Adaptive thresholds        │ │    │ │    - Multi-device coordination│  │
│ └─────────────────────────────────┘ │    │ └──────────────────────────────┘   │
│                ↓                    │    │                ↓                   │
│ ┌─────────────────────────────────┐ │    │ ┌──────────────────────────────┐   │
│ │      Enhanced UDP Protocol      │ │──→ │ │   ESP32-P4 Protocol Parser   │   │
│ │   - VERSION_WAKE_WORD (0x03)    │ │    │ │   - 24-byte headers          │   │
│ │   - 24-byte headers             │ │    │ │   - VAD + Wake word data     │   │
│ │   - Port 8000                   │ │    │ │   - Multi-device handling    │   │
│ └─────────────────────────────────┘ │    │ └──────────────────────────────┘   │
│                ↓                    │    │                ↓                   │
│ ┌─────────────────────────────────┐ │    │ ┌──────────────────────────────┐   │
│ │      WebSocket Client           │ │←─► │ │   VAD Feedback Server        │   │
│ │   - Real-time feedback          │ │    │ │   - Port 8001                │   │
│ │   - Threshold updates           │ │    │ │   - JSON protocol            │   │
│ │   - Wake word validation        │ │    │ │   - Multi-device sync        │   │
│ └─────────────────────────────────┘ │    │ └──────────────────────────────┘   │
│                ↓                    │    │                ↓                   │
│ ┌─────────────────────────────────┐ │    │ ┌──────────────────────────────┐   │
│ │     Round Display UI            │ │    │ │      VAD Coordinator         │   │
│ │   - Audio level visualization  │ │    │ │   - 5 fusion strategies      │   │
│ │   - Connection status           │ │    │ │   - Edge + server fusion     │   │
│ │   - Wake word feedback          │ │    │ │   - Adaptive learning        │   │
│ └─────────────────────────────────┘ │    │ └──────────────────────────────┘   │
└─────────────────────────────────────┘    └────────────────────────────────────┘
```

## Quick Start

### 1. ESP32-P4 Setup

1. **Configure your ESP32-P4 device** with Phase 6C firmware:
   ```c
   // In main/howdy_phase6_howdytts_integration.c
   #define WIFI_SSID     "your_wifi_network"
   #define WIFI_PASSWORD "your_wifi_password"  
   #define SERVER_IP     "192.168.1.100"      // HowdyTTS server IP
   #define WAKE_WORD_ENABLED true              // Enable wake word detection
   ```

2. **Build and flash** the Phase 6C firmware:
   ```bash
   cd /Users/silverlinings/Desktop/Coding/ESP32P4/HowdyScreen
   idf.py build
   idf.py -p /dev/cu.usbserial-* flash monitor
   ```

### 2. HowdyTTS Launch (Updated for Phase 6C)

1. **Start HowdyTTS with conda environment**:
   ```bash
   cd /Users/silverlinings/Desktop/Coding/RBP/HowdyTTS
   python launch_howdy_shell.py
   ```
   
   This automatically:
   - Activates `howdy310` conda environment
   - Starts FastWhisperAPI server (port 8000)
   - Launches `run_voice_assistant.py` with full ESP32-P4 integration

2. **Advanced Launch Options**:
   ```bash
   # Use wireless ESP32-P4 devices (auto-discovery)
   python run_voice_assistant.py --wireless
   
   # Target specific room device  
   python run_voice_assistant.py --wireless --room "Living Room"
   
   # Auto-detect best audio source (wireless first, local fallback)
   python run_voice_assistant.py --auto
   
   # List available ESP32-P4 devices
   python run_voice_assistant.py --list-devices
   ```

### 3. System Verification

Once both systems are running, you should see:

**ESP32-P4 Serial Output:**
```
I (12345) HOWDY: WiFi connected to your_wifi_network
I (12346) HOWDY: Enhanced VAD initialized  
I (12347) HOWDY: Wake word detection active - "Hey Howdy"
I (12348) HOWDY: WebSocket feedback client connected to 192.168.1.100:8001
I (12349) HOWDY: UDP audio streaming to 192.168.1.100:8000
```

**HowdyTTS Server Output:**
```
🤠 HowdyScreen device ESP32P4_001 connected (Living Room)
✅ VAD Coordinator: Enhanced VAD packet received (confidence: 0.85)  
🎯 Wake word "Hey Howdy" detected with 0.72 confidence
✓ Porcupine validation: Wake word confirmed
→ WebSocket feedback sent: threshold_update
```

## Usage Examples

### Basic Wireless Mode
```bash
python run_voice_assistant.py --wireless
```
- Automatically discovers and uses available ESP32P4 devices
- Falls back to first available device if no room assignment

### Room-Specific Mode
```bash
python run_voice_assistant.py --room "Living Room"
```
- Uses the ESP32P4 device assigned to "Living Room"
- You can assign rooms through the device management interface

### Device Discovery
```bash
python run_voice_assistant.py --list-devices
```
Output:
```
Available wireless devices:
  0: ESP32P4_001 (Living Room) - 192.168.1.101
  1: ESP32P4_002 (Kitchen) - 192.168.1.102
```

## Device Management

### Room Assignment

The system automatically discovers ESP32P4 devices on your network. To assign devices to rooms:

1. **Start the system** and note device IDs from the logs
2. **Assign rooms programmatically**:
   ```python
   from voice_assistant.wireless_device_manager import WirelessDeviceManager
   
   manager = WirelessDeviceManager()
   manager.start_monitoring()
   
   # Wait for device discovery
   time.sleep(5)
   
   # Assign device to room
   manager.assign_room("ESP32P4_001", "Living Room")
   manager.assign_room("ESP32P4_002", "Kitchen")
   ```

### Device Status Monitoring

Wireless devices report their status including:
- **Audio levels** for visual feedback
- **WiFi signal strength** (RSSI)
- **Battery level** (if applicable)
- **Connection status** (ready, recording, muted, error)

## Enhanced Audio Pipeline Integration (Phase 6C)

The Phase 6C implementation provides advanced bidirectional VAD and wake word integration:

### Bidirectional VAD System
- **Edge VAD**: ESP32-P4 enhanced VAD with spectral analysis and consistency checking
- **Server VAD**: Silero neural VAD with deep learning accuracy  
- **VAD Fusion**: 5 strategies for optimal performance:
  1. **Edge Priority**: Fast response using edge VAD with server backup
  2. **Server Priority**: High accuracy using server VAD with edge confirmation
  3. **Confidence Weighted**: Dynamic weighting based on confidence scores
  4. **Adaptive Learning**: Self-improving based on historical performance
  5. **Majority Vote**: Consensus-based decision making

### Wake Word Detection System
- **ESP32-P4 Edge Detection**: Lightweight "Hey Howdy" pattern matching with syllable counting
- **Porcupine Server Validation**: High-accuracy server-side wake word confirmation
- **Hybrid Consensus**: Combines edge speed with server accuracy
- **Real-time Adaptation**: WebSocket feedback for threshold optimization

### Enhanced Protocol Stack
- **VERSION_WAKE_WORD (0x03)**: Extended UDP protocol with wake word data
- **24-byte Headers**: VAD (12 bytes) + Wake word (12 bytes) metadata
- **WebSocket Feedback**: Port 8001 for bidirectional communication
- **Multi-device Coordination**: Synchronized wake word events across devices

### Processing Flow (Phase 6C)
```
ESP32-P4 Device:
Audio → Enhanced VAD → Wake Word Detection → Enhanced UDP (24-byte) → HowdyTTS

HowdyTTS Server:  
Enhanced UDP → Protocol Parser → VAD Coordinator → Wake Word Validation → Response
                     ↓                                      ↓
              WebSocket Feedback ←←←←←←←←← Porcupine Validation
                     ↓
              ESP32-P4 Threshold Updates + Wake Word Confirmation
```

## Network Configuration

### Firewall Settings
Ensure these ports are open on your HowdyTTS server:
- **UDP 8000**: Enhanced audio streaming from ESP32-P4 devices
- **TCP 8001**: WebSocket feedback server for VAD corrections
- **UDP 5353**: mDNS service discovery broadcasts

### Runtime Audio Source Control
HowdyTTS supports runtime switching between local and wireless microphones:

**Hotkey Controls** (if keyboard module available):
```
Ctrl+Alt+L - Switch to local microphone
Ctrl+Alt+W - Switch to wireless microphone  
Ctrl+Alt+T - Toggle audio source
Ctrl+Alt+I - Show audio source info
Ctrl+Alt+D - List wireless devices
```

**Programmatic Control**:
```python
from voice_assistant.audio_source_manager import get_audio_manager

# Get current audio manager
manager = get_audio_manager()

# Switch to wireless
manager.switch_to_wireless("Living Room")

# Switch to local  
manager.switch_to_local()

# Get current source info
info = manager.get_source_info()
print(f"Current source: {info['current_source']}")
```

### WiFi Optimization
For best audio quality:
- Use **5GHz WiFi** for lower latency
- Ensure **strong signal strength** (-50dBm or better)
- **Disable power saving** on ESP32P4 devices
- Consider **QoS settings** for audio traffic

## Troubleshooting

### No Devices Found
1. **Check network connectivity** - ESP32P4 and HowdyTTS on same network
2. **Verify firewall settings** - UDP ports 8000/8001 open
3. **Check ESP32P4 serial output** for connection errors
4. **Wait longer** - device discovery can take 10-15 seconds

### Audio Quality Issues
1. **Check WiFi signal strength** - move ESP32P4 closer to router
2. **Verify OPUS installation**: `pip install opuslib`
3. **Check ESP32P4 microphone** - test local recording first
4. **Network congestion** - try 5GHz band or different channel

### Connection Drops
1. **Power supply issues** - ensure stable 5V power to ESP32P4
2. **WiFi interference** - change WiFi channel
3. **Check device logs** for specific error messages
4. **Restart devices** - power cycle ESP32P4 and restart HowdyTTS

### Audio Latency
- **Expected latency**: 2-5ms for audio, 50-100ms total response time
- **High latency causes**: weak WiFi, network congestion, power saving enabled
- **Solutions**: use 5GHz WiFi, disable power saving, upgrade router firmware

## Advanced Configuration

### Custom Audio Settings
Modify `NetworkAudioSource` parameters:
```python
network_audio_source = NetworkAudioSource(
    target_room="Living Room",
    sample_rate=16000,  # Audio sample rate
    chunk_size=512,     # VAD chunk size (32ms)
    buffer_size=1000    # Audio buffer size
)
```

### Device Discovery Settings
Adjust discovery parameters in `WirelessDeviceManager`:
```python
manager = WirelessDeviceManager()
manager.device_timeout = 30.0      # Device timeout (seconds)
manager.discovery_timeout = 60.0   # Discovery interval (seconds)
```

### LED Ring Customization
The ESP32P4 LED ring responds to audio levels. Customize in `led_controller.c`:
- **Bass response**: Inner rings (red/orange)
- **Mid response**: Middle rings (green/yellow)  
- **Treble response**: Outer rings (blue/purple)
- **Sparkle effects**: High-frequency emphasis

## Monitoring and Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
python run_voice_assistant.py --wireless
```

### Device Statistics
Access real-time statistics:
```python
# Get audio server stats
stats = network_audio_source.get_stats()
print(f"Packets received: {stats['network_audio']['packets_received']}")
print(f"Active devices: {stats['device_manager']['active_devices']}")

# Get device-specific info
devices = network_audio_source.get_available_devices()
for device in devices:
    print(f"Device: {device.device_id}, RSSI: {device.signal_strength}dBm")
```

### ESP32P4 Serial Monitoring
```bash
cd /path/to/ESP32P4/HowdyScreen
idf.py monitor
```
Watch for:
- **WiFi connection status**
- **Audio processing statistics**
- **Network transmission errors**
- **Memory usage and performance**

## Integration with Existing Features

### LED Matrix Support
Wireless mode works alongside existing ESP32-S3 LED matrix:
```bash
# Use both wireless audio AND LED matrix
python run_voice_assistant.py --wireless
```

### Multi-Room TTS
Route TTS responses to specific devices:
```python
# Send TTS to specific room
network_audio_source.send_audio_to_room("Living Room", audio_data)

# Broadcast to all devices
network_audio_source.broadcast_audio(audio_data)
```

### Wake Word Detection
Currently uses local wake word detection. Future versions will support:
- **Distributed wake word** processing on ESP32P4 devices
- **Room-specific wake words**
- **Multi-device coordination**

## Performance Optimization

### ESP32P4 Optimization
- **Core affinity**: Audio processing on Core 0, UI on Core 1
- **Memory allocation**: TCM for audio buffers, PSRAM for large data
- **Network tuning**: Disable WiFi power saving, optimize socket buffers

### HowdyTTS Optimization
- **Audio buffering**: Configurable queue sizes for latency vs. reliability
- **OPUS codec**: Hardware acceleration where available
- **Memory management**: Automatic cleanup of audio resources

## Future Enhancements

Planned features:
- **WebSocket control channel** for device commands
- **Bidirectional TTS streaming** to ESP32P4 speakers
- **Device clustering** for improved coverage
- **Mobile app integration** for device management
- **Cloud connectivity** for remote monitoring

---

For technical support or feature requests, please check the project repository or contact the development team.