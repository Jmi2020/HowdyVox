# ESP32P4 HowdyScreen Integration Guide

This guide explains how to integrate your ESP32P4 HowdyScreen wireless microphone devices with HowdyVox.

## Overview

The ESP32P4 HowdyScreen integration allows you to use wireless microphone devices instead of local USB microphones. The system supports:

- **Real-time audio streaming** via UDP with OPUS compression
- **Multi-room support** with device discovery and assignment
- **Visual feedback** through the ESP32P4's display and LED ring
- **Seamless integration** with existing HowdyVox VAD and processing

## Architecture

```
ESP32P4 HowdyScreen Device          HowdyVox Server
┌─────────────────────────┐        ┌─────────────────────────┐
│ • Microphone capture    │  UDP   │ • Wireless Audio Server │
│ • OPUS encoding         │ ────── │ • Device Manager        │
│ • LED ring feedback     │  8000  │ • VAD & Processing      │
│ • Round display UI      │        │ • TTS Response          │
└─────────────────────────┘        └─────────────────────────┘
```

## Quick Start

### 1. ESP32P4 Setup

1. **Configure your ESP32P4 device** in `/main/main.c`:
   ```c
   #define WIFI_SSID     "your_wifi_network"
   #define WIFI_PASSWORD "your_wifi_password"
   #define SERVER_IP     "192.168.1.100"  // HowdyVox server IP
   ```

2. **Build and flash** the ESP32P4 firmware:
   ```bash
   cd /path/to/ESP32P4/HowdyScreen
   idf.py build
   idf.py -p /dev/cu.usbserial-* flash monitor
   ```

### 2. HowdyVox Setup

1. **Install additional dependencies**:
   ```bash
   pip install opuslib websocket-client
   ```

2. **Run with wireless support**:
   ```bash
   # Use any available wireless device
   python run_voice_assistant.py --wireless
   
   # Target specific room
   python run_voice_assistant.py --room "Living Room"
   
   # List available devices
   python run_voice_assistant.py --list-devices
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

## Audio Pipeline Integration

The wireless audio integration maintains compatibility with HowdyVox's existing audio processing:

### Voice Activity Detection (VAD)
- Uses the same **Silero neural VAD** as local microphones
- **Intelligent utterance detection** for natural conversation flow
- **Pre-speech buffering** to capture speech beginnings

### Audio Quality
- **16kHz mono audio** streaming
- **OPUS compression** for minimal latency (2-5ms)
- **Automatic gain control** and noise suppression on ESP32P4

### Processing Flow
```
ESP32P4 Device → UDP/OPUS → HowdyVox → VAD → Transcription → LLM → TTS → ESP32P4 Speaker
```

## Network Configuration

### Firewall Settings
Ensure these ports are open on your HowdyVox server:
- **UDP 8000**: Audio streaming from ESP32P4 devices
- **UDP 8001**: Device discovery broadcasts

### WiFi Optimization
For best audio quality:
- Use **5GHz WiFi** for lower latency
- Ensure **strong signal strength** (-50dBm or better)
- **Disable power saving** on ESP32P4 devices
- Consider **QoS settings** for audio traffic

## Troubleshooting

### No Devices Found
1. **Check network connectivity** - ESP32P4 and HowdyVox on same network
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
4. **Restart devices** - power cycle ESP32P4 and restart HowdyVox

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

### HowdyVox Optimization
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