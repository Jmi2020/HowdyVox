# HowdyTTS Quick Start Guide

## Single Command Launch 🚀

Launch the complete HowdyTTS system with ESP32-P4 wireless integration:

```bash
# Automatic launch with conda environment (howdy310)
python launch_howdy_shell.py
```

**This automatically:**
1. **Activates `howdy310` conda environment**
2. **Starts FastWhisperAPI server** (port 8000) 
3. **Launches HowdyTTS** with full ESP32-P4 integration
4. **Enables WebSocket feedback server** (port 8001)
5. **Shows output in real-time** for monitoring

### Advanced Launch Options

For direct control over audio sources:

```bash
# Use wireless ESP32-P4 devices (auto-discovery)
python run_voice_assistant.py --wireless

# Target specific room device
python run_voice_assistant.py --wireless --room "Living Room"

# Auto-detect best source (wireless first, local fallback)
python run_voice_assistant.py --auto

# List available ESP32-P4 devices
python run_voice_assistant.py --list-devices
```

### Manual conda environment specification:
```bash
# Specify a particular environment
python launch_howdy.py --conda-env your-env-name
```

## Audio Source Options

### Interactive Mode (Recommended)
```bash
python launch_howdy.py
```
- Shows a menu to choose audio source
- Lists available wireless devices
- Perfect for first-time setup

### Direct Audio Source Selection
```bash
# Use local microphone
python launch_howdy.py --local

# Use wireless ESP32P4 devices  
python launch_howdy.py --wireless

# Use specific room
python launch_howdy.py --room "Living Room"

# Auto-detect (try wireless, fallback to local)
python launch_howdy.py --auto
```

## Runtime Audio Switching 🎛️

While HowdyTTS is running, use these hotkeys to switch audio sources:

- **`Ctrl+Alt+L`** - Switch to local microphone
- **`Ctrl+Alt+W`** - Switch to wireless microphone  
- **`Ctrl+Alt+T`** - Toggle between sources
- **`Ctrl+Alt+I`** - Show current audio source info
- **`Ctrl+Alt+D`** - List available wireless devices

## ESP32-P4 HowdyScreen Setup (Phase 6C)

### Quick Setup Steps

1. **Configure your ESP32-P4** with Phase 6C firmware:
   ```c
   // In main/howdy_phase6_howdytts_integration.c
   #define WIFI_SSID     "your_network"
   #define WIFI_PASSWORD "your_password"
   #define SERVER_IP     "192.168.1.100"      // Your HowdyTTS server IP
   #define WAKE_WORD_ENABLED true              // Enable "Hey Howdy" detection
   ```

2. **Build and flash Phase 6C firmware**:
   ```bash
   cd /Users/silverlinings/Desktop/Coding/ESP32P4/HowdyScreen
   export IDF_TARGET=esp32p4
   idf.py build
   idf.py -p /dev/cu.usbserial-* flash monitor
   ```

3. **Verify integration** - You should see:
   ```
   I (12345) HOWDY: Enhanced VAD initialized
   I (12346) HOWDY: Wake word detection active - "Hey Howdy"  
   I (12347) HOWDY: WebSocket feedback client connected to 192.168.1.100:8001
   I (12348) HOWDY: UDP audio streaming to 192.168.1.100:8000
   ```

4. **Test wake word** - Say "Hey Howdy" and watch for:
   - ESP32-P4 round display animation
   - HowdyTTS server wake word confirmation
   - Bidirectional VAD coordination logs

### What's New in Phase 6C

- ✅ **Bidirectional VAD**: Edge (ESP32-P4) + Server (Silero) fusion
- ✅ **Wake Word Detection**: "Hey Howdy" with Porcupine server validation
- ✅ **WebSocket Feedback**: Real-time threshold adaptation (port 8001)
- ✅ **Enhanced UDP Protocol**: VERSION_WAKE_WORD (0x03) with 24-byte headers
- ✅ **Adaptive Learning**: System improves accuracy over time
- ✅ **Multi-device Support**: Room-based coordination

## System Architecture (Phase 6C)

### Complete Bidirectional Integration

```
ESP32-P4 HowdyScreen ←→ HowdyTTS Server
                     ↕
        Bidirectional Communication:
        • Enhanced UDP (port 8000) 
        • WebSocket feedback (port 8001)
        • VAD fusion & wake word validation
        • Real-time threshold adaptation
```

### What's New vs. Previous Phases

| Feature | Phase 1-5 | Phase 6C |
|---------|-----------|----------|
| **VAD** | Local only | Edge + Server fusion |
| **Wake Word** | Server only | Edge detection + Server validation |
| **Communication** | One-way UDP | Bidirectional UDP + WebSocket |
| **Learning** | Static | Adaptive thresholds |
| **Protocol** | Basic | Enhanced with 24-byte headers |
| **Multi-device** | None | Room-based coordination |

## Troubleshooting

### "No wireless devices found"
1. Check ESP32P4 is powered and connected to WiFi
2. Verify both devices are on same network
3. Check firewall allows UDP ports 8000/8001
4. Try `python launch_howdy.py --list-devices`

### Hotkeys not working
1. Install keyboard module: `pip install keyboard`
2. On macOS, grant terminal accessibility permissions
3. Try running as administrator/sudo if needed

### FastAPI startup issues
1. Check `fastwhisperapi/main.py` exists
2. Verify conda environment is activated
3. Install FastAPI: `pip install "fastapi[standard]"`

## Performance Notes

- **Local microphone**: Zero startup overhead
- **Wireless**: Lazy initialization only when first used
- **Hotkey switching**: Sub-second response time
- **Memory usage**: Minimal additional overhead vs. original system

## Development Workflow

```bash
# For development with frequent switching
python launch_howdy.py --auto

# Use hotkeys to switch between local and wireless during testing
# Ctrl+Alt+T to toggle sources quickly
```

The system intelligently manages resources - wireless components are only loaded when needed, and switching between sources is nearly instantaneous.