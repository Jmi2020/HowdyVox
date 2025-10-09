# HowdyVox Quick Start Guide

## Single Command Launch 🚀

Instead of managing two terminals, use the unified launcher:

```bash
# The launcher auto-detects and uses your conda environment!
python launch_howdy_shell.py
```

**No need to manually activate conda** - the launcher will:
1. Auto-detect your conda environment from:
   - Current active environment
   - `environment.yml` file  
   - Common environment names (`howdytts`, `howdy`, etc.)
2. Automatically run both FastWhisperAPI and HowdyVox in that environment

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

While HowdyVox is running, use these hotkeys to switch audio sources:

- **`Ctrl+Alt+L`** - Switch to local microphone
- **`Ctrl+Alt+W`** - Switch to wireless microphone  
- **`Ctrl+Alt+T`** - Toggle between sources
- **`Ctrl+Alt+I`** - Show current audio source info
- **`Ctrl+Alt+D`** - List available wireless devices

## ESP32P4 Setup

1. **Configure your ESP32P4** in `/main/main.c`:
   ```c
   #define WIFI_SSID     "your_network"
   #define WIFI_PASSWORD "your_password"
   #define SERVER_IP     "192.168.1.100"  // Your computer's IP
   ```

2. **Build and flash**:
   ```bash
   cd /path/to/ESP32P4/HowdyScreen
   idf.py build
   idf.py flash monitor
   ```

3. **The device will auto-discover** and appear in HowdyVox

## What's Improved

### Before ❌
- Required 2 terminals
- Manual FastAPI startup
- Complex wireless setup
- No runtime switching

### Now ✅  
- **Single command launch** - Everything starts automatically
- **Interactive audio selection** - Choose source at startup
- **Runtime hotkey switching** - Change sources while running
- **Minimal overhead** - Lazy initialization of wireless components
- **Auto-discovery** - ESP32P4 devices appear automatically

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