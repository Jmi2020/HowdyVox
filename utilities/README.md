# Utilities

This folder contains utility scripts and tools for HowdyVox configuration and testing.

## Voice Blending

- **`blend_voices.py`** - Create custom voice blends
- **`configure_blended_voice.py`** - Configure blended voice settings
- **`SupportedVoices.txt`** - List of available Kokoro voices
- **`VoiceBlending.txt`** - Voice blending documentation

## Microphone Setup

- **`microphone_manager.py`** - Manage multiple microphones
- **`microphone_test.py`** - Test microphone functionality
- **`setup_microphones.py`** - Configure multi-room microphone setup

## System Setup

- **`mac_setup.py`** - macOS-specific setup script
- **`setup_mac_voice_isolation.sh`** - Setup voice isolation on macOS
- **`setup.py`** - General setup utilities

## Other Tools

- **`create_rounded_icon.py`** - Create rounded app icons
- **`run_voice_assistant_supervisor.py`** - Supervisor for voice assistant process

## Audio Files

- **`activate.wav`** - Wake word activation sound
- **`blended_audio.wav`** - Sample blended voice output
- **`test.mp3`** - Audio test file

## Usage Examples

### Create a Custom Voice Blend
```bash
python utilities/blend_voices.py --voices "af_bella:40,am_michael:60" --text "Testing my custom voice"
```

### Test Your Microphone
```bash
python utilities/microphone_test.py
```

### Configure Multi-Room Setup
```bash
python utilities/setup_microphones.py
```

## Navigation

- Back to [Main README](../README.md)
- See [docs/](../docs/) for documentation
- See [face_modules/](../face_modules/) for face implementations
