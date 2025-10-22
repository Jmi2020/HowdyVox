# M3 Mac Setup Guide for HowdyVox

This guide provides specific instructions for setting up HowdyVox on M3 Mac (Apple Silicon) systems.

## Prerequisites

- macOS with M3 chip (Apple Silicon)
- [Homebrew](https://brew.sh/) package manager
- [Anaconda](https://www.anaconda.com/) or Miniconda
- Terminal access

## Issues on M3 Mac

The M3 Mac architecture requires specific setup steps that differ from Intel Macs:

1. **PyAudio Compilation**: PyAudio needs PortAudio library and proper compilation flags
2. **Opus Library Path**: The opuslib Python package needs help finding the Opus library for wireless audio support

## Installation Steps

### 1. Install System Libraries via Homebrew

```bash
# Install PortAudio (required for PyAudio)
brew install portaudio

# Install Opus (required for wireless audio with ESP32P4)
brew install opus
```

### 2. Create and Configure Conda Environment

```bash
# Create Python 3.10 environment
conda create -n howdy310 python=3.10

# Activate the environment
conda activate howdy310
```

### 3. Install PyAudio with Proper Compilation Flags

PyAudio requires special compilation flags to find PortAudio on M3 Mac:

```bash
pip install --no-cache-dir --force-reinstall \
  --global-option='build_ext' \
  --global-option="-I$(brew --prefix portaudio)/include" \
  --global-option="-L$(brew --prefix portaudio)/lib" \
  pyaudio
```

**Note**: You may see deprecation warnings about `--global-option`. This is expected and can be ignored.

### 4. Install All Other Requirements

```bash
pip install -r requirements.txt
```

This will install all dependencies including:
- Kokoro TTS with ONNX Runtime (optimized for Apple Silicon)
- Silero VAD with torchaudio
- PyObjC frameworks for macOS voice isolation
- Ollama, FastAPI, and other core dependencies

### 5. Set Up Environment Variables

Copy the example environment file:

```bash
cp example.env .env
```

Edit `.env` and add your Porcupine access key:

```env
PORCUPINE_ACCESS_KEY="your-porcupine-key-here"
LOCAL_MODEL_PATH="models"
ESP32_IP="192.168.1.xxx"  # Optional: LED matrix IP
```

Get a free Porcupine key from [picovoice.ai](https://picovoice.ai)

### 6. Run Quick Setup

```bash
python quick_setup.py
```

This will:
- Download the "Hey Howdy" wake word model
- Set up NLTK resources
- Verify your Porcupine access key

## Running HowdyVox on M3 Mac

### Option 1: Use the M3-Specific Launcher (Recommended)

```bash
./launch_howdy_m3.sh
```

This script automatically sets the required `DYLD_LIBRARY_PATH` for Opus library support.

### Option 2: Set Environment Variable Manually

If you prefer to run commands directly:

```bash
# Set the library path
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH

# Activate conda environment
conda activate howdy310

# Run the shell launcher
python launch_howdy_shell.py

# Or run the UI launcher
python launch_howdy_ui.py

# Or run directly
python run_voice_assistant.py
```

### Option 3: Add to Shell Profile (Permanent)

Add this to your `~/.zshrc` or `~/.bash_profile`:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH
```

Then you can run normally after `conda activate howdy310`.

## Verification

To verify everything is working correctly:

```bash
# Activate environment
conda activate howdy310

# Set library path
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH

# Test imports
python -c "import pyaudio; print('✓ PyAudio OK')"
python -c "import opuslib; print('✓ Opuslib OK')"
python -c "import onnxruntime; print('✓ ONNX Runtime OK')"
python -c "from voice_assistant.kokoro_manager import KokoroManager; print('✓ Kokoro TTS OK')"
```

All tests should pass without errors.

## M3-Specific Optimizations

HowdyVox automatically detects and enables optimizations for M3 Mac:

### ONNX Runtime with CoreML
- Uses CoreML execution provider for faster inference
- Automatically enabled when Apple Silicon detected
- Provides 2-3x speed improvement for Kokoro TTS

### macOS Voice Isolation
- Native macOS audio processing for noise reduction
- Requires macOS 12.0 or later
- Automatically enabled in `voice_assistant/config.py`

### PyTorch with MPS Backend
- Uses Metal Performance Shaders for GPU acceleration
- Automatically used by torchaudio for Silero VAD
- Provides significant speed boost for voice activity detection

## Troubleshooting

### PyAudio Build Fails

**Error**: `fatal error: 'portaudio.h' file not found`

**Solution**:
```bash
brew install portaudio
# Then retry PyAudio installation with compilation flags (see step 3)
```

### Opus Library Not Found

**Error**: `Exception: Could not find Opus library`

**Solution**:
```bash
# Install Opus
brew install opus

# Set library path
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH
```

### Conda Command Not Found

**Error**: `conda: command not found`

**Solution**:
```bash
# Initialize conda for your shell
/opt/anaconda3/bin/conda init zsh  # or bash

# Restart your terminal
# Then retry
```

### Permission Denied on Microphone

**Error**: Microphone access errors

**Solution**:
1. Open **System Settings** → **Privacy & Security** → **Microphone**
2. Enable microphone access for **Terminal** or **iTerm2**
3. Restart your terminal application

### ONNX Runtime Not Using CoreML

**Solution**:
```bash
# Verify Apple Silicon detection
python -c "import platform; print(platform.machine())"
# Should output: arm64

# Check ONNX providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should include: CoreMLExecutionProvider
```

## Performance Notes

On M3 Mac with proper setup:

- **First response**: ~1.5-2 seconds (model loading + inference)
- **Subsequent responses**: ~0.5-1 second (inference only)
- **Wake word detection**: <100ms latency
- **Voice isolation**: Real-time processing with <50ms latency

## Additional Resources

- [Main README](README.md) - Full project documentation
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [CLAUDE.md](CLAUDE.md) - Development guide
- [Homebrew](https://brew.sh/) - Package manager for macOS
- [Anaconda](https://www.anaconda.com/) - Python distribution

## Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](README.md) troubleshooting section
2. Verify all installation steps were completed
3. Check the logs for specific error messages
4. Try running the diagnostic tests: `python Tests_Fixes/check_components.py`

## Notes for Developers

When developing on M3 Mac, remember:

1. Always activate the conda environment: `conda activate howdy310`
2. Set `DYLD_LIBRARY_PATH` before running: `export DYLD_LIBRARY_PATH=/opt/homebrew/opt/opus/lib:$DYLD_LIBRARY_PATH`
3. PyAudio version is pinned to 0.2.14 for M3 compatibility
4. ONNX Runtime automatically uses CoreML on M3 - no configuration needed
5. Voice isolation requires macOS 12.0+ and uses native AVFoundation APIs

---

**Last Updated**: October 2025
**Tested On**: M3 Mac Studio, macOS Sequoia 15.0
