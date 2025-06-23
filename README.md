# HowdyTTS - A Voice Assistant with Cowboy Charm 🤠

Howdy, partner! I'm your friendly neighborhood voice assistant with a bit o' cowboy twang. I'm 100% offline, so you don't need to worry about them internet varmints listenin' in on our conversations!

## Features 🌟

- **Wake Word Detection**: Just say "Hey Howdy" to activate me
- **100% Offline Operation**: No cloud services required, everythin' runs right on your machine
- **Cowboy Voice**: Uses Kokoro TTS with the 'am_michael' voice model for that authentic cowboy sound
- **Fast Local Speech Recognition**: Powered by FastWhisperAPI
- **Local LLM Support**: Integrated with Ollama for text generation (works with Gemma 3 or any model of your choice)
- **Continuous Conversation**: Chat naturally without saying the wake word for each interaction
- **Voice Blending**: Create custom voices by blending multiple voice styles together (see [Voice Blending](#voice-blending-))
- **Enhanced TTS Performance**: Advanced stuttering fixes with adaptive chunk sizing and response-aware delays
- **Comprehensive Testing Suite**: Built-in testing scripts for all components and troubleshooting tools
- **Multi-Room Support**: Microphone manager for identifying and configuring multiple USB microphones
- **Memory Optimization**: Targeted garbage collection and audio buffer pooling for stable performance
- **Model Preloading**: Both Kokoro TTS and Ollama LLM models are preloaded at startup for faster response times

## Prerequisites ✅

- Python 3.10 or higher
- Virtual environment (recommended)
- PyAudio 0.2.12 (specifically this version for macOS compatibility)
- [Ollama](https://ollama.com/) installed and running locally
- [Porcupine](https://picovoice.ai/platform/porcupine/) access key (free for personal use)
- CUDA-capable GPU (optional, improves performance)
- For Apple Silicon Macs: Enhanced ONNX Runtime support for optimized performance

## Setup Instructions 🔧

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/HowdyTTS.git
cd HowdyTTS
```

### 2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Install PyAudio 0.2.12 specifically:
```bash
pip uninstall -y pyaudio
pip install pyaudio==0.2.12
```

### 5. Set up Porcupine wake word detection:

a. Get a free Porcupine access key from [Picovoice Console](https://console.picovoice.ai/)
b. Run the quick setup script to configure wake word detection:
```bash
python quick_setup.py
```
c. Or manually create a `.env` file in the project root with:
```
PORCUPINE_ACCESS_KEY="your-access-key-here"
LOCAL_MODEL_PATH="models"
ESP32_IP="192.168.1.xxx"  # Optional: For LED matrix display
```

### 6. Start the FastWhisperAPI service:
```bash
cd FastWhisperAPI
pip install -r requirements.txt
uvicorn main:app --reload
```

### 7. Install and start Ollama:
a. Download Ollama from [ollama.com](https://ollama.com/)
b. Start Ollama and pull your preferred model:
```bash
ollama run hf.co/unsloth/gemma-3-4b-it-GGUF:latest
```
You can substitute with any model of your choice.

### 8. (Optional) For Apple Silicon Macs - Enhanced ONNX Performance:
```bash
# Run the ONNX optimization script for better performance
python Tests_Fixes/fix_onnx_runtime.py
```

## Running the Assistant 🤠

Start the voice assistant:
```bash
python run_voice_assistant.py
```

The assistant will:
1. **Initialize Models**: Preload both Kokoro TTS and Ollama LLM models for faster response times
2. **Start Wake Word Detection**: Wait for the wake word "Hey Howdy" 
3. **Listen**: Record your voice input with enhanced audio processing
4. **Transcribe**: Convert speech to text using FastWhisperAPI
5. **Generate Response**: Create a cowboy-themed response using Ollama
6. **Speak**: Deliver the response using optimized Kokoro TTS with stuttering fixes

The conversation will continue until you say phrases like "goodbye", "that's all", or "thanks, that's all".

### Enhanced Performance Features:

- **Model Preloading**: Both TTS and LLM models load before wake word detection starts
- **Adaptive TTS Chunking**: Automatically adjusts chunk sizes based on response length
- **Memory Management**: Targeted garbage collection prevents memory leaks
- **Enhanced Audio Processing**: Improved recording and playback with reduced stuttering

## Configuration ⚙️

The system uses centralized configuration in `voice_assistant/config.py`. Key settings include:

```python
TRANSCRIPTION_MODEL = 'fastwhisperapi'  # Local speech recognition
RESPONSE_MODEL = 'ollama'               # Local LLM using Ollama
TTS_MODEL = 'kokoro'                    # Local TTS with cowboy voice
KOKORO_VOICE = 'am_michael'             # Default cowboy voice
OLLAMA_LLM = "hf.co/unsloth/gemma-3-4b-it-GGUF:latest"  # Default LLM model
```

## Voice Blending 🎭

HowdyTTS supports voice blending, allowing you to create custom voices by combining multiple voice styles with different ratios. This lets you fine-tune the perfect voice for your assistant.

### Quick Example
```bash
# Create a blend of two voices (40% Bella, 60% Michael)
python configure_blended_voice.py --name "my_custom_voice" --voices "af_bella:40,am_michael:60"

# Update config.py to use this voice
# KOKORO_VOICE = 'my_custom_voice'
```

Voice blending opens up endless possibilities for customization - from subtle voice tweaks to completely new voice personalities.

For detailed instructions and advanced usage, see the [VoiceBlend.md](VoiceBlend.md) guide.

## LED Matrix Display (Optional) 🌟

HowdyTTS supports connecting to an ESP32-S3 LED matrix display to show visual feedback of the assistant's state:

- **Waiting**: Displayed when waiting for wake word
- **Listening**: Shown when listening for your command
- **Thinking**: Displayed when generating a response
- **Speaking**: Shows the response text scrolling across the display
- **Ending**: Appears when ending a conversation

### Setting up the LED Matrix

1. Flash your ESP32-S3 with the HowdyTTS LED Matrix firmware (see [ESP32 directory](ESP32/))
2. Connect your ESP32-S3 to your WiFi network
3. Note the IP address assigned to your ESP32-S3
4. Add the ESP32 IP to your `.env` file:

```
ESP32_IP=192.168.1.xxx  # Replace with your ESP32's IP address
```

The LED matrix will automatically be used if the ESP32_IP environment variable is set.

## Recent Improvements & Technical Details 🚀

### TTS Stuttering Fix Implementation
HowdyTTS now includes comprehensive fixes for TTS stuttering issues:

- **Adaptive Chunk Sizing**: Automatically adjusts chunk sizes based on response length
  - Short texts (<100 chars): 150 character chunks
  - Medium texts (100-500 chars): 180 character chunks  
  - Long texts (>500 chars): 220 character chunks
- **Response-Aware Delays**: Stabilization delays based on response complexity
- **Enhanced Buffer Management**: Smarter pre-buffering and queue monitoring
- **Inter-chunk Gap Detection**: Automatically handles generation delays

### Memory Management & Performance
- **Targeted Garbage Collection**: Specifically cleans up audio-related objects
- **Audio Buffer Pooling**: Optimized memory usage for audio processing
- **Model Preloading**: Both Kokoro TTS and Ollama LLM models load at startup
- **Resource Cleanup**: Enhanced cleanup after conversations and errors

### Multi-Room Support
- **Microphone Manager**: Identify USB microphones by physical location
- **Persistent Mappings**: Maintain room assignments across restarts
- **USB Location Tracking**: Uses system profiler data to track microphone positions

### Enhanced Testing & Diagnostics
- **Comprehensive Test Suite**: 15+ testing scripts for all components
- **Automated Fixes**: Scripts that automatically diagnose and fix common issues
- **Component Verification**: Individual tests for each system component
- **Performance Monitoring**: Tools to track TTS performance and memory usage

For detailed technical information, see:
- [TTS_STUTTERING_FIX_README.md](TTS_STUTTERING_FIX_README.md) - Complete stuttering fix implementation
- [TTS_ENHANCEMENT_IMPLEMENTATION.md](TTS_ENHANCEMENT_IMPLEMENTATION.md) - Performance enhancements
- [Tests_Fixes/test_and_run_instructions.md](Tests_Fixes/test_and_run_instructions.md) - Comprehensive testing guide

## Testing the Setup 🧪

HowdyTTS now includes a comprehensive testing suite to help diagnose and fix issues:

### Component Tests:
```bash
# Test your microphone setup
python microphone_test.py

# Test all system components
python Tests_Fixes/check_components.py

# Test FastWhisperAPI connection
python Tests_Fixes/check_fastwhisper.py

# Test Kokoro TTS with ONNX optimizations
python Tests_Fixes/test_kokoro_onnx.py

# Test Porcupine wake word detection
python Tests_Fixes/test_porcupine_fixes.py

# Test memory management and garbage collection
python Tests_Fixes/test_targeted_gc.py

# Test voice blending functionality
python Tests_Fixes/test_voice_blending.py
```

### Automated Fixes:
```bash
# Run comprehensive system check and auto-fix issues
python Tests_Fixes/fix_all_issues.py

# Fix specific components if needed
python Tests_Fixes/fix_kokoro_voice_files.py
python Tests_Fixes/fix_porcupine_issues.py
python Tests_Fixes/fix_onnx_runtime.py
```

### Multi-Room Setup:
```bash
# Identify and configure multiple USB microphones
python microphone_manager.py

# Set up microphones for different rooms
python setup_microphones.py
```

## Troubleshooting 🔧

### Common Issues and Solutions:

- **Wake word detection not working**: Run `python quick_setup.py` to configure Porcupine
- **PyAudio errors on macOS**: Make sure you're using version 0.2.12
- **FastWhisperAPI errors**: Ensure it's running at http://localhost:8000
- **No response from Ollama**: Check that Ollama is running with `ollama list` and your model is downloaded
- **TTS stuttering or audio gaps**: The system now includes automatic stuttering fixes with adaptive chunking
- **Memory issues**: Run `python Tests_Fixes/test_targeted_gc.py` to test garbage collection
- **No audio output**: Check your system sound settings and run `python microphone_test.py`
- **ONNX Runtime issues on Apple Silicon**: Run `python Tests_Fixes/fix_onnx_runtime.py`
- **Model loading slow**: Models are now preloaded at startup for faster response times

### Advanced Troubleshooting:

```bash
# Check all system components
python Tests_Fixes/check_components.py

# Check environment configuration  
python Tests_Fixes/check_environment.py

# Run comprehensive diagnostics and fixes
python Tests_Fixes/fix_all_issues.py
```

### Performance Optimization:

- **For Apple Silicon Macs**: Use the enhanced ONNX Runtime setup for better performance
- **Memory Management**: The system now includes targeted garbage collection for audio resources
- **TTS Performance**: Enhanced with adaptive chunk sizing and response-aware delays
- **Model Preloading**: Both Kokoro and Ollama models preload at startup to reduce first-response latency

## License 📄

MIT License - See LICENSE file for details

---

*"Remember, partner - I'm always here to help! Just holler 'Hey Howdy' when ya need me!"* 🤠


