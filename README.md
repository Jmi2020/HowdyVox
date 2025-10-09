# HowdyVox - Your Local Conversational AI Companion 🎙️

Welcome to HowdyVox - a fully local, privacy-first conversational AI system that runs entirely on your machine. No cloud services, no data leaks, no subscriptions. Just you, your voice, and an AI personality that's completely under your control.

## What Makes HowdyVox Different 🌟

### Complete Privacy & Local Operation
Every component runs on your machine. Your conversations never leave your computer. No internet connection required once set up.

### Plug-and-Play Model Architecture
- **Bring Your Own LLM**: Works with any Ollama-compatible model (Gemma 3, Llama, Mistral, etc.)
- **Customizable Personality**: Edit the `SYSTEM_PROMPT` in `voice_assistant/config.py` to create any personality you want
- **Voice Flexibility**: Choose from 20+ built-in voices or blend them to create your own unique voice
- **Swap Models On-The-Fly**: Change the LLM model without restarting - just update `OLLAMA_LLM` in config

### Optimized for Low-Latency Conversations
- **Model Preloading**: Both TTS and LLM models load at startup for instant first responses
- **Adaptive Audio Chunking**: Automatically adjusts processing based on response length
- **Intelligent Buffering**: Pre-buffers audio to eliminate stuttering and gaps
- **Memory-Optimized**: Targeted garbage collection prevents memory leaks during long conversations

### Natural Conversation Flow
- **Wake Word Activation**: Say "Hey Howdy" to start, then chat naturally
- **Context Awareness**: Maintains conversation context until you explicitly end the session
- **Intelligent VAD**: Neural network-based voice activity detection knows when you've finished speaking
- **Multi-Room Support**: Configure USB microphones for different physical locations

### Developer-Friendly
- **Comprehensive Testing Suite**: 15+ diagnostic scripts to verify every component
- **Automated Fixes**: Run `fix_all_issues.py` to automatically diagnose and repair common problems
- **Modular Architecture**: Each component (STT, LLM, TTS) is independently replaceable
- **Extensive Documentation**: Detailed guides for every feature and troubleshooting scenario

## How It Works: The Mechanics 🔧

HowdyVox orchestrates multiple components working in harmony to create a seamless conversational experience. Here's what happens under the hood:

### The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. WAKE WORD DETECTION (Porcupine)                                 │
│    ↓ Listens continuously for "Hey Howdy"                          │
├─────────────────────────────────────────────────────────────────────┤
│ 2. VOICE ACTIVITY DETECTION (Silero Neural VAD)                    │
│    ↓ Intelligently detects when you start and stop speaking        │
├─────────────────────────────────────────────────────────────────────┤
│ 3. SPEECH-TO-TEXT (FastWhisperAPI - Local)                         │
│    ↓ Transcribes your audio to text                                │
├─────────────────────────────────────────────────────────────────────┤
│ 4. LANGUAGE MODEL (Ollama - Your Choice of Model)                  │
│    ↓ Generates response using your custom SYSTEM_PROMPT            │
├─────────────────────────────────────────────────────────────────────┤
│ 5. TEXT-TO-SPEECH (Kokoro ONNX - Your Choice of Voice)            │
│    ↓ Converts response to natural-sounding speech                  │
├─────────────────────────────────────────────────────────────────────┤
│ 6. AUDIO PLAYBACK                                                   │
│    ↓ Streams audio with adaptive chunking for smooth delivery      │
└─────────────────────────────────────────────────────────────────────┘
```

### The Startup Sequence

When you launch HowdyVox (via `launch_howdy_terminal.py`), here's what happens:

1. **Port Cleanup**: Checks for any existing FastWhisperAPI processes on port 8000 and terminates them
   ```python
   # Ensures clean startup without port conflicts
   lsof -ti :8000 | xargs kill -9
   ```

2. **FastWhisperAPI Launch**: Starts the local speech recognition server in the background
   ```bash
   conda run -n howdy310 uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   - Runs in a separate process group for independent lifecycle management
   - Uses `--no-capture-output` to preserve all logging output
   - Initializes in approximately 8 seconds

3. **Model Preloading**: While starting, `run_voice_assistant.py` performs critical optimizations:
   ```python
   # Preload Kokoro TTS models into memory
   kokoro_manager = KokoroManager()

   # Preload Ollama LLM model to avoid cold start
   preload_ollama_model(Config.OLLAMA_LLM)
   ```
   This means your **first response is just as fast as your tenth** - no waiting for models to load.

4. **Wake Word Listener**: Porcupine begins listening for "Hey Howdy"
   - Low CPU usage while idle
   - No recording until wake word detected
   - Privacy-preserving: audio is discarded if no wake word

5. **Conversation Loop**: Once activated, the system enters an intelligent conversation mode:
   ```python
   while conversation_active:
       # Record with intelligent VAD
       audio = record_audio_with_vad()

       # Transcribe locally
       text = transcribe_audio(audio)

       # Generate with your custom personality
       response = generate_response(text, SYSTEM_PROMPT)

       # Speak with adaptive chunking
       text_to_speech(response, adaptive_chunks=True)

       # Check for exit phrases
       if check_end_conversation(text):
           break
   ```

### Customizing Your AI's Personality

The magic happens in `voice_assistant/config.py`:

```python
# Your LLM of choice - swap anytime
OLLAMA_LLM = "hf.co/unsloth/gemma-3-4b-it-GGUF:latest"

# Your voice style - 20+ options or create blends
KOKORO_VOICE = 'am_michael'

# Your AI's personality - edit to match your needs
SYSTEM_PROMPT = (
    "You are George Carlin and Rodney Carrington as a single entity. "
    "Keep responses concise unless depth is essential. "
    "Maintain a neutral or lightly wry tone..."
)
```

Want a philosophical AI? A technical expert? A witty companion? Just edit the `SYSTEM_PROMPT`. The system will maintain this personality throughout your conversations while adapting to your questions and context.

### Audio Optimization: Eliminating Stuttering

HowdyVox uses **adaptive chunk sizing** to ensure smooth audio playback:

- **Short responses** (<100 chars): 150-char chunks with 50ms delays
- **Medium responses** (100-500 chars): 180-char chunks with 100ms delays
- **Long responses** (>500 chars): 220-char chunks with 150ms delays

The system pre-buffers chunks in the background while playing earlier chunks, creating seamless audio delivery even on longer responses. Combined with targeted garbage collection of audio buffers, you get stable performance during marathon conversation sessions.

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
git clone https://github.com/Jmi2020/HowdyVox.git
cd HowdyVox
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

### 6. Download Kokoro TTS models:

HowdyVox uses Kokoro ONNX for high-quality, local text-to-speech. You need to download the voice models:

**Option 1: Automatic Download (Recommended)**
```bash
pip install kokoro-onnx
```
This will automatically download models to `~/.kokoro_onnx/` on first use.

**Option 2: Manual Download**
If you want to use a specific model version or install locally:
```bash
python Tests_Fixes/download_kokoro_onnx_direct.py --type q8
```

**Available voice models:**
- `am_michael` - American male (default "cowboy" voice)
- `af_bella`, `af_nicole`, `af_sarah` - American female voices
- `bf_emma`, `bf_isabella` - British female voices
- `bm_lewis`, `bm_george` - British male voices
- Plus 15+ additional voices in various languages

List all available voices:
```bash
python blend_voices.py --list-voices
```

The models are lightweight (typically 50-100MB per voice) and run entirely offline.

### 7. Start the FastWhisperAPI service:
```bash
cd FastWhisperAPI
pip install -r requirements.txt
uvicorn main:app --reload
```

### 8. Install and configure Ollama (Language Model):

HowdyVox uses Ollama to run local language models. This is where your AI's "brain" lives.

**a. Download and install Ollama:**
- Visit [ollama.com](https://ollama.com/)
- Download for your platform (macOS, Linux, Windows)
- Run the installer

**b. Browse and choose your LLM:**
Visit the [Ollama Library](https://ollama.com/library) to explore 100+ models:

**Popular choices for conversational AI:**
- **Gemma 3** (4B) - Fast, efficient, great for conversation ⭐ Recommended
  ```bash
  ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:latest
  ```
- **Llama 3.2** (3B) - Meta's latest, excellent quality
  ```bash
  ollama pull llama3.2:latest
  ```
- **Mistral** (7B) - Powerful reasoning, slightly slower
  ```bash
  ollama pull mistral:latest
  ```
- **DeepSeek-R1** (7B) - Advanced reasoning capabilities
  ```bash
  ollama pull deepseek-r1:7b
  ```
- **Phi-3** (3.8B) - Microsoft's compact, fast model
  ```bash
  ollama pull phi3:latest
  ```

**Model size guide:**
- **3-4B parameters**: Fast responses, good quality, low RAM (~4-6GB)
- **7-8B parameters**: Better quality, slower, moderate RAM (~8-12GB)
- **13B+ parameters**: Highest quality, slowest, high RAM (~16GB+)

**c. Test your model:**
```bash
ollama run gemma-3-4b-it-GGUF:latest
```
Type a test question, then Ctrl+D to exit.

**d. Update HowdyVox to use your chosen model:**
Edit `voice_assistant/config.py` and set:
```python
OLLAMA_LLM = "llama3.2:latest"  # Or your chosen model
```

### 9. (Optional) For Apple Silicon Macs - Enhanced ONNX Performance:
```bash
# Run the ONNX optimization script for better performance
python Tests_Fixes/fix_onnx_runtime.py
```

## Running HowdyVox 🎙️

### Recommended: Terminal Launcher (One Command)
```bash
python launch_scripts_backup/launch_howdy_terminal.py
```

This unified launcher:
- Automatically kills any existing FastWhisperAPI processes
- Starts FastWhisperAPI in the background
- Launches the voice assistant in the foreground
- Handles cleanup when you exit (Ctrl+C)
- Preserves all logging output for debugging

**Configuring your conda environment:**

The launcher uses environment variables for flexibility:
```bash
# Set your conda environment name (default: howdy310)
export HOWDYVOX_CONDA_ENV="your-env-name"

# Set your conda path (default: /opt/anaconda3/bin/conda)
export CONDA_PATH="/path/to/your/conda"

# Then run the launcher
python launch_scripts_backup/launch_howdy_terminal.py
```

Or edit the script directly at the top:
```python
CONDA_ENV = "your-env-name"  # Change from default "howdy310"
CONDA_PATH = "/path/to/conda"  # Change if needed
```

### Alternative: Manual Two-Terminal Approach
```bash
# Terminal 1: Start FastWhisperAPI
cd FastWhisperAPI
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2: Start voice assistant
python run_voice_assistant.py
```

### What Happens When You Run It

1. **Model Preloading** (10-15 seconds)
   - Kokoro TTS voice models load into memory
   - Ollama LLM model initializes
   - This one-time startup cost ensures instant responses later

2. **Wake Word Detection**
   - System displays: "Listening for wake word 'Hey Howdy'..."
   - Porcupine is now actively listening
   - Say "Hey Howdy" to activate

3. **Conversation Mode**
   - LED indicator (if configured) shows "Listening"
   - Speak naturally - Silero VAD detects when you're done
   - Your audio transcribes locally via FastWhisperAPI
   - Ollama generates a response using your `SYSTEM_PROMPT` personality
   - Kokoro TTS speaks the response with your chosen voice
   - Conversation continues until you say "goodbye", "that's all", etc.

4. **Context Retention**
   - Each turn remembers previous exchanges
   - Ask follow-up questions naturally
   - No need to repeat "Hey Howdy" between turns
   - Context clears only when you explicitly end the conversation

### Enhanced Performance Features:

- **Model Preloading**: Both TTS and LLM models load before wake word detection starts
- **Adaptive TTS Chunking**: Automatically adjusts chunk sizes based on response length
- **Memory Management**: Targeted garbage collection prevents memory leaks
- **Enhanced Audio Processing**: Improved recording and playback with reduced stuttering

## Customization & Configuration ⚙️

HowdyVox is designed to be completely customizable. Everything is controlled through `voice_assistant/config.py`.

### Choose Your Language Model

```python
# Any Ollama-compatible model works
OLLAMA_LLM = "hf.co/unsloth/gemma-3-4b-it-GGUF:latest"  # Fast, compact
# OLLAMA_LLM = "llama3.2:latest"                        # Meta's Llama
# OLLAMA_LLM = "mistral:latest"                         # Mistral AI
# OLLAMA_LLM = "deepseek-r1:latest"                     # DeepSeek reasoning
```

First, pull your chosen model via Ollama:
```bash
ollama pull llama3.2:latest
```

Then update `OLLAMA_LLM` in `config.py` and restart HowdyVox. That's it.

### Design Your AI's Personality

The `SYSTEM_PROMPT` defines how your AI thinks and responds. The current default is a blend of George Carlin and Rodney Carrington - witty, direct, occasionally dark:

```python
SYSTEM_PROMPT = (
    "You are George Carlin and Rodney Carrington as a single entity. "
    "Keep responses concise unless depth is essential. "
    "Maintain a neutral or lightly wry tone, using dark humor sparingly..."
)
```

**Want something different?** Edit this to anything:

```python
# A philosophical guide
SYSTEM_PROMPT = "You are Socrates, asking probing questions to help the user think deeply."

# A technical expert
SYSTEM_PROMPT = "You are a senior software engineer with 20 years of experience. Be precise and cite best practices."

# A supportive friend
SYSTEM_PROMPT = "You are a warm, encouraging friend who celebrates wins and offers perspective during challenges."
```

The AI will embody this personality in all conversations while still adapting to your specific questions and context.

### Select Your Voice

```python
KOKORO_VOICE = 'am_michael'  # American male voice (default)
```

Choose from 20+ built-in voices:
- `af_bella`, `af_nicole`, `af_sarah` - American female voices
- `am_michael`, `am_eric`, `am_adam` - American male voices
- `bf_emma`, `bf_isabella` - British female voices
- `bm_lewis`, `bm_george` - British male voices
- And many more...

List all available voices:
```bash
python blend_voices.py --list-voices
```

### Model Configuration
```python
TRANSCRIPTION_MODEL = 'fastwhisperapi'  # Local Whisper API
RESPONSE_MODEL = 'ollama'               # Ollama LLM runtime
TTS_MODEL = 'kokoro'                    # Kokoro ONNX TTS
```

These model backends are the foundation of HowdyVox's offline-first architecture.

## Voice Blending 🎭

HowdyVox supports voice blending, allowing you to create custom voices by combining multiple voice styles with different ratios. This lets you fine-tune the perfect voice for your assistant.

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

HowdyVox supports connecting to an ESP32-S3 LED matrix display to show visual feedback of the assistant's state:

- **Waiting**: Displayed when waiting for wake word
- **Listening**: Shown when listening for your command
- **Thinking**: Displayed when generating a response
- **Speaking**: Shows the response text scrolling across the display
- **Ending**: Appears when ending a conversation

### Setting up the LED Matrix

1. Flash your ESP32-S3 with the HowdyVox LED Matrix firmware (see [ESP32 directory](ESP32/))
2. Connect your ESP32-S3 to your WiFi network
3. Note the IP address assigned to your ESP32-S3
4. Add the ESP32 IP to your `.env` file:

```
ESP32_IP=192.168.1.xxx  # Replace with your ESP32's IP address
```

The LED matrix will automatically be used if the ESP32_IP environment variable is set.

## Recent Improvements & Technical Details 🚀

### TTS Stuttering Fix Implementation
HowdyVox now includes comprehensive fixes for TTS stuttering issues:

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

HowdyVox includes a comprehensive testing suite to help diagnose and fix issues:

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

## Why HowdyVox? 🤔

In an era where AI assistants require constant internet connectivity and send your conversations to distant servers, HowdyVox takes a different approach:

- **Privacy First**: Your thoughts, questions, and conversations stay on your machine. Period.
- **Truly Yours**: Customize every aspect - the voice, the personality, the language model. Make it reflect your preferences, not a corporation's.
- **No Subscriptions**: No monthly fees, no API costs, no rate limits. Once set up, it's yours forever.
- **Transparent**: Every component is open source. You can see exactly how it works, modify it, improve it.
- **Offline Capable**: No internet? No problem. HowdyVox works anywhere, anytime.

HowdyVox proves that powerful AI assistants don't need to compromise your privacy or your wallet. It's conversational AI done right - local, fast, and completely under your control.

## Contributing 🤝

This project welcomes contributions! Whether it's:
- Bug fixes and improvements
- New voice personalities
- Additional language model integrations
- Documentation enhancements
- Performance optimizations

Feel free to open issues or submit pull requests on GitHub.

## License 📄

MIT License - See LICENSE file for details

---

*"Your AI, your rules. Welcome to HowdyVox."* 🎙️


