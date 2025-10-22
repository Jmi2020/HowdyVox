# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HowdyVox is a 100% offline voice assistant platform built on Python 3.10+. The system uses local-only models for speech recognition (FastWhisperAPI), text generation (Ollama), and text-to-speech (Kokoro TTS with ONNX). It features customizable personalities via the SYSTEM_PROMPT configuration and supports any Ollama-compatible language model. No cloud services are involved.

## Development Commands

### Starting the Assistant

```bash
# Shell launcher (recommended) - runs both services with single command
python launch_howdy_shell.py

# UI launcher - interactive terminal UI with animated face and visual feedback
python launch_howdy_ui.py

# With audio source options
python run_voice_assistant.py --wireless           # Use ESP32P4 wireless mics
python run_voice_assistant.py --room "Living Room" # Target specific room
python run_voice_assistant.py --list-devices       # Show available devices

# Traditional two-terminal approach
# Terminal 1: Start FastWhisperAPI
cd FastWhisperAPI
uvicorn main:app --reload

# Terminal 2: Start voice assistant
python run_voice_assistant.py
```

### Testing the Animated Face

```bash
# Test face animations without voice assistant
python test_face_ui.py

# Test face animator standalone with manual controls
python face_animator.py
```


### Testing

```bash
# Test all system components
python Tests_Fixes/check_components.py

# Run comprehensive diagnostics and fixes
python Tests_Fixes/fix_all_issues.py

# Component-specific tests
python microphone_test.py                      # Test microphone setup
python Tests_Fixes/check_fastwhisper.py        # Test FastWhisperAPI connection
python Tests_Fixes/test_kokoro_onnx.py         # Test Kokoro TTS with ONNX
python Tests_Fixes/test_porcupine_fixes.py     # Test wake word detection
python Tests_Fixes/test_targeted_gc.py         # Test memory management
python Tests_Fixes/test_voice_blending.py      # Test voice blending
python Tests_Fixes/test_tts_fix.py             # Test TTS stuttering fixes
```

### Multi-Room Setup

```bash
# Identify and configure multiple USB microphones
python microphone_manager.py

# Set up microphones for different rooms
python setup_microphones.py
```

### Voice Blending

```bash
# Create a blend of two voices (40% Bella, 60% Michael)
python configure_blended_voice.py --name "my_custom_voice" --voices "af_bella:40,am_michael:60"

# List available voices
python blend_voices.py --list-voices

# Test voice combinations
python blend_voices.py --voices "af_nicole:30,am_michael:70" --text "Hello, this is a test."
```

## Architecture & Key Components

### Core Configuration (`voice_assistant/config.py`)

Centralized configuration file that controls all model selection and system behavior. **This is the single source of truth for system settings.**

- **Model Selection**: All models are offline-only (`TRANSCRIPTION_MODEL`, `RESPONSE_MODEL`, `TTS_MODEL`)
- **Kokoro Settings**: Voice selection (`KOKORO_VOICE`), speed, and model paths
- **LLM Configuration**: Ollama model selection (`OLLAMA_LLM`)
- **System Prompt**: Personality configuration (currently George Carlin + Rodney Carrington blend)
- **VAD Settings**: Intelligent voice activity detection with neural networks (Silero VAD)
- **macOS Features**: Voice isolation settings, automatic gain control
- **ESP32 Integration**: LED matrix display and wireless device configuration

### Main Entry Point (`run_voice_assistant.py`)

The primary orchestration script that manages the conversation loop:

- **Global State Management**: Uses threading events to coordinate wake word detection, conversation state, and playback
- **LED State Updates**: `update_led_state()` function manages both LED matrix and console output
- **Conversation Flow Control**:
  - Wake word detection triggers conversation mode
  - `check_end_conversation()` determines when to return to wake word listening
  - Handles both explicit exit phrases and implicit conversation continuation
- **Audio Manager Integration**: Runtime audio source switching between local and wireless microphones
- **Hotkey Support**: Ctrl+Alt+L/W/T/I/D for runtime audio source control
- **Greeting System**: Dynamic wake word responses via `greeting_generator.py`
- **Cleanup Mechanisms**: Targeted garbage collection (`targeted_gc()`) for audio resources

### Audio Recording Architecture

The system uses a layered approach for audio capture:

1. **AudioSourceManager** (`voice_assistant/audio_source_manager.py`):
   - Central coordinator for all audio sources
   - Transparent switching between local microphone and wireless devices
   - Runtime hotkey support for source changes
   - Global singleton accessible via `get_audio_manager()`

2. **EnhancedAudioRecorder** (`voice_assistant/enhanced_audio.py`):
   - Implements intelligent VAD-based recording
   - Pre-speech buffering (500ms) to capture utterance beginnings
   - Chunk accumulation to handle partial PyAudio reads
   - Fixed 512-sample chunks for Silero VAD (32ms at 16kHz)
   - Wake word filtering to remove activation sound

3. **NetworkAudioSource** (`voice_assistant/network_audio_source.py`):
   - UDP-based wireless microphone support (ESP32P4 devices)
   - OPUS compression for low-latency streaming
   - Device discovery on port 8001
   - Audio streaming on port 8000

### Conversation State Machine

```
[Wake Word Detection] --"Hey Howdy"--> [Conversation Active]
         ↑                                      |
         |                                      |
         |                                      ↓
[Waiting/Idle] <--exit phrase-- [Listening → Thinking → Speaking]
         ↑                                      |
         |                                      |
         +------ continues looping -------------+
```

**Key State Variables** (threading.Event objects in `run_voice_assistant.py`):
- `wake_word_detected`: Signals wake word was heard
- `conversation_active`: Maintains conversation context
- `activation_sound_playing`: Prevents recording during greeting
- `is_first_turn_after_wake`: Triggers wake word filtering
- `playback_complete_event`: Coordinates audio playback completion

### Audio Pipeline

```
Microphone Input → Porcupine Wake Word → VAD/Utterance Detection →
FastWhisperAPI Transcription → Ollama LLM → Kokoro TTS → Audio Output
```

**Key features:**
- **Model Preloading**: Both Kokoro TTS and Ollama LLM models load at startup for faster first response
- **Adaptive TTS Chunking**: Automatically adjusts chunk sizes based on response length
  - Short texts (<100 chars): 150 char chunks with 50ms delay
  - Medium texts (100-500 chars): 180 char chunks with 100ms delay
  - Long texts (>500 chars): 220 char chunks with 150ms delay
- **Memory Management**: Targeted garbage collection for audio resources
- **Enhanced Buffer Management**: Pre-buffering and queue monitoring to prevent stuttering

### Voice Activity Detection (VAD)

The system uses intelligent neural VAD with adaptive utterance detection:

- **Silero VAD**: Neural network-based speech detection
- **Intelligent Parameters**:
  - `MIN_UTTERANCE_DURATION`: 0.5s minimum speech
  - `MAX_INITIAL_SILENCE`: 10s before speech starts
  - `MIN_FINAL_SILENCE`: 0.8s to end utterance
  - `MAX_FINAL_SILENCE`: 2s force ending
- **Adaptive Pausing**: Different pause factors for questions, incomplete sentences, filler words
- **macOS Voice Isolation**: Native macOS audio processing for enhanced noise reduction

### ESP32 Integration

**LED Matrix Display (ESP32-S3):**
- Visual feedback for assistant state (waiting, listening, thinking, speaking, ending)
- Configured via `ESP32_IP` environment variable
- See [ESP32 directory](ESP32/) for firmware

**Wireless Microphones (ESP32P4):**
- Real-time audio streaming via UDP with OPUS compression
- Multi-room support with device discovery and room assignment
- LED ring audio visualization
- Hotkey switching during runtime (Ctrl+Alt+L/W/T/I/D)
- See [ESP32P4_INTEGRATION.md](ESP32P4_INTEGRATION.md) for details

### Network Audio Source

The `NetworkAudioSource` class provides wireless microphone support:
- UDP audio streaming on port 8000
- Device discovery on port 8001
- OPUS audio codec for minimal latency (2-5ms)
- Automatic device management and reconnection
- Compatible with existing VAD and audio processing pipeline

### Text-to-Speech System

The TTS system uses a streaming architecture for low-latency responses:

**Key Components**:
- **KokoroManager** (`voice_assistant/kokoro_manager.py`): Singleton that manages ONNX TTS model
- **text_to_speech()** (`voice_assistant/text_to_speech.py`): Generates audio chunks in background thread
- **Chunk Queue**: Thread-safe queue that buffers generated audio chunks

**Adaptive Chunking Strategy**:
- Short responses (<100 chars): 150 char chunks, 50ms delay, 0.1s stabilization
- Medium responses (100-500 chars): 180 char chunks, 100ms delay, 0.2s stabilization
- Long responses (>500 chars): 220 char chunks, 150ms delay, 0.3s stabilization

**Playback Coordination** (in `run_voice_assistant.py:play_all_chunks()`):
1. Initial stabilization delay based on response length
2. Play first chunk immediately after delay
3. Monitor queue for subsequent chunks
4. Track inter-chunk gaps and log warnings for delays >1.5s
5. Continue until `generation_complete.is_set()` and queue is empty

See [TTS_STUTTERING_FIX_README.md](TTS_STUTTERING_FIX_README.md) for implementation details.

### Voice Blending System

Create custom voices by blending multiple Kokoro voice styles:
- Combine voice style vectors with specific ratios
- Available voices: `af_*` (female US), `am_*` (male US), `bf_*`, `bm_*`
- Configuration via `configure_blended_voice.py`
- See [VoiceBlend.md](VoiceBlend.md) for detailed instructions

## Directory Structure

```
HowdyVox/
├── voice_assistant/                # Core voice assistant logic
│   ├── config.py                  # Centralized configuration (EDIT THIS FOR SETTINGS)
│   ├── audio.py                   # Legacy audio recording (mostly replaced)
│   ├── enhanced_audio.py          # VAD-based recording (PRIMARY)
│   ├── text_to_speech.py          # TTS with adaptive chunking + queue
│   ├── intelligent_vad.py         # Silero VAD wrapper
│   ├── utterance_detector.py      # Utterance boundary detection
│   ├── mac_voice_isolation.py     # macOS native audio processing
│   ├── network_audio_source.py    # Wireless microphone (ESP32P4)
│   ├── audio_source_manager.py    # Audio source coordinator
│   ├── hotkey_manager.py          # Runtime hotkey controls
│   ├── led_matrix_controller.py   # ESP32-S3 LED matrix
│   ├── kokoro_manager.py          # Kokoro TTS singleton
│   ├── greeting_generator.py      # Dynamic wake word responses
│   ├── wake_word.py               # Porcupine wake word detection
│   ├── transcription.py           # FastWhisperAPI client
│   ├── response_generation.py     # Ollama LLM client
│   └── utils.py                   # Cleanup & GC utilities
├── FastWhisperAPI/                # Local speech recognition service
│   ├── main.py                    # FastAPI server
│   └── requirements.txt           # FastAPI dependencies
├── Tests_Fixes/                   # Comprehensive testing suite
│   ├── check_components.py        # System-wide diagnostics
│   ├── fix_all_issues.py          # Automated fixes
│   └── test_*.py                  # Individual component tests
├── ESP32/                         # ESP32-S3 LED matrix firmware
├── models/                        # Local model files (Kokoro voices)
├── run_voice_assistant.py         # Main entry point (ORCHESTRATION LOGIC)
├── launch_howdy_shell.py          # Shell launcher (single command)
├── launch_howdy_ui.py             # Interactive UI launcher with animated face
├── ui_interface.py                # Tkinter UI implementation
├── face_animator.py               # Animated 8-bit face widget
├── test_face_ui.py                # Face animation test utility
├── blend_voices.py                # Voice blending utility
└── requirements.txt               # Python dependencies
```

## Environment Setup

### Required Environment Variables (`.env` file)

```env
PORCUPINE_ACCESS_KEY="your-access-key-here"  # Free from picovoice.ai
LOCAL_MODEL_PATH="models"                     # Path to Kokoro models
ESP32_IP="192.168.1.xxx"                     # Optional: LED matrix IP
```

### Prerequisites

- **Python**: 3.10 or higher
- **PyAudio**: Specifically version 0.2.12 for macOS compatibility (0.2.14 in requirements.txt for newer systems)
- **Ollama**: Local LLM runtime (download from ollama.com)
- **Porcupine**: Wake word detection (free access key from picovoice.ai)
- **CUDA GPU**: Optional, improves performance
- **macOS**: Enhanced ONNX Runtime for Apple Silicon optimization

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install specific PyAudio version (macOS)
pip uninstall -y pyaudio
pip install pyaudio==0.2.12

# Set up Porcupine wake word
python quick_setup.py

# Install and start Ollama
ollama run hf.co/unsloth/gemma-3-4b-it-GGUF:latest

# Apple Silicon optimization (optional)
python Tests_Fixes/fix_onnx_runtime.py
```

## Key Technical Details

### Wake Word Detection

Uses Porcupine for "Hey Howdy" detection. The system waits for the wake word, then enters conversation mode until the user says goodbye/that's all/etc.

**Implementation**: `voice_assistant/wake_word.py`
- `WakeWordDetector`: Primary implementation using Porcupine
- `SpeechRecognitionWakeWord`: Fallback using Google Speech Recognition
- `cleanup_all_detectors()`: Global cleanup function for safe shutdown

### Conversation Flow

1. **Initialize**: Preload Kokoro TTS and Ollama LLM models (eliminates cold start)
2. **Wait**: Listen for "Hey Howdy" wake word via Porcupine
3. **Wake Greeting**: Generate dynamic greeting via LLM (`greeting_generator.py`)
4. **Listen**: Record audio with intelligent VAD (Silero neural network)
5. **Transcribe**: Send audio to FastWhisperAPI (local)
6. **Generate**: Query Ollama for response using system prompt personality
7. **Speak**: Generate audio via Kokoro TTS with adaptive chunking
8. **Loop**: Continue conversation until farewell phrase detected
9. **End**: Return to wake word listening, perform garbage collection

### Critical Threading Patterns

**Audio Playback Threading** (`run_voice_assistant.py:712-803`):
- Playback runs in separate daemon thread to avoid blocking conversation loop
- `playback_complete_event` coordinates between main loop and playback thread
- Main loop waits for playback completion before recording next utterance
- Cleanup of chunk files happens in `finally` block of playback thread

**Wake Word Detection Threading**:
- Wake word detector runs continuously in background thread
- Callback `handle_wake_word()` sets threading events to trigger conversation mode
- Detector is stopped and restarted between conversations to prevent memory leaks

**Greeting Generation Threading** (`run_voice_assistant.py:173-176`):
- Greeting plays in separate thread to avoid blocking wake word response
- `activation_sound_playing` event prevents recording during greeting
- Main loop waits up to 5 seconds for greeting completion before recording

### macOS-Specific Features

- **Voice Isolation**: Native macOS voice processing (requires macOS 12.0+)
- **Apple Silicon Support**: Optimized ONNX Runtime for M-series chips
- **PyAudio Compatibility**: Version pinning for macOS audio stack

### Performance Optimizations

- **Model Preloading**: Eliminates first-response latency
- **Audio Buffer Pooling**: Reduces memory allocations
- **Targeted Garbage Collection**: Audio-specific resource cleanup
- **Intelligent VAD**: Reduces unnecessary transcription calls
- **OPUS Compression**: Minimal latency for wireless audio (2-5ms)

### Testing Philosophy

The `Tests_Fixes/` directory contains 15+ testing scripts for every component. Use `fix_all_issues.py` for automated diagnostics and repairs. Each component has dedicated test scripts for isolated verification.

## Common Issues & Solutions

### Wake Word Not Working

Run `python quick_setup.py` to configure Porcupine access key.

### PyAudio Errors (macOS)

Ensure PyAudio 0.2.12 is installed: `pip install pyaudio==0.2.12`

### FastWhisperAPI Connection Failed

Verify FastAPI is running at <http://localhost:8000>. Check with `python Tests_Fixes/check_fastwhisper.py`

### No Ollama Response

Check Ollama is running: `ollama list`. Pull model: `ollama run hf.co/unsloth/gemma-3-4b-it-GGUF:latest`

### TTS Stuttering

Adaptive chunk sizing and delays are already implemented. If issues persist, run `python Tests_Fixes/test_tts_fix.py`

### Memory Issues

Test garbage collection: `python Tests_Fixes/test_targeted_gc.py`

### ONNX Runtime (Apple Silicon)

Run `python Tests_Fixes/fix_onnx_runtime.py` for optimization

### Wireless Devices Not Found

- Ensure ESP32P4 is on same network
- Verify UDP ports 8000/8001 are open in firewall
- Check ESP32P4 serial monitor for connection errors
- Wait 10-15 seconds for device discovery

### UI Output Duplication or Missing Messages

Recent fixes (commits be95330-812a481) address:

- Message duplication during initialization
- Multi-paragraph response handling
- Input/output truncation in UI capture
- Status indicator accuracy

If issues persist, check `update_led_state()` in `run_voice_assistant.py:53-118` for state tracking logic.

## Additional Documentation

- [README.md](README.md) - Comprehensive setup guide with features
- [QUICK_START.md](QUICK_START.md) - Single-command launch guide with audio sources
- [ANIMATED_FACE_README.md](ANIMATED_FACE_README.md) - Animated face feature guide
- [TTS_STUTTERING_FIX_README.md](TTS_STUTTERING_FIX_README.md) - Stuttering fix implementation
- [TTS_ENHANCEMENT_IMPLEMENTATION.md](TTS_ENHANCEMENT_IMPLEMENTATION.md) - Performance enhancements
- [VoiceBlend.md](VoiceBlend.md) - Voice blending guide
- [ESP32P4_INTEGRATION.md](ESP32P4_INTEGRATION.md) - Wireless microphone integration
- [Tests_Fixes/test_and_run_instructions.md](Tests_Fixes/test_and_run_instructions.md) - Testing suite guide
- [Upgrades/intelligent-vad-implementation-guide.md](Upgrades/intelligent-vad-implementation-guide.md) - VAD details
- [Upgrades/macos-voice-isolation-integration.md](Upgrades/macos-voice-isolation-integration.md) - macOS audio

## Runtime Hotkeys (with Wireless Support)

When running with wireless support, these hotkeys are available:

- **Ctrl+Alt+L**: Switch to local microphone
- **Ctrl+Alt+W**: Switch to wireless microphone
- **Ctrl+Alt+T**: Toggle between sources
- **Ctrl+Alt+I**: Show current audio source info
- **Ctrl+Alt+D**: List available wireless devices

Note: Requires `keyboard` module and may need accessibility permissions on macOS.

## Development & Debugging Guidelines

### When Modifying Audio Recording

Audio recording is critical and fragile. Follow these guidelines:

1. **Primary audio implementation**: `voice_assistant/enhanced_audio.py`
   - Uses Silero VAD for speech detection
   - Requires exactly 512 samples per chunk (32ms at 16kHz)
   - Pre-speech buffer captures first 500ms before speech detection
   - Chunk accumulator handles partial PyAudio reads

2. **Audio source abstraction**: `voice_assistant/audio_source_manager.py`
   - Global singleton pattern - use `get_audio_manager()` to access
   - Wraps both local and wireless audio sources
   - Don't bypass this abstraction layer

3. **Testing audio changes**:

   ```bash
   python microphone_test.py              # Basic audio I/O
   python Tests_Fixes/test_intelligent_vad.py  # VAD behavior
   python Tests_Fixes/debug_vad_audio.py  # Audio level monitoring
   ```

### When Modifying Conversation Flow

The conversation state machine is coordinated via threading events in `run_voice_assistant.py`:

1. **Always consider threading implications** - multiple threads access shared state
2. **Event coordination order matters**:
   - Set `conversation_active` before recording
   - Clear `activation_sound_playing` after greeting completes
   - Wait for `playback_complete_event` before next recording
3. **Test conversation edge cases**:
   - Wake word → immediate silence
   - Wake word → long pause → speech
   - Multiple rapid questions
   - Conversation end phrases

### When Modifying TTS

The TTS system uses background generation with chunked playback:

1. **Thread-safe queue**: `text_to_speech.py` generates chunks in background thread
2. **Playback coordination**: Main thread consumes chunks via `get_next_chunk()`
3. **Completion signaling**: `generation_complete` event signals end of generation
4. **File cleanup**: All chunk files must be cleaned up in playback thread's `finally` block

**Testing TTS changes**:

```bash
python Tests_Fixes/test_tts_fix.py           # Basic TTS functionality
python Tests_Fixes/test_tts_fix_enhanced.py  # Chunking behavior
```

### Debugging Tools

**Logging levels**: Set `logging.basicConfig(level=logging.DEBUG)` for verbose output

**Audio visualization**:

```bash
python Tests_Fixes/debug_vad_audio.py  # Real-time audio levels
python Tests_Fixes/diagnose_vad.py     # VAD behavior analysis
```

**Component isolation**:

```bash
python Tests_Fixes/check_components.py  # Test all components
python Tests_Fixes/test_kokoro_onnx.py  # Test TTS only
python Tests_Fixes/check_fastwhisper.py # Test STT only
```

### Memory Management

The system uses targeted garbage collection for audio resources:

- `targeted_gc()` in `voice_assistant/utils.py` specifically collects audio-related objects
- Called after conversation ends and on errors
- Prevents accumulation of PyAudio buffers and ONNX tensors

**When adding new audio processing**:

1. Ensure all audio buffers are released in `finally` blocks
2. Close PyAudio streams explicitly
3. Test with `python Tests_Fixes/test_targeted_gc.py`

### Common Development Pitfalls

1. **PyAudio partial reads**: Always accumulate chunks until you have exactly 512 samples for Silero VAD
2. **Threading deadlocks**: Never call blocking audio operations while holding threading locks
3. **Event race conditions**: Always set events before waiting on them in other threads
4. **File descriptor leaks**: Close all audio streams in `finally` blocks
5. **Wake word detector leaks**: Always call `cleanup_all_detectors()` on shutdown
