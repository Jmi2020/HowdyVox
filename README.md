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

## Prerequisites ✅

- Python 3.10 or higher
- Virtual environment (recommended)
- PyAudio 0.2.12 (specifically this version for macOS compatibility)
- [Ollama](https://ollama.com/) installed and running locally
- [Porcupine](https://picovoice.ai/platform/porcupine/) access key (free for personal use)
- CUDA-capable GPU (optional, improves performance)

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

## Running the Assistant 🤠

Start the voice assistant:
```bash
python run_voice_assistant.py
```

The assistant will:
1. Wait for the wake word "Hey Howdy"
2. Listen for your voice input
3. Transcribe it using FastWhisperAPI
4. Generate a response using Ollama with cowboy persona
5. Speak the response using Kokoro's cowboy voice

The conversation will continue until you say phrases like "goodbye", "that's all", or "thanks, that's all".

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

## Testing the Setup 🧪

1. Test your microphone:
```bash
python microphone_test.py
```

2. Test audio recording and transcription:
```bash
python test_audio.py
```

3. Test Kokoro TTS:
```bash
python test_kokoro_onnx.py
```

## Troubleshooting 🔧

- **Wake word detection not working**: Run `python quick_setup.py` to configure Porcupine
- **PyAudio errors on macOS**: Make sure you're using version 0.2.12
- **FastWhisperAPI errors**: Ensure it's running at http://localhost:8000
- **No response from Ollama**: Check that Ollama is running with `ollama list` and your model is downloaded
- **TTS errors**: Verify Kokoro model files are in the `models/` directory
- **No audio output**: Check your system sound settings and run `python microphone_test.py`

## License 📄

MIT License - See LICENSE file for details

---

*"Remember, partner - I'm always here to help! Just holler 'Hey Howdy' when ya need me!"* 🤠


