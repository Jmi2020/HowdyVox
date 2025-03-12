# HowdyTTS - A Voice Assistant with Cowboy Charm 🤠

A fully offline voice assistant that uses Kokoro TTS with a cowboy voice, FastWhisperAPI for transcription, and Ollama for text generation. This project focuses on local processing and maintaining a consistent cowboy persona.

## Features 🌟

- **100% Offline Operation**: No cloud services required
- **Cowboy Voice**: Uses Kokoro TTS with the 'am_michael' voice model
- **Fast Local Speech Recognition**: Powered by FastWhisperAPI
- **Local LLM Support**: Integrated with Ollama for text generation
- **Easy Testing Tools**: Includes microphone test and audio test utilities

## Prerequisites ✅

- Python 3.10 or higher
- Virtual environment (recommended)
- PyAudio 0.2.12 (specifically this version for macOS compatibility)
- [Ollama](https://ollama.com/) installed and running locally
- CUDA-capable GPU (optional, improves performance)

## Setup Instructions 🔧

1. Clone the repository:
```bash
git clone [your-repo-url]
cd HowdyTTS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyAudio 0.2.12 specifically:
```bash
pip uninstall -y pyaudio
pip install pyaudio==0.2.12
```

5. Start the FastWhisperAPI service:
```bash
cd FastWhisperAPI
pip install -r requirements.txt
fastapi run main.py
```

6. Make sure Ollama is running and has the required model:
```bash
ollama run llama3:8b
```

## Configuration ⚙️

The system uses a centralized configuration in `voice_assistant/config.py`. The default settings are:

```python
TRANSCRIPTION_MODEL = 'fastwhisperapi'  # Local speech recognition
RESPONSE_MODEL = 'ollama'               # Local LLM using Ollama
TTS_MODEL = 'kokoro'                    # Local TTS with cowboy voice
KOKORO_VOICE = 'am_michael'             # Default cowboy voice
OLLAMA_LLM = "llama3:8b"               # Default Ollama model
```

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

## Running the Assistant 🤠

Start the voice assistant:
```bash
python run_voice_assistant.py
```

The assistant will:
1. Listen for your voice input
2. Transcribe it using FastWhisperAPI
3. Generate a response using Ollama
4. Speak the response using Kokoro's cowboy voice

Say "goodbye" or "arrivederci" to exit.

## Project Structure 📂

- `voice_assistant/` - Core implementation
  - `audio.py` - Audio recording and playback
  - `transcription.py` - FastWhisperAPI integration
  - `response_generation.py` - Ollama integration
  - `text_to_speech.py` - Kokoro TTS implementation
  - `config.py` - Central configuration
- `FastWhisperAPI/` - Local transcription service
- `test_*.py` - Various test utilities
- `requirements.txt` - Project dependencies

## Troubleshooting 🔧

- If PyAudio fails on macOS, ensure you're using version 0.2.12
- If FastWhisperAPI fails, check it's running at http://localhost:8000
- If TTS fails, verify Kokoro model files are present
- If responses are slow, ensure Ollama is running with the correct model

## Contributing 🤝

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License 📄

MIT License - See LICENSE file for details


