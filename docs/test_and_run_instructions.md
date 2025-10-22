# Kokoro ONNX Setup and Testing Instructions

This document provides instructions for setting up and testing the Kokoro ONNX TTS integration in HowdyVox.

## Setup Instructions

### 1. Download the Kokoro ONNX Model

Download the model files with the script provided:

```bash
# Download the quantized model (recommended, smaller size)
python download_kokoro_onnx.py --type q8

# For better quality but larger model:
python download_kokoro_onnx.py --type fp32

# To install directly to project directory instead of ~/.kokoro_onnx:
python download_kokoro_onnx.py --type q8 --local
```

This will download the model files to `~/.kokoro_onnx/` (or to the local project directory if you use the `--local` flag).

### 2. Install Required Dependencies

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

The script requires:
- onnxruntime
- numpy
- scipy
- soundfile
- other dependencies listed in requirements.txt

### 3. Configure HowdyVox to Use Kokoro ONNX

Edit `voice_assistant/config.py` to use the Kokoro ONNX model:

```python
# Change this line in voice_assistant/config.py
TTS_MODEL = 'kokoro_onnx'  # Change from 'kokoro' to 'kokoro_onnx'

# Optionally specify a specific voice
KOKORO_VOICE = 'am_michael'  # Default cowboy voice

# Optionally specify a custom model path (if you used --local)
# KOKORO_ONNX_MODEL_PATH = '/path/to/model/directory'
```

## Testing

### 1. Test the Kokoro ONNX Model

Run the test script to verify that the model works correctly:

```bash
python test_kokoro_onnx.py
```

This will:
- Load the Kokoro ONNX model
- Display available voices
- Generate a test audio file with the default voice
- Save it to `test_kokoro_onnx.wav`

### 2. Testing with Custom Text

You can specify custom text for testing:

```bash
python test_kokoro_onnx.py --text "Howdy partner! This is a test of the Kokoro ONNX TTS system."
```

### 3. Test with a Specific Voice

If you want to test with a specific voice:

```bash
python test_kokoro_onnx.py --voice en
```

Available voices will depend on which files were successfully downloaded, but typically include:
- `am_michael` (American cowboy)
- `en` (English)
- `af` (Afrikaans)
- `zh` (Chinese)
- `fr` (French)
- `de` (German)
- and others

## Running the Voice Assistant

Run the voice assistant with Kokoro ONNX:

```bash
python run_voice_assistant.py
```

## Troubleshooting

### Missing Voices

If you see errors about missing voices:

1. The system will automatically try to use any available voice
2. Check which voices were downloaded:
   ```bash
   ls -la ~/.kokoro_onnx/voices/
   ```
3. Try running the download script again with different options
4. Manually create a symlink if needed:
   ```bash
   ln -s ~/.kokoro_onnx/voices/af.bin ~/.kokoro_onnx/voices/am_michael.bin
   ```

### Model Loading Errors

If the model fails to load:

1. Check that the model files exist:
   ```bash
   ls -la ~/.kokoro_onnx/onnx/
   ```
2. Try downloading with a different model type (--type q8 or --type fp16)
3. Check the ONNX runtime version:
   ```bash
   pip show onnxruntime
   ```
   Make sure it's at least version 1.14.0

### Audio Generation Issues

If audio generation fails:

1. Make sure Python's soundfile and numpy packages are properly installed
2. Try generating audio with a different voice
3. Check the error messages for specific details about what's failing

## Advanced Configuration

You can customize the Kokoro ONNX integration by modifying:

1. `voice_assistant/kokoro_onnx/tts.py` - Core TTS implementation
2. `voice_assistant/kokoro_onnx/integration.py` - Integration with the voice assistant
3. `voice_assistant/text_to_speech.py` - TTS selection logic