# Voice Blending in HowdyTTS

This guide explains how to use the voice blending capability in HowdyTTS, which allows you to create custom voices by combining multiple base voices at different ratios.

## Introduction

Voice blending combines multiple voice styles from Kokoro ONNX to create unique voice personalities. Instead of using a single predefined voice, you can create a blend that has qualities from multiple voices - perhaps combining the warmth of one voice with the clarity of another.

## Available Voices

HowdyTTS supports all the voices included in the Kokoro ONNX model. The voice IDs follow a specific naming pattern:

- `af_*` - Female English US voices (e.g., `af_nicole`, `af_bella`)
- `am_*` - Male English US voices (e.g., `am_michael`, `am_eric`) 
- `bf_*` - Other female voices (e.g., `bf_alice`)
- `bm_*` - Other male voices (e.g., `bm_daniel`)

You can view the complete list of available voices with:

```bash
python blend_voices.py --list-voices
```

## Voice Blending Basics

### How Voice Blending Works

Voice blending works by combining the voice style vectors of different voices with specific ratios. For example, you might create a voice that is 70% Michael and 30% Nicole:

```python
blend = np.add(nicole * (30 / 100), michael * (70 / 100))
```

The resulting blend has characteristics of both voices, creating a unique sound.

## Quick Start

### 1. Basic Blending

To create a simple blend of two voices and generate audio:

```bash
python blend_voices.py --voices "af_nicole:30,am_michael:70" --text "Hello, this is a blended voice test."
```

This command creates audio with a voice that is 30% Nicole and 70% Michael.

### 2. Testing Voice Combinations

To explore different blend ratios between two voices:

```bash
python Tests_Fixes/test_voice_blending.py --voice1 af_bella --voice2 am_eric --steps 5
```

This will generate audio samples with different blending ratios between Bella and Eric's voices (0%, 20%, 40%, 60%, 80%, 100% of each voice), allowing you to compare the results.

### 3. Multi-Voice Blending

You can blend more than two voices for more complex vocal characteristics:

```bash
python blend_voices.py --voices "af_bella:30,am_michael:50,am_eric:20" --text "This blends three different voices."
```

This creates a blend with 30% Bella, 50% Michael, and 20% Eric.

## Integrating Blended Voices Into Your Voice Assistant

After testing different blends, you can permanently configure your voice assistant to use your preferred blend.

### 1. Create a Voice Profile

First, create a named voice profile:

```bash
python configure_blended_voice.py --name "my_custom_voice" --voices "af_bella:40,am_michael:60"
```

This command:
1. Creates a blend with 40% Bella and 60% Michael
2. Registers it as a named voice profile called "my_custom_voice"
3. Updates your `.env` file with the blend configuration
4. Verifies that the blend works correctly

### 2. Update Your Configuration

Edit `voice_assistant/config.py` to use your new voice profile:

```python
# In voice_assistant/config.py
KOKORO_VOICE = 'my_custom_voice'  # Use your blend name here
```

### 3. Run Your Voice Assistant

Now run your voice assistant, and it will automatically use the blended voice:

```bash
python run_voice_assistant.py
```

## Advanced Blending

### Experimenting with Multi-Voice Combinations

For more sophisticated voice blends, you can try the multi-voice test script:

```bash
python Tests_Fixes/test_voice_blending.py --multi-voice
```

This generates several interesting combinations:
- `multi_female`: A blend of three female voices
- `multi_male`: A blend of three male voices
- `balanced_quartet`: An equal mix of two male and two female voices
- `mostly_michael`: Michael's voice with subtle influences from others
- `diverse_blend`: A varied mix including voices from different categories

### Custom Speech Speed

Adjust the speech speed with the `--speed` parameter:

```bash
python blend_voices.py --voices "af_bella:40,am_michael:60" --speed 1.2 --text "This is slightly faster speech."
```

Values above 1.0 make speech faster, while values below 1.0 make it slower.

## How It Works Behind The Scenes

### Voice Initialization Process

When you configure a blended voice and start the voice assistant:

1. The `voice_initializer.py` module runs during startup
2. It reads the voice name from `config.py` and looks for corresponding voice ratio settings in the `.env` file
3. It creates the blended voice vector and registers it with Kokoro
4. Each voice component is weighted by its percentage and added together 
5. The TTS system can then use this blend by name just like a regular voice

### Technical Implementation

The voice blending is implemented in these files:
- `blend_voices.py`: Standalone script for testing voice blends
- `test_voice_blending.py`: Tool to generate and compare multiple blends
- `configure_blended_voice.py`: Configuration tool to set up permanent voice profiles
- `voice_assistant/voice_initializer.py`: System module that initializes blended voices during startup

## Troubleshooting

### Voice Profile Not Recognized

If your voice assistant isn't using the blended voice:

1. Verify you used the exact same name in `config.py` as when creating the blend
2. Check your `.env` file to ensure it contains the voice ratio settings
3. Run `python configure_blended_voice.py` again to recreate the profile
4. Make sure to restart the voice assistant completely

### Audio Quality Issues

If a blend sounds distorted:

1. Try adjusting the blend ratios - some combinations work better than others
2. Avoid using too many voices in one blend (2-3 is usually optimal)
3. Make sure the ratio percentages add up to 100%

## Tips for Great Voice Blends

- **Start with 70/30 or 60/40 ratios** between two voices for best results
- **Similar voice types** often blend better (female+female or male+male)
- **Test with short phrases first** before committing to a blend for your assistant
- **The dominant voice** (higher percentage) sets the overall character
- **Keep a record** of blends you like for future reference

## Example Blend Recipes

Here are some pre-tested blend combinations that work well:

1. **Gentle Cowboy**: 70% am_michael + 30% af_nicole
   ```bash
   python configure_blended_voice.py --name "gentle_cowboy" --voices "am_michael:70,af_nicole:30"
   ```

2. **Professional Announcer**: 60% am_adam + 40% am_eric
   ```bash
   python configure_blended_voice.py --name "professional" --voices "am_adam:60,am_eric:40"
   ```

3. **Warm Female Voice**: 70% af_bella + 30% af_jessica
   ```bash
   python configure_blended_voice.py --name "warm_female" --voices "af_bella:70,af_jessica:30"
   ```

4. **Balanced Quartet**: 25% each of am_michael, am_adam, af_nicole, and af_bella
   ```bash
   python configure_blended_voice.py --name "quartet" --voices "am_michael:25,am_adam:25,af_nicole:25,af_bella:25"
   ```

Have fun experimenting with different voice combinations to find the perfect voice for your assistant!