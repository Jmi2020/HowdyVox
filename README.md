# HowdyVox - Your Snarky, Offline AI Companion 🤠

Howdy, partner! Welcome to HowdyVox - a fully local, privacy-first conversational AI that's more private than your therapist and cheaper than your bar tab. This ain't your typical cloud-dependent assistant that sends your embarrassing questions to some data center in Iowa. Everything runs on your machine, stays on your machine, and dies on your machine. Just the way it should be.

**The Catch:** You'll need a free Picovoice Porcupine license key to get the wake word detection working. Don't worry, it's actually free for personal use (unlike most things marketed as "free"). Without it, Howdy's just a folder of Python scripts with delusions of grandeur.

## What Makes This Thing Special? 🌟

### Privacy That Actually Means Something

Your conversations never leave your computer. No cloud services. No "telemetry." No "analytics to improve user experience." Just you, your voice, and an AI that couldn't snitch even if it wanted to (which it doesn't, because it's offline and has no concept of federal witness protection programs).

### Bring Your Own Everything

- **Any LLM You Want**: Works with any Ollama-compatible model. Gemma, Llama, Mistral, that weird experimental one you found on Hugging Face at 3 AM - they all work
- **Personality Editor**: Change the `SYSTEM_PROMPT` to make Howdy talk like Socrates, your grumpy uncle, or a motivational speaker having a bad day
- **Voice Buffet**: 20+ built-in voices or blend them together like you're running a vocal smoothie shop
- **Swap-Friendly**: Change the LLM mid-stream without restarting. It's like model hot-swapping but less dangerous

### Actually Fast (No, Really)

- **Model Preloading**: Both TTS and LLM load at startup, so your first response is just as zippy as your tenth
- **Adaptive Chunking**: Automatically figures out the best way to deliver audio without sounding like a skipping CD
- **Smart Buffering**: Pre-loads chunks while playing earlier ones, like a very organized relay race
- **Memory Management**: Targeted garbage collection means you can have marathon 3 AM conversations without the RAM usage looking like a crypto mining operation

### Natural Conversations (For Silicon Values of "Natural")

- **Wake Word**: Just say "Hey Howdy" and you're off to the races
- **Context Awareness**: Remembers what you talked about until you explicitly end the session (no goldfish memory here)
- **Intelligent VAD**: Neural network-based voice detection that actually knows when you've stopped rambling
- **Multi-Room**: Set up USB mics in different rooms because apparently one room isn't enough for your conversations with an AI

### Developer-Friendly (Translation: We Included Documentation)

- **15+ Test Scripts**: Verify every component works before blaming cosmic rays
- **Automated Fixes**: Run `fix_all_issues.py` and let the robots fix the robots
- **Modular Design**: STT, LLM, and TTS components are independently swappable, like LEGO but with more dependencies
- **Extensive Docs**: We wrote guides for everything. You're reading one right now. Meta!

## The Face That Launched a Thousand Commits 🎨

HowdyVox now sports an **audio-reactive face** that actually responds to speech characteristics in real-time. Think of it as giving your AI a face that does more than just sit there looking pretty (though it does that too).

### Choose Your Fighter: Two Face Styles

#### Option 1: GIF-Based Face (The Efficient One)
Load your own GIF animations and watch them react to audio features:
- **Your Art, Your Rules**: Drop in your own GIF files and they become the face
- **Audio-Reactive Speed**: Playback speed changes based on volume, sibilance, and emphasis
- **Low CPU Overhead**: ~2-5% CPU because not everyone has a NASA workstation
- **Simple Customization**: Just replace the GIF files. That's it. Done.

#### Option 2: EchoEar Face (The Fancy One)
Real-time rendered face with more expressiveness than a mime at an improv show:
- **Dynamic Rendering**: Eyes pulse, narrow, and the head nods based on actual speech analysis
- **Audio Feature Mapping**:
  - Volume (RMS) → Eye size (bigger eyes = louder speech)
  - Sibilance (ZCR) → Horizontal squeeze (narrow eyes for "s" and "sh" sounds)
  - Emphasis (Peaks) → Brief head nod (because even AIs should nod along)
- **Visual Polish**: Glowing cyan eyes with multi-layer effects and alpha blending
- **Moderate CPU**: ~5-12% CPU for significantly more expressiveness

**Both faces feature:**
- Custom rounded icon (that glowing face you see in the dock)
- Process name shows as "HowdyVox" instead of "python3.10" (fancy!)
- Can run on a separate device via UDP (Raspberry Pi face display, anyone?)

### How the Audio Magic Works

The system analyzes Howdy's speech in real-time using three deceptively simple features:

**1. RMS (Root Mean Square) - The Volume Knob**
```python
# Measures "loudness" of speech
rms = audioop.rms(pcm_chunk, sample_width)
# Translation: Louder speech = bigger eyes (or faster GIF)
# Because apparently volume should affect facial expressions
```

**2. ZCR (Zero-Crossing Rate) - The Sibilance Detector**
```python
# Counts how often the audio waveform crosses zero
# High ZCR = sibilants (s, sh, ch, f) = narrow eyes
# Low ZCR = vowels (a, e, i, o, u) = normal eyes
# It's like the AI is squinting at bright sounds
```

**3. Peak Detection - The Emphasis Spotter**
```python
# Detects sudden energy increases
if current_rms > threshold and no_recent_peak:
    trigger_head_nod()  # Brief 2-frame animation
# Even AIs should have a little body language
```

## How It Works (The Technical Deep Dive) 🔧

HowdyVox orchestrates multiple components like a conductor who's had too much coffee. Here's the pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. WAKE WORD (Porcupine)                                       │
│    ↓ Listens for "Hey Howdy" without recording everything     │
├─────────────────────────────────────────────────────────────────┤
│ 2. VOICE ACTIVITY DETECTION (Silero Neural VAD)                │
│    ↓ Knows when you've stopped talking (unlike some people)   │
├─────────────────────────────────────────────────────────────────┤
│ 3. SPEECH-TO-TEXT (FastWhisperAPI - Local)                     │
│    ↓ Transcribes your wisdom (or whatever)                    │
├─────────────────────────────────────────────────────────────────┤
│ 4. LANGUAGE MODEL (Ollama - Your Choice)                       │
│    ↓ Generates witty/helpful/sarcastic responses              │
├─────────────────────────────────────────────────────────────────┤
│ 5. TEXT-TO-SPEECH (Kokoro ONNX - Your Voice)                  │
│    ↓ Makes it sound human-ish                                 │
├─────────────────────────────────────────────────────────────────┤
│ 6. AUDIO PLAYBACK + FACE ANIMATION                             │
│    ↓ Streams audio while animating the face                   │
└─────────────────────────────────────────────────────────────────┘
```

### The Startup Sequence (Or: How Howdy Gets Out of Bed)

When you launch HowdyVox, here's what happens in the first 10-15 seconds:

1. **Loading Screen**: Animated "LOADING..." screen appears immediately (because waiting without feedback is torture)
2. **Port Cleanup**: Murders any zombie FastWhisperAPI processes squatting on port 8000
3. **FastWhisperAPI Launch**: Starts the local speech recognition server in a separate process (like a responsible parent)
4. **Model Preloading**: Loads Kokoro TTS and Ollama LLM into memory so your first response doesn't take geological time scales
5. **Face Initialization**: Loads your chosen face renderer (GIF or EchoEar) with that sweet rounded icon
6. **Loading Complete**: Screen transitions from loading animation to idle/waiting face
7. **Wake Word Listener**: Porcupine starts listening for "Hey Howdy" with minimal CPU usage
8. **Conversation Loop**: Once activated, Howdy enters conversation mode and won't shut up until you say "goodbye"

## Prerequisites (The Boring But Necessary Part) ✅

- **Python 3.10+**: Because backwards compatibility is for quitters
- **Virtual Environment**: Recommended unless you enjoy dependency hell
- **PyAudio 0.2.12**: Specifically this version for macOS compatibility (trust us)
- **Ollama**: Download from [ollama.com](https://ollama.com/) - this is Howdy's brain
- **Porcupine Key**: Free from [picovoice.ai](https://picovoice.ai/platform/porcupine/) - enables wake word detection
- **CUDA GPU**: Optional, but makes everything faster (like most hardware upgrades)
- **Apple Silicon**: Enhanced ONNX Runtime available for M-series Macs

### 🍎 M3 Mac (Apple Silicon) Users - Read This First!

If you're on an M3, M2, or M1 Mac, we've got you covered with automated setup:

```bash
# One-command automated setup
./setup_m3_mac.sh
```

This handles all the M3-specific quirks:
- Installs PortAudio and Opus via Homebrew
- Compiles PyAudio with proper Apple Silicon flags
- Configures library paths automatically
- Sets up conda environment activation scripts

**Or skip to the detailed guide:** See [M3_MAC_SETUP.md](M3_MAC_SETUP.md) for step-by-step instructions.

**Already have issues?** Run `./verify_installation.py` to diagnose problems.

## Setup (Let's Get This Show on the Road) 🔧

> **Note:** M3 Mac users should use `./setup_m3_mac.sh` instead of following these manual steps.

### 1. Clone this bad boy
```bash
git clone https://github.com/Jmi2020/HowdyVox.git
cd HowdyVox
```

### 2. Virtual environment time
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install all the things
```bash
pip install -r requirements.txt
```

### 4. PyAudio 0.2.12 specifically (yes, specifically)
```bash
pip uninstall -y pyaudio
pip install pyaudio==0.2.12
```

### 5. Porcupine wake word setup

Get a free key from [Picovoice Console](https://console.picovoice.ai/) then:
```bash
python quick_setup.py
```

Or manually create `.env`:
```
PORCUPINE_ACCESS_KEY="your-key-here"
LOCAL_MODEL_PATH="models"
ESP32_IP="192.168.1.xxx"  # Optional: For LED matrix display
```

### 6. Kokoro TTS voices

**Automatic (Easiest):**
```bash
pip install kokoro-onnx
```
Models auto-download to `~/.kokoro_onnx/` on first use.

**Manual (For control freaks):**
```bash
python Tests_Fixes/download_kokoro_onnx_direct.py --type q8
```

**Available voices:**
- `am_michael` - American male (default cowboy voice)
- `af_bella`, `af_nicole`, `af_sarah` - American female voices
- `bf_emma`, `bf_isabella` - British female voices
- `bm_lewis`, `bm_george` - British male voices
- Plus 15+ more in various languages

List them all:
```bash
python blend_voices.py --list-voices
```

### 7. FastWhisperAPI setup
```bash
cd FastWhisperAPI
pip install -r requirements.txt
uvicorn main:app --reload
```

### 8. Ollama (The Brain) setup

**Install Ollama** from [ollama.com](https://ollama.com/) then pick your poison:

**Fast & Efficient (Recommended for mortals):**
```bash
ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:latest
```

**Good Quality (For those with RAM to spare):**
```bash
ollama pull llama3.2:latest
```

**Big Brain Energy (For workstations that sound like jet engines):**
```bash
ollama pull mistral:latest
```

**Model Size Guide:**
- 3-4B parameters: Fast, low RAM (~4-6GB), good enough for most
- 7-8B parameters: Better quality, moderate RAM (~8-12GB), worth it
- 13B+ parameters: Highest quality, high RAM (~16GB+), overkill but fun

Test your model:
```bash
ollama run gemma-3-4b-it-GGUF:latest
```

Update `voice_assistant/config.py`:
```python
OLLAMA_LLM = "llama3.2:latest"  # Or your chosen model
```

### 9. (Optional) Apple Silicon optimization
```bash
python Tests_Fixes/fix_onnx_runtime.py
```

## Running HowdyVox (The Moment of Truth) 🎙️

### The New Hotness: Unified Face Launcher

```bash
# Launch with GIF face (low CPU, your custom animations)
python launch_howdy_face.py --face gif

# Launch with EchoEar face (more expressive, higher CPU)
python launch_howdy_face.py --face echoear

# Launch without face (for purists or potato computers)
python launch_howdy_face.py --face none
```

This launcher handles everything:
- Kills zombie FastWhisperAPI processes
- Starts FastWhisperAPI in the background
- Launches your chosen face renderer with that sweet rounded icon
- Starts the voice assistant with audio reactivity enabled
- Cleans up properly when you Ctrl+C

### The Old Reliable: Terminal-Only Launcher

```bash
python launch_scripts_backup/launch_howdy_terminal.py
```

Does the same thing but without the face. For minimalists and terminal purists.

### The Manual Way (For Those Who Trust Nothing)

```bash
# Terminal 1: FastWhisperAPI
cd FastWhisperAPI
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 (Optional): Face renderer
python gif_reactive_face.py  # or python echoear_face.py

# Terminal 3: Voice assistant
HOWDY_AUDIO_REACTIVE=1 python run_voice_assistant.py
```

### What Happens When You Run It

1. **Model Preloading** (10-15 seconds of anticipation)
   - Kokoro TTS loads voice models
   - Ollama LLM initializes
   - Face renderer loads and shows that beautiful rounded icon
   - This one-time cost means instant responses later

2. **Wake Word Mode** (The Waiting Game)
   - System says: "Listening for wake word 'Hey Howdy'..."
   - Porcupine listens with minimal CPU usage
   - Nothing is recorded until you say the magic words

3. **Conversation Mode** (The Main Event)
   - Face changes to "Listening" state
   - Speak naturally, Silero VAD knows when you're done
   - FastWhisperAPI transcribes locally (no cloud)
   - Ollama generates a response using your personality prompt
   - Kokoro TTS speaks with your chosen voice
   - Face animates in real-time based on speech characteristics
   - Repeat until you say goodbye (or an acceptable variant)

4. **Context Magic**
   - Each turn remembers previous exchanges
   - Ask follow-up questions naturally
   - No need to repeat "Hey Howdy" between turns
   - Context clears when you end the conversation

## Customization (Make It Your Own) ⚙️

### Personality Design (The Fun Part)

Edit `voice_assistant/config.py`:

```python
# Current default: George Carlin + Rodney Carrington (witty, direct, occasional darkness)
SYSTEM_PROMPT = (
    "You are George Carlin and Rodney Carrington as a single entity. "
    "Keep responses concise unless depth is essential. "
    "Maintain a neutral or lightly wry tone..."
)

# Or make it whatever you want:

# The Philosopher
SYSTEM_PROMPT = "You are Socrates, eternally asking 'but why?' until the user has an existential crisis."

# The Engineer
SYSTEM_PROMPT = "You are a senior software engineer with 20 years of experience and strong opinions about tabs vs spaces."

# The Motivational Speaker
SYSTEM_PROMPT = "You are a motivational speaker who believes everything can be solved with positive thinking and protein shakes."

# The Pessimist
SYSTEM_PROMPT = "You are Eeyore from Winnie the Pooh but with a computer science degree."
```

### Voice Selection

```python
KOKORO_VOICE = 'am_michael'  # Default cowboy voice
```

Choose from 20+ voices or blend them:
```bash
# Create a voice blend (40% Bella, 60% Michael)
python configure_blended_voice.py --name "my_blend" --voices "af_bella:40,am_michael:60"

# Then use it
KOKORO_VOICE = 'my_blend'
```

See [VoiceBlend.md](VoiceBlend.md) for the full guide.

### Face Customization

**For GIF Face:**
Just replace the files in `faceStates/` with your own animations:
- `waiting_blink_loop.gif` - Idle/waiting state
- `listening_glow_loop.gif` - User speaking
- `thinking_stars_motion.gif` - Processing
- `speaking_face.gif` - Assistant speaking

That's it. Done. The audio reactivity is automatic.

**For EchoEar Face:**
Edit `echoear_face.py`:
```python
CFG = {
    "size": 200,                 # Window size
    "bg": (0, 0, 0),            # Background color
    "eye_cyan": (0, 235, 255),  # Eye color (try different colors!)
    "ring": (40, 40, 40),       # Stage ring color
    "fps_speaking": 12,         # Higher = smoother but more CPU
    "head_nod_px": 4,           # How far the head nods
}
```

### Model Configuration

```python
TRANSCRIPTION_MODEL = 'fastwhisperapi'  # Local Whisper
RESPONSE_MODEL = 'ollama'               # Ollama LLM
TTS_MODEL = 'kokoro'                    # Kokoro ONNX
```

Change these if you want to swap in different backends. We won't judge (much).

## Features That Make This Special 🚀

### Audio Optimization That Actually Works

- **Adaptive Chunk Sizing**: Automatically adjusts based on response length
  - Short (<100 chars): 150-char chunks, 50ms delays
  - Medium (100-500): 180-char chunks, 100ms delays
  - Long (>500): 220-char chunks, 150ms delays
- **Pre-buffering**: Loads chunks while playing earlier ones
- **Gap Detection**: Handles generation delays gracefully
- **Result**: Smooth audio even on long responses. No stuttering. No weird pauses. Magic.

### Memory Management (Or: How to Not Eat All the RAM)

- **Targeted GC**: Specifically cleans up audio-related objects
- **Buffer Pooling**: Optimized memory usage for audio processing
- **Model Persistence**: Keeps models loaded but manages their memory footprint
- **Result**: Marathon conversations without memory leaks. Your RAM thanks you.

### Testing Suite (Because We're Not Animals)

15+ diagnostic scripts to verify everything works:

```bash
# The big kahuna
python Tests_Fixes/fix_all_issues.py

# Specific tests
python Tests_Fixes/test_kokoro_onnx.py      # TTS test
python Tests_Fixes/test_porcupine_fixes.py  # Wake word test
python Tests_Fixes/test_targeted_gc.py      # Memory management test
python microphone_test.py                    # Mic test
```

## Troubleshooting (When Things Go Sideways) 🔧

### Common Issues

**Wake word not working**
```bash
python quick_setup.py  # Reset Porcupine config
```

**PyAudio errors on macOS**
```bash
pip uninstall pyaudio && pip install pyaudio==0.2.12
```

**FastWhisperAPI connection failed**
```bash
# Check if it's running
curl http://localhost:8000
# If not, restart it
cd FastWhisperAPI && uvicorn main:app --reload
```

**Ollama not responding**
```bash
ollama list  # Check if model is installed
ollama pull your-model-name  # Install it
```

**Face not animating**
- Check if `HOWDY_AUDIO_REACTIVE=1` environment variable is set
- Verify UDP port 31337 isn't blocked
- Make sure the face window actually opened (check your dock)

**"python3.10" showing instead of "HowdyVox"**
```bash
pip install setproctitle  # Install the magic process rename library
```

**Audio stuttering**
Already fixed with adaptive chunking, but if it persists:
```bash
python Tests_Fixes/test_tts_fix.py
```

**ONNX Runtime issues (Apple Silicon)**
```bash
python Tests_Fixes/fix_onnx_runtime.py
```

### Advanced Troubleshooting

```bash
# Check all components
python Tests_Fixes/check_components.py

# Run comprehensive diagnostics
python Tests_Fixes/fix_all_issues.py

# Check environment
python Tests_Fixes/check_environment.py
```

## Optional Hardware Add-ons 🌟

### LED Matrix Display (ESP32-S3)

Flash an ESP32-S3 with the HowdyVox LED Matrix firmware (see `ESP32/` directory) for visual feedback:
- Waiting → Shows "Waiting" message
- Listening → Shows "Listening" indicator
- Thinking → Shows "Thinking" animation
- Speaking → Scrolls the response text
- Ending → Shows farewell message

Add to `.env`:
```
ESP32_IP=192.168.1.xxx
```

### Wireless ESP32P4 Microphones

Real-time audio streaming via UDP with OPUS compression for multi-room setups. See [ESP32P4_INTEGRATION.md](ESP32P4_INTEGRATION.md) for details.

## Why HowdyVox? (The Philosophy Section) 🤔

In an era where AI assistants require constant internet and send your conversations to data centers for "processing," HowdyVox takes a different approach:

- **Privacy First**: Your conversations stay on your machine. Full stop. End of story. No exceptions.
- **Actually Yours**: Customize everything. The voice, personality, model. Make it reflect your preferences, not a corporation's quarterly targets.
- **No Subscriptions**: No monthly fees. No API costs. No rate limits. Pay once (free, actually) and it's yours forever.
- **Open Source**: Every component is inspectable. You can see exactly how it works, modify it, improve it, or just make fun of our code comments.
- **Offline First**: No internet? No problem. HowdyVox works anywhere, anytime. On a plane, in a cabin, during the apocalypse.

HowdyVox proves that powerful AI assistants don't need to compromise your privacy or charge you monthly rent. It's conversational AI done right - local, fast, and completely under your control.

## Documentation Deep Dives 📚

Want to know more? We've got guides for days:

- [GIF_REACTIVE_FACE_GUIDE.md](GIF_REACTIVE_FACE_GUIDE.md) - Complete guide to GIF face customization
- [ECHOEAR_FACE_GUIDE.md](ECHOEAR_FACE_GUIDE.md) - EchoEar face technical documentation
- [VoiceBlend.md](VoiceBlend.md) - Voice blending guide
- [TTS_STUTTERING_FIX_README.md](TTS_STUTTERING_FIX_README.md) - Audio optimization details
- [TTS_ENHANCEMENT_IMPLEMENTATION.md](TTS_ENHANCEMENT_IMPLEMENTATION.md) - Performance enhancements
- [ESP32P4_INTEGRATION.md](ESP32P4_INTEGRATION.md) - Wireless microphone setup
- [Tests_Fixes/test_and_run_instructions.md](Tests_Fixes/test_and_run_instructions.md) - Testing suite guide

## Contributing (Join the Circus) 🤝

This project welcomes contributions! Whether it's:
- Bug fixes (we promise we left some for you)
- New voice personalities (the weirder the better)
- Additional LLM integrations (yes, that one too)
- Documentation improvements (make it even more entertaining)
- Performance optimizations (because faster is always better)

Feel free to open issues or submit pull requests. We're friendly, mostly.

## License 📄

MIT License - See LICENSE file for legalese

Translation: Do whatever you want with it. We're not your lawyer.

---

**"Your AI, your rules, your privacy. Welcome to HowdyVox."** 🤠

*P.S. - If you made it this far, you deserve a cookie. We don't have cookies, but we have code. Close enough.*
