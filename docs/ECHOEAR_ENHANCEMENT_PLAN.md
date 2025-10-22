# EchoEar-Style Face Enhancement Plan

## Current Implementation vs EchoEar Spec

### Current (Basic 8-bit Face)
**Pros:**
- ✅ Simple tkinter Canvas rendering
- ✅ Integrated directly into UI
- ✅ State-based animation (idle, listening, thinking, speaking)
- ✅ Low overhead (~2-3% CPU)
- ✅ Basic expressiveness (blinking, mouth cycle, thinking dots)

**Cons:**
- ❌ No audio reactivity (state-based only)
- ❌ Simple geometric shapes (rectangles)
- ❌ No visual polish (no glow, no dynamic scaling)
- ❌ Limited expressiveness (can't react to speech nuances)

### EchoEar-Style (Audio-Reactive)
**Pros:**
- ✅ **Audio-reactive** - responds to actual speech characteristics:
  - **RMS envelope** → eye size/glow intensity (louder = bigger eyes)
  - **Zero-crossing rate (ZCR)** → horizontal squeeze (sibilants = narrower eyes)
  - **Peak detection** → brief head nod (emphasis)
- ✅ Visual polish:
  - Circular stage with ring
  - Rounded-rectangle eyes with glow effect
  - Alpha blending for smooth visuals
- ✅ More expressive and lifelike
- ✅ Modular architecture (separate analyzer + renderer)

**Cons:**
- ⚠️ Requires pygame (additional dependency)
- ⚠️ Slightly higher complexity
- ⚠️ Need to wire audio analysis into TTS

## Key Enhancements

### 1. Audio Reactivity Features
```
Speech Characteristics → Visual Response
├── Volume (RMS)      → Eye size pulse (0.2 → 1.0 scale)
├── Sibilance (ZCR)   → Horizontal squeeze (narrow eyes for "s", "sh")
└── Emphasis (Peak)   → Brief head nod (2 frames)
```

### 2. Visual Quality
- **Circular stage** instead of rectangular head
- **Rounded-rectangle eyes** (cyan) with **glow effect**
- **Alpha blending** for smooth compositing
- **Precomputed surfaces** for performance

### 3. Modular Architecture
```
┌─────────────────────┐         ┌──────────────────┐
│  TTS Audio Stream   │         │   Face Renderer  │
│   (Mac/Primary)     │         │   (Any display)  │
└─────────┬───────────┘         └────────▲─────────┘
          │                              │
          ▼                              │
┌─────────────────────┐                 │
│  Audio Analyzer     │    UDP Messages │
│  (ReactiveMeter)    │─────────────────┘
│  - RMS envelope     │   ~12 Hz
│  - ZCR calculation  │   level:0.63;zcr=0.18;peak=1
│  - Peak detection   │
└─────────────────────┘
```

## Implementation Strategy

### Option A: Replace Existing Face (Recommended)
**Approach:** Keep tkinter UI, replace face_animator.py with pygame-based EchoEar face

**Pros:**
- Cleaner architecture
- Better performance (pygame optimized for graphics)
- Full EchoEar feature set

**Cons:**
- Need to handle pygame window separately from tkinter
- Slightly more complex setup

### Option B: Hybrid Approach
**Approach:** Keep both implementations, let user choose

**Pros:**
- Backwards compatibility
- Users can choose based on hardware

**Cons:**
- More code to maintain
- Confusing for users

### Option C: Enhance Current Implementation
**Approach:** Add audio analysis to existing tkinter Canvas face

**Pros:**
- Maintains integration with existing UI
- No new dependencies

**Cons:**
- tkinter Canvas less optimized for graphics
- Won't get full visual polish of pygame

## Recommended: Option A (Replace with EchoEar)

### Phase 1: Create EchoEar Face Renderer
- [x] Base from spec's echoear_face.py
- [ ] Adapt to HowdyVox states (idle, listening, thinking, speaking)
- [ ] Keep UDP interface for flexibility
- [ ] Test standalone with dummy audio data

### Phase 2: Implement Audio Analyzer
- [ ] Create tts_reactive_meter.py (from spec)
- [ ] Wire into existing TTS playback in text_to_speech.py
- [ ] Extract PCM data from Kokoro TTS chunks
- [ ] Test analyzer standalone

### Phase 3: Integration
- [ ] Launch EchoEar face as separate window (or embedded)
- [ ] Connect analyzer → face via UDP (localhost)
- [ ] Update launch_howdy_ui.py to start both
- [ ] Test end-to-end with real conversations

### Phase 4: Polish
- [ ] Add configuration options (size, colors, UDP port)
- [ ] Create test utilities
- [ ] Update documentation
- [ ] Performance profiling

## Performance Targets

- **Face Renderer**: < 5-10% CPU on typical hardware (160x160, 12 FPS speaking)
- **Audio Analyzer**: < 1% CPU overhead (C-accelerated audioop)
- **Total Overhead**: < 5-12% CPU (compared to 2-3% currently - acceptable tradeoff for expressiveness)
- **Latency**: < 50ms from audio → visual response (UDP @ 12 Hz)

## Technical Details

### Audio Features Explained

**1. RMS (Root Mean Square) - Volume/Energy**
```python
# Measures "loudness" of speech
rms = audioop.rms(pcm_chunk, sample_width)
# Mapped to 0.0-1.0 → controls eye size scale
```

**2. ZCR (Zero-Crossing Rate) - Brightness/Sibilance**
```python
# Counts how often signal crosses zero
# High ZCR = sibilants (s, sh, ch, f)
# Low ZCR = vowels (a, e, i, o, u)
zcr = count_zero_crossings(pcm_chunk) / len(pcm_chunk)
# Mapped to horizontal eye squeeze
```

**3. Peak Detection - Emphasis/Onset**
```python
# Detects sudden energy increases
if current_rms > previous_rms * threshold:
    peak_detected = True  # triggers head nod
```

### Why This Works

1. **Perceptual Alignment**: Visual cues match what humans perceive in speech
   - Loud = bigger, more intense (eye size)
   - Bright/sibilant = sharper, more focused (narrower eyes)
   - Emphasis = physical head movement (nod)

2. **Minimal CPU**: Uses stdlib `audioop` (C-accelerated)
   - No NumPy/SciPy required
   - No FFT or spectral analysis
   - Just simple time-domain features

3. **Low Bandwidth**: UDP messages @ 12 Hz
   - ~64 bytes per message
   - < 1 KB/s network traffic
   - Works great over local network or localhost

## Migration Path

### For Users
1. **No breaking changes** - launch command stays the same
2. **Optional fallback** - keep basic face as backup
3. **Configuration file** - choose face style in config

### For Developers
1. **Clear deprecation** - mark old face_animator.py as legacy
2. **Documentation** - update all guides with new features
3. **Testing** - comprehensive test suite for audio analysis

## Next Steps

1. ✅ Create this planning document
2. [ ] Implement echoear_face.py (pygame renderer)
3. [ ] Implement tts_reactive_meter.py (audio analyzer)
4. [ ] Wire analyzer into text_to_speech.py
5. [ ] Update launch scripts
6. [ ] Test and profile
7. [ ] Document new features

## Questions to Resolve

1. **Window management**: Separate pygame window or embedded in tkinter?
   - **Decision**: Separate window for now (simpler, better performance)

2. **Fallback**: Keep old face_animator.py as backup?
   - **Decision**: Yes, as legacy option

3. **Configuration**: How to let users customize?
   - **Decision**: Add config options to voice_assistant/config.py

4. **Testing**: How to test without full voice assistant?
   - **Decision**: Standalone test scripts for each component

## Expected Results

After implementation:
- ✅ Face reacts to speech volume (bigger eyes when loud)
- ✅ Face reacts to sibilants (narrower eyes for "s" sounds)
- ✅ Face nods on emphasis (peaks in speech)
- ✅ Smooth, expressive animations
- ✅ Minimal performance impact
- ✅ Works locally or over network (Pi support)

**Visual Demo Examples:**
- Speaking "**Hello**" - eyes pulse with "He-" and "-lo"
- Speaking "**Ssssnake**" - eyes squeeze narrow during "Ssss"
- Speaking "**YES!**" - eyes enlarge + brief head nod on emphasis
