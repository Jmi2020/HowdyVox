# Face Modules

This folder contains face rendering modules for HowdyVox's audio-reactive visual interface.

## Face Implementations

### EchoEar Face (Rendered)
- **`echoear_face.py`** - Real-time rendered face with audio-reactive features
- **`launch_howdy_echoear.py`** - Launcher for EchoEar face
- Dynamic eye sizing, horizontal squeeze, and head nodding based on audio analysis

### GIF Face (Efficient)
- **`gif_reactive_face.py`** - GIF-based audio-reactive face
- **`launch_howdy_face.py`** - Launcher for GIF face
- Uses your own GIF files with audio-reactive playback speed

## Utilities

- **`test_face_ui.py`** - Test face rendering without voice assistant
- **`tts_reactive_meter.py`** - Audio reactivity meter for testing

## Quick Start

### Run EchoEar Face
```bash
python face_modules/launch_howdy_echoear.py
```

### Run GIF Face
```bash
python face_modules/launch_howdy_face.py
```

### Test Face Rendering
```bash
python face_modules/test_face_ui.py
```

## Integration

Both face modules integrate with the main voice assistant:

- **Shell Launcher**: `python launch_howdy_shell.py` (no face)
- **UI Launcher**: `python launch_howdy_ui.py` (includes animated face)

The UI launcher uses `face_animator.py` from the root directory.

## Audio Features

Both face types react to:
- **Volume (RMS)** - Louder speech = bigger eyes or faster GIF playback
- **Sibilance (ZCR)** - "S" and "SH" sounds = horizontal squeeze
- **Emphasis (Peaks)** - Speech emphasis = head nod or frame changes

## Documentation

See [docs/](../docs/) for detailed guides:
- [ANIMATED_FACE_README.md](../docs/ANIMATED_FACE_README.md) - Overview
- [ECHOEAR_FACE_GUIDE.md](../docs/ECHOEAR_FACE_GUIDE.md) - EchoEar details
- [GIF_REACTIVE_FACE_GUIDE.md](../docs/GIF_REACTIVE_FACE_GUIDE.md) - GIF face details

## Navigation

- Back to [Main README](../README.md)
- See [docs/](../docs/) for documentation
- See [utilities/](../utilities/) for helper scripts
