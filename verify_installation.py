#!/usr/bin/env python3
"""
Installation Verification Script for HowdyVox
Checks all dependencies and configurations
"""

import sys
import os
import platform
import subprocess

# Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

errors = []
warnings = []

def print_header(text):
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{text:^60}{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")

def check(condition, success_msg, error_msg, is_warning=False):
    """Check a condition and print result"""
    if condition:
        print(f"{GREEN}✓{NC} {success_msg}")
        return True
    else:
        if is_warning:
            print(f"{YELLOW}⚠{NC} {error_msg}")
            warnings.append(error_msg)
        else:
            print(f"{RED}✗{NC} {error_msg}")
            errors.append(error_msg)
        return False

def test_import(module_name, display_name=None):
    """Test if a module can be imported"""
    if display_name is None:
        display_name = module_name

    try:
        __import__(module_name)
        print(f"{GREEN}✓{NC} {display_name}")
        return True
    except ImportError as e:
        print(f"{RED}✗{NC} {display_name} - {e}")
        errors.append(f"Failed to import {display_name}")
        return False

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

print_header("HowdyVox Installation Verification")

# System Info
print(f"{BLUE}System Information:{NC}")
print(f"  Platform: {platform.system()}")
print(f"  Machine: {platform.machine()}")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Python Path: {sys.executable}")

# Check Python version
print_header("Python Environment")
py_version = sys.version_info
check(
    py_version.major == 3 and py_version.minor == 10,
    f"Python 3.10.x (found {py_version.major}.{py_version.minor}.{py_version.micro})",
    f"Python 3.10 required (found {py_version.major}.{py_version.minor}.{py_version.micro})"
)

# Check if in conda environment
in_conda = "CONDA_DEFAULT_ENV" in os.environ
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "None")
check(
    in_conda,
    f"Running in conda environment: {conda_env}",
    "Not in conda environment (expected 'howdy310')",
    is_warning=True
)

# Check M3-specific requirements
if platform.machine() == "arm64" and platform.system() == "Darwin":
    print_header("Apple Silicon (M3) Configuration")

    # Check Homebrew
    has_brew = run_command("which brew")
    check(has_brew, "Homebrew installed", "Homebrew not found")

    if has_brew:
        # Check PortAudio
        has_portaudio = run_command("brew list portaudio")
        check(has_portaudio, "PortAudio installed", "PortAudio not installed (brew install portaudio)")

        # Check Opus
        has_opus = run_command("brew list opus")
        check(has_opus, "Opus installed", "Opus not installed (brew install opus)")

        # Check library path
        opus_path = "/opt/homebrew/opt/opus/lib"
        dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
        check(
            opus_path in dyld_path or os.path.exists(opus_path),
            "Opus library path configured",
            f"DYLD_LIBRARY_PATH not set (export DYLD_LIBRARY_PATH={opus_path}:$DYLD_LIBRARY_PATH)",
            is_warning=True
        )

# Check core Python packages
print_header("Core Dependencies")

test_import("pyaudio", "PyAudio")
test_import("pygame", "Pygame")
test_import("dotenv", "python-dotenv")
test_import("colorama", "Colorama")
test_import("requests", "Requests")

# Check audio libraries
print_header("Audio Processing")

test_import("sounddevice", "SoundDevice")
test_import("soundfile", "SoundFile")
test_import("pydub", "Pydub")
test_import("speech_recognition", "SpeechRecognition")

# Check PyObjC (macOS only)
if platform.system() == "Darwin":
    print_header("macOS Frameworks")
    test_import("objc", "PyObjC Core")
    test_import("AVFoundation", "PyObjC AVFoundation")
    test_import("CoreAudio", "PyObjC CoreAudio")

# Check ML/AI libraries
print_header("AI/ML Libraries")

test_import("onnxruntime", "ONNX Runtime")
test_import("torch", "PyTorch")
test_import("torchaudio", "TorchAudio")
test_import("numpy", "NumPy")

# Check service clients
print_header("Service Clients")

test_import("ollama", "Ollama")
test_import("openai", "OpenAI")
test_import("groq", "Groq")
test_import("deepgram", "Deepgram SDK")

# Check TTS/STT
print_header("Speech Synthesis & Recognition")

test_import("kokoro_onnx", "Kokoro ONNX")
test_import("pvporcupine", "Porcupine")

# Check wireless support
print_header("Wireless Audio Support")

test_import("opuslib", "OpusLib")
test_import("websocket", "WebSocket Client")

# Check voice assistant modules
print_header("Voice Assistant Modules")

test_import("voice_assistant.config", "Config")
test_import("voice_assistant.kokoro_manager", "Kokoro Manager")
test_import("voice_assistant.transcription", "Transcription")
test_import("voice_assistant.response_generation", "Response Generation")
test_import("voice_assistant.text_to_speech", "Text-to-Speech")
test_import("voice_assistant.wake_word", "Wake Word")

# Check environment file
print_header("Configuration Files")

env_exists = os.path.exists(".env")
check(env_exists, ".env file exists", ".env file not found (copy from example.env)")

if env_exists:
    from dotenv import load_dotenv
    load_dotenv()

    porcupine_key = os.getenv("PORCUPINE_ACCESS_KEY")
    check(
        porcupine_key and porcupine_key != "your-porcupine-key-here",
        "Porcupine access key configured",
        "Porcupine access key not set in .env",
        is_warning=True
    )

    model_path = os.getenv("LOCAL_MODEL_PATH", "models")
    check(
        os.path.exists(model_path),
        f"Model path exists: {model_path}",
        f"Model path not found: {model_path}",
        is_warning=True
    )

# Check Ollama
print_header("Ollama LLM")

ollama_running = run_command("curl -s http://localhost:11434/api/tags > /dev/null 2>&1")
check(
    ollama_running,
    "Ollama running on port 11434",
    "Ollama not running (start with: ollama serve)",
    is_warning=True
)

# Check FastWhisperAPI directory
print_header("FastWhisperAPI")

fastapi_exists = os.path.exists("FastWhisperAPI")
check(
    fastapi_exists,
    "FastWhisperAPI directory exists",
    "FastWhisperAPI directory not found"
)

if fastapi_exists:
    check(
        os.path.exists("FastWhisperAPI/main.py"),
        "FastWhisperAPI main.py found",
        "FastWhisperAPI main.py not found"
    )

# Summary
print_header("Verification Summary")

total_checks = len(errors) + len(warnings)
print(f"Errors: {len(errors)}")
print(f"Warnings: {len(warnings)}")
print()

if errors:
    print(f"{RED}Critical Issues:{NC}")
    for error in errors:
        print(f"  • {error}")
    print()

if warnings:
    print(f"{YELLOW}Warnings:{NC}")
    for warning in warnings:
        print(f"  • {warning}")
    print()

if not errors and not warnings:
    print(f"{GREEN}✓ All checks passed! HowdyVox is ready to run.{NC}")
    print()
    print(f"{BLUE}To start HowdyVox:{NC}")
    if platform.machine() == "arm64":
        print(f"  ./launch_howdy_m3.sh")
    else:
        print(f"  python launch_howdy_shell.py")
    print(f"  # or")
    print(f"  python launch_howdy_ui.py")
    sys.exit(0)
elif not errors:
    print(f"{YELLOW}Setup is mostly complete with some warnings.{NC}")
    print(f"{YELLOW}HowdyVox should work but may have limited functionality.{NC}")
    sys.exit(0)
else:
    print(f"{RED}Setup incomplete. Please fix the errors above.{NC}")
    print()
    print(f"{BLUE}For M3 Mac, try running:{NC}")
    print(f"  ./setup_m3_mac.sh")
    print()
    print(f"{BLUE}Or follow the manual setup guide:{NC}")
    print(f"  M3_MAC_SETUP.md")
    sys.exit(1)
