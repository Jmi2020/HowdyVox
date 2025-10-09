#!/usr/bin/env python3
"""
Comprehensive check of all HowdyVox components
"""

import os
import sys
import platform
import subprocess
import importlib.util
import time
import json
from pathlib import Path

def check_command(command):
    """Check if a command is available"""
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_file_exists(file_path):
    """Check if a file exists"""
    return os.path.exists(file_path)

def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n----- Checking Ollama -----")
    
    # Check if ollama command is available
    if not check_command(["ollama", "--version"]):
        print("❌ Ollama command not found. Please install Ollama from https://ollama.com/")
        return False
    
    # Check if Ollama service is running
    try:
        result = subprocess.run(["ollama", "list"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               check=True)
        print("✅ Ollama is installed and running")
        print(f"Available models: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ollama is installed but not running or returned an error")
        return False

def check_fastwhisper():
    """Check if FastWhisperAPI is set up and running"""
    print("\n----- Checking FastWhisperAPI -----")
    
    # Check if FastWhisperAPI directory exists
    fastwhisper_dir = os.path.join(os.getcwd(), "FastWhisperAPI")
    if not os.path.isdir(fastwhisper_dir):
        print("❌ FastWhisperAPI directory not found")
        return False
    
    # Check if requirements are installed
    if not check_import("faster_whisper"):
        print("❌ faster_whisper not installed. Run: pip install faster_whisper")
        return False
    
    if not check_import("fastapi"):
        print("❌ fastapi not installed. Run: pip install fastapi[standard]")
        return False
    
    if not check_import("uvicorn"):
        print("❌ uvicorn not installed. Run: pip install uvicorn")
        return False
    
    # Check if service is running
    import requests
    try:
        response = requests.get("http://localhost:8000/info", timeout=2)
        if response.status_code == 200:
            print("✅ FastWhisperAPI is running")
            print(f"API info: {response.json()}")
            return True
        else:
            print("❌ FastWhisperAPI returned an error")
            return False
    except requests.exceptions.RequestException:
        print("❌ FastWhisperAPI is not running")
        print("   Start it with: cd FastWhisperAPI && uvicorn main:app --reload")
        return False

def check_kokoro():
    """Check if Kokoro TTS is installed and model files exist"""
    print("\n----- Checking Kokoro TTS -----")
    
    # Check if kokoro_onnx is installed
    if not check_import("kokoro_onnx"):
        print("❌ kokoro_onnx not installed. Run: pip install kokoro-onnx")
        return False
    
    # Check ONNX Runtime
    if check_import("onnxruntime_silicon"):
        print("✅ onnxruntime-silicon is installed (optimized for Apple Silicon)")
    elif check_import("onnxruntime"):
        print("⚠️ Standard onnxruntime is installed (not optimized for Apple Silicon)")
        print("   Consider installing onnxruntime-silicon for better performance")
    else:
        print("❌ No version of onnxruntime is installed")
        print("   Run: pip install onnxruntime-silicon==1.16.3")
        return False
    
    # Check model files
    model_file = os.path.join(os.getcwd(), "models", "kokoro-v1.0.onnx")
    voices_file = os.path.join(os.getcwd(), "models", "voices-v1.0.bin")
    
    if not check_file_exists(model_file):
        print(f"❌ Kokoro model file not found: {model_file}")
        return False
    
    if not check_file_exists(voices_file):
        print(f"❌ Kokoro voices file not found: {voices_file}")
        return False
    
    print("✅ Kokoro TTS model and voices files found")
    
    # Check voice files
    voices_dir = os.path.join(os.getcwd(), "models", "voices")
    if not os.path.isdir(voices_dir):
        print("⚠️ Voices directory not found")
    else:
        voices = [f for f in os.listdir(voices_dir) if f.endswith('.bin')]
        if voices:
            print(f"✅ Found {len(voices)} voice files")
            if "am_michael.bin" in voices:
                print("✅ Default cowboy voice (am_michael.bin) is available")
            else:
                print("⚠️ Default cowboy voice (am_michael.bin) not found")
        else:
            print("⚠️ No voice files found in voices directory")
    
    return True

def check_porcupine():
    """Check if Porcupine wake word detection is set up"""
    print("\n----- Checking Porcupine Wake Word Detection -----")
    
    # Check if pvporcupine is installed
    if not check_import("pvporcupine"):
        print("❌ pvporcupine not installed. Run: pip install pvporcupine")
        return False
    
    # Check for access key in .env file
    env_file = os.path.join(os.getcwd(), ".env")
    has_access_key = False
    
    if check_file_exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
            if "PORCUPINE_ACCESS_KEY" in content:
                has_access_key = True
    
    if not has_access_key:
        print("❌ PORCUPINE_ACCESS_KEY not found in .env file")
        print("   Create .env file with: PORCUPINE_ACCESS_KEY='your-key-here'")
        return False
    
    # Check wake word model
    model_file = os.path.join(os.getcwd(), "models", "Hey-howdy_en_mac_v3_0_0.ppn")
    if not check_file_exists(model_file):
        print(f"❌ Porcupine wake word model not found: {model_file}")
        return False
    
    print("✅ Porcupine wake word detection is properly set up")
    return True

def check_audio():
    """Check audio setup for recording and playback"""
    print("\n----- Checking Audio Setup -----")
    
    # Check PyAudio
    if not check_import("pyaudio"):
        print("❌ PyAudio not installed. Run: pip install PyAudio==0.2.14")
        return False
    
    import pyaudio
    print(f"✅ PyAudio is installed (version {pyaudio.__version__})")
    
    # List audio devices
    p = pyaudio.PyAudio()
    input_devices = []
    output_devices = []
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            input_devices.append(device_info)
        if device_info['maxOutputChannels'] > 0:
            output_devices.append(device_info)
    
    p.terminate()
    
    if input_devices:
        print(f"✅ Found {len(input_devices)} input (microphone) devices")
    else:
        print("❌ No microphone devices found")
    
    if output_devices:
        print(f"✅ Found {len(output_devices)} output (speaker) devices")
    else:
        print("❌ No speaker devices found")
    
    # Check additional audio libraries
    audio_libs = ["pygame", "soundfile", "sounddevice", "pydub"]
    missing = []
    
    for lib in audio_libs:
        if not check_import(lib):
            missing.append(lib)
    
    if missing:
        print(f"❌ Missing audio libraries: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
    else:
        print("✅ All required audio libraries are installed")
    
    return len(missing) == 0

def main():
    print("\n===== HowdyVox Component Check =====\n")
    
    # System info
    print("----- System Information -----")
    print(f"Python version: {sys.version}")
    print(f"System: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    # Check each component
    audio_ok = check_audio()
    kokoro_ok = check_kokoro()
    porcupine_ok = check_porcupine()
    fastwhisper_ok = check_fastwhisper()
    ollama_ok = check_ollama()
    
    # Summary
    print("\n===== Component Check Summary =====")
    components = {
        "Audio Setup": audio_ok,
        "Kokoro TTS": kokoro_ok,
        "Porcupine Wake Word": porcupine_ok,
        "FastWhisperAPI": fastwhisper_ok,
        "Ollama LLM": ollama_ok
    }
    
    all_ok = True
    for component, status in components.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\n✅ All components are properly set up! You can run HowdyVox now.")
        print("   Use: python run_voice_assistant.py")
    else:
        print("\n❌ Some components are missing or misconfigured.")
        print("   Please fix the issues above before running HowdyVox.")
    
    print("\n===== Component Check Complete =====")

if __name__ == "__main__":
    main()