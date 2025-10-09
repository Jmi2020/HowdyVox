#!/usr/bin/env python3
"""
Terminal-based launcher that preserves all output
"""

import subprocess
import sys
import os
import signal
import time

# Configurable conda environment (can be set via env variable or edit this default)
CONDA_ENV = os.getenv("HOWDYVOX_CONDA_ENV", "howdy310")
CONDA_PATH = os.getenv("CONDA_PATH", "/opt/anaconda3/bin/conda")

# Store process references for cleanup
processes = []

def signal_handler(sig, frame):
    print("\n\nShutting down HowdyVox...")
    for proc in processes:
        try:
            proc.terminate()
        except:
            pass
    time.sleep(1)
    for proc in processes:
        try:
            proc.kill()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("🎙️ HowdyVox Terminal Launcher")
print("=" * 50)
print(f"Using conda environment: {CONDA_ENV}")
print(f"Conda path: {CONDA_PATH}")
print("=" * 50)

# Kill any existing process on port 8000
print("\nChecking port 8000...")
try:
    result = subprocess.run(['lsof', '-ti', ':8000'], capture_output=True, text=True)
    if result.stdout.strip():
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            subprocess.run(['kill', '-9', pid])
        print(f"Killed existing process(es): {', '.join(pids)}")
        time.sleep(2)
except:
    pass

# Start FastAPI in background
print("\n1️⃣ Starting FastWhisperAPI in background...")
fastapi_cmd = [
    CONDA_PATH, "run", "-n", CONDA_ENV,
    "--no-capture-output",  # Important: don't capture output
    "python", "-u", "-m", "uvicorn", "main:app",
    "--host", "127.0.0.1", "--port", "8000"
]

# Start FastAPI without capturing output - let it go to terminal
fastapi_proc = subprocess.Popen(
    fastapi_cmd,
    cwd="FastWhisperAPI",
    preexec_fn=os.setsid  # Create new process group
)
processes.append(fastapi_proc)

print("   FastAPI starting in background...")
print("   Waiting 8 seconds for initialization...")
time.sleep(8)

# Now start voice assistant in foreground
print("\n2️⃣ Starting Voice Assistant (foreground)...")
print("=" * 50)
print()

va_cmd = [
    CONDA_PATH, "run", "-n", CONDA_ENV,
    "--no-capture-output",  # Important: don't capture output
    "python", "-u", "run_voice_assistant.py"
]

try:
    # Run voice assistant in foreground so we see all output
    va_proc = subprocess.run(va_cmd)
except KeyboardInterrupt:
    print("\nShutdown initiated...")

# Cleanup
signal_handler(None, None)