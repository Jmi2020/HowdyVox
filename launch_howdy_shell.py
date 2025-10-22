#!/usr/bin/env python3
"""
Shell-based launcher that runs commands directly in terminal
"""

import subprocess
import os
import time
import signal
import sys
import platform

# Track background process
bg_proc = None

def cleanup(sig=None, frame=None):
    global bg_proc
    print("\n\nShutting down...")
    if bg_proc:
        try:
            os.killpg(os.getpgid(bg_proc.pid), signal.SIGTERM)
        except:
            pass
    subprocess.run("pkill -f 'uvicorn main:app'", shell=True)
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

print("🤠 HowdyVox Shell Launcher")
print("=" * 50)

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Auto-configure library paths for M3 Mac
if platform.machine() == "arm64" and platform.system() == "Darwin":
    opus_lib_path = "/opt/homebrew/opt/opus/lib"
    if os.path.exists(opus_lib_path):
        current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
        if opus_lib_path not in current_path:
            os.environ["DYLD_LIBRARY_PATH"] = f"{opus_lib_path}:{current_path}"
            print(f"✓ Auto-configured Opus library path for M3 Mac")
    else:
        print(f"⚠️  Warning: Opus library not found at {opus_lib_path}")
        print(f"   Wireless audio may not work. Run: brew install opus")

# Kill existing FastAPI
subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)
time.sleep(1)

# Start FastAPI in background using shell
print("\n1️⃣ Starting FastWhisperAPI...")
cmd = "cd FastWhisperAPI && /opt/anaconda3/bin/conda run -n howdy310 --no-capture-output python -u -m uvicorn main:app --host 127.0.0.1 --port 8000 &"
bg_proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

print("   Waiting for FastAPI to initialize...")
time.sleep(8)

# Run voice assistant in foreground
print("\n2️⃣ Starting Voice Assistant...")
print("=" * 50)
print()

# This will show all output in the terminal
os.system("/opt/anaconda3/bin/conda run -n howdy310 --no-capture-output python -u run_voice_assistant.py")

# Cleanup when voice assistant exits
cleanup()