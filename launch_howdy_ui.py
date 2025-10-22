#!/usr/bin/env python3
"""
UI-based launcher for HowdyVox Voice Assistant
Launches FastWhisperAPI and voice assistant with a lightweight tkinter UI
"""

import subprocess
import os
import time
import signal
import sys
import threading
import platform
from ui_interface import create_ui

# Track background processes
fastapi_proc = None
voice_proc = None
ui_root = None
ui_instance = None


def cleanup(sig=None, frame=None):
    """Cleanup all processes on exit"""
    global fastapi_proc, voice_proc, ui_root
    print("\n\nShutting down HowdyVox...")

    # Kill voice assistant
    if voice_proc:
        try:
            voice_proc.terminate()
            voice_proc.wait(timeout=3)
        except:
            try:
                voice_proc.kill()
            except:
                pass

    # Kill FastAPI
    if fastapi_proc:
        try:
            os.killpg(os.getpgid(fastapi_proc.pid), signal.SIGTERM)
        except:
            pass

    subprocess.run("pkill -f 'uvicorn main:app'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)

    # Close UI
    if ui_root:
        try:
            ui_root.quit()
        except:
            pass

    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


def start_fastapi():
    """Start FastWhisperAPI in background"""
    global fastapi_proc

    print("🤠 HowdyVox UI Launcher")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Kill any existing FastAPI instances
    subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)
    time.sleep(1)

    # Start FastAPI in background
    print("\n1️⃣ Starting FastWhisperAPI...")
    cmd = "cd FastWhisperAPI && /opt/anaconda3/bin/conda run -n howdy310 --no-capture-output python -u -m uvicorn main:app --host 127.0.0.1 --port 8000"
    fastapi_proc = subprocess.Popen(
        cmd,
        shell=True,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("   Waiting for FastAPI to initialize...")
    time.sleep(8)
    print("   ✓ FastWhisperAPI ready")


def start_voice_assistant():
    """Start voice assistant as subprocess"""
    global voice_proc

    print("\n2️⃣ Starting Voice Assistant with UI...")
    print("=" * 50)
    print()

    cmd = [
        "/opt/anaconda3/bin/conda", "run", "-n", "howdy310",
        "--no-capture-output", "python", "-u", "run_voice_assistant.py"
    ]

    voice_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1
    )

    return voice_proc


def monitor_voice_assistant():
    """Monitor voice assistant in separate thread"""
    if ui_instance and voice_proc:
        ui_instance.monitor_process(voice_proc)


def main():
    """Main launcher function"""
    global ui_root, ui_instance

    # Auto-configure library paths for M3 Mac
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        opus_lib_path = "/opt/homebrew/opt/opus/lib"
        if os.path.exists(opus_lib_path):
            current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            if opus_lib_path not in current_path:
                os.environ["DYLD_LIBRARY_PATH"] = f"{opus_lib_path}:{current_path}"
                print("✓ Auto-configured Opus library path for M3 Mac")

    try:
        # Start FastAPI
        start_fastapi()

        # Create UI
        print("3️⃣ Launching UI...")
        ui_root, ui_instance = create_ui()

        # Add initial message
        ui_instance.add_message("HowdyVox initialized. Say 'Hey Howdy' to start!", 'system')

        # Start voice assistant
        voice_proc = start_voice_assistant()

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_voice_assistant, daemon=True)
        monitor_thread.start()

        print("   ✓ UI launched successfully")
        print("\n" + "=" * 50)
        print("HowdyVox is running! Check the UI window.")
        print("=" * 50 + "\n")

        # Run UI main loop
        ui_root.protocol("WM_DELETE_WINDOW", cleanup)
        ui_root.mainloop()

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
