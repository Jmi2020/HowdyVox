#!/usr/bin/env python3
"""
HowdyVox Launcher with EchoEar-style Audio-Reactive Face
Launches FastWhisperAPI, Voice Assistant, and EchoEar face renderer
"""

import subprocess
import os
import time
import signal
import sys
import platform

# Track all processes
fastapi_proc = None
voice_proc = None
face_proc = None


def cleanup(sig=None, frame=None):
    """Cleanup all processes on exit"""
    global fastapi_proc, voice_proc, face_proc
    print("\n\nShutting down HowdyVox...")

    # Kill face renderer
    if face_proc:
        try:
            face_proc.terminate()
            face_proc.wait(timeout=3)
        except:
            try:
                face_proc.kill()
            except:
                pass

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
    subprocess.run("pkill -f 'echoear_face'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)

    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


def main():
    """Main launcher function"""
    global fastapi_proc, voice_proc, face_proc

    print("🤠 HowdyVox with EchoEar Face Launcher")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Kill any existing instances
    subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'echoear_face'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 1. Start FastWhisperAPI
    print("\n1️⃣ Starting FastWhisperAPI...")
    cmd = "cd FastWhisperAPI && /opt/anaconda3/bin/conda run -n howdy310 --no-capture-output python -u -m uvicorn main:app --host 127.0.0.1 --port 8000"
    fastapi_proc = subprocess.Popen(
        cmd, shell=True, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print("   Waiting for FastAPI to initialize...")
    time.sleep(8)
    print("   ✓ FastWhisperAPI ready")

    # 2. Start EchoEar Face Renderer
    print("\n2️⃣ Starting EchoEar Face Renderer...")
    face_cmd = ["/opt/anaconda3/bin/conda", "run", "-n", "howdy310", "python", "-u", "echoear_face.py"]

    face_proc = subprocess.Popen(
        face_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    time.sleep(2)
    print("   ✓ EchoEar Face ready (UDP port 31337)")

    # 3. Start Voice Assistant with reactive audio
    print("\n3️⃣ Starting Voice Assistant...")
    print("=" * 50)
    print()

    # Set environment variable to enable audio reactivity
    env = os.environ.copy()
    env["HOWDY_AUDIO_REACTIVE"] = "1"

    # Auto-configure library paths for M3 Mac
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        opus_lib_path = "/opt/homebrew/opt/opus/lib"
        if os.path.exists(opus_lib_path):
            current_path = env.get("DYLD_LIBRARY_PATH", "")
            if opus_lib_path not in current_path:
                env["DYLD_LIBRARY_PATH"] = f"{opus_lib_path}:{current_path}"

    # Use shell command to ensure DYLD_LIBRARY_PATH is passed through conda
    voice_cmd = f"cd '{os.getcwd()}' && /opt/anaconda3/bin/conda run -n howdy310 --no-capture-output python -u run_voice_assistant.py"

    voice_proc = subprocess.Popen(voice_cmd, env=env, shell=True)

    print("=" * 50)
    print("✓ All systems running!")
    print("=" * 50)
    print("\n🎨 EchoEar Face Features:")
    print("  • Eyes pulse with speech volume")
    print("  • Eyes narrow on sibilants (s, sh)")
    print("  • Head nods on emphasis")
    print("\n💡 Say 'Hey Howdy' to start a conversation")
    print("=" * 50)

    # Wait for voice assistant to finish
    try:
        voice_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup()
