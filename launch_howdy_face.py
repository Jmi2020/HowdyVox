#!/usr/bin/env python3
"""
HowdyVox Launcher with Audio-Reactive Face Options
Supports both EchoEar (rendered) and GIF-based (pre-rendered) faces
"""

import subprocess
import os
import time
import signal
import sys
import argparse

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
    subprocess.run("pkill -f 'gif_reactive_face'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)

    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


def main():
    """Main launcher function"""
    global fastapi_proc, voice_proc, face_proc

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch HowdyVox with audio-reactive face")
    parser.add_argument(
        "--face",
        choices=["gif", "echoear", "none"],
        default="gif",
        help="Face renderer type: gif (pre-rendered), echoear (dynamic), or none"
    )
    parser.add_argument(
        "--conda-env",
        default="howdy310",
        help="Conda environment name (default: howdy310)"
    )
    args = parser.parse_args()

    print("🤠 HowdyVox Audio-Reactive Face Launcher")
    print("=" * 50)
    print(f"Face mode: {args.face.upper()}")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Kill any existing instances
    subprocess.run("lsof -ti:8000 | xargs kill -9 2>/dev/null", shell=True)
    subprocess.run("pkill -f 'echoear_face'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("pkill -f 'gif_reactive_face'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 1. Start FastWhisperAPI
    print("\n1️⃣ Starting FastWhisperAPI...")
    cmd = f"cd FastWhisperAPI && /opt/anaconda3/bin/conda run -n {args.conda_env} --no-capture-output python -u -m uvicorn main:app --host 127.0.0.1 --port 8000"
    fastapi_proc = subprocess.Popen(
        cmd, shell=True, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print("   Waiting for FastAPI to initialize...")
    time.sleep(8)
    print("   ✓ FastWhisperAPI ready")

    # 2. Start Face Renderer (if requested)
    if args.face != "none":
        if args.face == "gif":
            print("\n2️⃣ Starting GIF-Based Audio-Reactive Face...")
            face_script = "gif_reactive_face.py"

            # Check if GIF directory exists
            if not os.path.exists("faceStates"):
                print("   ⚠️  WARNING: faceStates/ directory not found!")
                print("   Face renderer will show error. Place GIF files in faceStates/")

        else:  # echoear
            print("\n2️⃣ Starting EchoEar Face Renderer...")
            face_script = "echoear_face.py"

        face_cmd = ["/opt/anaconda3/bin/conda", "run", "-n", args.conda_env, "python", "-u", face_script]

        face_proc = subprocess.Popen(
            face_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        time.sleep(2)
        print(f"   ✓ {args.face.upper()} Face ready (UDP port 31337)")

    # 3. Start Voice Assistant with reactive audio
    print("\n3️⃣ Starting Voice Assistant...")
    print("=" * 50)
    print()

    # Set environment variable to enable audio reactivity
    env = os.environ.copy()
    env["HOWDY_AUDIO_REACTIVE"] = "1" if args.face != "none" else "0"

    voice_cmd = [
        "/opt/anaconda3/bin/conda",
        "run",
        "-n",
        args.conda_env,
        "--no-capture-output",
        "python",
        "-u",
        "run_voice_assistant.py",
    ]

    voice_proc = subprocess.Popen(voice_cmd, env=env)

    print("=" * 50)
    print("✓ All systems running!")
    print("=" * 50)

    if args.face == "gif":
        print("\n🎨 GIF Audio-Reactive Face Features:")
        print("  • Playback speed adjusts with speech volume")
        print("  • Peak detection triggers speedups")
        print("  • Smooth looping animations")
    elif args.face == "echoear":
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
