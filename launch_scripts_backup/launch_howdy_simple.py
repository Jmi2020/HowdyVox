#!/usr/bin/env python3
"""
Simple launcher for HowdyTTS that starts FastWhisperAPI and Voice Assistant.
Shows all output for debugging and monitoring.
"""

import subprocess
import time
import os
import signal
import sys
import threading
import select
from pathlib import Path

class SimpleLauncher:
    def __init__(self):
        self.processes = []
    
    def cleanup(self, sig=None, frame=None):
        print("\n\nShutting down...")
        for proc in self.processes:
            try:
                # First try graceful termination
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                try:
                    proc.kill()
                    proc.wait()
                except:
                    pass
            except Exception as e:
                print(f"Warning during cleanup: {e}")
        print("Cleanup complete.")
        sys.exit(0)
    
    def check_port(self, port=8000):
        """Check if port is available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except:
            return False
    
    def run(self):
        # Set up signal handler
        signal.signal(signal.SIGINT, self.cleanup)
        
        print("🤠 HowdyTTS Simple Launcher")
        print("=" * 50)
        
        # Check if we're in the right directory
        if not Path("FastWhisperAPI").exists():
            print("❌ FastWhisperAPI directory not found")
            return
        
        # Check if FastAPI is installed in the conda environment
        print("Checking FastAPI installation...")
        check_cmd = [
            "/opt/anaconda3/bin/conda", "run", "-n", "howdy310",
            "python", "-c", "import fastapi; print('FastAPI version:', fastapi.__version__)"
        ]
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
            else:
                print("❌ FastAPI not found in howdy310 environment")
                print("Error:", result.stderr)
                print("\nTry running: conda activate howdy310 && pip install fastapi uvicorn")
                return
        except Exception as e:
            print(f"⚠️  Could not check FastAPI: {e}")
        
        # Kill any existing process on port 8000
        if not self.check_port(8000):
            print("Port 8000 is busy, killing existing process...")
            try:
                # Get PIDs using port 8000
                result = subprocess.run(['lsof', '-ti', ':8000'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        subprocess.run(['kill', '-9', pid])
                    print(f"Killed process(es): {', '.join(pids)}")
                    time.sleep(2)
            except Exception as e:
                print(f"Warning: Could not kill process on port 8000: {e}")
        
        # Start FastWhisperAPI
        print("\n1️⃣ Starting FastWhisperAPI...")
        
        # Use the full conda path to ensure it works
        conda_path = "/opt/anaconda3/bin/conda"
        
        # Build command as list for better reliability
        fastapi_cmd = [
            conda_path, "run", "-n", "howdy310",
            "fastapi", "run", "main.py"
        ]
        
        # Use PYTHONUNBUFFERED to ensure real-time output
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        fastapi_proc = subprocess.Popen(
            fastapi_cmd,
            cwd="FastWhisperAPI",  # Change directory properly
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Separate stderr to catch errors
            universal_newlines=True,
            bufsize=0,  # Unbuffered
            env=env
        )
        self.processes.append(fastapi_proc)
        
        # Monitor FastAPI startup with threading to prevent blocking
        import threading
        
        def monitor_output(proc, prefix, show_raw=False):
            """Monitor process output in a separate thread"""
            while True:
                try:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if show_raw:
                        # For HowdyTTS, show output without prefix for cleaner display
                        print(line.rstrip(), flush=True)
                    else:
                        print(f"   [{prefix}] {line.rstrip()}", flush=True)
                    if "Application startup complete" in line or "Uvicorn running on" in line:
                        print(f"   ✅ {prefix} is ready!", flush=True)
                except Exception as e:
                    print(f"   [{prefix}] Error reading output: {e}", flush=True)
                    break
        
        def monitor_error(proc, prefix, show_raw=False):
            """Monitor process errors in a separate thread"""
            while True:
                try:
                    line = proc.stderr.readline()
                    if not line:
                        break
                    if show_raw:
                        # For HowdyTTS, show stderr as normal output since it contains logs
                        print(line.rstrip(), flush=True)
                    else:
                        # FastAPI logs go to stderr by default, so not all stderr is errors
                        if "INFO" in line or "WARNING" in line:
                            print(f"   [{prefix}] {line.rstrip()}", flush=True)
                        else:
                            print(f"   [{prefix} ERROR] {line.rstrip()}", flush=True)
                except Exception as e:
                    print(f"   [{prefix}] Error reading stderr: {e}", flush=True)
                    break
        
        # Start monitoring threads for FastAPI
        stdout_thread = threading.Thread(target=monitor_output, args=(fastapi_proc, "FastAPI", False))
        stderr_thread = threading.Thread(target=monitor_error, args=(fastapi_proc, "FastAPI", False))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Give FastAPI time to start without blocking
        print("   Waiting for FastAPI to initialize...")
        time.sleep(8)  # Give it more time to load models
        
        # Check if process is still running
        if fastapi_proc.poll() is not None:
            print("   ❌ FastAPI failed to start!")
            return
        
        # Small delay
        time.sleep(2)
        
        # Start Voice Assistant
        print("\n2️⃣ Starting Voice Assistant...")
        
        # Build command as list
        # Note: run_voice_assistant.py defaults to local microphone when no args provided
        va_cmd = [
            conda_path, "run", "-n", "howdy310",
            "python", "-u", "run_voice_assistant.py"  # -u for unbuffered Python output
        ]
        
        va_proc = subprocess.Popen(
            va_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=0,  # Unbuffered
            env=env
        )
        self.processes.append(va_proc)
        
        # Monitor Voice Assistant output in threads too
        # Use show_raw=True to display HowdyTTS output without prefixes
        va_stdout_thread = threading.Thread(target=monitor_output, args=(va_proc, "HowdyTTS", True))
        va_stderr_thread = threading.Thread(target=monitor_error, args=(va_proc, "HowdyTTS", True))
        va_stdout_thread.daemon = True
        va_stderr_thread.daemon = True
        va_stdout_thread.start()
        va_stderr_thread.start()
        
        print("\n✅ Both processes started!")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        print("\n--- HowdyTTS Voice Assistant Output ---\n")
        
        # Continue monitoring in main thread
        try:
            while True:
                # Check if processes are still running
                if fastapi_proc.poll() is not None:
                    print("❌ FastAPI process ended")
                    # Try to get any remaining output
                    remaining = fastapi_proc.stderr.read()
                    if remaining:
                        print(f"FastAPI final error output: {remaining}")
                    break
                if va_proc.poll() is not None:
                    print("❌ Voice Assistant process ended")
                    break
                    
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            self.cleanup()

if __name__ == "__main__":
    launcher = SimpleLauncher()
    launcher.run()