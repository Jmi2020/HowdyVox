#!/usr/bin/env python3
"""
Direct launcher that shows all output in real-time
"""

import subprocess
import sys
import os
import signal
import threading
import time

class DirectLauncher:
    def __init__(self):
        self.processes = []
        signal.signal(signal.SIGINT, self.cleanup)
    
    def cleanup(self, sig=None, frame=None):
        print("\n\nShutting down...")
        for proc in self.processes:
            try:
                proc.terminate()
                time.sleep(1)
                if proc.poll() is None:
                    proc.kill()
            except:
                pass
        sys.exit(0)
    
    def run_command(self, cmd, cwd=None, prefix=""):
        """Run command and stream output in real-time"""
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            universal_newlines=True,
            bufsize=0,
            env=env
        )
        
        self.processes.append(proc)
        
        # Stream output in a thread
        def stream_output():
            for line in iter(proc.stdout.readline, ''):
                if line:
                    if prefix:
                        print(f"[{prefix}] {line.rstrip()}")
                    else:
                        print(line.rstrip())
                    sys.stdout.flush()
        
        thread = threading.Thread(target=stream_output)
        thread.daemon = True
        thread.start()
        
        return proc
    
    def run(self):
        print("🤠 HowdyVox Direct Launcher")
        print("=" * 50)
        
        # Start FastAPI
        print("\n1️⃣ Starting FastWhisperAPI...")
        fastapi_proc = self.run_command(
            ["/opt/anaconda3/bin/conda", "run", "-n", "howdy310", 
             "python", "-u", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd="FastWhisperAPI",
            prefix="FastAPI"
        )
        
        # Wait a bit for FastAPI to start
        print("   Waiting for FastAPI to initialize...")
        time.sleep(8)
        
        # Start Voice Assistant
        print("\n2️⃣ Starting Voice Assistant...")
        print("=" * 50)
        print()
        
        # Run voice assistant without prefix for cleaner output
        va_proc = self.run_command(
            ["/opt/anaconda3/bin/conda", "run", "-n", "howdy310", 
             "python", "-u", "run_voice_assistant.py"],
            prefix=""
        )
        
        print("\n✅ Both processes started!")
        print("Press Ctrl+C to stop")
        print("=" * 50 + "\n")
        
        # Wait for processes
        try:
            while True:
                if fastapi_proc.poll() is not None:
                    print("\n❌ FastAPI process ended")
                    break
                if va_proc.poll() is not None:
                    print("\n❌ Voice Assistant process ended")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    launcher = DirectLauncher()
    launcher.run()