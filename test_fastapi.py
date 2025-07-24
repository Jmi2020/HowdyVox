#!/usr/bin/env python3

import subprocess
import time
import sys

def test_fastapi_startup():
    """Test FastAPI startup to diagnose issues"""
    print("🔍 Testing FastAPI startup...")
    
    try:
        # Start FastAPI with conda
        process = subprocess.Popen(
            ["conda", "run", "-n", "howdy310", "fastapi", "run", "main.py", "--host", "127.0.0.1", "--port", "8000"],
            cwd="FastWhisperAPI",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("FastAPI process started, monitoring output...")
        start_time = time.time()
        
        while time.time() - start_time < 60:  # 60 second timeout
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    print(f"❌ Process ended with return code: {process.returncode}")
                    break
                continue
            
            print(f"[FastAPI] {line.strip()}")
            
            # Check for ready signals
            if any(pattern in line for pattern in [
                "Application startup complete",
                "Server started at http://",
                "server   Server started at"
            ]):
                print("✅ FastAPI is ready!")
                process.terminate()
                return True
        
        print("⏰ Timeout reached")
        process.terminate()
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_fastapi_startup()
    print(f"\n{'✅ Success' if success else '❌ Failed'}")