#!/usr/bin/env python3
"""
Test and fix FastWhisperAPI configuration
"""

import os
import sys
import subprocess
import requests
import time
import signal
import platform

def is_service_running(url):
    """Check if the FastWhisperAPI service is running"""
    try:
        response = requests.get(f"{url}/info", timeout=3)
        return response.status_code == 200
    except:
        return False

def check_transcribe_endpoint(url):
    """Check if the transcribe endpoint is working"""
    try:
        # Find a sample audio file
        sample_files = [
            os.path.join("voice_samples", "sample1.mp3"),
            os.path.join("test_audio", "sample.wav"),
            os.path.join("test_recordings", "test1.wav")
        ]
        
        audio_file = None
        for file_path in sample_files:
            if os.path.exists(file_path):
                audio_file = file_path
                break
        
        if audio_file is None:
            print("❌ No sample audio file found for testing")
            return False
        
        print(f"Using sample audio file: {audio_file}")
        
        # Test the transcribe endpoint
        with open(audio_file, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_file), f, "audio/wav")}
            response = requests.post(
                f"{url}/transcribe", 
                files=files
            )
        
        if response.status_code == 200:
            print(f"✅ Transcription endpoint working")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Transcription endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing transcribe endpoint: {e}")
        return False

def check_api_files():
    """Check if FastWhisperAPI files are present and correct"""
    api_dir = os.path.join(os.getcwd(), "FastWhisperAPI")
    
    if not os.path.isdir(api_dir):
        print("❌ FastWhisperAPI directory not found")
        return False
    
    required_files = ["main.py", "requirements.txt"]
    for file in required_files:
        file_path = os.path.join(api_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            return False
    
    print("✅ All required FastWhisperAPI files found")
    return True

def restart_api():
    """Restart the FastWhisperAPI service"""
    print("\nRestarting FastWhisperAPI service...")
    
    # Kill any running uvicorn processes for FastWhisperAPI
    if platform.system() == "Windows":
        os.system('taskkill /f /im uvicorn.exe')
    else:
        try:
            pid_cmd = "ps aux | grep 'uvicorn main:app' | grep -v grep | awk '{print $2}'"
            pids = subprocess.check_output(pid_cmd, shell=True).decode().strip().split('\n')
            for pid in pids:
                if pid:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"Terminated process {pid}")
        except:
            pass
    
    # Change to the FastWhisperAPI directory
    api_dir = os.path.join(os.getcwd(), "FastWhisperAPI")
    os.chdir(api_dir)
    
    # Start the server in a new process
    if platform.system() == "Windows":
        subprocess.Popen(
            ["start", "cmd", "/c", "uvicorn", "main:app", "--reload", "--port", "8000"],
            shell=True
        )
    else:
        subprocess.Popen(
            ["uvicorn", "main:app", "--reload", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Give it some time to start
    print("Waiting for service to start...")
    for i in range(10):
        time.sleep(1)
        if is_service_running("http://localhost:8000"):
            print(f"✅ FastWhisperAPI started successfully")
            return True
        print(".", end="", flush=True)
    
    print("\n❌ Failed to start FastWhisperAPI")
    return False

def install_dependencies():
    """Install FastWhisperAPI dependencies"""
    print("\nInstalling FastWhisperAPI dependencies...")
    
    api_dir = os.path.join(os.getcwd(), "FastWhisperAPI")
    req_file = os.path.join(api_dir, "requirements.txt")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def fix_api_endpoints():
    """Fix the FastWhisperAPI endpoints if needed"""
    api_dir = os.path.join(os.getcwd(), "FastWhisperAPI")
    main_py = os.path.join(api_dir, "main.py")
    
    with open(main_py, 'r') as f:
        content = f.read()
    
    # Check if transcribe endpoint exists
    if "async def transcribe(" not in content:
        print("❌ Transcribe endpoint not found in main.py")
        print("This indicates a corrupted or incomplete FastWhisperAPI installation.")
        print("You might need to reinstall FastWhisperAPI.")
        return False
    
    print("✅ Transcribe endpoint found in main.py")
    return True

def main():
    print("\n===== FastWhisperAPI Test and Fix =====\n")
    
    # Check if API files exist
    if not check_api_files():
        print("❌ Critical files missing. Cannot continue.")
        return
    
    # Check if service is running
    api_url = "http://localhost:8000"
    if is_service_running(api_url):
        print("✅ FastWhisperAPI is running")
        
        # Check if transcribe endpoint works
        if check_transcribe_endpoint(api_url):
            print("✅ FastWhisperAPI is fully functional")
        else:
            print("⚠️ FastWhisperAPI is running but the transcribe endpoint is not working")
            
            # Try to fix it
            print("\nAttempting to fix the API...")
            if fix_api_endpoints():
                print("API code appears to be correct.")
                
                # Reinstall dependencies and restart
                install_dependencies()
                if restart_api():
                    # Test again
                    time.sleep(2)
                    if check_transcribe_endpoint(api_url):
                        print("✅ FastWhisperAPI is now fully functional")
                    else:
                        print("❌ Still having issues with the transcribe endpoint")
            else:
                print("❌ Could not fix the API automatically")
    else:
        print("❌ FastWhisperAPI is not running")
        
        # Install dependencies
        install_dependencies()
        
        # Start the API
        if restart_api():
            # Test transcribe endpoint
            time.sleep(2)
            if check_transcribe_endpoint(api_url):
                print("✅ FastWhisperAPI is now fully functional")
            else:
                print("⚠️ FastWhisperAPI is running but the transcribe endpoint is not working")
                
                # Try to fix it
                if fix_api_endpoints():
                    # Restart again
                    restart_api()
                    time.sleep(2)
                    if check_transcribe_endpoint(api_url):
                        print("✅ FastWhisperAPI is now fully functional")
                    else:
                        print("❌ Still having issues with the transcribe endpoint")
                else:
                    print("❌ Could not fix the API automatically")
        else:
            print("❌ Failed to start FastWhisperAPI")
    
    print("\n===== FastWhisperAPI Test and Fix Complete =====")

if __name__ == "__main__":
    main()