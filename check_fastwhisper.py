#!/usr/bin/env python3
"""
Test script to check if FastWhisperAPI is running and working correctly
"""

import requests
import sys
import os
import time
import subprocess
import platform

def is_service_running(url):
    """Check if a service is running at the specified URL"""
    try:
        response = requests.get(f"{url}/info", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    api_url = "http://localhost:8000"
    print("\n===== FastWhisperAPI Test =====\n")
    
    # Check if service is running
    print(f"Checking if FastWhisperAPI is running at {api_url}...")
    if is_service_running(api_url):
        print("✅ FastWhisperAPI is running")
    else:
        print("❌ FastWhisperAPI is not running")
        print("\nWould you like to start it? (y/n)")
        choice = input().lower()
        
        if choice == 'y':
            print("\nStarting FastWhisperAPI...")
            
            # Change directory to FastWhisperAPI
            os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FastWhisperAPI"))
            
            # Start the service in a new process
            if platform.system() == "Windows":
                subprocess.Popen(["start", "cmd", "/c", "uvicorn", "main:app", "--reload"], shell=True)
            else:
                subprocess.Popen(["uvicorn", "main:app", "--reload"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
            
            # Wait for service to start
            print("Waiting for service to start...")
            for _ in range(10):  # Try for 10 seconds
                time.sleep(1)
                if is_service_running(api_url):
                    print("✅ FastWhisperAPI has been started successfully")
                    break
            else:
                print("❌ Failed to start FastWhisperAPI")
                return
        else:
            print("Skipping FastWhisperAPI startup")
            return
    
    # Test the API
    print("\nTesting FastWhisperAPI transcription...")
    
    # Check for a sample audio file
    sample_files = [
        os.path.join("test_audio", "sample.wav"),
        os.path.join("voice_samples", "sample1.mp3"),
        os.path.join("test_recordings", "test1.wav")
    ]
    
    audio_file = None
    for file_path in sample_files:
        if os.path.exists(file_path):
            audio_file = file_path
            break
    
    if audio_file is None:
        print("❌ No sample audio file found for testing")
        return
    
    print(f"Using sample audio file: {audio_file}")
    
    # Send a request to the transcription API
    try:
        with open(audio_file, "rb") as f:
            files = {"audio_file": (os.path.basename(audio_file), f, "audio/wav")}
            response = requests.post(
                f"{api_url}/transcribe",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Transcription successful:")
            print(f"Text: {result.get('text', 'No text in response')}")
            print(f"Duration: {result.get('duration', 'Unknown')} seconds")
            print(f"Processing Time: {result.get('processing_time', 'Unknown')} seconds")
        else:
            print(f"❌ Transcription failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error during transcription request: {e}")
    
    print("\n===== FastWhisperAPI Test Complete =====")

if __name__ == "__main__":
    main()