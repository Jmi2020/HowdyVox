#!/usr/bin/env python3
# test_audio.py - A simple script to test audio recording functionality

import os
import sys
import time
from colorama import Fore, init
from voice_assistant.audio import record_audio
from voice_assistant.transcription import transcribe_audio
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key

# Initialize colorama
init(autoreset=True)

def test_audio_recording():
    """Test audio recording functionality in isolation."""
    print(f"{Fore.CYAN}Audio Recording Test Utility{Fore.RESET}")
    print("-------------------------------")
    print("This script tests the microphone recording and transcription.")
    print("Press Ctrl+C to exit\n")
    
    # Create a test directory
    os.makedirs("test_audio", exist_ok=True)
    
    # Loop for multiple test recordings
    counter = 1
    while True:
        try:
            # File path for this test
            wav_path = f"test_audio/test_{counter}.wav"
            
            print(f"\n{Fore.YELLOW}Test Recording #{counter}{Fore.RESET}")
            print(f"Recording to {wav_path}")
            
            # Record audio with different energy thresholds to find what works
            energy_threshold = 300  # Start with a very low threshold
            
            # Record audio
            record_audio(
                wav_path,
                timeout=None,        # Wait indefinitely for audio
                phrase_time_limit=10,  # Record up to 10 seconds
                energy_threshold=energy_threshold,
                calibration_duration=1.0
            )
            
            # Verify the file was created
            if not os.path.exists(wav_path):
                print(f"{Fore.RED}Error: No audio file was created!{Fore.RESET}")
                continue
                
            file_size = os.path.getsize(wav_path)
            print(f"Recorded audio file size: {file_size} bytes")
            
            # Optional: Test transcription
            if file_size > 1000:  # Only attempt transcription if file has content
                try:
                    print("\nAttempting transcription...")
                    
                    # Check if FastWhisperAPI is available
                    try:
                        import requests
                        response = requests.get("http://localhost:8000/info", timeout=1.0)
                        if response.status_code == 200:
                            # Try FastWhisperAPI
                            transcription_api_key = get_transcription_api_key()
                            text = transcribe_audio(
                                Config.TRANSCRIPTION_MODEL,
                                transcription_api_key,
                                wav_path,
                                Config.LOCAL_MODEL_PATH
                            )
                            
                            if text:
                                print(f"{Fore.GREEN}Transcription: {text}{Fore.RESET}")
                            else:
                                print(f"{Fore.YELLOW}No transcription result{Fore.RESET}")
                        else:
                            print("FastWhisperAPI not available, skipping transcription")
                    except:
                        print("Could not connect to FastWhisperAPI, skipping transcription")
                        
                except Exception as e:
                    print(f"{Fore.RED}Transcription error: {e}{Fore.RESET}")
            
            # Ask if user wants to continue
            choice = input("\nRecord another sample? (y/n): ").strip().lower()
            if choice != 'y':
                break
                
            counter += 1
            
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Fore.RESET}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    
    print("\nTest completed.")

if __name__ == "__main__":
    test_audio_recording()