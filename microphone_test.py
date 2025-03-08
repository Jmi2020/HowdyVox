#!/usr/bin/env python3
"""
Simple microphone test script that records audio and immediately plays it back.
This helps verify that your microphone is working properly.
"""

import os
import sys
import time
import wave
import pyaudio
import speech_recognition as sr
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

def list_microphones():
    """List all available microphones."""
    print(f"\n{Fore.CYAN}Available Microphones:{Fore.RESET}")
    
    mic_list = sr.Microphone.list_microphone_names()
    for i, mic in enumerate(mic_list):
        print(f"{i}: {mic}")
    
    return mic_list

def test_microphone(device_index=None):
    """Test microphone by recording and playing back audio."""
    
    print(f"\n{Fore.YELLOW}Testing Microphone{Fore.RESET}")
    if device_index is not None:
        print(f"Using device index: {device_index}")
    
    # Initialize the recognizer
    r = sr.Recognizer()
    
    # Set very low energy threshold to detect any audio
    r.energy_threshold = 50
    r.dynamic_energy_threshold = False
    
    # Create output directory
    os.makedirs("test_recordings", exist_ok=True)
    output_file = "test_recordings/microphone_test.wav"
    
    # Start recording
    print(f"{Fore.YELLOW}Recording 5 seconds of audio...{Fore.RESET}")
    print("Please speak into your microphone...")
    
    try:
        with sr.Microphone(device_index=device_index) as source:
            # Adjust for ambient noise
            print("Calibrating for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            print(f"Energy threshold after calibration: {r.energy_threshold}")
            
            # Record audio
            print(f"{Fore.GREEN}Recording... (speak now){Fore.RESET}")
            audio = r.listen(source, timeout=None, phrase_time_limit=5)
            print("Recording finished")
            
            # Get audio data
            data_length = len(audio.frame_data)
            print(f"Recorded {data_length} bytes of audio data")
            
            # Save to WAV file
            print(f"Saving to {output_file}...")
            with open(output_file, "wb") as f:
                f.write(audio.get_wav_data())
            
            # Verify the file exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"WAV file created: {file_size} bytes")
                
                # Play the recorded audio
                print(f"\n{Fore.YELLOW}Playing back recorded audio...{Fore.RESET}")
                play_audio(output_file)
                
                print(f"\n{Fore.GREEN}Microphone test completed successfully!{Fore.RESET}")
                return True
            else:
                print(f"{Fore.RED}Failed to create output file{Fore.RESET}")
                return False
    
    except Exception as e:
        print(f"{Fore.RED}Error testing microphone: {str(e)}{Fore.RESET}")
        import traceback
        traceback.print_exc()
        return False

def play_audio(file_path):
    """Play an audio file."""
    chunk = 1024
    
    try:
        # Open the WAV file
        wf = wave.open(file_path, 'rb')
        
        # Create PyAudio object
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        
        # Read data
        data = wf.readframes(chunk)
        
        # Play stream
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        
        # Close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return True
    except Exception as e:
        print(f"{Fore.RED}Error playing audio: {str(e)}{Fore.RESET}")
        return False

def main():
    """Main function to run the microphone test."""
    print(f"{Fore.CYAN}=== Microphone Test Utility ==={Fore.RESET}")
    print("This utility will help you test your microphone setup")
    
    # List available microphones
    mic_list = list_microphones()
    
    # Ask user to select a microphone
    print("\nOptions:")
    print("1. Test default microphone")
    print("2. Select a specific microphone")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        test_microphone()
    elif choice == '2':
        if not mic_list:
            print(f"{Fore.RED}No microphones found{Fore.RESET}")
            return
        
        device_index = input(f"Enter microphone number (0-{len(mic_list)-1}): ").strip()
        try:
            device_index = int(device_index)
            if 0 <= device_index < len(mic_list):
                test_microphone(device_index)
            else:
                print(f"{Fore.RED}Invalid device index{Fore.RESET}")
        except ValueError:
            print(f"{Fore.RED}Invalid input{Fore.RESET}")
    elif choice == '3':
        print("Exiting...")
    else:
        print(f"{Fore.RED}Invalid choice{Fore.RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {str(e)}{Fore.RESET}")
        import traceback
        traceback.print_exc()