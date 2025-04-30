#!/usr/bin/env python3
"""
Simple test script for Porcupine wake word detection v2.2.0 on Apple Silicon
"""

import os
import platform
import struct
import time
import pyaudio
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def main():
    print(f"{Fore.CYAN}=== Testing Porcupine Wake Word Detection v2.2.0 ==={Style.RESET_ALL}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    print(f"Apple Silicon: {is_apple_silicon}")
    
    # Load environment variables
    load_dotenv()
    
    # Import pvporcupine
    import pvporcupine
    
    # Get access key from environment
    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    if not access_key:
        print(f"{Fore.RED}PORCUPINE_ACCESS_KEY not found in environment variables{Style.RESET_ALL}")
        return
    
    # Clean up access key
    access_key = access_key.strip().strip('"').strip("'")
    print(f"Using access key: {access_key[:5]}...{access_key[-5:]}")
    
    # Path to custom wake word model
    model_path = os.path.join("models", "Hey-Howdy_en_mac_v3_0_0.ppn")
    
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Custom model not found at: {model_path}{Style.RESET_ALL}")
        print("Falling back to built-in wake word 'jarvis'")
        try:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=["jarvis"],
                sensitivities=[0.7]
            )
            print(f"{Fore.GREEN}Porcupine created successfully with 'jarvis' wake word{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            return
    else:
        print(f"Found model at: {model_path}")
        try:
            porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path],
                sensitivities=[0.7]
            )
            print(f"{Fore.GREEN}Porcupine created successfully with custom wake word{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error with custom model: {e}{Style.RESET_ALL}")
            print("Falling back to built-in wake word 'jarvis'")
            try:
                porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=["jarvis"],
                    sensitivities=[0.7]
                )
                print(f"{Fore.GREEN}Porcupine created successfully with 'jarvis' wake word{Style.RESET_ALL}")
            except Exception as e2:
                print(f"{Fore.RED}Error with built-in wake word: {e2}{Style.RESET_ALL}")
                return
    
    print(f"Sample rate: {porcupine.sample_rate}")
    print(f"Frame length: {porcupine.frame_length}")
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Create audio stream
    audio_stream = audio.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    
    print(f"\n{Fore.CYAN}Listening for wake word... (Press Ctrl+C to exit){Style.RESET_ALL}")
    
    # Main detection loop
    try:
        while True:
            # Read audio frame
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            
            # Convert to the format Porcupine expects
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            # Process with Porcupine
            keyword_index = porcupine.process(pcm)
            
            # If wake word detected
            if keyword_index >= 0:
                print(f"\n{Fore.GREEN}Wake word detected!{Style.RESET_ALL}")
                # Brief pause to prevent multiple detections
                time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Stopped by user{Style.RESET_ALL}")
    
    # Clean up resources
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()
    porcupine.delete()

if __name__ == "__main__":
    main()
