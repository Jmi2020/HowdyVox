#!/usr/bin/env python3
"""
Test script for running Porcupine wake word detection in HowdyTTS
"""

import os
import time
import logging
import pyaudio
import struct
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print(f"{Fore.CYAN}=== Testing Porcupine Wake Word Detection ==={Style.RESET_ALL}")
    print("Make sure you're using pvporcupine version 2.2.0 for Apple Silicon compatibility")
    
    # Load environment variables
    load_dotenv()
    
    # Import here to ensure we're using the right version
    import pvporcupine
    import pkg_resources
    
    # Get version using pkg_resources
    pv_version = pkg_resources.get_distribution("pvporcupine").version
    print(f"Using pvporcupine version: {pv_version}")
    
    # Get access key from environment
    access_key = os.getenv("PORCUPINE_ACCESS_KEY")
    if not access_key:
        print(f"{Fore.RED}PORCUPINE_ACCESS_KEY not found in environment variables{Style.RESET_ALL}")
        return 1
    
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
            print("Trying with built-in 'Jarvis' wake word instead...")
            try:
                porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=["jarvis"],
                    sensitivities=[0.7]
                )
                print(f"{Fore.GREEN}Porcupine created successfully with 'jarvis' wake word{Style.RESET_ALL}")
            except Exception as e2:
                print(f"{Fore.RED}Error with built-in wake word: {e2}{Style.RESET_ALL}")
            return 1
    
    print(f"Found model at: {model_path}")
    
    try:
        # Try to create a Porcupine instance
        print(f"{Fore.YELLOW}Creating Porcupine with custom 'Hey Howdy' wake word...{Style.RESET_ALL}")
        porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[model_path],
            sensitivities=[0.7]  # Higher sensitivity for better detection
        )
        
        print(f"{Fore.GREEN}Successfully created Porcupine!{Style.RESET_ALL}")
        print(f"Sample rate: {porcupine.sample_rate}")
        print(f"Frame length: {porcupine.frame_length}")
        
        # Initialize PyAudio for microphone input
        audio = pyaudio.PyAudio()
        
        # Create audio stream
        audio_stream = audio.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        
        print(f"\n{Fore.CYAN}Listening for 'Hey Howdy'... (Press Ctrl+C to exit){Style.RESET_ALL}")
        
        # Main detection loop
        try:
            while True:
                # Read audio frame
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                
                # Convert to the format Porcupine expects
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                
                # Process with Porcupine
                keyword_index = porcupine.process(pcm)
                
                # If wake word detected (keyword_index >= 0)
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
        
        return 0
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        
        # Try with built-in wake word as fallback
        try:
            print(f"\n{Fore.YELLOW}Trying with built-in 'Jarvis' wake word instead...{Style.RESET_ALL}")
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=["jarvis"],
                sensitivities=[0.7]
            )
            
            print(f"{Fore.GREEN}Successfully created Porcupine with 'Jarvis' wake word!{Style.RESET_ALL}")
            print(f"Sample rate: {porcupine.sample_rate}")
            print(f"Frame length: {porcupine.frame_length}")
            
            # Initialize PyAudio for microphone input
            audio = pyaudio.PyAudio()
            
            # Create audio stream
            audio_stream = audio.open(
                rate=porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=porcupine.frame_length
            )
            
            print(f"\n{Fore.CYAN}Listening for 'Jarvis'... (Press Ctrl+C to exit){Style.RESET_ALL}")
            
            # Main detection loop
            try:
                while True:
                    # Read audio frame
                    pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                    
                    # Convert to the format Porcupine expects
                    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
                    
                    # Process with Porcupine
                    keyword_index = porcupine.process(pcm)
                    
                    # If wake word detected (keyword_index >= 0)
                    if keyword_index >= 0:
                        print(f"\n{Fore.GREEN}Wake word 'Jarvis' detected!{Style.RESET_ALL}")
                        # Brief pause to prevent multiple detections
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Stopped by user{Style.RESET_ALL}")
            
            # Clean up resources
            audio_stream.stop_stream()
            audio_stream.close()
            audio.terminate()
            porcupine.delete()
            
            return 0
            
        except Exception as e2:
            print(f"{Fore.RED}Error with built-in wake word: {e2}{Style.RESET_ALL}")
            return 1

if __name__ == "__main__":
    main()
