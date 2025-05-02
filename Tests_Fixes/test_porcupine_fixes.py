#!/usr/bin/env python3
"""
Test script to validate Porcupine wake word detection fixes.
This script tests the improved error handling and platform compatibility checks.
"""

import os
import sys
import logging
from colorama import Fore, init
import pvporcupine

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voice_assistant.wake_word import WakeWordDetector

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def on_wake_word():
    """Callback for when wake word is detected"""
    print(f"{Fore.GREEN}Wake word detected!{Fore.RESET}")

def main():
    """Test Porcupine wake word detection"""
    print(f"{Fore.BLUE}Testing Porcupine wake word detection with fixes...{Fore.RESET}")
    print(f"{Fore.BLUE}Porcupine version: {pvporcupine.__version__}{Fore.RESET}")
    
    try:
        # Attempt to create a WakeWordDetector instance
        detector = WakeWordDetector(on_wake_word, sensitivity=0.5)
        print(f"{Fore.GREEN}Successfully created detector instance!{Fore.RESET}")
        
        # Start wake word detection
        detector.start()
        print(f"{Fore.CYAN}Starting wake word detection, say 'Hey Howdy' (or alternative wake word){Fore.RESET}")
        print("Press Ctrl+C to exit...")
        
        # Run for a few seconds
        try:
            # Keep the main thread alive
            while True:
                pass
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}Stopping wake word detection...{Fore.RESET}")
        finally:
            # Clean up
            detector.stop()
            
    except Exception as e:
        print(f"{Fore.RED}Error testing wake word detection: {e}{Fore.RESET}")
        
        # Check for specific error signatures
        error_str = str(e)
        if "00000136" in error_str:
            print(f"{Fore.YELLOW}This appears to be a platform compatibility issue (error 00000136).{Fore.RESET}")
            print(f"{Fore.YELLOW}Your wake word model is not compatible with Apple Silicon.{Fore.RESET}")
            print(f"{Fore.YELLOW}Solution: Generate a new wake word model specifically for macOS arm64.{Fore.RESET}")
        elif "access key" in error_str.lower():
            print(f"{Fore.YELLOW}This appears to be an access key issue.{Fore.RESET}")
            print(f"{Fore.YELLOW}Make sure your PORCUPINE_ACCESS_KEY environment variable is set correctly.{Fore.RESET}")
        
    print(f"{Fore.BLUE}Test completed.{Fore.RESET}")

if __name__ == "__main__":
    main()