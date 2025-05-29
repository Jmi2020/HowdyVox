# setup_microphones.py

#!/usr/bin/env python3
"""
Microphone Setup Wizard for Multi-Room HowdyTTS

This wizard helps you assign specific USB microphones to different rooms.
"""

import sys
import time
import pyaudio
import wave
from colorama import Fore, init
from voice_assistant.microphone_manager import MicrophoneManager
import speech_recognition as sr

init(autoreset=True)

def test_microphone(index, duration=3):
    """
    Record a short audio sample from a specific microphone.
    
    Args:
        index: Microphone index
        duration: Recording duration in seconds
    """
    print(f"\n{Fore.YELLOW}Recording from microphone {index} for {duration} seconds...{Fore.RESET}")
    print(f"{Fore.CYAN}Please speak into the microphone to identify it!{Fore.RESET}")
    
    r = sr.Recognizer()
    
    try:
        with sr.Microphone(device_index=index) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source, duration=0.5)
            
            # Record audio
            audio = r.listen(source, timeout=duration, phrase_time_limit=duration)
            
            print(f"{Fore.GREEN}Recording complete!{Fore.RESET}")
            
            # Try to transcribe to verify it's working
            try:
                text = r.recognize_google(audio)
                print(f"{Fore.GREEN}Heard: '{text}'{Fore.RESET}")
            except:
                print(f"{Fore.YELLOW}Could not transcribe, but audio was recorded{Fore.RESET}")
                
            return True
            
    except Exception as e:
        print(f"{Fore.RED}Error testing microphone: {e}{Fore.RESET}")
        return False

def setup_wizard():
    """Run the microphone setup wizard."""
    print(f"\n{Fore.CYAN}=== HowdyTTS Multi-Room Microphone Setup ==={Fore.RESET}")
    print(f"\nThis wizard will help you assign USB microphones to different rooms.")
    
    # Create microphone manager
    manager = MicrophoneManager()
    
    # List all microphones
    print(f"\n{Fore.YELLOW}Detecting microphones...{Fore.RESET}")
    microphones = manager.list_all_microphones()
    
    # Filter for UAC 1.0 Microphones
    usb_mics = [m for m in microphones if "UAC 1.0 Microphone" in m['name']]
    
    if len(usb_mics) < 3:
        print(f"{Fore.RED}Warning: Found only {len(usb_mics)} UAC 1.0 Microphone(s).{Fore.RESET}")
        print(f"Expected 3 for a complete setup.")
    
    print(f"\n{Fore.GREEN}Found {len(usb_mics)} UAC 1.0 Microphone(s):{Fore.RESET}")
    for mic in usb_mics:
        print(f"  Index {mic['index']}: {mic['name']}")
        if mic['usb_location']:
            print(f"    USB Location: {mic['usb_location']}")
    
    # Room setup
    rooms = []
    print(f"\n{Fore.CYAN}Let's set up your rooms:{Fore.RESET}")
    
    while True:
        room_name = input(f"\nEnter room name (or 'done' to finish): ").strip()
        if room_name.lower() == 'done':
            break
            
        if not room_name:
            continue
            
        rooms.append(room_name)
        
        if len(rooms) >= 3:
            print(f"{Fore.YELLOW}You've set up 3 rooms. Add more? (y/n): {Fore.RESET}", end='')
            if input().lower() != 'y':
                break
    
    if not rooms:
        print(f"{Fore.RED}No rooms configured. Exiting.{Fore.RESET}")
        return
    
    print(f"\n{Fore.GREEN}Rooms to configure: {', '.join(rooms)}{Fore.RESET}")
    
    # Assign microphones to rooms
    print(f"\n{Fore.CYAN}Now let's assign microphones to rooms.{Fore.RESET}")
    print(f"I'll test each microphone so you can identify which is which.")
    
    used_indices = set()
    
    for room in rooms:
        print(f"\n{Fore.YELLOW}=== Setting up: {room} ==={Fore.RESET}")
        
        while True:
            # Show available microphones
            print(f"\nAvailable microphones:")
            for mic in usb_mics:
                if mic['index'] not in used_indices:
                    print(f"  {mic['index']}: {mic['name']}")
            
            try:
                mic_index = int(input(f"\nEnter microphone index for {room}: "))
                
                if mic_index not in [m['index'] for m in usb_mics]:
                    print(f"{Fore.RED}Invalid microphone index{Fore.RESET}")
                    continue
                    
                if mic_index in used_indices:
                    print(f"{Fore.RED}This microphone is already assigned{Fore.RESET}")
                    continue
                
                # Test the microphone
                if test_microphone(mic_index):
                    confirm = input(f"\nAssign this microphone to {room}? (y/n): ").lower()
                    if confirm == 'y':
                        manager.create_room_mapping(room, mic_index)
                        used_indices.add(mic_index)
                        print(f"{Fore.GREEN}✓ Microphone assigned to {room}{Fore.RESET}")
                        break
                    
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number{Fore.RESET}")
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Setup cancelled{Fore.RESET}")
                return
    
    print(f"\n{Fore.GREEN}=== Setup Complete ==={Fore.RESET}")
    print(f"\nMicrophone assignments:")
    for room in rooms:
        mic_index = manager.get_microphone_for_room(room)
        if mic_index is not None:
            print(f"  {room}: Microphone {mic_index}")
    
    print(f"\n{Fore.CYAN}You can now launch HowdyTTS with:{Fore.RESET}")
    for room in rooms:
        print(f"  python run_voice_assistant.py --room \"{room}\"")

if __name__ == "__main__":
    setup_wizard()