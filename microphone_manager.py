# voice_assistant/microphone_manager.py

import speech_recognition as sr
import pyaudio
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
import subprocess
import re

class MicrophoneManager:
    """
    Manages microphone identification and mapping for multi-room setup.
    This class helps identify USB microphones by their location and maintains
    persistent mappings between physical microphones and room assignments.
    """
    
    def __init__(self, config_path: str = "microphone_config.json"):
        self.config_path = config_path
        self.mappings = self.load_mappings()
        
    def list_all_microphones(self) -> List[Dict]:
        """
        List all available microphones with detailed information.
        
        Returns:
            List of dictionaries containing microphone information
        """
        # First, get the speech_recognition view of microphones
        sr_mics = sr.Microphone.list_microphone_names()
        
        # Get system profiler data for USB devices (macOS specific)
        usb_info = self._get_usb_audio_devices()
        
        microphones = []
        
        # For each microphone found by speech_recognition
        for index, name in enumerate(sr_mics):
            mic_info = {
                'index': index,
                'name': name,
                'usb_location': None,
                'serial_number': None
            }
            
            # Try to match with USB information
            if "UAC 1.0 Microphone" in name:
                # Find matching USB device by partial name match
                for usb_device in usb_info:
                    if "UAC 1.0 Microphone" in usb_device.get('name', ''):
                        mic_info['usb_location'] = usb_device.get('location_id')
                        mic_info['serial_number'] = usb_device.get('serial_number')
                        break
            
            microphones.append(mic_info)
            
        return microphones
    
    def _get_usb_audio_devices(self) -> List[Dict]:
        """
        Get USB audio device information using system_profiler on macOS.
        
        Returns:
            List of USB audio devices with their location IDs
        """
        try:
            # Run system_profiler to get USB information
            result = subprocess.run(
                ['system_profiler', 'SPUSBDataType', '-json'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logging.warning("Failed to get USB information from system_profiler")
                return []
            
            import json
            data = json.loads(result.stdout)
            
            audio_devices = []
            
            # Parse the USB tree to find audio devices
            def parse_usb_tree(items, parent_location=""):
                for item in items:
                    # Check if this is an audio device
                    if 'UAC 1.0 Microphone' in item.get('_name', ''):
                        device_info = {
                            'name': item.get('_name'),
                            'location_id': item.get('location_id', ''),
                            'serial_number': item.get('serial_num', 'Unknown')
                        }
                        audio_devices.append(device_info)
                    
                    # Recursively check items
                    if '_items' in item:
                        parse_usb_tree(item['_items'], item.get('location_id', ''))
            
            # Start parsing from SPUSBDataType
            if 'SPUSBDataType' in data:
                parse_usb_tree(data['SPUSBDataType'])
            
            return audio_devices
            
        except Exception as e:
            logging.error(f"Error getting USB information: {e}")
            return []
    
    def create_room_mapping(self, room_name: str, microphone_index: int) -> bool:
        """
        Create a mapping between a room name and a microphone index.
        
        Args:
            room_name: Name of the room (e.g., "Living Room", "Bedroom", "Kitchen")
            microphone_index: The speech_recognition microphone index
            
        Returns:
            bool: True if successful
        """
        # Get current microphone list
        mics = self.list_all_microphones()
        
        # Find the microphone
        mic_info = None
        for mic in mics:
            if mic['index'] == microphone_index:
                mic_info = mic
                break
                
        if not mic_info:
            logging.error(f"Microphone index {microphone_index} not found")
            return False
        
        # Store the mapping
        self.mappings[room_name] = {
            'microphone_index': microphone_index,
            'microphone_name': mic_info['name'],
            'usb_location': mic_info.get('usb_location'),
            'last_seen': True
        }
        
        self.save_mappings()
        logging.info(f"Mapped '{room_name}' to microphone {microphone_index}: {mic_info['name']}")
        return True
    
    def get_microphone_for_room(self, room_name: str) -> Optional[int]:
        """
        Get the microphone index for a specific room.
        
        Args:
            room_name: Name of the room
            
        Returns:
            Microphone index or None if not found
        """
        if room_name not in self.mappings:
            logging.error(f"No mapping found for room '{room_name}'")
            return None
            
        # Verify the microphone still exists at that index
        mic_index = self.mappings[room_name]['microphone_index']
        current_mics = sr.Microphone.list_microphone_names()
        
        if mic_index >= len(current_mics):
            logging.error(f"Microphone index {mic_index} no longer valid")
            return None
            
        # Verify it's still the same microphone (by name)
        expected_name = self.mappings[room_name]['microphone_name']
        actual_name = current_mics[mic_index]
        
        if expected_name != actual_name:
            logging.warning(f"Microphone at index {mic_index} has changed. Expected '{expected_name}', got '{actual_name}'")
            # Try to find it at a different index
            for i, name in enumerate(current_mics):
                if name == expected_name:
                    logging.info(f"Found microphone at new index {i}")
                    # Update the mapping
                    self.mappings[room_name]['microphone_index'] = i
                    self.save_mappings()
                    return i
            
            logging.error(f"Could not find microphone '{expected_name}'")
            return None
            
        return mic_index
    
    def save_mappings(self):
        """Save mappings to configuration file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.mappings, f, indent=2)
            
    def load_mappings(self) -> Dict:
        """Load mappings from configuration file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading mappings: {e}")
        return {}