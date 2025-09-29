#!/usr/bin/env python3

import logging
import threading
import time
from typing import Optional, Dict, Any, Callable
from enum import Enum

from .audio import record_audio as local_record_audio
try:
    from .network_audio_source import NetworkAudioSource
    WIRELESS_AVAILABLE = True
except ImportError:
    WIRELESS_AVAILABLE = False
    logging.warning("Wireless audio support not available")

class AudioSourceType(Enum):
    LOCAL = "local"
    WIRELESS = "wireless"

class AudioSourceManager:
    """
    Lightweight audio source manager that handles switching between local and wireless
    microphones with minimal overhead. Only initializes wireless components when needed.
    """
    
    def __init__(self, initial_source: AudioSourceType = AudioSourceType.LOCAL, target_room: Optional[str] = None):
        self.current_source = initial_source
        self.target_room = target_room
        
        # Audio sources (lazy initialization)
        self._network_source: Optional[NetworkAudioSource] = None
        self._network_initialized = False
        self._network_failed = False
        
        # Current record function
        self._current_record_func = local_record_audio
        
        # Statistics
        self.stats = {
            'source_switches': 0,
            'wireless_initializations': 0,
            'local_recordings': 0,
            'wireless_recordings': 0
        }
        
        # Callbacks
        self.source_changed_callback: Optional[Callable[[AudioSourceType, bool], None]] = None
        
        logging.info(f"AudioSourceManager initialized with {initial_source.value} source")

        if self.current_source == AudioSourceType.WIRELESS and WIRELESS_AVAILABLE:
            if not self.switch_to_wireless(target_room):
                logging.warning("Failed to activate wireless source during initialization; falling back to local")
                self.current_source = AudioSourceType.LOCAL
                self._current_record_func = local_record_audio
    
    def set_source_changed_callback(self, callback: Callable[[AudioSourceType, bool], None]):
        """
        Set callback for source change events.
        Callback receives (new_source_type, success) parameters.
        """
        self.source_changed_callback = callback
    
    def get_current_source(self) -> AudioSourceType:
        """Get the currently active audio source type."""
        return self.current_source
    
    def get_current_record_function(self):
        """Get the current record function (low overhead access)."""
        return self._current_record_func
    
    def switch_to_local(self) -> bool:
        """Switch to local microphone (always available, minimal overhead)."""
        if self.current_source == AudioSourceType.LOCAL:
            return True
        
        self.current_source = AudioSourceType.LOCAL
        self._current_record_func = local_record_audio
        self.stats['source_switches'] += 1
        
        logging.info("Switched to local microphone")
        
        if self.source_changed_callback:
            self.source_changed_callback(AudioSourceType.LOCAL, True)
        
        return True
    
    def switch_to_wireless(self, target_room: Optional[str] = None) -> bool:
        """
        Switch to wireless microphone with lazy initialization.
        Only initializes wireless components when first needed.
        """
        if not WIRELESS_AVAILABLE:
            logging.error("Wireless audio support not available")
            return False
        
        # Use provided room or fall back to instance room
        room = target_room or self.target_room
        
        # Check if we need to initialize or reinitialize wireless
        need_init = (
            not self._network_initialized or 
            self._network_failed or
            (self._network_source and self._network_source.target_room != room)
        )
        
        if need_init:
            success = self._initialize_wireless(room)
            if not success:
                return False
        
        # Switch to wireless
        if self._network_source:
            if self.current_source != AudioSourceType.WIRELESS:
                self.current_source = AudioSourceType.WIRELESS
                self._current_record_func = self._network_source.record_audio
                self.stats['source_switches'] += 1
                self.target_room = room
                
                logging.info(f"Switched to wireless microphone (room: {room or 'auto'})")
                
                if self.source_changed_callback:
                    self.source_changed_callback(AudioSourceType.WIRELESS, True)
            else:
                logging.info(f"Already using wireless microphone (room: {room or 'auto'})")
            
            return True
        
        return False
    
    def _initialize_wireless(self, room: Optional[str] = None) -> bool:
        """Initialize wireless audio source (called only when needed)."""
        if self._network_failed:
            # Reset failed state for retry
            self._network_failed = False
        
        try:
            # Clean up existing network source if different room
            if self._network_source and self._network_source.target_room != room:
                self._network_source.stop()
                self._network_source = None
                self._network_initialized = False
            
            # Create new network source if needed
            if not self._network_source:
                logging.info(f"Initializing wireless audio system (room: {room or 'auto'})...")
                self._network_source = NetworkAudioSource(target_room=room)
                self.stats['wireless_initializations'] += 1
            
            # Start if not already initialized
            if not self._network_initialized:
                if self._network_source.start():
                    self._network_initialized = True
                    
                    # Wait for ESP32-P4 devices to boot and send discovery packets
                    # ESP32-P4 typically takes 3-5 seconds to fully boot and start discovery
                    time.sleep(5.0)
                    
                    # Check if devices are available
                    devices = self._network_source.get_available_devices()
                    if devices:
                        logging.info(f"Wireless initialization successful ({len(devices)} device(s))")
                        return True
                    else:
                        logging.warning("Wireless initialized but no devices found")
                        # Don't mark as failed - devices might appear later
                        return True
                else:
                    logging.error("Failed to start wireless audio system")
                    self._network_failed = True
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Wireless initialization error: {e}")
            self._network_failed = True
            return False
    
    def toggle_source(self) -> bool:
        """Toggle between local and wireless sources."""
        if self.current_source == AudioSourceType.LOCAL:
            return self.switch_to_wireless()
        else:
            return self.switch_to_local()
    
    def auto_select_source(self) -> AudioSourceType:
        """
        Auto-select best available source.
        Tries wireless first, falls back to local.
        """
        if WIRELESS_AVAILABLE:
            if self.switch_to_wireless():
                return AudioSourceType.WIRELESS
        
        # Fallback to local
        self.switch_to_local()
        return AudioSourceType.LOCAL
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about current audio source."""
        info = {
            'current_source': self.current_source.value,
            'target_room': self.target_room,
            'wireless_available': WIRELESS_AVAILABLE,
            'wireless_initialized': self._network_initialized,
            'wireless_failed': self._network_failed
        }
        
        if self._network_source and self._network_initialized:
            devices = self._network_source.get_available_devices()
            info.update({
                'wireless_devices': len(devices),
                'device_info': self._network_source.get_device_info() if devices else "No devices"
            })
        
        return info
    
    def get_available_devices(self) -> list:
        """Get list of available wireless devices (initializes wireless if needed)."""
        if not WIRELESS_AVAILABLE:
            return []
        
        if not self._network_initialized:
            if not self._initialize_wireless():
                return []
        
        return self._network_source.get_available_devices() if self._network_source else []
    
    def record_audio(self, *args, **kwargs):
        """
        Main record function that delegates to current source.
        This is the primary interface used by the voice assistant.
        """
        # Track usage statistics
        if self.current_source == AudioSourceType.LOCAL:
            self.stats['local_recordings'] += 1
        else:
            self.stats['wireless_recordings'] += 1
        
        # Call the current record function
        return self._current_record_func(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = self.stats.copy()
        
        if self._network_source:
            network_stats = self._network_source.get_stats()
            stats['wireless_details'] = network_stats
        
        return stats
    
    def cleanup(self):
        """Clean up resources (call on shutdown)."""
        if self._network_source:
            logging.info("Cleaning up wireless audio source...")
            self._network_source.stop()
            self._network_source = None
            self._network_initialized = False
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()


# Convenience functions for global access
_global_audio_manager: Optional[AudioSourceManager] = None

def get_audio_manager() -> AudioSourceManager:
    """Get the global audio source manager."""
    global _global_audio_manager
    if _global_audio_manager is None:
        _global_audio_manager = AudioSourceManager()
    return _global_audio_manager

def set_audio_manager(manager: AudioSourceManager):
    """Set the global audio source manager."""
    global _global_audio_manager
    if _global_audio_manager:
        _global_audio_manager.cleanup()
    _global_audio_manager = manager

def cleanup_audio_manager():
    """Clean up the global audio source manager."""
    global _global_audio_manager
    if _global_audio_manager:
        _global_audio_manager.cleanup()
        _global_audio_manager = None


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create audio source manager
    manager = AudioSourceManager(AudioSourceType.LOCAL)
    
    # Test source switching
    print("Testing audio source manager...")
    
    # Get initial info
    info = manager.get_source_info()
    print(f"Initial source: {info}")
    
    # Try switching to wireless
    if manager.switch_to_wireless():
        print("Successfully switched to wireless")
        
        # List devices
        devices = manager.get_available_devices()
        print(f"Available devices: {devices}")
        
        # Switch back to local
        manager.switch_to_local()
        print("Switched back to local")
    else:
        print("Wireless switch failed, staying on local")
    
    # Show final stats
    stats = manager.get_stats()
    print(f"Final stats: {stats}")
    
    # Cleanup
    manager.cleanup()
    print("Test complete")