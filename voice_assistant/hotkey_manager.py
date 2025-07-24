#!/usr/bin/env python3

import logging
import threading
import time
from typing import Optional, Dict, Callable

# Disable keyboard module on macOS due to Bus error issues
import platform

KEYBOARD_AVAILABLE = False
if platform.system() != 'Darwin':  # Not macOS
    try:
        import keyboard
        KEYBOARD_AVAILABLE = True
    except ImportError:
        KEYBOARD_AVAILABLE = False
        logging.warning("keyboard module not available - hotkeys disabled")
else:
    logging.info("Hotkeys disabled on macOS due to compatibility issues")

from .audio_source_manager import get_audio_manager, AudioSourceType

class HotkeyManager:
    """
    Lightweight hotkey manager for runtime audio source switching.
    Provides keyboard shortcuts for switching between local and wireless microphones.
    """
    
    def __init__(self):
        self.enabled = KEYBOARD_AVAILABLE
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Hotkey mappings
        self.hotkeys = {
            'ctrl+alt+l': self._switch_to_local,
            'ctrl+alt+w': self._switch_to_wireless,
            'ctrl+alt+t': self._toggle_source,
            'ctrl+alt+i': self._show_audio_info,
            'ctrl+alt+d': self._list_devices
        }
        
        # Statistics
        self.stats = {
            'hotkey_presses': 0,
            'source_switches': 0,
            'last_hotkey': None,
            'last_switch_time': 0
        }
        
        if not self.enabled:
            logging.info("HotkeyManager initialized but disabled (keyboard module not available)")
        else:
            logging.info("HotkeyManager initialized with hotkeys enabled")
    
    def start(self):
        """Start hotkey monitoring in background thread."""
        if not self.enabled:
            logging.warning("Cannot start hotkeys - keyboard module not available")
            return False
        
        if self.running:
            return True
        
        self.running = True
        
        # Register hotkeys
        try:
            for hotkey, callback in self.hotkeys.items():
                keyboard.add_hotkey(hotkey, callback, suppress=False)
            
            logging.info("Hotkeys registered:")
            logging.info("  Ctrl+Alt+L - Switch to local microphone")
            logging.info("  Ctrl+Alt+W - Switch to wireless microphone")
            logging.info("  Ctrl+Alt+T - Toggle audio source")
            logging.info("  Ctrl+Alt+I - Show audio source info")
            logging.info("  Ctrl+Alt+D - List wireless devices")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to register hotkeys: {e}")
            self.running = False
            return False
    
    def stop(self):
        """Stop hotkey monitoring and cleanup."""
        if not self.enabled or not self.running:
            return
        
        self.running = False
        
        try:
            # Remove all hotkeys
            keyboard.unhook_all_hotkeys()
            logging.info("Hotkeys unregistered")
            
        except Exception as e:
            logging.error(f"Error stopping hotkeys: {e}")
    
    def _switch_to_local(self):
        """Hotkey callback: switch to local microphone."""
        self._handle_hotkey('switch_to_local')
        
        audio_manager = get_audio_manager()
        if audio_manager:
            success = audio_manager.switch_to_local()
            if success:
                print(f"\n🎤 Switched to LOCAL microphone")
                self.stats['source_switches'] += 1
            else:
                print(f"\n❌ Failed to switch to local microphone")
    
    def _switch_to_wireless(self):
        """Hotkey callback: switch to wireless microphone."""
        self._handle_hotkey('switch_to_wireless')
        
        audio_manager = get_audio_manager()
        if audio_manager:
            success = audio_manager.switch_to_wireless()
            if success:
                print(f"\n📡 Switched to WIRELESS microphone")
                self.stats['source_switches'] += 1
            else:
                print(f"\n❌ Failed to switch to wireless microphone")
    
    def _toggle_source(self):
        """Hotkey callback: toggle between audio sources."""
        self._handle_hotkey('toggle_source')
        
        audio_manager = get_audio_manager()
        if audio_manager:
            current = audio_manager.get_current_source()
            success = audio_manager.toggle_source()
            
            if success:
                new_source = audio_manager.get_current_source()
                icon = "📡" if new_source == AudioSourceType.WIRELESS else "🎤"
                print(f"\n{icon} Toggled to {new_source.value.upper()} microphone")
                self.stats['source_switches'] += 1
            else:
                print(f"\n❌ Failed to toggle audio source")
    
    def _show_audio_info(self):
        """Hotkey callback: show current audio source information."""
        self._handle_hotkey('show_info')
        
        audio_manager = get_audio_manager()
        if audio_manager:
            info = audio_manager.get_source_info()
            current = info['current_source'].upper()
            icon = "📡" if info['current_source'] == 'wireless' else "🎤"
            
            print(f"\n{icon} Current audio source: {current}")
            
            if info.get('target_room'):
                print(f"   Target room: {info['target_room']}")
            
            if info.get('wireless_devices', 0) > 0:
                print(f"   Wireless devices: {info['wireless_devices']}")
            elif info['current_source'] == 'wireless':
                print(f"   ⚠️  No wireless devices currently available")
    
    def _list_devices(self):
        """Hotkey callback: list available wireless devices."""
        self._handle_hotkey('list_devices')
        
        audio_manager = get_audio_manager()
        if audio_manager:
            print(f"\n📋 Discovering wireless devices...")
            
            try:
                devices = audio_manager.get_available_devices()
                
                if devices:
                    print(f"   Found {len(devices)} wireless device(s):")
                    for idx, (_, name, ip) in enumerate(devices):
                        print(f"     {idx + 1}. {name} - {ip}")
                else:
                    print(f"   No wireless devices found")
                    
            except Exception as e:
                print(f"   ❌ Error listing devices: {e}")
    
    def _handle_hotkey(self, action: str):
        """Common hotkey handling logic."""
        self.stats['hotkey_presses'] += 1
        self.stats['last_hotkey'] = action
        self.stats['last_switch_time'] = time.time()
        
        logging.debug(f"Hotkey triggered: {action}")
    
    def get_stats(self) -> Dict:
        """Get hotkey usage statistics."""
        return self.stats.copy()
    
    def is_enabled(self) -> bool:
        """Check if hotkeys are enabled and available."""
        return self.enabled and self.running


# Global hotkey manager instance
_global_hotkey_manager: Optional[HotkeyManager] = None

def get_hotkey_manager() -> HotkeyManager:
    """Get the global hotkey manager."""
    global _global_hotkey_manager
    if _global_hotkey_manager is None:
        _global_hotkey_manager = HotkeyManager()
    return _global_hotkey_manager

def start_hotkeys() -> bool:
    """Start the global hotkey manager."""
    manager = get_hotkey_manager()
    return manager.start()

def stop_hotkeys():
    """Stop the global hotkey manager."""
    global _global_hotkey_manager
    if _global_hotkey_manager:
        _global_hotkey_manager.stop()
        _global_hotkey_manager = None


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test hotkey manager
    print("Testing hotkey manager...")
    print("Available hotkeys:")
    print("  Ctrl+Alt+L - Switch to local microphone")
    print("  Ctrl+Alt+W - Switch to wireless microphone") 
    print("  Ctrl+Alt+T - Toggle audio source")
    print("  Ctrl+Alt+I - Show audio source info")
    print("  Ctrl+Alt+D - List wireless devices")
    print("\nPress any hotkey to test, or Ctrl+C to exit...")
    
    manager = HotkeyManager()
    
    if manager.start():
        try:
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping hotkey manager...")
        finally:
            manager.stop()
    else:
        print("Failed to start hotkey manager")
    
    print("Hotkey manager test complete")