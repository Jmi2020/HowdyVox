#!/usr/bin/env python3

import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import socket
import struct

@dataclass
class WirelessDevice:
    """Represents a wireless audio device (ESP32P4 HowdyScreen)."""
    device_id: str
    device_type: str = "ESP32P4_HowdyScreen"
    ip_address: str = ""
    port: int = 8000
    room: str = ""
    last_seen: float = 0.0
    is_active: bool = False
    audio_level: float = 0.0
    battery_level: int = -1  # -1 = unknown, 0-100 = percentage
    signal_strength: int = -1  # WiFi RSSI in dBm
    status: str = "unknown"  # unknown, ready, recording, muted, error
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WirelessDevice':
        return cls(**data)

class WirelessDeviceManager:
    """
    Manages wireless audio devices (ESP32P4 HowdyScreen units).
    Handles device discovery, registration, status monitoring, and room assignments.
    """
    
    def __init__(self, config_file: str = "wireless_devices.json"):
        self.config_file = Path(config_file)
        self.devices: Dict[str, WirelessDevice] = {}
        self.device_timeout = 30.0  # seconds
        self.discovery_timeout = 60.0  # seconds
        
        # Callbacks
        self.device_connected_callback: Optional[Callable[[WirelessDevice], None]] = None
        self.device_disconnected_callback: Optional[Callable[[WirelessDevice], None]] = None
        self.device_status_callback: Optional[Callable[[WirelessDevice], None]] = None
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Discovery
        self.discovery_socket: Optional[socket.socket] = None
        self.discovery_running = False
        self.discovery_thread: Optional[threading.Thread] = None
        
        # Load existing configuration
        self.load_config()
        
        logging.info(f"WirelessDeviceManager initialized with {len(self.devices)} saved devices")
    
    def set_callbacks(self, 
                     connected: Optional[Callable[[WirelessDevice], None]] = None,
                     disconnected: Optional[Callable[[WirelessDevice], None]] = None,
                     status_update: Optional[Callable[[WirelessDevice], None]] = None):
        """Set callback functions for device events."""
        self.device_connected_callback = connected
        self.device_disconnected_callback = disconnected
        self.device_status_callback = status_update
        logging.info("Device event callbacks registered")
    
    def start_monitoring(self):
        """Start device monitoring and discovery."""
        if self.monitoring:
            logging.warning("Device monitoring already running")
            return
        
        self.monitoring = True
        
        # Start status monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start device discovery
        self.start_discovery()
        
        logging.info("Wireless device monitoring started")
    
    def stop_monitoring(self):
        """Stop device monitoring and discovery."""
        logging.info("Stopping wireless device monitoring...")
        
        self.monitoring = False
        self.stop_discovery()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # Save configuration
        self.save_config()
        
        logging.info("Wireless device monitoring stopped")
    
    def start_discovery(self):
        """Start mDNS/UDP discovery for new devices."""
        if self.discovery_running:
            return
        
        try:
            # Create UDP socket for discovery
            self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.discovery_socket.settimeout(1.0)
            
            self.discovery_running = True
            self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
            self.discovery_thread.start()
            
            logging.info("Device discovery started")
            
        except Exception as e:
            logging.error(f"Failed to start device discovery: {e}")
    
    def stop_discovery(self):
        """Stop device discovery."""
        self.discovery_running = False
        
        if self.discovery_socket:
            self.discovery_socket.close()
            self.discovery_socket = None
        
        if self.discovery_thread and self.discovery_thread.is_alive():
            self.discovery_thread.join(timeout=2.0)
    
    def register_device(self, device_id: str, ip_address: str, port: int = 8000) -> WirelessDevice:
        """Register a new wireless device or update existing one."""
        current_time = time.time()
        
        if device_id in self.devices:
            # Update existing device
            device = self.devices[device_id]
            was_active = device.is_active
            device.ip_address = ip_address
            device.port = port
            device.last_seen = current_time
            device.is_active = True
            
            if not was_active and self.device_connected_callback:
                self.device_connected_callback(device)
                logging.info(f"Wireless device reconnected: {device_id} ({ip_address})")
        else:
            # Create new device
            device = WirelessDevice(
                device_id=device_id,
                ip_address=ip_address,
                port=port,
                last_seen=current_time,
                is_active=True,
                status="ready"
            )
            self.devices[device_id] = device
            
            if self.device_connected_callback:
                self.device_connected_callback(device)
            
            logging.info(f"New wireless device registered: {device_id} ({ip_address})")
        
        # Save updated configuration
        self.save_config()
        return device
    
    def update_device_status(self, device_id: str, **kwargs):
        """Update device status information."""
        if device_id not in self.devices:
            logging.warning(f"Attempt to update unknown device: {device_id}")
            return
        
        device = self.devices[device_id]
        updated = False
        
        for key, value in kwargs.items():
            if hasattr(device, key):
                setattr(device, key, value)
                updated = True
        
        if updated:
            device.last_seen = time.time()
            if self.device_status_callback:
                self.device_status_callback(device)
    
    def assign_room(self, device_id: str, room: str) -> bool:
        """Assign a device to a specific room."""
        if device_id not in self.devices:
            logging.error(f"Cannot assign room to unknown device: {device_id}")
            return False
        
        # Check if room is already assigned to another device
        for other_id, other_device in self.devices.items():
            if other_id != device_id and other_device.room == room:
                logging.warning(f"Room '{room}' already assigned to device {other_id}")
                return False
        
        self.devices[device_id].room = room
        self.save_config()
        
        logging.info(f"Device {device_id} assigned to room: {room}")
        return True
    
    def get_device(self, device_id: str) -> Optional[WirelessDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)
    
    def get_device_by_room(self, room: str) -> Optional[WirelessDevice]:
        """Get device assigned to a specific room."""
        for device in self.devices.values():
            if device.room == room and device.is_active:
                return device
        return None
    
    def get_active_devices(self) -> List[WirelessDevice]:
        """Get list of currently active devices."""
        return [device for device in self.devices.values() if device.is_active]
    
    def get_all_devices(self) -> List[WirelessDevice]:
        """Get list of all registered devices."""
        return list(self.devices.values())
    
    def remove_device(self, device_id: str) -> bool:
        """Remove a device from the registry."""
        if device_id not in self.devices:
            return False
        
        device = self.devices[device_id]
        if device.is_active and self.device_disconnected_callback:
            self.device_disconnected_callback(device)
        
        del self.devices[device_id]
        self.save_config()
        
        logging.info(f"Device removed: {device_id}")
        return True
    
    def send_command(self, device_id: str, command: str, data: Any = None) -> bool:
        """Send a command to a specific device (placeholder for WebSocket implementation)."""
        device = self.get_device(device_id)
        if not device or not device.is_active:
            logging.error(f"Cannot send command to inactive device: {device_id}")
            return False
        
        # TODO: Implement WebSocket command sending
        logging.info(f"Command '{command}' sent to device {device_id}: {data}")
        return True
    
    def broadcast_command(self, command: str, data: Any = None) -> int:
        """Broadcast a command to all active devices."""
        count = 0
        for device in self.get_active_devices():
            if self.send_command(device.device_id, command, data):
                count += 1
        return count
    
    def _monitor_loop(self):
        """Main monitoring loop - checks device timeouts."""
        logging.info("Device monitoring loop started")
        
        while self.monitoring:
            try:
                current_time = time.time()
                inactive_devices = []
                
                for device_id, device in self.devices.items():
                    if device.is_active and (current_time - device.last_seen) > self.device_timeout:
                        inactive_devices.append(device_id)
                
                # Mark devices as inactive
                for device_id in inactive_devices:
                    device = self.devices[device_id]
                    device.is_active = False
                    device.status = "disconnected"
                    
                    if self.device_disconnected_callback:
                        self.device_disconnected_callback(device)
                    
                    logging.warning(f"Device {device_id} marked as inactive (timeout)")
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error in device monitoring loop: {e}")
        
        logging.info("Device monitoring loop ended")
    
    def _discovery_loop(self):
        """Device discovery loop - broadcasts discovery packets."""
        logging.info("Device discovery loop started")
        
        while self.discovery_running:
            try:
                # Send discovery broadcast
                discovery_packet = b"HOWDYTTS_DISCOVERY"
                self.discovery_socket.sendto(discovery_packet, ("255.255.255.255", 8001))
                
                # Listen for responses
                try:
                    data, addr = self.discovery_socket.recvfrom(1024)
                    self._handle_discovery_response(data, addr)
                except socket.timeout:
                    pass
                
                time.sleep(self.discovery_timeout)  # Discovery interval
                
            except Exception as e:
                if self.discovery_running:
                    logging.error(f"Error in discovery loop: {e}")
        
        logging.info("Device discovery loop ended")
    
    def _handle_discovery_response(self, data: bytes, addr: tuple):
        """Handle discovery response from device."""
        try:
            response = data.decode('utf-8')
            if response.startswith("HOWDYSCREEN_"):
                # Extract device info from response
                parts = response.split("_")
                if len(parts) >= 3:
                    device_type = parts[1]
                    device_id = parts[2]
                    
                    # Register or update device
                    self.register_device(device_id, addr[0])
                    
        except Exception as e:
            logging.debug(f"Error handling discovery response: {e}")
    
    def load_config(self):
        """Load device configuration from file."""
        if not self.config_file.exists():
            logging.info("No wireless device configuration file found")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.devices = {}
            for device_data in config.get('devices', []):
                device = WirelessDevice.from_dict(device_data)
                device.is_active = False  # Will be updated by monitoring
                self.devices[device.device_id] = device
            
            logging.info(f"Loaded {len(self.devices)} devices from configuration")
            
        except Exception as e:
            logging.error(f"Failed to load device configuration: {e}")
    
    def save_config(self):
        """Save device configuration to file."""
        try:
            config = {
                'devices': [device.to_dict() for device in self.devices.values()],
                'last_updated': time.time()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logging.debug(f"Device configuration saved to {self.config_file}")
            
        except Exception as e:
            logging.error(f"Failed to save device configuration: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get device manager statistics."""
        active_devices = self.get_active_devices()
        
        return {
            'total_devices': len(self.devices),
            'active_devices': len(active_devices),
            'rooms_assigned': len([d for d in self.devices.values() if d.room]),
            'device_types': list(set(d.device_type for d in self.devices.values())),
            'monitoring': self.monitoring,
            'discovery_running': self.discovery_running
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_device_connected(device: WirelessDevice):
        print(f"Device connected: {device.device_id} at {device.ip_address}")
    
    def on_device_disconnected(device: WirelessDevice):
        print(f"Device disconnected: {device.device_id}")
    
    def on_device_status(device: WirelessDevice):
        print(f"Device status update: {device.device_id} - {device.status}")
    
    # Create device manager
    manager = WirelessDeviceManager("test_wireless_devices.json")
    manager.set_callbacks(on_device_connected, on_device_disconnected, on_device_status)
    
    # Start monitoring
    manager.start_monitoring()
    
    try:
        # Simulate device registration
        time.sleep(2)
        device = manager.register_device("ESP32P4_001", "192.168.1.100")
        manager.assign_room("ESP32P4_001", "Living Room")
        
        # Update device status
        manager.update_device_status("ESP32P4_001", 
                                   status="recording", 
                                   audio_level=0.75, 
                                   signal_strength=-45)
        
        # Test for a while
        for i in range(10):
            time.sleep(3)
            stats = manager.get_stats()
            devices = manager.get_active_devices()
            print(f"Stats: {stats}")
            print(f"Active devices: {[d.device_id for d in devices]}")
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        manager.stop_monitoring()
    
    print("Device manager test complete")