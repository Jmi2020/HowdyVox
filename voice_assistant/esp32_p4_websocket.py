#!/usr/bin/env python3

import asyncio
import websockets
import json
import logging
import time
import struct
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict

@dataclass
class DeviceInfo:
    """Information about connected ESP32-P4 device."""
    device_id: str
    websocket: Any  # websockets.WebSocketServerProtocol
    ip_address: str
    port: int
    last_seen: float
    capabilities: Dict[str, Any]
    vad_accuracy: float = 0.0
    wake_word_accuracy: float = 0.0
    last_stats: Optional[Dict[str, Any]] = None
    last_stats_time: float = 0.0

class FeedbackMessageType(Enum):
    """Types of feedback messages sent to ESP32-P4 devices."""
    VAD_CORRECTION = "vad_correction"
    WAKE_WORD_VALIDATION = "wake_word_validation"
    WAKE_WORD_REJECTION = "wake_word_rejection"
    VAD_SENSITIVITY_UPDATE = "vad_sensitivity_update"
    WAKE_WORD_SENSITIVITY_UPDATE = "wake_word_sensitivity_update"
    DEVICE_SYNC = "device_sync"
    SYSTEM_STATUS = "system_status"

@dataclass
class FeedbackMessage:
    """Feedback message to ESP32-P4 device."""
    message_type: FeedbackMessageType
    device_id: str
    timestamp: float
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

class ESP32P4WebSocketServer:
    """
    WebSocket server for real-time feedback and coordination with ESP32-P4 devices.
    
    Features:
    - Real-time VAD correction feedback
    - Wake word validation/rejection signals
    - Multi-device wake word synchronization
    - Adaptive sensitivity updates
    - Device status monitoring
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 8001,
                 max_devices: int = 10):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host address
            port: WebSocket server port
            max_devices: Maximum number of connected devices
        """
        self.host = host
        self.port = port
        self.max_devices = max_devices
        
        # Connected devices
        self.connected_devices: Dict[str, DeviceInfo] = {}
        self.device_lock = threading.RLock()
        
        # Event loop and server
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server = None
        self.running = False
        
        # Message queues
        self.outbound_messages: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        
        # Callbacks
        self.device_connected_callback: Optional[Callable] = None
        self.device_disconnected_callback: Optional[Callable] = None
        self.wake_word_sync_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'active_devices': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'vad_corrections_sent': 0,
            'wake_word_validations_sent': 0,
            'wake_word_rejections_sent': 0,
            'sync_events': 0
        }
        
        logging.info(f"ESP32-P4 WebSocket server initialized on {host}:{port}")
    
    def start(self) -> bool:
        """Start the WebSocket server in a separate thread."""
        try:
            # Start server thread
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            # Wait for server to start
            time.sleep(0.5)
            
            logging.info(f"ESP32-P4 WebSocket server started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start WebSocket server: {e}")
            return False
    
    def stop(self):
        """Stop the WebSocket server."""
        self.running = False
        
        if self.server and self.loop:
            asyncio.run_coroutine_threadsafe(self.server.close(), self.loop)
        
        logging.info("ESP32-P4 WebSocket server stopped")
    
    def _run_server(self):
        """Run the WebSocket server event loop."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            self.running = True
            
            # Start WebSocket server
            start_server = websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.server = self.loop.run_until_complete(start_server)
            
            # Run until stopped
            self.loop.run_forever()
            
        except Exception as e:
            logging.error(f"WebSocket server error: {e}")
        finally:
            if self.loop:
                self.loop.close()
    
    async def _handle_client(self, websocket, path):
        """Handle individual WebSocket client connections."""
        device_id = None
        
        try:
            # Wait for device registration
            registration_msg = await asyncio.wait_for(
                websocket.recv(), 
                timeout=10.0
            )
            
            registration = json.loads(registration_msg)
            if registration.get('type') != 'device_registration':
                await websocket.close(code=1003, reason="Invalid registration")
                return
            
            device_id = registration.get('device_id')
            if not device_id:
                await websocket.close(code=1003, reason="Missing device_id")
                return
            
            # Check device limit
            with self.device_lock:
                if len(self.connected_devices) >= self.max_devices:
                    await websocket.close(code=1013, reason="Server full")
                    return
                
                # Register device
                device_info = DeviceInfo(
                    device_id=device_id,
                    websocket=websocket,
                    ip_address=websocket.remote_address[0],
                    port=websocket.remote_address[1],
                    last_seen=time.time(),
                    capabilities=registration.get('capabilities', {})
                )
                
                self.connected_devices[device_id] = device_info
                self.stats['total_connections'] += 1
                self.stats['active_devices'] = len(self.connected_devices)
            
            logging.info(f"ESP32-P4 device {device_id} connected from {websocket.remote_address}")
            
            # Notify callback
            if self.device_connected_callback:
                self.device_connected_callback(device_info)
            
            # Send welcome message
            await self._send_message(websocket, {
                'type': 'welcome',
                'server_time': time.time(),
                'device_id': device_id
            })
            
            # Start message sender task
            sender_task = asyncio.create_task(self._message_sender(device_id))
            
            try:
                # Handle incoming messages
                async for message in websocket:
                    await self._handle_message(device_id, message)
                    
                    # Update last seen
                    with self.device_lock:
                        if device_id in self.connected_devices:
                            self.connected_devices[device_id].last_seen = time.time()
            
            except websockets.exceptions.ConnectionClosed:
                logging.info(f"ESP32-P4 device {device_id} disconnected")
            finally:
                sender_task.cancel()
        
        except asyncio.TimeoutError:
            logging.warning(f"Registration timeout for {websocket.remote_address}")
            await websocket.close(code=1000, reason="Registration timeout")
        
        except Exception as e:
            logging.error(f"Error handling WebSocket client: {e}")
        
        finally:
            # Clean up device registration
            if device_id:
                with self.device_lock:
                    if device_id in self.connected_devices:
                        device_info = self.connected_devices.pop(device_id)
                        self.stats['active_devices'] = len(self.connected_devices)
                        
                        # Notify callback
                        if self.device_disconnected_callback:
                            self.device_disconnected_callback(device_info)
    
    async def _message_sender(self, device_id: str):
        """Send queued messages to device."""
        try:
            queue = self.outbound_messages[device_id]
            
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    with self.device_lock:
                        if device_id not in self.connected_devices:
                            break
                        
                        device_info = self.connected_devices[device_id]
                        await self._send_message(device_info.websocket, message)
                        self.stats['messages_sent'] += 1
                
                except asyncio.TimeoutError:
                    # Check if device still connected
                    with self.device_lock:
                        if device_id not in self.connected_devices:
                            break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Error in message sender for {device_id}: {e}")
    
    async def _handle_message(self, device_id: str, message: str):
        """Handle incoming message from ESP32-P4 device."""
        try:
            data = json.loads(message)
            self.stats['messages_received'] += 1
            
            msg_type = data.get('type')
            
            if msg_type == 'heartbeat':
                await self._handle_heartbeat(device_id, data)
            elif msg_type == 'wake_word_detected':
                await self._handle_wake_word_sync(device_id, data)
            elif msg_type == 'vad_feedback':
                await self._handle_vad_feedback(device_id, data)
            elif msg_type == 'status_update':
                await self._handle_status_update(device_id, data)
            elif msg_type == 'device_statistics':
                await self._handle_device_statistics(device_id, data)
            else:
                logging.warning(f"Unknown message type from {device_id}: {msg_type}")
        
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON from {device_id}: {message}")
        except Exception as e:
            logging.error(f"Error handling message from {device_id}: {e}")
    
    async def _handle_heartbeat(self, device_id: str, data: Dict):
        """Handle heartbeat message."""
        # Update device status if provided
        if 'status' in data:
            with self.device_lock:
                if device_id in self.connected_devices:
                    device = self.connected_devices[device_id]
                    device.capabilities.update(data.get('status', {}))
    
    async def _handle_wake_word_sync(self, device_id: str, data: Dict):
        """Handle wake word synchronization."""
        self.stats['sync_events'] += 1
        
        # Notify all other devices about wake word detection
        sync_message = {
            'type': 'wake_word_sync',
            'source_device': device_id,
            'keyword_id': data.get('keyword_id'),
            'confidence': data.get('confidence'),
            'timestamp': time.time()
        }
        
        await self._broadcast_message(sync_message, exclude_device=device_id)
        
        # Call sync callback
        if self.wake_word_sync_callback:
            self.wake_word_sync_callback(device_id, data)
    
    async def _handle_vad_feedback(self, device_id: str, data: Dict):
        """Handle VAD accuracy feedback."""
        with self.device_lock:
            if device_id in self.connected_devices:
                device = self.connected_devices[device_id]
                if 'vad_accuracy' in data:
                    device.vad_accuracy = data['vad_accuracy']
                if 'wake_word_accuracy' in data:
                    device.wake_word_accuracy = data['wake_word_accuracy']
    
    async def _handle_status_update(self, device_id: str, data: Dict):
        """Handle device status updates."""
        with self.device_lock:
            if device_id in self.connected_devices:
                device = self.connected_devices[device_id]
                device.capabilities.update(data.get('capabilities', {}))

    async def _handle_device_statistics(self, device_id: str, data: Dict):
        """Handle periodic statistics from ESP32-P4 devices."""
        with self.device_lock:
            if device_id not in self.connected_devices:
                return
            device = self.connected_devices[device_id]
            device.last_stats = data
            device.last_stats_time = time.time()

        wake_stats = data.get('wake_word_stats', {})
        vad_stats = data.get('vad_stats', {})
        logging.debug(
            "📈 Device stats %s - detections=%s tp=%s fp=%s thr=%s voice=%s silence=%s",
            device_id,
            wake_stats.get('total_detections'),
            wake_stats.get('true_positives'),
            wake_stats.get('false_positives'),
            wake_stats.get('current_threshold'),
            vad_stats.get('voice_packets'),
            vad_stats.get('silence_packets')
        )
    
    async def _send_message(self, websocket, message: Dict):
        """Send message to WebSocket client."""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            pass  # Client disconnected
    
    async def _broadcast_message(self, message: Dict, exclude_device: Optional[str] = None):
        """Broadcast message to all connected devices."""
        with self.device_lock:
            for device_id, device_info in self.connected_devices.items():
                if exclude_device and device_id == exclude_device:
                    continue
                
                try:
                    await self.outbound_messages[device_id].put(message)
                except Exception as e:
                    logging.error(f"Error queuing message for {device_id}: {e}")
    
    def send_vad_correction(self, 
                           device_id: str, 
                           is_correct_decision: bool,
                           timestamp: float,
                           confidence_adjustment: float = 0.0):
        """Send VAD correction feedback to specific device."""
        message = {
            'type': FeedbackMessageType.VAD_CORRECTION.value,
            'timestamp': timestamp,
            'is_correct': is_correct_decision,
            'confidence_adjustment': confidence_adjustment,
            'server_time': time.time()
        }
        
        self._queue_message(device_id, message)
        self.stats['vad_corrections_sent'] += 1
        
        logging.debug(f"Sent VAD correction to {device_id}: correct={is_correct_decision}")
    
    def send_wake_word_validation(self, 
                                 device_id: str, 
                                 keyword_id: int, 
                                 validation_confidence: float,
                                 correlation_id: Optional[str] = None):
        """Send wake word validation to device."""
        message = {
            'type': FeedbackMessageType.WAKE_WORD_VALIDATION.value,
            'keyword_id': keyword_id,
            'validation_confidence': validation_confidence,
            'correlation_id': correlation_id,
            'server_time': time.time()
        }
        
        self._queue_message(device_id, message)
        self.stats['wake_word_validations_sent'] += 1
        
        logging.info(f"Validated wake word from {device_id}: keyword={keyword_id}, confidence={validation_confidence:.2f}")
    
    def send_wake_word_rejection(self, 
                                device_id: str, 
                                keyword_id: int, 
                                reason: str = "false_positive",
                                correlation_id: Optional[str] = None):
        """Send wake word rejection to device."""
        message = {
            'type': FeedbackMessageType.WAKE_WORD_REJECTION.value,
            'keyword_id': keyword_id,
            'reason': reason,
            'correlation_id': correlation_id,
            'server_time': time.time()
        }
        
        self._queue_message(device_id, message)
        self.stats['wake_word_rejections_sent'] += 1
        
        logging.info(f"Rejected wake word from {device_id}: keyword={keyword_id}, reason={reason}")
    
    def broadcast_wake_word_sync(self, 
                                source_device_id: str, 
                                keyword_id: int, 
                                confidence: float):
        """Broadcast wake word detection to all devices for synchronization."""
        message = {
            'type': 'wake_word_sync',
            'source_device': source_device_id,
            'keyword_id': keyword_id,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        asyncio.run_coroutine_threadsafe(
            self._broadcast_message(message, exclude_device=source_device_id),
            self.loop
        )
        
        self.stats['sync_events'] += 1
    
    def update_device_sensitivity(self, 
                                 device_id: str, 
                                 vad_sensitivity: Optional[float] = None,
                                 wake_word_sensitivity: Optional[float] = None):
        """Update device sensitivity settings."""
        message = {
            'type': 'sensitivity_update',
            'server_time': time.time()
        }
        
        if vad_sensitivity is not None:
            message['vad_sensitivity'] = vad_sensitivity
        
        if wake_word_sensitivity is not None:
            message['wake_word_sensitivity'] = wake_word_sensitivity
        
        self._queue_message(device_id, message)
        
        logging.info(f"Updated sensitivity for {device_id}: VAD={vad_sensitivity}, Wake={wake_word_sensitivity}")
    
    def _queue_message(self, device_id: str, message: Dict):
        """Queue message for sending to device."""
        if self.loop and device_id in self.connected_devices:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.outbound_messages[device_id].put(message),
                    self.loop
                )
            except Exception as e:
                logging.error(f"Error queuing message for {device_id}: {e}")
    
    def get_connected_devices(self) -> Dict[str, DeviceInfo]:
        """Get list of connected devices."""
        with self.device_lock:
            return self.connected_devices.copy()
    
    def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get information about specific device."""
        with self.device_lock:
            return self.connected_devices.get(device_id)
    
    def is_device_connected(self, device_id: str) -> bool:
        """Check if device is connected."""
        with self.device_lock:
            return device_id in self.connected_devices
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return self.stats.copy()
    
    def set_callbacks(self, 
                     device_connected: Optional[Callable] = None,
                     device_disconnected: Optional[Callable] = None,
                     wake_word_sync: Optional[Callable] = None):
        """Set event callbacks."""
        if device_connected:
            self.device_connected_callback = device_connected
        if device_disconnected:
            self.device_disconnected_callback = device_disconnected
        if wake_word_sync:
            self.wake_word_sync_callback = wake_word_sync


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_device_connected(device_info):
        print(f"Device connected: {device_info.device_id} from {device_info.ip_address}")
    
    def on_device_disconnected(device_info):
        print(f"Device disconnected: {device_info.device_id}")
    
    def on_wake_word_sync(device_id, data):
        print(f"Wake word sync from {device_id}: {data}")
    
    # Create and start server
    server = ESP32P4WebSocketServer(host="0.0.0.0", port=8001)
    server.set_callbacks(
        device_connected=on_device_connected,
        device_disconnected=on_device_disconnected,
        wake_word_sync=on_wake_word_sync
    )
    
    if server.start():
        print("WebSocket server started. Press Ctrl+C to stop.")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
                
                # Print stats every 30 seconds
                if int(time.time()) % 30 == 0:
                    stats = server.get_stats()
                    devices = server.get_connected_devices()
                    print(f"Stats: {stats}")
                    print(f"Connected devices: {list(devices.keys())}")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.stop()
    
    print("Server stopped")