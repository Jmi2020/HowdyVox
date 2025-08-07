#!/usr/bin/env python3

import asyncio
import json
import logging
import numpy as np
import threading
import time
from typing import Dict, Optional, Callable, Any
import websockets
from websockets.server import WebSocketServerProtocol
import socket

class WebSocketTTSServer:
    """
    WebSocket TTS Audio Server for ESP32-P4 HowdyScreen devices.
    Handles bidirectional communication for TTS audio streaming and VAD feedback.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002):
        self.host = host
        self.port = port
        self.server = None
        self.running = False
        
        # Connected devices
        self.devices: Dict[str, WebSocketServerProtocol] = {}
        self.device_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self.tts_request_callback: Optional[Callable[[str, str], None]] = None
        self.vad_feedback_callback: Optional[Callable[[str, Dict], None]] = None
        
        # Thread for running the server
        self.server_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        logging.info(f"WebSocket TTS Server initialized for {host}:{port}")
    
    def set_tts_request_callback(self, callback: Callable[[str, str], None]):
        """Set callback for handling TTS requests from ESP32-P4."""
        self.tts_request_callback = callback
    
    def set_vad_feedback_callback(self, callback: Callable[[str, Dict], None]):
        """Set callback for handling VAD feedback from ESP32-P4."""
        self.vad_feedback_callback = callback
    
    def start_server(self):
        """Start the WebSocket TTS server in a background thread."""
        if self.running:
            logging.warning("WebSocket TTS server already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start
        time.sleep(0.5)
        logging.info(f"🔊 WebSocket TTS server started on {self.host}:{self.port}")
    
    def stop_server(self):
        """Stop the WebSocket TTS server."""
        if not self.running:
            return
        
        logging.info("Stopping WebSocket TTS server...")
        self.running = False
        
        # Stop the server
        if self.loop and self.server:
            asyncio.run_coroutine_threadsafe(self.server.close(), self.loop)
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        
        logging.info("WebSocket TTS server stopped")
    
    def _run_server(self):
        """Run the WebSocket server in its own thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._start_websocket_server())
        except Exception as e:
            logging.error(f"WebSocket TTS server error: {e}")
    
    async def _start_websocket_server(self):
        """Start the WebSocket server."""
        # Use a wrapper function to handle websockets version compatibility
        async def websocket_handler(websocket, path=None):
            if path is None:
                # For newer websockets versions that don't pass path
                path = websocket.path if hasattr(websocket, 'path') else "/"
            return await self._handle_client_connection(websocket, path)
        
        self.server = await websockets.serve(
            websocket_handler,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logging.info(f"📡 WebSocket TTS server listening on ws://{self.host}:{self.port}")
        
        # Keep server running
        while self.running:
            await asyncio.sleep(1)
    
    async def _handle_client_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new ESP32-P4 client connections."""
        client_ip = websocket.remote_address[0]
        device_id = f"esp32p4_{client_ip.replace('.', '_')}"
        
        logging.info(f"🔗 ESP32-P4 connected: {device_id} ({client_ip}) on path: {path}")
        
        # Register device
        self.devices[device_id] = websocket
        self.device_sessions[device_id] = {
            'connected_at': time.time(),
            'last_seen': time.time(),
            'tts_sessions': {},
            'vad_stats': {'corrections': 0, 'wake_words': 0}
        }
        
        try:
            # Send welcome message requesting device registration
            welcome_msg = {
                'type': 'server_info',
                'server_name': socket.gethostname(),
                'timestamp': int(time.time() * 1000),
                'supported_features': ['tts_audio', 'vad_feedback', 'wake_word_validation', 'device_registration'],
                'request': 'device_registration'  # Request device to send its info
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle client messages
            async for message in websocket:
                await self._handle_client_message(device_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"🔌 ESP32-P4 disconnected: {device_id}")
        except Exception as e:
            logging.error(f"Error handling ESP32-P4 {device_id}: {e}")
        finally:
            # Cleanup
            if device_id in self.devices:
                del self.devices[device_id]
            if device_id in self.device_sessions:
                del self.device_sessions[device_id]
    
    async def _handle_client_message(self, device_id: str, message: str):
        """Handle messages from ESP32-P4 clients."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', 'unknown')
            
            # Update last seen
            if device_id in self.device_sessions:
                self.device_sessions[device_id]['last_seen'] = time.time()
            
            logging.debug(f"📨 Received from {device_id}: {msg_type}")
            
            if msg_type == 'device_registration':
                # ESP32-P4 sending device registration info
                device_name = data.get('device_name', device_id)
                device_room = data.get('room', 'Unknown')
                firmware_version = data.get('firmware_version', 'Unknown')
                capabilities = data.get('capabilities', {})
                
                logging.info(f"📱 Device registration from {device_id}: {device_name} in {device_room}")
                
                # Update device session with registration info
                if device_id in self.device_sessions:
                    self.device_sessions[device_id]['device_name'] = device_name
                    self.device_sessions[device_id]['room'] = device_room
                    self.device_sessions[device_id]['firmware_version'] = firmware_version
                    self.device_sessions[device_id]['capabilities'] = capabilities
                    
                    logging.info(f"✅ Registered ESP32-P4: {device_name} ({device_room})")
                
                # Send registration confirmation
                confirm_msg = {
                    'type': 'registration_confirmed',
                    'device_id': device_id,
                    'server_time': time.time()
                }
                await websocket.send(json.dumps(confirm_msg))
            
            elif msg_type == 'tts_request':
                # ESP32-P4 requesting TTS for user speech
                text = data.get('text', '')
                session_id = data.get('session_id', f"tts_{int(time.time())}")
                
                logging.info(f"🎤 TTS request from {device_id}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                if self.tts_request_callback:
                    # Trigger TTS generation - callback should call send_tts_audio() when ready
                    self.tts_request_callback(device_id, text)
            
            elif msg_type == 'vad_feedback':
                # ESP32-P4 sending VAD detection feedback
                vad_data = data.get('vad_data', {})
                
                logging.debug(f"📊 VAD feedback from {device_id}: confidence={vad_data.get('confidence', 0)}")
                
                if self.vad_feedback_callback:
                    self.vad_feedback_callback(device_id, vad_data)
            
            elif msg_type == 'wake_word_detected':
                # ESP32-P4 detected wake word
                wake_word = data.get('wake_word', 'unknown')
                confidence = data.get('confidence', 0.0)
                
                logging.info(f"👂 Wake word detected by {device_id}: '{wake_word}' (confidence: {confidence:.2f})")
                
                # Update stats
                if device_id in self.device_sessions:
                    self.device_sessions[device_id]['vad_stats']['wake_words'] += 1
            
            elif msg_type == 'ping':
                # Respond to ping
                pong_msg = {'type': 'pong', 'timestamp': int(time.time() * 1000)}
                await self.devices[device_id].send(json.dumps(pong_msg))
            
            else:
                logging.warning(f"Unknown message type from {device_id}: {msg_type}")
        
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON from {device_id}: {message[:100]}")
        except Exception as e:
            logging.error(f"Error processing message from {device_id}: {e}")
    
    async def send_tts_audio(self, device_id: str, audio_data: bytes, session_id: str = None):
        """Send TTS audio to specific ESP32-P4 device."""
        if device_id not in self.devices:
            logging.warning(f"Device {device_id} not connected - cannot send TTS audio")
            return False
        
        try:
            # Convert audio data to base64 for JSON transport
            import base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            tts_msg = {
                'type': 'tts_audio',
                'session_id': session_id or f"tts_{int(time.time())}",
                'audio_format': 'pcm_16bit_mono_16khz',
                'audio_data': audio_b64,
                'timestamp': int(time.time() * 1000)
            }
            
            await self.devices[device_id].send(json.dumps(tts_msg))
            logging.debug(f"🔊 Sent TTS audio to {device_id}: {len(audio_data)} bytes")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send TTS audio to {device_id}: {e}")
            return False
    
    def send_tts_audio_sync(self, device_id: str, audio_data: bytes, session_id: str = None):
        """Send TTS audio synchronously (thread-safe)."""
        if not self.loop or not self.running:
            return False
        
        future = asyncio.run_coroutine_threadsafe(
            self.send_tts_audio(device_id, audio_data, session_id),
            self.loop
        )
        
        try:
            return future.result(timeout=5.0)
        except Exception as e:
            logging.error(f"Failed to send TTS audio sync to {device_id}: {e}")
            return False
    
    def get_connected_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get list of connected ESP32-P4 devices."""
        return {
            device_id: {
                'ip': self.devices[device_id].remote_address[0],
                'connected_at': session['connected_at'],
                'last_seen': session['last_seen'],
                'vad_stats': session['vad_stats'],
                'device_name': session.get('device_name', device_id),
                'room': session.get('room', 'Unknown'),
                'firmware_version': session.get('firmware_version', 'Unknown'),
                'capabilities': session.get('capabilities', {})
            }
            for device_id, session in self.device_sessions.items()
            if device_id in self.devices
        }
    
    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected ESP32-P4 devices."""
        if not self.loop or not self.running:
            return
        
        async def _broadcast():
            if not self.devices:
                return
            
            msg_json = json.dumps(message)
            disconnected = []
            
            for device_id, websocket in self.devices.items():
                try:
                    await websocket.send(msg_json)
                except Exception as e:
                    logging.warning(f"Failed to broadcast to {device_id}: {e}")
                    disconnected.append(device_id)
            
            # Cleanup disconnected devices
            for device_id in disconnected:
                if device_id in self.devices:
                    del self.devices[device_id]
                if device_id in self.device_sessions:
                    del self.device_sessions[device_id]
        
        asyncio.run_coroutine_threadsafe(_broadcast(), self.loop)


# Global WebSocket TTS server instance
_websocket_tts_server: Optional[WebSocketTTSServer] = None

def get_websocket_tts_server() -> Optional[WebSocketTTSServer]:
    """Get the global WebSocket TTS server instance."""
    return _websocket_tts_server

def start_websocket_tts_server(host: str = "0.0.0.0", port: int = 8002) -> WebSocketTTSServer:
    """Start the global WebSocket TTS server."""
    global _websocket_tts_server
    
    if _websocket_tts_server:
        logging.warning("WebSocket TTS server already running")
        return _websocket_tts_server
    
    _websocket_tts_server = WebSocketTTSServer(host, port)
    _websocket_tts_server.start_server()
    return _websocket_tts_server

def stop_websocket_tts_server():
    """Stop the global WebSocket TTS server."""
    global _websocket_tts_server
    
    if _websocket_tts_server:
        _websocket_tts_server.stop_server()
        _websocket_tts_server = None