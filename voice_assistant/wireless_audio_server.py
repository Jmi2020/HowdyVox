#!/usr/bin/env python3

import socket
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
import struct
import numpy as np
from queue import Queue, Empty

try:
    import opuslib
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False
    logging.warning("opuslib not available - OPUS decoding disabled")

class WirelessAudioServer:
    """
    UDP server for receiving wireless audio from ESP32P4 HowdyScreen devices.
    Handles OPUS decoding and integrates with HowdyTTS audio pipeline.
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 8000,
                 sample_rate: int = 16000,
                 channels: int = 1):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Network components
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.receive_thread: Optional[threading.Thread] = None
        
        # Audio processing
        self.opus_decoder = None
        if OPUS_AVAILABLE:
            try:
                self.opus_decoder = opuslib.Decoder(sample_rate, channels)
                logging.info("OPUS decoder initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize OPUS decoder: {e}")
                self.opus_decoder = None
        
        # Audio callback and buffering
        self.audio_callback: Optional[Callable[[np.ndarray], None]] = None
        self.audio_queue = Queue(maxsize=50)  # ~1 second buffer at 20ms frames
        self.process_thread: Optional[threading.Thread] = None
        
        # Statistics and monitoring
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'opus_decode_errors': 0,
            'last_packet_time': 0,
            'connected_devices': set()
        }
        
        # Device registry
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.device_timeout = 10.0  # seconds
        
        logging.info(f"WirelessAudioServer initialized for {host}:{port}")
    
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function to receive decoded audio data."""
        self.audio_callback = callback
        logging.info("Audio callback registered")
    
    def start(self) -> bool:
        """Start the UDP server and processing threads."""
        if self.running:
            logging.warning("Server already running")
            return True
        
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set socket buffer sizes for real-time audio
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            
            # Bind to address
            self.socket.bind((self.host, self.port))
            
            # Set socket timeout for graceful shutdown
            self.socket.settimeout(1.0)
            
            self.running = True
            
            # Start receiving thread
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            
            # Start audio processing thread
            self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.process_thread.start()
            
            # Start device cleanup thread
            cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            cleanup_thread.start()
            
            logging.info(f"WirelessAudioServer started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the server and all threads."""
        logging.info("Stopping WirelessAudioServer...")
        self.running = False
        
        if self.socket:
            self.socket.close()
            self.socket = None
        
        # Wait for threads to finish
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        logging.info("WirelessAudioServer stopped")
    
    def _receive_loop(self):
        """Main receive loop - runs in separate thread."""
        logging.info("Audio receive loop started")
        
        while self.running:
            try:
                if not self.socket:
                    break
                
                # Receive UDP packet
                data, addr = self.socket.recvfrom(2048)  # Max OPUS packet size
                
                # Update statistics
                self.stats['packets_received'] += 1
                self.stats['last_packet_time'] = time.time()
                
                # Register/update device
                device_id = f"{addr[0]}:{addr[1]}"
                self._register_device(device_id, addr)
                
                # Queue packet for processing
                try:
                    packet_info = {
                        'data': data,
                        'addr': addr,
                        'timestamp': time.time()
                    }
                    self.audio_queue.put_nowait(packet_info)
                except:
                    self.stats['packets_dropped'] += 1
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Error in receive loop: {e}")
                break
        
        logging.info("Audio receive loop ended")
    
    def _process_loop(self):
        """Audio processing loop - decodes OPUS and calls callback."""
        logging.info("Audio processing loop started")
        
        while self.running:
            try:
                # Get packet from queue with timeout
                packet_info = self.audio_queue.get(timeout=0.1)
                
                # Decode audio
                audio_data = self._decode_audio(packet_info['data'])
                
                if audio_data is not None and self.audio_callback:
                    # Convert to numpy array and call callback
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.audio_callback(audio_array)
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
        
        logging.info("Audio processing loop ended")
    
    def _decode_audio(self, opus_data: bytes) -> Optional[bytes]:
        """Decode OPUS-encoded audio data."""
        if not self.opus_decoder or not OPUS_AVAILABLE:
            # Without OPUS, assume raw PCM data
            return opus_data
        
        try:
            # Decode OPUS to PCM
            pcm_data = self.opus_decoder.decode(opus_data, frame_size=320)  # 20ms at 16kHz
            return pcm_data
            
        except Exception as e:
            self.stats['opus_decode_errors'] += 1
            logging.debug(f"OPUS decode error: {e}")
            return None
    
    def _register_device(self, device_id: str, addr: tuple):
        """Register or update device information."""
        current_time = time.time()
        
        if device_id not in self.devices:
            logging.info(f"New wireless device connected: {device_id}")
            self.stats['connected_devices'].add(device_id)
        
        self.devices[device_id] = {
            'addr': addr,
            'last_seen': current_time,
            'packets_received': self.devices.get(device_id, {}).get('packets_received', 0) + 1
        }
    
    def _cleanup_loop(self):
        """Cleanup loop - removes inactive devices."""
        while self.running:
            try:
                current_time = time.time()
                inactive_devices = []
                
                for device_id, info in self.devices.items():
                    if current_time - info['last_seen'] > self.device_timeout:
                        inactive_devices.append(device_id)
                
                for device_id in inactive_devices:
                    logging.info(f"Device {device_id} timed out - removing")
                    del self.devices[device_id]
                    self.stats['connected_devices'].discard(device_id)
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error in cleanup loop: {e}")
    
    def get_connected_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get list of currently connected devices."""
        return self.devices.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        stats = self.stats.copy()
        stats['connected_device_count'] = len(self.devices)
        stats['queue_size'] = self.audio_queue.qsize()
        return stats
    
    def send_audio(self, audio_data: np.ndarray, target_device: Optional[str] = None):
        """
        Send audio data to connected devices (for TTS playback).
        
        Args:
            audio_data: Audio samples as numpy array
            target_device: Specific device ID, or None for all devices
        """
        if not self.socket or not self.running:
            return
        
        # Convert audio to int16 PCM
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
        
        audio_bytes = audio_int16.tobytes()
        
        # Determine target devices
        targets = []
        if target_device and target_device in self.devices:
            targets = [self.devices[target_device]['addr']]
        else:
            targets = [info['addr'] for info in self.devices.values()]
        
        # Send to all target devices
        for addr in targets:
            try:
                self.socket.sendto(audio_bytes, addr)
            except Exception as e:
                logging.debug(f"Failed to send audio to {addr}: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def audio_callback(audio_data):
        """Example callback that just logs audio level."""
        level = np.abs(audio_data).mean()
        if level > 0.01:  # Only log if there's significant audio
            logging.info(f"Received audio: {len(audio_data)} samples, level: {level:.4f}")
    
    # Create and start server
    server = WirelessAudioServer()
    server.set_audio_callback(audio_callback)
    
    if server.start():
        try:
            # Run for testing
            while True:
                time.sleep(5)
                stats = server.get_stats()
                devices = server.get_connected_devices()
                
                logging.info(f"Stats: {stats}")
                logging.info(f"Connected devices: {list(devices.keys())}")
                
        except KeyboardInterrupt:
            logging.info("Shutting down...")
        finally:
            server.stop()
    
    logging.info("Server test complete")