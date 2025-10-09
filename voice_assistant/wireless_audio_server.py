#!/usr/bin/env python3

import socket
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any, Tuple
import struct
import numpy as np
from queue import Queue, Empty
from collections import defaultdict
import binascii

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
    
    Enhanced with comprehensive debugging instrumentation for troubleshooting
    ESP32-P4 audio reception issues including:
    - Detailed packet logging with source addresses
    - UDP header format verification matching ESP32-P4 structure
    - Network interface binding validation
    - Per-device packet counters and diagnostics
    - Raw packet data inspection capabilities
    """
    
    # ESP32-P4 UDP header format - EXACTLY matches device header structure
    # typedef struct {
    #     uint32_t sequence_number;    // 4 bytes
    #     uint16_t sample_count;       // 2 bytes  
    #     uint16_t sample_rate;        // 2 bytes
    #     uint8_t channels;            // 1 byte
    #     uint8_t bits_per_sample;     // 1 byte
    #     uint16_t flags;              // 2 bytes
    #     // Total header: 12 bytes
    #     // Followed by PCM int16 samples
    # } udp_audio_header_t;
    UDP_HEADER_FORMAT = '<I H H B B H'  # Little-endian: uint32, uint16, uint16, uint8, uint8, uint16
    UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)  # Should be exactly 12 bytes
    
    # Validate header size matches ESP32-P4 expectation
    assert UDP_HEADER_SIZE == 12, f"Header size mismatch! Expected 12 bytes, got {UDP_HEADER_SIZE}"
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 8003,
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
        
        # Enhanced statistics and monitoring with debugging instrumentation
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'opus_decode_errors': 0,
            'last_packet_time': 0,
            'connected_devices': set(),
            'total_bytes_received': 0,
            'malformed_packets': 0,
            'valid_headers': 0,
            'invalid_headers': 0,
            'size_mismatch_errors': 0,
            'pcm_extraction_errors': 0,
            'format_validation_errors': 0,
            'start_time': time.time()
        }
        
        # Per-device debugging statistics with comprehensive error categorization
        self.device_debug_stats = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'first_seen': 0,
            'last_seen': 0,
            'last_sequence': None,
            'sequence_gaps': 0,
            'late_packets': 0,
            'valid_headers': 0,
            'invalid_headers': 0,
            'malformed_packets': 0,
            'size_mismatch_errors': 0,
            'pcm_extraction_errors': 0,
            'format_validation_errors': 0,
            'audio_quality_metrics': {
                'sample_rate_consistency': True,
                'format_consistency': True,
                'last_sample_rate': None,
                'last_channels': None,
                'last_bits_per_sample': None,
                'inconsistent_formats_count': 0
            },
            'error_categories': {
                'header_too_small': 0,
                'header_parse_failed': 0,
                'invalid_sample_count': 0,
                'invalid_sample_rate': 0,
                'invalid_channels': 0,
                'invalid_bits_per_sample': 0,
                'payload_size_mismatch': 0,
                'pcm_data_corrupt': 0
            }
        })
        
        # Debug configuration
        self.debug_enabled = True
        self.hex_dump_enabled = False
        self.packet_log_interval = 100  # Log every N packets
        
        # Device registry
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.device_timeout = 10.0  # seconds
        
        logging.info(f"WirelessAudioServer initialized for {host}:{port}")
        logging.info(f"Debug instrumentation enabled - UDP header format: {self.UDP_HEADER_FORMAT} ({self.UDP_HEADER_SIZE} bytes)")
    
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
        """Main receive loop with comprehensive debugging - runs in separate thread."""
        logging.info("Audio receive loop started with debug instrumentation")
        logging.info(f"Listening on {self.host}:{self.port} for ESP32-P4 audio packets")
        
        packet_count = 0
        
        while self.running:
            try:
                if not self.socket:
                    break
                
                # Receive UDP packet
                data, addr = self.socket.recvfrom(2048)  # Max OPUS packet size
                packet_count += 1
                
                # Update global statistics
                self.stats['packets_received'] += 1
                self.stats['total_bytes_received'] += len(data)
                self.stats['last_packet_time'] = time.time()
                
                # Device identification and debug tracking
                device_id = f"{addr[0]}:{addr[1]}"
                
                # Parse and validate UDP header for debugging
                valid_header, header_info = self._parse_udp_header(data)
                
                # Update device debug statistics
                self._update_device_debug_stats(device_id, addr, len(data), header_info)
                
                # Debug logging based on interval
                if self.debug_enabled and (packet_count % self.packet_log_interval == 1 or packet_count <= 10):
                    self._log_packet_debug(device_id, data, header_info, packet_count)
                
                # Register/update device
                self._register_device(device_id, addr)
                
                # Queue packet for processing
                try:
                    packet_info = {
                        'data': data,
                        'addr': addr,
                        'timestamp': time.time(),
                        'header_info': header_info,
                        'valid_header': valid_header
                    }
                    self.audio_queue.put_nowait(packet_info)
                except:
                    self.stats['packets_dropped'] += 1
                    logging.warning(f"Dropped packet from {device_id} - queue full")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Error in receive loop: {e}")
                break
        
        # Log final debug statistics
        if self.debug_enabled:
            self._log_final_debug_stats()
        
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
                    # Pass audio data, raw packet data, and source address for ESP32-P4 processing
                    try:
                        # Try new callback signature with packet data
                        self.audio_callback(audio_array, packet_info['data'], packet_info['addr'])
                    except TypeError:
                        # Fallback to old callback signature for backward compatibility
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
        """Get comprehensive server statistics including debug information."""
        stats = self.stats.copy()
        stats['connected_device_count'] = len(self.devices)
        stats['queue_size'] = self.audio_queue.qsize()
        
        # Add runtime statistics
        runtime = time.time() - self.stats['start_time']
        stats['runtime_seconds'] = runtime
        
        if runtime > 0:
            stats['packets_per_second'] = self.stats['packets_received'] / runtime
            stats['bytes_per_second'] = self.stats['total_bytes_received'] / runtime
        else:
            stats['packets_per_second'] = 0
            stats['bytes_per_second'] = 0
        
        # Add device debug statistics
        stats['device_debug_stats'] = dict(self.device_debug_stats)
        
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
    
    def _parse_udp_header(self, data: bytes) -> Tuple[bool, Dict[str, Any]]:
        """Parse ESP32-P4 UDP audio header with precise validation matching device structure."""
        # Validate minimum packet size for header
        if len(data) < self.UDP_HEADER_SIZE:
            self.stats['malformed_packets'] += 1
            return False, {
                'error': f'Packet too small for header: {len(data)} < {self.UDP_HEADER_SIZE}',
                'packet_size': len(data),
                'error_category': 'header_too_small'
            }
        
        try:
            # Unpack header exactly matching ESP32-P4 structure:
            # uint32_t sequence_number, uint16_t sample_count, uint16_t sample_rate,
            # uint8_t channels, uint8_t bits_per_sample, uint16_t flags
            header = struct.unpack(self.UDP_HEADER_FORMAT, data[:self.UDP_HEADER_SIZE])
            
            header_info = {
                'sequence_number': header[0],
                'sample_count': header[1],
                'sample_rate': header[2],
                'channels': header[3],
                'bits_per_sample': header[4],
                'flags': header[5],
                'header_size': self.UDP_HEADER_SIZE,
                'payload_size': len(data) - self.UDP_HEADER_SIZE,
                'total_size': len(data)
            }
            
            # Comprehensive header validation
            is_valid, validation_msg, error_category = self._validate_header_values(header_info)
            if is_valid:
                self.stats['valid_headers'] += 1
                # Extract and validate PCM samples
                pcm_valid, pcm_info = self._validate_and_extract_pcm_samples(data, header_info)
                header_info.update(pcm_info)
                if not pcm_valid:
                    is_valid = False
                    validation_msg = pcm_info.get('error', 'PCM extraction failed')
                    error_category = pcm_info.get('error_category', 'pcm_extraction_error')
            else:
                self.stats['invalid_headers'] += 1
                header_info['validation_error'] = validation_msg
                header_info['error_category'] = error_category
            
            return is_valid, header_info
            
        except struct.error as e:
            self.stats['malformed_packets'] += 1
            return False, {
                'error': f'Header parsing failed: {e}',
                'packet_size': len(data),
                'error_category': 'header_parse_failed'
            }
    
    def _validate_header_values(self, header: Dict[str, Any]) -> Tuple[bool, str, str]:
        """Validate ESP32-P4 audio header values with detailed error categorization."""
        # Sample count validation - ESP32-P4 typically sends 160-480 samples per packet
        if not (1 <= header['sample_count'] <= 1024):
            return False, f"Invalid sample_count: {header['sample_count']} (expected 1-1024)", 'invalid_sample_count'
        
        # Sample rate validation - common rates for ESP32-P4
        valid_sample_rates = [8000, 16000, 22050, 44100, 48000]
        if header['sample_rate'] not in valid_sample_rates:
            return False, f"Invalid sample_rate: {header['sample_rate']} (expected one of {valid_sample_rates})", 'invalid_sample_rate'
        
        # Channels validation - ESP32-P4 typically mono or stereo
        if not (1 <= header['channels'] <= 2):
            return False, f"Invalid channels: {header['channels']} (expected 1-2)", 'invalid_channels'
        
        # Bits per sample validation - ESP32-P4 typically 16-bit PCM
        valid_bits = [16, 24, 32]
        if header['bits_per_sample'] not in valid_bits:
            return False, f"Invalid bits_per_sample: {header['bits_per_sample']} (expected one of {valid_bits})", 'invalid_bits_per_sample'
        
        # Calculate expected payload size for PCM data
        bytes_per_sample = header['bits_per_sample'] // 8
        expected_payload = header['sample_count'] * header['channels'] * bytes_per_sample
        actual_payload = header['payload_size']
        
        # Allow minimal tolerance for alignment/padding (ESP32-P4 should be exact)
        tolerance = 4  # Allow up to 4 bytes difference for alignment
        if abs(expected_payload - actual_payload) > tolerance:
            self.stats['size_mismatch_errors'] += 1
            return False, (f"Payload size mismatch: expected {expected_payload} bytes "
                         f"({header['sample_count']} samples × {header['channels']} channels × {bytes_per_sample} bytes/sample), "
                         f"got {actual_payload} bytes (difference: {actual_payload - expected_payload})"), 'payload_size_mismatch'
        
        return True, "Valid header", 'valid'
    
    def _update_device_debug_stats(self, device_id: str, addr: Tuple[str, int], packet_size: int, header_info: Dict[str, Any]):
        """Update per-device debugging statistics with comprehensive error categorization."""
        device = self.device_debug_stats[device_id]
        current_time = time.time()
        
        # Basic counters
        device['packet_count'] += 1
        device['byte_count'] += packet_size
        device['last_seen'] = current_time
        
        if device['first_seen'] == 0:
            device['first_seen'] = current_time
            logging.info(f"🔌 ESP32-P4 device connected: {device_id} ({addr[0]}:{addr[1]})")
        
        # Sequence number tracking with late packet detection
        if 'sequence_number' in header_info and 'error' not in header_info:
            seq_num = header_info['sequence_number']
            if device['last_sequence'] is not None:
                expected = (device['last_sequence'] + 1) & 0xFFFFFFFF  # Handle 32-bit wraparound
                if seq_num != expected:
                    if seq_num < device['last_sequence']:  # Late packet
                        device['late_packets'] += 1
                        if self.debug_enabled:
                            logging.warning(f"⏰ {device_id}: Late packet detected! Expected >= {expected}, got {seq_num}")
                    else:  # Gap in sequence
                        gaps = seq_num - expected
                        device['sequence_gaps'] += gaps
                        if self.debug_enabled:
                            logging.warning(f"📊 {device_id}: Sequence gap detected! Expected {expected}, got {seq_num} (gap: {gaps})")
            device['last_sequence'] = seq_num
        
        # Error categorization tracking
        if 'error' not in header_info:
            if 'validation_error' not in header_info:
                device['valid_headers'] += 1
                # Track audio quality metrics for format consistency
                self._update_audio_quality_metrics(device, header_info)
            else:
                device['invalid_headers'] += 1
                # Track specific error categories
                error_category = header_info.get('error_category', 'unknown')
                if error_category in device['error_categories']:
                    device['error_categories'][error_category] += 1
                
                # Track global error counters
                if error_category == 'payload_size_mismatch':
                    device['size_mismatch_errors'] += 1
                elif 'pcm' in error_category:
                    device['pcm_extraction_errors'] += 1
                else:
                    device['format_validation_errors'] += 1
        else:
            device['malformed_packets'] += 1
            # Track malformed packet error categories
            error_category = header_info.get('error_category', 'unknown')
            if error_category in device['error_categories']:
                device['error_categories'][error_category] += 1
    
    def _log_packet_debug(self, device_id: str, data: bytes, header_info: Dict[str, Any], packet_count: int):
        """Log detailed packet debugging information."""
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]
        
        if 'error' in header_info:
            logging.error(f"❌ [{timestamp}] MALFORMED packet #{packet_count} from {device_id}: "
                         f"size={len(data)}B, error={header_info['error']}")
        elif 'validation_error' in header_info:
            logging.warning(f"⚠️  [{timestamp}] INVALID packet #{packet_count} from {device_id}: "
                          f"{header_info['validation_error']}")
            if self.debug_enabled:
                logging.warning(f"    Header details: seq={header_info.get('sequence_number', 'N/A')}, "
                              f"samples={header_info.get('sample_count', 'N/A')}, "
                              f"rate={header_info.get('sample_rate', 'N/A')}Hz, "
                              f"payload={header_info.get('payload_size', 'N/A')}B")
        else:
            # Enhanced valid packet logging with PCM analysis
            pcm_info = ""
            if 'pcm_samples' in header_info:
                pcm_info = f", PCM_amp={header_info['amplitude_ratio']:.3f}"
            elif 'pcm_payload_raw' in header_info:
                pcm_info = f", PCM_raw={header_info['pcm_payload_size']}B"
            elif header_info.get('sample_data_valid', False):
                pcm_info = f", PCM_ok"
            
            logging.info(f"📦 [{timestamp}] VALID packet #{packet_count} from {device_id}: "
                        f"seq={header_info['sequence_number']}, "
                        f"samples={header_info['sample_count']}, "
                        f"rate={header_info['sample_rate']}Hz, "
                        f"ch={header_info['channels']}, "
                        f"bits={header_info['bits_per_sample']}, "
                        f"flags=0x{header_info['flags']:04x}, "
                        f"payload={header_info['payload_size']}B{pcm_info}")
        
        # Optional hex dump for deep debugging
        if self.hex_dump_enabled:
            self._log_hex_dump(device_id, data)
    
    def _log_hex_dump(self, device_id: str, data: bytes, max_bytes: int = 64):
        """Log hex dump of packet data for deep debugging."""
        dump_size = min(len(data), max_bytes)
        hex_str = binascii.hexlify(data[:dump_size]).decode('ascii')
        
        logging.info(f"🔍 Hex dump of packet from {device_id}:")
        # Format as 16 bytes per line
        for i in range(0, len(hex_str), 32):
            line = hex_str[i:i+32]
            formatted = ' '.join(line[j:j+2] for j in range(0, len(line), 2))
            offset = i // 2
            logging.info(f"    {offset:04x}: {formatted}")
        
        if dump_size < len(data):
            logging.info(f"    ... ({len(data) - dump_size} more bytes)")
    
    def _log_final_debug_stats(self):
        """Log comprehensive final debugging statistics with enhanced error categorization."""
        runtime = time.time() - self.stats['start_time']
        
        logging.info("=" * 80)
        logging.info("📊 WIRELESS AUDIO SERVER DEBUG STATISTICS")
        logging.info("=" * 80)
        logging.info(f"Runtime: {runtime:.1f}s")
        logging.info(f"Total packets: {self.stats['packets_received']}")
        logging.info(f"Total bytes: {self.stats['total_bytes_received']:,}")
        
        if runtime > 0:
            logging.info(f"Average rate: {self.stats['packets_received']/runtime:.1f} pkt/s, {self.stats['total_bytes_received']/runtime:.0f} B/s")
        
        # Enhanced global statistics
        logging.info(f"Valid headers: {self.stats['valid_headers']}")
        logging.info(f"Invalid headers: {self.stats['invalid_headers']}")
        logging.info(f"Malformed packets: {self.stats['malformed_packets']}")
        logging.info(f"Dropped packets: {self.stats['packets_dropped']}")
        logging.info(f"Size mismatch errors: {self.stats['size_mismatch_errors']}")
        logging.info(f"PCM extraction errors: {self.stats['pcm_extraction_errors']}")
        logging.info(f"Format validation errors: {self.stats['format_validation_errors']}")
        
        if self.device_debug_stats:
            logging.info("")
            logging.info("📱 PER-DEVICE DEBUG STATISTICS WITH COMPREHENSIVE ERROR ANALYSIS:")
            for device_id, device in self.device_debug_stats.items():
                device_runtime = device['last_seen'] - device['first_seen']
                device_rate = device['packet_count'] / device_runtime if device_runtime > 0 else 0
                
                logging.info(f"  🔹 {device_id}:")
                logging.info(f"    📦 Packets: {device['packet_count']} ({device_rate:.1f} pkt/s)")
                logging.info(f"    💾 Bytes: {device['byte_count']:,}")
                logging.info(f"    ✅ Valid headers: {device['valid_headers']}")
                logging.info(f"    ❌ Invalid headers: {device['invalid_headers']}")
                logging.info(f"    ⚠️  Malformed packets: {device['malformed_packets']}")
                logging.info(f"    📊 Sequence gaps: {device['sequence_gaps']}")
                logging.info(f"    ⏰ Late packets: {device['late_packets']}")
                logging.info(f"    🔢 Last sequence: {device['last_sequence']}")
                
                # Audio quality metrics
                quality = device['audio_quality_metrics']
                logging.info(f"    🎵 Audio Quality:")
                logging.info(f"      Sample rate consistent: {quality['sample_rate_consistency']}")
                logging.info(f"      Format consistent: {quality['format_consistency']}")
                logging.info(f"      Last format: {quality['last_sample_rate']}Hz, {quality['last_channels']}ch, {quality['last_bits_per_sample']}bit")
                logging.info(f"      Inconsistent formats: {quality['inconsistent_formats_count']}")
                
                # Error categorization
                if any(device['error_categories'].values()):
                    logging.info(f"    🐛 Error Categories:")
                    for error_type, count in device['error_categories'].items():
                        if count > 0:
                            logging.info(f"      {error_type}: {count}")
        
        logging.info("=" * 80)
    
    def _validate_and_extract_pcm_samples(self, data: bytes, header_info: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate and extract PCM samples from UDP packet after 12-byte header."""
        try:
            # Extract PCM payload after header
            pcm_payload = data[self.UDP_HEADER_SIZE:]
            
            # Calculate expected PCM data size
            sample_count = header_info['sample_count']
            channels = header_info['channels']
            bits_per_sample = header_info['bits_per_sample']
            bytes_per_sample = bits_per_sample // 8
            
            expected_pcm_size = sample_count * channels * bytes_per_sample
            actual_pcm_size = len(pcm_payload)
            
            # Validate PCM payload size exactly matches header
            if actual_pcm_size != expected_pcm_size:
                self.stats['pcm_extraction_errors'] += 1
                return False, {
                    'error': f'PCM payload size mismatch: expected {expected_pcm_size}, got {actual_pcm_size}',
                    'error_category': 'pcm_size_mismatch',
                    'expected_pcm_size': expected_pcm_size,
                    'actual_pcm_size': actual_pcm_size
                }
            
            # For 16-bit PCM (most common), validate sample data integrity
            if bits_per_sample == 16:
                try:
                    # Unpack PCM samples as int16 (little-endian)
                    pcm_format = f'<{sample_count * channels}h'  # h = signed short (16-bit)
                    pcm_samples = struct.unpack(pcm_format, pcm_payload)
                    
                    # Basic PCM validation - check for reasonable amplitude range
                    max_amplitude = max(abs(sample) for sample in pcm_samples)
                    
                    return True, {
                        'pcm_samples': pcm_samples,
                        'pcm_payload_size': actual_pcm_size,
                        'max_amplitude': max_amplitude,
                        'amplitude_ratio': max_amplitude / 32767.0 if max_amplitude > 0 else 0.0,
                        'sample_data_valid': True
                    }
                    
                except struct.error as e:
                    self.stats['pcm_extraction_errors'] += 1
                    return False, {
                        'error': f'PCM sample unpacking failed: {e}',
                        'error_category': 'pcm_data_corrupt',
                        'pcm_payload_size': actual_pcm_size
                    }
            else:
                # For non-16-bit formats, just validate size and return raw data
                return True, {
                    'pcm_payload_raw': pcm_payload,
                    'pcm_payload_size': actual_pcm_size,
                    'sample_data_valid': True,
                    'note': f'{bits_per_sample}-bit PCM not unpacked (not 16-bit)'
                }
                
        except Exception as e:
            self.stats['pcm_extraction_errors'] += 1
            return False, {
                'error': f'PCM extraction failed: {e}',
                'error_category': 'pcm_extraction_error'
            }
    
    def _update_audio_quality_metrics(self, device_stats: Dict[str, Any], header_info: Dict[str, Any]):
        """Update audio quality metrics for format consistency validation."""
        quality = device_stats['audio_quality_metrics']
        
        current_sample_rate = header_info['sample_rate']
        current_channels = header_info['channels']
        current_bits = header_info['bits_per_sample']
        
        # Check for format consistency
        if quality['last_sample_rate'] is not None:
            if (quality['last_sample_rate'] != current_sample_rate or
                quality['last_channels'] != current_channels or
                quality['last_bits_per_sample'] != current_bits):
                
                quality['sample_rate_consistency'] = False
                quality['format_consistency'] = False
                quality['inconsistent_formats_count'] += 1
                
                if self.debug_enabled:
                    logging.warning(f"🎵 Format change detected! "
                                  f"Previous: {quality['last_sample_rate']}Hz, {quality['last_channels']}ch, {quality['last_bits_per_sample']}bit "
                                  f"Current: {current_sample_rate}Hz, {current_channels}ch, {current_bits}bit")
        
        # Update last seen format
        quality['last_sample_rate'] = current_sample_rate
        quality['last_channels'] = current_channels
        quality['last_bits_per_sample'] = current_bits
    
    def enable_debug_logging(self, enable: bool = True, hex_dump: bool = False, packet_interval: int = 100):
        """Enable or disable debug logging with optional hex dumps."""
        self.debug_enabled = enable
        self.hex_dump_enabled = hex_dump
        self.packet_log_interval = packet_interval
        
        if enable:
            logging.info(f"📝 Debug logging enabled (hex_dump={hex_dump}, interval={packet_interval})")
        else:
            logging.info("📝 Debug logging disabled")
    
    def get_debug_stats(self) -> Dict[str, Any]:
        """Get detailed debugging statistics."""
        return {
            'global_stats': self.stats.copy(),
            'device_stats': dict(self.device_debug_stats),
            'runtime': time.time() - self.stats['start_time']
        }
    
    def print_debug_summary(self):
        """Print a summary of current debugging statistics."""
        if not self.debug_enabled:
            logging.info("Debug logging is disabled. Enable with enable_debug_logging()")
            return
        
        self._log_final_debug_stats()
    
    def get_comprehensive_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive packet validation statistics for analysis."""
        runtime = time.time() - self.stats['start_time']
        
        # Calculate success rates
        total_packets = self.stats['packets_received']
        success_rate = self.stats['valid_headers'] / total_packets if total_packets > 0 else 0.0
        
        validation_stats = {
            'runtime_seconds': runtime,
            'total_packets_received': total_packets,
            'packet_validation_summary': {
                'valid_headers': self.stats['valid_headers'],
                'invalid_headers': self.stats['invalid_headers'],
                'malformed_packets': self.stats['malformed_packets'],
                'success_rate_percentage': success_rate * 100,
                'error_rate_percentage': (1 - success_rate) * 100
            },
            'error_breakdown': {
                'size_mismatch_errors': self.stats['size_mismatch_errors'],
                'pcm_extraction_errors': self.stats['pcm_extraction_errors'],
                'format_validation_errors': self.stats['format_validation_errors'],
                'packets_dropped': self.stats['packets_dropped']
            },
            'performance_metrics': {
                'packets_per_second': self.stats['packets_received'] / runtime if runtime > 0 else 0,
                'bytes_per_second': self.stats['total_bytes_received'] / runtime if runtime > 0 else 0,
                'average_packet_size': self.stats['total_bytes_received'] / total_packets if total_packets > 0 else 0
            },
            'device_summaries': {}
        }
        
        # Add per-device validation summaries
        for device_id, device_stats in self.device_debug_stats.items():
            device_packets = device_stats['packet_count']
            device_success_rate = device_stats['valid_headers'] / device_packets if device_packets > 0 else 0.0
            
            validation_stats['device_summaries'][device_id] = {
                'packet_count': device_packets,
                'success_rate_percentage': device_success_rate * 100,
                'sequence_gaps': device_stats['sequence_gaps'],
                'late_packets': device_stats['late_packets'],
                'audio_quality': device_stats['audio_quality_metrics'],
                'error_categories': {k: v for k, v in device_stats['error_categories'].items() if v > 0}
            }
        
        return validation_stats
    
    def validate_esp32_p4_compatibility(self) -> Dict[str, Any]:
        """Validate server configuration for ESP32-P4 compatibility."""
        compatibility_report = {
            'header_format_validation': {
                'format_string': self.UDP_HEADER_FORMAT,
                'header_size_bytes': self.UDP_HEADER_SIZE,
                'expected_size': 12,
                'format_correct': self.UDP_HEADER_FORMAT == '<I H H B B H',
                'size_correct': self.UDP_HEADER_SIZE == 12
            },
            'esp32_p4_requirements': {
                'little_endian_format': self.UDP_HEADER_FORMAT.startswith('<'),
                'sequence_number_uint32': 'I' in self.UDP_HEADER_FORMAT,
                'sample_count_uint16': 'H' in self.UDP_HEADER_FORMAT,
                'sample_rate_uint16': self.UDP_HEADER_FORMAT.count('H') >= 2,
                'channels_uint8': 'B' in self.UDP_HEADER_FORMAT,
                'bits_per_sample_uint8': self.UDP_HEADER_FORMAT.count('B') >= 2
            },
            'validation_capabilities': {
                'pcm_sample_extraction': hasattr(self, '_validate_and_extract_pcm_samples'),
                'audio_quality_metrics': hasattr(self, '_update_audio_quality_metrics'),
                'comprehensive_error_categorization': 'error_categories' in str(self.device_debug_stats),
                'sequence_gap_detection': True,
                'late_packet_detection': True
            }
        }
        
        # Overall compatibility score
        format_checks = list(compatibility_report['header_format_validation'].values())
        requirement_checks = list(compatibility_report['esp32_p4_requirements'].values())
        capability_checks = list(compatibility_report['validation_capabilities'].values())
        
        all_checks = format_checks[2:] + requirement_checks + capability_checks  # Skip format_string and header_size_bytes
        compatibility_score = sum(all_checks) / len(all_checks) * 100
        
        compatibility_report['overall_compatibility_score'] = compatibility_score
        compatibility_report['esp32_p4_ready'] = compatibility_score >= 95.0
        
        return compatibility_report


# Example usage and testing with debug instrumentation
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments for debug testing
    parser = argparse.ArgumentParser(description='HowdyTTS Wireless Audio Server with Debug Instrumentation')
    parser.add_argument('--port', type=int, default=8003, help='UDP port to listen on (default: 8003)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug logging')
    parser.add_argument('--hex-dump', action='store_true', help='Enable hex dump of packets')
    parser.add_argument('--packet-interval', type=int, default=100, 
                       help='Log every N packets (default: 100)')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    def audio_callback(audio_data, raw_packet_data=None, source_addr=None):
        """Enhanced callback with packet debugging information."""
        level = np.abs(audio_data).mean()
        if level > 0.01:  # Only log if there's significant audio
            if source_addr:
                logging.info(f"🎵 Audio from {source_addr[0]}:{source_addr[1]}: "
                           f"{len(audio_data)} samples, level: {level:.4f}")
            else:
                logging.info(f"🎵 Received audio: {len(audio_data)} samples, level: {level:.4f}")
    
    # Create and start server with debug configuration
    server = WirelessAudioServer(port=args.port)
    server.set_audio_callback(audio_callback)
    
    # Enable debug logging based on command line arguments
    server.enable_debug_logging(
        enable=True, 
        hex_dump=args.hex_dump, 
        packet_interval=args.packet_interval
    )
    
    logging.info(f"🚀 Starting HowdyTTS Wireless Audio Server with debug instrumentation")
    logging.info(f"📡 Listening on 0.0.0.0:{args.port} for ESP32-P4 audio packets")
    logging.info(f"📝 Debug mode: verbose={args.verbose}, hex_dump={args.hex_dump}, interval={args.packet_interval}")
    logging.info(f"🔍 Waiting for ESP32-P4 devices... (Press Ctrl+C to stop)")
    
    if server.start():
        try:
            stats_interval = 10.0  # Print stats every 10 seconds
            last_stats_time = time.time()
            
            # Run for testing with periodic statistics
            while True:
                time.sleep(1)
                
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    stats = server.get_stats()
                    devices = server.get_connected_devices()
                    
                    if stats['packets_received'] > 0:
                        logging.info("=" * 60)
                        logging.info(f"📊 PERIODIC STATS SUMMARY:")
                        logging.info(f"   Total packets: {stats['packets_received']}")
                        logging.info(f"   Total bytes: {stats['total_bytes_received']:,}")
                        logging.info(f"   Packet rate: {stats.get('packets_per_second', 0):.1f} pkt/s")
                        logging.info(f"   Valid headers: {stats['valid_headers']}")
                        logging.info(f"   Invalid headers: {stats['invalid_headers']}")
                        logging.info(f"   Connected devices: {list(devices.keys())}")
                        logging.info("=" * 60)
                    else:
                        logging.info("⏳ No packets received yet. Waiting for ESP32-P4 devices...")
                    
                    last_stats_time = current_time
                
        except KeyboardInterrupt:
            logging.info("\n🛑 Shutdown requested by user")
        finally:
            server.stop()
    else:
        logging.error("❌ Failed to start server")
    
    logging.info("✅ Server test complete")