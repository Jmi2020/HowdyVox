#!/usr/bin/env python3
"""
Standalone UDP Debug Receiver for HowdyTTS ESP32-P4 Audio Debugging

This minimal UDP receiver script helps debug why the HowdyTTS server 
is not receiving audio packets from ESP32-P4 devices. It provides:
- Comprehensive packet logging with source addresses
- Header format verification matching ESP32-P4 structure  
- Network interface binding validation
- Per-device packet counters and diagnostics
- Raw packet data inspection for troubleshooting

Usage:
    python udp_debug_receiver.py [--port 8003] [--verbose] [--hex-dump]
    
UDP Header Format from ESP32-P4:
    typedef struct {
        uint32_t sequence_number;
        uint16_t sample_count; 
        uint16_t sample_rate;
        uint8_t channels;
        uint8_t bits_per_sample;
        uint16_t flags;
    } udp_audio_header_t;  // Total: 12 bytes
"""

import socket
import struct
import time
import argparse
import logging
from collections import defaultdict
from typing import Dict, Any, Tuple
import binascii

class UDPDebugReceiver:
    """Minimal UDP receiver with comprehensive debugging for ESP32-P4 audio packets."""
    
    # ESP32-P4 UDP header format: <I H H B B H (little-endian)
    UDP_HEADER_FORMAT = '<I H H B B H'
    UDP_HEADER_SIZE = struct.calcsize(UDP_HEADER_FORMAT)
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8003, verbose: bool = False, hex_dump: bool = False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.hex_dump = hex_dump
        self.socket = None
        self.running = False
        
        # Statistics tracking
        self.stats = {
            'total_packets': 0,
            'total_bytes': 0,
            'start_time': time.time(),
            'last_packet_time': 0,
            'devices': defaultdict(lambda: {
                'packet_count': 0,
                'byte_count': 0,
                'first_seen': 0,
                'last_seen': 0,
                'last_sequence': None,
                'sequence_gaps': 0,
                'valid_headers': 0,
                'invalid_headers': 0
            })
        }
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def parse_udp_header(self, data: bytes) -> Tuple[bool, Dict[str, Any]]:
        """Parse ESP32-P4 UDP audio header."""
        if len(data) < self.UDP_HEADER_SIZE:
            return False, {'error': f'Packet too small: {len(data)} < {self.UDP_HEADER_SIZE}'}
        
        try:
            # Unpack header: sequence_number, sample_count, sample_rate, channels, bits_per_sample, flags
            header = struct.unpack(self.UDP_HEADER_FORMAT, data[:self.UDP_HEADER_SIZE])
            
            return True, {
                'sequence_number': header[0],
                'sample_count': header[1],
                'sample_rate': header[2],
                'channels': header[3],
                'bits_per_sample': header[4],
                'flags': header[5],
                'header_size': self.UDP_HEADER_SIZE,
                'payload_size': len(data) - self.UDP_HEADER_SIZE
            }
        except struct.error as e:
            return False, {'error': f'Header parsing failed: {e}'}
    
    def validate_header(self, header: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate ESP32-P4 audio header values."""
        if 'error' in header:
            return False, header['error']
        
        # Reasonable validation ranges
        if not (1 <= header['sample_count'] <= 1024):
            return False, f"Invalid sample_count: {header['sample_count']}"
        
        if header['sample_rate'] not in [8000, 16000, 22050, 44100, 48000]:
            return False, f"Invalid sample_rate: {header['sample_rate']}"
        
        if not (1 <= header['channels'] <= 2):
            return False, f"Invalid channels: {header['channels']}"
        
        if header['bits_per_sample'] not in [16, 24, 32]:
            return False, f"Invalid bits_per_sample: {header['bits_per_sample']}"
        
        # Calculate expected payload size
        expected_payload = header['sample_count'] * header['channels'] * (header['bits_per_sample'] // 8)
        actual_payload = header['payload_size']
        
        if abs(expected_payload - actual_payload) > 64:  # Allow some tolerance
            return False, f"Payload size mismatch: expected {expected_payload}, got {actual_payload}"
        
        return True, "Valid header"
    
    def print_hex_dump(self, data: bytes, max_bytes: int = 64):
        """Print hex dump of packet data."""
        dump_size = min(len(data), max_bytes)
        hex_str = binascii.hexlify(data[:dump_size]).decode('ascii')
        
        # Format as 16 bytes per line
        for i in range(0, len(hex_str), 32):
            line = hex_str[i:i+32]
            formatted = ' '.join(line[j:j+2] for j in range(0, len(line), 2))
            offset = i // 2
            self.logger.info(f"    {offset:04x}: {formatted}")
        
        if dump_size < len(data):
            self.logger.info(f"    ... ({len(data) - dump_size} more bytes)")
    
    def update_device_stats(self, device_id: str, addr: Tuple[str, int], packet_size: int, header: Dict[str, Any]):
        """Update per-device statistics."""
        device = self.stats['devices'][device_id]
        current_time = time.time()
        
        # Basic counters
        device['packet_count'] += 1
        device['byte_count'] += packet_size
        device['last_seen'] = current_time
        
        if device['first_seen'] == 0:
            device['first_seen'] = current_time
            self.logger.info(f"🔌 New device detected: {device_id} ({addr[0]}:{addr[1]})")
        
        # Sequence number tracking
        if 'sequence_number' in header:
            seq_num = header['sequence_number']
            if device['last_sequence'] is not None:
                expected = (device['last_sequence'] + 1) & 0xFFFFFFFF  # Handle 32-bit wraparound
                if seq_num != expected:
                    device['sequence_gaps'] += 1
                    if self.verbose:
                        self.logger.warning(f"📊 {device_id}: Sequence gap! Expected {expected}, got {seq_num}")
            device['last_sequence'] = seq_num
        
        # Header validation tracking
        if 'error' not in header:
            device['valid_headers'] += 1
        else:
            device['invalid_headers'] += 1
    
    def print_statistics(self):
        """Print comprehensive statistics."""
        current_time = time.time()
        runtime = current_time - self.stats['start_time']
        
        self.logger.info("=" * 80)
        self.logger.info("📊 UDP DEBUG RECEIVER STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Runtime: {runtime:.1f}s")
        self.logger.info(f"Total packets: {self.stats['total_packets']}")
        self.logger.info(f"Total bytes: {self.stats['total_bytes']:,}")
        self.logger.info(f"Average rate: {self.stats['total_packets']/runtime:.1f} pkt/s, {self.stats['total_bytes']/runtime:.0f} B/s")
        
        if self.stats['last_packet_time'] > 0:
            last_ago = current_time - self.stats['last_packet_time']
            self.logger.info(f"Last packet: {last_ago:.1f}s ago")
        
        self.logger.info("")
        self.logger.info("📱 DEVICE STATISTICS:")
        for device_id, device in self.stats['devices'].items():
            device_runtime = device['last_seen'] - device['first_seen']
            if device_runtime > 0:
                device_rate = device['packet_count'] / device_runtime
            else:
                device_rate = 0
            
            self.logger.info(f"  🔹 {device_id}:")
            self.logger.info(f"    Packets: {device['packet_count']} ({device_rate:.1f} pkt/s)")
            self.logger.info(f"    Bytes: {device['byte_count']:,}")
            self.logger.info(f"    Valid headers: {device['valid_headers']}")
            self.logger.info(f"    Invalid headers: {device['invalid_headers']}")
            self.logger.info(f"    Sequence gaps: {device['sequence_gaps']}")
            self.logger.info(f"    Last sequence: {device['last_sequence']}")
        self.logger.info("=" * 80)
    
    def start(self) -> bool:
        """Start the UDP debug receiver."""
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set larger buffer for debug capture
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB buffer
            
            # Bind to address with detailed logging
            self.logger.info(f"🌐 Binding UDP socket to {self.host}:{self.port}")
            self.socket.bind((self.host, self.port))
            
            # Set timeout for periodic stats
            self.socket.settimeout(5.0)
            
            self.running = True
            self.logger.info(f"✅ UDP Debug Receiver started successfully")
            self.logger.info(f"📡 Listening for ESP32-P4 audio packets on {self.host}:{self.port}")
            self.logger.info(f"📝 Header format: {self.UDP_HEADER_FORMAT} ({self.UDP_HEADER_SIZE} bytes)")
            self.logger.info("🔍 Waiting for packets... (Press Ctrl+C to stop)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start UDP receiver: {e}")
            return False
    
    def run(self):
        """Main packet receiving loop."""
        if not self.start():
            return
        
        last_stats_time = time.time()
        stats_interval = 10.0  # Print stats every 10 seconds
        
        try:
            while self.running:
                try:
                    # Receive UDP packet
                    data, addr = self.socket.recvfrom(2048)
                    
                    # Update global statistics
                    self.stats['total_packets'] += 1
                    self.stats['total_bytes'] += len(data)
                    self.stats['last_packet_time'] = time.time()
                    
                    # Device identification
                    device_id = f"{addr[0]}:{addr[1]}"
                    
                    # Parse UDP header
                    valid_header, header = self.parse_udp_header(data)
                    
                    # Update device statistics
                    self.update_device_stats(device_id, addr, len(data), header)
                    
                    # Log packet reception
                    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
                    if valid_header and 'error' not in header:
                        # Validate header values
                        is_valid, validation_msg = self.validate_header(header)
                        
                        if is_valid:
                            self.logger.info(f"📦 [{timestamp}] VALID packet from {device_id}: "
                                           f"seq={header['sequence_number']}, "
                                           f"samples={header['sample_count']}, "
                                           f"rate={header['sample_rate']}Hz, "
                                           f"ch={header['channels']}, "
                                           f"bits={header['bits_per_sample']}, "
                                           f"flags=0x{header['flags']:04x}, "
                                           f"payload={header['payload_size']}B")
                        else:
                            self.logger.warning(f"⚠️  [{timestamp}] INVALID packet from {device_id}: {validation_msg}")
                            if self.verbose:
                                self.logger.warning(f"    Header: {header}")
                    else:
                        self.logger.error(f"❌ [{timestamp}] MALFORMED packet from {device_id}: "
                                        f"size={len(data)}B, error={header.get('error', 'Unknown')}")
                    
                    # Hex dump if requested
                    if self.hex_dump:
                        self.logger.info(f"🔍 Hex dump of packet from {device_id}:")
                        self.print_hex_dump(data)
                    
                    # Periodic statistics
                    current_time = time.time()
                    if current_time - last_stats_time >= stats_interval:
                        self.print_statistics()
                        last_stats_time = current_time
                
                except socket.timeout:
                    # Periodic timeout for stats and status check
                    current_time = time.time()
                    if current_time - last_stats_time >= stats_interval:
                        if self.stats['total_packets'] == 0:
                            self.logger.info("⏳ No packets received yet. Listening...")
                        else:
                            self.print_statistics()
                        last_stats_time = current_time
                    continue
                
                except Exception as e:
                    self.logger.error(f"❌ Error receiving packet: {e}")
                    if not self.running:
                        break
        
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Shutdown requested by user")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the UDP debug receiver."""
        self.logger.info("🔄 Stopping UDP Debug Receiver...")
        self.running = False
        
        if self.socket:
            self.socket.close()
            self.socket = None
        
        # Print final statistics
        if self.stats['total_packets'] > 0:
            self.print_statistics()
        else:
            self.logger.info("📊 No packets were received during this session")
        
        self.logger.info("✅ UDP Debug Receiver stopped")

def main():
    """Main function for standalone UDP debug receiver."""
    parser = argparse.ArgumentParser(
        description='HowdyTTS UDP Debug Receiver for ESP32-P4 Audio Debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python udp_debug_receiver.py                    # Listen on default port 8003
  python udp_debug_receiver.py --port 8003        # Explicitly set port
  python udp_debug_receiver.py --verbose          # Detailed logging
  python udp_debug_receiver.py --hex-dump         # Include hex dumps
  python udp_debug_receiver.py --verbose --hex-dump  # Full debugging

Network Interface Binding:
  This receiver binds to 0.0.0.0:8003 by default, which accepts
  packets from any network interface (WiFi, Ethernet, etc.).
  This ensures reception regardless of subnet configuration.

ESP32-P4 Integration:
  The ESP32-P4 device should discover this server automatically
  via UDP broadcast and start sending audio packets to port 8003.
        """
    )
    
    parser.add_argument('--port', type=int, default=8003,
                       help='UDP port to listen on (default: 8003)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address to bind to (default: 0.0.0.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging with detailed packet analysis')
    parser.add_argument('--hex-dump', action='store_true',
                       help='Include hex dump of packet contents (first 64 bytes)')
    
    args = parser.parse_args()
    
    # Create and run debug receiver
    receiver = UDPDebugReceiver(
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        hex_dump=args.hex_dump
    )
    
    receiver.run()

if __name__ == "__main__":
    main()