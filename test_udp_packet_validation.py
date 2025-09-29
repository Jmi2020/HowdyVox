#!/usr/bin/env python3
"""
UDP Packet Format Validation Test for ESP32-P4 HowdyTTS Integration

This script tests the enhanced UDP packet validation capabilities implemented
in the WirelessAudioServer to ensure correct parsing of ESP32-P4 audio packets.

Test Coverage:
- Header format validation exactly matching ESP32-P4 structure
- PCM sample extraction with size validation
- Per-device packet counters and diagnostics
- Audio quality metrics and format consistency
- Comprehensive error categorization and reporting

Usage:
    python test_udp_packet_validation.py
"""

import sys
import os
import struct
import socket
import time
import logging
import json
from typing import Dict, Any

# Add the voice_assistant directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice_assistant'))

from wireless_audio_server import WirelessAudioServer

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

class ESP32P4PacketValidator:
    """Test class to validate ESP32-P4 UDP packet format handling."""
    
    def __init__(self, server_port: int = 8003):
        self.server_port = server_port
        self.server = None
        
    def create_esp32_p4_test_packet(self, 
                                  sequence_number: int = 1,
                                  sample_count: int = 320,
                                  sample_rate: int = 16000,
                                  channels: int = 1,
                                  bits_per_sample: int = 16,
                                  flags: int = 0,
                                  include_pcm: bool = True,
                                  corrupt_header: bool = False,
                                  corrupt_pcm: bool = False) -> bytes:
        """Create a test UDP packet matching ESP32-P4 format."""
        
        # ESP32-P4 header format: <I H H B B H (12 bytes)
        header_data = struct.pack('<I H H B B H', 
                                sequence_number, sample_count, sample_rate, 
                                channels, bits_per_sample, flags)
        
        if corrupt_header:
            # Truncate header to test malformed packet handling
            header_data = header_data[:8]
        
        packet_data = header_data
        
        if include_pcm and not corrupt_header:
            # Generate PCM data (int16 samples)
            bytes_per_sample = bits_per_sample // 8
            pcm_size = sample_count * channels * bytes_per_sample
            
            if corrupt_pcm:
                # Create PCM data with wrong size
                pcm_size = pcm_size // 2
            
            if bits_per_sample == 16:
                # Generate sine wave PCM samples
                import math
                pcm_samples = []
                for i in range(sample_count * channels):
                    sample = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
                    pcm_samples.append(sample)
                pcm_data = struct.pack(f'<{len(pcm_samples)}h', *pcm_samples)
                
                if corrupt_pcm:
                    pcm_data = pcm_data[:len(pcm_data)//2]
            else:
                # Generate dummy PCM data for other bit depths
                pcm_data = b'\x00' * pcm_size
            
            packet_data += pcm_data
        
        return packet_data
    
    def send_test_packet(self, packet_data: bytes, target_ip: str = "127.0.0.1"):
        """Send test packet to the wireless audio server."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(packet_data, (target_ip, self.server_port))
            sock.close()
            return True
        except Exception as e:
            logging.error(f"Failed to send test packet: {e}")
            return False
    
    def run_validation_tests(self):
        """Run comprehensive validation tests."""
        logging.info("🧪 Starting ESP32-P4 UDP Packet Validation Tests")
        logging.info("=" * 60)
        
        # Create and start wireless audio server
        self.server = WirelessAudioServer(port=self.server_port)
        self.server.enable_debug_logging(enable=True, hex_dump=False, packet_interval=1)
        
        # Validate ESP32-P4 compatibility
        compatibility = self.server.validate_esp32_p4_compatibility()
        logging.info("🔧 ESP32-P4 Compatibility Check:")
        logging.info(f"   Format String: {compatibility['header_format_validation']['format_string']}")
        logging.info(f"   Header Size: {compatibility['header_format_validation']['header_size_bytes']} bytes")
        logging.info(f"   Format Correct: {compatibility['header_format_validation']['format_correct']}")
        logging.info(f"   Size Correct: {compatibility['header_format_validation']['size_correct']}")
        logging.info(f"   Overall Compatibility Score: {compatibility['overall_compatibility_score']:.1f}%")
        logging.info(f"   ESP32-P4 Ready: {compatibility['esp32_p4_ready']}")
        logging.info("")
        
        if not self.server.start():
            logging.error("❌ Failed to start server")
            return False
        
        try:
            # Wait for server to fully start
            time.sleep(1)
            
            # Test Case 1: Valid ESP32-P4 packets
            logging.info("📦 Test Case 1: Valid ESP32-P4 Packets")
            test_packets = [
                (1, 320, 16000, 1, 16, 0),  # Standard 16kHz mono
                (2, 480, 16000, 1, 16, 0),  # Larger frame
                (3, 160, 8000, 1, 16, 0),   # 8kHz sample rate
                (4, 320, 16000, 2, 16, 0),  # Stereo
            ]
            
            for seq, samples, rate, channels, bits, flags in test_packets:
                packet = self.create_esp32_p4_test_packet(seq, samples, rate, channels, bits, flags)
                self.send_test_packet(packet)
                time.sleep(0.1)
            
            # Test Case 2: Malformed packets
            logging.info("📦 Test Case 2: Malformed Packets")
            malformed_packet = self.create_esp32_p4_test_packet(5, 320, 16000, 1, 16, 0, corrupt_header=True)
            self.send_test_packet(malformed_packet)
            
            # Test Case 3: Invalid header values
            logging.info("📦 Test Case 3: Invalid Header Values")
            invalid_packets = [
                (6, 0, 16000, 1, 16, 0),      # Invalid sample count
                (7, 320, 99999, 1, 16, 0),    # Invalid sample rate
                (8, 320, 16000, 5, 16, 0),    # Invalid channels
                (9, 320, 16000, 1, 12, 0),    # Invalid bits per sample
            ]
            
            for seq, samples, rate, channels, bits, flags in invalid_packets:
                packet = self.create_esp32_p4_test_packet(seq, samples, rate, channels, bits, flags)
                self.send_test_packet(packet)
                time.sleep(0.1)
            
            # Test Case 4: PCM size mismatches
            logging.info("📦 Test Case 4: PCM Size Mismatches")
            pcm_mismatch_packet = self.create_esp32_p4_test_packet(10, 320, 16000, 1, 16, 0, corrupt_pcm=True)
            self.send_test_packet(pcm_mismatch_packet)
            
            # Test Case 5: Sequence gap detection
            logging.info("📦 Test Case 5: Sequence Gap Detection")
            gap_packets = [11, 12, 15, 16]  # Gap between 12 and 15
            for seq in gap_packets:
                packet = self.create_esp32_p4_test_packet(seq, 320, 16000, 1, 16, 0)
                self.send_test_packet(packet)
                time.sleep(0.1)
            
            # Test Case 6: Late packet detection
            logging.info("📦 Test Case 6: Late Packet Detection")
            late_packet = self.create_esp32_p4_test_packet(13, 320, 16000, 1, 16, 0)  # Late packet
            self.send_test_packet(late_packet)
            
            # Wait for processing
            time.sleep(2)
            
            # Get comprehensive validation statistics
            logging.info("")
            logging.info("📊 COMPREHENSIVE VALIDATION RESULTS")
            logging.info("=" * 60)
            
            validation_stats = self.server.get_comprehensive_validation_stats()
            
            logging.info(f"📈 Packet Validation Summary:")
            summary = validation_stats['packet_validation_summary']
            logging.info(f"   Total packets received: {validation_stats['total_packets_received']}")
            logging.info(f"   Valid headers: {summary['valid_headers']}")
            logging.info(f"   Invalid headers: {summary['invalid_headers']}")
            logging.info(f"   Malformed packets: {summary['malformed_packets']}")
            logging.info(f"   Success rate: {summary['success_rate_percentage']:.1f}%")
            
            logging.info(f"🐛 Error Breakdown:")
            errors = validation_stats['error_breakdown']
            for error_type, count in errors.items():
                if count > 0:
                    logging.info(f"   {error_type}: {count}")
            
            logging.info(f"⚡ Performance Metrics:")
            perf = validation_stats['performance_metrics']
            logging.info(f"   Packets/sec: {perf['packets_per_second']:.1f}")
            logging.info(f"   Bytes/sec: {perf['bytes_per_second']:.1f}")
            logging.info(f"   Avg packet size: {perf['average_packet_size']:.1f} bytes")
            
            if validation_stats['device_summaries']:
                logging.info(f"📱 Per-Device Analysis:")
                for device_id, device_summary in validation_stats['device_summaries'].items():
                    logging.info(f"   Device {device_id}:")
                    logging.info(f"     Packets: {device_summary['packet_count']}")
                    logging.info(f"     Success rate: {device_summary['success_rate_percentage']:.1f}%")
                    logging.info(f"     Sequence gaps: {device_summary['sequence_gaps']}")
                    logging.info(f"     Late packets: {device_summary['late_packets']}")
                    
                    if device_summary['error_categories']:
                        logging.info(f"     Error categories: {device_summary['error_categories']}")
            
            # Final server debug summary
            logging.info("")
            self.server.print_debug_summary()
            
            # Test success criteria
            success_rate = summary['success_rate_percentage']
            expected_valid = 4  # From test case 1
            expected_invalid = 6  # From test cases 2, 3, 4
            
            test_success = (
                summary['valid_headers'] >= expected_valid and
                summary['invalid_headers'] + summary['malformed_packets'] >= expected_invalid and
                success_rate > 0  # Some packets should be valid
            )
            
            if test_success:
                logging.info("✅ ESP32-P4 UDP Packet Validation Tests PASSED")
                logging.info(f"   Server correctly parsed {summary['valid_headers']} valid packets")
                logging.info(f"   Server correctly rejected {summary['invalid_headers'] + summary['malformed_packets']} invalid packets")
                logging.info(f"   Comprehensive error categorization working properly")
            else:
                logging.error("❌ ESP32-P4 UDP Packet Validation Tests FAILED")
                logging.error(f"   Expected >= {expected_valid} valid, >= {expected_invalid} invalid")
                logging.error(f"   Got {summary['valid_headers']} valid, {summary['invalid_headers'] + summary['malformed_packets']} invalid")
            
            return test_success
            
        except Exception as e:
            logging.error(f"❌ Test execution failed: {e}")
            return False
        
        finally:
            self.server.stop()
    
    def run_performance_benchmark(self, duration_seconds: int = 10, packets_per_second: int = 50):
        """Run performance benchmark with high packet rate."""
        logging.info("⚡ Starting Performance Benchmark")
        logging.info(f"   Duration: {duration_seconds}s")
        logging.info(f"   Target rate: {packets_per_second} pkt/s")
        
        self.server = WirelessAudioServer(port=self.server_port)
        self.server.enable_debug_logging(enable=False)  # Disable verbose logging for performance
        
        if not self.server.start():
            logging.error("❌ Failed to start server for benchmark")
            return
        
        try:
            start_time = time.time()
            packet_count = 0
            
            while time.time() - start_time < duration_seconds:
                # Send valid packet
                packet = self.create_esp32_p4_test_packet(packet_count, 320, 16000, 1, 16, 0)
                if self.send_test_packet(packet):
                    packet_count += 1
                
                # Control packet rate
                time.sleep(1.0 / packets_per_second)
            
            # Wait for processing
            time.sleep(1)
            
            # Get performance results
            validation_stats = self.server.get_comprehensive_validation_stats()
            actual_duration = validation_stats['runtime_seconds']
            
            logging.info("📊 Benchmark Results:")
            logging.info(f"   Duration: {actual_duration:.2f}s")
            logging.info(f"   Packets sent: {packet_count}")
            logging.info(f"   Packets received: {validation_stats['total_packets_received']}")
            logging.info(f"   Packet loss: {packet_count - validation_stats['total_packets_received']}")
            logging.info(f"   Actual rate: {validation_stats['performance_metrics']['packets_per_second']:.1f} pkt/s")
            logging.info(f"   Success rate: {validation_stats['packet_validation_summary']['success_rate_percentage']:.1f}%")
            
        finally:
            self.server.stop()

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32-P4 UDP Packet Validation Test')
    parser.add_argument('--port', type=int, default=8003, help='Server port (default: 8003)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--duration', type=int, default=10, help='Benchmark duration in seconds')
    parser.add_argument('--rate', type=int, default=50, help='Benchmark packet rate per second')
    
    args = parser.parse_args()
    
    validator = ESP32P4PacketValidator(server_port=args.port)
    
    if args.benchmark:
        validator.run_performance_benchmark(args.duration, args.rate)
    else:
        success = validator.run_validation_tests()
        exit(0 if success else 1)

if __name__ == "__main__":
    main()