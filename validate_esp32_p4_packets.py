#!/usr/bin/env python3
"""
ESP32-P4 Packet Validation Demonstration Script

This script demonstrates the enhanced UDP packet format validation
implemented for the HowdyTTS server to ensure correct parsing of
ESP32-P4 audio packets.

Key Features Demonstrated:
1. Precise UDP packet format validation matching ESP32-P4 exactly
2. Robust PCM sample extraction with size validation  
3. Per-device statistics tracking with error categorization
4. Audio quality metrics and format consistency validation
5. Comprehensive diagnostic reporting

Usage:
    python validate_esp32_p4_packets.py [--port 8003]
"""

import sys
import os
import logging
import json

# Add voice_assistant directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice_assistant'))

from wireless_audio_server import WirelessAudioServer

def main():
    """Demonstrate ESP32-P4 packet validation capabilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32-P4 Packet Validation Demo')
    parser.add_argument('--port', type=int, default=8003, help='UDP port (default: 8003)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose debug output')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logging.info("🚀 ESP32-P4 Packet Validation Demonstration")
    logging.info("=" * 60)
    
    # Create wireless audio server with enhanced validation
    server = WirelessAudioServer(port=args.port)
    
    # Enable comprehensive debug logging
    server.enable_debug_logging(enable=True, hex_dump=False, packet_interval=1)
    
    # Validate ESP32-P4 compatibility
    logging.info("🔧 Validating ESP32-P4 Compatibility...")
    compatibility = server.validate_esp32_p4_compatibility()
    
    logging.info("📋 ESP32-P4 Compatibility Report:")
    logging.info(f"   Header Format: {compatibility['header_format_validation']['format_string']}")
    logging.info(f"   Header Size: {compatibility['header_format_validation']['header_size_bytes']} bytes (expected: 12)")
    logging.info(f"   Format Match: {'✅' if compatibility['header_format_validation']['format_correct'] else '❌'}")
    logging.info(f"   Size Match: {'✅' if compatibility['header_format_validation']['size_correct'] else '❌'}")
    
    logging.info("🔍 ESP32-P4 Requirements Check:")
    requirements = compatibility['esp32_p4_requirements']
    for req_name, req_met in requirements.items():
        status = '✅' if req_met else '❌'
        logging.info(f"   {req_name}: {status}")
    
    logging.info("⚙️ Validation Capabilities:")
    capabilities = compatibility['validation_capabilities']
    for cap_name, cap_available in capabilities.items():
        status = '✅' if cap_available else '❌'
        logging.info(f"   {cap_name}: {status}")
    
    logging.info(f"📊 Overall Compatibility Score: {compatibility['overall_compatibility_score']:.1f}%")
    
    if compatibility['esp32_p4_ready']:
        logging.info("✅ Server is ESP32-P4 Ready!")
    else:
        logging.warning("⚠️ Server configuration needs adjustment for ESP32-P4")
        return 1
    
    logging.info("")
    logging.info("🎧 Starting Enhanced Audio Server...")
    logging.info(f"📡 Listening on 0.0.0.0:{args.port} for ESP32-P4 devices")
    logging.info("🔍 Enhanced validation features active:")
    logging.info("   • Precise header format validation (12-byte ESP32-P4 structure)")
    logging.info("   • PCM sample extraction with size validation") 
    logging.info("   • Per-device packet counters and sequence analysis")
    logging.info("   • Audio quality metrics and format consistency tracking")
    logging.info("   • Comprehensive error categorization and diagnostics")
    logging.info("")
    
    if server.start():
        try:
            import time
            stats_interval = 30.0  # Print detailed stats every 30 seconds
            last_stats_time = time.time()
            
            logging.info("⏳ Waiting for ESP32-P4 devices... (Press Ctrl+C to stop)")
            
            while True:
                time.sleep(1)
                
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    # Get comprehensive validation statistics
                    validation_stats = server.get_comprehensive_validation_stats()
                    
                    if validation_stats['total_packets_received'] > 0:
                        logging.info("")
                        logging.info("📊 COMPREHENSIVE VALIDATION STATISTICS")
                        logging.info("=" * 50)
                        
                        # Packet validation summary
                        summary = validation_stats['packet_validation_summary']
                        logging.info(f"📦 Packet Summary:")
                        logging.info(f"   Total received: {validation_stats['total_packets_received']}")
                        logging.info(f"   Valid headers: {summary['valid_headers']}")
                        logging.info(f"   Invalid headers: {summary['invalid_headers']}")
                        logging.info(f"   Malformed packets: {summary['malformed_packets']}")
                        logging.info(f"   Success rate: {summary['success_rate_percentage']:.1f}%")
                        
                        # Error breakdown
                        errors = validation_stats['error_breakdown']
                        if any(errors.values()):
                            logging.info(f"🐛 Error Analysis:")
                            for error_type, count in errors.items():
                                if count > 0:
                                    logging.info(f"   {error_type}: {count}")
                        
                        # Performance metrics
                        perf = validation_stats['performance_metrics']
                        logging.info(f"⚡ Performance:")
                        logging.info(f"   Rate: {perf['packets_per_second']:.1f} pkt/s")
                        logging.info(f"   Throughput: {perf['bytes_per_second']:.1f} B/s")
                        logging.info(f"   Avg packet size: {perf['average_packet_size']:.1f} bytes")
                        
                        # Per-device analysis
                        if validation_stats['device_summaries']:
                            logging.info(f"📱 Device Analysis:")
                            for device_id, device_stats in validation_stats['device_summaries'].items():
                                logging.info(f"   🔹 {device_id}:")
                                logging.info(f"     Packets: {device_stats['packet_count']}")
                                logging.info(f"     Success: {device_stats['success_rate_percentage']:.1f}%")
                                logging.info(f"     Seq gaps: {device_stats['sequence_gaps']}")
                                logging.info(f"     Late packets: {device_stats['late_packets']}")
                                
                                quality = device_stats['audio_quality']
                                logging.info(f"     Audio quality: Rate consistent: {quality['sample_rate_consistency']}, "
                                           f"Format consistent: {quality['format_consistency']}")
                                
                                if device_stats['error_categories']:
                                    logging.info(f"     Errors: {device_stats['error_categories']}")
                        
                        logging.info("=" * 50)
                    else:
                        logging.info("⏳ No ESP32-P4 packets received yet...")
                    
                    last_stats_time = current_time
                    
        except KeyboardInterrupt:
            logging.info("")
            logging.info("🛑 Shutdown requested")
            
        finally:
            # Print final comprehensive statistics
            logging.info("")
            logging.info("📊 FINAL VALIDATION REPORT")
            server.print_debug_summary()
            server.stop()
            
    else:
        logging.error("❌ Failed to start server")
        return 1
    
    logging.info("✅ ESP32-P4 Packet Validation Demo Complete")
    return 0

if __name__ == "__main__":
    exit(main())