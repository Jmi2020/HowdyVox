#!/usr/bin/env python3
"""
UDP Debug Integration Script for HowdyTTS Voice Assistant

This script enables comprehensive UDP debugging in the main HowdyTTS voice assistant
when running in wireless mode. It patches the wireless audio server to enable
debug logging for troubleshooting ESP32-P4 audio reception issues.

Usage:
    python enable_udp_debug.py [--verbose] [--hex-dump] [--packet-interval N]
    
This will start the voice assistant with enhanced UDP debugging enabled.
"""

import sys
import argparse
import logging
from voice_assistant.wireless_audio_server import WirelessAudioServer

def patch_wireless_debug():
    """Patch the WirelessAudioServer class to enable debug logging by default."""
    
    # Store original __init__ method
    original_init = WirelessAudioServer.__init__
    
    def debug_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Enable debug logging by default in wireless mode
        self.enable_debug_logging(
            enable=True,
            hex_dump=getattr(patch_wireless_debug, 'hex_dump_enabled', False),
            packet_interval=getattr(patch_wireless_debug, 'packet_log_interval', 10)
        )
        
        logging.info("🔧 UDP Debug mode automatically enabled for wireless audio server")
    
    # Replace __init__ with debug version
    WirelessAudioServer.__init__ = debug_init

def main():
    """Enable UDP debugging and run the voice assistant."""
    parser = argparse.ArgumentParser(
        description='HowdyTTS Voice Assistant with UDP Debug Instrumentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script enables comprehensive UDP debugging for troubleshooting why the 
HowdyTTS server is not receiving audio packets from ESP32-P4 devices.

Debug Features Enabled:
  - Detailed packet logging with source addresses
  - UDP header format verification matching ESP32-P4 structure
  - Network interface binding validation  
  - Per-device packet counters and diagnostics
  - Raw packet data inspection (optional)

Usage Examples:
  python enable_udp_debug.py --wireless
  python enable_udp_debug.py --wireless --verbose
  python enable_udp_debug.py --wireless --hex-dump --packet-interval 10
        """
    )
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    parser.add_argument('--hex-dump', action='store_true',
                       help='Enable hex dump of packet contents')
    parser.add_argument('--packet-interval', type=int, default=10,
                       help='Log every N packets (default: 10 for debug mode)')
    parser.add_argument('--wireless', action='store_true',
                       help='Force wireless mode (recommended for debugging)')
    
    args = parser.parse_args()
    
    # Configure debug parameters
    patch_wireless_debug.hex_dump_enabled = args.hex_dump
    patch_wireless_debug.packet_log_interval = args.packet_interval
    
    # Configure enhanced logging for debug mode
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Apply debug patches
    patch_wireless_debug()
    
    # Import and run voice assistant after patching
    logging.info("🔧 Applying UDP debug patches to voice assistant...")
    
    # Add wireless argument to sys.argv if not present
    if args.wireless and '--wireless' not in sys.argv:
        sys.argv.append('--wireless')
    
    # Import and run the main voice assistant
    try:
        from run_voice_assistant import main as voice_assistant_main
        
        logging.info("🚀 Starting HowdyTTS Voice Assistant with UDP debug instrumentation")
        logging.info("📡 Enhanced UDP debugging is now active for ESP32-P4 connectivity")
        logging.info(f"📝 Debug configuration: verbose={args.verbose}, hex_dump={args.hex_dump}, interval={args.packet_interval}")
        logging.info("🔍 Monitor logs for detailed UDP packet analysis")
        
        voice_assistant_main()
        
    except ImportError as e:
        logging.error(f"❌ Failed to import voice assistant: {e}")
        logging.error("Make sure you're running this from the HowdyTTS directory")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Voice assistant error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()