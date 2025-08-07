#!/usr/bin/env python3

"""
ESP32-P4 HowdyTTS Integration Test

This script validates the complete ESP32-P4 to HowdyTTS conversation loop:
1. ESP32-P4 device discovery and connection
2. Audio streaming from ESP32-P4 to HowdyTTS server
3. Voice assistant processing (STT → AI → TTS)
4. TTS audio playback on ESP32-P4 via WebSocket

Usage:
    python tools/test_esp32p4_integration.py [--duration 60] [--room "Living Room"]
"""

import sys
import os
import time
import logging
import threading
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice_assistant.network_audio_source import NetworkAudioSource
from voice_assistant.websocket_tts_server import start_websocket_tts_server, get_websocket_tts_server
from voice_assistant.wireless_device_manager import WirelessDeviceManager

class ESP32P4IntegrationTester:
    """
    Comprehensive integration tester for ESP32-P4 HowdyTTS system.
    """
    
    def __init__(self, target_room: Optional[str] = None):
        self.target_room = target_room
        self.network_audio = None
        self.test_results = {
            'discovery': False,
            'audio_streaming': False,
            'websocket_connection': False,
            'tts_playback': False,
            'complete_loop': False
        }
        self.test_start_time = time.time()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def test_device_discovery(self, timeout: float = 30.0) -> bool:
        """Test ESP32-P4 device discovery and connection."""
        self.logger.info("🔍 Testing ESP32-P4 device discovery...")
        
        try:
            # Initialize network audio source
            self.network_audio = NetworkAudioSource(target_room=self.target_room)
            
            if not self.network_audio.start():
                self.logger.error("❌ Failed to start NetworkAudioSource")
                return False
            
            # Wait for device discovery
            start_time = time.time()
            device_found = False
            
            while (time.time() - start_time) < timeout:
                devices = self.network_audio.get_available_devices()
                if devices:
                    device_found = True
                    self.logger.info(f"✅ ESP32-P4 device(s) discovered: {len(devices)} device(s)")
                    for idx, device_name, device_ip in devices:
                        self.logger.info(f"   📱 Device {idx}: {device_name} - {device_ip}")
                    break
                
                time.sleep(1.0)
                elapsed = int(time.time() - start_time)
                print(f"\\r⏳ Waiting for ESP32-P4 discovery... {elapsed}s", end="", flush=True)
            
            print()  # New line after waiting animation
            
            if device_found:
                self.test_results['discovery'] = True
                return True
            else:
                self.logger.error(f"❌ No ESP32-P4 devices found after {timeout}s timeout")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Device discovery test failed: {e}")
            return False
    
    def test_audio_streaming(self, duration: float = 10.0) -> bool:
        """Test audio streaming from ESP32-P4."""
        self.logger.info("🎵 Testing ESP32-P4 audio streaming...")
        
        if not self.network_audio:
            self.logger.error("❌ NetworkAudioSource not initialized")
            return False
        
        try:
            # Create test recording
            test_file = "/tmp/esp32p4_test_recording.wav"
            
            self.logger.info(f"🎤 Recording {duration}s of audio from ESP32-P4...")
            self.logger.info("   💬 Please speak into the ESP32-P4 microphone now!")
            
            # Start recording
            success = self.network_audio.record_audio(
                file_path=test_file,
                max_duration=duration,
                silence_timeout=2.0
            )
            
            if success and os.path.exists(test_file):
                file_size = os.path.getsize(test_file)
                self.logger.info(f"✅ Audio streaming successful: {file_size} bytes recorded")
                
                # Clean up test file
                try:
                    os.remove(test_file)
                except:
                    pass
                
                self.test_results['audio_streaming'] = True
                return True
            else:
                self.logger.error("❌ Audio streaming failed - no audio recorded")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Audio streaming test failed: {e}")
            return False
    
    def test_websocket_connection(self) -> bool:
        """Test WebSocket TTS server connection."""
        self.logger.info("🔌 Testing WebSocket TTS server...")
        
        try:
            # WebSocket server should already be started by NetworkAudioSource
            tts_server = get_websocket_tts_server()
            
            if not tts_server:
                self.logger.error("❌ WebSocket TTS server not running")
                return False
            
            # Wait a moment for ESP32-P4 to connect via WebSocket
            time.sleep(2.0)
            
            connected_devices = tts_server.get_connected_devices()
            if connected_devices:
                self.logger.info(f"✅ WebSocket connections active: {len(connected_devices)} device(s)")
                for device_id, info in connected_devices.items():
                    self.logger.info(f"   🔗 {device_id}: {info['ip']} (connected {time.time() - info['connected_at']:.1f}s ago)")
                
                self.test_results['websocket_connection'] = True
                return True
            else:
                self.logger.warning("⚠️ No WebSocket connections from ESP32-P4 devices")
                # This might be OK if ESP32-P4 firmware doesn't support WebSocket yet
                return False
                
        except Exception as e:
            self.logger.error(f"❌ WebSocket connection test failed: {e}")
            return False
    
    def test_tts_playback(self) -> bool:
        """Test TTS audio playback to ESP32-P4."""
        self.logger.info("🔊 Testing TTS audio playback to ESP32-P4...")
        
        try:
            # Create a simple test TTS audio file
            test_text = "Hello from HowdyTTS server! This is a test message."
            test_audio_file = "/tmp/test_tts.wav"
            
            # Generate test TTS audio using Kokoro
            from voice_assistant.kokoro_manager import KokoroManager
            from voice_assistant.config import Config
            import soundfile as sf
            
            kokoro = KokoroManager.get_instance(local_model_path=Config.LOCAL_MODEL_PATH)
            samples, sample_rate = kokoro.create(test_text, voice=Config.KOKORO_VOICE, speed=1.0, lang="en-us")
            
            # Save as 16kHz mono for ESP32-P4
            sf.write(test_audio_file, samples, sample_rate)
            
            # Send to ESP32-P4 devices
            if self.network_audio:
                success = self.network_audio.send_tts_audio_to_devices(test_audio_file, test_text)
                
                # Clean up test file
                try:
                    os.remove(test_audio_file)
                except:
                    pass
                
                if success:
                    self.logger.info("✅ TTS audio sent to ESP32-P4 devices successfully")
                    self.logger.info("   👂 You should hear the test message on ESP32-P4 speakers")
                    self.test_results['tts_playback'] = True
                    return True
                else:
                    self.logger.error("❌ Failed to send TTS audio to ESP32-P4 devices")
                    return False
            else:
                self.logger.error("❌ NetworkAudioSource not available")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ TTS playback test failed: {e}")
            return False
    
    def test_complete_conversation_loop(self, timeout: float = 30.0) -> bool:
        """Test complete conversation loop simulation."""
        self.logger.info("💬 Testing complete conversation loop...")
        self.logger.info("   🗣️ This test simulates the full voice assistant interaction:")
        self.logger.info("   📝 ESP32-P4 audio → STT → AI response → TTS → ESP32-P4 playback")
        
        try:
            if not all([
                self.test_results['discovery'],
                self.test_results['audio_streaming']
            ]):
                self.logger.error("❌ Prerequisites not met for conversation loop test")
                return False
            
            # Simulate the conversation loop by using the same components as run_voice_assistant.py
            from voice_assistant.transcription import transcribe_audio
            from voice_assistant.response_generation import generate_response
            from voice_assistant.text_to_speech import text_to_speech
            from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
            from voice_assistant.config import Config
            
            self.logger.info("🎤 Please speak a test phrase into ESP32-P4 microphone...")
            
            # Step 1: Record audio from ESP32-P4
            audio_file = "/tmp/conversation_test.wav"
            recording_success = self.network_audio.record_audio(
                file_path=audio_file,
                max_duration=10.0,
                silence_timeout=3.0
            )
            
            if not recording_success or not os.path.exists(audio_file):
                self.logger.error("❌ Failed to record audio for conversation test")
                return False
            
            # Step 2: Transcribe audio
            self.logger.info("🔤 Transcribing audio...")
            transcription_api_key = get_transcription_api_key()
            user_text = transcribe_audio(Config.TRANSCRIPTION_MODEL, transcription_api_key, audio_file, Config.LOCAL_MODEL_PATH)
            
            if not user_text:
                self.logger.error("❌ No transcription result")
                os.remove(audio_file)
                return False
            
            self.logger.info(f"📝 Transcription: '{user_text}'")
            
            # Step 3: Generate AI response
            self.logger.info("🤖 Generating AI response...")
            chat_history = [
                {"role": "system", "content": "You are Howdy, a helpful assistant. Keep responses short for testing."},
                {"role": "user", "content": user_text}
            ]
            
            response_api_key = get_response_api_key()
            ai_response = generate_response(Config.RESPONSE_MODEL, response_api_key, chat_history, Config.LOCAL_MODEL_PATH)
            
            if not ai_response:
                self.logger.error("❌ No AI response generated")
                os.remove(audio_file)
                return False
            
            self.logger.info(f"🤖 AI Response: '{ai_response}'")
            
            # Step 4: Generate TTS
            self.logger.info("🔊 Generating TTS audio...")
            tts_api_key = get_tts_api_key()
            tts_success, tts_file = text_to_speech(Config.TTS_MODEL, tts_api_key, ai_response, "/tmp/conversation_tts.wav", Config.LOCAL_MODEL_PATH)
            
            if not tts_success or not tts_file:
                self.logger.error("❌ TTS generation failed")
                os.remove(audio_file)
                return False
            
            # Step 5: Send TTS to ESP32-P4
            self.logger.info("📡 Sending TTS to ESP32-P4...")
            playback_success = self.network_audio.send_tts_audio_to_devices(tts_file, ai_response)
            
            # Clean up files
            try:
                os.remove(audio_file)
                os.remove(tts_file)
            except:
                pass
            
            if playback_success:
                self.logger.info("✅ Complete conversation loop successful!")
                self.logger.info("   👂 You should hear the AI response on ESP32-P4 speakers")
                self.test_results['complete_loop'] = True
                return True
            else:
                self.logger.error("❌ Failed to send TTS response to ESP32-P4")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Conversation loop test failed: {e}")
            return False
    
    def run_all_tests(self, test_duration: float = 60.0) -> Dict[str, bool]:
        """Run all integration tests."""
        self.logger.info("🚀 Starting ESP32-P4 HowdyTTS Integration Tests")
        self.logger.info(f"   🎯 Target room: {self.target_room or 'Any'}")
        self.logger.info(f"   ⏱️ Test duration: {test_duration}s")
        self.logger.info("=" * 60)
        
        try:
            # Test 1: Device Discovery
            if not self.test_device_discovery(timeout=30.0):
                self.logger.error("🛑 Device discovery failed - aborting remaining tests")
                return self.test_results
            
            time.sleep(2.0)
            
            # Test 2: Audio Streaming
            if not self.test_audio_streaming(duration=8.0):
                self.logger.error("⚠️ Audio streaming failed - continuing with other tests")
            
            time.sleep(2.0)
            
            # Test 3: WebSocket Connection
            if not self.test_websocket_connection():
                self.logger.warning("⚠️ WebSocket connection test failed - this may be expected if ESP32-P4 firmware doesn't support WebSocket yet")
            
            time.sleep(2.0)
            
            # Test 4: TTS Playback
            if not self.test_tts_playback():
                self.logger.error("⚠️ TTS playback failed - continuing with conversation test")
            
            time.sleep(2.0)
            
            # Test 5: Complete Conversation Loop
            if self.test_results['discovery'] and self.test_results['audio_streaming']:
                self.test_complete_conversation_loop(timeout=30.0)
            
            return self.test_results
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Tests interrupted by user")
            return self.test_results
        
        except Exception as e:
            self.logger.error(f"❌ Test suite failed: {e}")
            return self.test_results
        
        finally:
            # Clean up
            if self.network_audio:
                try:
                    self.network_audio.stop()
                except Exception as e:
                    self.logger.warning(f"Error during cleanup: {e}")
    
    def print_results(self):
        """Print test results summary."""
        total_time = time.time() - self.test_start_time
        
        print("\n" + "=" * 60)
        print("📊 ESP32-P4 HowdyTTS Integration Test Results")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            test_display = test_name.replace('_', ' ').title()
            print(f"{test_display:<25} {status}")
        
        passed = sum(self.test_results.values())
        total = len(self.test_results)
        
        print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        print(f"⏱️ Total time: {total_time:.1f}s")
        
        if passed == total:
            print("🎉 All tests passed! ESP32-P4 integration is working correctly.")
        elif passed >= 3:
            print("⚠️ Most tests passed. ESP32-P4 integration is partially working.")
        else:
            print("❌ Multiple tests failed. ESP32-P4 integration needs attention.")
        
        print("=" * 60)


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ESP32-P4 HowdyTTS Integration Tester')
    parser.add_argument('--duration', type=float, default=60.0, help='Test duration in seconds')
    parser.add_argument('--room', type=str, help='Target specific room for testing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run tester
    tester = ESP32P4IntegrationTester(target_room=args.room)
    
    try:
        results = tester.run_all_tests(test_duration=args.duration)
        tester.print_results()
        
        # Exit with appropriate code
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            exit(0)  # All tests passed
        elif passed >= 3:
            exit(1)  # Partial success
        else:
            exit(2)  # Multiple failures
            
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        exit(3)


if __name__ == "__main__":
    main()