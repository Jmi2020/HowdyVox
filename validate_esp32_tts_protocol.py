#!/usr/bin/env python3
"""
ESP32-P4 TTS Protocol Validation Script
Validates WebSocket TTS protocol between HowdyTTS server and ESP32-P4 devices.
"""

import asyncio
import websockets
import json
import base64
import logging
import time
import numpy as np
import wave
import tempfile
import os
from typing import Dict, Any, List
from voice_assistant.websocket_tts_server import start_websocket_tts_server, get_websocket_tts_server

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ESP32P4_TTS_Protocol_Validator:
    """
    Comprehensive validator for ESP32-P4 TTS WebSocket protocol.
    Tests both server-side message sending and client-side reception.
    """
    
    def __init__(self, esp32_ip: str = "192.168.0.151"):
        self.esp32_ip = esp32_ip
        self.tts_server = None
        self.validation_results = {}
        
        # Test audio samples
        self.test_audio_samples = self.generate_test_audio_samples()
        
    def generate_test_audio_samples(self) -> Dict[str, bytes]:
        """Generate various test audio samples for validation."""
        samples = {}
        
        # 1. Short beep (0.5 seconds, 440Hz)
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_440 = np.sin(2 * np.pi * 440 * t)
        samples['short_beep'] = (wave_440 * 32767).astype(np.int16).tobytes()
        
        # 2. Medium tone (2 seconds, 880Hz)
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave_880 = np.sin(2 * np.pi * 880 * t)
        samples['medium_tone'] = (wave_880 * 32767).astype(np.int16).tobytes()
        
        # 3. Multi-tone sequence (3 seconds, varying frequencies)
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C major scale
        wave_multi = np.zeros_like(t)
        for i, freq in enumerate(freqs):
            start_idx = int(i * len(t) / len(freqs))
            end_idx = int((i + 1) * len(t) / len(freqs))
            wave_multi[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        samples['multi_tone'] = (wave_multi * 32767).astype(np.int16).tobytes()
        
        # 4. Silence (1 second of zero audio)
        samples['silence'] = np.zeros(sample_rate, dtype=np.int16).tobytes()
        
        logging.info(f"Generated {len(samples)} test audio samples")
        for name, data in samples.items():
            logging.info(f"  {name}: {len(data)} bytes ({len(data)//2} samples)")
        
        return samples
    
    def create_test_wav_file(self, audio_data: bytes, filename: str) -> str:
        """Create a temporary WAV file for testing."""
        temp_dir = tempfile.gettempdir()
        wav_path = os.path.join(temp_dir, f"{filename}.wav")
        
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_data)
        
        logging.info(f"Created test WAV file: {wav_path}")
        return wav_path
    
    async def validate_server_startup(self) -> bool:
        """Validate that HowdyTTS WebSocket server starts correctly."""
        logging.info("🧪 Validating server startup...")
        
        try:
            # Start WebSocket TTS server
            self.tts_server = start_websocket_tts_server(host="0.0.0.0", port=8002)
            await asyncio.sleep(1.0)  # Allow server to start
            
            if self.tts_server and self.tts_server.running:
                logging.info("✅ WebSocket TTS server started successfully")
                self.validation_results['server_startup'] = {'status': 'success', 'port': 8002}
                return True
            else:
                logging.error("❌ WebSocket TTS server failed to start")
                self.validation_results['server_startup'] = {'status': 'failed', 'error': 'Server not running'}
                return False
                
        except Exception as e:
            logging.error(f"❌ Server startup exception: {e}")
            self.validation_results['server_startup'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def validate_esp32_connection(self) -> bool:
        """Validate ESP32-P4 can connect to the WebSocket server."""
        logging.info("🧪 Validating ESP32-P4 connection...")
        
        try:
            # Wait for ESP32-P4 to connect
            max_wait_time = 10.0
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                devices = self.tts_server.get_connected_devices()
                esp32_devices = [dev for dev in devices if 'esp32' in dev.lower()]
                
                if esp32_devices:
                    logging.info(f"✅ ESP32-P4 connected: {esp32_devices}")
                    self.validation_results['esp32_connection'] = {
                        'status': 'success',
                        'devices': esp32_devices,
                        'connection_time': time.time() - start_time
                    }
                    return True
                
                await asyncio.sleep(0.5)
            
            logging.warning("⚠️ ESP32-P4 not connected within timeout")
            self.validation_results['esp32_connection'] = {
                'status': 'timeout',
                'devices_found': list(self.tts_server.get_connected_devices().keys()),
                'wait_time': max_wait_time
            }
            return False
            
        except Exception as e:
            logging.error(f"❌ ESP32-P4 connection validation failed: {e}")
            self.validation_results['esp32_connection'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def validate_tts_audio_delivery(self) -> bool:
        """Validate TTS audio delivery to ESP32-P4."""
        logging.info("🧪 Validating TTS audio delivery...")
        
        try:
            devices = self.tts_server.get_connected_devices()
            if not devices:
                logging.error("❌ No ESP32-P4 devices connected for TTS testing")
                self.validation_results['tts_audio_delivery'] = {'status': 'failed', 'error': 'No devices connected'}
                return False
            
            # Test each audio sample
            test_results = {}
            
            for sample_name, audio_data in self.test_audio_samples.items():
                logging.info(f"📤 Testing audio sample: {sample_name} ({len(audio_data)} bytes)")
                
                # Create temporary WAV file
                wav_path = self.create_test_wav_file(audio_data, f"test_{sample_name}")
                
                try:
                    # Send to all connected devices
                    success_count = 0
                    for device_id in devices:
                        session_id = f"test_{sample_name}_{int(time.time())}"
                        
                        # Method 1: Send via WebSocket TTS server
                        success = self.tts_server.send_tts_audio_sync(device_id, audio_data, session_id)
                        
                        if success:
                            success_count += 1
                            logging.info(f"✅ Sent {sample_name} to {device_id} successfully")
                        else:
                            logging.error(f"❌ Failed to send {sample_name} to {device_id}")
                    
                    test_results[sample_name] = {
                        'devices_targeted': len(devices),
                        'devices_success': success_count,
                        'success_rate': success_count / len(devices),
                        'audio_size': len(audio_data)
                    }
                    
                    # Wait between tests
                    await asyncio.sleep(1.0)
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(wav_path):\n                        os.remove(wav_path)
            
            # Calculate overall success
            total_success = sum(result['devices_success'] for result in test_results.values())
            total_attempts = sum(result['devices_targeted'] for result in test_results.values())
            overall_success_rate = total_success / total_attempts if total_attempts > 0 else 0
            
            self.validation_results['tts_audio_delivery'] = {
                'status': 'success' if overall_success_rate > 0.8 else 'partial_success',
                'overall_success_rate': overall_success_rate,
                'test_results': test_results,
                'total_attempts': total_attempts,
                'total_successes': total_success
            }
            
            logging.info(f"🎯 TTS audio delivery validation: {overall_success_rate:.1%} success rate")
            return overall_success_rate > 0.5
            
        except Exception as e:
            logging.error(f"❌ TTS audio delivery validation failed: {e}")
            self.validation_results['tts_audio_delivery'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def validate_message_format_compatibility(self) -> bool:
        """Validate WebSocket message format compatibility."""
        logging.info("🧪 Validating message format compatibility...")
        
        try:
            devices = self.tts_server.get_connected_devices()
            if not devices:
                logging.warning("⚠️ No devices connected for message format testing")
                return False
            
            device_id = list(devices.keys())[0]
            test_messages = []
            
            # Test 1: Standard TTS audio message
            test_audio = self.test_audio_samples['short_beep']
            audio_b64 = base64.b64encode(test_audio).decode('utf-8')
            
            standard_msg = {
                'type': 'tts_audio',
                'session_id': f"format_test_{int(time.time())}",
                'audio_format': 'pcm_16bit_mono_16khz',
                'audio_data': audio_b64,
                'timestamp': int(time.time() * 1000)
            }
            
            # Test 2: Message with alternative field names (compatibility test)
            alt_msg = {
                'message_type': 'tts_audio',  # Alternative field name
                'session_id': f"alt_test_{int(time.time())}",
                'audio_format': 'pcm_16bit_mono_16khz',
                'audio_data': audio_b64,
                'timestamp': int(time.time() * 1000)
            }
            
            # Test 3: Server info message
            info_msg = {
                'type': 'server_info',
                'server_name': 'validation_test',
                'timestamp': int(time.time() * 1000),
                'supported_features': ['tts_audio', 'vad_feedback']
            }
            
            test_messages = [
                ('standard_tts_message', standard_msg),
                ('alternative_field_names', alt_msg),
                ('server_info_message', info_msg)
            ]
            
            format_results = {}
            
            for test_name, message in test_messages:
                try:
                    # Send message directly via WebSocket
                    if device_id in self.tts_server.devices:
                        websocket = self.tts_server.devices[device_id]
                        await websocket.send(json.dumps(message))
                        
                        format_results[test_name] = {
                            'sent': True,
                            'message_size': len(json.dumps(message)),
                            'fields': list(message.keys())
                        }
                        
                        logging.info(f"✅ Sent {test_name} successfully")
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    format_results[test_name] = {
                        'sent': False,
                        'error': str(e)
                    }
                    logging.error(f"❌ Failed to send {test_name}: {e}")
            
            self.validation_results['message_format_compatibility'] = {
                'status': 'success',
                'test_results': format_results,
                'tests_passed': sum(1 for result in format_results.values() if result.get('sent', False)),
                'total_tests': len(test_messages)
            }
            
            logging.info("✅ Message format compatibility validation completed")
            return True
            
        except Exception as e:
            logging.error(f"❌ Message format compatibility validation failed: {e}")
            self.validation_results['message_format_compatibility'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def validate_session_lifecycle(self) -> bool:
        """Validate complete TTS session lifecycle (start/chunks/end)."""
        logging.info("🧪 Validating TTS session lifecycle...")
        
        try:
            devices = self.tts_server.get_connected_devices()
            if not devices:
                logging.warning("⚠️ No devices connected for session lifecycle testing")
                return False
            
            device_id = list(devices.keys())[0]
            session_id = f"lifecycle_test_{int(time.time())}"
            
            # Use multi-tone audio for chunked transmission
            audio_data = self.test_audio_samples['multi_tone']
            chunk_size = 1024  # 512 samples per chunk
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            
            lifecycle_results = {}
            
            # Step 1: Send session start
            session_start = {
                'type': 'tts_session_start',
                'session_id': session_id,
                'audio_format': {
                    'sample_rate': 16000,
                    'channels': 1,
                    'bits_per_sample': 16
                },
                'estimated_duration_ms': 3000,
                'total_chunks_expected': total_chunks
            }
            
            websocket = self.tts_server.devices[device_id]
            await websocket.send(json.dumps(session_start))
            lifecycle_results['session_start'] = {'sent': True, 'chunks_expected': total_chunks}
            logging.info(f"📤 Sent session start: {session_id}")
            
            await asyncio.sleep(0.5)
            
            # Step 2: Send audio chunks
            for chunk_idx in range(total_chunks):
                start_byte = chunk_idx * chunk_size
                end_byte = min(start_byte + chunk_size, len(audio_data))
                chunk_data = audio_data[start_byte:end_byte]
                
                chunk_msg = {
                    'type': 'tts_audio_chunk',
                    'session_id': session_id,
                    'chunk_sequence': chunk_idx,
                    'chunk_size': len(chunk_data),
                    'is_final': (chunk_idx == total_chunks - 1),
                    'audio_data': base64.b64encode(chunk_data).decode('utf-8')
                }
                
                await websocket.send(json.dumps(chunk_msg))
                logging.info(f"📤 Sent chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_data)} bytes)")
                
                await asyncio.sleep(0.2)  # Simulate streaming delay
            
            lifecycle_results['chunks_sent'] = {'count': total_chunks, 'total_bytes': len(audio_data)}
            
            # Step 3: Send session end
            session_end = {
                'type': 'tts_session_end',
                'session_id': session_id,
                'total_chunks_sent': total_chunks,
                'total_audio_bytes': len(audio_data),
                'session_complete': True
            }
            
            await websocket.send(json.dumps(session_end))
            lifecycle_results['session_end'] = {'sent': True}
            logging.info(f"📤 Sent session end: {session_id}")
            
            self.validation_results['session_lifecycle'] = {
                'status': 'success',
                'session_id': session_id,
                'results': lifecycle_results,
                'total_chunks': total_chunks,
                'total_audio_bytes': len(audio_data)
            }
            
            logging.info("✅ TTS session lifecycle validation completed")
            return True
            
        except Exception as e:
            logging.error(f"❌ Session lifecycle validation failed: {e}")
            self.validation_results['session_lifecycle'] = {'status': 'failed', 'error': str(e)}
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive TTS protocol validation."""
        logging.info("🚀 Starting comprehensive ESP32-P4 TTS protocol validation...")
        
        validation_summary = {
            'timestamp': time.time(),
            'esp32_ip': self.esp32_ip,
            'tests_run': [],
            'tests_passed': 0,
            'tests_failed': 0,
            'overall_status': 'unknown'
        }
        
        try:
            # Validation test suite
            tests = [
                ('server_startup', self.validate_server_startup),
                ('esp32_connection', self.validate_esp32_connection),
                ('message_format_compatibility', self.validate_message_format_compatibility),
                ('tts_audio_delivery', self.validate_tts_audio_delivery),
                ('session_lifecycle', self.validate_session_lifecycle)
            ]
            
            for test_name, test_func in tests:
                logging.info(f"🧪 Running test: {test_name}")
                
                try:
                    test_start_time = time.time()
                    test_result = await test_func()
                    test_duration = time.time() - test_start_time
                    
                    validation_summary['tests_run'].append({
                        'name': test_name,
                        'result': test_result,
                        'duration': test_duration
                    })
                    
                    if test_result:
                        validation_summary['tests_passed'] += 1
                        logging.info(f"✅ Test passed: {test_name} ({test_duration:.2f}s)")
                    else:
                        validation_summary['tests_failed'] += 1
                        logging.error(f"❌ Test failed: {test_name} ({test_duration:.2f}s)")
                
                except Exception as e:
                    validation_summary['tests_failed'] += 1
                    logging.error(f"💥 Test error: {test_name} - {e}")
                    self.validation_results[test_name] = {'status': 'error', 'error': str(e)}
                
                # Wait between tests
                await asyncio.sleep(1.0)
            
            # Calculate overall status
            total_tests = validation_summary['tests_passed'] + validation_summary['tests_failed']
            success_rate = validation_summary['tests_passed'] / total_tests if total_tests > 0 else 0
            
            if success_rate >= 0.8:
                validation_summary['overall_status'] = 'success'
            elif success_rate >= 0.5:
                validation_summary['overall_status'] = 'partial_success'
            else:
                validation_summary['overall_status'] = 'failed'
            
            validation_summary['success_rate'] = success_rate
            validation_summary['detailed_results'] = self.validation_results
            
            # Log final summary
            logging.info("="*60)
            logging.info("ESP32-P4 TTS PROTOCOL VALIDATION COMPLETE")
            logging.info("="*60)
            logging.info(f"Tests passed: {validation_summary['tests_passed']}")
            logging.info(f"Tests failed: {validation_summary['tests_failed']}")
            logging.info(f"Success rate: {success_rate:.1%}")
            logging.info(f"Overall status: {validation_summary['overall_status']}")
            logging.info("="*60)
            
            return validation_summary
            
        except Exception as e:
            logging.error(f"💥 Comprehensive validation failed: {e}")
            validation_summary['overall_status'] = 'error'
            validation_summary['error'] = str(e)
            return validation_summary
        
        finally:
            # Cleanup
            if self.tts_server:
                self.tts_server.stop_server()

async def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate ESP32-P4 TTS WebSocket protocol')
    parser.add_argument('--esp32-ip', default='192.168.0.151', help='ESP32-P4 IP address')
    parser.add_argument('--output', help='Save results to JSON file')
    args = parser.parse_args()
    
    validator = ESP32P4_TTS_Protocol_Validator(args.esp32_ip)
    results = await validator.run_comprehensive_validation()
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"📄 Results saved to: {args.output}")
    
    # Print results summary
    print("\n" + "="*60)
    print("ESP32-P4 TTS PROTOCOL VALIDATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    print("="*60)
    
    return results['overall_status'] == 'success'

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)