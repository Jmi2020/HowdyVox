#!/usr/bin/env python3

import logging
import threading
import time
import numpy as np
from typing import Optional, Callable, Tuple
from collections import deque
import pyaudio

from .wireless_audio_server import WirelessAudioServer
from .wireless_device_manager import WirelessDeviceManager, WirelessDevice
from .intelligent_vad import IntelligentVAD
from .utterance_detector import IntelligentUtteranceDetector, UtteranceContext
from .esp32_p4_vad_coordinator import ESP32P4VADCoordinator, VADFusionStrategy, VADDecision
from .esp32_p4_protocol import ESP32P4ProtocolParser
from .websocket_tts_server import start_websocket_tts_server, get_websocket_tts_server
from .config import Config

class NetworkAudioSource:
    """
    Network-based audio source that integrates wireless ESP32P4 devices
    with HowdyTTS's existing audio pipeline and VAD system.
    
    This class provides the same interface as the local microphone
    but receives audio from wireless devices over UDP.
    """
    
    def __init__(self, target_room: Optional[str] = None):
        """
        Initialize network audio source.
        
        Args:
            target_room: Specific room to listen to, or None for auto-selection
        """
        self.target_room = target_room
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 512  # 32ms chunks for Silero VAD
        
        # Audio components
        self.audio_server = WirelessAudioServer(
            host="0.0.0.0",
            port=8000,
            sample_rate=self.sample_rate,
            channels=self.channels
        )
        
        self.device_manager = WirelessDeviceManager()
        
        # VAD and utterance detection
        self.vad = IntelligentVAD(
            sample_rate=self.sample_rate,
            chunk_duration_ms=32
        )
        self.utterance_detector = IntelligentUtteranceDetector()
        
        # ESP32-P4 VAD coordination
        self.vad_coordinator = ESP32P4VADCoordinator(
            server_vad=self.vad,
            fusion_strategy=VADFusionStrategy.ADAPTIVE
        )
        self.protocol_parser = ESP32P4ProtocolParser()
        
        # Audio buffering
        self.audio_buffer = deque(maxlen=1000)  # ~32 seconds at 32ms chunks
        self.pre_speech_buffer = deque(maxlen=16)  # ~500ms pre-speech buffer
        
        # Recording state
        self.is_recording = False
        self.active_device: Optional[WirelessDevice] = None
        self.recording_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'packets_received': 0,
            'audio_processed': 0,
            'vad_detections': 0,
            'device_switches': 0,
            'esp32p4_enhanced_packets': 0,
            'vad_coordinated_decisions': 0
        }
        
        # Set up callbacks
        self.audio_server.set_audio_callback(self._on_audio_received)
        self.device_manager.set_callbacks(
            connected=self._on_device_connected,
            disconnected=self._on_device_disconnected,
            status_update=self._on_device_status
        )
        
        logging.info(f"NetworkAudioSource initialized for room: {target_room or 'auto'}")
    
    def start(self) -> bool:
        """Start the network audio source."""
        try:
            # Start wireless components
            if not self.audio_server.start():
                logging.error("Failed to start wireless audio server")
                return False
            
            self.device_manager.start_monitoring()
            
            # Start WebSocket TTS server for ESP32-P4 audio playback
            logging.info("🔊 Starting WebSocket TTS server for ESP32-P4 audio playback")
            tts_server = start_websocket_tts_server(host="0.0.0.0", port=8002)
            if tts_server:
                logging.info("✅ WebSocket TTS server started on port 8002")
                
                # Set up TTS request callback
                def handle_tts_request(device_id: str, text: str):
                    logging.info(f"🎤 TTS request from {device_id}: {text[:50]}{'...' if len(text) > 50 else ''}")
                    # TTS generation will be handled by the main voice assistant
                
                tts_server.set_tts_request_callback(handle_tts_request)
            else:
                logging.warning("⚠️ Failed to start WebSocket TTS server")
            
            # Wait a moment for device discovery
            time.sleep(2.0)
            
            # Select active device
            self._select_active_device()
            
            logging.info("NetworkAudioSource started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start NetworkAudioSource: {e}")
            return False
    
    def stop(self):
        """Stop the network audio source."""
        logging.info("Stopping NetworkAudioSource...")
        
        self.is_recording = False
        
        # Stop components
        self.audio_server.stop()
        self.device_manager.stop_monitoring()
        
        # Stop WebSocket TTS server
        from .websocket_tts_server import stop_websocket_tts_server
        stop_websocket_tts_server()
        logging.info("🔇 WebSocket TTS server stopped")
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        logging.info("NetworkAudioSource stopped")
    
    def send_tts_audio_to_devices(self, audio_data: bytes, text: str = "") -> bool:
        """Send TTS audio to all connected ESP32-P4 devices."""
        tts_server = get_websocket_tts_server()
        if not tts_server:
            logging.warning("WebSocket TTS server not available")
            return False
        
        devices = tts_server.get_connected_devices()
        if not devices:
            logging.info("No ESP32-P4 devices connected for TTS playback")
            return False
        
        success_count = 0
        for device_id in devices:
            if tts_server.send_tts_audio_sync(device_id, audio_data):
                success_count += 1
                logging.info(f"🔊 Sent TTS audio to {device_id}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        
        logging.info(f"📡 TTS audio sent to {success_count}/{len(devices)} ESP32-P4 devices")
        return success_count > 0
    
    def record_audio(self, 
                    file_path: str,
                    max_duration: float = 30.0,
                    silence_timeout: float = 2.0,
                    energy_threshold: Optional[float] = None,
                    **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Record audio from wireless device using intelligent VAD.
        
        This method provides the same interface as EnhancedAudioRecorder.record_audio()
        but receives audio from network instead of local microphone.
        
        Args:
            file_path: Path to save the recorded audio
            max_duration: Maximum recording duration in seconds
            silence_timeout: Maximum silence before stopping
            energy_threshold: Minimum energy threshold (unused for network source)
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Tuple of (success, error_message)
        """
        if not self.active_device:
            return False, "No active wireless device available"
        
        if self.is_recording:
            return False, "Already recording"
        
        logging.info(f"Starting network audio recording from device {self.active_device.device_id}")
        
        # Clear buffers
        self.audio_buffer.clear()
        self.pre_speech_buffer.clear()
        
        # Reset VAD state
        self.vad.reset()
        
        # Start recording
        self.is_recording = True
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(file_path, max_duration, silence_timeout),
            daemon=True
        )
        self.recording_thread.start()
        
        # Wait for recording to complete
        self.recording_thread.join()
        
        success = len(self.audio_buffer) > 0
        error_message = None if success else "No audio data received"
        
        logging.info(f"Network audio recording completed: {success}")
        return success, error_message
    
    def get_available_devices(self) -> list:
        """Get list of available wireless audio devices (compatibility method)."""
        devices = self.device_manager.get_active_devices()
        return [(i, f"{d.device_id} ({d.room or 'No room'})", d.ip_address) 
                for i, d in enumerate(devices)]
    
    def set_device(self, device_index: int) -> bool:
        """Set active device by index (compatibility method)."""
        devices = self.device_manager.get_active_devices()
        if 0 <= device_index < len(devices):
            self.active_device = devices[device_index]
            logging.info(f"Active device set to: {self.active_device.device_id}")
            return True
        return False
    
    def get_device_info(self) -> str:
        """Get current device information."""
        if self.active_device:
            return f"Wireless Device: {self.active_device.device_id} ({self.active_device.ip_address})"
        return "No active wireless device"
    
    def test_device(self, duration: float = 3.0) -> bool:
        """Test the current device by recording for a short duration."""
        if not self.active_device:
            return False
        
        # Simple test - check if we receive audio data
        start_time = time.time()
        initial_count = len(self.audio_buffer)
        
        while time.time() - start_time < duration:
            time.sleep(0.1)
            if len(self.audio_buffer) > initial_count + 10:  # Received some audio
                return True
        
        return False
    
    def _select_active_device(self):
        """Select the active device based on target room or availability."""
        active_devices = self.device_manager.get_active_devices()
        
        if not active_devices:
            logging.warning("No active wireless devices found")
            return
        
        # Try to find device for target room
        if self.target_room:
            for device in active_devices:
                if device.room == self.target_room:
                    self.active_device = device
                    logging.info(f"Selected device for room '{self.target_room}': {device.device_id}")
                    return
            
            logging.warning(f"No device found for room '{self.target_room}', using first available")
        
        # Use first available device
        self.active_device = active_devices[0]
        logging.info(f"Selected active device: {self.active_device.device_id}")
    
    def _on_audio_received(self, audio_data: np.ndarray, raw_packet_data: bytes = None, source_addr: tuple = None):
        """Callback for receiving audio data from wireless server with ESP32-P4 packet parsing."""
        if not self.active_device or not self.is_recording:
            return
        
        # Store raw packet info for ESP32-P4 processing
        packet_info = None
        if raw_packet_data and source_addr:
            packet_info = self.protocol_parser.parse_packet(raw_packet_data, source_addr)
            if packet_info and self.protocol_parser.is_enhanced_packet(packet_info):
                self.stats['esp32p4_enhanced_packets'] += 1
        
        # Convert to int16 for compatibility
        if audio_data.dtype != np.int16:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data
        
        # Ensure correct chunk size
        if len(audio_int16) != self.chunk_size:
            # Pad or truncate to correct size
            if len(audio_int16) < self.chunk_size:
                audio_int16 = np.pad(audio_int16, (0, self.chunk_size - len(audio_int16)))
            else:
                audio_int16 = audio_int16[:self.chunk_size]
        
        # Store both processed audio and packet info for VAD coordination
        audio_entry = {
            'audio_data': audio_int16,
            'packet_info': packet_info,
            'timestamp': time.time()
        }
        self.audio_buffer.append(audio_entry)
        self.stats['packets_received'] += 1
        
        # Update device audio level
        audio_level = np.abs(audio_data).mean()
        self.device_manager.update_device_status(
            self.active_device.device_id,
            audio_level=float(audio_level),
            last_seen=time.time()
        )
    
    def _recording_loop(self, file_path: str, max_duration: float, silence_timeout: float):
        """Main recording loop with VAD and utterance detection."""
        recording_started = False
        speech_detected = False
        silence_start = None
        recorded_chunks = []
        
        start_time = time.time()
        
        while self.is_recording and (time.time() - start_time) < max_duration:
            try:
                # Wait for audio data
                if not self.audio_buffer:
                    time.sleep(0.01)
                    continue
                
                # Get next audio entry
                audio_entry = self.audio_buffer.popleft()
                if isinstance(audio_entry, dict):
                    audio_chunk = audio_entry['audio_data']
                    packet_info = audio_entry.get('packet_info')
                else:
                    # Backward compatibility with old format
                    audio_chunk = audio_entry
                    packet_info = None
                
                self.stats['audio_processed'] += 1
                
                # Add to pre-speech buffer
                self.pre_speech_buffer.append(audio_chunk)
                
                # Run coordinated VAD
                if packet_info:
                    # Use ESP32-P4 VAD coordination
                    vad_result = self.vad_coordinator.process_packet(
                        packet_info, 
                        audio_chunk.astype(np.float32) / 32768.0
                    )
                    is_speech = vad_result.decision in [VADDecision.SPEECH_DETECTED, VADDecision.SPEECH_START]
                    self.stats['vad_coordinated_decisions'] += 1
                    
                    # Log enhanced VAD information
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"VAD Coordination: {vad_result.decision.value}, "
                                    f"confidence: {vad_result.confidence:.3f}, "
                                    f"method: {vad_result.coordination_method}")
                else:
                    # Fallback to server-only VAD
                    is_speech, _ = self.vad.process_chunk(audio_chunk.astype(np.float32) / 32768.0)
                
                if is_speech:
                    self.stats['vad_detections'] += 1
                    
                    if not recording_started:
                        # Start recording - include pre-speech buffer
                        recorded_chunks.extend(list(self.pre_speech_buffer))
                        recording_started = True
                        speech_detected = True
                        logging.info("Speech detected - started recording")
                    
                    # Reset silence timer
                    silence_start = None
                else:
                    # No speech detected
                    if recording_started:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > silence_timeout:
                            # Silence timeout reached
                            logging.info("Silence timeout - stopping recording")
                            break
                
                # Add chunk to recording if we're recording
                if recording_started:
                    recorded_chunks.append(audio_chunk)
                
            except Exception as e:
                logging.error(f"Error in recording loop: {e}")
                break
        
        self.is_recording = False
        
        # Save recorded audio
        if recorded_chunks and speech_detected:
            self._save_audio(recorded_chunks, file_path)
            logging.info(f"Saved {len(recorded_chunks)} audio chunks to {file_path}")
        else:
            logging.warning("No speech detected during recording")
    
    def _save_audio(self, chunks: list, file_path: str):
        """Save recorded audio chunks to file."""
        try:
            # Concatenate all chunks
            audio_data = np.concatenate(chunks)
            
            # Save as WAV file
            import wave
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            logging.info(f"Audio saved: {file_path} ({len(audio_data)} samples)")
            
        except Exception as e:
            logging.error(f"Failed to save audio: {e}")
    
    def _on_device_connected(self, device: WirelessDevice):
        """Handle device connection."""
        logging.info(f"Wireless device connected: {device.device_id}")
        
        # Auto-select if we don't have an active device
        if not self.active_device:
            self._select_active_device()
    
    def _on_device_disconnected(self, device: WirelessDevice):
        """Handle device disconnection."""
        logging.warning(f"Wireless device disconnected: {device.device_id}")
        
        # If this was our active device, select a new one
        if self.active_device and self.active_device.device_id == device.device_id:
            self.active_device = None
            self._select_active_device()
            self.stats['device_switches'] += 1
    
    def _on_device_status(self, device: WirelessDevice):
        """Handle device status updates."""
        logging.debug(f"Device status update: {device.device_id} - {device.status}")
    
    def get_stats(self) -> dict:
        """Get network audio source statistics."""
        server_stats = self.audio_server.get_stats()
        device_stats = self.device_manager.get_stats()
        vad_stats = self.vad_coordinator.get_performance_metrics()
        protocol_stats = self.protocol_parser.get_stats()
        
        return {
            'network_audio': self.stats,
            'audio_server': server_stats,
            'device_manager': device_stats,
            'vad_coordination': vad_stats._asdict(),
            'esp32p4_protocol': protocol_stats,
            'active_device': self.active_device.device_id if self.active_device else None,
            'target_room': self.target_room,
            'is_recording': self.is_recording
        }
    
    def set_vad_fusion_strategy(self, strategy: VADFusionStrategy):
        """Change VAD fusion strategy at runtime."""
        self.vad_coordinator.set_fusion_strategy(strategy)
        logging.info(f"VAD fusion strategy updated to {strategy.value}")
    
    def provide_vad_feedback(self, is_correct: bool, decision_timestamp: float = None):
        """
        Provide feedback on VAD decision accuracy for learning.
        
        Args:
            is_correct: Whether the VAD decision was correct
            decision_timestamp: Timestamp of decision, or None for most recent
        """
        if decision_timestamp is None:
            decision_timestamp = time.time()
        
        self.vad_coordinator.provide_feedback(is_correct, decision_timestamp)
        logging.debug(f"VAD feedback provided: {'correct' if is_correct else 'incorrect'}")
    
    def get_esp32p4_device_states(self) -> dict:
        """Get ESP32-P4 device VAD states."""
        return self.vad_coordinator.get_device_states()
    
    def reset_vad_metrics(self):
        """Reset VAD coordination metrics."""
        self.vad_coordinator.reset_metrics()
        self.protocol_parser.reset_stats()
        logging.info("VAD coordination metrics reset")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create network audio source
    audio_source = NetworkAudioSource(target_room="Living Room")
    
    if audio_source.start():
        try:
            # Wait for devices
            time.sleep(5)
            
            # Show available devices
            devices = audio_source.get_available_devices()
            print(f"Available devices: {devices}")
            
            # Test recording
            if devices:
                print("Testing network audio recording...")
                success, error = audio_source.record_audio(
                    "test_network_recording.wav",
                    max_duration=10.0,
                    silence_timeout=3.0
                )
                print(f"Recording result: {success}, Error: {error}")
                
                # Show stats
                stats = audio_source.get_stats()
                print(f"Stats: {stats}")
            
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            audio_source.stop()
    
    print("Network audio source test complete")