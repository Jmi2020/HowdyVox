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

        # Transmission statistics
        self.transmission_stats = {
            'total_sessions': 0,
            'total_chunks_sent': 0,
            'total_bytes_sent': 0,
            'failed_transmissions': 0,
            'per_device': {}
        }
        
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

        # Print transmission statistics
        self.print_transmission_stats()
    
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
                await self.devices[device_id].send(json.dumps(confirm_msg))
            
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
                logging.debug(f"📡 Pong sent to {device_id}")
            
            else:
                logging.warning(f"Unknown message type from {device_id}: {msg_type}")
        
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON from {device_id}: {message[:100]}")
        except Exception as e:
            logging.error(f"Error processing message from {device_id}: {e}")
    
    async def send_tts_audio(self, device_id: str, audio_data: bytes, session_id: str = None, use_opus: bool = True):
        """Send TTS audio to specific ESP32-P4 device in chunks to avoid SDIO buffer overflow.

        Args:
            device_id: Target ESP32-P4 device ID
            audio_data: PCM audio data (16-bit mono @ 16kHz)
            session_id: Optional session ID for tracking
            use_opus: If True, encode to Opus before sending (default, recommended for bandwidth)
        """
        if device_id not in self.devices:
            logging.warning(f"Device {device_id} not connected - cannot send TTS audio")
            self.transmission_stats['failed_transmissions'] += 1
            return False

        try:
            import base64

            # Encode to Opus if requested (reduces size by ~10x)
            if use_opus:
                try:
                    import opuslib

                    # Opus encoder configuration
                    SAMPLE_RATE = 16000
                    CHANNELS = 1
                    FRAME_SIZE = 320  # 20ms @ 16kHz
                    BITRATE = 24000  # 24 kbps for speech

                    # Create Opus encoder
                    encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
                    encoder.bitrate = BITRATE

                    # Encode PCM to Opus in frames
                    opus_frames = []
                    pcm_offset = 0
                    original_size = len(audio_data)

                    while pcm_offset < len(audio_data):
                        # Extract one frame (320 samples × 2 bytes = 640 bytes)
                        frame_end = min(pcm_offset + (FRAME_SIZE * 2), len(audio_data))
                        pcm_frame = audio_data[pcm_offset:frame_end]

                        # Pad last frame if needed
                        if len(pcm_frame) < FRAME_SIZE * 2:
                            pcm_frame += b'\x00' * (FRAME_SIZE * 2 - len(pcm_frame))

                        # Encode frame
                        opus_frame = encoder.encode(pcm_frame, FRAME_SIZE)
                        opus_frames.append(opus_frame)
                        pcm_offset = frame_end

                    # Concatenate Opus frames with 2-byte length headers for ESP32 parsing
                    # Format: [len_bytes(2)] [opus_frame] [len_bytes(2)] [opus_frame] ...
                    encoded_audio = b''
                    for opus_frame in opus_frames:
                        encoded_audio += len(opus_frame).to_bytes(2, 'little')
                        encoded_audio += opus_frame
                    audio_format = 'opus_16khz_mono'

                    raw_opus_size = sum(len(f) for f in opus_frames)
                    logging.info(f"🗜️  Opus encoding: {original_size} bytes PCM → {raw_opus_size} bytes Opus + {len(opus_frames)*2} bytes headers = {len(encoded_audio)} total ({raw_opus_size/original_size*100:.1f}% compression)")

                except ImportError:
                    logging.warning("opuslib not available, falling back to PCM")
                    encoded_audio = audio_data
                    audio_format = 'pcm_16bit_mono_16khz'
                    use_opus = False
                except Exception as e:
                    logging.error(f"Opus encoding failed: {e}, falling back to PCM")
                    encoded_audio = audio_data
                    audio_format = 'pcm_16bit_mono_16khz'
                    use_opus = False
            else:
                encoded_audio = audio_data
                audio_format = 'pcm_16bit_mono_16khz'

            # 1KB chunks - proven reliable for ESP32 WebSocket buffer
            # ESP32 WS buffer: 4096 bytes (CONFIG_WS_BUFFER_SIZE)
            # 1KB binary → 1.37KB base64 + ~300B JSON = ~1.7KB total (safe, tested)
            # ESP32 150-chunk PSRAM buffer provides smooth playback despite smaller chunks
            CHUNK_SIZE = 1024  # 1KB per chunk (proven reliable)
            total_bytes = len(encoded_audio)
            num_chunks = (total_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE
            session = session_id or f"tts_{int(time.time())}"

            # Initialize device stats if needed
            if device_id not in self.transmission_stats['per_device']:
                self.transmission_stats['per_device'][device_id] = {
                    'sessions': 0,
                    'chunks_sent': 0,
                    'bytes_sent': 0,
                    'failed': 0
                }

            # Track session start
            self.transmission_stats['total_sessions'] += 1
            self.transmission_stats['per_device'][device_id]['sessions'] += 1

            logging.info(f"📤 Sending TTS audio to {device_id}: {total_bytes} bytes ({audio_format}) in {num_chunks} chunks")

            # Send session start message first
            session_start_msg = {
                'type': 'tts_audio_start',
                'session_info': {
                    'session_id': session,
                    'estimated_duration_ms': int((total_bytes / 2) / 16000 * 1000) if not use_opus else 0,
                    'total_chunks_expected': num_chunks,
                    'audio_format': audio_format
                }
            }
            await self.devices[device_id].send(json.dumps(session_start_msg))
            logging.info(f"🎬 Sent TTS session start: {session}")

            # Track transmission timing
            transmission_start = time.time()

            # Send audio in chunks (matching ESP32's expected format)
            for chunk_index in range(num_chunks):
                start = chunk_index * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, total_bytes)
                chunk_data = encoded_audio[start:end]  # Use encoded_audio (Opus or PCM)
                chunk_b64 = base64.b64encode(chunk_data).decode('utf-8')

                # Calculate duration based on format
                if use_opus:
                    # Opus: approximate duration based on bitrate
                    # At 24 kbps: ~3 KB/sec, so 1 KB ≈ 333ms
                    chunk_duration_ms = int((len(chunk_data) / 3000) * 1000)
                else:
                    # PCM: samples = bytes / 2 (16-bit), duration = samples / sample_rate
                    chunk_duration_ms = int((len(chunk_data) / 2) / 16000 * 1000)

                tts_msg = {
                    'type': 'tts_audio_chunk',
                    'chunk_info': {
                        'session_id': session,
                        'chunk_sequence': chunk_index,  # ESP32 uses chunk_sequence not chunk_index
                        'chunk_size': len(chunk_data),
                        'is_final': (chunk_index == num_chunks - 1),
                        'audio_data': chunk_b64  # ESP32 expects audio_data INSIDE chunk_info
                    },
                    'timing': {
                        'chunk_start_time_ms': chunk_index * chunk_duration_ms,
                        'chunk_duration_ms': chunk_duration_ms
                    },
                    'audio_format': audio_format  # 'opus_16khz_mono' or 'pcm_16bit_mono_16khz'
                }

                await self.devices[device_id].send(json.dumps(tts_msg))

                # Track successful chunk transmission
                self.transmission_stats['total_chunks_sent'] += 1
                self.transmission_stats['total_bytes_sent'] += len(chunk_data)
                self.transmission_stats['per_device'][device_id]['chunks_sent'] += 1
                self.transmission_stats['per_device'][device_id]['bytes_sent'] += len(chunk_data)

                # NO DELAY - mirroring ESP32's reliable STT audio streaming pattern
                # ESP32 sends 640-byte chunks (20ms audio) with zero delay
                # ESP32 150-chunk PSRAM buffer (300KB, ~9s) handles continuous streaming
                # No artificial delays needed - let the PSRAM buffer do its job
                pass  # No delay between chunks

            # Calculate actual transmission time
            transmission_time_ms = int((time.time() - transmission_start) * 1000)

            # Send session end message
            session_end_msg = {
                'type': 'tts_audio_end',
                'session_summary': {
                    'session_id': session,
                    'total_chunks_sent': num_chunks,
                    'total_audio_bytes': total_bytes,
                    'actual_duration_ms': int((total_bytes / 2) / 16000 * 1000) if not use_opus else 0,
                    'transmission_time_ms': transmission_time_ms,
                    'return_to_listening': True
                }
            }
            await self.devices[device_id].send(json.dumps(session_end_msg))

            logging.info(f"🔊 Sent TTS audio to {device_id}: {total_bytes} bytes in {num_chunks} chunks (session: {session})")
            logging.info(f"🏁 Sent TTS session end: {session}")
            return True

        except Exception as e:
            logging.error(f"Failed to send TTS audio to {device_id}: {e}")
            self.transmission_stats['failed_transmissions'] += 1
            if device_id in self.transmission_stats['per_device']:
                self.transmission_stats['per_device'][device_id]['failed'] += 1
            return False
    
    def send_tts_audio_sync(self, device_id: str, audio_data: bytes, session_id: str = None, use_opus: bool = False, pre_encoded: bool = False):
        """Send TTS audio synchronously (thread-safe).

        Args:
            device_id: Target ESP32-P4 device ID
            audio_data: PCM audio data (16-bit mono @ 16kHz) OR pre-encoded Opus if pre_encoded=True
            session_id: Optional session ID for tracking
            use_opus: If True and pre_encoded=False, encode PCM to Opus (default)
            pre_encoded: If True, audio_data is already Opus-encoded (skips encoding)
        """
        if not self.loop or not self.running:
            return False

        # Handle pre-encoded Opus: skip encoding, set format correctly
        if pre_encoded:
            use_opus = False  # Don't encode again
            # TODO: Need to modify send_tts_audio to accept audio_format parameter
            # For now, log a warning
            logging.warning("Pre-encoded Opus support incomplete - audio may not play correctly")

        future = asyncio.run_coroutine_threadsafe(
            self.send_tts_audio(device_id, audio_data, session_id, use_opus),
            self.loop
        )

        try:
            # Timeout must accommodate: num_chunks × 20ms delay + network overhead
            # With Opus: ~35 chunks (was 343), so ~1s minimum, use 30s for safety
            # With PCM: ~343 chunks, ~8s minimum, use 30s for safety
            return future.result(timeout=30.0)
        except Exception as e:
            logging.error(f"Failed to send TTS audio sync to {device_id}: {e}")
            return False
    
    async def send_tts_audio_streaming(self, device_id: str, text: str, session_id: str = None):
        """
        Stream TTS audio as it's generated (natural rate-limiting).

        Generates TTS in small text chunks and sends each audio chunk immediately,
        providing natural pacing from TTS generation speed. Eliminates artificial
        delays and prevents ESP32 buffer overflow.

        Args:
            device_id: Target ESP32-P4 device ID
            text: Text to convert to speech
            session_id: Optional session ID for tracking

        Returns:
            bool: True if streaming succeeded, False otherwise
        """
        if device_id not in self.devices:
            logging.error(f"Device {device_id} not connected")
            return False

        try:
            import re
            import base64
            import soundfile as sf
            from voice_assistant.kokoro_manager import KokoroManager
            from voice_assistant.config import Config

            # Get Kokoro TTS instance
            kokoro = KokoroManager.get_instance()
            session = session_id or f"tts_stream_{int(time.time())}"

            # Clean text and split into small chunks (sentences/phrases)
            # Split on sentence boundaries for natural pacing
            text = text.strip()
            sentence_chunks = re.split(r'([.!?]+\s+)', text)

            # Reconstruct with punctuation
            chunks = []
            for i in range(0, len(sentence_chunks)-1, 2):
                chunk = sentence_chunks[i] + (sentence_chunks[i+1] if i+1 < len(sentence_chunks) else '')
                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
            # Add remaining text if any
            if len(sentence_chunks) % 2 == 1 and sentence_chunks[-1].strip():
                chunks.append(sentence_chunks[-1].strip())

            if not chunks:
                logging.warning("No text chunks to generate")
                return False

            logging.info(f"📝 Streaming TTS for {device_id}: '{text[:50]}...' ({len(chunks)} sentence chunks)")

            # Send session start
            session_start_msg = {
                'type': 'tts_audio_start',
                'session_info': {
                    'session_id': session,
                    'total_chunks_expected': -1,  # Unknown - streaming mode
                    'audio_format': 'pcm_16bit_mono_16khz',
                    'streaming': True
                }
            }
            await self.devices[device_id].send(json.dumps(session_start_msg))

            total_bytes = 0
            total_chunks_sent = 0
            generation_start = time.time()

            # Generate and send each chunk immediately
            for chunk_idx, text_chunk in enumerate(chunks):
                try:
                    # Generate TTS for this chunk (natural rate-limiting here)
                    gen_start = time.time()
                    samples, sample_rate = kokoro.create(
                        text_chunk,
                        voice=Config.KOKORO_VOICE,
                        speed=Config.KOKORO_SPEED,
                        lang="en-us"
                    )
                    gen_time = time.time() - gen_start

                    # Convert to 16-bit PCM bytes
                    audio_data = (samples * 32767).astype(np.int16).tobytes()

                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        import io
                        from scipy import signal

                        # Write to temporary buffer
                        buffer = io.BytesIO()
                        sf.write(buffer, samples, sample_rate, format='WAV')
                        buffer.seek(0)

                        # Resample
                        resampled_samples = signal.resample(samples, int(len(samples) * 16000 / sample_rate))
                        audio_data = (resampled_samples * 32767).astype(np.int16).tobytes()

                    # Send this audio chunk immediately
                    # Split into 1KB WebSocket chunks for ESP32 buffer
                    CHUNK_SIZE = 1024
                    audio_bytes = len(audio_data)
                    num_ws_chunks = (audio_bytes + CHUNK_SIZE - 1) // CHUNK_SIZE

                    logging.info(f"🎵 Chunk {chunk_idx+1}/{len(chunks)}: '{text_chunk[:30]}...' → {audio_bytes}B audio ({gen_time:.2f}s TTS gen) → {num_ws_chunks} WS chunks")

                    for ws_chunk_idx in range(num_ws_chunks):
                        start_idx = ws_chunk_idx * CHUNK_SIZE
                        end_idx = min(start_idx + CHUNK_SIZE, audio_bytes)
                        chunk_data = audio_data[start_idx:end_idx]

                        # Send WebSocket chunk
                        chunk_msg = {
                            'type': 'tts_audio_chunk',
                            'chunk_info': {
                                'session_id': session,
                                'chunk_sequence': total_chunks_sent,
                                'chunk_data': base64.b64encode(chunk_data).decode('utf-8'),
                                'audio_format': 'pcm_16bit_mono_16khz',
                                'text_chunk_index': chunk_idx,
                                'is_last_in_text_chunk': (ws_chunk_idx == num_ws_chunks - 1)
                            }
                        }
                        await self.devices[device_id].send(json.dumps(chunk_msg))
                        total_chunks_sent += 1

                        # Rate-limiting: 30ms delay per chunk to match ESP32 playback
                        # ESP32 playback: 16kHz × 2 bytes = 32 bytes/ms
                        # 1KB chunk = 1024 bytes ÷ 32 bytes/ms = 32ms playback time
                        # Send every 30ms to stay ahead but not overwhelm buffer
                        if ws_chunk_idx < num_ws_chunks - 1:  # Skip delay on last chunk of sentence
                            await asyncio.sleep(0.030)  # 30ms delay

                    total_bytes += audio_bytes

                    # No additional delay between sentences - 30ms/chunk provides enough pacing

                except Exception as e:
                    logging.error(f"Error generating/sending chunk {chunk_idx}: {e}")
                    continue

            total_time = time.time() - generation_start

            # Send session end
            session_end_msg = {
                'type': 'tts_audio_end',
                'session_info': {
                    'session_id': session,
                    'total_chunks_sent': total_chunks_sent,
                    'total_audio_bytes': total_bytes,
                    'actual_duration_ms': int((total_bytes / 2) / 16000 * 1000),
                    'generation_time_ms': int(total_time * 1000),
                    'return_to_listening': True
                }
            }
            await self.devices[device_id].send(json.dumps(session_end_msg))

            logging.info(f"✅ Streaming TTS complete: {total_bytes}B in {total_chunks_sent} chunks ({total_time:.2f}s total)")
            return True

        except Exception as e:
            logging.error(f"Failed to stream TTS audio to {device_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def send_tts_audio_streaming_sync(self, device_id: str, text: str, session_id: str = None):
        """Send TTS audio using streaming generation (thread-safe)."""
        if not self.loop or not self.running:
            return False

        future = asyncio.run_coroutine_threadsafe(
            self.send_tts_audio_streaming(device_id, text, session_id),
            self.loop
        )

        try:
            # Generous timeout for streaming generation
            return future.result(timeout=60.0)
        except Exception as e:
            logging.error(f"Failed to stream TTS audio sync to {device_id}: {e}")
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

    def get_transmission_stats(self) -> Dict[str, Any]:
        """Get TTS transmission statistics."""
        return self.transmission_stats.copy()

    def print_transmission_stats(self):
        """Print TTS transmission statistics."""
        stats = self.transmission_stats
        logging.info("=" * 60)
        logging.info("TTS Transmission Statistics:")
        logging.info(f"  Total sessions: {stats['total_sessions']}")
        logging.info(f"  Total chunks sent: {stats['total_chunks_sent']}")
        logging.info(f"  Total bytes sent: {stats['total_bytes_sent']:,} ({stats['total_bytes_sent'] / 1024:.1f} KB)")
        logging.info(f"  Failed transmissions: {stats['failed_transmissions']}")

        if stats['per_device']:
            logging.info("\n  Per-Device Statistics:")
            for device_id, device_stats in stats['per_device'].items():
                logging.info(f"    {device_id}:")
                logging.info(f"      Sessions: {device_stats['sessions']}")
                logging.info(f"      Chunks sent: {device_stats['chunks_sent']}")
                logging.info(f"      Bytes sent: {device_stats['bytes_sent']:,} ({device_stats['bytes_sent'] / 1024:.1f} KB)")
                logging.info(f"      Failed: {device_stats['failed']}")
        logging.info("=" * 60)

    def reset_transmission_stats(self):
        """Reset transmission statistics."""
        self.transmission_stats = {
            'total_sessions': 0,
            'total_chunks_sent': 0,
            'total_bytes_sent': 0,
            'failed_transmissions': 0,
            'per_device': {}
        }
        logging.info("Transmission statistics reset")


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