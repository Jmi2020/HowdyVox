#!/usr/bin/env python3
"""
RTP Audio Receiver
Receives RTP/UDP audio stream from ESP32-P4 device and decodes to PCM

This module receives G.711 μ-law encoded RTP packets on UDP port 5004,
decodes them to PCM, and provides the audio data to HowdyTTS STT pipeline.
"""

import socket
import struct
import audioop
import logging
import threading
import time
from typing import Optional, Callable
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RTPPacket:
    """
    RTP packet structure (RFC 3550)
    """
    version: int  # RTP version (should be 2)
    padding: bool  # Padding flag
    extension: bool  # Extension flag
    csrc_count: int  # CSRC count
    marker: bool  # Marker bit
    payload_type: int  # Payload type (0 = μ-law)
    sequence_number: int  # Sequence number
    timestamp: int  # RTP timestamp
    ssrc: int  # Synchronization source identifier
    payload: bytes  # Audio payload (μ-law encoded)


class RTPReceiver:
    """
    Receives and decodes RTP audio packets from ESP32-P4 device
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5004,
                 audio_callback: Optional[Callable[[bytes], None]] = None):
        """
        Initialize RTP receiver

        Args:
            host: IP address to bind to (0.0.0.0 for all interfaces)
            port: UDP port to receive RTP packets (default 5004)
            audio_callback: Callback function for decoded PCM audio
        """
        self.host = host
        self.port = port
        self.audio_callback = audio_callback

        # Socket
        self.socket: Optional[socket.socket] = None

        # State
        self.running = False
        self.receive_thread: Optional[threading.Thread] = None

        # Packet tracking
        self.last_sequence = None
        self.expected_sequence = 0
        self.packets_received = 0
        self.packets_lost = 0
        self.packets_out_of_order = 0

        # Audio configuration
        self.sample_rate = 16000
        self.channels = 1
        self.frame_samples = 320  # 20ms @ 16kHz

        # Jitter buffer (simple implementation)
        self.jitter_buffer = deque(maxlen=8)  # 160ms max buffering
        self.jitter_buffer_enabled = True

        logger.info(f"RTP receiver initialized: {host}:{port}")

    def start(self):
        """
        Start RTP receiver
        """
        if self.running:
            logger.warning("RTP receiver already running")
            return

        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(1.0)  # 1 second timeout for clean shutdown

            logger.info(f"✓ RTP receiver listening on {self.host}:{self.port}")

            # Start receive thread
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()

            logger.info("✓ RTP receiver started")

        except Exception as e:
            logger.error(f"Failed to start RTP receiver: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            raise

    def stop(self):
        """
        Stop RTP receiver
        """
        if not self.running:
            return

        logger.info("Stopping RTP receiver...")
        self.running = False

        # Wait for receive thread to finish
        if self.receive_thread:
            self.receive_thread.join(timeout=2.0)
            self.receive_thread = None

        # Close socket
        if self.socket:
            self.socket.close()
            self.socket = None

        logger.info("✓ RTP receiver stopped")
        self.print_stats()

    def _receive_loop(self):
        """
        Main receive loop - runs in separate thread
        """
        logger.info("RTP receive loop started")

        while self.running:
            try:
                # Receive RTP packet
                data, addr = self.socket.recvfrom(2048)

                if not data:
                    continue

                # Parse RTP packet
                packet = self._parse_rtp_packet(data)

                if not packet:
                    continue

                # Track sequence numbers
                self._track_sequence(packet.sequence_number)

                # Decode μ-law to PCM
                pcm_data = self._decode_ulaw(packet.payload)

                # Handle jitter buffer
                if self.jitter_buffer_enabled:
                    self._jitter_buffer_add(packet.sequence_number, pcm_data)
                    pcm_data = self._jitter_buffer_get()

                    if not pcm_data:
                        continue

                # Call audio callback
                if self.audio_callback and pcm_data:
                    self.audio_callback(pcm_data)

            except socket.timeout:
                # Timeout is normal - allows clean shutdown check
                continue
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    logger.error(f"Error in RTP receive loop: {e}")
                    time.sleep(0.1)

        logger.info("RTP receive loop exited")

    def _parse_rtp_packet(self, data: bytes) -> Optional[RTPPacket]:
        """
        Parse RTP packet according to RFC 3550

        Args:
            data: Raw packet data

        Returns:
            RTPPacket object or None if invalid
        """
        if len(data) < 12:
            logger.warning(f"Packet too short: {len(data)} bytes")
            return None

        try:
            # Parse RTP header (12 bytes)
            byte0 = data[0]
            byte1 = data[1]

            version = (byte0 >> 6) & 0x03
            padding = bool((byte0 >> 5) & 0x01)
            extension = bool((byte0 >> 4) & 0x01)
            csrc_count = byte0 & 0x0F

            marker = bool((byte1 >> 7) & 0x01)
            payload_type = byte1 & 0x7F

            sequence_number = struct.unpack('!H', data[2:4])[0]
            timestamp = struct.unpack('!I', data[4:8])[0]
            ssrc = struct.unpack('!I', data[8:12])[0]

            # Validate RTP version
            if version != 2:
                logger.warning(f"Invalid RTP version: {version}")
                return None

            # Validate payload type (0 = μ-law)
            if payload_type != 0:
                logger.warning(f"Unsupported payload type: {payload_type}")
                return None

            # Extract payload (skip CSRC identifiers if present)
            header_len = 12 + (csrc_count * 4)
            payload = data[header_len:]

            packet = RTPPacket(
                version=version,
                padding=padding,
                extension=extension,
                csrc_count=csrc_count,
                marker=marker,
                payload_type=payload_type,
                sequence_number=sequence_number,
                timestamp=timestamp,
                ssrc=ssrc,
                payload=payload
            )

            self.packets_received += 1
            return packet

        except Exception as e:
            logger.error(f"Failed to parse RTP packet: {e}")
            return None

    def _track_sequence(self, sequence_number: int):
        """
        Track sequence numbers to detect packet loss

        Args:
            sequence_number: Current packet sequence number
        """
        if self.last_sequence is not None:
            expected = (self.last_sequence + 1) & 0xFFFF  # Wrap at 65535

            if sequence_number != expected:
                if sequence_number > expected:
                    lost = sequence_number - expected
                    self.packets_lost += lost
                    logger.warning(f"Packet loss detected: {lost} packets (seq {expected}-{sequence_number-1})")
                else:
                    self.packets_out_of_order += 1
                    logger.debug(f"Out of order packet: {sequence_number} (expected {expected})")

        self.last_sequence = sequence_number

    def _decode_ulaw(self, ulaw_data: bytes) -> bytes:
        """
        Decode G.711 μ-law to 16-bit PCM

        Args:
            ulaw_data: μ-law encoded audio

        Returns:
            16-bit PCM audio data
        """
        try:
            # Use Python's audioop module for μ-law decoding
            pcm_data = audioop.ulaw2lin(ulaw_data, 2)  # 2 = 16-bit samples
            return pcm_data
        except Exception as e:
            logger.error(f"Failed to decode μ-law: {e}")
            return b''

    def _jitter_buffer_add(self, sequence: int, pcm_data: bytes):
        """
        Add packet to jitter buffer (simple implementation)

        Args:
            sequence: Packet sequence number
            pcm_data: PCM audio data
        """
        self.jitter_buffer.append((sequence, pcm_data))

    def _jitter_buffer_get(self) -> Optional[bytes]:
        """
        Get next packet from jitter buffer

        Returns:
            PCM audio data or None if buffer not ready
        """
        # Simple strategy: wait for buffer to have at least 2 packets
        # then always return oldest packet
        if len(self.jitter_buffer) < 2:
            return None

        # Return oldest packet (FIFO)
        sequence, pcm_data = self.jitter_buffer.popleft()
        return pcm_data

    def get_stats(self) -> dict:
        """
        Get receiver statistics

        Returns:
            Dictionary of statistics
        """
        return {
            'packets_received': self.packets_received,
            'packets_lost': self.packets_lost,
            'packets_out_of_order': self.packets_out_of_order,
            'jitter_buffer_size': len(self.jitter_buffer),
            'sample_rate': self.sample_rate,
            'channels': self.channels
        }

    def print_stats(self):
        """
        Print receiver statistics
        """
        stats = self.get_stats()
        logger.info("RTP Receiver Statistics:")
        logger.info(f"  Packets received: {stats['packets_received']}")
        logger.info(f"  Packets lost: {stats['packets_lost']}")
        logger.info(f"  Packets out of order: {stats['packets_out_of_order']}")
        logger.info(f"  Jitter buffer size: {stats['jitter_buffer_size']}")

        if stats['packets_received'] > 0:
            loss_rate = (stats['packets_lost'] / (stats['packets_received'] + stats['packets_lost'])) * 100
            logger.info(f"  Packet loss rate: {loss_rate:.2f}%")


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    def audio_callback(pcm_data: bytes):
        logger.info(f"Received PCM audio: {len(pcm_data)} bytes")

    receiver = RTPReceiver(audio_callback=audio_callback)

    try:
        receiver.start()
        logger.info("RTP receiver running - press Ctrl+C to stop")

        # Run until interrupted
        while True:
            time.sleep(1)
            stats = receiver.get_stats()
            logger.info(f"Stats: {stats['packets_received']} received, {stats['packets_lost']} lost")

    except KeyboardInterrupt:
        logger.info("\nStopping...")
        receiver.stop()