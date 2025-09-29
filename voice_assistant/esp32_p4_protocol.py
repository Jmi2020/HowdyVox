#!/usr/bin/env python3

import struct
import logging
import time
from typing import Optional, Tuple, NamedTuple
from enum import IntFlag
import numpy as np

class ESP32P4VADFlags(IntFlag):
    """ESP32-P4 VAD flags for enhanced voice activity detection."""
    VOICE_ACTIVE = 0x01
    SPEECH_START = 0x02  
    SPEECH_END = 0x04
    HIGH_CONFIDENCE = 0x08

class ESP32P4WakeWordFlags(IntFlag):
    """ESP32-P4 wake word detection flags for Porcupine integration."""
    WAKE_WORD_DETECTED = 0x01
    WAKE_WORD_END = 0x02
    HIGH_CONFIDENCE_WAKE = 0x04
    MULTIPLE_KEYWORDS = 0x08
    WAKE_WORD_VALIDATED = 0x10  # Server validated the wake word
    WAKE_WORD_REJECTED = 0x20   # Server rejected the wake word
    RESERVED_1 = 0x40
    RESERVED_2 = 0x80

class ESP32P4AudioHeader(NamedTuple):
    """Basic ESP32-P4 UDP audio packet header (12 bytes)."""
    sequence: int
    sample_count: int
    sample_rate: int
    channels: int
    bits_per_sample: int
    flags: int

class ESP32P4VADHeader(NamedTuple):
    """Enhanced ESP32-P4 VAD header extension (12 bytes)."""
    version: int
    vad_flags: int
    vad_confidence: int
    detection_quality: int
    max_amplitude: int
    noise_floor: int
    zero_crossing_rate: int
    snr_db_scaled: int
    reserved: int

class ESP32P4WakeWordHeader(NamedTuple):
    """ESP32-P4 wake word detection header extension (12 bytes)."""
    version: int
    wake_word_flags: int
    wake_confidence: int
    keyword_id: int
    detection_start_ms: int
    detection_duration_ms: int
    wake_word_quality: int
    validation_confidence: int  # Server validation confidence
    reserved: int

class ESP32P4PacketInfo(NamedTuple):
    """Complete ESP32-P4 packet information."""
    basic_header: ESP32P4AudioHeader
    vad_header: Optional[ESP32P4VADHeader]
    wake_word_header: Optional[ESP32P4WakeWordHeader]
    audio_data: np.ndarray
    raw_data: bytes
    timestamp: float
    source_addr: Tuple[str, int]

class ESP32P4ProtocolParser:
    """
    Enhanced UDP packet parser for ESP32-P4 devices with VAD extensions.
    
    Handles both basic UDP audio packets and enhanced packets with 12-byte VAD headers.
    Provides backward compatibility with basic UDP audio while enabling advanced
    edge VAD coordination for ESP32-P4 devices.
    """
    
    # Packet format constants
    BASIC_HEADER_SIZE = 12
    VAD_HEADER_SIZE = 12
    WAKE_WORD_HEADER_SIZE = 12
    ENHANCED_HEADER_SIZE = BASIC_HEADER_SIZE + VAD_HEADER_SIZE
    WAKE_WORD_HEADER_SIZE_TOTAL = BASIC_HEADER_SIZE + VAD_HEADER_SIZE + WAKE_WORD_HEADER_SIZE
    
    # Protocol versions
    VERSION_BASIC = 0x01
    VERSION_ENHANCED = 0x02
    VERSION_WAKE_WORD = 0x03  # Supports wake word detection
    
    def __init__(self):
        """Initialize the ESP32-P4 protocol parser."""
        self.stats = {
            'total_packets': 0,
            'basic_packets': 0,
            'enhanced_packets': 0,
            'wake_word_packets': 0,
            'parse_errors': 0,
            'invalid_headers': 0,
            'version_mismatches': 0
        }
        
        # Sequence tracking for packet loss detection
        self.device_sequences = {}  # device_id -> last_sequence
        
        logging.info("ESP32-P4 protocol parser initialized")
    
    def parse_packet(self, 
                    raw_data: bytes, 
                    source_addr: Tuple[str, int]) -> Optional[ESP32P4PacketInfo]:
        """
        Parse ESP32-P4 UDP audio packet with optional VAD extensions.
        
        Args:
            raw_data: Raw UDP packet data
            source_addr: Source IP and port tuple
            
        Returns:
            ESP32P4PacketInfo if parsing successful, None otherwise
        """
        self.stats['total_packets'] += 1
        
        try:
            # Check minimum packet size
            if len(raw_data) < self.BASIC_HEADER_SIZE:
                logging.warning(f"Packet too small: {len(raw_data)} bytes")
                self.stats['invalid_headers'] += 1
                return None
            
            # Parse basic header
            basic_header = self._parse_basic_header(raw_data[:self.BASIC_HEADER_SIZE])
            if not basic_header:
                return None
            
            # Determine packet type and parse headers
            vad_header = None
            wake_word_header = None
            audio_offset = self.BASIC_HEADER_SIZE
            
            # Check for wake word packet (has all three headers)
            if len(raw_data) >= self.WAKE_WORD_HEADER_SIZE_TOTAL:
                # Try to parse VAD header first
                vad_data = raw_data[self.BASIC_HEADER_SIZE:self.ENHANCED_HEADER_SIZE]
                potential_vad = self._parse_vad_header(vad_data)
                
                if potential_vad and potential_vad.version == self.VERSION_WAKE_WORD:
                    # Parse wake word header
                    wake_word_data = raw_data[self.ENHANCED_HEADER_SIZE:self.WAKE_WORD_HEADER_SIZE_TOTAL]
                    potential_wake_word = self._parse_wake_word_header(wake_word_data)
                    
                    if potential_wake_word and potential_wake_word.version == self.VERSION_WAKE_WORD:
                        vad_header = potential_vad
                        wake_word_header = potential_wake_word
                        audio_offset = self.WAKE_WORD_HEADER_SIZE_TOTAL
                        self.stats['enhanced_packets'] += 1
                        if 'wake_word_packets' not in self.stats:
                            self.stats['wake_word_packets'] = 0
                        self.stats['wake_word_packets'] += 1
            
            # Check for enhanced packet with VAD extension (but no wake word)
            if vad_header is None and len(raw_data) >= self.ENHANCED_HEADER_SIZE:
                # Try to parse VAD header
                vad_data = raw_data[self.BASIC_HEADER_SIZE:self.ENHANCED_HEADER_SIZE]
                potential_vad = self._parse_vad_header(vad_data)
                
                # Validate VAD header
                if potential_vad and potential_vad.version == self.VERSION_ENHANCED:
                    vad_header = potential_vad
                    audio_offset = self.ENHANCED_HEADER_SIZE
                    self.stats['enhanced_packets'] += 1
            
            # If no enhanced headers found, treat as basic packet
            if vad_header is None:
                self.stats['basic_packets'] += 1
            
            # Extract audio data
            audio_data = self._extract_audio_data(
                raw_data[audio_offset:], 
                basic_header
            )
            
            if audio_data is None:
                return None
            
            # Track sequence for packet loss detection
            self._track_sequence(source_addr, basic_header.sequence)
            
            # Create packet info
            packet_info = ESP32P4PacketInfo(
                basic_header=basic_header,
                vad_header=vad_header,
                wake_word_header=wake_word_header,
                audio_data=audio_data,
                raw_data=raw_data,
                timestamp=time.time(),
                source_addr=source_addr
            )
            
            return packet_info
            
        except Exception as e:
            logging.error(f"Error parsing ESP32-P4 packet: {e}")
            self.stats['parse_errors'] += 1
            return None
    
    def _parse_basic_header(self, header_data: bytes) -> Optional[ESP32P4AudioHeader]:
        """Parse the basic ESP32-P4 UDP audio header (supports 12-byte and 11-byte variants)."""
        # Try 12-byte variant first: <IHHBBH> (flags is 16-bit)
        for fmt in ('<IHHBBH', '<IHHBBB'):
            try:
                unpacked = struct.unpack(fmt, header_data[:struct.calcsize(fmt)])
                if fmt == '<IHHBBH':
                    sequence, sample_count, sample_rate, channels, bits_per_sample, flags = unpacked
                else:
                    sequence, sample_count, sample_rate, channels, bits_per_sample, flags8 = unpacked
                    flags = flags8  # 8-bit flags
                
                # Validate header fields
                if not self._validate_basic_header(sequence, sample_count, sample_rate, channels, bits_per_sample):
                    continue
                
                return ESP32P4AudioHeader(
                    sequence=sequence,
                    sample_count=sample_count,
                    sample_rate=sample_rate,
                    channels=channels,
                    bits_per_sample=bits_per_sample,
                    flags=flags
                )
            except struct.error:
                continue
        
        logging.debug("Basic header parse error: unsupported format")
        self.stats['invalid_headers'] += 1
        return None
    
    def _parse_vad_header(self, vad_data: bytes) -> Optional[ESP32P4VADHeader]:
        """Parse the enhanced 12-byte VAD header."""
        try:
            # Unpack VAD header: version, vad_flags, vad_confidence, detection_quality,
            # max_amplitude, noise_floor, zero_crossing_rate, snr_db_scaled, reserved
            unpacked = struct.unpack('<BBBBBHHBB', vad_data)
            
            if len(unpacked) != 9:
                logging.debug(f"VAD header unpacking mismatch: got {len(unpacked)} fields")
                return None
            
            (version, vad_flags, vad_confidence, detection_quality,
             max_amplitude, noise_floor, zero_crossing_rate, 
             snr_db_scaled, reserved) = unpacked
            
            # Validate version
            if version != self.VERSION_ENHANCED:
                self.stats['version_mismatches'] += 1
                return None
            
            # Validate VAD fields
            if not self._validate_vad_header(vad_confidence, detection_quality):
                return None
            
            return ESP32P4VADHeader(
                version=version,
                vad_flags=vad_flags,
                vad_confidence=vad_confidence,
                detection_quality=detection_quality,
                max_amplitude=max_amplitude,
                noise_floor=noise_floor,
                zero_crossing_rate=zero_crossing_rate,
                snr_db_scaled=snr_db_scaled,
                reserved=reserved
            )
            
        except struct.error as e:
            logging.debug(f"VAD header parse error: {e}")
            return None
    
    def _parse_wake_word_header(self, wake_word_data: bytes) -> Optional[ESP32P4WakeWordHeader]:
        """Parse the wake word detection header (12 bytes)."""
        try:
            # Unpack wake word header: version, wake_word_flags, wake_confidence, keyword_id,
            # detection_start_ms, detection_duration_ms, wake_word_quality, validation_confidence, reserved
            unpacked = struct.unpack('<BBBBHHBBB', wake_word_data)
            
            if len(unpacked) != 9:
                logging.debug(f"Wake word header unpacking mismatch: got {len(unpacked)} fields")
                return None
            
            (version, wake_word_flags, wake_confidence, keyword_id,
             detection_start_ms, detection_duration_ms, wake_word_quality,
             validation_confidence, reserved) = unpacked
            
            # Validate version
            if version != self.VERSION_WAKE_WORD:
                self.stats['version_mismatches'] += 1
                return None
            
            # Validate wake word fields
            if not self._validate_wake_word_header(wake_confidence, wake_word_quality):
                return None
            
            return ESP32P4WakeWordHeader(
                version=version,
                wake_word_flags=wake_word_flags,
                wake_confidence=wake_confidence,
                keyword_id=keyword_id,
                detection_start_ms=detection_start_ms,
                detection_duration_ms=detection_duration_ms,
                wake_word_quality=wake_word_quality,
                validation_confidence=validation_confidence,
                reserved=reserved
            )
            
        except struct.error as e:
            logging.debug(f"Wake word header parse error: {e}")
            return None
    
    def _extract_audio_data(self, 
                          audio_bytes: bytes, 
                          header: ESP32P4AudioHeader) -> Optional[np.ndarray]:
        """Extract and validate audio data from packet."""
        try:
            # Calculate expected audio data size
            bytes_per_sample = header.bits_per_sample // 8
            expected_size = header.sample_count * header.channels * bytes_per_sample
            
            # Validate audio data size
            if len(audio_bytes) < expected_size:
                logging.debug(f"Audio data too small: got {len(audio_bytes)}, "
                            f"expected {expected_size}")
                return None
            
            # Extract audio data
            audio_data = audio_bytes[:expected_size]
            
            # Convert to numpy array based on bit depth
            if header.bits_per_sample == 16:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif header.bits_per_sample == 8:
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                # Convert to signed 16-bit
                audio_array = ((audio_array.astype(np.int16) - 128) * 256)
            else:
                logging.warning(f"Unsupported bit depth: {header.bits_per_sample}")
                return None
            
            # Reshape for multi-channel audio
            if header.channels > 1:
                audio_array = audio_array.reshape(-1, header.channels)
            
            return audio_array
            
        except Exception as e:
            logging.debug(f"Audio extraction error: {e}")
            return None
    
    def _validate_basic_header(self, 
                             sequence: int, 
                             sample_count: int, 
                             sample_rate: int,
                             channels: int, 
                             bits_per_sample: int) -> bool:
        """Validate basic header fields for reasonable values."""
        # Sequence validation (allow wrap-around)
        if sequence > 0xFFFFFFFF:
            return False
        
        # Sample count validation (reasonable chunk sizes)
        if sample_count < 32 or sample_count > 2048:
            logging.debug(f"Invalid sample count: {sample_count}")
            return False
        
        # Sample rate validation
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if sample_rate not in valid_rates:
            logging.debug(f"Invalid sample rate: {sample_rate}")
            return False
        
        # Channel validation
        if channels < 1 or channels > 2:
            logging.debug(f"Invalid channel count: {channels}")
            return False
        
        # Bit depth validation
        if bits_per_sample not in [8, 16, 24, 32]:
            logging.debug(f"Invalid bit depth: {bits_per_sample}")
            return False
        
        return True
    
    def _validate_vad_header(self, confidence: int, quality: int) -> bool:
        """Validate VAD header fields."""
        # Confidence should be 0-255
        if confidence < 0 or confidence > 255:
            return False
        
        # Quality should be 0-255
        if quality < 0 or quality > 255:
            return False
        
        return True
    
    def _validate_wake_word_header(self, confidence: int, quality: int) -> bool:
        """Validate wake word header fields."""
        # Confidence should be 0-255
        if confidence < 0 or confidence > 255:
            return False
        
        # Quality should be 0-255
        if quality < 0 or quality > 255:
            return False
        
        return True
    
    def _track_sequence(self, source_addr: Tuple[str, int], sequence: int):
        """Track packet sequences for loss detection."""
        device_id = f"{source_addr[0]}:{source_addr[1]}"
        
        if device_id in self.device_sequences:
            last_seq = self.device_sequences[device_id]
            
            # Check for sequence gaps (accounting for wrap-around)
            if sequence != (last_seq + 1) % 0x100000000:
                gap = (sequence - last_seq - 1) % 0x100000000
                if gap < 1000:  # Reasonable gap threshold
                    logging.debug(f"Packet loss detected for {device_id}: "
                                f"gap of {gap} packets")
        
        self.device_sequences[device_id] = sequence
    
    def is_enhanced_packet(self, packet_info: ESP32P4PacketInfo) -> bool:
        """Check if packet has enhanced VAD information."""
        return packet_info.vad_header is not None
    
    def is_wake_word_packet(self, packet_info: ESP32P4PacketInfo) -> bool:
        """Check if packet has wake word detection information."""
        return packet_info.wake_word_header is not None
    
    def has_wake_word_detected(self, packet_info: ESP32P4PacketInfo) -> bool:
        """Check if packet indicates a wake word was detected."""
        if not self.is_wake_word_packet(packet_info):
            return False
        
        wake_flags = ESP32P4WakeWordFlags(packet_info.wake_word_header.wake_word_flags)
        return bool(wake_flags & ESP32P4WakeWordFlags.WAKE_WORD_DETECTED)
    
    def get_vad_state(self, packet_info: ESP32P4PacketInfo) -> dict:
        """Extract VAD state information from enhanced packet."""
        if not self.is_enhanced_packet(packet_info):
            return {
                'has_vad': False,
                'voice_active': False,
                'confidence': 0.0,
                'quality': 0.0
            }
        
        vad = packet_info.vad_header
        flags = ESP32P4VADFlags(vad.vad_flags)
        
        return {
            'has_vad': True,
            'voice_active': bool(flags & ESP32P4VADFlags.VOICE_ACTIVE),
            'speech_start': bool(flags & ESP32P4VADFlags.SPEECH_START),
            'speech_end': bool(flags & ESP32P4VADFlags.SPEECH_END),
            'high_confidence': bool(flags & ESP32P4VADFlags.HIGH_CONFIDENCE),
            'confidence': vad.vad_confidence / 255.0,  # Normalize to 0-1
            'quality': vad.detection_quality / 255.0,   # Normalize to 0-1
            'max_amplitude': vad.max_amplitude,
            'noise_floor': vad.noise_floor,
            'zero_crossing_rate': vad.zero_crossing_rate,
            'snr_db': vad.snr_db_scaled / 2.0  # Unscale SNR
        }
    
    def get_wake_word_state(self, packet_info: ESP32P4PacketInfo) -> dict:
        """Extract wake word state information from wake word packet."""
        if not self.is_wake_word_packet(packet_info):
            return {
                'has_wake_word': False,
                'wake_detected': False,
                'confidence': 0.0,
                'keyword_id': None,
                'duration_ms': 0
            }
        
        wake_word = packet_info.wake_word_header
        flags = ESP32P4WakeWordFlags(wake_word.wake_word_flags)
        
        return {
            'has_wake_word': True,
            'wake_detected': bool(flags & ESP32P4WakeWordFlags.WAKE_WORD_DETECTED),
            'wake_end': bool(flags & ESP32P4WakeWordFlags.WAKE_WORD_END),
            'high_confidence_wake': bool(flags & ESP32P4WakeWordFlags.HIGH_CONFIDENCE_WAKE),
            'multiple_keywords': bool(flags & ESP32P4WakeWordFlags.MULTIPLE_KEYWORDS),
            'wake_validated': bool(flags & ESP32P4WakeWordFlags.WAKE_WORD_VALIDATED),
            'wake_rejected': bool(flags & ESP32P4WakeWordFlags.WAKE_WORD_REJECTED),
            'confidence': wake_word.wake_confidence / 255.0,  # Normalize to 0-1
            'quality': wake_word.wake_word_quality / 255.0,   # Normalize to 0-1
            'keyword_id': wake_word.keyword_id,
            'detection_start_ms': wake_word.detection_start_ms,
            'detection_duration_ms': wake_word.detection_duration_ms,
            'validation_confidence': wake_word.validation_confidence / 255.0  # Server validation
        }
    
    def get_audio_info(self, packet_info: ESP32P4PacketInfo) -> dict:
        """Get audio format information from packet."""
        header = packet_info.basic_header
        return {
            'sample_rate': header.sample_rate,
            'channels': header.channels,
            'bits_per_sample': header.bits_per_sample,
            'sample_count': header.sample_count,
            'sequence': header.sequence,
            'duration_ms': (header.sample_count / header.sample_rate) * 1000
        }
    
    def get_stats(self) -> dict:
        """Get protocol parser statistics."""
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats['total_packets'] > 0:
            stats['enhanced_ratio'] = (stats['enhanced_packets'] / 
                                     stats['total_packets'])
            stats['error_rate'] = (stats['parse_errors'] / 
                                 stats['total_packets'])
        else:
            stats['enhanced_ratio'] = 0.0
            stats['error_rate'] = 0.0
        
        stats['tracked_devices'] = len(self.device_sequences)
        
        return stats
    
    def reset_stats(self):
        """Reset parser statistics."""
        self.stats = {
            'total_packets': 0,
            'basic_packets': 0,
            'enhanced_packets': 0,
            'wake_word_packets': 0,
            'parse_errors': 0,
            'invalid_headers': 0,
            'version_mismatches': 0
        }
        logging.info("Protocol parser statistics reset")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create parser
    parser = ESP32P4ProtocolParser()
    
    # Create test enhanced packet
    def create_test_packet():
        # Basic header
        basic_data = struct.pack('<IHHBBB', 
                               12345,      # sequence
                               512,        # sample_count
                               16000,      # sample_rate
                               1,          # channels
                               16,         # bits_per_sample
                               0)          # flags (padded)
        
        # VAD header (for wake word packet, use version 0x03)
        vad_data = struct.pack('<BBBBBHHBB',
                             0x03,        # version (VERSION_WAKE_WORD)
                             0x09,        # vad_flags (VOICE_ACTIVE | HIGH_CONFIDENCE)
                             200,         # vad_confidence
                             180,         # detection_quality
                             32000,       # max_amplitude
                             1500,        # noise_floor
                             450,         # zero_crossing_rate
                             24,          # snr_db_scaled (12dB * 2)
                             0)           # reserved
        
        # Wake word header
        wake_word_data = struct.pack('<BBBBHHBBB',
                                   0x03,        # version (VERSION_WAKE_WORD)
                                   0x05,        # wake_word_flags (WAKE_WORD_DETECTED | HIGH_CONFIDENCE_WAKE)
                                   220,         # wake_confidence
                                   1,           # keyword_id (Hey Howdy = 1)
                                   1234,        # detection_start_ms
                                   500,         # detection_duration_ms (500ms)
                                   190,         # wake_word_quality
                                   0,           # validation_confidence (not yet validated)
                                   0)           # reserved
        
        # Audio data (512 samples of 16-bit audio)
        audio_data = np.random.randint(-32768, 32767, 512, dtype=np.int16).tobytes()
        
        return basic_data + vad_data + wake_word_data + audio_data
    
    # Test parsing
    test_packet = create_test_packet()
    source_addr = ("192.168.1.100", 8000)
    
    packet_info = parser.parse_packet(test_packet, source_addr)
    
    if packet_info:
        print("Packet parsed successfully!")
        print(f"Enhanced packet: {parser.is_enhanced_packet(packet_info)}")
        print(f"Wake word packet: {parser.is_wake_word_packet(packet_info)}")
        print(f"Wake word detected: {parser.has_wake_word_detected(packet_info)}")
        print(f"VAD state: {parser.get_vad_state(packet_info)}")
        print(f"Wake word state: {parser.get_wake_word_state(packet_info)}")
        print(f"Audio info: {parser.get_audio_info(packet_info)}")
        print(f"Audio data shape: {packet_info.audio_data.shape}")
    else:
        print("Failed to parse packet")
    
    # Show statistics
    print(f"Parser stats: {parser.get_stats()}")
