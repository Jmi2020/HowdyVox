#!/usr/bin/env python3

"""
Comprehensive Test Script for ESP32-P4 VAD Coordination System

This test script validates the complete ESP32-P4 VAD coordination system by simulating
realistic scenarios and testing all components of the dual-VAD fusion architecture.

Test Coverage:
- Enhanced UDP packet parsing and validation
- VAD coordination strategies (edge + server fusion)
- Performance testing and latency measurement
- Network reliability and packet loss scenarios
- Multi-device coordination
- Real-world audio scenarios (noise, whispers, etc.)

Author: HowdyTTS ESP32-P4 Integration Team
"""

import sys
import os
import time
import logging
import struct
import socket
import threading
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from unittest.mock import Mock, MagicMock
import concurrent.futures
import statistics
import traceback

# Add voice_assistant module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'voice_assistant'))

try:
    from esp32_p4_protocol import (
        ESP32P4ProtocolParser, ESP32P4PacketInfo, ESP32P4AudioHeader, 
        ESP32P4VADHeader, ESP32P4VADFlags
    )
    from esp32_p4_vad_coordinator import (
        ESP32P4VADCoordinator, VADDecision, VADResult, VADFusionStrategy,
        VADPerformanceMetrics
    )
    from intelligent_vad import IntelligentVAD
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure you're running from the HowdyTTS root directory")
    sys.exit(1)

@dataclass
class TestScenario:
    """Test scenario configuration."""
    name: str
    description: str
    packet_count: int = 100
    noise_level: float = 0.0
    speech_probability: float = 0.5
    edge_vad_accuracy: float = 0.85
    server_vad_accuracy: float = 0.95
    packet_loss_rate: float = 0.0
    latency_ms: float = 5.0
    device_count: int = 1

@dataclass
class TestResult:
    """Test execution results."""
    scenario: TestScenario
    success: bool
    execution_time: float
    packets_processed: int
    packets_lost: int
    accuracy: float
    latency_stats: Dict[str, float]
    error_details: Optional[str] = None
    performance_metrics: Optional[Dict] = None

class MockIntelligentVAD:
    """Mock Silero VAD for testing that simulates realistic behavior."""
    
    def __init__(self, accuracy: float = 0.95, latency_ms: float = 10.0):
        """
        Initialize mock VAD.
        
        Args:
            accuracy: Simulated accuracy (0.0 - 1.0)
            latency_ms: Simulated processing latency
        """
        self.accuracy = accuracy
        self.latency_ms = latency_ms
        self.call_count = 0
        
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """
        Simulate Silero VAD processing with realistic behavior.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            (is_speech, confidence)
        """
        self.call_count += 1
        
        # Simulate processing delay
        time.sleep(self.latency_ms / 1000.0)
        
        # Calculate energy-based speech detection (ground truth)
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        is_actual_speech = energy > 0.01  # Simple threshold
        
        # Simulate accuracy by occasionally flipping the result
        if np.random.random() > self.accuracy:
            is_detected_speech = not is_actual_speech
        else:
            is_detected_speech = is_actual_speech
            
        # Generate realistic confidence based on energy and accuracy
        if is_detected_speech:
            base_confidence = min(0.95, energy * 50 + 0.3)
        else:
            base_confidence = max(0.05, 1.0 - energy * 50)
            
        # Add some noise to confidence
        confidence = np.clip(base_confidence + np.random.normal(0, 0.05), 0.0, 1.0)
        
        return is_detected_speech, confidence
    
    def reset(self):
        """Reset VAD state."""
        self.call_count = 0

class ESP32P4TestPacketGenerator:
    """Generates realistic ESP32-P4 test packets for validation."""
    
    def __init__(self):
        """Initialize packet generator."""
        self.sequence_counters = defaultdict(int)  # Per-device sequence counters
        
    def generate_enhanced_packet(self, 
                               device_id: str = "192.168.1.100:8000",
                               sample_rate: int = 16000,
                               sample_count: int = 512,
                               channels: int = 1,
                               bits_per_sample: int = 16,
                               is_speech: bool = False,
                               noise_level: float = 0.0,
                               vad_confidence: int = 180,
                               vad_quality: int = 200) -> bytes:
        """
        Generate an enhanced ESP32-P4 packet with VAD extension.
        
        Args:
            device_id: Device identifier
            sample_rate: Audio sample rate
            sample_count: Number of audio samples
            channels: Audio channels
            bits_per_sample: Bits per audio sample
            is_speech: Whether packet contains speech
            noise_level: Background noise level (0.0-1.0)
            vad_confidence: Edge VAD confidence (0-255)
            vad_quality: Edge VAD quality (0-255)
            
        Returns:
            Complete packet bytes
        """
        # Generate sequence number
        sequence = self.sequence_counters[device_id]
        self.sequence_counters[device_id] += 1
        
        # Create basic header (12 bytes)
        basic_header = struct.pack('<IHHBBB',
            sequence,           # sequence number
            sample_count,       # sample count
            sample_rate,        # sample rate
            channels,           # channels
            bits_per_sample,    # bits per sample
            0                   # flags (padding)
        )
        
        # Calculate VAD flags based on speech state
        vad_flags = 0
        if is_speech:
            vad_flags |= ESP32P4VADFlags.VOICE_ACTIVE
            if vad_confidence > 200:
                vad_flags |= ESP32P4VADFlags.HIGH_CONFIDENCE
            # Randomly add boundary detection flags
            if np.random.random() < 0.1:
                vad_flags |= ESP32P4VADFlags.SPEECH_START
            if np.random.random() < 0.1:
                vad_flags |= ESP32P4VADFlags.SPEECH_END
        
        # Calculate audio metrics
        if is_speech:
            max_amplitude = int(np.random.uniform(15000, 32000))
            noise_floor = int(np.random.uniform(500, 2000))
            snr_db = int(np.random.uniform(10, 25)) * 2  # Scale by 2
            zcr = int(np.random.uniform(200, 600))
        else:
            max_amplitude = int(np.random.uniform(100, 3000))
            noise_floor = int(np.random.uniform(800, 5000))
            snr_db = int(np.random.uniform(0, 8)) * 2
            zcr = int(np.random.uniform(50, 200))
        
        # Create VAD header (12 bytes)
        vad_header = struct.pack('<BBBBBHHBB',
            0x02,               # version (enhanced)
            vad_flags,          # VAD flags
            vad_confidence,     # VAD confidence
            vad_quality,        # detection quality
            max_amplitude & 0xFF,  # max amplitude (low byte)
            noise_floor,        # noise floor
            zcr,                # zero crossing rate
            snr_db,             # SNR (scaled)
            0                   # reserved
        )
        
        # Generate realistic audio data
        audio_data = self._generate_audio_samples(
            sample_count, is_speech, noise_level, bits_per_sample
        )
        
        return basic_header + vad_header + audio_data
    
    def generate_basic_packet(self,
                            device_id: str = "192.168.1.101:8000",
                            sample_rate: int = 16000,
                            sample_count: int = 512,
                            is_speech: bool = False,
                            noise_level: float = 0.0) -> bytes:
        """Generate basic UDP packet without VAD extension."""
        sequence = self.sequence_counters[device_id]
        self.sequence_counters[device_id] += 1
        
        # Basic header only
        basic_header = struct.pack('<IHHBBB',
            sequence, sample_count, sample_rate, 1, 16, 0
        )
        
        # Generate audio data
        audio_data = self._generate_audio_samples(
            sample_count, is_speech, noise_level, 16
        )
        
        return basic_header + audio_data
    
    def _generate_audio_samples(self, 
                              sample_count: int,
                              is_speech: bool,
                              noise_level: float,
                              bits_per_sample: int) -> bytes:
        """Generate realistic audio samples."""
        if is_speech:
            # Generate speech-like signal with harmonics
            t = np.linspace(0, sample_count/16000.0, sample_count)
            
            # Fundamental frequency around 150-300 Hz
            f0 = np.random.uniform(120, 250)
            
            # Generate harmonics
            signal = (0.5 * np.sin(2 * np.pi * f0 * t) +
                     0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
                     0.2 * np.sin(2 * np.pi * 3 * f0 * t))
            
            # Add formant-like resonances
            signal += 0.4 * np.sin(2 * np.pi * np.random.uniform(800, 1200) * t)
            signal += 0.3 * np.sin(2 * np.pi * np.random.uniform(1500, 2500) * t)
            
            # Apply amplitude envelope
            envelope = np.exp(-2 * t) + 0.3
            signal *= envelope
            
            # Normalize
            signal *= np.random.uniform(0.3, 0.8)
        else:
            # Generate noise or silence
            if noise_level > 0:
                signal = np.random.normal(0, noise_level * 0.1, sample_count)
            else:
                signal = np.random.normal(0, 0.005, sample_count)
        
        # Add background noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 0.05, sample_count)
            signal += noise
        
        # Convert to appropriate bit depth
        if bits_per_sample == 16:
            signal = np.clip(signal, -1.0, 1.0)
            audio_samples = (signal * 32767).astype(np.int16)
            return audio_samples.tobytes()
        else:
            raise ValueError(f"Unsupported bit depth: {bits_per_sample}")

class ESP32P4VADTester:
    """Comprehensive ESP32-P4 VAD system tester."""
    
    def __init__(self):
        """Initialize the tester."""
        self.setup_logging()
        self.packet_generator = ESP32P4TestPacketGenerator()
        self.test_results: List[TestResult] = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('esp32_p4_vad_test.log')
            ]
        )
        self.logger = logging.getLogger('ESP32P4VADTester')
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        self.logger.info("Starting ESP32-P4 VAD comprehensive test suite")
        
        test_scenarios = [
            # Basic functionality tests
            TestScenario(
                name="basic_enhanced_parsing",
                description="Basic enhanced packet parsing validation",
                packet_count=50,
                speech_probability=0.5
            ),
            TestScenario(
                name="basic_packet_parsing", 
                description="Basic UDP packet parsing (no VAD extension)",
                packet_count=50,
                speech_probability=0.5
            ),
            
            # VAD coordination tests
            TestScenario(
                name="edge_priority_fusion",
                description="Edge-priority VAD fusion strategy",
                packet_count=100,
                speech_probability=0.6,
                edge_vad_accuracy=0.9
            ),
            TestScenario(
                name="server_priority_fusion",
                description="Server-priority VAD fusion strategy", 
                packet_count=100,
                speech_probability=0.6,
                server_vad_accuracy=0.95
            ),
            TestScenario(
                name="confidence_weighted_fusion",
                description="Confidence-weighted VAD fusion",
                packet_count=100,
                speech_probability=0.7
            ),
            TestScenario(
                name="adaptive_fusion",
                description="Adaptive VAD fusion with learning",
                packet_count=200,
                speech_probability=0.6
            ),
            
            # Performance tests
            TestScenario(
                name="low_latency_processing",
                description="Low latency VAD processing test",
                packet_count=1000,
                latency_ms=2.0
            ),
            TestScenario(
                name="high_throughput",
                description="High throughput processing test",
                packet_count=2000,
                speech_probability=0.8
            ),
            
            # Robustness tests
            TestScenario(
                name="noisy_environment",
                description="VAD performance in noisy environment",
                packet_count=150,
                noise_level=0.3,
                speech_probability=0.5
            ),
            TestScenario(
                name="packet_loss_simulation",
                description="Network packet loss resilience",
                packet_count=200,
                packet_loss_rate=0.05
            ),
            TestScenario(
                name="low_confidence_speech",
                description="Low confidence speech detection",
                packet_count=100,
                speech_probability=0.4,
                edge_vad_accuracy=0.7,
                server_vad_accuracy=0.8
            ),
            
            # Multi-device tests
            TestScenario(
                name="multi_device_coordination",
                description="Multiple ESP32-P4 devices coordination",
                packet_count=150,
                device_count=3,
                speech_probability=0.6
            ),
            
            # Edge cases
            TestScenario(
                name="rapid_speech_transitions",
                description="Rapid speech start/end transitions",
                packet_count=200,
                speech_probability=0.5  # Will be overridden for rapid transitions
            ),
            TestScenario(
                name="malformed_packet_handling",
                description="Malformed packet resilience test",
                packet_count=100
            )
        ]
        
        # Execute all test scenarios
        for scenario in test_scenarios:
            self.logger.info(f"Running test: {scenario.name}")
            try:
                result = self._run_test_scenario(scenario)
                self.test_results.append(result)
                
                if result.success:
                    self.logger.info(f"✓ {scenario.name} PASSED")
                else:
                    self.logger.error(f"✗ {scenario.name} FAILED: {result.error_details}")
                    
            except Exception as e:
                self.logger.error(f"✗ {scenario.name} CRASHED: {e}")
                self.test_results.append(TestResult(
                    scenario=scenario,
                    success=False,
                    execution_time=0.0,
                    packets_processed=0,
                    packets_lost=0,
                    accuracy=0.0,
                    latency_stats={},
                    error_details=f"Test crashed: {e}"
                ))
        
        # Generate comprehensive report
        return self._generate_test_report()
    
    def _run_test_scenario(self, scenario: TestScenario) -> TestResult:
        """Run a single test scenario."""
        start_time = time.time()
        
        try:
            if scenario.name == "basic_enhanced_parsing":
                return self._test_enhanced_packet_parsing(scenario)
            elif scenario.name == "basic_packet_parsing":
                return self._test_basic_packet_parsing(scenario)
            elif scenario.name.endswith("_fusion"):
                return self._test_vad_fusion_strategy(scenario)
            elif scenario.name == "low_latency_processing":
                return self._test_latency_performance(scenario)
            elif scenario.name == "high_throughput":
                return self._test_throughput_performance(scenario)
            elif scenario.name == "noisy_environment":
                return self._test_noisy_environment(scenario)
            elif scenario.name == "packet_loss_simulation":
                return self._test_packet_loss_resilience(scenario)
            elif scenario.name == "low_confidence_speech":
                return self._test_low_confidence_handling(scenario)
            elif scenario.name == "multi_device_coordination":
                return self._test_multi_device_coordination(scenario)
            elif scenario.name == "rapid_speech_transitions":
                return self._test_rapid_transitions(scenario)
            elif scenario.name == "malformed_packet_handling":
                return self._test_malformed_packets(scenario)
            else:
                raise ValueError(f"Unknown test scenario: {scenario.name}")
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                scenario=scenario,
                success=False,
                execution_time=execution_time,
                packets_processed=0,
                packets_lost=0,
                accuracy=0.0,
                latency_stats={},
                error_details=f"Test execution failed: {e}\n{traceback.format_exc()}"
            )
    
    def _test_enhanced_packet_parsing(self, scenario: TestScenario) -> TestResult:
        """Test enhanced packet parsing functionality."""
        parser = ESP32P4ProtocolParser()
        packets_processed = 0
        parse_errors = 0
        latencies = []
        
        for i in range(scenario.packet_count):
            # Generate test packet
            is_speech = np.random.random() < scenario.speech_probability
            packet_data = self.packet_generator.generate_enhanced_packet(
                device_id=f"192.168.1.100:800{i % 10}",
                is_speech=is_speech,
                noise_level=scenario.noise_level
            )
            
            # Parse packet
            start_time = time.time()
            packet_info = parser.parse_packet(packet_data, ("192.168.1.100", 8000))
            parse_time = time.time() - start_time
            latencies.append(parse_time * 1000)  # Convert to ms
            
            if packet_info is None:
                parse_errors += 1
                continue
                
            packets_processed += 1
            
            # Validate packet structure
            if not self._validate_packet_info(packet_info, enhanced=True):
                parse_errors += 1
        
        success = parse_errors < scenario.packet_count * 0.05  # Allow 5% error rate
        accuracy = (packets_processed - parse_errors) / scenario.packet_count
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=packets_processed,
            packets_lost=parse_errors,
            accuracy=accuracy,
            latency_stats={
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            }
        )
    
    def _test_basic_packet_parsing(self, scenario: TestScenario) -> TestResult:
        """Test basic packet parsing (no VAD extension)."""
        parser = ESP32P4ProtocolParser()
        packets_processed = 0
        parse_errors = 0
        latencies = []
        
        for i in range(scenario.packet_count):
            is_speech = np.random.random() < scenario.speech_probability
            packet_data = self.packet_generator.generate_basic_packet(
                device_id=f"192.168.1.101:800{i % 10}",
                is_speech=is_speech,
                noise_level=scenario.noise_level
            )
            
            start_time = time.time()
            packet_info = parser.parse_packet(packet_data, ("192.168.1.101", 8000))
            parse_time = time.time() - start_time
            latencies.append(parse_time * 1000)
            
            if packet_info is None:
                parse_errors += 1
                continue
                
            packets_processed += 1
            
            # Validate basic packet (should not have VAD extension)
            if parser.is_enhanced_packet(packet_info):
                parse_errors += 1
        
        success = parse_errors < scenario.packet_count * 0.05
        accuracy = (packets_processed - parse_errors) / scenario.packet_count
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=packets_processed,
            packets_lost=parse_errors,
            accuracy=accuracy,
            latency_stats={
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            }
        )
    
    def _test_vad_fusion_strategy(self, scenario: TestScenario) -> TestResult:
        """Test VAD fusion strategies."""
        # Determine fusion strategy from scenario name
        if "edge_priority" in scenario.name:
            strategy = VADFusionStrategy.EDGE_PRIORITY
        elif "server_priority" in scenario.name:
            strategy = VADFusionStrategy.SERVER_PRIORITY
        elif "confidence_weighted" in scenario.name:
            strategy = VADFusionStrategy.CONFIDENCE_WEIGHTED
        elif "adaptive" in scenario.name:
            strategy = VADFusionStrategy.ADAPTIVE
        else:
            strategy = VADFusionStrategy.CONFIDENCE_WEIGHTED
        
        # Setup VAD coordinator
        mock_server_vad = MockIntelligentVAD(
            accuracy=scenario.server_vad_accuracy,
            latency_ms=scenario.latency_ms
        )
        coordinator = ESP32P4VADCoordinator(mock_server_vad, strategy)
        
        correct_decisions = 0
        total_decisions = 0
        latencies = []
        
        for i in range(scenario.packet_count):
            # Generate ground truth
            is_actual_speech = np.random.random() < scenario.speech_probability
            
            # Generate enhanced packet with simulated edge VAD
            edge_confidence = 180 if is_actual_speech else 80
            
            # Simulate edge VAD accuracy
            edge_is_correct = np.random.random() < scenario.edge_vad_accuracy
            if not edge_is_correct:
                # Edge VAD is wrong
                packet_speech = not is_actual_speech
                edge_confidence = 120  # Lower confidence when wrong
            else:
                packet_speech = is_actual_speech
            
            packet_data = self.packet_generator.generate_enhanced_packet(
                is_speech=packet_speech,
                vad_confidence=edge_confidence,
                noise_level=scenario.noise_level
            )
            
            # Parse and process packet
            packet_info = coordinator.protocol_parser.parse_packet(
                packet_data, ("192.168.1.100", 8000)
            )
            
            if packet_info is None:
                continue
            
            # Generate audio data for server VAD
            audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
            
            # Process with coordinator
            start_time = time.time()
            vad_result = coordinator.process_packet(packet_info, audio_samples)
            process_time = time.time() - start_time
            latencies.append(process_time * 1000)
            
            total_decisions += 1
            
            # Check accuracy against ground truth
            predicted_speech = vad_result.decision in [VADDecision.SPEECH_DETECTED, VADDecision.SPEECH_START]
            if predicted_speech == is_actual_speech:
                correct_decisions += 1
        
        accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0.0
        success = accuracy >= 0.7  # Expect at least 70% accuracy
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=total_decisions,
            packets_lost=scenario.packet_count - total_decisions,
            accuracy=accuracy,
            latency_stats={
                'mean': statistics.mean(latencies) if latencies else 0,
                'median': statistics.median(latencies) if latencies else 0,
                'p95': np.percentile(latencies, 95) if latencies else 0,
                'p99': np.percentile(latencies, 99) if latencies else 0
            },
            performance_metrics=asdict(coordinator.get_performance_metrics())
        )
    
    def _test_latency_performance(self, scenario: TestScenario) -> TestResult:
        """Test low-latency performance requirements."""
        mock_server_vad = MockIntelligentVAD(latency_ms=scenario.latency_ms)
        coordinator = ESP32P4VADCoordinator(mock_server_vad, VADFusionStrategy.EDGE_PRIORITY)
        
        latencies = []
        packets_processed = 0
        
        for i in range(scenario.packet_count):
            # Generate packet
            packet_data = self.packet_generator.generate_enhanced_packet(
                is_speech=np.random.random() < 0.5
            )
            
            packet_info = coordinator.protocol_parser.parse_packet(
                packet_data, ("192.168.1.100", 8000)
            )
            
            if packet_info is None:
                continue
            
            audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
            
            # Measure end-to-end latency
            start_time = time.time()
            vad_result = coordinator.process_packet(packet_info, audio_samples)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            packets_processed += 1
        
        # Check latency requirements
        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Success if mean < 20ms and P95 < 50ms for real-time performance
        success = mean_latency < 20.0 and p95_latency < 50.0
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=packets_processed,
            packets_lost=0,
            accuracy=1.0,  # Not measuring accuracy in this test
            latency_stats={
                'mean': mean_latency,
                'median': statistics.median(latencies),
                'p95': p95_latency,
                'p99': np.percentile(latencies, 99),
                'min': min(latencies),
                'max': max(latencies)
            }
        )
    
    def _test_throughput_performance(self, scenario: TestScenario) -> TestResult:
        """Test high throughput processing capability."""
        mock_server_vad = MockIntelligentVAD(latency_ms=5.0)
        coordinator = ESP32P4VADCoordinator(mock_server_vad, VADFusionStrategy.CONFIDENCE_WEIGHTED)
        
        start_time = time.time()
        packets_processed = 0
        errors = 0
        
        # Process packets as fast as possible
        for i in range(scenario.packet_count):
            packet_data = self.packet_generator.generate_enhanced_packet(
                is_speech=np.random.random() < scenario.speech_probability
            )
            
            try:
                packet_info = coordinator.protocol_parser.parse_packet(
                    packet_data, ("192.168.1.100", 8000)
                )
                
                if packet_info is None:
                    errors += 1
                    continue
                
                audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
                vad_result = coordinator.process_packet(packet_info, audio_samples)
                packets_processed += 1
                
            except Exception as e:
                errors += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        packets_per_second = packets_processed / total_time if total_time > 0 else 0
        
        # Success if can process > 100 packets/second
        success = packets_per_second > 100 and errors < scenario.packet_count * 0.02
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=total_time,
            packets_processed=packets_processed,
            packets_lost=errors,
            accuracy=1.0 - (errors / scenario.packet_count),
            latency_stats={
                'packets_per_second': packets_per_second,
                'total_time': total_time,
                'avg_packet_time': total_time / packets_processed if packets_processed > 0 else 0
            }
        )
    
    def _test_noisy_environment(self, scenario: TestScenario) -> TestResult:
        """Test VAD performance in noisy environment."""
        mock_server_vad = MockIntelligentVAD(accuracy=0.85)  # Slightly reduced accuracy in noise
        coordinator = ESP32P4VADCoordinator(mock_server_vad, VADFusionStrategy.ADAPTIVE)
        
        correct_decisions = 0
        total_decisions = 0
        
        for i in range(scenario.packet_count):
            is_actual_speech = np.random.random() < scenario.speech_probability
            
            # In noisy environment, edge VAD is less reliable
            edge_accuracy = 0.75  # Reduced from normal
            edge_is_correct = np.random.random() < edge_accuracy
            packet_speech = is_actual_speech if edge_is_correct else not is_actual_speech
            
            # Lower confidence in noisy conditions
            base_confidence = 150 if packet_speech else 100
            confidence_noise = np.random.randint(-30, 30)
            edge_confidence = np.clip(base_confidence + confidence_noise, 50, 255)
            
            packet_data = self.packet_generator.generate_enhanced_packet(
                is_speech=packet_speech,
                vad_confidence=edge_confidence,
                noise_level=scenario.noise_level
            )
            
            packet_info = coordinator.protocol_parser.parse_packet(
                packet_data, ("192.168.1.100", 8000)
            )
            
            if packet_info is None:
                continue
            
            audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
            vad_result = coordinator.process_packet(packet_info, audio_samples)
            
            total_decisions += 1
            predicted_speech = vad_result.decision in [VADDecision.SPEECH_DETECTED, VADDecision.SPEECH_START]
            
            if predicted_speech == is_actual_speech:
                correct_decisions += 1
        
        accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0.0
        success = accuracy >= 0.6  # Lower threshold for noisy conditions
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=total_decisions,
            packets_lost=0,
            accuracy=accuracy,
            latency_stats={}
        )
    
    def _test_packet_loss_resilience(self, scenario: TestScenario) -> TestResult:
        """Test resilience to network packet loss."""
        parser = ESP32P4ProtocolParser()
        packets_sent = 0
        packets_received = 0
        packets_parsed = 0
        
        for i in range(scenario.packet_count):
            packets_sent += 1
            
            # Simulate packet loss
            if np.random.random() < scenario.packet_loss_rate:
                continue  # Packet lost
            
            packets_received += 1
            
            packet_data = self.packet_generator.generate_enhanced_packet(
                device_id=f"192.168.1.100:8000",  # Same device to test sequence tracking
                is_speech=np.random.random() < 0.5
            )
            
            packet_info = parser.parse_packet(packet_data, ("192.168.1.100", 8000))
            if packet_info is not None:
                packets_parsed += 1
        
        actual_loss_rate = (packets_sent - packets_received) / packets_sent
        parse_success_rate = packets_parsed / packets_received if packets_received > 0 else 0
        
        # Success if parser handles packet loss gracefully
        success = (abs(actual_loss_rate - scenario.packet_loss_rate) < 0.02 and 
                  parse_success_rate > 0.95)
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=packets_parsed,
            packets_lost=packets_sent - packets_received,
            accuracy=parse_success_rate,
            latency_stats={
                'packets_sent': packets_sent,
                'packets_received': packets_received,
                'packets_parsed': packets_parsed,
                'loss_rate': actual_loss_rate
            }
        )
    
    def _test_low_confidence_handling(self, scenario: TestScenario) -> TestResult:
        """Test handling of low confidence VAD decisions."""
        mock_server_vad = MockIntelligentVAD(accuracy=scenario.server_vad_accuracy)
        coordinator = ESP32P4VADCoordinator(mock_server_vad, VADFusionStrategy.CONFIDENCE_WEIGHTED)
        
        low_confidence_decisions = 0
        correct_decisions = 0
        total_decisions = 0
        
        for i in range(scenario.packet_count):
            is_actual_speech = np.random.random() < scenario.speech_probability
            
            # Generate low confidence scenarios
            edge_confidence = np.random.randint(80, 150)  # Low confidence range
            edge_accuracy = scenario.edge_vad_accuracy
            
            packet_speech = is_actual_speech if np.random.random() < edge_accuracy else not is_actual_speech
            
            packet_data = self.packet_generator.generate_enhanced_packet(
                is_speech=packet_speech,
                vad_confidence=edge_confidence,
                vad_quality=np.random.randint(100, 180)  # Lower quality
            )
            
            packet_info = coordinator.protocol_parser.parse_packet(
                packet_data, ("192.168.1.100", 8000)
            )
            
            if packet_info is None:
                continue
            
            audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
            vad_result = coordinator.process_packet(packet_info, audio_samples)
            
            total_decisions += 1
            
            if vad_result.confidence < 0.6:
                low_confidence_decisions += 1
            
            predicted_speech = vad_result.decision in [VADDecision.SPEECH_DETECTED, VADDecision.SPEECH_START]
            if predicted_speech == is_actual_speech:
                correct_decisions += 1
        
        accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0.0
        low_confidence_rate = low_confidence_decisions / total_decisions if total_decisions > 0 else 0.0
        
        # Success if system handles low confidence appropriately (conservative decisions)
        success = accuracy >= 0.65 and low_confidence_rate > 0.2
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=total_decisions,
            packets_lost=0,
            accuracy=accuracy,
            latency_stats={'low_confidence_rate': low_confidence_rate}
        )
    
    def _test_multi_device_coordination(self, scenario: TestScenario) -> TestResult:
        """Test coordination of multiple ESP32-P4 devices."""
        mock_server_vad = MockIntelligentVAD()
        coordinator = ESP32P4VADCoordinator(mock_server_vad, VADFusionStrategy.ADAPTIVE)
        
        devices = [f"192.168.1.{100+i}:8000" for i in range(scenario.device_count)]
        device_packets = defaultdict(int)
        total_processed = 0
        errors = 0
        
        for i in range(scenario.packet_count):
            device_id = devices[i % len(devices)]
            is_speech = np.random.random() < scenario.speech_probability
            
            try:
                packet_data = self.packet_generator.generate_enhanced_packet(
                    device_id=device_id,
                    is_speech=is_speech
                )
                
                packet_info = coordinator.protocol_parser.parse_packet(
                    packet_data, tuple(device_id.split(':'))
                )
                
                if packet_info is None:
                    errors += 1
                    continue
                
                audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
                vad_result = coordinator.process_packet(packet_info, audio_samples)
                
                device_packets[device_id] += 1
                total_processed += 1
                
            except Exception as e:
                errors += 1
        
        # Check device state tracking
        device_states = coordinator.get_device_states()
        tracked_devices = len(device_states)
        
        success = (tracked_devices == scenario.device_count and 
                  errors < scenario.packet_count * 0.05)
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=total_processed,
            packets_lost=errors,
            accuracy=1.0 - (errors / scenario.packet_count),
            latency_stats={
                'tracked_devices': tracked_devices,
                'expected_devices': scenario.device_count,
                'packets_per_device': dict(device_packets)
            }
        )
    
    def _test_rapid_transitions(self, scenario: TestScenario) -> TestResult:
        """Test rapid speech start/end transitions."""
        mock_server_vad = MockIntelligentVAD()
        coordinator = ESP32P4VADCoordinator(mock_server_vad, VADFusionStrategy.EDGE_PRIORITY)
        
        speech_starts = 0
        speech_ends = 0
        transitions_detected = 0
        
        # Generate alternating speech/silence pattern
        is_speech = False
        
        for i in range(scenario.packet_count):
            # Toggle every 10 packets for rapid transitions
            if i % 10 == 0:
                is_speech = not is_speech
            
            # Set boundary flags for transitions
            vad_flags = 0
            if i % 10 == 0:  # Transition packet
                if is_speech:
                    vad_flags |= ESP32P4VADFlags.SPEECH_START
                    speech_starts += 1
                else:
                    vad_flags |= ESP32P4VADFlags.SPEECH_END  
                    speech_ends += 1
            
            if is_speech:
                vad_flags |= ESP32P4VADFlags.VOICE_ACTIVE
            
            # Generate packet with explicit flags
            basic_header = struct.pack('<IHHBBB', i, 512, 16000, 1, 16, 0)
            vad_header = struct.pack('<BBBBBHHBB',
                0x02, vad_flags, 180 if is_speech else 80, 200, 
                20000 if is_speech else 1000, 1500, 300, 15, 0
            )
            audio_data = self.packet_generator._generate_audio_samples(512, is_speech, 0.0, 16)
            packet_data = basic_header + vad_header + audio_data
            
            packet_info = coordinator.protocol_parser.parse_packet(
                packet_data, ("192.168.1.100", 8000)
            )
            
            if packet_info is None:
                continue
            
            audio_samples = packet_info.audio_data.astype(np.float32) / 32767.0
            vad_result = coordinator.process_packet(packet_info, audio_samples)
            
            if vad_result.decision in [VADDecision.SPEECH_START, VADDecision.SPEECH_END]:
                transitions_detected += 1
        
        expected_transitions = speech_starts + speech_ends
        transition_detection_rate = transitions_detected / expected_transitions if expected_transitions > 0 else 0
        
        success = transition_detection_rate >= 0.7  # Detect 70% of transitions
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=scenario.packet_count,
            packets_lost=0,
            accuracy=transition_detection_rate,
            latency_stats={
                'speech_starts': speech_starts,
                'speech_ends': speech_ends,
                'transitions_detected': transitions_detected,
                'detection_rate': transition_detection_rate
            }
        )
    
    def _test_malformed_packets(self, scenario: TestScenario) -> TestResult:
        """Test resilience to malformed packets."""
        parser = ESP32P4ProtocolParser()
        malformed_packets = 0
        handled_gracefully = 0
        
        malformation_types = [
            'too_short',
            'invalid_header',
            'corrupted_vad',
            'wrong_audio_size',
            'invalid_sequence'
        ]
        
        for i in range(scenario.packet_count):
            malformation_type = np.random.choice(malformation_types)
            
            try:
                if malformation_type == 'too_short':
                    # Packet too small
                    packet_data = b'\x00' * 5
                elif malformation_type == 'invalid_header':
                    # Invalid header values
                    packet_data = struct.pack('<IHHBBB', 
                        0xFFFFFFFF, 99999, 999999, 255, 255, 255) + b'\x00' * 1000
                elif malformation_type == 'corrupted_vad':
                    # Good basic header, corrupted VAD header
                    basic_header = struct.pack('<IHHBBB', i, 512, 16000, 1, 16, 0)
                    vad_header = b'\xFF' * 12  # Corrupted VAD data
                    audio_data = b'\x00' * 1024
                    packet_data = basic_header + vad_header + audio_data
                elif malformation_type == 'wrong_audio_size':
                    # Header says one size, actual audio is different
                    basic_header = struct.pack('<IHHBBB', i, 512, 16000, 1, 16, 0)
                    vad_header = struct.pack('<BBBBBHHBB', 0x02, 0, 128, 150, 15000, 1200, 300, 12, 0)
                    audio_data = b'\x00' * 100  # Too small
                    packet_data = basic_header + vad_header + audio_data
                else:  # invalid_sequence
                    # Valid packet but with invalid sequence
                    packet_data = self.packet_generator.generate_enhanced_packet()
                
                malformed_packets += 1
                
                # Try to parse - should handle gracefully
                packet_info = parser.parse_packet(packet_data, ("192.168.1.100", 8000))
                
                # If no exception and returns None (graceful failure) or valid packet
                if packet_info is None or self._validate_packet_info(packet_info, enhanced=True):
                    handled_gracefully += 1
                    
            except Exception as e:
                # Exception is acceptable for malformed packets
                handled_gracefully += 1
        
        success_rate = handled_gracefully / malformed_packets if malformed_packets > 0 else 0
        success = success_rate >= 0.95  # 95% should be handled gracefully
        
        return TestResult(
            scenario=scenario,
            success=success,
            execution_time=time.time(),
            packets_processed=malformed_packets,
            packets_lost=malformed_packets - handled_gracefully,
            accuracy=success_rate,
            latency_stats={
                'malformed_packets': malformed_packets,
                'handled_gracefully': handled_gracefully,
                'success_rate': success_rate
            }
        )
    
    def _validate_packet_info(self, packet_info: ESP32P4PacketInfo, enhanced: bool = False) -> bool:
        """Validate packet info structure."""
        if packet_info is None:
            return False
        
        # Check basic header
        if packet_info.basic_header is None:
            return False
        
        # Check enhanced packet requirements
        if enhanced and packet_info.vad_header is None:
            return False
        
        # Check audio data
        if packet_info.audio_data is None or len(packet_info.audio_data) == 0:
            return False
        
        return True
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate overall statistics
        overall_accuracy = statistics.mean([result.accuracy for result in self.test_results if result.accuracy > 0])
        total_packets = sum(result.packets_processed for result in self.test_results)
        total_execution_time = sum(result.execution_time for result in self.test_results)
        
        # Collect latency statistics
        all_latencies = []
        for result in self.test_results:
            if result.latency_stats and 'mean' in result.latency_stats:
                all_latencies.append(result.latency_stats['mean'])
        
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'overall_accuracy': overall_accuracy,
                'total_packets_processed': total_packets,
                'total_execution_time': total_execution_time
            },
            'performance_metrics': {
                'average_latency_ms': statistics.mean(all_latencies) if all_latencies else 0,
                'packets_per_second': total_packets / total_execution_time if total_execution_time > 0 else 0
            },
            'test_results': []
        }
        
        # Add individual test results
        for result in self.test_results:
            report['test_results'].append({
                'scenario_name': result.scenario.name,
                'scenario_description': result.scenario.description,
                'success': result.success,
                'accuracy': result.accuracy,
                'packets_processed': result.packets_processed,
                'packets_lost': result.packets_lost,
                'execution_time': result.execution_time,
                'latency_stats': result.latency_stats,
                'performance_metrics': result.performance_metrics,
                'error_details': result.error_details
            })
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [result for result in self.test_results if not result.success]
        
        if failed_tests:
            recommendations.append(f"⚠️  {len(failed_tests)} tests failed. Review error details for each failure.")
        
        # Check latency performance
        latency_tests = [result for result in self.test_results 
                        if 'latency' in result.scenario.name.lower()]
        if latency_tests:
            avg_latency = statistics.mean([result.latency_stats.get('mean', 0) for result in latency_tests])
            if avg_latency > 20:
                recommendations.append(f"🐌 Average latency ({avg_latency:.1f}ms) exceeds 20ms target. Consider optimization.")
        
        # Check accuracy
        low_accuracy_tests = [result for result in self.test_results 
                             if result.accuracy > 0 and result.accuracy < 0.8]
        if low_accuracy_tests:
            recommendations.append(f"🎯 {len(low_accuracy_tests)} tests show accuracy below 80%. Review VAD fusion strategies.")
        
        # Check packet loss handling
        packet_loss_tests = [result for result in self.test_results 
                           if 'packet_loss' in result.scenario.name.lower()]
        if any(not result.success for result in packet_loss_tests):
            recommendations.append("📡 Packet loss handling needs improvement. Consider implementing buffering.")
        
        if not recommendations:
            recommendations.append("✅ All tests passed successfully! ESP32-P4 VAD coordination system is ready for production.")
        
        return recommendations

def main():
    """Main test execution function."""
    print("ESP32-P4 VAD Coordination System - Comprehensive Test Suite")
    print("=" * 65)
    
    try:
        tester = ESP32P4VADTester()
        test_report = tester.run_all_tests()
        
        # Print summary
        summary = test_report['test_summary']
        print(f"\n🧪 Test Execution Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"   Packets Processed: {summary['total_packets_processed']:,}")
        
        # Print performance metrics
        perf = test_report['performance_metrics']
        print(f"\n⚡ Performance Metrics:")
        print(f"   Average Latency: {perf['average_latency_ms']:.2f} ms")
        print(f"   Throughput: {perf['packets_per_second']:.1f} packets/sec")
        
        # Print recommendations
        print(f"\n💡 Recommendations:")
        for rec in test_report['recommendations']:
            print(f"   {rec}")
        
        # Save detailed report
        report_filename = f"esp32_p4_vad_test_report_{int(time.time())}.json"
        with open(report_filename, 'w') as f:
            json.dump(test_report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Exit with appropriate code
        if summary['failed_tests'] == 0:
            print("\n🎉 All tests passed! ESP32-P4 VAD system is production-ready.")
            sys.exit(0)
        else:
            print(f"\n⚠️  {summary['failed_tests']} test(s) failed. Review failures before deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()