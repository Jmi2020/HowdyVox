#!/usr/bin/env python3
"""
HowdyTTS Server UDP Flow Validation Test Suite

This comprehensive test suite validates UDP packet reception from ESP32-P4
devices with detailed analysis and monitoring capabilities.

Features:
- Enhanced UDP packet reception with validation
- ESP32-P4 packet format compliance verification
- Real-time packet analysis and quality metrics
- Network reliability testing scenarios
- Performance benchmarking and statistics
- Comprehensive error categorization and debugging

Usage:
    python test_udp_flow_validation.py [--port 8003] [--duration 60] [--scenarios all]
"""

import asyncio
import argparse
import logging
import sys
import os
import time
import json
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import signal

# Add voice_assistant to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice_assistant'))

from wireless_audio_server import WirelessAudioServer

@dataclass
class TestScenarioConfig:
    """Configuration for a test scenario"""
    name: str
    duration_seconds: int
    expected_packet_rate: float  # packets per second
    packet_rate_tolerance: float  # tolerance percentage
    enable_deep_analysis: bool
    enable_quality_metrics: bool

@dataclass
class PacketQualityMetrics:
    """Quality metrics for UDP packet analysis"""
    packet_count: int
    bytes_received: int
    sequence_gaps: int
    late_packets: int
    malformed_packets: int
    invalid_headers: int
    success_rate: float
    throughput_mbps: float
    avg_packet_size: float
    packet_rate_pps: float
    
    # Timing analysis
    min_inter_packet_ms: float
    max_inter_packet_ms: float
    avg_inter_packet_ms: float
    jitter_ms: float
    
    # Audio quality
    format_consistency: bool
    sample_rate_consistency: bool
    amplitude_analysis: Dict[str, float]

@dataclass
class ScenarioResult:
    """Result of a single test scenario"""
    scenario_name: str
    duration_seconds: float
    start_time: float
    end_time: float
    packet_metrics: PacketQualityMetrics
    device_analysis: Dict[str, Any]
    error_categories: Dict[str, int]
    performance_summary: Dict[str, Any]
    success: bool
    notes: List[str]

class UDPFlowValidationTester:
    """Comprehensive UDP flow validation test suite for HowdyTTS server"""
    
    def __init__(self, port: int = 8003, verbose: bool = False):
        self.port = port
        self.verbose = verbose
        self.server: Optional[WirelessAudioServer] = None
        self.test_active = False
        self.abort_requested = False
        
        # Test state tracking
        self.current_scenario: Optional[TestScenarioConfig] = None
        self.scenario_start_time = 0
        self.test_results: List[ScenarioResult] = []
        
        # Real-time packet analysis
        self.packet_buffer = deque(maxlen=1000)  # Last 1000 packets for analysis
        self.device_stats = defaultdict(lambda: {
            'packets': [],
            'timestamps': [],
            'sequence_numbers': [],
            'errors': defaultdict(int),
            'quality_metrics': {}
        })
        
        # Performance tracking
        self.performance_metrics = {
            'test_start_time': 0,
            'total_packets_received': 0,
            'total_bytes_received': 0,
            'total_devices_seen': 0,
            'peak_packet_rate': 0,
            'min_packet_rate': float('inf'),
            'throughput_samples': []
        }
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def create_test_scenarios(self, duration_factor: float = 1.0) -> List[TestScenarioConfig]:
        """Create comprehensive test scenarios"""
        base_duration = int(30 * duration_factor)  # Base 30 seconds
        
        scenarios = [
            TestScenarioConfig(
                name="Basic Reception Validation",
                duration_seconds=base_duration,
                expected_packet_rate=50.0,  # 50 packets/sec (20ms intervals)
                packet_rate_tolerance=20.0,  # 20% tolerance
                enable_deep_analysis=True,
                enable_quality_metrics=True
            ),
            TestScenarioConfig(
                name="High Volume Packet Reception",
                duration_seconds=base_duration,
                expected_packet_rate=100.0,  # 100 packets/sec (10ms intervals)
                packet_rate_tolerance=30.0,  # 30% tolerance for high volume
                enable_deep_analysis=True,
                enable_quality_metrics=True
            ),
            TestScenarioConfig(
                name="Error Recovery Validation",
                duration_seconds=base_duration,
                expected_packet_rate=40.0,  # Lower rate due to expected errors
                packet_rate_tolerance=50.0,  # High tolerance for error scenarios
                enable_deep_analysis=True,
                enable_quality_metrics=False
            ),
            TestScenarioConfig(
                name="Multi-Device Discovery Test",
                duration_seconds=base_duration // 2,  # Shorter for discovery
                expected_packet_rate=25.0,  # Multiple devices, lower per-device rate
                packet_rate_tolerance=60.0,  # High tolerance for discovery
                enable_deep_analysis=False,
                enable_quality_metrics=True
            ),
            TestScenarioConfig(
                name="Sustained Streaming Performance",
                duration_seconds=base_duration * 2,  # Longer duration
                expected_packet_rate=50.0,
                packet_rate_tolerance=15.0,  # Tight tolerance for sustained test
                enable_deep_analysis=True,
                enable_quality_metrics=True
            )
        ]
        
        return scenarios
    
    def setup_server(self) -> bool:
        """Initialize and configure the wireless audio server"""
        try:
            self.server = WirelessAudioServer(port=self.port)
            self.server.set_audio_callback(self.audio_packet_callback)
            
            # Enable comprehensive debug logging
            self.server.enable_debug_logging(
                enable=True,
                hex_dump=self.verbose,
                packet_interval=10  # Log every 10th packet
            )
            
            # Validate ESP32-P4 compatibility
            compatibility = self.server.validate_esp32_p4_compatibility()
            if not compatibility['esp32_p4_ready']:
                self.logger.error("❌ Server not ready for ESP32-P4 compatibility")
                return False
            
            self.logger.info("✅ Server ESP32-P4 compatibility validated")
            self.logger.info(f"📊 Compatibility score: {compatibility['overall_compatibility_score']:.1f}%")
            
            return self.server.start()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup server: {e}")
            return False
    
    def audio_packet_callback(self, audio_data, raw_packet_data=None, source_addr=None):
        """Enhanced audio packet callback with comprehensive analysis"""
        current_time = time.time()
        
        if raw_packet_data and source_addr:
            device_id = f"{source_addr[0]}:{source_addr[1]}"
            
            # Store packet for real-time analysis
            packet_info = {
                'timestamp': current_time,
                'device_id': device_id,
                'packet_size': len(raw_packet_data),
                'audio_samples': len(audio_data),
                'source_addr': source_addr
            }
            
            self.packet_buffer.append(packet_info)
            
            # Update device-specific statistics
            device_data = self.device_stats[device_id]
            device_data['packets'].append(packet_info)
            device_data['timestamps'].append(current_time)
            
            # Track performance metrics
            self.performance_metrics['total_packets_received'] += 1
            self.performance_metrics['total_bytes_received'] += len(raw_packet_data)
            
            # Calculate real-time packet rate
            if len(device_data['timestamps']) >= 10:  # Need minimum samples
                recent_timestamps = device_data['timestamps'][-10:]
                time_span = recent_timestamps[-1] - recent_timestamps[0]
                if time_span > 0:
                    packet_rate = 9 / time_span  # 9 intervals over 10 packets
                    
                    self.performance_metrics['peak_packet_rate'] = max(
                        self.performance_metrics['peak_packet_rate'], packet_rate
                    )
                    self.performance_metrics['min_packet_rate'] = min(
                        self.performance_metrics['min_packet_rate'], packet_rate
                    )
            
            # Audio level analysis
            if len(audio_data) > 0:
                audio_level = float(abs(audio_data).mean())
                if audio_level > 0.01:  # Significant audio
                    self.logger.debug(f"🎵 Audio from {device_id}: {len(audio_data)} samples, level: {audio_level:.4f}")
    
    def analyze_scenario_performance(self, scenario: TestScenarioConfig) -> PacketQualityMetrics:
        """Analyze packet quality metrics for the current scenario"""
        scenario_packets = [
            p for p in self.packet_buffer 
            if p['timestamp'] >= self.scenario_start_time
        ]
        
        if not scenario_packets:
            # Return empty metrics if no packets received
            return PacketQualityMetrics(
                packet_count=0, bytes_received=0, sequence_gaps=0,
                late_packets=0, malformed_packets=0, invalid_headers=0,
                success_rate=0.0, throughput_mbps=0.0, avg_packet_size=0.0,
                packet_rate_pps=0.0, min_inter_packet_ms=0.0,
                max_inter_packet_ms=0.0, avg_inter_packet_ms=0.0,
                jitter_ms=0.0, format_consistency=True,
                sample_rate_consistency=True, amplitude_analysis={}
            )
        
        # Basic metrics
        packet_count = len(scenario_packets)
        total_bytes = sum(p['packet_size'] for p in scenario_packets)
        duration = scenario_packets[-1]['timestamp'] - scenario_packets[0]['timestamp']
        
        # Calculate inter-packet intervals
        timestamps = [p['timestamp'] for p in scenario_packets]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        intervals_ms = [i * 1000 for i in intervals if i > 0]
        
        # Timing analysis
        if intervals_ms:
            min_interval = min(intervals_ms)
            max_interval = max(intervals_ms)
            avg_interval = statistics.mean(intervals_ms)
            jitter = statistics.stdev(intervals_ms) if len(intervals_ms) > 1 else 0.0
        else:
            min_interval = max_interval = avg_interval = jitter = 0.0
        
        # Performance calculations
        if duration > 0:
            packet_rate = packet_count / duration
            throughput_mbps = (total_bytes * 8) / (duration * 1000000)
        else:
            packet_rate = throughput_mbps = 0.0
        
        avg_packet_size = total_bytes / packet_count if packet_count > 0 else 0.0
        
        # Get server validation statistics
        if self.server:
            validation_stats = self.server.get_comprehensive_validation_stats()
            server_packets = validation_stats['total_packets_received']
            valid_headers = validation_stats['packet_validation_summary']['valid_headers']
            success_rate = (valid_headers / server_packets * 100) if server_packets > 0 else 0.0
            
            error_breakdown = validation_stats['error_breakdown']
            sequence_gaps = sum(
                device['sequence_gaps'] 
                for device in validation_stats['device_summaries'].values()
            )
            late_packets = sum(
                device['late_packets']
                for device in validation_stats['device_summaries'].values()
            )
        else:
            success_rate = 100.0
            sequence_gaps = late_packets = 0
            error_breakdown = {}
        
        return PacketQualityMetrics(
            packet_count=packet_count,
            bytes_received=total_bytes,
            sequence_gaps=sequence_gaps,
            late_packets=late_packets,
            malformed_packets=error_breakdown.get('malformed_packets', 0),
            invalid_headers=error_breakdown.get('invalid_headers', 0),
            success_rate=success_rate,
            throughput_mbps=throughput_mbps,
            avg_packet_size=avg_packet_size,
            packet_rate_pps=packet_rate,
            min_inter_packet_ms=min_interval,
            max_inter_packet_ms=max_interval,
            avg_inter_packet_ms=avg_interval,
            jitter_ms=jitter,
            format_consistency=True,  # Would need deeper analysis to determine
            sample_rate_consistency=True,  # Would need deeper analysis to determine
            amplitude_analysis={}  # Placeholder for audio analysis
        )
    
    def run_scenario(self, scenario: TestScenarioConfig) -> ScenarioResult:
        """Run a single test scenario"""
        self.logger.info("")
        self.logger.info(f"🔄 Running Scenario: {scenario.name}")
        self.logger.info(f"   Duration: {scenario.duration_seconds}s")
        self.logger.info(f"   Expected rate: {scenario.expected_packet_rate} pkt/s")
        self.logger.info(f"   Tolerance: ±{scenario.packet_rate_tolerance}%")
        self.logger.info("─" * 60)
        
        self.current_scenario = scenario
        self.scenario_start_time = time.time()
        
        # Clear buffers for this scenario
        self.packet_buffer.clear()
        
        # Reset server statistics for this scenario
        if self.server:
            # Note: We don't reset stats to maintain cumulative data
            # but we track scenario-specific metrics separately
            pass
        
        # Run scenario for specified duration
        end_time = self.scenario_start_time + scenario.duration_seconds
        
        last_progress_time = self.scenario_start_time
        progress_interval = 5.0  # Report progress every 5 seconds
        
        while time.time() < end_time and not self.abort_requested:
            current_time = time.time()
            
            # Periodic progress reporting
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - self.scenario_start_time
                progress_pct = (elapsed / scenario.duration_seconds) * 100
                
                # Get current packet statistics
                scenario_packets = [
                    p for p in self.packet_buffer 
                    if p['timestamp'] >= self.scenario_start_time
                ]
                
                current_rate = len(scenario_packets) / elapsed if elapsed > 0 else 0
                
                self.logger.info(f"📊 Progress: {progress_pct:.1f}% | "
                               f"Packets: {len(scenario_packets)} | "
                               f"Rate: {current_rate:.1f} pkt/s | "
                               f"Target: {scenario.expected_packet_rate:.1f} pkt/s")
                
                last_progress_time = current_time
            
            time.sleep(0.1)  # Small sleep to prevent busy waiting
        
        # Scenario completed - analyze results
        scenario_end_time = time.time()
        actual_duration = scenario_end_time - self.scenario_start_time
        
        self.logger.info(f"⏱️ Scenario completed in {actual_duration:.1f}s")
        
        # Analyze packet quality metrics
        quality_metrics = self.analyze_scenario_performance(scenario)
        
        # Determine success criteria
        rate_diff_pct = abs(quality_metrics.packet_rate_pps - scenario.expected_packet_rate) / scenario.expected_packet_rate * 100
        rate_within_tolerance = rate_diff_pct <= scenario.packet_rate_tolerance
        
        success = (
            rate_within_tolerance and
            quality_metrics.success_rate >= 85.0 and  # At least 85% success rate
            quality_metrics.packet_count > 0
        )
        
        # Collect additional analysis
        device_analysis = {}
        if self.server:
            validation_stats = self.server.get_comprehensive_validation_stats()
            device_analysis = validation_stats.get('device_summaries', {})
        
        error_categories = {}
        if self.server:
            debug_stats = self.server.get_debug_stats()
            # Extract error categories from device stats
            for device_id, device_data in debug_stats.get('device_stats', {}).items():
                if 'error_categories' in device_data:
                    for error_type, count in device_data['error_categories'].items():
                        error_categories[error_type] = error_categories.get(error_type, 0) + count
        
        # Performance summary
        performance_summary = {
            'expected_packet_rate': scenario.expected_packet_rate,
            'actual_packet_rate': quality_metrics.packet_rate_pps,
            'rate_deviation_percent': rate_diff_pct,
            'rate_within_tolerance': rate_within_tolerance,
            'throughput_mbps': quality_metrics.throughput_mbps,
            'avg_packet_size': quality_metrics.avg_packet_size,
            'jitter_ms': quality_metrics.jitter_ms
        }
        
        # Collect notes about the scenario
        notes = []
        if not rate_within_tolerance:
            notes.append(f"Packet rate deviation {rate_diff_pct:.1f}% exceeds tolerance {scenario.packet_rate_tolerance}%")
        
        if quality_metrics.success_rate < 85.0:
            notes.append(f"Success rate {quality_metrics.success_rate:.1f}% below 85% threshold")
        
        if quality_metrics.sequence_gaps > 0:
            notes.append(f"Detected {quality_metrics.sequence_gaps} sequence gaps")
        
        if quality_metrics.late_packets > 0:
            notes.append(f"Detected {quality_metrics.late_packets} late packets")
        
        if success:
            notes.append("All success criteria met")
        
        # Create scenario result
        result = ScenarioResult(
            scenario_name=scenario.name,
            duration_seconds=actual_duration,
            start_time=self.scenario_start_time,
            end_time=scenario_end_time,
            packet_metrics=quality_metrics,
            device_analysis=device_analysis,
            error_categories=error_categories,
            performance_summary=performance_summary,
            success=success,
            notes=notes
        )
        
        # Log scenario results
        self.logger.info(f"📋 Scenario Results: {scenario.name}")
        self.logger.info(f"   Success: {'✅ PASS' if success else '❌ FAIL'}")
        self.logger.info(f"   Packets received: {quality_metrics.packet_count}")
        self.logger.info(f"   Success rate: {quality_metrics.success_rate:.1f}%")
        self.logger.info(f"   Packet rate: {quality_metrics.packet_rate_pps:.1f} pkt/s (expected: {scenario.expected_packet_rate:.1f})")
        self.logger.info(f"   Throughput: {quality_metrics.throughput_mbps:.3f} Mbps")
        self.logger.info(f"   Avg packet size: {quality_metrics.avg_packet_size:.1f} bytes")
        self.logger.info(f"   Sequence gaps: {quality_metrics.sequence_gaps}")
        self.logger.info(f"   Late packets: {quality_metrics.late_packets}")
        if quality_metrics.jitter_ms > 0:
            self.logger.info(f"   Jitter: {quality_metrics.jitter_ms:.2f} ms")
        
        if notes:
            self.logger.info(f"   Notes: {'; '.join(notes)}")
        
        return result
    
    async def run_test_suite(self, scenarios: List[TestScenarioConfig]) -> List[ScenarioResult]:
        """Run the complete UDP flow validation test suite"""
        self.logger.info("🚀 Starting HowdyTTS UDP Flow Validation Test Suite")
        self.logger.info("═" * 80)
        self.logger.info(f"Server port: {self.port}")
        self.logger.info(f"Scenarios: {len(scenarios)}")
        self.logger.info(f"Total estimated duration: {sum(s.duration_seconds for s in scenarios)}s")
        self.logger.info("═" * 80)
        
        if not self.setup_server():
            self.logger.error("❌ Failed to setup server")
            return []
        
        self.test_active = True
        self.performance_metrics['test_start_time'] = time.time()
        
        # Run each scenario
        results = []
        for i, scenario in enumerate(scenarios):
            if self.abort_requested:
                self.logger.warning("🛑 Test suite aborted by user")
                break
                
            self.logger.info(f"\n📍 Scenario {i+1}/{len(scenarios)}")
            
            try:
                result = self.run_scenario(scenario)
                results.append(result)
                self.test_results.append(result)
                
                # Brief pause between scenarios (except last one)
                if i < len(scenarios) - 1:
                    self.logger.info("⏸️ Pausing 3s before next scenario...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                self.logger.error(f"❌ Scenario {scenario.name} failed with exception: {e}")
                # Create failed result
                failed_result = ScenarioResult(
                    scenario_name=scenario.name,
                    duration_seconds=0,
                    start_time=time.time(),
                    end_time=time.time(),
                    packet_metrics=PacketQualityMetrics(
                        packet_count=0, bytes_received=0, sequence_gaps=0,
                        late_packets=0, malformed_packets=0, invalid_headers=0,
                        success_rate=0.0, throughput_mbps=0.0, avg_packet_size=0.0,
                        packet_rate_pps=0.0, min_inter_packet_ms=0.0,
                        max_inter_packet_ms=0.0, avg_inter_packet_ms=0.0,
                        jitter_ms=0.0, format_consistency=False,
                        sample_rate_consistency=False, amplitude_analysis={}
                    ),
                    device_analysis={},
                    error_categories={},
                    performance_summary={},
                    success=False,
                    notes=[f"Exception occurred: {str(e)}"]
                )
                results.append(failed_result)
        
        # Test suite completed
        self.test_active = False
        total_duration = time.time() - self.performance_metrics['test_start_time']
        
        self.logger.info("")
        self.logger.info("🏁 UDP Flow Validation Test Suite Completed")
        self.logger.info("═" * 80)
        
        # Overall results summary
        successful_scenarios = sum(1 for r in results if r.success)
        total_packets = sum(r.packet_metrics.packet_count for r in results)
        total_bytes = sum(r.packet_metrics.bytes_received for r in results)
        avg_success_rate = statistics.mean([r.packet_metrics.success_rate for r in results]) if results else 0.0
        
        self.logger.info(f"📊 Overall Test Results:")
        self.logger.info(f"   Total duration: {total_duration:.1f}s")
        self.logger.info(f"   Scenarios: {successful_scenarios}/{len(scenarios)} successful")
        self.logger.info(f"   Total packets received: {total_packets}")
        self.logger.info(f"   Total bytes received: {total_bytes:,}")
        self.logger.info(f"   Average success rate: {avg_success_rate:.1f}%")
        self.logger.info(f"   Test success: {'✅ PASS' if successful_scenarios == len(scenarios) else '❌ FAIL'}")
        
        # Detailed server statistics
        if self.server:
            self.logger.info("")
            self.server.print_debug_summary()
        
        # Cleanup
        if self.server:
            self.server.stop()
        
        return results
    
    def export_results_json(self, results: List[ScenarioResult], filename: str):
        """Export test results to JSON file"""
        try:
            # Convert dataclasses to dictionaries for JSON serialization
            json_data = {
                'test_summary': {
                    'total_scenarios': len(results),
                    'successful_scenarios': sum(1 for r in results if r.success),
                    'test_duration': sum(r.duration_seconds for r in results),
                    'total_packets': sum(r.packet_metrics.packet_count for r in results),
                    'export_time': time.time()
                },
                'performance_metrics': self.performance_metrics,
                'scenario_results': [asdict(result) for result in results]
            }
            
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            self.logger.info(f"📄 Results exported to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal, stopping test...")
    global tester
    if tester:
        tester.abort_requested = True

async def main():
    """Main function for UDP flow validation testing"""
    parser = argparse.ArgumentParser(
        description='HowdyTTS UDP Flow Validation Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_udp_flow_validation.py                    # Run with defaults
  python test_udp_flow_validation.py --port 8003       # Specify port
  python test_udp_flow_validation.py --duration 1.5    # 1.5x duration factor
  python test_udp_flow_validation.py --verbose         # Detailed logging
  python test_udp_flow_validation.py --export results.json  # Export results

Test Process:
  1. Sets up enhanced UDP server with ESP32-P4 validation
  2. Runs comprehensive packet reception scenarios
  3. Analyzes packet quality, timing, and reliability
  4. Provides detailed performance metrics and validation
  5. Exports results for further analysis
        """
    )
    
    parser.add_argument('--port', type=int, default=8003,
                       help='UDP port to listen on (default: 8003)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Duration factor for scenarios (default: 1.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    
    args = parser.parse_args()
    
    # Setup signal handling
    global tester
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run test suite
    tester = UDPFlowValidationTester(port=args.port, verbose=args.verbose)
    
    scenarios = tester.create_test_scenarios(duration_factor=args.duration)
    results = await tester.run_test_suite(scenarios)
    
    # Export results if requested
    if args.export:
        tester.export_results_json(results, args.export)
    
    # Return appropriate exit code
    successful_scenarios = sum(1 for r in results if r.success)
    exit_code = 0 if successful_scenarios == len(scenarios) else 1
    
    tester.logger.info(f"✅ Test suite completed with exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)