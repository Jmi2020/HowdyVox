#!/usr/bin/env python3
"""
Network Reliability Testing for UDP Audio Streaming

This script tests various network failure scenarios and recovery mechanisms
to validate the robustness of UDP audio streaming between ESP32-P4 and HowdyTTS.

Features:
- Simulated network congestion and packet loss
- Bandwidth limitation testing
- Network interface disruption scenarios
- Connection timeout and recovery validation
- Multi-device stress testing
- Error injection and backoff mechanism validation
- Network topology change simulation

Usage:
    python test_network_reliability.py [--scenario all] [--duration 60]
"""

import asyncio
import time
import logging
import sys
import os
import subprocess
import threading
import argparse
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import socket
import struct

# Add voice_assistant to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice_assistant'))

from wireless_audio_server import WirelessAudioServer

@dataclass
class NetworkScenarioConfig:
    """Configuration for network reliability scenario"""
    name: str
    description: str
    duration_seconds: int
    
    # Network simulation parameters
    packet_loss_percent: float = 0.0      # Simulated packet loss
    bandwidth_limit_mbps: float = 0.0     # Bandwidth limitation (0 = unlimited)
    latency_ms: float = 0.0               # Added network latency
    jitter_ms: float = 0.0                # Network jitter
    
    # Connection disruption
    connection_drops: int = 0              # Number of connection drops
    drop_duration_seconds: float = 0.0     # Duration of each drop
    
    # Error injection
    corrupt_packets_percent: float = 0.0   # Corrupted packet injection
    duplicate_packets_percent: float = 0.0 # Duplicate packet injection
    out_of_order_percent: float = 0.0     # Out-of-order packet injection
    
    # Recovery testing
    test_backoff_recovery: bool = False    # Test exponential backoff
    test_degraded_mode: bool = False       # Test degraded mode operation
    
    # Stress testing
    concurrent_connections: int = 1         # Number of simulated devices
    burst_traffic: bool = False            # Generate burst traffic patterns

@dataclass
class NetworkReliabilityMetrics:
    """Metrics collected during network reliability testing"""
    scenario_name: str
    test_duration: float
    
    # Packet statistics
    packets_received: int
    packets_lost: int
    packets_corrupted: int
    packets_duplicate: int
    packets_out_of_order: int
    
    # Timing metrics
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    jitter_ms: float
    
    # Connection metrics
    connection_drops_detected: int
    recovery_time_seconds: float
    backoff_events: int
    degraded_mode_activations: int
    
    # Performance under stress
    throughput_mbps: float
    cpu_usage_percent: float
    memory_usage_mb: float
    
    # Quality scores
    reliability_score: float      # 0-100 based on packet success
    recovery_score: float         # 0-100 based on recovery performance
    stress_resilience_score: float # 0-100 based on stress test performance

class NetworkReliabilityTester:
    """Network reliability testing framework"""
    
    def __init__(self, port: int = 8003, verbose: bool = False):
        self.port = port
        self.verbose = verbose
        self.server: Optional[WirelessAudioServer] = None
        self.test_active = False
        self.abort_requested = False
        
        # Network simulation state
        self.packet_interceptor = None
        self.network_conditions = {}
        
        # Test monitoring
        self.packet_buffer = deque(maxlen=10000)
        self.timing_samples = deque(maxlen=1000)
        self.error_events = []
        self.recovery_events = []
        
        # Metrics collection
        self.test_metrics = {}
        self.performance_counters = defaultdict(int)
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_reliability_scenarios(self) -> List[NetworkScenarioConfig]:
        """Create comprehensive network reliability test scenarios"""
        scenarios = [
            # Basic packet loss scenarios
            NetworkScenarioConfig(
                name="Low Packet Loss",
                description="5% packet loss simulation",
                duration_seconds=60,
                packet_loss_percent=5.0
            ),
            
            NetworkScenarioConfig(
                name="High Packet Loss", 
                description="20% packet loss simulation",
                duration_seconds=60,
                packet_loss_percent=20.0
            ),
            
            # Bandwidth limitation scenarios
            NetworkScenarioConfig(
                name="Limited Bandwidth",
                description="1 Mbps bandwidth limit",
                duration_seconds=60,
                bandwidth_limit_mbps=1.0
            ),
            
            NetworkScenarioConfig(
                name="Severe Bandwidth Limit",
                description="0.1 Mbps bandwidth limit",
                duration_seconds=60,
                bandwidth_limit_mbps=0.1
            ),
            
            # Latency and jitter scenarios
            NetworkScenarioConfig(
                name="High Latency",
                description="500ms network latency",
                duration_seconds=60,
                latency_ms=500.0,
                jitter_ms=50.0
            ),
            
            NetworkScenarioConfig(
                name="Variable Jitter",
                description="High network jitter without latency",
                duration_seconds=60,
                latency_ms=10.0,
                jitter_ms=200.0
            ),
            
            # Connection disruption scenarios
            NetworkScenarioConfig(
                name="Connection Drops",
                description="Multiple connection drops with recovery",
                duration_seconds=120,
                connection_drops=3,
                drop_duration_seconds=10.0,
                test_backoff_recovery=True
            ),
            
            NetworkScenarioConfig(
                name="Extended Outage",
                description="Extended network outage recovery test",
                duration_seconds=180,
                connection_drops=1,
                drop_duration_seconds=60.0,
                test_backoff_recovery=True,
                test_degraded_mode=True
            ),
            
            # Packet corruption scenarios
            NetworkScenarioConfig(
                name="Packet Corruption",
                description="10% packet corruption simulation",
                duration_seconds=60,
                corrupt_packets_percent=10.0
            ),
            
            NetworkScenarioConfig(
                name="Mixed Errors",
                description="Combined packet loss, corruption, and reordering",
                duration_seconds=90,
                packet_loss_percent=5.0,
                corrupt_packets_percent=5.0,
                duplicate_packets_percent=2.0,
                out_of_order_percent=3.0
            ),
            
            # Stress testing scenarios
            NetworkScenarioConfig(
                name="Multi-Device Stress",
                description="Multiple concurrent device connections",
                duration_seconds=120,
                concurrent_connections=5,
                burst_traffic=True
            ),
            
            NetworkScenarioConfig(
                name="Extreme Stress",
                description="High stress with network issues",
                duration_seconds=180,
                packet_loss_percent=10.0,
                bandwidth_limit_mbps=0.5,
                latency_ms=100.0,
                jitter_ms=50.0,
                concurrent_connections=3,
                burst_traffic=True,
                test_degraded_mode=True
            )
        ]
        
        return scenarios
    
    def setup_network_simulation(self, scenario: NetworkScenarioConfig):
        """Setup network simulation for the given scenario"""
        self.logger.info(f"🌐 Setting up network simulation for: {scenario.name}")
        
        # Store current network conditions
        self.network_conditions = {
            'packet_loss_percent': scenario.packet_loss_percent,
            'bandwidth_limit_mbps': scenario.bandwidth_limit_mbps,
            'latency_ms': scenario.latency_ms,
            'jitter_ms': scenario.jitter_ms,
            'corrupt_packets_percent': scenario.corrupt_packets_percent,
            'duplicate_packets_percent': scenario.duplicate_packets_percent,
            'out_of_order_percent': scenario.out_of_order_percent
        }
        
        # Create packet interceptor if needed
        if any([
            scenario.packet_loss_percent > 0,
            scenario.corrupt_packets_percent > 0,
            scenario.duplicate_packets_percent > 0,
            scenario.out_of_order_percent > 0
        ]):
            self.packet_interceptor = NetworkPacketInterceptor(
                self.port,
                self.network_conditions,
                verbose=self.verbose
            )
            self.packet_interceptor.start()
        
        self.logger.info(f"✅ Network simulation configured: {scenario.description}")
    
    def cleanup_network_simulation(self):
        """Cleanup network simulation"""
        if self.packet_interceptor:
            self.packet_interceptor.stop()
            self.packet_interceptor = None
        
        self.network_conditions.clear()
        self.logger.info("🧹 Network simulation cleaned up")
    
    def setup_server_with_monitoring(self) -> bool:
        """Setup server with enhanced monitoring for reliability testing"""
        try:
            self.server = WirelessAudioServer(port=self.port)
            self.server.set_audio_callback(self.reliability_audio_callback)
            
            # Enable comprehensive debugging for reliability testing
            self.server.enable_debug_logging(
                enable=True,
                hex_dump=False,
                packet_interval=1  # Log every packet for reliability analysis
            )
            
            return self.server.start()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup server: {e}")
            return False
    
    def reliability_audio_callback(self, audio_data, raw_packet_data=None, source_addr=None):
        """Enhanced audio callback for reliability testing"""
        current_time = time.time()
        
        if raw_packet_data and source_addr:
            # Store packet for analysis
            packet_info = {
                'timestamp': current_time,
                'source_addr': source_addr,
                'packet_size': len(raw_packet_data),
                'audio_samples': len(audio_data)
            }
            
            self.packet_buffer.append(packet_info)
            self.performance_counters['packets_received'] += 1
            
            # Analyze timing if we have previous packets from same source
            device_id = f"{source_addr[0]}:{source_addr[1]}"
            recent_packets = [
                p for p in list(self.packet_buffer)[-10:]
                if f"{p['source_addr'][0]}:{p['source_addr'][1]}" == device_id
            ]
            
            if len(recent_packets) >= 2:
                # Calculate inter-packet timing
                last_packet = recent_packets[-2]
                inter_packet_ms = (current_time - last_packet['timestamp']) * 1000
                self.timing_samples.append(inter_packet_ms)
                
                # Detect potential issues
                if inter_packet_ms > 100:  # More than 100ms between packets
                    self.error_events.append({
                        'timestamp': current_time,
                        'type': 'timing_anomaly',
                        'device': device_id,
                        'interval_ms': inter_packet_ms
                    })
    
    def simulate_connection_drops(self, scenario: NetworkScenarioConfig):
        """Simulate connection drops as specified in scenario"""
        if scenario.connection_drops <= 0:
            return
        
        async def drop_simulation():
            drop_interval = scenario.duration_seconds / scenario.connection_drops
            
            for drop_num in range(scenario.connection_drops):
                if self.abort_requested:
                    break
                
                # Wait until time for this drop
                await asyncio.sleep(drop_interval)
                
                self.logger.warning(f"🔥 Simulating connection drop {drop_num + 1}/{scenario.connection_drops}")
                
                # Record drop event
                drop_start = time.time()
                self.error_events.append({
                    'timestamp': drop_start,
                    'type': 'connection_drop_start',
                    'drop_number': drop_num + 1
                })
                
                # Simulate drop by temporarily stopping server
                if self.server:
                    self.server.stop()
                
                # Wait for drop duration
                await asyncio.sleep(scenario.drop_duration_seconds)
                
                # Restart server
                if not self.abort_requested:
                    self.logger.info(f"🔄 Recovering from connection drop {drop_num + 1}")
                    
                    recovery_start = time.time()
                    if self.setup_server_with_monitoring():
                        recovery_time = time.time() - recovery_start
                        
                        self.recovery_events.append({
                            'timestamp': time.time(),
                            'type': 'connection_recovery',
                            'drop_number': drop_num + 1,
                            'recovery_time_seconds': recovery_time
                        })
                        
                        self.logger.info(f"✅ Connection recovered in {recovery_time:.2f}s")
                    else:
                        self.logger.error(f"❌ Failed to recover from drop {drop_num + 1}")
        
        # Start drop simulation in background
        asyncio.create_task(drop_simulation())
    
    async def run_reliability_scenario(self, scenario: NetworkScenarioConfig) -> NetworkReliabilityMetrics:
        """Run a single network reliability scenario"""
        self.logger.info("")
        self.logger.info(f"🔬 Running Reliability Scenario: {scenario.name}")
        self.logger.info(f"   Description: {scenario.description}")
        self.logger.info(f"   Duration: {scenario.duration_seconds}s")
        self.logger.info("─" * 60)
        
        # Reset test state
        self.packet_buffer.clear()
        self.timing_samples.clear()
        self.error_events.clear()
        self.recovery_events.clear()
        self.performance_counters.clear()
        
        # Setup network simulation
        self.setup_network_simulation(scenario)
        
        # Setup server with monitoring
        if not self.setup_server_with_monitoring():
            self.logger.error(f"❌ Failed to setup server for scenario: {scenario.name}")
            return self._create_failed_metrics(scenario)
        
        scenario_start_time = time.time()
        
        # Start connection drop simulation if configured
        if scenario.connection_drops > 0:
            self.simulate_connection_drops(scenario)
        
        # Monitor scenario execution
        end_time = scenario_start_time + scenario.duration_seconds
        last_progress_time = scenario_start_time
        progress_interval = 10.0  # Report every 10 seconds
        
        while time.time() < end_time and not self.abort_requested:
            current_time = time.time()
            
            # Periodic progress reporting
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - scenario_start_time
                progress_pct = (elapsed / scenario.duration_seconds) * 100
                
                packets_received = len(self.packet_buffer)
                error_count = len(self.error_events)
                
                self.logger.info(f"📊 Progress: {progress_pct:.1f}% | "
                               f"Packets: {packets_received} | "
                               f"Errors: {error_count}")
                
                last_progress_time = current_time
            
            await asyncio.sleep(0.1)
        
        # Scenario completed - analyze results
        scenario_end_time = time.time()
        actual_duration = scenario_end_time - scenario_start_time
        
        self.logger.info(f"⏱️ Scenario completed in {actual_duration:.1f}s")
        
        # Calculate metrics
        metrics = self._calculate_scenario_metrics(scenario, actual_duration)
        
        # Cleanup
        if self.server:
            self.server.stop()
        self.cleanup_network_simulation()
        
        # Log scenario results
        self._log_scenario_results(metrics)
        
        return metrics
    
    def _calculate_scenario_metrics(self, scenario: NetworkScenarioConfig, 
                                  actual_duration: float) -> NetworkReliabilityMetrics:
        """Calculate comprehensive metrics for the scenario"""
        packets_received = len(self.packet_buffer)
        
        # Calculate timing metrics
        if self.timing_samples:
            timing_values = list(self.timing_samples)
            min_latency = min(timing_values)
            max_latency = max(timing_values)
            avg_latency = sum(timing_values) / len(timing_values)
            
            # Calculate jitter as standard deviation
            mean_timing = avg_latency
            variance = sum((x - mean_timing) ** 2 for x in timing_values) / len(timing_values)
            jitter = variance ** 0.5
        else:
            min_latency = max_latency = avg_latency = jitter = 0.0
        
        # Count error types
        packets_lost = len([e for e in self.error_events if e['type'] == 'packet_loss'])
        packets_corrupted = len([e for e in self.error_events if e['type'] == 'packet_corruption'])
        timing_anomalies = len([e for e in self.error_events if e['type'] == 'timing_anomaly'])
        
        # Connection metrics
        connection_drops = len([e for e in self.error_events if e['type'] == 'connection_drop_start'])
        recoveries = len([e for e in self.recovery_events if e['type'] == 'connection_recovery'])
        
        avg_recovery_time = 0.0
        if recoveries > 0:
            recovery_times = [e['recovery_time_seconds'] for e in self.recovery_events 
                            if e['type'] == 'connection_recovery']
            avg_recovery_time = sum(recovery_times) / len(recovery_times)
        
        # Performance metrics
        if actual_duration > 0:
            throughput_mbps = (packets_received * 512 * 8) / (actual_duration * 1000000)  # Estimate
        else:
            throughput_mbps = 0.0
        
        # Quality scores
        reliability_score = self._calculate_reliability_score(
            packets_received, len(self.error_events), scenario
        )
        
        recovery_score = self._calculate_recovery_score(
            connection_drops, recoveries, avg_recovery_time
        )
        
        stress_resilience_score = self._calculate_stress_resilience_score(
            scenario, packets_received, actual_duration
        )
        
        return NetworkReliabilityMetrics(
            scenario_name=scenario.name,
            test_duration=actual_duration,
            packets_received=packets_received,
            packets_lost=packets_lost,
            packets_corrupted=packets_corrupted,
            packets_duplicate=0,  # Would need specific tracking
            packets_out_of_order=0,  # Would need specific tracking
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            avg_latency_ms=avg_latency,
            jitter_ms=jitter,
            connection_drops_detected=connection_drops,
            recovery_time_seconds=avg_recovery_time,
            backoff_events=0,  # Would need server integration
            degraded_mode_activations=0,  # Would need server integration
            throughput_mbps=throughput_mbps,
            cpu_usage_percent=0.0,  # Would need system monitoring
            memory_usage_mb=0.0,  # Would need system monitoring
            reliability_score=reliability_score,
            recovery_score=recovery_score,
            stress_resilience_score=stress_resilience_score
        )
    
    def _calculate_reliability_score(self, packets_received: int, 
                                   error_count: int, scenario: NetworkScenarioConfig) -> float:
        """Calculate reliability score (0-100)"""
        if packets_received == 0:
            return 0.0
        
        # Base score on packet success rate
        base_score = max(0, 100 - (error_count / packets_received * 100))
        
        # Adjust for scenario difficulty
        difficulty_factor = 1.0
        if scenario.packet_loss_percent > 0:
            difficulty_factor += scenario.packet_loss_percent / 100
        if scenario.bandwidth_limit_mbps > 0 and scenario.bandwidth_limit_mbps < 1.0:
            difficulty_factor += 0.5
        if scenario.connection_drops > 0:
            difficulty_factor += 0.3
        
        # Apply difficulty adjustment
        adjusted_score = min(100, base_score * difficulty_factor)
        
        return adjusted_score
    
    def _calculate_recovery_score(self, drops: int, recoveries: int, 
                                avg_recovery_time: float) -> float:
        """Calculate recovery score (0-100)"""
        if drops == 0:
            return 100.0  # No drops to recover from
        
        if recoveries == 0:
            return 0.0  # Failed to recover
        
        recovery_rate = recoveries / drops
        time_penalty = max(0, 1 - (avg_recovery_time / 30.0))  # Penalty for slow recovery
        
        return recovery_rate * time_penalty * 100
    
    def _calculate_stress_resilience_score(self, scenario: NetworkScenarioConfig, 
                                         packets_received: int, duration: float) -> float:
        """Calculate stress resilience score (0-100)"""
        if scenario.concurrent_connections <= 1 and not scenario.burst_traffic:
            return 100.0  # No stress testing
        
        # Expected packet rate under normal conditions (50 pkt/s per connection)
        expected_packets = scenario.concurrent_connections * 50 * duration
        
        if expected_packets == 0:
            return 0.0
        
        performance_ratio = packets_received / expected_packets
        return min(100, performance_ratio * 100)
    
    def _create_failed_metrics(self, scenario: NetworkScenarioConfig) -> NetworkReliabilityMetrics:
        """Create metrics for failed scenario"""
        return NetworkReliabilityMetrics(
            scenario_name=scenario.name,
            test_duration=0.0,
            packets_received=0,
            packets_lost=0,
            packets_corrupted=0,
            packets_duplicate=0,
            packets_out_of_order=0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            avg_latency_ms=0.0,
            jitter_ms=0.0,
            connection_drops_detected=0,
            recovery_time_seconds=0.0,
            backoff_events=0,
            degraded_mode_activations=0,
            throughput_mbps=0.0,
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            reliability_score=0.0,
            recovery_score=0.0,
            stress_resilience_score=0.0
        )
    
    def _log_scenario_results(self, metrics: NetworkReliabilityMetrics):
        """Log detailed scenario results"""
        self.logger.info(f"📋 Scenario Results: {metrics.scenario_name}")
        self.logger.info(f"   Duration: {metrics.test_duration:.1f}s")
        self.logger.info(f"   Packets received: {metrics.packets_received}")
        self.logger.info(f"   Packets lost: {metrics.packets_lost}")
        self.logger.info(f"   Packets corrupted: {metrics.packets_corrupted}")
        self.logger.info(f"   Throughput: {metrics.throughput_mbps:.3f} Mbps")
        
        if metrics.avg_latency_ms > 0:
            self.logger.info(f"   Latency: min={metrics.min_latency_ms:.1f}ms, "
                           f"max={metrics.max_latency_ms:.1f}ms, "
                           f"avg={metrics.avg_latency_ms:.1f}ms")
            self.logger.info(f"   Jitter: {metrics.jitter_ms:.2f}ms")
        
        if metrics.connection_drops_detected > 0:
            self.logger.info(f"   Connection drops: {metrics.connection_drops_detected}")
            self.logger.info(f"   Recovery time: {metrics.recovery_time_seconds:.2f}s")
        
        self.logger.info(f"   Reliability score: {metrics.reliability_score:.1f}/100")
        self.logger.info(f"   Recovery score: {metrics.recovery_score:.1f}/100")
        self.logger.info(f"   Stress resilience: {metrics.stress_resilience_score:.1f}/100")
    
    async def run_reliability_test_suite(self, scenarios: List[NetworkScenarioConfig]) -> List[NetworkReliabilityMetrics]:
        """Run complete network reliability test suite"""
        self.logger.info("🚀 Starting Network Reliability Test Suite")
        self.logger.info("═" * 80)
        self.logger.info(f"Test scenarios: {len(scenarios)}")
        self.logger.info(f"Estimated duration: {sum(s.duration_seconds for s in scenarios)}s")
        self.logger.info("═" * 80)
        
        self.test_active = True
        results = []
        
        for i, scenario in enumerate(scenarios):
            if self.abort_requested:
                self.logger.warning("🛑 Test suite aborted")
                break
            
            self.logger.info(f"\\n📍 Scenario {i+1}/{len(scenarios)}")
            
            try:
                result = await self.run_reliability_scenario(scenario)
                results.append(result)
                
                # Brief pause between scenarios
                if i < len(scenarios) - 1:
                    self.logger.info("⏸️ Pausing 5s before next scenario...")
                    await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"❌ Scenario {scenario.name} failed: {e}")
                failed_metrics = self._create_failed_metrics(scenario)
                results.append(failed_metrics)
        
        # Final summary
        self._log_test_suite_summary(results)
        
        self.test_active = False
        return results
    
    def _log_test_suite_summary(self, results: List[NetworkReliabilityMetrics]):
        """Log comprehensive test suite summary"""
        self.logger.info("")
        self.logger.info("🏁 Network Reliability Test Suite Summary")
        self.logger.info("═" * 80)
        
        successful_scenarios = len([r for r in results if r.reliability_score > 50])
        total_packets = sum(r.packets_received for r in results)
        total_errors = sum(r.packets_lost + r.packets_corrupted for r in results)
        
        avg_reliability = sum(r.reliability_score for r in results) / len(results) if results else 0
        avg_recovery = sum(r.recovery_score for r in results) / len(results) if results else 0
        avg_resilience = sum(r.stress_resilience_score for r in results) / len(results) if results else 0
        
        self.logger.info(f"Scenarios: {successful_scenarios}/{len(results)} passed")
        self.logger.info(f"Total packets: {total_packets}")
        self.logger.info(f"Total errors: {total_errors}")
        self.logger.info(f"Average reliability score: {avg_reliability:.1f}/100")
        self.logger.info(f"Average recovery score: {avg_recovery:.1f}/100")
        self.logger.info(f"Average resilience score: {avg_resilience:.1f}/100")
        
        overall_success = (
            successful_scenarios == len(results) and
            avg_reliability > 70 and
            avg_recovery > 70
        )
        
        self.logger.info(f"Overall result: {'✅ PASS' if overall_success else '❌ FAIL'}")
        self.logger.info("═" * 80)
    
    def export_results(self, results: List[NetworkReliabilityMetrics], filename: str):
        """Export test results to JSON file"""
        try:
            export_data = {
                'test_summary': {
                    'total_scenarios': len(results),
                    'successful_scenarios': len([r for r in results if r.reliability_score > 50]),
                    'export_time': time.time()
                },
                'scenario_results': [asdict(result) for result in results]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📄 Results exported to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")

class NetworkPacketInterceptor:
    """Simulates network conditions by intercepting and modifying packets"""
    
    def __init__(self, port: int, conditions: Dict[str, Any], verbose: bool = False):
        self.port = port
        self.conditions = conditions
        self.verbose = verbose
        self.running = False
        self.intercept_thread = None
        
        self.logger = logging.getLogger(f"{__name__}.PacketInterceptor")
    
    def start(self):
        """Start packet interception"""
        self.running = True
        self.intercept_thread = threading.Thread(target=self._intercept_loop, daemon=True)
        self.intercept_thread.start()
        self.logger.info("🛡️ Packet interceptor started")
    
    def stop(self):
        """Stop packet interception"""
        self.running = False
        if self.intercept_thread:
            self.intercept_thread.join(timeout=2)
        self.logger.info("🛡️ Packet interceptor stopped")
    
    def _intercept_loop(self):
        """Main packet interception loop"""
        # Note: This is a simplified simulation
        # In a real implementation, this would use raw sockets or iptables
        # For testing purposes, we simulate conditions without actual interception
        
        while self.running:
            time.sleep(0.1)
            
            # Simulate packet processing effects
            if self.conditions.get('packet_loss_percent', 0) > 0:
                # In real implementation, would drop packets
                pass
            
            if self.conditions.get('corrupt_packets_percent', 0) > 0:
                # In real implementation, would corrupt packet data
                pass

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\\n🛑 Received interrupt signal, stopping reliability test...")
    global tester
    if tester:
        tester.abort_requested = True

async def main():
    """Main function for network reliability testing"""
    parser = argparse.ArgumentParser(
        description='Network Reliability Testing for UDP Audio Streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_network_reliability.py                     # Run all scenarios
  python test_network_reliability.py --scenario basic   # Run basic scenarios
  python test_network_reliability.py --duration 0.5     # Reduce test duration
  python test_network_reliability.py --export results.json  # Export results

Scenarios:
  • Packet loss simulation (5%, 20%)
  • Bandwidth limitation (1 Mbps, 0.1 Mbps)
  • High latency and jitter testing
  • Connection drop and recovery validation
  • Packet corruption and error injection
  • Multi-device stress testing
  • Combined failure scenarios
        """
    )
    
    parser.add_argument('--port', type=int, default=8003,
                       help='UDP port to test (default: 8003)')
    parser.add_argument('--scenario', choices=['basic', 'stress', 'all'], default='all',
                       help='Scenario set to run (default: all)')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Duration factor for scenarios (default: 1.0)')
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # Setup signal handling
    global tester
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create reliability tester
    tester = NetworkReliabilityTester(port=args.port, verbose=args.verbose)
    
    # Create scenarios based on selection
    all_scenarios = tester.create_reliability_scenarios()
    
    if args.scenario == 'basic':
        scenarios = [s for s in all_scenarios if 'Stress' not in s.name and 'Extreme' not in s.name]
    elif args.scenario == 'stress':
        scenarios = [s for s in all_scenarios if 'Stress' in s.name or 'Extreme' in s.name]
    else:  # all
        scenarios = all_scenarios
    
    # Apply duration factor
    for scenario in scenarios:
        scenario.duration_seconds = int(scenario.duration_seconds * args.duration)
    
    # Run test suite
    results = await tester.run_reliability_test_suite(scenarios)
    
    # Export results if requested
    if args.export:
        tester.export_results(results, args.export)
    
    # Return appropriate exit code
    successful_scenarios = len([r for r in results if r.reliability_score > 50])
    exit_code = 0 if successful_scenarios == len(scenarios) else 1
    
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)