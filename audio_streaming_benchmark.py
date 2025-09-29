#!/usr/bin/env python3
"""
Audio Streaming Performance Benchmark Tool

This tool provides comprehensive performance benchmarking for sustained
UDP audio streaming between ESP32-P4 devices and HowdyTTS server.

Features:
- Sustained streaming performance analysis
- Throughput and latency measurements
- Resource utilization monitoring (CPU, memory, network)
- Scalability testing with multiple concurrent streams
- Long-duration stress testing
- Quality degradation analysis under load
- Performance regression detection
- Benchmark result comparison and reporting

Usage:
    python audio_streaming_benchmark.py [--duration 300] [--concurrent 5] [--export results.json]
"""

import asyncio
import time
import psutil
import threading
import logging
import sys
import os
import json
import argparse
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import concurrent.futures

# Add voice_assistant to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice_assistant'))

from wireless_audio_server import WirelessAudioServer

@dataclass
class SystemResourceMetrics:
    """System resource utilization metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    network_errors: int
    disk_io_read_mb: float
    disk_io_write_mb: float
    open_files: int
    threads: int

@dataclass
class AudioStreamMetrics:
    """Audio stream performance metrics"""
    stream_id: str
    duration_seconds: float
    
    # Packet statistics
    packets_received: int
    packets_expected: int
    packet_loss_rate: float
    packet_success_rate: float
    
    # Throughput metrics
    bytes_received: int
    bits_per_second: float
    packets_per_second: float
    audio_data_rate_kbps: float
    
    # Latency metrics
    min_latency_ms: float
    max_latency_ms: float
    avg_latency_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    # Jitter and timing
    min_inter_packet_ms: float
    max_inter_packet_ms: float
    avg_inter_packet_ms: float
    jitter_ms: float
    timing_stability_score: float  # 0-100
    
    # Quality metrics
    audio_quality_score: float  # 0-100
    format_consistency: bool
    sequence_integrity_score: float  # 0-100
    
    # Performance grades
    throughput_grade: str  # A, B, C, D, F
    latency_grade: str     # A, B, C, D, F
    stability_grade: str   # A, B, C, D, F
    overall_grade: str     # A, B, C, D, F

@dataclass
class BenchmarkScenario:
    """Benchmark test scenario configuration"""
    name: str
    description: str
    duration_seconds: int
    concurrent_streams: int
    target_packet_rate: float  # packets per second per stream
    stress_cpu: bool = False    # Enable CPU stress testing
    stress_memory: bool = False # Enable memory stress testing
    stress_network: bool = False # Enable network stress testing
    quality_threshold: float = 80.0  # Minimum acceptable quality score

@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    scenario: BenchmarkScenario
    start_time: float
    end_time: float
    actual_duration: float
    
    # Stream results
    stream_metrics: List[AudioStreamMetrics]
    
    # System performance
    resource_metrics: List[SystemResourceMetrics]
    
    # Aggregate metrics
    total_packets_received: int
    total_bytes_received: int
    overall_packet_loss_rate: float
    overall_throughput_mbps: float
    peak_cpu_percent: float
    peak_memory_mb: float
    avg_cpu_percent: float
    avg_memory_mb: float
    
    # Performance scores
    throughput_score: float    # 0-100
    latency_score: float      # 0-100
    stability_score: float    # 0-100
    scalability_score: float # 0-100
    efficiency_score: float  # 0-100
    overall_score: float     # 0-100
    
    # Pass/fail status
    meets_requirements: bool
    performance_grade: str
    
    # Issues and recommendations
    issues_detected: List[str]
    recommendations: List[str]

class AudioStreamingBenchmark:
    """Comprehensive audio streaming performance benchmark tool"""
    
    def __init__(self, port: int = 8003, verbose: bool = False):
        self.port = port
        self.verbose = verbose
        
        # Server and monitoring
        self.server: Optional[WirelessAudioServer] = None
        self.benchmark_active = False
        self.abort_requested = False
        
        # Data collection
        self.stream_data = defaultdict(lambda: {
            'packets': deque(maxlen=100000),
            'timestamps': deque(maxlen=100000),
            'latencies': deque(maxlen=10000),
            'sizes': deque(maxlen=100000),
            'start_time': 0,
            'last_packet_time': 0
        })
        
        # System monitoring
        self.resource_monitor = None
        self.resource_data = deque(maxlen=10000)
        
        # Benchmark configuration
        self.current_scenario: Optional[BenchmarkScenario] = None
        self.scenario_start_time = 0
        
        # Results
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_benchmark_scenarios(self) -> List[BenchmarkScenario]:
        """Create comprehensive benchmark scenarios"""
        scenarios = [
            # Basic performance baseline
            BenchmarkScenario(
                name="Baseline Performance",
                description="Single stream baseline performance measurement",
                duration_seconds=60,
                concurrent_streams=1,
                target_packet_rate=50.0  # 50 packets/second (20ms intervals)
            ),
            
            # Sustained streaming test
            BenchmarkScenario(
                name="Sustained Streaming",
                description="Long-duration single stream stability test",
                duration_seconds=300,  # 5 minutes
                concurrent_streams=1,
                target_packet_rate=50.0
            ),
            
            # Multi-stream scalability
            BenchmarkScenario(
                name="Dual Stream Scalability",
                description="Two concurrent audio streams",
                duration_seconds=120,
                concurrent_streams=2,
                target_packet_rate=50.0
            ),
            
            BenchmarkScenario(
                name="Multi-Stream Scalability",
                description="Five concurrent audio streams",
                duration_seconds=120,
                concurrent_streams=5,
                target_packet_rate=50.0
            ),
            
            BenchmarkScenario(
                name="High Concurrency Stress",
                description="Ten concurrent audio streams",
                duration_seconds=90,
                concurrent_streams=10,
                target_packet_rate=50.0,
                quality_threshold=70.0  # Lower threshold for stress test
            ),
            
            # High throughput tests
            BenchmarkScenario(
                name="High Packet Rate",
                description="Single stream at 100 packets/second",
                duration_seconds=60,
                concurrent_streams=1,
                target_packet_rate=100.0
            ),
            
            BenchmarkScenario(
                name="Maximum Throughput",
                description="Single stream at maximum sustainable rate",
                duration_seconds=60,
                concurrent_streams=1,
                target_packet_rate=200.0,
                quality_threshold=70.0
            ),
            
            # System stress tests
            BenchmarkScenario(
                name="CPU Stress Test",
                description="Audio streaming under CPU load",
                duration_seconds=90,
                concurrent_streams=3,
                target_packet_rate=50.0,
                stress_cpu=True,
                quality_threshold=75.0
            ),
            
            BenchmarkScenario(
                name="Memory Stress Test",
                description="Audio streaming under memory pressure",
                duration_seconds=90,
                concurrent_streams=3,
                target_packet_rate=50.0,
                stress_memory=True,
                quality_threshold=75.0
            ),
            
            # Endurance test
            BenchmarkScenario(
                name="Endurance Test",
                description="Extended duration stability test",
                duration_seconds=600,  # 10 minutes
                concurrent_streams=2,
                target_packet_rate=50.0,
                quality_threshold=85.0
            )
        ]
        
        return scenarios
    
    def setup_server_for_benchmark(self) -> bool:
        """Setup server with benchmark-specific configuration"""
        try:
            self.server = WirelessAudioServer(port=self.port)
            self.server.set_audio_callback(self.benchmark_audio_callback)
            
            # Enable detailed logging for benchmarking
            self.server.enable_debug_logging(
                enable=False,  # Disable verbose logging during benchmarks
                hex_dump=False,
                packet_interval=1000  # Minimal logging to reduce overhead
            )
            
            return self.server.start()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup server: {e}")
            return False
    
    def benchmark_audio_callback(self, audio_data, raw_packet_data=None, source_addr=None):
        """Enhanced audio callback for benchmark data collection"""
        current_time = time.time()
        
        if raw_packet_data and source_addr:
            device_id = f"{source_addr[0]}:{source_addr[1]}"
            
            # Parse packet for timing information
            packet_info = {
                'timestamp': current_time,
                'size': len(raw_packet_data),
                'audio_samples': len(audio_data),
                'source_addr': source_addr
            }
            
            # Store packet data
            stream = self.stream_data[device_id]
            stream['packets'].append(packet_info)
            stream['timestamps'].append(current_time)
            stream['sizes'].append(len(raw_packet_data))
            stream['last_packet_time'] = current_time
            
            if stream['start_time'] == 0:
                stream['start_time'] = current_time
            
            # Calculate latency (inter-packet timing)
            if len(stream['timestamps']) >= 2:
                inter_packet_ms = (current_time - stream['timestamps'][-2]) * 1000
                stream['latencies'].append(inter_packet_ms)
    
    def start_resource_monitoring(self):
        """Start system resource monitoring"""
        def monitor_resources():
            self.logger.info("📊 System resource monitoring started")
            
            # Get initial network stats
            net_io_start = psutil.net_io_counters()
            disk_io_start = psutil.disk_io_counters()
            
            while self.benchmark_active:
                try:
                    # CPU and memory
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Network I/O
                    net_io = psutil.net_io_counters()
                    
                    # Disk I/O
                    disk_io = psutil.disk_io_counters()
                    
                    # Process info
                    process = psutil.Process()
                    
                    metrics = SystemResourceMetrics(
                        timestamp=time.time(),
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        memory_mb=memory.used / 1024 / 1024,
                        network_bytes_sent=net_io.bytes_sent - net_io_start.bytes_sent,
                        network_bytes_recv=net_io.bytes_recv - net_io_start.bytes_recv,
                        network_packets_sent=net_io.packets_sent - net_io_start.packets_sent,
                        network_packets_recv=net_io.packets_recv - net_io_start.packets_recv,
                        network_errors=net_io.errin + net_io.errout,
                        disk_io_read_mb=(disk_io.read_bytes - disk_io_start.read_bytes) / 1024 / 1024,
                        disk_io_write_mb=(disk_io.write_bytes - disk_io_start.write_bytes) / 1024 / 1024,
                        open_files=process.num_fds() if hasattr(process, 'num_fds') else 0,
                        threads=process.num_threads()
                    )
                    
                    self.resource_data.append(metrics)
                    
                except Exception as e:
                    self.logger.debug(f"Resource monitoring error: {e}")
                    time.sleep(1)
        
        self.resource_monitor = threading.Thread(target=monitor_resources, daemon=True)
        self.resource_monitor.start()
    
    def stop_resource_monitoring(self):
        """Stop system resource monitoring"""
        if self.resource_monitor:
            self.logger.info("📊 System resource monitoring stopped")
    
    def create_stress_conditions(self, scenario: BenchmarkScenario):
        """Create system stress conditions for testing"""
        stress_tasks = []
        
        if scenario.stress_cpu:
            self.logger.info("💻 Creating CPU stress load...")
            
            def cpu_stress():
                # CPU-intensive computation
                end_time = time.time() + scenario.duration_seconds
                while time.time() < end_time and self.benchmark_active:
                    # Busy computation
                    for _ in range(10000):
                        x = sum(i*i for i in range(100))
                    time.sleep(0.001)  # Brief pause
            
            for _ in range(psutil.cpu_count() // 2):  # Use half the CPU cores
                task = threading.Thread(target=cpu_stress, daemon=True)
                task.start()
                stress_tasks.append(task)
        
        if scenario.stress_memory:
            self.logger.info("💾 Creating memory pressure...")
            
            def memory_stress():
                # Allocate and hold memory
                memory_blocks = []
                block_size = 10 * 1024 * 1024  # 10MB blocks
                max_blocks = 50  # 500MB total
                
                try:
                    for i in range(max_blocks):
                        if not self.benchmark_active:
                            break
                        block = bytearray(block_size)
                        memory_blocks.append(block)
                        time.sleep(0.1)
                    
                    # Hold memory for scenario duration
                    time.sleep(scenario.duration_seconds)
                    
                finally:
                    # Clean up
                    memory_blocks.clear()
            
            task = threading.Thread(target=memory_stress, daemon=True)
            task.start()
            stress_tasks.append(task)
        
        return stress_tasks
    
    async def run_benchmark_scenario(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Run a single benchmark scenario"""
        self.logger.info("")
        self.logger.info(f"🚀 Running Benchmark: {scenario.name}")
        self.logger.info(f"   Description: {scenario.description}")
        self.logger.info(f"   Duration: {scenario.duration_seconds}s")
        self.logger.info(f"   Concurrent streams: {scenario.concurrent_streams}")
        self.logger.info(f"   Target rate: {scenario.target_packet_rate} pkt/s")
        self.logger.info("─" * 60)
        
        self.current_scenario = scenario
        self.scenario_start_time = time.time()
        
        # Reset data collection
        self.stream_data.clear()
        self.resource_data.clear()
        
        # Setup server
        if not self.setup_server_for_benchmark():
            return self._create_failed_result(scenario, "Server setup failed")
        
        # Start resource monitoring
        self.benchmark_active = True
        self.start_resource_monitoring()
        
        # Create stress conditions if needed
        stress_tasks = self.create_stress_conditions(scenario)
        
        # Run benchmark
        end_time = self.scenario_start_time + scenario.duration_seconds
        last_progress_time = self.scenario_start_time
        progress_interval = max(10, scenario.duration_seconds // 10)  # Progress every 10% or 10s
        
        while time.time() < end_time and not self.abort_requested:
            current_time = time.time()
            
            # Progress reporting
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - self.scenario_start_time
                progress_pct = (elapsed / scenario.duration_seconds) * 100
                
                # Count active streams
                active_streams = len([s for s in self.stream_data.values() 
                                    if current_time - s.get('last_packet_time', 0) < 5])
                
                total_packets = sum(len(s['packets']) for s in self.stream_data.values())
                
                self.logger.info(f"📊 Progress: {progress_pct:.1f}% | "
                               f"Active streams: {active_streams}/{scenario.concurrent_streams} | "
                               f"Total packets: {total_packets}")
                
                last_progress_time = current_time
            
            await asyncio.sleep(0.1)
        
        # Benchmark completed
        self.benchmark_active = False
        actual_duration = time.time() - self.scenario_start_time
        
        self.logger.info(f"⏱️ Benchmark completed in {actual_duration:.1f}s")
        
        # Stop monitoring and cleanup
        self.stop_resource_monitoring()
        if self.server:
            self.server.stop()
        
        # Analyze results
        result = self._analyze_benchmark_results(scenario, actual_duration)
        
        self.logger.info(f"📋 Benchmark Results: {scenario.name}")
        self.logger.info(f"   Overall Score: {result.overall_score:.1f}/100")
        self.logger.info(f"   Performance Grade: {result.performance_grade}")
        self.logger.info(f"   Meets Requirements: {'✅ YES' if result.meets_requirements else '❌ NO'}")
        
        return result
    
    def _analyze_benchmark_results(self, scenario: BenchmarkScenario, 
                                 actual_duration: float) -> BenchmarkResult:
        """Analyze benchmark results and calculate performance metrics"""
        
        # Analyze each stream
        stream_metrics = []
        for device_id, stream_data in self.stream_data.items():
            if len(stream_data['packets']) > 0:
                metrics = self._analyze_stream_performance(device_id, stream_data, scenario)
                stream_metrics.append(metrics)
        
        # Calculate aggregate metrics
        total_packets = sum(len(s['packets']) for s in self.stream_data.values())
        total_bytes = sum(sum(p['size'] for p in s['packets']) for s in self.stream_data.values())
        
        expected_packets = scenario.concurrent_streams * scenario.target_packet_rate * actual_duration
        packet_loss_rate = max(0, (expected_packets - total_packets) / expected_packets * 100) if expected_packets > 0 else 0
        
        throughput_mbps = (total_bytes * 8) / (actual_duration * 1000000) if actual_duration > 0 else 0
        
        # Resource utilization analysis
        if self.resource_data:
            peak_cpu = max(r.cpu_percent for r in self.resource_data)
            peak_memory = max(r.memory_mb for r in self.resource_data)
            avg_cpu = statistics.mean(r.cpu_percent for r in self.resource_data)
            avg_memory = statistics.mean(r.memory_mb for r in self.resource_data)
        else:
            peak_cpu = avg_cpu = peak_memory = avg_memory = 0
        
        # Calculate performance scores
        scores = self._calculate_performance_scores(scenario, stream_metrics, packet_loss_rate, 
                                                  peak_cpu, avg_cpu)
        
        # Determine pass/fail and grade
        meets_requirements = (
            scores['overall_score'] >= scenario.quality_threshold and
            packet_loss_rate < 5.0 and  # Less than 5% packet loss
            all(m.audio_quality_score >= scenario.quality_threshold for m in stream_metrics)
        )
        
        performance_grade = self._score_to_grade(scores['overall_score'])
        
        # Detect issues and generate recommendations
        issues, recommendations = self._analyze_issues_and_recommendations(
            scenario, stream_metrics, packet_loss_rate, peak_cpu, peak_memory
        )
        
        return BenchmarkResult(
            scenario=scenario,
            start_time=self.scenario_start_time,
            end_time=self.scenario_start_time + actual_duration,
            actual_duration=actual_duration,
            stream_metrics=stream_metrics,
            resource_metrics=list(self.resource_data),
            total_packets_received=total_packets,
            total_bytes_received=total_bytes,
            overall_packet_loss_rate=packet_loss_rate,
            overall_throughput_mbps=throughput_mbps,
            peak_cpu_percent=peak_cpu,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            avg_memory_mb=avg_memory,
            throughput_score=scores['throughput_score'],
            latency_score=scores['latency_score'],
            stability_score=scores['stability_score'],
            scalability_score=scores['scalability_score'],
            efficiency_score=scores['efficiency_score'],
            overall_score=scores['overall_score'],
            meets_requirements=meets_requirements,
            performance_grade=performance_grade,
            issues_detected=issues,
            recommendations=recommendations
        )
    
    def _analyze_stream_performance(self, device_id: str, stream_data: Dict[str, Any], 
                                  scenario: BenchmarkScenario) -> AudioStreamMetrics:
        """Analyze performance metrics for a single stream"""
        packets = stream_data['packets']
        timestamps = list(stream_data['timestamps'])
        latencies = list(stream_data['latencies'])
        sizes = list(stream_data['sizes'])
        
        if not packets:
            return self._create_empty_stream_metrics(device_id)
        
        # Basic metrics
        duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
        packets_received = len(packets)
        bytes_received = sum(p['size'] for p in packets)
        
        # Expected vs actual
        expected_packets = scenario.target_packet_rate * duration if duration > 0 else 0
        packet_loss_rate = max(0, (expected_packets - packets_received) / expected_packets * 100) if expected_packets > 0 else 0
        packet_success_rate = 100 - packet_loss_rate
        
        # Throughput
        bits_per_second = (bytes_received * 8) / duration if duration > 0 else 0
        packets_per_second = packets_received / duration if duration > 0 else 0
        audio_data_rate = bits_per_second / 1000  # kbps
        
        # Latency analysis
        if latencies:
            min_latency = min(latencies)
            max_latency = max(latencies)
            avg_latency = statistics.mean(latencies)
            
            # Percentiles
            sorted_latencies = sorted(latencies)
            p95_idx = int(0.95 * len(sorted_latencies))
            p99_idx = int(0.99 * len(sorted_latencies))
            latency_p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else max_latency
            latency_p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else max_latency
        else:
            min_latency = max_latency = avg_latency = latency_p95 = latency_p99 = 0
        
        # Inter-packet timing
        if len(timestamps) >= 2:
            intervals = [(timestamps[i+1] - timestamps[i]) * 1000 
                        for i in range(len(timestamps)-1)]
            
            min_interval = min(intervals)
            max_interval = max(intervals)
            avg_interval = statistics.mean(intervals)
            
            # Jitter calculation
            jitter = statistics.stdev(intervals) if len(intervals) > 1 else 0
            
            # Stability score based on coefficient of variation
            cv = jitter / avg_interval if avg_interval > 0 else float('inf')
            stability_score = max(0, 100 - (cv * 100))
        else:
            min_interval = max_interval = avg_interval = jitter = 0
            stability_score = 0
        
        # Quality metrics
        audio_quality_score = self._calculate_stream_quality_score(
            packet_success_rate, avg_latency, jitter
        )
        
        format_consistency = True  # Would need packet format analysis
        sequence_integrity_score = packet_success_rate  # Simplified
        
        # Performance grades
        throughput_grade = self._score_to_grade(min(100, packets_per_second / scenario.target_packet_rate * 100))
        latency_grade = self._score_to_grade(max(0, 100 - avg_latency))
        stability_grade = self._score_to_grade(stability_score)
        overall_grade = self._score_to_grade(audio_quality_score)
        
        return AudioStreamMetrics(
            stream_id=device_id,
            duration_seconds=duration,
            packets_received=packets_received,
            packets_expected=int(expected_packets),
            packet_loss_rate=packet_loss_rate,
            packet_success_rate=packet_success_rate,
            bytes_received=bytes_received,
            bits_per_second=bits_per_second,
            packets_per_second=packets_per_second,
            audio_data_rate_kbps=audio_data_rate,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            avg_latency_ms=avg_latency,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            min_inter_packet_ms=min_interval,
            max_inter_packet_ms=max_interval,
            avg_inter_packet_ms=avg_interval,
            jitter_ms=jitter,
            timing_stability_score=stability_score,
            audio_quality_score=audio_quality_score,
            format_consistency=format_consistency,
            sequence_integrity_score=sequence_integrity_score,
            throughput_grade=throughput_grade,
            latency_grade=latency_grade,
            stability_grade=stability_grade,
            overall_grade=overall_grade
        )
    
    def _calculate_stream_quality_score(self, packet_success_rate: float, 
                                      avg_latency: float, jitter: float) -> float:
        """Calculate overall stream quality score"""
        # Weighted scoring
        weights = {
            'packet_success': 0.4,
            'latency': 0.3,
            'jitter': 0.3
        }
        
        # Convert metrics to scores (0-100)
        packet_score = packet_success_rate
        latency_score = max(0, 100 - avg_latency)  # Lower latency = higher score
        jitter_score = max(0, 100 - (jitter * 2))  # Lower jitter = higher score
        
        quality_score = (
            packet_score * weights['packet_success'] +
            latency_score * weights['latency'] +
            jitter_score * weights['jitter']
        )
        
        return min(100, max(0, quality_score))
    
    def _calculate_performance_scores(self, scenario: BenchmarkScenario, 
                                    stream_metrics: List[AudioStreamMetrics],
                                    packet_loss_rate: float, peak_cpu: float, 
                                    avg_cpu: float) -> Dict[str, float]:
        """Calculate comprehensive performance scores"""
        
        if not stream_metrics:
            return {
                'throughput_score': 0, 'latency_score': 0, 'stability_score': 0,
                'scalability_score': 0, 'efficiency_score': 0, 'overall_score': 0
            }
        
        # Throughput score
        avg_throughput_ratio = statistics.mean([
            m.packets_per_second / scenario.target_packet_rate 
            for m in stream_metrics
        ])
        throughput_score = min(100, avg_throughput_ratio * 100)
        
        # Latency score
        avg_latency = statistics.mean([m.avg_latency_ms for m in stream_metrics])
        latency_score = max(0, 100 - avg_latency)
        
        # Stability score
        stability_score = statistics.mean([m.timing_stability_score for m in stream_metrics])
        
        # Scalability score (based on performance consistency across streams)
        if len(stream_metrics) > 1:
            quality_scores = [m.audio_quality_score for m in stream_metrics]
            quality_std = statistics.stdev(quality_scores)
            scalability_score = max(0, 100 - (quality_std * 2))
        else:
            scalability_score = 100  # Single stream = perfect scalability
        
        # Efficiency score (based on resource utilization)
        cpu_efficiency = max(0, 100 - peak_cpu) if peak_cpu > 0 else 100
        packet_efficiency = max(0, 100 - packet_loss_rate)
        efficiency_score = (cpu_efficiency + packet_efficiency) / 2
        
        # Overall score (weighted average)
        weights = {
            'throughput': 0.25,
            'latency': 0.2,
            'stability': 0.2,
            'scalability': 0.15,
            'efficiency': 0.2
        }
        
        overall_score = (
            throughput_score * weights['throughput'] +
            latency_score * weights['latency'] +
            stability_score * weights['stability'] +
            scalability_score * weights['scalability'] +
            efficiency_score * weights['efficiency']
        )
        
        return {
            'throughput_score': throughput_score,
            'latency_score': latency_score,
            'stability_score': stability_score,
            'scalability_score': scalability_score,
            'efficiency_score': efficiency_score,
            'overall_score': overall_score
        }
    
    def _analyze_issues_and_recommendations(self, scenario: BenchmarkScenario,
                                          stream_metrics: List[AudioStreamMetrics],
                                          packet_loss_rate: float, peak_cpu: float,
                                          peak_memory: float) -> Tuple[List[str], List[str]]:
        """Analyze performance issues and generate recommendations"""
        issues = []
        recommendations = []
        
        # Packet loss issues
        if packet_loss_rate > 5:
            issues.append(f"High packet loss rate: {packet_loss_rate:.1f}%")
            recommendations.append("Investigate network reliability and UDP buffer sizes")
        
        # CPU utilization issues
        if peak_cpu > 80:
            issues.append(f"High CPU utilization: {peak_cpu:.1f}%")
            recommendations.append("Consider CPU optimization or load balancing")
        
        # Latency issues
        if stream_metrics:
            avg_latency = statistics.mean([m.avg_latency_ms for m in stream_metrics])
            if avg_latency > 50:
                issues.append(f"High average latency: {avg_latency:.1f}ms")
                recommendations.append("Optimize network path and processing pipeline")
        
        # Jitter issues
        if stream_metrics:
            avg_jitter = statistics.mean([m.jitter_ms for m in stream_metrics])
            if avg_jitter > 10:
                issues.append(f"High network jitter: {avg_jitter:.1f}ms")
                recommendations.append("Implement jitter buffer or QoS policies")
        
        # Scalability issues
        if len(stream_metrics) > 1:
            quality_scores = [m.audio_quality_score for m in stream_metrics]
            if min(quality_scores) < max(quality_scores) - 20:
                issues.append("Inconsistent performance across streams")
                recommendations.append("Review load balancing and resource allocation")
        
        # Throughput issues
        if stream_metrics:
            underperforming_streams = [
                m for m in stream_metrics 
                if m.packets_per_second < scenario.target_packet_rate * 0.9
            ]
            if underperforming_streams:
                issues.append(f"{len(underperforming_streams)} streams underperforming")
                recommendations.append("Check network capacity and device configuration")
        
        return issues, recommendations
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _create_empty_stream_metrics(self, device_id: str) -> AudioStreamMetrics:
        """Create empty stream metrics"""
        return AudioStreamMetrics(
            stream_id=device_id, duration_seconds=0, packets_received=0,
            packets_expected=0, packet_loss_rate=0, packet_success_rate=0,
            bytes_received=0, bits_per_second=0, packets_per_second=0,
            audio_data_rate_kbps=0, min_latency_ms=0, max_latency_ms=0,
            avg_latency_ms=0, latency_p95_ms=0, latency_p99_ms=0,
            min_inter_packet_ms=0, max_inter_packet_ms=0, avg_inter_packet_ms=0,
            jitter_ms=0, timing_stability_score=0, audio_quality_score=0,
            format_consistency=True, sequence_integrity_score=0,
            throughput_grade="F", latency_grade="F", stability_grade="F",
            overall_grade="F"
        )
    
    def _create_failed_result(self, scenario: BenchmarkScenario, reason: str) -> BenchmarkResult:
        """Create result for failed benchmark"""
        return BenchmarkResult(
            scenario=scenario, start_time=time.time(), end_time=time.time(),
            actual_duration=0, stream_metrics=[], resource_metrics=[],
            total_packets_received=0, total_bytes_received=0,
            overall_packet_loss_rate=100, overall_throughput_mbps=0,
            peak_cpu_percent=0, peak_memory_mb=0, avg_cpu_percent=0,
            avg_memory_mb=0, throughput_score=0, latency_score=0,
            stability_score=0, scalability_score=0, efficiency_score=0,
            overall_score=0, meets_requirements=False, performance_grade="F",
            issues_detected=[reason], recommendations=["Fix setup issues and retry"]
        )
    
    async def run_benchmark_suite(self, scenarios: List[BenchmarkScenario]) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        self.logger.info("🚀 Starting Audio Streaming Benchmark Suite")
        self.logger.info("═" * 80)
        self.logger.info(f"Benchmark scenarios: {len(scenarios)}")
        self.logger.info(f"Estimated duration: {sum(s.duration_seconds for s in scenarios)}s")
        self.logger.info("═" * 80)
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            if self.abort_requested:
                self.logger.warning("🛑 Benchmark suite aborted")
                break
            
            self.logger.info(f"\\n📍 Benchmark {i+1}/{len(scenarios)}")
            
            try:
                result = await self.run_benchmark_scenario(scenario)
                results.append(result)
                self.benchmark_results.append(result)
                
                # Brief pause between benchmarks
                if i < len(scenarios) - 1:
                    self.logger.info("⏸️ Pausing 10s before next benchmark...")
                    await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"❌ Benchmark {scenario.name} failed: {e}")
                failed_result = self._create_failed_result(scenario, str(e))
                results.append(failed_result)
        
        # Final summary
        self._log_benchmark_suite_summary(results)
        
        return results
    
    def _log_benchmark_suite_summary(self, results: List[BenchmarkResult]):
        """Log comprehensive benchmark suite summary"""
        self.logger.info("")
        self.logger.info("🏁 Audio Streaming Benchmark Suite Summary")
        self.logger.info("═" * 80)
        
        passed_benchmarks = len([r for r in results if r.meets_requirements])
        total_packets = sum(r.total_packets_received for r in results)
        avg_score = statistics.mean([r.overall_score for r in results]) if results else 0
        
        self.logger.info(f"Benchmarks: {passed_benchmarks}/{len(results)} passed")
        self.logger.info(f"Total packets processed: {total_packets:,}")
        self.logger.info(f"Average performance score: {avg_score:.1f}/100")
        
        # Performance breakdown
        if results:
            avg_throughput = statistics.mean([r.throughput_score for r in results])
            avg_latency = statistics.mean([r.latency_score for r in results])
            avg_stability = statistics.mean([r.stability_score for r in results])
            
            self.logger.info(f"Average throughput score: {avg_throughput:.1f}/100")
            self.logger.info(f"Average latency score: {avg_latency:.1f}/100")
            self.logger.info(f"Average stability score: {avg_stability:.1f}/100")
        
        # Grade distribution
        grades = [r.performance_grade for r in results]
        grade_counts = {grade: grades.count(grade) for grade in set(grades)}
        self.logger.info(f"Grade distribution: {grade_counts}")
        
        overall_success = passed_benchmarks == len(results) and avg_score >= 80
        self.logger.info(f"Overall result: {'✅ PASS' if overall_success else '❌ FAIL'}")
        self.logger.info("═" * 80)
    
    def export_results(self, results: List[BenchmarkResult], filename: str):
        """Export benchmark results to JSON file"""
        try:
            export_data = {
                'benchmark_summary': {
                    'total_scenarios': len(results),
                    'passed_scenarios': len([r for r in results if r.meets_requirements]),
                    'average_score': statistics.mean([r.overall_score for r in results]) if results else 0,
                    'export_time': time.time()
                },
                'scenario_results': [asdict(result) for result in results]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📄 Benchmark results exported to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\\n🛑 Received interrupt signal, stopping benchmark...")
    global benchmark
    if benchmark:
        benchmark.abort_requested = True

async def main():
    """Main function for audio streaming benchmark"""
    parser = argparse.ArgumentParser(
        description='Audio Streaming Performance Benchmark Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_streaming_benchmark.py                    # Run all benchmarks
  python audio_streaming_benchmark.py --quick           # Quick benchmark suite
  python audio_streaming_benchmark.py --stress          # Stress test scenarios
  python audio_streaming_benchmark.py --duration 0.5    # Reduce test duration
  python audio_streaming_benchmark.py --export results.json  # Export results

Benchmark Categories:
  • Baseline performance (single stream)
  • Sustained streaming stability
  • Multi-stream scalability
  • High throughput testing
  • System stress testing
  • Long-duration endurance testing
        """
    )
    
    parser.add_argument('--port', type=int, default=8003,
                       help='UDP port to test (default: 8003)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark suite (reduced duration)')
    parser.add_argument('--stress', action='store_true',
                       help='Run stress test scenarios only')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Duration factor for scenarios (default: 1.0)')
    parser.add_argument('--export', type=str,
                       help='Export results to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # Setup signal handling
    global benchmark
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create benchmark tool
    benchmark = AudioStreamingBenchmark(port=args.port, verbose=args.verbose)
    
    # Create scenarios based on selection
    all_scenarios = benchmark.create_benchmark_scenarios()
    
    if args.quick:
        # Quick suite - shorter durations, fewer scenarios
        scenarios = [s for s in all_scenarios if 'Stress' not in s.name and 'Endurance' not in s.name]
        for scenario in scenarios:
            scenario.duration_seconds = min(30, scenario.duration_seconds)
    elif args.stress:
        # Stress scenarios only
        scenarios = [s for s in all_scenarios if 'Stress' in s.name or 'High' in s.name]
    else:
        # Full suite
        scenarios = all_scenarios
    
    # Apply duration factor
    for scenario in scenarios:
        scenario.duration_seconds = int(scenario.duration_seconds * args.duration)
    
    # Run benchmark suite
    results = await benchmark.run_benchmark_suite(scenarios)
    
    # Export results if requested
    if args.export:
        benchmark.export_results(results, args.export)
    
    # Return appropriate exit code
    passed_benchmarks = len([r for r in results if r.meets_requirements])
    exit_code = 0 if passed_benchmarks == len(scenarios) else 1
    
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)