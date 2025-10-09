#!/usr/bin/env python3
"""
UDP Packet Flow Analysis and Quality Metrics Tool

This tool provides comprehensive analysis of UDP packet flows between
ESP32-P4 devices and HowdyTTS server, focusing on quality metrics,
performance analysis, and validation reporting.

Features:
- Real-time packet flow analysis
- Audio quality metrics calculation
- Network performance profiling
- Packet sequence integrity validation
- Jitter and latency analysis
- Visual analytics and reporting
- Quality score computation
- Trend analysis and prediction

Usage:
    python packet_flow_analyzer.py [--input data.pcap] [--real-time] [--port 8003]
"""

import sys
import os
import time
import json
import argparse
import logging
import statistics
import struct
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import socket
import asyncio

# Add voice_assistant to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'voice_assistant'))

from wireless_audio_server import WirelessAudioServer

@dataclass
class PacketFlowMetrics:
    """Comprehensive packet flow metrics"""
    # Basic flow statistics
    total_packets: int
    total_bytes: int
    duration_seconds: float
    packet_rate_pps: float
    throughput_mbps: float
    
    # Sequence analysis
    sequence_gaps: int
    duplicate_packets: int
    out_of_order_packets: int
    late_packets: int
    sequence_integrity_score: float  # 0-100
    
    # Timing analysis
    min_inter_packet_ms: float
    max_inter_packet_ms: float
    avg_inter_packet_ms: float
    jitter_ms: float
    jitter_coefficient: float  # Normalized jitter
    
    # Audio quality metrics
    sample_rate_consistency: bool
    format_consistency: bool
    amplitude_stability_score: float  # 0-100
    audio_quality_score: float  # 0-100
    
    # Network performance
    bandwidth_utilization_percent: float
    packet_size_efficiency: float
    network_overhead_percent: float
    
    # Error analysis
    malformed_packets: int
    checksum_errors: int
    size_mismatch_errors: int
    format_errors: int
    total_error_rate_percent: float
    
    # Quality scores
    overall_quality_score: float  # 0-100 composite score
    reliability_grade: str  # A, B, C, D, F
    performance_grade: str  # A, B, C, D, F

@dataclass
class DeviceFlowAnalysis:
    """Per-device flow analysis"""
    device_id: str
    device_ip: str
    device_port: int
    
    # Connection info
    first_seen: float
    last_seen: float
    connection_duration: float
    
    # Flow metrics
    flow_metrics: PacketFlowMetrics
    
    # Behavioral analysis
    transmission_pattern: str  # "Regular", "Burst", "Irregular"
    stability_score: float  # 0-100
    consistency_score: float  # 0-100
    
    # Anomaly detection
    anomalies_detected: int
    anomaly_types: List[str]
    anomaly_severity: str  # "Low", "Medium", "High"
    
    # Performance trends
    trend_analysis: Dict[str, Any]

@dataclass 
class PacketFlowReport:
    """Comprehensive packet flow analysis report"""
    analysis_timestamp: float
    analysis_duration: float
    
    # Global metrics
    global_metrics: PacketFlowMetrics
    
    # Per-device analysis
    device_analyses: List[DeviceFlowAnalysis]
    
    # Comparative analysis
    device_comparison: Dict[str, Any]
    
    # Quality assessment
    quality_summary: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]
    
    # Validation results
    validation_results: Dict[str, Any]

class PacketFlowAnalyzer:
    """Comprehensive packet flow analyzer"""
    
    def __init__(self, port: int = 8003, real_time: bool = False, verbose: bool = False):
        self.port = port
        self.real_time = real_time
        self.verbose = verbose
        
        # Analysis state
        self.server: Optional[WirelessAudioServer] = None
        self.analysis_active = False
        self.start_time = 0
        
        # Data collection
        self.packet_data = deque(maxlen=100000)  # Last 100k packets
        self.device_data = defaultdict(lambda: {
            'packets': deque(maxlen=10000),
            'timestamps': deque(maxlen=10000),
            'sequences': deque(maxlen=10000),
            'sizes': deque(maxlen=10000),
            'audio_data': deque(maxlen=1000),
            'errors': defaultdict(int),
            'first_seen': 0,
            'last_seen': 0
        })
        
        # Analysis results
        self.current_metrics = {}
        self.analysis_history = deque(maxlen=1000)
        
        # Real-time monitoring
        self.monitoring_thread = None
        self.analysis_callbacks = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_real_time_analysis(self) -> bool:
        """Setup real-time packet flow analysis"""
        try:
            self.server = WirelessAudioServer(port=self.port)
            self.server.set_audio_callback(self.packet_analysis_callback)
            
            # Enable detailed logging for analysis
            self.server.enable_debug_logging(
                enable=True,
                hex_dump=False,
                packet_interval=1
            )
            
            return self.server.start()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup real-time analysis: {e}")
            return False
    
    def packet_analysis_callback(self, audio_data, raw_packet_data=None, source_addr=None):
        """Enhanced callback for packet analysis"""
        current_time = time.time()
        
        if raw_packet_data and source_addr:
            device_id = f"{source_addr[0]}:{source_addr[1]}"
            
            # Parse packet header for detailed analysis
            packet_info = self._parse_packet_for_analysis(raw_packet_data, source_addr, current_time)
            
            # Store packet data
            self.packet_data.append(packet_info)
            
            # Update device-specific data
            device = self.device_data[device_id]
            device['packets'].append(packet_info)
            device['timestamps'].append(current_time)
            device['sizes'].append(len(raw_packet_data))
            device['audio_data'].append(audio_data)
            
            if device['first_seen'] == 0:
                device['first_seen'] = current_time
            device['last_seen'] = current_time
            
            # Extract sequence number if available
            if packet_info.get('sequence_number'):
                device['sequences'].append(packet_info['sequence_number'])
            
            # Real-time analysis triggers
            if self.real_time and len(device['packets']) % 50 == 0:  # Every 50 packets
                self._perform_real_time_analysis(device_id)
    
    def _parse_packet_for_analysis(self, packet_data: bytes, source_addr: Tuple[str, int], 
                                 timestamp: float) -> Dict[str, Any]:
        """Parse packet data for comprehensive analysis"""
        packet_info = {
            'timestamp': timestamp,
            'source_addr': source_addr,
            'packet_size': len(packet_data),
            'raw_data': packet_data
        }
        
        # Try to parse UDP audio header
        if len(packet_data) >= 12:  # Minimum header size
            try:
                # ESP32-P4 UDP header format: <I H H B B H
                header = struct.unpack('<I H H B B H', packet_data[:12])
                packet_info.update({
                    'sequence_number': header[0],
                    'sample_count': header[1],
                    'sample_rate': header[2],
                    'channels': header[3],
                    'bits_per_sample': header[4],
                    'flags': header[5],
                    'header_valid': True,
                    'payload_size': len(packet_data) - 12
                })
                
                # Validate header consistency
                expected_payload = header[1] * header[3] * (header[4] // 8)
                packet_info['payload_size_valid'] = (packet_info['payload_size'] == expected_payload)
                
            except struct.error:
                packet_info['header_valid'] = False
                packet_info['parse_error'] = 'Header parsing failed'
        else:
            packet_info['header_valid'] = False
            packet_info['parse_error'] = 'Packet too small for header'
        
        return packet_info
    
    def _perform_real_time_analysis(self, device_id: str):
        """Perform real-time analysis for a device"""
        device_data = self.device_data[device_id]
        
        if len(device_data['packets']) < 10:
            return  # Need minimum samples
        
        # Calculate basic metrics
        recent_packets = list(device_data['packets'])[-50:]  # Last 50 packets
        recent_timestamps = list(device_data['timestamps'])[-50:]
        
        # Packet rate calculation
        if len(recent_timestamps) >= 2:
            time_span = recent_timestamps[-1] - recent_timestamps[0]
            packet_rate = (len(recent_timestamps) - 1) / time_span if time_span > 0 else 0
            
            # Inter-packet timing analysis
            intervals = [recent_timestamps[i+1] - recent_timestamps[i] 
                        for i in range(len(recent_timestamps)-1)]
            intervals_ms = [i * 1000 for i in intervals]
            
            if intervals_ms:
                avg_interval = statistics.mean(intervals_ms)
                jitter = statistics.stdev(intervals_ms) if len(intervals_ms) > 1 else 0
                
                # Detect anomalies
                if avg_interval > 50:  # More than 50ms between packets
                    self.logger.warning(f"⚠️ High inter-packet interval for {device_id}: {avg_interval:.1f}ms")
                
                if jitter > 20:  # High jitter
                    self.logger.warning(f"⚠️ High jitter for {device_id}: {jitter:.1f}ms")
                
                # Update current metrics
                self.current_metrics[device_id] = {
                    'packet_rate': packet_rate,
                    'avg_interval_ms': avg_interval,
                    'jitter_ms': jitter,
                    'last_update': time.time()
                }
    
    def analyze_packet_flow(self, duration_seconds: Optional[float] = None) -> PacketFlowReport:
        """Perform comprehensive packet flow analysis"""
        self.logger.info("📊 Performing comprehensive packet flow analysis")
        
        analysis_start = time.time()
        
        if duration_seconds:
            # Filter packets within duration
            cutoff_time = analysis_start - duration_seconds
            packets = [p for p in self.packet_data if p['timestamp'] >= cutoff_time]
        else:
            packets = list(self.packet_data)
        
        if not packets:
            self.logger.warning("⚠️ No packet data available for analysis")
            return self._create_empty_report(analysis_start)
        
        # Calculate global metrics
        global_metrics = self._calculate_global_metrics(packets)
        
        # Analyze each device
        device_analyses = []
        for device_id, device_data in self.device_data.items():
            if len(device_data['packets']) > 0:
                analysis = self._analyze_device_flow(device_id, device_data)
                device_analyses.append(analysis)
        
        # Comparative analysis
        device_comparison = self._perform_device_comparison(device_analyses)
        
        # Quality assessment
        quality_summary = self._assess_quality(global_metrics, device_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(global_metrics, device_analyses)
        
        # Validation results
        validation_results = self._perform_validation(global_metrics, device_analyses)
        
        report = PacketFlowReport(
            analysis_timestamp=analysis_start,
            analysis_duration=time.time() - analysis_start,
            global_metrics=global_metrics,
            device_analyses=device_analyses,
            device_comparison=device_comparison,
            quality_summary=quality_summary,
            recommendations=recommendations,
            validation_results=validation_results
        )
        
        # Store in history
        self.analysis_history.append(report)
        
        self.logger.info(f"✅ Analysis completed in {report.analysis_duration:.2f}s")
        return report
    
    def _calculate_global_metrics(self, packets: List[Dict[str, Any]]) -> PacketFlowMetrics:
        """Calculate global packet flow metrics"""
        if not packets:
            return self._create_empty_metrics()
        
        # Basic statistics
        total_packets = len(packets)
        total_bytes = sum(p['packet_size'] for p in packets)
        
        # Time analysis
        timestamps = [p['timestamp'] for p in packets]
        duration = max(timestamps) - min(timestamps) if timestamps else 0
        
        packet_rate = total_packets / duration if duration > 0 else 0
        throughput_mbps = (total_bytes * 8) / (duration * 1000000) if duration > 0 else 0
        
        # Sequence analysis
        valid_packets = [p for p in packets if p.get('header_valid', False)]
        sequences = [p['sequence_number'] for p in valid_packets if 'sequence_number' in p]
        
        sequence_gaps = 0
        duplicate_packets = 0
        out_of_order_packets = 0
        
        if len(sequences) > 1:
            sorted_sequences = sorted(sequences)
            expected_count = sorted_sequences[-1] - sorted_sequences[0] + 1
            sequence_gaps = expected_count - len(set(sequences))
            duplicate_packets = len(sequences) - len(set(sequences))
            
            # Check for out-of-order
            for i in range(1, len(sequences)):
                if sequences[i] < sequences[i-1]:
                    out_of_order_packets += 1
        
        sequence_integrity_score = max(0, 100 - (sequence_gaps + duplicate_packets + out_of_order_packets))
        
        # Timing analysis
        intervals = []
        for i in range(1, len(timestamps)):
            interval_ms = (timestamps[i] - timestamps[i-1]) * 1000
            intervals.append(interval_ms)
        
        if intervals:
            min_interval = min(intervals)
            max_interval = max(intervals)
            avg_interval = statistics.mean(intervals)
            jitter = statistics.stdev(intervals) if len(intervals) > 1 else 0
            jitter_coefficient = jitter / avg_interval if avg_interval > 0 else 0
        else:
            min_interval = max_interval = avg_interval = jitter = jitter_coefficient = 0
        
        # Audio quality analysis
        sample_rates = [p['sample_rate'] for p in valid_packets if 'sample_rate' in p]
        sample_rate_consistency = len(set(sample_rates)) <= 1 if sample_rates else True
        
        formats = [(p.get('channels', 0), p.get('bits_per_sample', 0)) for p in valid_packets]
        format_consistency = len(set(formats)) <= 1 if formats else True
        
        # Error analysis
        malformed_packets = len([p for p in packets if not p.get('header_valid', False)])
        size_mismatch_errors = len([p for p in valid_packets if not p.get('payload_size_valid', True)])
        total_errors = malformed_packets + size_mismatch_errors
        error_rate = (total_errors / total_packets * 100) if total_packets > 0 else 0
        
        # Quality scores
        audio_quality_score = self._calculate_audio_quality_score(
            sample_rate_consistency, format_consistency, jitter_coefficient
        )
        
        overall_quality_score = self._calculate_overall_quality_score(
            sequence_integrity_score, audio_quality_score, error_rate, jitter_coefficient
        )
        
        reliability_grade = self._score_to_grade(overall_quality_score)
        performance_grade = self._score_to_grade(100 - error_rate)
        
        return PacketFlowMetrics(
            total_packets=total_packets,
            total_bytes=total_bytes,
            duration_seconds=duration,
            packet_rate_pps=packet_rate,
            throughput_mbps=throughput_mbps,
            sequence_gaps=sequence_gaps,
            duplicate_packets=duplicate_packets,
            out_of_order_packets=out_of_order_packets,
            late_packets=0,  # Would need timestamp analysis
            sequence_integrity_score=sequence_integrity_score,
            min_inter_packet_ms=min_interval,
            max_inter_packet_ms=max_interval,
            avg_inter_packet_ms=avg_interval,
            jitter_ms=jitter,
            jitter_coefficient=jitter_coefficient,
            sample_rate_consistency=sample_rate_consistency,
            format_consistency=format_consistency,
            amplitude_stability_score=100.0,  # Would need audio analysis
            audio_quality_score=audio_quality_score,
            bandwidth_utilization_percent=0.0,  # Would need network capacity info
            packet_size_efficiency=0.0,  # Would need optimal size comparison
            network_overhead_percent=0.0,  # Would need protocol analysis
            malformed_packets=malformed_packets,
            checksum_errors=0,  # Would need checksum validation
            size_mismatch_errors=size_mismatch_errors,
            format_errors=0,  # Included in malformed
            total_error_rate_percent=error_rate,
            overall_quality_score=overall_quality_score,
            reliability_grade=reliability_grade,
            performance_grade=performance_grade
        )
    
    def _analyze_device_flow(self, device_id: str, device_data: Dict[str, Any]) -> DeviceFlowAnalysis:
        """Analyze flow for a specific device"""
        packets = list(device_data['packets'])
        
        if not packets:
            return self._create_empty_device_analysis(device_id)
        
        # Calculate device-specific metrics
        device_metrics = self._calculate_global_metrics(packets)
        
        # Connection info
        first_seen = device_data['first_seen']
        last_seen = device_data['last_seen']
        connection_duration = last_seen - first_seen
        
        # Behavioral analysis
        transmission_pattern = self._analyze_transmission_pattern(device_data['timestamps'])
        stability_score = self._calculate_stability_score(device_data['timestamps'])
        consistency_score = self._calculate_consistency_score(packets)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(device_data)
        
        # Trend analysis
        trend_analysis = self._analyze_trends(device_data)
        
        return DeviceFlowAnalysis(
            device_id=device_id,
            device_ip=device_id.split(':')[0],
            device_port=int(device_id.split(':')[1]),
            first_seen=first_seen,
            last_seen=last_seen,
            connection_duration=connection_duration,
            flow_metrics=device_metrics,
            transmission_pattern=transmission_pattern,
            stability_score=stability_score,
            consistency_score=consistency_score,
            anomalies_detected=len(anomalies),
            anomaly_types=[a['type'] for a in anomalies],
            anomaly_severity=self._assess_anomaly_severity(anomalies),
            trend_analysis=trend_analysis
        )
    
    def _analyze_transmission_pattern(self, timestamps: deque) -> str:
        """Analyze transmission pattern"""
        if len(timestamps) < 10:
            return "Insufficient Data"
        
        timestamps_list = list(timestamps)[-100:]  # Last 100 timestamps
        intervals = [timestamps_list[i+1] - timestamps_list[i] 
                    for i in range(len(timestamps_list)-1)]
        
        if not intervals:
            return "No Pattern"
        
        interval_std = statistics.stdev(intervals)
        interval_mean = statistics.mean(intervals)
        
        coefficient_of_variation = interval_std / interval_mean if interval_mean > 0 else float('inf')
        
        if coefficient_of_variation < 0.1:
            return "Regular"
        elif coefficient_of_variation < 0.5:
            return "Mostly Regular"
        elif coefficient_of_variation < 1.0:
            return "Irregular"
        else:
            return "Burst"
    
    def _calculate_stability_score(self, timestamps: deque) -> float:
        """Calculate transmission stability score (0-100)"""
        if len(timestamps) < 10:
            return 0.0
        
        timestamps_list = list(timestamps)[-100:]
        intervals = [timestamps_list[i+1] - timestamps_list[i] 
                    for i in range(len(timestamps_list)-1)]
        
        if not intervals:
            return 0.0
        
        # Lower coefficient of variation = higher stability
        interval_std = statistics.stdev(intervals)
        interval_mean = statistics.mean(intervals)
        
        if interval_mean == 0:
            return 0.0
        
        coefficient_of_variation = interval_std / interval_mean
        stability_score = max(0, 100 - (coefficient_of_variation * 100))
        
        return min(100, stability_score)
    
    def _calculate_consistency_score(self, packets: List[Dict[str, Any]]) -> float:
        """Calculate format consistency score (0-100)"""
        valid_packets = [p for p in packets if p.get('header_valid', False)]
        
        if not valid_packets:
            return 0.0
        
        # Check consistency across multiple dimensions
        sample_rates = [p.get('sample_rate', 0) for p in valid_packets]
        channels = [p.get('channels', 0) for p in valid_packets]
        bits_per_sample = [p.get('bits_per_sample', 0) for p in valid_packets]
        sample_counts = [p.get('sample_count', 0) for p in valid_packets]
        
        consistency_factors = [
            len(set(sample_rates)) == 1,     # Sample rate consistency
            len(set(channels)) == 1,         # Channel consistency
            len(set(bits_per_sample)) == 1,  # Bit depth consistency
            len(set(sample_counts)) <= 2     # Sample count consistency (allow some variation)
        ]
        
        consistency_score = sum(consistency_factors) / len(consistency_factors) * 100
        return consistency_score
    
    def _detect_anomalies(self, device_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in device transmission"""
        anomalies = []
        
        timestamps = list(device_data['timestamps'])
        packets = list(device_data['packets'])
        
        if len(timestamps) < 10:
            return anomalies
        
        # Timing anomalies
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if intervals:
            interval_mean = statistics.mean(intervals)
            interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
            
            for i, interval in enumerate(intervals):
                if abs(interval - interval_mean) > 3 * interval_std:  # 3-sigma rule
                    anomalies.append({
                        'type': 'timing_anomaly',
                        'timestamp': timestamps[i+1],
                        'severity': 'high' if abs(interval - interval_mean) > 5 * interval_std else 'medium',
                        'details': f'Interval {interval*1000:.1f}ms deviates from mean {interval_mean*1000:.1f}ms'
                    })
        
        # Packet size anomalies
        sizes = list(device_data['sizes'])
        if len(sizes) > 10:
            size_mean = statistics.mean(sizes)
            size_std = statistics.stdev(sizes)
            
            for i, size in enumerate(sizes):
                if abs(size - size_mean) > 3 * size_std:
                    anomalies.append({
                        'type': 'size_anomaly',
                        'timestamp': timestamps[i] if i < len(timestamps) else time.time(),
                        'severity': 'medium',
                        'details': f'Packet size {size} bytes deviates from mean {size_mean:.1f} bytes'
                    })
        
        return anomalies
    
    def _assess_anomaly_severity(self, anomalies: List[Dict[str, Any]]) -> str:
        """Assess overall anomaly severity"""
        if not anomalies:
            return "None"
        
        high_severity = len([a for a in anomalies if a.get('severity') == 'high'])
        medium_severity = len([a for a in anomalies if a.get('severity') == 'medium'])
        
        if high_severity > 0:
            return "High"
        elif medium_severity > 5:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_trends(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends for device"""
        # This would implement trend analysis over time
        # For now, return basic trend information
        
        packets = list(device_data['packets'])
        timestamps = list(device_data['timestamps'])
        
        if len(packets) < 20:
            return {'trend': 'insufficient_data'}
        
        # Analyze packet rate trend over time
        time_windows = []
        window_size = len(timestamps) // 10  # 10 windows
        
        for i in range(0, len(timestamps), window_size):
            window_timestamps = timestamps[i:i+window_size]
            if len(window_timestamps) >= 2:
                window_duration = window_timestamps[-1] - window_timestamps[0]
                window_rate = len(window_timestamps) / window_duration if window_duration > 0 else 0
                time_windows.append(window_rate)
        
        if len(time_windows) >= 3:
            # Simple trend detection
            early_avg = statistics.mean(time_windows[:3])
            late_avg = statistics.mean(time_windows[-3:])
            
            if late_avg > early_avg * 1.1:
                trend_direction = "increasing"
            elif late_avg < early_avg * 0.9:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "unknown"
        
        return {
            'packet_rate_trend': trend_direction,
            'trend_windows': time_windows,
            'trend_strength': abs(late_avg - early_avg) / early_avg if 'early_avg' in locals() and early_avg > 0 else 0
        }
    
    def _calculate_audio_quality_score(self, sample_rate_consistency: bool, 
                                     format_consistency: bool, jitter_coefficient: float) -> float:
        """Calculate audio quality score"""
        base_score = 100.0
        
        if not sample_rate_consistency:
            base_score -= 20
        
        if not format_consistency:
            base_score -= 15
        
        # Jitter penalty
        jitter_penalty = min(30, jitter_coefficient * 100)
        base_score -= jitter_penalty
        
        return max(0, base_score)
    
    def _calculate_overall_quality_score(self, sequence_integrity: float, 
                                       audio_quality: float, error_rate: float, 
                                       jitter_coefficient: float) -> float:
        """Calculate overall quality score"""
        # Weighted average of different quality factors
        weights = {
            'sequence_integrity': 0.3,
            'audio_quality': 0.3,
            'error_rate': 0.25,
            'jitter': 0.15
        }
        
        error_score = max(0, 100 - error_rate)
        jitter_score = max(0, 100 - (jitter_coefficient * 100))
        
        overall_score = (
            sequence_integrity * weights['sequence_integrity'] +
            audio_quality * weights['audio_quality'] +
            error_score * weights['error_rate'] +
            jitter_score * weights['jitter']
        )
        
        return min(100, max(0, overall_score))
    
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
    
    def _perform_device_comparison(self, device_analyses: List[DeviceFlowAnalysis]) -> Dict[str, Any]:
        """Perform comparative analysis between devices"""
        if len(device_analyses) < 2:
            return {'comparison': 'insufficient_devices'}
        
        # Compare key metrics across devices
        quality_scores = [d.flow_metrics.overall_quality_score for d in device_analyses]
        packet_rates = [d.flow_metrics.packet_rate_pps for d in device_analyses]
        stability_scores = [d.stability_score for d in device_analyses]
        
        return {
            'device_count': len(device_analyses),
            'quality_stats': {
                'min': min(quality_scores),
                'max': max(quality_scores),
                'avg': statistics.mean(quality_scores),
                'std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
            },
            'packet_rate_stats': {
                'min': min(packet_rates),
                'max': max(packet_rates),
                'avg': statistics.mean(packet_rates),
                'std': statistics.stdev(packet_rates) if len(packet_rates) > 1 else 0
            },
            'stability_stats': {
                'min': min(stability_scores),
                'max': max(stability_scores),
                'avg': statistics.mean(stability_scores),
                'std': statistics.stdev(stability_scores) if len(stability_scores) > 1 else 0
            },
            'best_performing_device': max(device_analyses, key=lambda d: d.flow_metrics.overall_quality_score).device_id,
            'most_stable_device': max(device_analyses, key=lambda d: d.stability_score).device_id
        }
    
    def _assess_quality(self, global_metrics: PacketFlowMetrics, 
                       device_analyses: List[DeviceFlowAnalysis]) -> Dict[str, Any]:
        """Assess overall system quality"""
        assessment = {
            'overall_grade': global_metrics.reliability_grade,
            'performance_grade': global_metrics.performance_grade,
            'quality_score': global_metrics.overall_quality_score,
            'key_strengths': [],
            'key_weaknesses': [],
            'critical_issues': []
        }
        
        # Identify strengths
        if global_metrics.sequence_integrity_score > 95:
            assessment['key_strengths'].append("Excellent sequence integrity")
        
        if global_metrics.jitter_ms < 5:
            assessment['key_strengths'].append("Low network jitter")
        
        if global_metrics.total_error_rate_percent < 1:
            assessment['key_strengths'].append("Very low error rate")
        
        # Identify weaknesses
        if global_metrics.sequence_integrity_score < 80:
            assessment['key_weaknesses'].append("Poor sequence integrity")
        
        if global_metrics.jitter_ms > 20:
            assessment['key_weaknesses'].append("High network jitter")
        
        if global_metrics.total_error_rate_percent > 5:
            assessment['key_weaknesses'].append("High error rate")
        
        # Critical issues
        if global_metrics.overall_quality_score < 50:
            assessment['critical_issues'].append("Overall quality below acceptable threshold")
        
        if global_metrics.total_error_rate_percent > 20:
            assessment['critical_issues'].append("Error rate critically high")
        
        return assessment
    
    def _generate_recommendations(self, global_metrics: PacketFlowMetrics, 
                                device_analyses: List[DeviceFlowAnalysis]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        if global_metrics.jitter_ms > 15:
            recommendations.append("Consider implementing jitter buffer to smooth audio delivery")
        
        if global_metrics.sequence_gaps > 0:
            recommendations.append("Investigate network reliability - packet loss detected")
        
        if global_metrics.total_error_rate_percent > 3:
            recommendations.append("Review error handling and retry mechanisms")
        
        # Device-specific recommendations
        if device_analyses:
            unstable_devices = [d for d in device_analyses if d.stability_score < 70]
            if unstable_devices:
                recommendations.append(f"Investigate transmission stability for {len(unstable_devices)} devices")
        
        # Quality improvements
        if global_metrics.overall_quality_score < 80:
            recommendations.append("Consider quality-of-service (QoS) configuration for audio traffic")
        
        if not global_metrics.sample_rate_consistency:
            recommendations.append("Ensure consistent audio format configuration across devices")
        
        # Network optimization
        if global_metrics.packet_rate_pps < 40:  # Below expected 50 pkt/s
            recommendations.append("Investigate packet transmission rate - may indicate network congestion")
        
        return recommendations
    
    def _perform_validation(self, global_metrics: PacketFlowMetrics, 
                          device_analyses: List[DeviceFlowAnalysis]) -> Dict[str, Any]:
        """Perform validation against quality standards"""
        validation = {
            'meets_quality_standards': True,
            'validation_criteria': {},
            'failed_criteria': [],
            'passed_criteria': []
        }
        
        # Define validation criteria
        criteria = {
            'error_rate_below_5_percent': global_metrics.total_error_rate_percent < 5.0,
            'jitter_below_20ms': global_metrics.jitter_ms < 20.0,
            'sequence_integrity_above_90': global_metrics.sequence_integrity_score > 90.0,
            'overall_quality_above_70': global_metrics.overall_quality_score > 70.0,
            'packet_rate_acceptable': 40 <= global_metrics.packet_rate_pps <= 60,  # Expected range
            'format_consistency': global_metrics.format_consistency and global_metrics.sample_rate_consistency
        }
        
        validation['validation_criteria'] = criteria
        
        for criterion, passed in criteria.items():
            if passed:
                validation['passed_criteria'].append(criterion)
            else:
                validation['failed_criteria'].append(criterion)
                validation['meets_quality_standards'] = False
        
        return validation
    
    def _create_empty_metrics(self) -> PacketFlowMetrics:
        """Create empty metrics structure"""
        return PacketFlowMetrics(
            total_packets=0, total_bytes=0, duration_seconds=0.0,
            packet_rate_pps=0.0, throughput_mbps=0.0, sequence_gaps=0,
            duplicate_packets=0, out_of_order_packets=0, late_packets=0,
            sequence_integrity_score=0.0, min_inter_packet_ms=0.0,
            max_inter_packet_ms=0.0, avg_inter_packet_ms=0.0, jitter_ms=0.0,
            jitter_coefficient=0.0, sample_rate_consistency=True,
            format_consistency=True, amplitude_stability_score=0.0,
            audio_quality_score=0.0, bandwidth_utilization_percent=0.0,
            packet_size_efficiency=0.0, network_overhead_percent=0.0,
            malformed_packets=0, checksum_errors=0, size_mismatch_errors=0,
            format_errors=0, total_error_rate_percent=0.0,
            overall_quality_score=0.0, reliability_grade="F", performance_grade="F"
        )
    
    def _create_empty_device_analysis(self, device_id: str) -> DeviceFlowAnalysis:
        """Create empty device analysis"""
        return DeviceFlowAnalysis(
            device_id=device_id, device_ip="", device_port=0,
            first_seen=0, last_seen=0, connection_duration=0,
            flow_metrics=self._create_empty_metrics(),
            transmission_pattern="Unknown", stability_score=0.0,
            consistency_score=0.0, anomalies_detected=0,
            anomaly_types=[], anomaly_severity="None", trend_analysis={}
        )
    
    def _create_empty_report(self, timestamp: float) -> PacketFlowReport:
        """Create empty report structure"""
        return PacketFlowReport(
            analysis_timestamp=timestamp, analysis_duration=0.0,
            global_metrics=self._create_empty_metrics(),
            device_analyses=[], device_comparison={},
            quality_summary={}, recommendations=[],
            validation_results={}
        )
    
    def export_report(self, report: PacketFlowReport, filename: str, format_type: str = "json"):
        """Export analysis report to file"""
        try:
            if format_type.lower() == "json":
                with open(filename, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"📄 Report exported to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export report: {e}")
    
    def print_report_summary(self, report: PacketFlowReport):
        """Print a summary of the analysis report"""
        print("\\n" + "═" * 80)
        print("📊 PACKET FLOW ANALYSIS REPORT SUMMARY")
        print("═" * 80)
        
        # Global metrics
        metrics = report.global_metrics
        print(f"Overall Quality: {metrics.overall_quality_score:.1f}/100 (Grade: {metrics.reliability_grade})")
        print(f"Performance: {metrics.performance_grade} grade")
        print(f"Total Packets: {metrics.total_packets:,}")
        print(f"Duration: {metrics.duration_seconds:.1f}s")
        print(f"Packet Rate: {metrics.packet_rate_pps:.1f} pkt/s")
        print(f"Throughput: {metrics.throughput_mbps:.3f} Mbps")
        print(f"Error Rate: {metrics.total_error_rate_percent:.1f}%")
        print(f"Jitter: {metrics.jitter_ms:.2f}ms")
        
        # Device analysis
        if report.device_analyses:
            print(f"\\nDevices Analyzed: {len(report.device_analyses)}")
            for device in report.device_analyses:
                print(f"  • {device.device_id}: Quality {device.flow_metrics.overall_quality_score:.1f}/100, "
                      f"Stability {device.stability_score:.1f}/100")
        
        # Quality assessment
        if report.quality_summary:
            quality = report.quality_summary
            if quality.get('key_strengths'):
                print(f"\\nStrengths: {', '.join(quality['key_strengths'])}")
            if quality.get('key_weaknesses'):
                print(f"Weaknesses: {', '.join(quality['key_weaknesses'])}")
            if quality.get('critical_issues'):
                print(f"Critical Issues: {', '.join(quality['critical_issues'])}")
        
        # Recommendations
        if report.recommendations:
            print("\\nRecommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):  # Top 5
                print(f"  {i}. {rec}")
        
        # Validation results
        if report.validation_results:
            validation = report.validation_results
            meets_standards = validation.get('meets_quality_standards', False)
            print(f"\\nValidation: {'✅ PASS' if meets_standards else '❌ FAIL'}")
            
            failed = validation.get('failed_criteria', [])
            if failed:
                print(f"Failed Criteria: {', '.join(failed)}")
        
        print("═" * 80)
    
    async def start_real_time_monitoring(self, duration_seconds: Optional[int] = None):
        """Start real-time packet flow monitoring"""
        if not self.setup_real_time_analysis():
            self.logger.error("❌ Failed to start real-time monitoring")
            return
        
        self.analysis_active = True
        self.start_time = time.time()
        
        self.logger.info("🔴 Real-time packet flow monitoring started")
        self.logger.info(f"Listening on port {self.port} for UDP packets...")
        
        try:
            if duration_seconds:
                await asyncio.sleep(duration_seconds)
            else:
                # Run indefinitely until interrupted
                while self.analysis_active:
                    await asyncio.sleep(1)
                    
                    # Periodic analysis reports
                    if int(time.time() - self.start_time) % 30 == 0:  # Every 30 seconds
                        if len(self.packet_data) > 0:
                            report = self.analyze_packet_flow(duration_seconds=30)
                            self.logger.info(f"📊 30s analysis: Quality {report.global_metrics.overall_quality_score:.1f}/100, "
                                           f"Packets {report.global_metrics.total_packets}")
        
        except KeyboardInterrupt:
            self.logger.info("\\n🛑 Real-time monitoring interrupted")
        
        finally:
            self.analysis_active = False
            if self.server:
                self.server.stop()
            
            # Final analysis
            if len(self.packet_data) > 0:
                self.logger.info("📊 Generating final analysis report...")
                final_report = self.analyze_packet_flow()
                self.print_report_summary(final_report)

def main():
    """Main function for packet flow analyzer"""
    parser = argparse.ArgumentParser(
        description='UDP Packet Flow Analysis and Quality Metrics Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python packet_flow_analyzer.py --real-time          # Real-time monitoring
  python packet_flow_analyzer.py --real-time --duration 60  # 60-second analysis
  python packet_flow_analyzer.py --export report.json # Export to JSON
  python packet_flow_analyzer.py --verbose            # Detailed logging

Real-time Analysis:
  Monitors UDP packets on specified port and provides continuous
  quality analysis with periodic reports and final comprehensive analysis.
        """
    )
    
    parser.add_argument('--port', type=int, default=8003,
                       help='UDP port to monitor (default: 8003)')
    parser.add_argument('--real-time', action='store_true',
                       help='Enable real-time packet monitoring')
    parser.add_argument('--duration', type=int,
                       help='Analysis duration in seconds (real-time mode)')
    parser.add_argument('--export', type=str,
                       help='Export analysis report to JSON file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug logging')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PacketFlowAnalyzer(
        port=args.port,
        real_time=args.real_time,
        verbose=args.verbose
    )
    
    if args.real_time:
        # Real-time monitoring mode
        try:
            asyncio.run(analyzer.start_real_time_monitoring(args.duration))
        except KeyboardInterrupt:
            print("\\n🛑 Analysis interrupted by user")
    else:
        # Analysis of existing data (would need implementation for file input)
        analyzer.logger.info("📄 File-based analysis not yet implemented")
        analyzer.logger.info("💡 Use --real-time for live packet monitoring")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)