#!/usr/bin/env python3
"""
Network Interface Diagnostics for HowdyTTS ESP32-P4 Integration

This script diagnoses network interface binding issues that may prevent
the HowdyTTS server from receiving UDP packets from ESP32-P4 devices.

Key Diagnostic Areas:
1. Network interface enumeration and IP addresses
2. UDP socket binding validation on different interfaces  
3. Subnet configuration analysis (192.168.86.x vs 192.168.0.x)
4. Firewall and port accessibility testing
5. Broadcast reception testing

Usage:
    python network_interface_diagnostics.py [--port 8003] [--test-broadcast]
"""

import socket
import struct
import logging
import time
import threading
from typing import List, Dict, Any, Tuple
import argparse
import subprocess
import platform
import netifaces

class NetworkInterfaceDiagnostics:
    """Comprehensive network interface diagnostics for UDP reception."""
    
    def __init__(self, port: int = 8003):
        self.port = port
        self.results = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_network_interfaces(self) -> List[Dict[str, Any]]:
        """Get all network interfaces with their IP addresses."""
        interfaces = []
        
        try:
            for interface_name in netifaces.interfaces():
                interface_info = {
                    'name': interface_name,
                    'addresses': {}
                }
                
                # Get address information for this interface
                addr_info = netifaces.ifaddresses(interface_name)
                
                # IPv4 addresses
                if netifaces.AF_INET in addr_info:
                    ipv4_addrs = []
                    for addr in addr_info[netifaces.AF_INET]:
                        ipv4_addrs.append({
                            'ip': addr.get('addr'),
                            'netmask': addr.get('netmask'),
                            'broadcast': addr.get('broadcast')
                        })
                    interface_info['addresses']['ipv4'] = ipv4_addrs
                
                # Only include interfaces with IPv4 addresses
                if 'ipv4' in interface_info['addresses'] and interface_info['addresses']['ipv4']:
                    interfaces.append(interface_info)
                    
        except Exception as e:
            self.logger.error(f"Error getting network interfaces: {e}")
            
            # Fallback method using socket
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                interfaces.append({
                    'name': 'default',
                    'addresses': {
                        'ipv4': [{'ip': local_ip, 'netmask': None, 'broadcast': None}]
                    }
                })
            except Exception as fallback_e:
                self.logger.error(f"Fallback method also failed: {fallback_e}")
        
        return interfaces
    
    def calculate_network_info(self, ip: str, netmask: str) -> Dict[str, str]:
        """Calculate network address and broadcast address from IP and netmask."""
        if not ip or not netmask:
            return {}
        
        try:
            # Convert IP and netmask to integers
            ip_int = struct.unpack("!I", socket.inet_aton(ip))[0]
            mask_int = struct.unpack("!I", socket.inet_aton(netmask))[0]
            
            # Calculate network and broadcast
            network_int = ip_int & mask_int
            broadcast_int = network_int | (~mask_int & 0xFFFFFFFF)
            
            # Convert back to strings
            network = socket.inet_ntoa(struct.pack("!I", network_int))
            broadcast = socket.inet_ntoa(struct.pack("!I", broadcast_int))
            
            return {
                'network': network,
                'broadcast': broadcast
            }
        except Exception as e:
            self.logger.error(f"Error calculating network info for {ip}/{netmask}: {e}")
            return {}
    
    def test_udp_binding(self, interface_ip: str = "0.0.0.0") -> Dict[str, Any]:
        """Test UDP socket binding on specific interface."""
        result = {
            'interface': interface_ip,
            'port': self.port,
            'bind_success': False,
            'error': None,
            'socket_info': {}
        }
        
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Try binding
            sock.bind((interface_ip, self.port))
            result['bind_success'] = True
            
            # Get socket information
            sock_name = sock.getsockname()
            result['socket_info'] = {
                'bound_address': sock_name[0],
                'bound_port': sock_name[1]
            }
            
            # Test socket buffer sizes
            try:
                recv_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                send_buf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                result['socket_info']['recv_buffer'] = recv_buf
                result['socket_info']['send_buffer'] = send_buf
            except Exception as buf_e:
                self.logger.debug(f"Could not get buffer sizes: {buf_e}")
            
            sock.close()
            
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def test_broadcast_reception(self, duration: int = 5) -> Dict[str, Any]:
        """Test ability to receive broadcast packets."""
        result = {
            'test_duration': duration,
            'packets_received': 0,
            'unique_senders': set(),
            'errors': []
        }
        
        try:
            # Create broadcast receiver
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.settimeout(1.0)
            
            self.logger.info(f"🔍 Testing broadcast reception on port {self.port} for {duration} seconds...")
            
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    data, addr = sock.recvfrom(1024)
                    result['packets_received'] += 1
                    result['unique_senders'].add(addr[0])
                    
                    if result['packets_received'] <= 5:  # Log first few packets
                        self.logger.info(f"📦 Received packet from {addr[0]}:{addr[1]}: {len(data)} bytes")
                        
                except socket.timeout:
                    continue
                except Exception as recv_e:
                    result['errors'].append(f"Receive error: {recv_e}")
            
            sock.close()
            result['unique_senders'] = list(result['unique_senders'])
            
        except Exception as e:
            result['errors'].append(f"Broadcast test error: {e}")
        
        return result
    
    def check_port_accessibility(self) -> Dict[str, Any]:
        """Check if the UDP port is accessible and not blocked."""
        result = {
            'port': self.port,
            'accessible': False,
            'listening_processes': [],
            'error': None
        }
        
        try:
            # Check if anything is already listening on the port
            if platform.system() != "Windows":
                try:
                    cmd = f"netstat -ulnp | grep :{self.port}"
                    output = subprocess.check_output(cmd, shell=True, text=True)
                    if output.strip():
                        result['listening_processes'] = output.strip().split('\n')
                        self.logger.warning(f"⚠️ Port {self.port} may already be in use:")
                        for line in result['listening_processes']:
                            self.logger.warning(f"   {line}")
                except subprocess.CalledProcessError:
                    # No processes listening - this is good
                    pass
            
            # Test if we can bind to the port
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_sock.bind(("0.0.0.0", self.port))
            test_sock.close()
            result['accessible'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def analyze_esp32_connectivity(self, esp32_ip: str = "192.168.0.151") -> Dict[str, Any]:
        """Analyze connectivity to specific ESP32-P4 device."""
        result = {
            'esp32_ip': esp32_ip,
            'reachable': False,
            'same_subnet': False,
            'route_info': {},
            'ping_result': None
        }
        
        try:
            # Check if ESP32 IP is reachable
            if platform.system() == "Windows":
                ping_cmd = f"ping -n 1 -W 1000 {esp32_ip}"
            else:
                ping_cmd = f"ping -c 1 -W 1 {esp32_ip}"
            
            try:
                ping_output = subprocess.check_output(ping_cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                result['reachable'] = True
                result['ping_result'] = "Success"
            except subprocess.CalledProcessError:
                result['ping_result'] = "Failed"
            
            # Check if ESP32 is on same subnet as any local interface
            interfaces = self.get_network_interfaces()
            for interface in interfaces:
                for ipv4_addr in interface['addresses'].get('ipv4', []):
                    local_ip = ipv4_addr['ip']
                    netmask = ipv4_addr.get('netmask')
                    
                    if local_ip and netmask and esp32_ip:
                        # Check if ESP32 is in same subnet
                        try:
                            local_int = struct.unpack("!I", socket.inet_aton(local_ip))[0]
                            esp32_int = struct.unpack("!I", socket.inet_aton(esp32_ip))[0]
                            mask_int = struct.unpack("!I", socket.inet_aton(netmask))[0]
                            
                            if (local_int & mask_int) == (esp32_int & mask_int):
                                result['same_subnet'] = True
                                result['route_info'] = {
                                    'local_interface': interface['name'],
                                    'local_ip': local_ip,
                                    'netmask': netmask
                                }
                                break
                        except Exception:
                            continue
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def run_full_diagnostics(self, esp32_ip: str = "192.168.0.151", test_broadcast: bool = False) -> Dict[str, Any]:
        """Run complete network diagnostics suite."""
        self.logger.info("🚀 Starting comprehensive network interface diagnostics...")
        self.logger.info("=" * 80)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'port': self.port,
            'esp32_ip': esp32_ip
        }
        
        # 1. Network Interface Analysis
        self.logger.info("📡 1. NETWORK INTERFACE ANALYSIS")
        self.logger.info("-" * 40)
        
        interfaces = self.get_network_interfaces()
        results['interfaces'] = interfaces
        
        for interface in interfaces:
            self.logger.info(f"Interface: {interface['name']}")
            for ipv4_addr in interface['addresses'].get('ipv4', []):
                ip = ipv4_addr['ip']
                netmask = ipv4_addr.get('netmask', 'Unknown')
                broadcast = ipv4_addr.get('broadcast', 'Unknown')
                
                self.logger.info(f"  IP: {ip}")
                self.logger.info(f"  Netmask: {netmask}")
                self.logger.info(f"  Broadcast: {broadcast}")
                
                # Calculate network info
                if netmask and netmask != 'Unknown':
                    network_info = self.calculate_network_info(ip, netmask)
                    if network_info:
                        self.logger.info(f"  Network: {network_info.get('network', 'Unknown')}")
            self.logger.info("")
        
        # 2. UDP Socket Binding Tests
        self.logger.info("🔌 2. UDP SOCKET BINDING TESTS")
        self.logger.info("-" * 40)
        
        binding_tests = []
        
        # Test binding to all interfaces
        test_result = self.test_udp_binding("0.0.0.0")
        binding_tests.append(test_result)
        
        if test_result['bind_success']:
            self.logger.info(f"✅ Successfully bound to 0.0.0.0:{self.port}")
            self.logger.info(f"   Socket info: {test_result['socket_info']}")
        else:
            self.logger.error(f"❌ Failed to bind to 0.0.0.0:{self.port}: {test_result['error']}")
        
        # Test binding to specific interfaces
        for interface in interfaces:
            for ipv4_addr in interface['addresses'].get('ipv4', []):
                ip = ipv4_addr['ip']
                if ip and ip != "127.0.0.1":  # Skip localhost
                    test_result = self.test_udp_binding(ip)
                    binding_tests.append(test_result)
                    
                    if test_result['bind_success']:
                        self.logger.info(f"✅ Successfully bound to {ip}:{self.port}")
                    else:
                        self.logger.error(f"❌ Failed to bind to {ip}:{self.port}: {test_result['error']}")
        
        results['binding_tests'] = binding_tests
        self.logger.info("")
        
        # 3. Port Accessibility Check
        self.logger.info("🚪 3. PORT ACCESSIBILITY CHECK")
        self.logger.info("-" * 40)
        
        port_check = self.check_port_accessibility()
        results['port_accessibility'] = port_check
        
        if port_check['accessible']:
            self.logger.info(f"✅ Port {self.port} is accessible")
        else:
            self.logger.error(f"❌ Port {self.port} is not accessible: {port_check['error']}")
        
        if port_check['listening_processes']:
            self.logger.warning(f"⚠️ Processes already listening on port {self.port}:")
            for process in port_check['listening_processes']:
                self.logger.warning(f"   {process}")
        
        self.logger.info("")
        
        # 4. ESP32-P4 Connectivity Analysis
        self.logger.info(f"📱 4. ESP32-P4 CONNECTIVITY ANALYSIS ({esp32_ip})")
        self.logger.info("-" * 40)
        
        esp32_analysis = self.analyze_esp32_connectivity(esp32_ip)
        results['esp32_connectivity'] = esp32_analysis
        
        if esp32_analysis['reachable']:
            self.logger.info(f"✅ ESP32-P4 at {esp32_ip} is reachable")
        else:
            self.logger.warning(f"⚠️ ESP32-P4 at {esp32_ip} is not reachable via ping")
        
        if esp32_analysis['same_subnet']:
            route_info = esp32_analysis['route_info']
            self.logger.info(f"✅ ESP32-P4 is on same subnet as local interface {route_info['local_interface']}")
            self.logger.info(f"   Local IP: {route_info['local_ip']}")
            self.logger.info(f"   Netmask: {route_info['netmask']}")
        else:
            self.logger.warning(f"⚠️ ESP32-P4 is not on same subnet as any local interface")
            self.logger.warning("   This may cause UDP broadcast/discovery issues")
        
        self.logger.info("")
        
        # 5. Broadcast Reception Test (Optional)
        if test_broadcast:
            self.logger.info("📻 5. BROADCAST RECEPTION TEST")
            self.logger.info("-" * 40)
            
            broadcast_test = self.test_broadcast_reception(duration=10)
            results['broadcast_test'] = broadcast_test
            
            if broadcast_test['packets_received'] > 0:
                self.logger.info(f"✅ Received {broadcast_test['packets_received']} broadcast packets")
                self.logger.info(f"   From {len(broadcast_test['unique_senders'])} unique senders: {broadcast_test['unique_senders']}")
            else:
                self.logger.warning("⚠️ No broadcast packets received during test")
            
            if broadcast_test['errors']:
                for error in broadcast_test['errors']:
                    self.logger.error(f"❌ {error}")
            
            self.logger.info("")
        
        # 6. Summary and Recommendations
        self.logger.info("📋 6. SUMMARY AND RECOMMENDATIONS")
        self.logger.info("-" * 40)
        
        # Analyze results and provide recommendations
        recommendations = []
        
        if not any(test['bind_success'] for test in binding_tests):
            recommendations.append("❌ CRITICAL: Cannot bind UDP socket - check firewall and permissions")
        
        if not port_check['accessible']:
            recommendations.append(f"❌ CRITICAL: Port {self.port} is not accessible - check firewall rules")
        
        if port_check['listening_processes']:
            recommendations.append(f"⚠️ WARNING: Port {self.port} may be in use by other processes")
        
        if not esp32_analysis['same_subnet']:
            recommendations.append("⚠️ WARNING: ESP32-P4 not on same subnet - may affect UDP discovery")
        
        if not esp32_analysis['reachable']:
            recommendations.append("⚠️ WARNING: ESP32-P4 not reachable - check network connectivity")
        
        # Check for common subnet mismatches
        esp32_subnet = '.'.join(esp32_ip.split('.')[:-1])
        local_subnets = []
        for interface in interfaces:
            for ipv4_addr in interface['addresses'].get('ipv4', []):
                ip = ipv4_addr['ip']
                if ip and ip != "127.0.0.1":
                    local_subnet = '.'.join(ip.split('.')[:-1])
                    local_subnets.append(local_subnet)
        
        if esp32_subnet not in local_subnets:
            recommendations.append(f"⚠️ WARNING: ESP32 subnet ({esp32_subnet}.x) differs from local subnets ({local_subnets})")
        
        if not recommendations:
            recommendations.append("✅ All network diagnostics passed - configuration looks good!")
        
        for rec in recommendations:
            self.logger.info(rec)
        
        results['recommendations'] = recommendations
        
        self.logger.info("=" * 80)
        self.logger.info("🏁 Network diagnostics complete!")
        
        return results

def main():
    """Main function for network interface diagnostics."""
    parser = argparse.ArgumentParser(
        description='Network Interface Diagnostics for HowdyTTS ESP32-P4 Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool diagnoses network interface issues that may prevent the HowdyTTS 
server from receiving UDP audio packets from ESP32-P4 devices.

Common Issues Detected:
  - UDP socket binding failures on 0.0.0.0:8003
  - Firewall blocking UDP port 8003
  - Subnet mismatches (192.168.86.x vs 192.168.0.x)
  - Network interface configuration problems
  - ESP32-P4 connectivity issues

Examples:
  python network_interface_diagnostics.py
  python network_interface_diagnostics.py --port 8003 --esp32-ip 192.168.0.151
  python network_interface_diagnostics.py --test-broadcast
        """
    )
    
    parser.add_argument('--port', type=int, default=8003,
                       help='UDP port to test (default: 8003)')
    parser.add_argument('--esp32-ip', type=str, default='192.168.0.151',
                       help='ESP32-P4 device IP address (default: 192.168.0.151)')
    parser.add_argument('--test-broadcast', action='store_true',
                       help='Include broadcast reception test (takes 10 seconds)')
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnostics = NetworkInterfaceDiagnostics(port=args.port)
    results = diagnostics.run_full_diagnostics(
        esp32_ip=args.esp32_ip,
        test_broadcast=args.test_broadcast
    )
    
    # Print final status
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSTIC RESULTS SUMMARY")
    print("=" * 80)
    
    success_count = 0
    total_tests = 4
    
    if any(test['bind_success'] for test in results['binding_tests']):
        print("✅ UDP socket binding: PASS")
        success_count += 1
    else:
        print("❌ UDP socket binding: FAIL")
    
    if results['port_accessibility']['accessible']:
        print(f"✅ Port {args.port} accessibility: PASS")
        success_count += 1
    else:
        print(f"❌ Port {args.port} accessibility: FAIL")
    
    if results['esp32_connectivity']['same_subnet']:
        print("✅ ESP32-P4 subnet compatibility: PASS")
        success_count += 1
    else:
        print("⚠️ ESP32-P4 subnet compatibility: WARNING")
    
    if results['esp32_connectivity']['reachable']:
        print("✅ ESP32-P4 reachability: PASS")
        success_count += 1
    else:
        print("⚠️ ESP32-P4 reachability: WARNING")
    
    print("-" * 80)
    print(f"Overall Status: {success_count}/{total_tests} tests passed")
    
    if success_count >= 2:
        print("🎉 Network configuration should support ESP32-P4 UDP audio reception!")
    else:
        print("🔧 Network configuration issues detected - see recommendations above")

if __name__ == "__main__":
    main()