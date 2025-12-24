#!/usr/bin/env python3
"""
NirogScan Health Monitor - Python BLE Client
============================================

Real-time health monitoring data receiver and visualizer for ESP32-S3 NirogScan device.

Features:
- BLE GATT client for data reception
- Real-time ECG and PPG signal visualization
- Data logging and export capabilities
- System monitoring (battery, temperature, signal quality)
- Industrial-grade error handling and data validation

Author: Embedded Systems Engineer
Version: 1.0.1
Date: 2025
"""

import asyncio
import struct
import time
import threading
import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Button
    from bleak import BleakClient, BleakScanner
    from bleak.backends.characteristic import BleakGATTCharacteristic
except ImportError as e:
    print(f"Required dependency missing: {e}")
    print("Install with: pip install bleak numpy matplotlib")
    sys.exit(1)

# Configuration Constants
DEVICE_NAME = "NirogScan"
HEALTH_SERVICE_UUID = "0000180D-0000-1000-8000-00805F9B34FB"  # Heart Rate Service
DATA_CHARACTERISTIC_UUID = "00002A37-0000-1000-8000-00805F9B34FB"  # Heart Rate Measurement

# Data Configuration
ECG_SAMPLES_PER_PACKET = 10
PPG_SAMPLES_PER_PACKET = 4
ECG_SAMPLE_RATE = 125  # Hz
PPG_SAMPLE_RATE = 50   # Hz
PACKET_RATE = 12.5     # Hz
PACKET_SIZE = 92       # bytes

# Visualization Configuration
PLOT_WINDOW_SECONDS = 10
ECG_BUFFER_SIZE = int(ECG_SAMPLE_RATE * PLOT_WINDOW_SECONDS)
PPG_BUFFER_SIZE = int(PPG_SAMPLE_RATE * PLOT_WINDOW_SECONDS)
MAX_PLOT_POINTS = 2000

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nirog_scan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class HealthDataPacket:
    """Health monitoring data packet structure matching ESP32 implementation."""
    sequence_number: int
    timestamp_start_us: int
    timestamp_end_us: int
    ecg_values: List[int]          # 10 samples
    leads_off_status: List[int]    # 10 samples
    ppg_red: List[int]            # 4 samples
    ppg_ir: List[int]             # 4 samples
    battery_voltage: float
    battery_percentage: float
    temperature: float
    ecg_quality: int              # 0-100%
    ppg_quality: int              # 0-100%
    system_status: int
    checksum: int
    
    @property
    def packet_duration_ms(self) -> float:
        """Calculate packet duration in milliseconds."""
        return (self.timestamp_end_us - self.timestamp_start_us) / 1000.0
    
    @property
    def is_ecg_leads_connected(self) -> bool:
        """Check if ECG leads are properly connected."""
        return self.ecg_quality >= 80
    
    @property
    def is_ppg_signal_good(self) -> bool:
        """Check if PPG signal quality is acceptable."""
        return self.ppg_quality >= 70

class DataValidator:
    """Validates incoming health monitoring data for integrity and quality."""
    
    @staticmethod
    def calculate_checksum(data: bytes) -> int:
        """Calculate simple checksum for data integrity validation."""
        return sum(data[:-2]) & 0xFFFF  # Exclude checksum bytes
    
    @staticmethod
    def validate_packet(packet: HealthDataPacket, raw_data: bytes) -> bool:
        """Validate health data packet integrity and ranges."""
        try:
            # Checksum validation
            calculated_checksum = DataValidator.calculate_checksum(raw_data)
            if calculated_checksum != packet.checksum:
                logger.warning(f"Checksum mismatch: calc={calculated_checksum}, recv={packet.checksum}")
                return False
            
            # Range validation
            if not (0 <= packet.ecg_quality <= 100):
                logger.warning(f"Invalid ECG quality: {packet.ecg_quality}")
                return False
                
            if not (0 <= packet.ppg_quality <= 100):
                logger.warning(f"Invalid PPG quality: {packet.ppg_quality}")
                return False
                
            if not (2.5 <= packet.battery_voltage <= 4.5):
                logger.warning(f"Invalid battery voltage: {packet.battery_voltage}V")
                return False
                
            if not (-40.0 <= packet.temperature <= 85.0):
                logger.warning(f"Invalid temperature: {packet.temperature}°C")
                return False
            
            # ECG range validation (12-bit ADC)
            for ecg_val in packet.ecg_values:
                if not (0 <= ecg_val <= 4095):
                    logger.warning(f"ECG value out of range: {ecg_val}")
                    return False
            
            # PPG range validation (18-bit)
            for ppg_val in packet.ppg_red + packet.ppg_ir:
                if not (0 <= ppg_val <= 262143):
                    logger.warning(f"PPG value out of range: {ppg_val}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

class DataParser:
    """Parses binary health monitoring data packets from ESP32."""
    
    @staticmethod
    def parse_packet(data: bytes) -> Optional[HealthDataPacket]:
        """Parse binary data packet into HealthDataPacket structure."""
        if len(data) != PACKET_SIZE:
            logger.error(f"Invalid packet size: {len(data)}, expected {PACKET_SIZE}")
            return None
        
        try:
            # Unpack binary data according to ESP32 structure
            # Note: Adjusted for non-packed structure to avoid alignment issues
            values = struct.unpack('<HLL10h10B4L4LfffBBBH', data)
            
            idx = 0
            sequence_number = values[idx]; idx += 1
            timestamp_start_us = values[idx]; idx += 1
            timestamp_end_us = values[idx]; idx += 1
            
            ecg_values = list(values[idx:idx+10]); idx += 10
            leads_off_status = list(values[idx:idx+10]); idx += 10
            
            ppg_red = list(values[idx:idx+4]); idx += 4
            ppg_ir = list(values[idx:idx+4]); idx += 4
            
            battery_voltage = values[idx]; idx += 1
            battery_percentage = values[idx]; idx += 1
            temperature = values[idx]; idx += 1
            
            ecg_quality = values[idx]; idx += 1
            ppg_quality = values[idx]; idx += 1
            system_status = values[idx]; idx += 1
            checksum = values[idx]
            
            packet = HealthDataPacket(
                sequence_number=sequence_number,
                timestamp_start_us=timestamp_start_us,
                timestamp_end_us=timestamp_end_us,
                ecg_values=ecg_values,
                leads_off_status=leads_off_status,
                ppg_red=ppg_red,
                ppg_ir=ppg_ir,
                battery_voltage=battery_voltage,
                battery_percentage=battery_percentage,
                temperature=temperature,
                ecg_quality=ecg_quality,
                ppg_quality=ppg_quality,
                system_status=system_status,
                checksum=checksum
            )
            
            # Validate packet
            if not DataValidator.validate_packet(packet, data):
                return None
                
            return packet
            
        except struct.error as e:
            logger.error(f"Struct unpacking error: {e}")
            return None
        except Exception as e:
            logger.error(f"Packet parsing error: {e}")
            return None

class DataLogger:
    """Handles data logging to CSV and JSON formats."""
    
    def __init__(self, base_filename: str):
        """Initialize data logger with base filename."""
        self.base_filename = base_filename
        self.csv_filename = f"{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.json_filename = f"{base_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Initialize CSV file
        self._init_csv()
        
        # JSON data accumulator
        self.json_data = []
        
    def _init_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                headers = [
                    'timestamp', 'sequence', 'ecg_quality', 'ppg_quality',
                    'battery_voltage', 'battery_percentage', 'temperature',
                    'system_status', 'packet_duration_ms'
                ]
                
                # Add ECG sample headers
                for i in range(ECG_SAMPLES_PER_PACKET):
                    headers.append(f'ecg_{i}')
                    headers.append(f'lo_status_{i}')
                
                # Add PPG sample headers  
                for i in range(PPG_SAMPLES_PER_PACKET):
                    headers.append(f'ppg_red_{i}')
                    headers.append(f'ppg_ir_{i}')
                
                writer.writerow(headers)
                
            logger.info(f"CSV logging initialized: {self.csv_filename}")
            
        except Exception as e:
            logger.error(f"CSV initialization failed: {e}")
    
    def log_packet(self, packet: HealthDataPacket):
        """Log health data packet to CSV and JSON."""
        timestamp = datetime.now().isoformat()
        
        try:
            # CSV logging
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                row = [
                    timestamp, packet.sequence_number, packet.ecg_quality,
                    packet.ppg_quality, packet.battery_voltage,
                    packet.battery_percentage, packet.temperature,
                    packet.system_status, packet.packet_duration_ms
                ]
                
                # Add ECG data
                for i in range(ECG_SAMPLES_PER_PACKET):
                    row.append(packet.ecg_values[i])
                    row.append(packet.leads_off_status[i])
                
                # Add PPG data
                for i in range(PPG_SAMPLES_PER_PACKET):
                    row.append(packet.ppg_red[i])
                    row.append(packet.ppg_ir[i])
                
                writer.writerow(row)
            
            # JSON data accumulation
            packet_dict = {
                'timestamp': timestamp,
                'sequence_number': packet.sequence_number,
                'timestamp_start_us': packet.timestamp_start_us,
                'timestamp_end_us': packet.timestamp_end_us,
                'ecg_values': packet.ecg_values,
                'leads_off_status': packet.leads_off_status,
                'ppg_red': packet.ppg_red,
                'ppg_ir': packet.ppg_ir,
                'battery_voltage': packet.battery_voltage,
                'battery_percentage': packet.battery_percentage,
                'temperature': packet.temperature,
                'ecg_quality': packet.ecg_quality,
                'ppg_quality': packet.ppg_quality,
                'system_status': packet.system_status,
                'packet_duration_ms': packet.packet_duration_ms
            }
            
            self.json_data.append(packet_dict)
            
        except Exception as e:
            logger.error(f"Data logging error: {e}")
    
    def save_json(self):
        """Save accumulated JSON data to file."""
        try:
            with open(self.json_filename, 'w') as jsonfile:
                json.dump(self.json_data, jsonfile, indent=2)
            logger.info(f"JSON data saved: {self.json_filename}")
        except Exception as e:
            logger.error(f"JSON save error: {e}")

class RealTimePlotter:
    """Real-time visualization of ECG and PPG signals."""
    
    def __init__(self):
        """Initialize real-time plotter with multiple subplots."""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('NirogScan Real-Time Health Monitor', fontsize=14, fontweight='bold')
        
        # Data buffers
        self.ecg_buffer = deque(maxlen=ECG_BUFFER_SIZE)
        self.ppg_red_buffer = deque(maxlen=PPG_BUFFER_SIZE)
        self.ppg_ir_buffer = deque(maxlen=PPG_BUFFER_SIZE)
        
        # Time buffers
        self.ecg_time_buffer = deque(maxlen=ECG_BUFFER_SIZE)
        self.ppg_time_buffer = deque(maxlen=PPG_BUFFER_SIZE)
        
        # Line objects
        self.ecg_line, = self.axes[0].plot([], [], 'b-', linewidth=1, label='ECG')
        self.ppg_red_line, = self.axes[1].plot([], [], 'r-', linewidth=1, label='PPG Red')
        self.ppg_ir_line, = self.axes[2].plot([], [], 'darkred', linewidth=1, label='PPG IR')
        
        # Setup axes
        self._setup_axes()
        
        # Status text
        self.status_text = self.fig.text(0.02, 0.95, '', fontsize=10, 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Animation
        self.animation = animation.FuncAnimation(
            self.fig, self._update_plots, interval=50, blit=False
        )
        
        # Current time reference
        self.start_time = time.time()
        
        # Latest packet for display
        self.latest_packet: Optional[HealthDataPacket] = None
        
        plt.tight_layout()
        
    def _setup_axes(self):
        """Configure plot axes properties."""
        # ECG plot
        self.axes[0].set_title('ECG Signal (125 Hz)', fontweight='bold')
        self.axes[0].set_ylabel('ADC Value')
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].legend()
        self.axes[0].set_ylim(0, 4095)
        
        # PPG Red plot
        self.axes[1].set_title('PPG Red Channel (50 Hz)', fontweight='bold')
        self.axes[1].set_ylabel('ADC Value')
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].legend()
        self.axes[1].set_ylim(0, 262143)
        
        # PPG IR plot
        self.axes[2].set_title('PPG IR Channel (50 Hz)', fontweight='bold')
        self.axes[2].set_ylabel('ADC Value')
        self.axes[2].set_xlabel('Time (seconds)')
        self.axes[2].grid(True, alpha=0.3)
        self.axes[2].legend()
        self.axes[2].set_ylim(0, 262143)
        
        # Set time range for all plots
        for ax in self.axes:
            ax.set_xlim(0, PLOT_WINDOW_SECONDS)
    
    def add_data(self, packet: HealthDataPacket):
        """Add new data packet to plotting buffers."""
        self.latest_packet = packet
        current_time = time.time() - self.start_time
        
        # Add ECG data
        for i, ecg_val in enumerate(packet.ecg_values):
            sample_time = current_time + (i / ECG_SAMPLE_RATE)
            self.ecg_buffer.append(ecg_val)
            self.ecg_time_buffer.append(sample_time)
        
        # Add PPG data
        for i, (red_val, ir_val) in enumerate(zip(packet.ppg_red, packet.ppg_ir)):
            sample_time = current_time + (i / PPG_SAMPLE_RATE)
            self.ppg_red_buffer.append(red_val)
            self.ppg_ir_buffer.append(ir_val)
            self.ppg_time_buffer.append(sample_time)
    
    def _update_plots(self, frame):
        """Update plot data and axes (called by animation)."""
        current_time = time.time() - self.start_time
        
        # Update time windows
        time_start = max(0, current_time - PLOT_WINDOW_SECONDS)
        time_end = current_time
        
        # Update ECG plot
        if self.ecg_buffer and self.ecg_time_buffer:
            ecg_times = np.array(self.ecg_time_buffer)
            ecg_values = np.array(self.ecg_buffer)
            
            # Filter data within time window
            mask = (ecg_times >= time_start) & (ecg_times <= time_end)
            if np.any(mask):
                self.ecg_line.set_data(ecg_times[mask], ecg_values[mask])
        
        # Update PPG plots
        if self.ppg_red_buffer and self.ppg_ir_buffer and self.ppg_time_buffer:
            ppg_times = np.array(self.ppg_time_buffer)
            ppg_red_values = np.array(self.ppg_red_buffer)
            ppg_ir_values = np.array(self.ppg_ir_buffer)
            
            # Filter data within time window
            mask = (ppg_times >= time_start) & (ppg_times <= time_end)
            if np.any(mask):
                self.ppg_red_line.set_data(ppg_times[mask], ppg_red_values[mask])
                self.ppg_ir_line.set_data(ppg_times[mask], ppg_ir_values[mask])
        
        # Update time axes
        for ax in self.axes:
            ax.set_xlim(time_start, time_end)
        
        # Update status text
        if self.latest_packet:
            status_text = (
                f"Seq: {self.latest_packet.sequence_number} | "
                f"ECG Quality: {self.latest_packet.ecg_quality}% | "
                f"PPG Quality: {self.latest_packet.ppg_quality}% | "
                f"Battery: {self.latest_packet.battery_voltage:.2f}V ({self.latest_packet.battery_percentage:.1f}%) | "
                f"Temp: {self.latest_packet.temperature:.1f}°C"
            )
            
            # Color code based on quality
            if self.latest_packet.ecg_quality < 50 or self.latest_packet.ppg_quality < 50:
                color = "lightcoral"
            elif self.latest_packet.ecg_quality < 80 or self.latest_packet.ppg_quality < 80:
                color = "lightyellow"
            else:
                color = "lightgreen"
                
            self.status_text.set_text(status_text)
            self.status_text.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=color))
        
        return [self.ecg_line, self.ppg_red_line, self.ppg_ir_line]

class NirogScanClient:
    """Main BLE client for NirogScan health monitoring device."""
    
    def __init__(self, enable_plotting: bool = True, enable_logging: bool = True):
        """Initialize NirogScan BLE client."""
        self.client: Optional[BleakClient] = None
        self.device_address: Optional[str] = None
        self.connected = False
        self.running = False
        
        # Data handling
        self.packet_count = 0
        self.error_count = 0
        self.last_sequence = 0
        
        # Optional components
        self.plotter = RealTimePlotter() if enable_plotting else None
        self.logger_obj = DataLogger("nirog_scan_data") if enable_logging else None
        
        # Statistics
        self.start_time = time.time()
        self.last_stats_time = time.time()
    
    async def scan_devices(self, timeout: float = 10.0) -> Optional[str]:
        """Scan for NirogScan BLE devices."""
        logger.info(f"Scanning for {DEVICE_NAME} devices...")
        
        try:
            devices = await BleakScanner.discover(timeout=timeout)
            
            for device in devices:
                if device.name and DEVICE_NAME in device.name:
                    logger.info(f"Found {DEVICE_NAME} device: {device.address}")
                    return device.address
                    
            logger.warning(f"No {DEVICE_NAME} devices found")
            return None
            
        except Exception as e:
            logger.error(f"Device scanning error: {e}")
            return None
    
    async def connect(self, device_address: Optional[str] = None) -> bool:
        """Connect to NirogScan BLE device."""
        try:
            # Auto-discover if no address provided
            if device_address is None:
                device_address = await self.scan_devices()
                if device_address is None:
                    return False
            
            self.device_address = device_address
            self.client = BleakClient(device_address)
            
            logger.info(f"Connecting to {device_address}...")
            await self.client.connect()
            
            if not self.client.is_connected:
                logger.error("Failed to connect to device")
                return False
            
            logger.info("Connected successfully")
            
            # Start notifications
            await self.client.start_notify(DATA_CHARACTERISTIC_UUID, self._notification_handler)
            logger.info("Notifications enabled")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from BLE device."""
        try:
            if self.client and self.client.is_connected:
                await self.client.stop_notify(DATA_CHARACTERISTIC_UUID)
                await self.client.disconnect()
                logger.info("Disconnected from device")
                
            self.connected = False
            
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
    
    def _notification_handler(self, sender: BleakGATTCharacteristic, data: bytearray):
        """Handle incoming BLE notifications with health data."""
        try:
            # Parse packet
            packet = DataParser.parse_packet(bytes(data))
            if packet is None:
                self.error_count += 1
                return
            
            # Check for missing packets
            expected_sequence = (self.last_sequence + 1) % 65536
            if self.packet_count > 0 and packet.sequence_number != expected_sequence:
                missed = (packet.sequence_number - expected_sequence) % 65536
                logger.warning(f"Missed {missed} packets (got {packet.sequence_number}, expected {expected_sequence})")
            
            self.last_sequence = packet.sequence_number
            self.packet_count += 1
            
            # Log data
            if self.logger_obj:
                self.logger_obj.log_packet(packet)
            
            # Update plots
            if self.plotter:
                self.plotter.add_data(packet)
            
            # Print periodic stats
            current_time = time.time()
            if current_time - self.last_stats_time >= 5.0:
                self._print_stats(packet)
                self.last_stats_time = current_time
                
        except Exception as e:
            logger.error(f"Notification handling error: {e}")
            self.error_count += 1
    
    def _print_stats(self, packet: HealthDataPacket):
        """Print periodic statistics."""
        runtime = time.time() - self.start_time
        packet_rate = self.packet_count / runtime if runtime > 0 else 0
        error_rate = (self.error_count / (self.packet_count + self.error_count)) * 100 if (self.packet_count + self.error_count) > 0 else 0
        
        print(f"\n=== NirogScan Statistics (Runtime: {runtime:.1f}s) ===")
        print(f"Packets: {self.packet_count} | Rate: {packet_rate:.1f} Hz | Errors: {error_rate:.1f}%")
        print(f"Sequence: {packet.sequence_number} | Duration: {packet.packet_duration_ms:.1f}ms")
        print(f"ECG Quality: {packet.ecg_quality}% | PPG Quality: {packet.ppg_quality}%")
        print(f"Battery: {packet.battery_voltage:.2f}V ({packet.battery_percentage:.1f}%)")
        print(f"Temperature: {packet.temperature:.1f}°C")
        print(f"Leads Connected: {'Yes' if packet.is_ecg_leads_connected else 'No'}")
        print(f"PPG Signal Good: {'Yes' if packet.is_ppg_signal_good else 'No'}")
    
    async def run(self):
        """Main execution loop."""
        self.running = True
        
        try:
            while self.running and self.connected:
                await asyncio.sleep(1.0)
                
                # Check connection status
                if self.client and not self.client.is_connected:
                    logger.warning("Connection lost")
                    self.connected = False
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            await self.disconnect()
            
            # Save final JSON data
            if self.logger_obj:
                self.logger_obj.save_json()
    
    def stop(self):
        """Stop the client."""
        self.running = False

async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='NirogScan Health Monitor - Python BLE Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nirog_scan_receiver.py                    # Auto-discover and connect
  python nirog_scan_receiver.py -a AA:BB:CC:DD:EE:FF  # Connect to specific device
  python nirog_scan_receiver.py --no-plot          # Disable real-time plotting
  python nirog_scan_receiver.py --no-log           # Disable data logging
        """
    )
    
    parser.add_argument('-a', '--address', type=str, help='BLE device MAC address')
    parser.add_argument('--no-plot', action='store_true', help='Disable real-time plotting')
    parser.add_argument('--no-log', action='store_true', help='Disable data logging')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize client
    client = NirogScanClient(
        enable_plotting=not args.no_plot,
        enable_logging=not args.no_log
    )
    
    try:
        # Connect to device
        if not await client.connect(args.address):
            logger.error("Failed to connect to NirogScan device")
            return 1
        
        logger.info("Starting data collection...")
        logger.info("Press Ctrl+C to stop")
        
        # Start plotting in separate thread if enabled
        if client.plotter:
            plot_thread = threading.Thread(target=plt.show, daemon=True)
            plot_thread.start()
        
        # Run client
        await client.run()
        
        logger.info("Data collection completed")
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    finally:
        client.stop()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)