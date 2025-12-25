import asyncio
import struct
import sys
from datetime import datetime

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    print("Error: bleak library not installed")
    print("Install with: pip install bleak")
    sys.exit(1)

DEVICE_NAME = "NirogScan"
SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

ECG_SAMPLES_PER_PACKET = 10
PPG_SAMPLES_PER_PACKET = 4

class HealthDataPacket:
    def __init__(self, data):
        if len(data) < 89:
            raise ValueError(f"Packet too short: {len(data)} bytes")
        
        offset = 0
        
        self.sequence_number = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        
        self.timestamp_start_us = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        self.timestamp_end_us = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        self.ecg_values = list(struct.unpack('<' + 'h' * ECG_SAMPLES_PER_PACKET, 
                                            data[offset:offset+ECG_SAMPLES_PER_PACKET*2]))
        offset += ECG_SAMPLES_PER_PACKET * 2
        
        self.leads_off_status = list(struct.unpack('<' + 'B' * ECG_SAMPLES_PER_PACKET,
                                                   data[offset:offset+ECG_SAMPLES_PER_PACKET]))
        offset += ECG_SAMPLES_PER_PACKET
        
        self.ppg_red = list(struct.unpack('<' + 'I' * PPG_SAMPLES_PER_PACKET,
                                         data[offset:offset+PPG_SAMPLES_PER_PACKET*4]))
        offset += PPG_SAMPLES_PER_PACKET * 4
        
        self.ppg_ir = list(struct.unpack('<' + 'I' * PPG_SAMPLES_PER_PACKET,
                                        data[offset:offset+PPG_SAMPLES_PER_PACKET*4]))
        offset += PPG_SAMPLES_PER_PACKET * 4
        
        self.battery_voltage = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        self.battery_percentage = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        self.temperature = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        self.ecg_quality = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        
        self.ppg_quality = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        
        self.system_status = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        
        self.checksum = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        
    def __str__(self):
        return (f"Packet #{self.sequence_number:04d} | "
                f"ECG: {self.ecg_quality}% | "
                f"PPG: {self.ppg_quality}% | "
                f"Batt: {self.battery_voltage:.2f}V ({self.battery_percentage:.0f}%) | "
                f"Temp: {self.temperature:.1f}°C")
    
    def to_dict(self):
        return {
            'sequence': self.sequence_number,
            'timestamp_start': self.timestamp_start_us,
            'timestamp_end': self.timestamp_end_us,
            'ecg_values': self.ecg_values,
            'ecg_quality': self.ecg_quality,
            'ppg_red': self.ppg_red,
            'ppg_ir': self.ppg_ir,
            'ppg_quality': self.ppg_quality,
            'battery_voltage': self.battery_voltage,
            'battery_percentage': self.battery_percentage,
            'temperature': self.temperature,
            'system_status': self.system_status
        }

class NirogScanClient:
    def __init__(self):
        self.client = None
        self.device = None
        self.packet_count = 0
        self.last_sequence = None
        
    async def scan_for_device(self, timeout=10.0):
        print(f"Scanning for {DEVICE_NAME}...")
        devices = await BleakScanner.discover(timeout=timeout)
        
        for device in devices:
            if device.name and DEVICE_NAME in device.name:
                self.device = device
                print(f"Found device: {device.name} ({device.address})")
                return True
        
        print(f"Device {DEVICE_NAME} not found")
        return False
    
    def notification_handler(self, sender, data):
        try:
            packet = HealthDataPacket(data)
            
            if self.last_sequence is not None:
                expected = (self.last_sequence + 1) & 0xFFFF
                if packet.sequence_number != expected:
                    print(f"⚠ Packet loss detected! Expected {expected}, got {packet.sequence_number}")
            
            self.last_sequence = packet.sequence_number
            self.packet_count += 1
            
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {packet}")
            
            if self.packet_count % 10 == 0:
                print(f"\n--- Sample ECG values: {packet.ecg_values[:5]} ---")
                print(f"--- Sample PPG (Red): {packet.ppg_red[:2]} ---")
                print(f"--- Sample PPG (IR):  {packet.ppg_ir[:2]} ---\n")
                
        except Exception as e:
            print(f"Error parsing packet: {e}")
    
    async def connect_and_stream(self):
        if not self.device:
            if not await self.scan_for_device():
                return False
        
        print(f"\nConnecting to {self.device.name}...")
        
        async with BleakClient(self.device.address) as client:
            self.client = client
            print(f"Connected: {client.is_connected}")
            
            print("\nServices:")
            for service in client.services:
                print(f"  [{service.uuid}] {service.description}")
                for char in service.characteristics:
                    print(f"    [{char.uuid}] {','.join(char.properties)}")
            
            print(f"\nEnabling notifications on characteristic {CHAR_UUID}...")
            await client.start_notify(CHAR_UUID, self.notification_handler)
            print("Notifications enabled. Receiving data...\n")
            
            try:
                while client.is_connected:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n\nStopping...")
            finally:
                await client.stop_notify(CHAR_UUID)
                print(f"\nTotal packets received: {self.packet_count}")
        
        return True

async def main():
    print("=" * 70)
    print("NirogScan BLE Client")
    print("=" * 70)
    print()
    
    client = NirogScanClient()
    
    try:
        await client.connect_and_stream()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)