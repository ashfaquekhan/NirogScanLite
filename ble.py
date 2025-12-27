import sys
import asyncio
import struct
from collections import deque
from threading import Lock

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont

import pyqtgraph as pg
pg.setConfigOptions(antialias=True)

from bleak import BleakClient, BleakScanner
import qasync
from qasync import QEventLoop, asyncSlot

import numpy as np

DEVICE_NAME = "NirogScan"
SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

PLOT_WINDOW = 500
UPDATE_RATE_MS = 50
PACKET_SIZE = 47


def crc16_modbus(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class HealthPacket:
    __slots__ = ['timestamp', 'seq', 'ecg', 'leads', 'red', 'ir', 
                 'battery_v', 'battery_pct', 'temp', 'crc', 'crc_valid']
    
    def __init__(self, data):
        if len(data) < PACKET_SIZE:
            raise ValueError(f"Packet too short: {len(data)} bytes, expected {PACKET_SIZE}")
        
        computed_crc = crc16_modbus(data[:-2])
        received_crc = struct.unpack('<H', data[-2:])[0]
        self.crc_valid = (computed_crc == received_crc)
        
        offset = 0
        self.timestamp = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        self.seq = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2
        
        self.ecg = list(struct.unpack('<hhhhh', data[offset:offset+10]))
        offset += 10
        
        self.leads = struct.unpack('<B', data[offset:offset+1])[0]
        offset += 1
        
        self.red = list(struct.unpack('<II', data[offset:offset+8]))
        offset += 8
        
        self.ir = list(struct.unpack('<II', data[offset:offset+8]))
        offset += 8
        
        self.battery_v = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        self.battery_pct = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        self.temp = struct.unpack('<f', data[offset:offset+4])[0]
        offset += 4
        
        self.crc = received_crc


class DataSignals(QObject):
    new_packet = pyqtSignal(object)
    connection_changed = pyqtSignal(bool)
    status_message = pyqtSignal(str)


class BLEManager:
    def __init__(self, signals):
        self.signals = signals
        self.client = None
        self.connected = False
        self.packet_count = 0
        self.last_seq = -1
        self.dropped = 0
        self.crc_errors = 0
        
    async def scan_devices(self):
        self.signals.status_message.emit("Scanning...")
        devices = await BleakScanner.discover(timeout=10.0)
        return [d for d in devices if d.name and DEVICE_NAME in d.name]
    
    def notification_handler(self, sender, data):
        try:
            packet = HealthPacket(data)
            
            if not packet.crc_valid:
                self.crc_errors += 1
                print(f"[WARN] CRC error #{self.crc_errors} (seq={packet.seq})")
                return
            
            if self.last_seq >= 0:
                expected = (self.last_seq + 1) & 0xFFFF
                if packet.seq != expected:
                    dropped = (packet.seq - expected) & 0xFFFF
                    if dropped < 1000:
                        self.dropped += dropped
                        print(f"[WARN] Dropped {dropped} packets (seq {expected} -> {packet.seq})")
            
            self.last_seq = packet.seq
            self.packet_count += 1
            
            if self.packet_count % 250 == 0:
                print(f"[INFO] Packets: {self.packet_count}, Dropped: {self.dropped}, CRC Errors: {self.crc_errors}")
            
            self.signals.new_packet.emit(packet)
        except Exception as e:
            print(f"[ERROR] Parse error: {e}, len={len(data)}")
    
    async def connect(self, device):
        try:
            self.signals.status_message.emit("Connecting...")
            self.client = BleakClient(device.address)
            await self.client.connect()
            
            if self.client.is_connected:
                try:
                    await self.client.request_mtu(185)
                except Exception:
                    pass
                
                await self.client.start_notify(CHAR_UUID, self.notification_handler)
                self.connected = True
                self.packet_count = 0
                self.last_seq = -1
                self.dropped = 0
                self.crc_errors = 0
                self.signals.connection_changed.emit(True)
                self.signals.status_message.emit("Connected")
                return True
        except Exception as e:
            self.signals.status_message.emit(f"Failed: {e}")
            self.connected = False
            self.signals.connection_changed.emit(False)
        return False
    
    async def disconnect(self):
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(CHAR_UUID)
                await self.client.disconnect()
            except Exception:
                pass
        self.connected = False
        self.signals.connection_changed.emit(False)
        self.signals.status_message.emit("Disconnected")


class CircularBuffer:
    def __init__(self, size, dtype=np.float32):
        self.size = size
        self.data = np.zeros(size, dtype=dtype)
        self.write_idx = 0
        self.count = 0
        self.lock = Lock()
    
    def append(self, value):
        with self.lock:
            self.data[self.write_idx] = value
            self.write_idx = (self.write_idx + 1) % self.size
            if self.count < self.size:
                self.count += 1
    
    def extend(self, values):
        with self.lock:
            for v in values:
                self.data[self.write_idx] = v
                self.write_idx = (self.write_idx + 1) % self.size
            self.count = min(self.count + len(values), self.size)
    
    def get_ordered(self):
        with self.lock:
            if self.count < self.size:
                return self.data[:self.count].copy()
            else:
                return np.concatenate([
                    self.data[self.write_idx:],
                    self.data[:self.write_idx]
                ])
    
    def clear(self):
        with self.lock:
            self.data.fill(0)
            self.write_idx = 0
            self.count = 0


class NirogScanGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NirogScan v2.3")
        self.setGeometry(100, 100, 1400, 900)
        
        self.signals = DataSignals()
        self.ble_manager = BLEManager(self.signals)
        
        self.ecg_buffer = CircularBuffer(PLOT_WINDOW, dtype=np.int16)
        self.ppg_red_buffer = CircularBuffer(PLOT_WINDOW, dtype=np.uint32)
        self.ppg_ir_buffer = CircularBuffer(PLOT_WINDOW, dtype=np.uint32)
        
        self.packet_count = 0
        self.last_packet = None
        self.data_lock = Lock()
        
        self.setup_ui()
        self.connect_signals()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(UPDATE_RATE_MS)
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        ctrl_group = QGroupBox("Connection")
        ctrl_layout = QHBoxLayout()
        
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.on_scan_clicked)
        ctrl_layout.addWidget(self.scan_btn)
        
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        ctrl_layout.addWidget(self.device_combo)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect_clicked)
        self.connect_btn.setEnabled(False)
        ctrl_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.on_disconnect_clicked)
        self.disconnect_btn.setEnabled(False)
        ctrl_layout.addWidget(self.disconnect_btn)
        
        ctrl_layout.addStretch()
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)
        
        status_group = QGroupBox("Status")
        status_layout = QGridLayout()
        
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        
        labels = [
            ("Packets:", "packet_label", "0"),
            ("Dropped:", "dropped_label", "0"),
            ("CRC Err:", "crc_label", "0"),
            ("Battery:", "battery_label", "-.--V"),
            ("Temp:", "temp_label", "--.-°C"),
            ("Leads:", "leads_label", "--"),
            ("Seq:", "seq_label", "0"),
        ]
        
        for col, (text, attr, default) in enumerate(labels):
            status_layout.addWidget(QLabel(text), 0, col * 2)
            label = QLabel(default)
            label.setFont(font)
            setattr(self, attr, label)
            status_layout.addWidget(label, 0, col * 2 + 1)
        
        status_layout.setColumnStretch(len(labels) * 2, 1)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        self.ecg_plot = pg.PlotWidget(title="ECG (125 Hz)")
        self.ecg_plot.setBackground('#1e1e1e')
        self.ecg_plot.setLabel('left', 'ADC')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.setDownsampling(mode='peak')
        self.ecg_plot.setClipToView(True)
        self.ecg_plot.setRange(xRange=[0, PLOT_WINDOW], yRange=[0, 4096])
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen(color='#00ff00', width=1))
        layout.addWidget(self.ecg_plot)
        
        self.ppg_red_plot = pg.PlotWidget(title="PPG Red (25 Hz)")
        self.ppg_red_plot.setBackground('#1e1e1e')
        self.ppg_red_plot.setLabel('left', 'Counts')
        self.ppg_red_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_red_plot.setDownsampling(mode='peak')
        self.ppg_red_plot.setClipToView(True)
        self.ppg_red_curve = self.ppg_red_plot.plot(pen=pg.mkPen(color='#ff0000', width=1))
        layout.addWidget(self.ppg_red_plot)
        
        self.ppg_ir_plot = pg.PlotWidget(title="PPG IR (25 Hz)")
        self.ppg_ir_plot.setBackground('#1e1e1e')
        self.ppg_ir_plot.setLabel('left', 'Counts')
        self.ppg_ir_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_ir_plot.setDownsampling(mode='peak')
        self.ppg_ir_plot.setClipToView(True)
        self.ppg_ir_curve = self.ppg_ir_plot.plot(pen=pg.mkPen(color='#ff00ff', width=1))
        layout.addWidget(self.ppg_ir_plot)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ffffff;
            }
            QLabel { color: #ffffff; }
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover { background-color: #4d4d4d; }
            QPushButton:disabled { background-color: #2d2d2d; color: #666666; }
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
            }
        """)
    
    def connect_signals(self):
        self.signals.new_packet.connect(self.on_new_packet)
        self.signals.connection_changed.connect(self.on_connection_changed)
        self.signals.status_message.connect(self.on_status_message)
    
    @asyncSlot()
    async def on_scan_clicked(self):
        self.scan_btn.setEnabled(False)
        self.device_combo.clear()
        
        try:
            devices = await self.ble_manager.scan_devices()
            if devices:
                for device in devices:
                    self.device_combo.addItem(f"{device.name} ({device.address})", device)
                self.connect_btn.setEnabled(True)
                self.signals.status_message.emit(f"Found {len(devices)} device(s)")
            else:
                self.signals.status_message.emit("No devices found")
        except Exception as e:
            self.signals.status_message.emit(f"Scan error: {e}")
        finally:
            self.scan_btn.setEnabled(True)
    
    @asyncSlot()
    async def on_connect_clicked(self):
        if self.device_combo.currentIndex() >= 0:
            device = self.device_combo.currentData()
            self.connect_btn.setEnabled(False)
            success = await self.ble_manager.connect(device)
            if not success:
                self.connect_btn.setEnabled(True)
            else:
                self.packet_count = 0
                self.ecg_buffer.clear()
                self.ppg_red_buffer.clear()
                self.ppg_ir_buffer.clear()
    
    @asyncSlot()
    async def on_disconnect_clicked(self):
        self.disconnect_btn.setEnabled(False)
        await self.ble_manager.disconnect()
    
    def on_connection_changed(self, connected):
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.scan_btn.setEnabled(not connected)
        self.device_combo.setEnabled(not connected)
    
    def on_status_message(self, message):
        self.statusBar.showMessage(message)
    
    def on_new_packet(self, packet):
        self.packet_count += 1
        
        self.ecg_buffer.extend(packet.ecg)
        
        if packet.red[0] > 0:
            self.ppg_red_buffer.append(packet.red[0])
            self.ppg_ir_buffer.append(packet.ir[0])
        
        with self.data_lock:
            self.last_packet = packet
    
    def update_plots(self):
        ecg_data = self.ecg_buffer.get_ordered()
        ppg_red_data = self.ppg_red_buffer.get_ordered()
        ppg_ir_data = self.ppg_ir_buffer.get_ordered()
        
        self.ecg_curve.setData(ecg_data)
        self.ppg_red_curve.setData(ppg_red_data)
        self.ppg_ir_curve.setData(ppg_ir_data)
        
        with self.data_lock:
            packet = self.last_packet
        
        if packet:
            self.packet_label.setText(str(self.packet_count))
            self.dropped_label.setText(str(self.ble_manager.dropped))
            self.crc_label.setText(str(self.ble_manager.crc_errors))
            self.battery_label.setText(f"{packet.battery_v:.2f}V ({packet.battery_pct:.0f}%)")
            self.temp_label.setText(f"{packet.temp:.1f}°C")
            self.seq_label.setText(str(packet.seq))
            
            if packet.leads == 0:
                self.leads_label.setText("OK")
                self.leads_label.setStyleSheet("color: #55ff55; font-weight: bold;")
            else:
                status = []
                if packet.leads & 0x01:
                    status.append("LO+")
                if packet.leads & 0x02:
                    status.append("LO-")
                self.leads_label.setText(" ".join(status) if status else "ERR")
                self.leads_label.setStyleSheet("color: #ff5555; font-weight: bold;")
    
    def closeEvent(self, event):
        if self.ble_manager.connected:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.ble_manager.disconnect())
        event.accept()


def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = NirogScanGUI()
    window.show()
    with loop:
        loop.run_forever()


if __name__ == '__main__':
    main()