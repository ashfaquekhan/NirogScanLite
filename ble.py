import sys
import asyncio
import struct
from datetime import datetime
from collections import deque

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QComboBox, 
                                 QGroupBox, QGridLayout, QStatusBar)
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
    from PyQt5.QtGui import QFont, QPalette, QColor
except ImportError:
    print("Error: PyQt5 not installed")
    print("Install with: pip install PyQt5")
    sys.exit(1)

try:
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, useOpenGL=True)
except ImportError:
    print("Error: pyqtgraph not installed")
    print("Install with: pip install pyqtgraph")
    sys.exit(1)

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    print("Error: bleak not installed")
    print("Install with: pip install bleak")
    sys.exit(1)

import qasync
from qasync import QEventLoop, asyncSlot

DEVICE_NAME = "NirogScan"
SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

ECG_SAMPLES_PER_PACKET = 10
PPG_SAMPLES_PER_PACKET = 4
PLOT_WINDOW = 1000

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

class DataSignals(QObject):
    new_packet = pyqtSignal(object)
    connection_changed = pyqtSignal(bool)
    status_message = pyqtSignal(str)

class BLEManager:
    def __init__(self, signals):
        self.signals = signals
        self.client = None
        self.device = None
        self.connected = False
        self.packet_count = 0
        
    async def scan_devices(self):
        self.signals.status_message.emit("Scanning for devices...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        nirog_devices = []
        for device in devices:
            if device.name and DEVICE_NAME in device.name:
                nirog_devices.append(device)
        
        return nirog_devices
    
    def notification_handler(self, sender, data):
        try:
            packet = HealthDataPacket(data)
            self.packet_count += 1
            self.signals.new_packet.emit(packet)
        except Exception as e:
            self.signals.status_message.emit(f"Packet parse error: {e}")
    
    async def connect(self, device):
        try:
            self.device = device
            self.signals.status_message.emit(f"Connecting to {device.name}...")
            
            self.client = BleakClient(device.address)
            await self.client.connect()
            
            if self.client.is_connected:
                await self.client.start_notify(CHAR_UUID, self.notification_handler)
                self.connected = True
                self.signals.connection_changed.emit(True)
                self.signals.status_message.emit(f"Connected to {device.name}")
                return True
            
        except Exception as e:
            self.signals.status_message.emit(f"Connection failed: {e}")
            self.connected = False
            self.signals.connection_changed.emit(False)
            return False
    
    async def disconnect(self):
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(CHAR_UUID)
                await self.client.disconnect()
            except:
                pass
        
        self.connected = False
        self.signals.connection_changed.emit(False)
        self.signals.status_message.emit("Disconnected")

class NirogScanGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NirogScan - Real-time Health Monitor")
        self.setGeometry(100, 100, 1400, 900)
        
        self.signals = DataSignals()
        self.ble_manager = BLEManager(self.signals)
        
        self.ecg_data = deque(maxlen=PLOT_WINDOW)
        self.ppg_red_data = deque(maxlen=PLOT_WINDOW)
        self.ppg_ir_data = deque(maxlen=PLOT_WINDOW)
        self.time_data = deque(maxlen=PLOT_WINDOW)
        
        self.packet_count = 0
        self.start_time = None
        
        self.setup_ui()
        self.connect_signals()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(50)
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        status_panel = self.create_status_panel()
        main_layout.addWidget(status_panel)
        
        plots_layout = self.create_plots()
        main_layout.addLayout(plots_layout)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666666;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
        """)
    
    def create_control_panel(self):
        group = QGroupBox("Connection Control")
        layout = QHBoxLayout()
        
        self.scan_button = QPushButton("Scan for Devices")
        self.scan_button.clicked.connect(self.on_scan_clicked)
        layout.addWidget(self.scan_button)
        
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(200)
        layout.addWidget(self.device_combo)
        
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.on_connect_clicked)
        self.connect_button.setEnabled(False)
        layout.addWidget(self.connect_button)
        
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.on_disconnect_clicked)
        self.disconnect_button.setEnabled(False)
        layout.addWidget(self.disconnect_button)
        
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_status_panel(self):
        group = QGroupBox("System Status")
        layout = QGridLayout()
        
        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setBold(True)
        
        value_font = QFont()
        value_font.setPointSize(11)
        
        row = 0
        
        layout.addWidget(QLabel("ECG Quality:"), row, 0)
        self.ecg_quality_label = QLabel("---%")
        self.ecg_quality_label.setFont(value_font)
        layout.addWidget(self.ecg_quality_label, row, 1)
        
        layout.addWidget(QLabel("PPG Quality:"), row, 2)
        self.ppg_quality_label = QLabel("---%")
        self.ppg_quality_label.setFont(value_font)
        layout.addWidget(self.ppg_quality_label, row, 3)
        
        layout.addWidget(QLabel("Leads Off:"), row, 4)
        self.leads_off_label = QLabel("Unknown")
        self.leads_off_label.setFont(value_font)
        layout.addWidget(self.leads_off_label, row, 5)
        
        row += 1
        
        layout.addWidget(QLabel("Battery:"), row, 0)
        self.battery_label = QLabel("-.--V (---%)")
        self.battery_label.setFont(value_font)
        layout.addWidget(self.battery_label, row, 1)
        
        layout.addWidget(QLabel("Temperature:"), row, 2)
        self.temp_label = QLabel("--.-°C")
        self.temp_label.setFont(value_font)
        layout.addWidget(self.temp_label, row, 3)
        
        layout.addWidget(QLabel("Packets:"), row, 4)
        self.packet_label = QLabel("0")
        self.packet_label.setFont(value_font)
        layout.addWidget(self.packet_label, row, 5)
        
        layout.setColumnStretch(6, 1)
        
        group.setLayout(layout)
        return group
    
    def create_plots(self):
        layout = QVBoxLayout()
        
        self.ecg_plot = pg.PlotWidget(title="ECG Signal (125 Hz)")
        self.ecg_plot.setBackground('#1e1e1e')
        self.ecg_plot.setLabel('left', 'Amplitude', units='ADC')
        self.ecg_plot.setLabel('bottom', 'Sample')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen(color='#00ff00', width=2))
        layout.addWidget(self.ecg_plot)
        
        self.ppg_red_plot = pg.PlotWidget(title="PPG Red Channel (50 Hz)")
        self.ppg_red_plot.setBackground('#1e1e1e')
        self.ppg_red_plot.setLabel('left', 'Amplitude', units='counts')
        self.ppg_red_plot.setLabel('bottom', 'Sample')
        self.ppg_red_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_red_curve = self.ppg_red_plot.plot(pen=pg.mkPen(color='#ff0000', width=2))
        layout.addWidget(self.ppg_red_plot)
        
        self.ppg_ir_plot = pg.PlotWidget(title="PPG IR Channel (50 Hz)")
        self.ppg_ir_plot.setBackground('#1e1e1e')
        self.ppg_ir_plot.setLabel('left', 'Amplitude', units='counts')
        self.ppg_ir_plot.setLabel('bottom', 'Sample')
        self.ppg_ir_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_ir_curve = self.ppg_ir_plot.plot(pen=pg.mkPen(color='#ff00ff', width=2))
        layout.addWidget(self.ppg_ir_plot)
        
        return layout
    
    def connect_signals(self):
        self.signals.new_packet.connect(self.on_new_packet)
        self.signals.connection_changed.connect(self.on_connection_changed)
        self.signals.status_message.connect(self.on_status_message)
    
    @asyncSlot()
    async def on_scan_clicked(self):
        self.scan_button.setEnabled(False)
        self.device_combo.clear()
        
        try:
            devices = await self.ble_manager.scan_devices()
            
            if devices:
                for device in devices:
                    self.device_combo.addItem(f"{device.name} ({device.address})", device)
                self.connect_button.setEnabled(True)
                self.signals.status_message.emit(f"Found {len(devices)} device(s)")
            else:
                self.signals.status_message.emit("No devices found")
                self.connect_button.setEnabled(False)
        
        except Exception as e:
            self.signals.status_message.emit(f"Scan error: {e}")
        
        finally:
            self.scan_button.setEnabled(True)
    
    @asyncSlot()
    async def on_connect_clicked(self):
        if self.device_combo.currentIndex() >= 0:
            device = self.device_combo.currentData()
            self.connect_button.setEnabled(False)
            
            success = await self.ble_manager.connect(device)
            
            if not success:
                self.connect_button.setEnabled(True)
            else:
                self.start_time = datetime.now()
                self.packet_count = 0
                self.ecg_data.clear()
                self.ppg_red_data.clear()
                self.ppg_ir_data.clear()
                self.time_data.clear()
    
    @asyncSlot()
    async def on_disconnect_clicked(self):
        self.disconnect_button.setEnabled(False)
        await self.ble_manager.disconnect()
    
    def on_connection_changed(self, connected):
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.scan_button.setEnabled(not connected)
        self.device_combo.setEnabled(not connected)
        
        if not connected:
            self.ecg_quality_label.setText("---%")
            self.ppg_quality_label.setText("---%")
            self.leads_off_label.setText("Unknown")
            self.battery_label.setText("-.--V (---%)")
            self.temp_label.setText("--.-°C")
    
    def on_status_message(self, message):
        self.statusBar.showMessage(message)
    
    def on_new_packet(self, packet):
        self.packet_count += 1
        
        for ecg_val in packet.ecg_values:
            self.ecg_data.append(ecg_val)
        
        for red_val in packet.ppg_red:
            self.ppg_red_data.append(red_val)
        
        for ir_val in packet.ppg_ir:
            self.ppg_ir_data.append(ir_val)
        
        self.ecg_quality_label.setText(f"{packet.ecg_quality}%")
        self.set_quality_color(self.ecg_quality_label, packet.ecg_quality)
        
        self.ppg_quality_label.setText(f"{packet.ppg_quality}%")
        self.set_quality_color(self.ppg_quality_label, packet.ppg_quality)
        
        leads_off = packet.leads_off_status[0]
        if leads_off == 0:
            self.leads_off_label.setText("✓ Connected")
            self.leads_off_label.setStyleSheet("color: #00ff00;")
        elif leads_off == 0xFF:
            self.leads_off_label.setText("? Unknown")
            self.leads_off_label.setStyleSheet("color: #ffaa00;")
        else:
            status_text = []
            if leads_off & 0x01:
                status_text.append("LO+")
            if leads_off & 0x02:
                status_text.append("LO-")
            self.leads_off_label.setText(f"✗ {' '.join(status_text)}")
            self.leads_off_label.setStyleSheet("color: #ff0000;")
        
        self.battery_label.setText(f"{packet.battery_voltage:.2f}V ({packet.battery_percentage:.0f}%)")
        if packet.battery_percentage < 20:
            self.battery_label.setStyleSheet("color: #ff0000;")
        elif packet.battery_percentage < 50:
            self.battery_label.setStyleSheet("color: #ffaa00;")
        else:
            self.battery_label.setStyleSheet("color: #00ff00;")
        
        self.temp_label.setText(f"{packet.temperature:.1f}°C")
        
        self.packet_label.setText(str(self.packet_count))
    
    def set_quality_color(self, label, quality):
        if quality >= 80:
            label.setStyleSheet("color: #00ff00;")
        elif quality >= 50:
            label.setStyleSheet("color: #ffaa00;")
        else:
            label.setStyleSheet("color: #ff0000;")
    
    def update_plots(self):
        if len(self.ecg_data) > 0:
            self.ecg_curve.setData(list(self.ecg_data))
        
        if len(self.ppg_red_data) > 0:
            self.ppg_red_curve.setData(list(self.ppg_red_data))
        
        if len(self.ppg_ir_data) > 0:
            self.ppg_ir_curve.setData(list(self.ppg_ir_data))
    
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