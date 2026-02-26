#!/usr/bin/env python3
"""
NirogScan v5.0 Fixed - Enhanced ECG/PPG Analysis
Features:
- ECG: Medical grid plots, PQRST detection & labeling, interval analysis
- PPG: Age Index, PAT, Respiration Rate, Baseline Shift
- Proper NeuroKit2 integration
- Fixed logging (no zeros)
"""

import sys, os, asyncio, struct, time, queue, warnings
from threading import Lock
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QStatusBar, QFileDialog,
    QDialog, QFormLayout, QLineEdit, QTextEdit, QMessageBox, QListWidget,
    QListWidgetItem
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, QThread
from PyQt5.QtGui import QFont, QIntValidator

import pyqtgraph as pg
pg.setConfigOptions(antialias=True)

from bleak import BleakClient, BleakScanner
import qasync
from qasync import QEventLoop, asyncSlot

from scipy.signal import butter, filtfilt, find_peaks as scipy_find_peaks, detrend, hilbert, welch
from scipy.stats import zscore

warnings.filterwarnings('ignore')

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
    print(f"[INIT] NeuroKit2 v{nk.__version__} loaded")
except:
    NEUROKIT_AVAILABLE = False
    nk = None
    print("[INIT] WARNING: NeuroKit2 not available")

# Constants
DEVICE_NAME = "NirogScan"
SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

PLOT_WINDOW = 500
UPDATE_RATE_MS = 50
PACKET_SIZE = 47

ADC_VREF_MV = 3300
ADC_MAX_VALUE = 4095
AD8232_GAIN = 1100

ECG_FS = 125
PPG_FS = 25

MIN_RPEAKS = 5

SPO2_LOOKUP = [
    95,95,95,96,96,96,97,97,97,97,97,98,98,98,98,98,
    99,99,99,99,99,99,99,99,100,100,100,100,100,100,100,100,
    100,100,100,100,100,100,100,100,100,100,100,100,99,99,99,99,
    99,99,99,99,98,98,98,98,98,98,97,97,97,97,96,96,
    96,96,95,95,95,94,94,94,93,93,93,92,92,92,91,91,
    90,90,89,89,89,88,88,87,87,86,86,85,85,84,84,83,
    82,82,81,81,80,80,79,78,78,77,76,76,75,74,74,73,
    72,72,71,70,69,69,68,67,66,66,65,64,63,62,62,61,
    60,59,58,57,56,56,55,54,53,52,51,50,49,48,47,46,
    45,44,43,42,41,40,39,38,37,36,35,34,33,31,30,29,
    28,27,26,25,23,22,21,20,19,17,16,15,14,12,11,10,
    9,7,6,5,3,2,1
]

DARK_STYLE = """
    QMainWindow, QWidget { background-color: #1e1e1e; color: #ffffff; }
    QGroupBox {
        border: 1px solid #3d3d3d; border-radius: 5px;
        margin-top: 10px; padding-top: 10px;
        font-weight: bold; color: #ffffff;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QLabel { color: #ffffff; }
    QPushButton {
        background-color: #3d3d3d; color: #ffffff;
        border: 1px solid #555555; border-radius: 4px;
        padding: 6px 12px; min-height: 20px;
    }
    QPushButton:hover { background-color: #4d4d4d; }
    QPushButton:pressed { background-color: #2d2d2d; }
    QPushButton:disabled { background-color: #2d2d2d; color: #666666; }
    QComboBox {
        background-color: #3d3d3d; color: #ffffff;
        border: 1px solid #555555; border-radius: 4px;
        padding: 5px; min-height: 20px;
    }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView {
        background-color: #3d3d3d; color: #ffffff;
        selection-background-color: #5d5d5d;
    }
    QLineEdit, QTextEdit {
        background-color: #3d3d3d; color: #ffffff;
        border: 1px solid #555555; border-radius: 4px; padding: 5px;
    }
    QListWidget {
        background-color: #3d3d3d; color: #ffffff;
        border: 1px solid #555555; border-radius: 4px;
    }
    QListWidget::item:selected { background-color: #5d8aa8; }
    QStatusBar { background-color: #2d2d2d; color: #888888; }
    QDialog { background-color: #1e1e1e; }
"""

def adc_to_uv(adc_val: int) -> float:
    centered = adc_val - 2048
    return (centered * ADC_VREF_MV * 1000.0) / (ADC_MAX_VALUE * AD8232_GAIN)

def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = fs / 2.0
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, padlen=min(len(signal)-1, 3*max(len(a),len(b))))

class HealthPacket:
    __slots__ = ['timestamp','seq','ecg','leads','red','ir','battery_v','battery_pct','temp','crc','crc_valid']

    def __init__(self, data: bytes):
        if len(data) < PACKET_SIZE:
            raise ValueError(f"Packet too short")
        computed_crc = crc16_modbus(data[:-2])
        received_crc = struct.unpack('<H', data[-2:])[0]
        self.crc_valid = computed_crc == received_crc
        offset = 0
        self.timestamp = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        self.seq = struct.unpack('<H', data[offset:offset+2])[0]; offset += 2
        self.ecg = list(struct.unpack('<hhhhh', data[offset:offset+10])); offset += 10
        self.leads = struct.unpack('<B', data[offset:offset+1])[0]; offset += 1
        self.red = list(struct.unpack('<II', data[offset:offset+8])); offset += 8
        self.ir = list(struct.unpack('<II', data[offset:offset+8])); offset += 8
        self.battery_v = struct.unpack('<f', data[offset:offset+4])[0]; offset += 4
        self.battery_pct = struct.unpack('<f', data[offset:offset+4])[0]; offset += 4
        self.temp = struct.unpack('<f', data[offset:offset+4])[0]
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
                return
            if self.last_seq >= 0:
                expected = (self.last_seq + 1) & 0xFFFF
                if packet.seq != expected:
                    dropped = (packet.seq - expected) & 0xFFFF
                    if dropped < 1000:
                        self.dropped += dropped
            self.last_seq = packet.seq
            self.packet_count += 1
            self.signals.new_packet.emit(packet)
        except:
            pass

    async def connect(self, device):
        try:
            self.signals.status_message.emit("Connecting...")
            self.client = BleakClient(device.address)
            await self.client.connect()
            if self.client.is_connected:
                try:
                    await self.client.request_mtu(185)
                except:
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
            except:
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
            return np.concatenate([self.data[self.write_idx:], self.data[:self.write_idx]])

    def clear(self):
        with self.lock:
            self.data.fill(0)
            self.write_idx = 0
            self.count = 0

class LogThread(QThread):
    def __init__(self, log_queue, session_name, save_path):
        super().__init__()
        self.log_queue = log_queue
        self.session_name = session_name
        self.save_path = save_path
        self.running = False
        self.counts = {'ecg': 0, 'ppg_red': 0, 'ppg_ir': 0, 'vitals': 0}
        self.files = {}

    def run(self):
        self.running = True
        os.makedirs(self.save_path, exist_ok=True)
        try:
            self.files['ecg'] = open(os.path.join(self.save_path, f"{self.session_name}_ecg.txt"), 'w')
            self.files['ecg'].write("Packet_Number,Timestamp,Data\n")
            self.files['ppg_red'] = open(os.path.join(self.save_path, f"{self.session_name}_ppg_red.txt"), 'w')
            self.files['ppg_red'].write("Packet_Number,Timestamp,Data\n")
            self.files['ppg_ir'] = open(os.path.join(self.save_path, f"{self.session_name}_ppg_ir.txt"), 'w')
            self.files['ppg_ir'].write("Packet_Number,Timestamp,Data\n")
            self.files['vitals'] = open(os.path.join(self.save_path, f"{self.session_name}_vitals.txt"), 'w')
            self.files['vitals'].write("Packet_Number,Timestamp,Battery_V,Battery_Pct,Temp,Leads\n")
            while self.running:
                try:
                    data = self.log_queue.get(timeout=0.5)
                    if data is None:
                        break
                    self._write_data(data)
                except queue.Empty:
                    continue
        finally:
            self._convert_to_csv()

    def _write_data(self, data):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        dtype = data.get('type')
        if dtype == 'ecg':
            samples = data.get('samples', [])
            if samples:
                self.counts['ecg'] += 1
                self.files['ecg'].write(f"{self.counts['ecg']},{ts},[{'|'.join(map(str, samples))}]\n")
                self.files['ecg'].flush()
        elif dtype == 'ppg_red':
            values = data.get('values', [])
            if values:
                self.counts['ppg_red'] += 1
                self.files['ppg_red'].write(f"{self.counts['ppg_red']},{ts},[{'|'.join(map(str, values))}]\n")
                self.files['ppg_red'].flush()
        elif dtype == 'ppg_ir':
            values = data.get('values', [])
            if values:
                self.counts['ppg_ir'] += 1
                self.files['ppg_ir'].write(f"{self.counts['ppg_ir']},{ts},[{'|'.join(map(str, values))}]\n")
                self.files['ppg_ir'].flush()
        elif dtype == 'vitals':
            self.counts['vitals'] += 1
            self.files['vitals'].write(
                f"{self.counts['vitals']},{ts},"
                f"{data.get('battery_v', 0):.2f},{data.get('battery_pct', 0):.1f},"
                f"{data.get('temp', 0):.2f},{data.get('leads', 0)}\n")
            self.files['vitals'].flush()

    def _convert_to_csv(self):
        for key, f in self.files.items():
            if f and not f.closed:
                f.close()
        for key in self.files:
            txt_path = os.path.join(self.save_path, f"{self.session_name}_{key}.txt")
            csv_path = os.path.join(self.save_path, f"{self.session_name}_{key}.csv")
            if os.path.exists(txt_path):
                try:
                    os.rename(txt_path, csv_path)
                except:
                    pass

    def stop(self):
        self.running = False

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_snr(signal, fs):
    """Calculate Signal-to-Noise Ratio"""
    if len(signal) < fs:
        return float('-inf')
    try:
        signal = signal - np.mean(signal)
        filtered = bandpass_filter(signal, 0.5, 40.0, fs)
        signal_power = np.var(filtered)
        noise = signal - filtered
        noise_power = np.var(noise)
        if noise_power < 1e-10:
            return 50.0
        return 10 * np.log10(signal_power / noise_power)
    except:
        return float('-inf')

def find_best_ecg_window(ecg_data, fs, window_sec=5.0):
    """Find best quality ECG window"""
    window_samples = int(window_sec * fs)
    step_samples = int(0.25 * fs)
    
    print(f"[WINDOW] Searching {len(ecg_data)/fs:.1f}s for best {window_sec}s window")
    
    if len(ecg_data) < window_samples:
        return 0, len(ecg_data), calculate_snr(ecg_data, fs), 0
    
    best_score = -999
    best_idx = 0
    best_snr = 0
    best_peaks = 0
    
    for start in range(0, len(ecg_data) - window_samples + 1, step_samples):
        end = start + window_samples
        segment = ecg_data[start:end]
        
        snr = calculate_snr(segment, fs)
        
        # Count R-peaks using NeuroKit2
        rpeak_count = 0
        if NEUROKIT_AVAILABLE:
            try:
                cleaned = nk.ecg_clean(segment, sampling_rate=fs)
                _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
                peaks = rpeaks_info.get('ECG_R_Peaks', [])
                rpeak_count = len(peaks)
            except:
                pass
        
        score = snr + (rpeak_count * 2)  # Bonus for R-peaks
        
        if score > best_score and rpeak_count >= MIN_RPEAKS:
            best_score = score
            best_idx = start
            best_snr = snr
            best_peaks = rpeak_count
    
    print(f"[WINDOW] Best: {best_idx/fs:.2f}s-{(best_idx+window_samples)/fs:.2f}s, SNR={best_snr:.1f}dB, R-peaks={best_peaks}")
    return best_idx, best_idx + window_samples, best_snr, best_peaks

def process_ecg(ecg_uv, fs):
    """Process ECG using NeuroKit2"""
    print(f"\n{'='*70}")
    print(f"[ECG] Processing {len(ecg_uv)} samples @ {fs}Hz ({len(ecg_uv)/fs:.2f}s)")
    print(f"{'='*70}")
    
    result = {
        'raw_uv': ecg_uv,
        'raw_mv': ecg_uv / 1000.0,
        'cleaned_uv': None,
        'cleaned_mv': None,
        'r_peaks': np.array([]),
        'heart_rate': None,
        'rr_intervals_ms': np.array([]),
        'hrv': {},
        'waves': {},
        'intervals': {},
    }
    
    if not NEUROKIT_AVAILABLE:
        result['cleaned_uv'] = ecg_uv
        result['cleaned_mv'] = ecg_uv / 1000.0
        return result
    
    # Clean
    try:
        cleaned = nk.ecg_clean(ecg_uv, sampling_rate=fs)
        result['cleaned_uv'] = cleaned
        result['cleaned_mv'] = cleaned / 1000.0
        print(f"[ECG] Cleaned: {cleaned.min():.1f} to {cleaned.max():.1f} µV")
    except Exception as e:
        print(f"[ECG] Clean failed: {e}")
        cleaned = ecg_uv
        result['cleaned_uv'] = ecg_uv
        result['cleaned_mv'] = ecg_uv / 1000.0
    
    # R-peaks
    try:
        _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        r_peaks = np.array(rpeaks_info.get('ECG_R_Peaks', []))
        result['r_peaks'] = r_peaks
        print(f"[ECG] R-peaks: {len(r_peaks)}")
        
        if len(r_peaks) >= 2:
            rr_samples = np.diff(r_peaks)
            rr_ms = rr_samples / fs * 1000
            rr_valid = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
            result['rr_intervals_ms'] = rr_valid
            
            if len(rr_valid) > 0:
                hr_bpm = 60000 / rr_valid
                result['heart_rate'] = {
                    'mean': float(np.mean(hr_bpm)),
                    'std': float(np.std(hr_bpm)),
                    'min': float(np.min(hr_bpm)),
                    'max': float(np.max(hr_bpm)),
                }
                print(f"[ECG] HR: {result['heart_rate']['mean']:.1f} ± {result['heart_rate']['std']:.1f} bpm")
    except Exception as e:
        print(f"[ECG] Peak detection failed: {e}")
    
    # PQRST delineation
    if len(r_peaks) >= 3:
        try:
            _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=fs, method='dwt')
            result['waves'] = waves
            
            # Calculate intervals
            intervals = {}
            
            # PR interval
            p_onsets = waves.get('ECG_P_Onsets', [])
            pr_vals = []
            for i, r in enumerate(r_peaks):
                if i < len(p_onsets) and p_onsets[i] is not None and not np.isnan(p_onsets[i]):
                    pr = (r - p_onsets[i]) / fs * 1000
                    if 80 < pr < 300:
                        pr_vals.append(pr)
            if pr_vals:
                intervals['PR_Interval'] = float(np.mean(pr_vals))
            
            # QRS duration
            q_peaks = waves.get('ECG_Q_Peaks', [])
            s_peaks = waves.get('ECG_S_Peaks', [])
            qrs_vals = []
            for i in range(len(r_peaks)):
                if i < len(q_peaks) and i < len(s_peaks):
                    q, s = q_peaks[i], s_peaks[i]
                    if q is not None and s is not None and not np.isnan(q) and not np.isnan(s):
                        qrs = (s - q) / fs * 1000
                        if 40 < qrs < 200:
                            qrs_vals.append(qrs)
            if qrs_vals:
                intervals['QRS_Duration'] = float(np.mean(qrs_vals))
            
            # QT interval
            t_offsets = waves.get('ECG_T_Offsets', [])
            qt_vals = []
            for i in range(len(r_peaks)):
                if i < len(q_peaks) and i < len(t_offsets):
                    q, t = q_peaks[i], t_offsets[i]
                    if q is not None and t is not None and not np.isnan(q) and not np.isnan(t):
                        qt = (t - q) / fs * 1000
                        if 200 < qt < 600:
                            qt_vals.append(qt)
            if qt_vals:
                intervals['QT_Interval'] = float(np.mean(qt_vals))
                if result.get('heart_rate'):
                    hr = result['heart_rate']['mean']
                    rr_sec = 60.0 / hr
                    intervals['QTc'] = float(intervals['QT_Interval'] / np.sqrt(rr_sec))
            
            result['intervals'] = intervals
            print(f"[ECG] Intervals: {len(intervals)} calculated")
        except Exception as e:
            print(f"[ECG] Delineation failed: {e}")
    
    # HRV
    if len(r_peaks) >= 4:
        try:
            hrv_time = nk.hrv_time(r_peaks, sampling_rate=fs, show=False)
            for col in ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50']:
                if col in hrv_time.columns:
                    val = hrv_time[col].values[0]
                    if val is not None and not np.isnan(val):
                        result['hrv'][col] = float(val)
            print(f"[ECG] HRV: {len(result['hrv'])} metrics")
        except Exception as e:
            print(f"[ECG] HRV failed: {e}")
    
    return result

def process_ppg(red_data, ir_data, fs, patient_height_m=1.70):
    """Process PPG with Age Index, PAT, Respiration Rate"""
    print(f"\n{'='*70}")
    print(f"[PPG] Processing {len(ir_data)} samples @ {fs}Hz ({len(ir_data)/fs:.2f}s)")
    print(f"{'='*70}")
    
    result = {
        'red_raw': red_data,
        'ir_raw': ir_data,
        'red_filtered': None,
        'ir_filtered': None,
        'red_peaks': np.array([]),
        'ir_peaks': np.array([]),
        'heart_rate': None,
        'spo2': None,
        'perfusion_index': None,
        'age_index': None,
        'respiration_rate': None,
        'baseline_shift': {},
    }
    
    # Preprocess IR
    ir_signal = np.array(ir_data, dtype=np.float64)
    ir_dc_removed = ir_signal - np.mean(ir_signal)
    ir_detrended = detrend(ir_dc_removed)
    ir_filtered = bandpass_filter(ir_detrended, 0.5, 8.0, fs, order=3)
    ir_normalized = (ir_filtered - np.mean(ir_filtered)) / (np.std(ir_filtered) + 1e-10)
    result['ir_filtered'] = ir_filtered
    
    # Preprocess Red
    red_signal = np.array(red_data, dtype=np.float64)
    red_dc_removed = red_signal - np.mean(red_signal)
    red_detrended = detrend(red_dc_removed)
    red_filtered = bandpass_filter(red_detrended, 0.5, 8.0, fs, order=3)
    red_normalized = (red_filtered - np.mean(red_filtered)) / (np.std(red_filtered) + 1e-10)
    result['red_filtered'] = red_filtered
    
    # Find peaks
    if NEUROKIT_AVAILABLE:
        try:
            peaks_info = nk.ppg_findpeaks(ir_normalized, sampling_rate=fs)
            ir_peaks = np.array(peaks_info.get('PPG_Peaks', []))
            result['ir_peaks'] = ir_peaks
            print(f"[PPG] IR peaks: {len(ir_peaks)}")
        except:
            ir_peaks = np.array([])
    else:
        ir_peaks = np.array([])
    
    if NEUROKIT_AVAILABLE:
        try:
            peaks_info = nk.ppg_findpeaks(red_normalized, sampling_rate=fs)
            red_peaks = np.array(peaks_info.get('PPG_Peaks', []))
            result['red_peaks'] = red_peaks
            print(f"[PPG] Red peaks: {len(red_peaks)}")
        except:
            red_peaks = np.array([])
    else:
        red_peaks = np.array([])
    
    # Heart rate from peaks
    peaks = ir_peaks if len(ir_peaks) >= len(red_peaks) else red_peaks
    if len(peaks) >= 2:
        pp_samples = np.diff(peaks)
        pp_ms = pp_samples / fs * 1000
        pp_valid = pp_ms[(pp_ms > 300) & (pp_ms < 2000)]
        
        if len(pp_valid) > 0:
            hr_bpm = 60000 / pp_valid
            result['heart_rate'] = {
                'mean': float(np.mean(hr_bpm)),
                'std': float(np.std(hr_bpm)),
                'min': float(np.min(hr_bpm)),
                'max': float(np.max(hr_bpm)),
            }
            print(f"[PPG] HR: {result['heart_rate']['mean']:.1f} ± {result['heart_rate']['std']:.1f} bpm")
    
    # SpO2
    if len(ir_peaks) >= 3 and len(red_peaks) >= 3:
        r_ratios = []
        for i in range(min(len(ir_peaks), len(red_peaks)) - 1):
            start_idx = max(ir_peaks[i], red_peaks[i])
            end_idx = min(ir_peaks[i + 1], red_peaks[i + 1]) if i + 1 < min(len(ir_peaks), len(red_peaks)) else len(red_data)
            
            if end_idx - start_idx < 3 or end_idx >= len(red_data):
                continue
            
            red_segment = red_data[start_idx:end_idx]
            ir_segment = ir_data[start_idx:end_idx]
            
            red_dc = np.mean(red_segment)
            ir_dc = np.mean(ir_segment)
            
            if red_dc <= 0 or ir_dc <= 0:
                continue
            
            red_ac = np.max(red_segment) - np.min(red_segment)
            ir_ac = np.max(ir_segment) - np.min(ir_segment)
            
            if ir_ac <= 0 or ir_dc <= 0:
                continue
            
            r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            
            if 0.2 < r_ratio < 2.0:
                r_ratios.append(r_ratio)
        
        if len(r_ratios) >= 2:
            r_avg = np.mean(r_ratios)
            lookup_index = int(r_avg * 100)
            if 0 <= lookup_index < len(SPO2_LOOKUP):
                result['spo2'] = SPO2_LOOKUP[lookup_index]
            else:
                result['spo2'] = max(0, min(100, int(-45.060 * r_avg * r_avg + 30.354 * r_avg + 94.845)))
            print(f"[PPG] SpO2: {result['spo2']}%")
            
            # Perfusion Index
            ir_dc = np.mean(ir_data)
            ir_ac = np.max(ir_data) - np.min(ir_data)
            if ir_dc > 0:
                result['perfusion_index'] = (ir_ac / ir_dc) * 100
                print(f"[PPG] PI: {result['perfusion_index']:.2f}%")
    
    # Age Index (Stiffness Index)
    if len(ir_peaks) >= 3:
        try:
            delta_t_values = []
            for i in range(len(ir_peaks) - 1):
                peak_idx = ir_peaks[i]
                next_peak_idx = ir_peaks[i + 1]
                
                # Search for dicrotic notch between peaks
                search_start = peak_idx + int(0.12 * fs)  # 120ms after peak
                search_end = min(peak_idx + int(0.35 * fs), next_peak_idx)  # Up to 350ms
                
                if search_end > search_start and search_end < len(ir_normalized):
                    segment = ir_normalized[search_start:search_end]
                    notch_idx_rel = np.argmin(segment)
                    notch_idx = search_start + notch_idx_rel
                    
                    delta_t = (notch_idx - peak_idx) / fs  # seconds
                    
                    if 0.1 < delta_t < 0.35:  # Physiological range
                        delta_t_values.append(delta_t)
            
            if len(delta_t_values) >= 2:
                avg_delta_t = np.mean(delta_t_values)
                age_index = patient_height_m / avg_delta_t  # m/s
                result['age_index'] = age_index
                print(f"[PPG] Age Index: {age_index:.2f} m/s (Δt={avg_delta_t*1000:.1f}ms)")
        except Exception as e:
            print(f"[PPG] Age Index failed: {e}")
    
    # Respiration Rate
    if len(ir_data) >= fs * 30:  # Need at least 30 seconds
        try:
            # Extract envelope using Hilbert transform
            analytic_signal = hilbert(ir_filtered)
            amplitude_envelope = np.abs(analytic_signal)
            
            # Bandpass filter envelope for respiratory frequencies (0.1-0.5 Hz = 6-30 bpm)
            resp_signal = bandpass_filter(amplitude_envelope, 0.1, 0.5, fs, order=2)
            
            # Find peaks in respiratory signal
            min_distance = int(2.0 * fs)  # Min 2 seconds between breaths
            peaks, _ = scipy_find_peaks(resp_signal, distance=min_distance)
            
            if len(peaks) >= 3:
                breath_intervals = np.diff(peaks) / fs  # seconds
                breath_rate = 60.0 / np.mean(breath_intervals)  # breaths per minute
                
                if 5 < breath_rate < 40:  # Physiological range
                    result['respiration_rate'] = breath_rate
                    print(f"[PPG] Respiration: {breath_rate:.1f} breaths/min")
        except Exception as e:
            print(f"[PPG] Respiration failed: {e}")
    
    # Baseline Shift Analysis
    try:
        # Linear trend
        x = np.arange(len(ir_filtered))
        coeffs = np.polyfit(x, ir_filtered, 1)
        drift_per_sample = coeffs[0]
        drift_per_sec = drift_per_sample * fs
        
        # Detrended signal stats
        trend_line = np.polyval(coeffs, x)
        detrended_sig = ir_filtered - trend_line
        std_dev = np.std(detrended_sig)
        
        drift_magnitude = np.max(ir_filtered) - np.min(ir_filtered)
        
        result['baseline_shift'] = {
            'drift_per_sec': drift_per_sec,
            'std_deviation': std_dev,
            'drift_magnitude': drift_magnitude,
        }
        print(f"[PPG] Baseline: drift={drift_per_sec:.3f}/s, std={std_dev:.2f}")
    except Exception as e:
        print(f"[PPG] Baseline analysis failed: {e}")
    
    return result

def calculate_pat(ecg_r_peaks, ppg_peaks, ecg_fs, ppg_fs):
    """Calculate Pulse Arrival Time"""
    if len(ecg_r_peaks) < 2 or len(ppg_peaks) < 2:
        return None
    
    # Convert peak indices to time
    ecg_times = ecg_r_peaks / ecg_fs
    ppg_times = ppg_peaks / ppg_fs
    
    pat_values = []
    
    for r_time in ecg_times:
        # Find next PPG peak after this R-peak
        later_ppg = ppg_times[ppg_times > r_time]
        if len(later_ppg) > 0:
            ppg_time = later_ppg[0]
            pat = (ppg_time - r_time) * 1000  # Convert to ms
            
            # Physiological range: 80-400ms
            if 80 < pat < 400:
                pat_values.append(pat)
    
    if len(pat_values) >= 2:
        avg_pat = np.mean(pat_values)
        print(f"[PAT] Pulse Arrival Time: {avg_pat:.1f}ms")
        return avg_pat
    
    return None

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, session_dir, output_path):
        super().__init__()
        self.session_dir = session_dir
        self.output_path = output_path
        self.ecg_data = None
        self.ppg_red_data = None
        self.ppg_ir_data = None
        self.patient_info = {}

    def run(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.gridspec as gridspec
            
            self.plt = plt
            self.PdfPages = PdfPages
            self.gridspec = gridspec

            self.progress.emit("Loading session data...")
            self._load_data()

            if self.ecg_data is None or len(self.ecg_data) < ECG_FS * 3:
                self.finished.emit(False, "Insufficient ECG data (need 3+ seconds)")
                return

            self.progress.emit("Finding best ECG window...")
            start_idx, end_idx, snr, rpeak_count = find_best_ecg_window(self.ecg_data, ECG_FS)
            ecg_window = self.ecg_data[start_idx:end_idx]
            
            self.progress.emit("Analyzing ECG...")
            ecg_result = process_ecg(ecg_window, ECG_FS)

            ppg_result = None
            pat = None
            if self.ppg_ir_data is not None and self.ppg_red_data is not None:
                if len(self.ppg_ir_data) > PPG_FS * 5:
                    self.progress.emit("Analyzing PPG...")
                    
                    # Get patient height if available
                    height_m = 1.70  # default
                    if 'Height' in self.patient_info:
                        try:
                            height_cm = float(self.patient_info['Height'])
                            if height_cm > 50 and height_cm < 250:
                                height_m = height_cm / 100.0
                        except:
                            pass
                    
                    ppg_result = process_ppg(self.ppg_red_data, self.ppg_ir_data, PPG_FS, height_m)
                    
                    # Calculate PAT if we have both ECG and PPG peaks
                    if len(ecg_result['r_peaks']) >= 2 and len(ppg_result['ir_peaks']) >= 2:
                        self.progress.emit("Calculating PAT...")
                        pat = calculate_pat(
                            ecg_result['r_peaks'] + start_idx,  # Adjust for window offset
                            ppg_result['ir_peaks'],
                            ECG_FS,
                            PPG_FS
                        )
                        if pat:
                            ppg_result['pat'] = pat

            self.progress.emit("Generating PDF report...")
            self._create_pdf(ecg_result, ppg_result, start_idx, end_idx, snr)
            
            self.finished.emit(True, self.output_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e))

    def _load_data(self):
        for filename in os.listdir(self.session_dir):
            filepath = os.path.join(self.session_dir, filename)
            if filename.endswith('_ecg.csv'):
                self.ecg_data = self._parse_waveform(filepath)
            elif filename.endswith('_ppg_red.csv'):
                self.ppg_red_data = self._parse_waveform(filepath)
            elif filename.endswith('_ppg_ir.csv'):
                self.ppg_ir_data = self._parse_waveform(filepath)
            elif filename == 'patient_info.txt':
                self._load_patient_info(filepath)

    def _parse_waveform(self, filepath):
        try:
            df = pd.read_csv(filepath)
            if 'Data' not in df.columns:
                return None
            samples = []
            for row in df['Data']:
                s = str(row).strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                    samples.extend([float(x) for x in s.split('|') if x.strip()])
            return np.array(samples, dtype=np.float64) if samples else None
        except:
            return None

    def _load_patient_info(self, filepath):
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        self.patient_info[key.strip()] = value.strip()
        except:
            pass

    def _draw_ecg_grid(self, ax, duration, y_min=-2.0, y_max=2.0):
        """Draw medical ECG grid (25mm/s, 10mm/mV)"""
        ax.set_facecolor('white')
        # Minor grid (1mm = 0.04s, 0.1mV)
        for x in np.arange(0, duration + 0.04, 0.04):
            ax.axvline(x, color='#ffdddd', lw=0.3, zorder=0)
        for y in np.arange(y_min, y_max + 0.1, 0.1):
            ax.axhline(y, color='#ffdddd', lw=0.3, zorder=0)
        # Major grid (5mm = 0.2s, 0.5mV)
        for x in np.arange(0, duration + 0.2, 0.2):
            ax.axvline(x, color='#ffaaaa', lw=0.5, zorder=0)
        for y in np.arange(y_min, y_max + 0.5, 0.5):
            ax.axhline(y, color='#ffaaaa', lw=0.5, zorder=0)
        ax.axhline(0, color='#ff8888', lw=0.8, zorder=0)

    def _create_pdf(self, ecg_result, ppg_result, start_idx, end_idx, snr):
        plt = self.plt
        
        with self.PdfPages(self.output_path) as pdf:
            # Page 1: Cover
            self._page_cover(pdf, plt, ecg_result, ppg_result, snr)
            
            # Page 2: ECG Medical Grid with R-peaks and intervals
            self._page_ecg_medical(pdf, plt, ecg_result)
            
            # Page 3: ECG PQRST Labeled Beats
            self._page_ecg_pqrst(pdf, plt, ecg_result)
            
            # Page 4: ECG HRV Analysis
            if ecg_result.get('hrv'):
                self._page_ecg_hrv(pdf, plt, ecg_result)
            
            # Page 5: PPG Signal with Peaks
            if ppg_result:
                self._page_ppg_signals(pdf, plt, ppg_result)
            
            # Page 6: PPG Analysis (SpO2, Age Index, PAT, Respiration)
            if ppg_result:
                self._page_ppg_analysis(pdf, plt, ppg_result)
            
            # Page 7: Summary
            self._page_summary(pdf, plt, ecg_result, ppg_result)

    def _page_cover(self, pdf, plt, ecg_result, ppg_result, snr):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('NirogScan Health Report', fontsize=24, fontweight='bold', y=0.95)
        ax = fig.add_subplot(111)
        ax.axis('off')

        y = 0.85
        ax.text(0.5, y, 'Patient Information', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        y -= 0.04
        for key, value in self.patient_info.items():
            ax.text(0.3, y, f'{key}:', ha='right', fontsize=10, transform=ax.transAxes)
            ax.text(0.32, y, str(value), ha='left', fontsize=10, transform=ax.transAxes)
            y -= 0.025

        y -= 0.02
        ax.text(0.5, y, 'Vital Signs', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        y -= 0.04

        items = []
        hr = ecg_result.get('heart_rate')
        if hr:
            items.append(('Heart Rate (ECG)', f'{hr["mean"]:.1f} ± {hr["std"]:.1f} bpm'))
        
        if ppg_result:
            if ppg_result.get('spo2'):
                items.append(('SpO2', f'{ppg_result["spo2"]}%'))
            if ppg_result.get('heart_rate'):
                items.append(('Pulse Rate (PPG)', f'{ppg_result["heart_rate"]["mean"]:.1f} bpm'))
            if ppg_result.get('age_index'):
                items.append(('Age Index', f'{ppg_result["age_index"]:.2f} m/s'))
            if ppg_result.get('pat'):
                items.append(('PAT', f'{ppg_result["pat"]:.1f} ms'))
            if ppg_result.get('respiration_rate'):
                items.append(('Respiration Rate', f'{ppg_result["respiration_rate"]:.1f} breaths/min'))
        
        for label, value in items:
            ax.text(0.3, y, f'{label}:', ha='right', fontsize=10, transform=ax.transAxes)
            ax.text(0.32, y, value, ha='left', fontsize=10, fontfamily='monospace', transform=ax.transAxes)
            y -= 0.025

        y -= 0.03
        ax.text(0.5, y, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=9, ha='center', transform=ax.transAxes, style='italic', color='gray')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ecg_medical(self, pdf, plt, ecg_result):
        """ECG with medical grid, R-peaks, and interval labels"""
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('ECG Medical Grid Analysis', fontsize=14, fontweight='bold')

        clean_mv = ecg_result['cleaned_mv']
        r_peaks = ecg_result.get('r_peaks', np.array([]))
        duration = len(clean_mv) / ECG_FS
        t = np.arange(len(clean_mv)) / ECG_FS

        # Plot 1: Full cleaned ECG
        ax = axes[0]
        self._draw_ecg_grid(ax, duration)
        ax.plot(t, clean_mv, 'k-', lw=0.8, zorder=5, label='Cleaned ECG')
        if len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < len(clean_mv)]
            ax.scatter(valid_peaks/ECG_FS, clean_mv[valid_peaks], color='red', s=100, marker='v', 
                      zorder=10, label='R-peaks')
            
            # Label RR intervals
            for i in range(len(valid_peaks) - 1):
                rr_ms = (valid_peaks[i+1] - valid_peaks[i]) / ECG_FS * 1000
                mid_t = (valid_peaks[i] + valid_peaks[i+1]) / 2 / ECG_FS
                ax.annotate(f'RR={rr_ms:.0f}ms', xy=(mid_t, -1.7), fontsize=7, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', ec='orange', alpha=0.9))
        
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('Amplitude (mV)', fontsize=10)
        ax.set_title('Cleaned ECG with R-peaks')
        ax.legend(loc='upper right', fontsize=8)

        # Plot 2: Single beat zoomed
        ax = axes[1]
        if len(r_peaks) >= 2:
            idx = len(r_peaks) // 2
            r = r_peaks[idx]
            pre, post = int(0.3 * ECG_FS), int(0.4 * ECG_FS)
            start, end = max(0, r - pre), min(len(clean_mv), r + post)
            beat = clean_mv[start:end]
            t_beat = (np.arange(len(beat)) - (r - start)) / ECG_FS * 1000
            
            beat_duration = len(beat) / ECG_FS
            self._draw_ecg_grid(ax, beat_duration * 0.001, -1.5, 1.5)  # Smaller y-range for single beat
            ax.plot(t_beat, beat, 'b-', lw=2)
            ax.axvline(0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='R-peak')
            
            # Show intervals if available
            intervals = ecg_result.get('intervals', {})
            info = ""
            for k in ['PR_Interval', 'QRS_Duration', 'QT_Interval', 'QTc']:
                if k in intervals:
                    info += f"{k.replace('_', ' ')}: {intervals[k]:.0f}ms\n"
            if info:
                ax.text(0.02, 0.98, info.strip(), transform=ax.transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
            
            ax.set_xlabel('Time (ms)', fontsize=10)
            ax.set_ylabel('Amplitude (mV)', fontsize=10)
            ax.set_title('Single Beat Detail')
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for single beat view', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ecg_pqrst(self, pdf, plt, ecg_result):
        """PQRST labeled beats with colored segments"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('ECG PQRST Wave Detection', fontsize=14, fontweight='bold')
        
        gs = self.gridspec.GridSpec(2, 2, figure=fig)
        
        clean_mv = ecg_result['cleaned_mv']
        r_peaks = ecg_result.get('r_peaks', np.array([]))
        waves = ecg_result.get('waves', {})
        
        if len(r_peaks) < 3 or not waves:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'PQRST delineation not available\n(requires 3+ R-peaks)', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Plot 1: Beat with PQRST points labeled
        ax = fig.add_subplot(gs[0, 0])
        if len(r_peaks) >= 2:
            idx = len(r_peaks) // 2
            r = r_peaks[idx]
            pre, post = int(0.3 * ECG_FS), int(0.4 * ECG_FS)
            start, end = max(0, r - pre), min(len(clean_mv), r + post)
            beat = clean_mv[start:end]
            t_beat = (np.arange(len(beat)) - (r - start)) / ECG_FS * 1000
            
            ax.plot(t_beat, beat, 'b-', lw=2)
            ax.axvline(0, color='red', linestyle='--', lw=1, alpha=0.5)
            
            # Label PQRST peaks
            markers = {
                'ECG_P_Peaks': ('P', 'green'),
                'ECG_Q_Peaks': ('Q', 'orange'),
                'ECG_S_Peaks': ('S', 'purple'),
                'ECG_T_Peaks': ('T', 'brown')
            }
            for wkey, (label, color) in markers.items():
                if wkey in waves:
                    arr = np.array(waves[wkey])
                    if idx < len(arr) and not np.isnan(arr[idx]):
                        w_idx = int(arr[idx])
                        if start <= w_idx < end:
                            w_t = (w_idx - r) / ECG_FS * 1000
                            ax.scatter([w_t], [clean_mv[w_idx]], c=color, s=120, zorder=10)
                            ax.annotate(label, (w_t, clean_mv[w_idx]), xytext=(0, 15),
                                       textcoords='offset points', ha='center', fontsize=11,
                                       fontweight='bold', color=color)
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('mV')
            ax.set_title('PQRST Points Labeled')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Beat with colored segments
        ax = fig.add_subplot(gs[0, 1])
        if len(r_peaks) >= 2:
            idx = len(r_peaks) // 2
            r = r_peaks[idx]
            pre, post = int(0.3 * ECG_FS), int(0.4 * ECG_FS)
            start, end = max(0, r - pre), min(len(clean_mv), r + post)
            beat = clean_mv[start:end]
            t_beat = np.arange(len(beat)) / ECG_FS * 1000 - (r - start) / ECG_FS * 1000
            
            # Default plot
            ax.plot(t_beat, beat, 'gray', lw=1, alpha=0.5)
            
            # Color code segments
            def get_idx(key):
                if key in waves:
                    arr = np.array(waves[key])
                    if idx < len(arr) and not np.isnan(arr[idx]):
                        return int(arr[idx])
                return None
            
            p_on = get_idx('ECG_P_Onsets')
            p_peak = get_idx('ECG_P_Peaks')
            p_off = get_idx('ECG_P_Offsets')
            q_peak = get_idx('ECG_Q_Peaks')
            s_peak = get_idx('ECG_S_Peaks')
            t_on = get_idx('ECG_T_Onsets')
            t_peak = get_idx('ECG_T_Peaks')
            t_off = get_idx('ECG_T_Offsets')
            
            # P-wave (green)
            if p_on and p_off and start <= p_on < p_off < end:
                p_seg = beat[p_on-start:p_off-start+1]
                p_t = t_beat[p_on-start:p_off-start+1]
                ax.plot(p_t, p_seg, 'green', lw=3, label='P-wave')
            
            # QRS complex (red)
            if q_peak and s_peak and start <= q_peak < s_peak < end:
                qrs_seg = beat[q_peak-start:s_peak-start+1]
                qrs_t = t_beat[q_peak-start:s_peak-start+1]
                ax.plot(qrs_t, qrs_seg, 'red', lw=3, label='QRS')
            
            # T-wave (blue)
            if t_on and t_off and start <= t_on < t_off < end:
                t_seg = beat[t_on-start:t_off-start+1]
                t_t = t_beat[t_on-start:t_off-start+1]
                ax.plot(t_t, t_seg, 'blue', lw=3, label='T-wave')
            
            ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('mV')
            ax.set_title('Colored Segments')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot 3: RR Tachogram
        ax = fig.add_subplot(gs[1, 0])
        rr_ms = ecg_result.get('rr_intervals_ms', np.array([]))
        if len(rr_ms) >= 2:
            ax.plot(rr_ms, 'b-o', markersize=5, lw=1.5)
            ax.axhline(np.mean(rr_ms), color='red', linestyle='--', lw=2, 
                      label=f'Mean: {np.mean(rr_ms):.0f}ms')
            ax.fill_between(range(len(rr_ms)), 
                           np.mean(rr_ms)-np.std(rr_ms), 
                           np.mean(rr_ms)+np.std(rr_ms),
                           alpha=0.2, color='red')
            ax.set_xlabel('Beat #')
            ax.set_ylabel('RR Interval (ms)')
            ax.set_title('RR Tachogram')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient RR data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # Plot 4: Poincaré Plot
        ax = fig.add_subplot(gs[1, 1])
        if len(rr_ms) >= 3:
            rr1, rr2 = rr_ms[:-1], rr_ms[1:]
            ax.scatter(rr1, rr2, alpha=0.7, s=50, c='blue', edgecolors='darkblue')
            min_v = min(rr1.min(), rr2.min()) * 0.95
            max_v = max(rr1.max(), rr2.max()) * 1.05
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5)
            ax.set_xlabel('RR(n) ms')
            ax.set_ylabel('RR(n+1) ms')
            ax.set_title('Poincaré Plot')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ecg_hrv(self, pdf, plt, ecg_result):
        """HRV analysis page"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Heart Rate Variability Analysis', fontsize=14, fontweight='bold')
        
        hrv = ecg_result.get('hrv', {})
        
        # HRV Metrics
        ax = axes[0, 0]
        ax.axis('off')
        ax.text(0.5, 0.98, 'HRV Time Domain', fontsize=12, fontweight='bold', 
               ha='center', transform=ax.transAxes)
        y = 0.85
        for key in ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50']:
            if key in hrv:
                ax.text(0.1, y, f'{key}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.6, y, f'{hrv[key]:.2f}', fontsize=10, fontfamily='monospace',
                       fontweight='bold', transform=ax.transAxes)
                y -= 0.12
        
        # RR Histogram
        ax = axes[0, 1]
        rr_ms = ecg_result.get('rr_intervals_ms', np.array([]))
        if len(rr_ms) >= 3:
            ax.hist(rr_ms, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(rr_ms), color='red', linestyle='--', lw=2, 
                      label=f'Mean: {np.mean(rr_ms):.0f}ms')
            ax.set_xlabel('RR Interval (ms)')
            ax.set_ylabel('Count')
            ax.set_title('RR Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # Placeholder plots
        for idx in [2, 3]:
            ax = axes.flatten()[idx]
            ax.text(0.5, 0.5, 'Additional HRV metrics\n(Frequency domain requires\n60+ seconds)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ppg_signals(self, pdf, plt, ppg_result):
        """PPG signals with peaks labeled"""
        fig, axes = plt.subplots(3, 1, figsize=(11, 8.5))
        fig.suptitle('PPG Signal Analysis', fontsize=14, fontweight='bold')
        
        ir_raw = ppg_result['ir_raw']
        red_raw = ppg_result['red_raw']
        ir_filtered = ppg_result.get('ir_filtered', ir_raw)
        red_filtered = ppg_result.get('red_filtered', red_raw)
        ir_peaks = ppg_result.get('ir_peaks', np.array([]))
        red_peaks = ppg_result.get('red_peaks', np.array([]))
        
        window = min(len(ir_raw), PPG_FS * 15)
        t = np.arange(window) / PPG_FS
        
        # IR Channel
        ax = axes[0]
        ax.plot(t, ir_raw[:window], 'purple', lw=0.8, alpha=0.4, label='Raw IR')
        ax.plot(t, ir_filtered[:window], 'b-', lw=1.2, label='Filtered IR')
        peaks_win = ir_peaks[ir_peaks < window]
        if len(peaks_win) > 0:
            ax.scatter(peaks_win/PPG_FS, ir_filtered[peaks_win], 
                      color='red', s=80, marker='v', zorder=10, label='Peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ADC Counts')
        ax.set_title(f'IR Channel - {len(ir_peaks)} peaks detected')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Red Channel
        ax = axes[1]
        ax.plot(t, red_raw[:window], 'red', lw=0.8, alpha=0.4, label='Raw Red')
        ax.plot(t, red_filtered[:window], 'darkred', lw=1.2, label='Filtered Red')
        peaks_win = red_peaks[red_peaks < window]
        if len(peaks_win) > 0:
            ax.scatter(peaks_win/PPG_FS, red_filtered[peaks_win], 
                      color='green', s=80, marker='v', zorder=10, label='Peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ADC Counts')
        ax.set_title(f'Red Channel - {len(red_peaks)} peaks detected')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Baseline Shift Plot
        ax = axes[2]
        if 'baseline_shift' in ppg_result and ppg_result['baseline_shift']:
            # Show detrended signal
            x = np.arange(len(ir_filtered))
            coeffs = np.polyfit(x, ir_filtered, 1)
            trend_line = np.polyval(coeffs, x)
            detrended = ir_filtered - trend_line
            
            t_full = np.arange(len(ir_filtered)) / PPG_FS
            ax.plot(t_full, ir_filtered, 'b-', lw=0.8, alpha=0.5, label='Original')
            ax.plot(t_full, trend_line, 'r--', lw=2, label='Trend')
            ax.plot(t_full, detrended, 'g-', lw=0.8, label='Detrended')
            
            bs = ppg_result['baseline_shift']
            info = f"Drift: {bs['drift_per_sec']:.3f}/s\nStd: {bs['std_deviation']:.2f}"
            ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ADC Counts')
            ax.set_title('Baseline Shift Analysis')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Baseline analysis not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ppg_analysis(self, pdf, plt, ppg_result):
        """PPG analysis: SpO2, Age Index, PAT, Respiration"""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('PPG Advanced Analysis', fontsize=14, fontweight='bold')
        
        gs = self.gridspec.GridSpec(2, 2, figure=fig)
        
        # SpO2
        ax = fig.add_subplot(gs[0, 0])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Oxygen Saturation', fontsize=12, fontweight='bold', 
               ha='center', transform=ax.transAxes)
        spo2 = ppg_result.get('spo2')
        if spo2:
            color = 'green' if spo2 >= 95 else ('orange' if spo2 >= 90 else 'red')
            ax.text(0.5, 0.6, f'{spo2}%', fontsize=48, fontweight='bold', 
                   ha='center', transform=ax.transAxes, color=color)
            ax.text(0.5, 0.35, 'SpO2', fontsize=14, ha='center', transform=ax.transAxes)
            
            pi = ppg_result.get('perfusion_index')
            if pi:
                ax.text(0.5, 0.2, f'Perfusion Index: {pi:.2f}%', fontsize=10, 
                       ha='center', transform=ax.transAxes, color='gray')
        else:
            ax.text(0.5, 0.5, 'SpO2\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
        
        # Age Index (Stiffness)
        ax = fig.add_subplot(gs[0, 1])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Age Index (Stiffness)', fontsize=12, fontweight='bold', 
               ha='center', transform=ax.transAxes)
        age_idx = ppg_result.get('age_index')
        if age_idx:
            ax.text(0.5, 0.6, f'{age_idx:.2f}', fontsize=42, fontweight='bold', 
                   ha='center', transform=ax.transAxes, color='purple')
            ax.text(0.5, 0.4, 'm/s', fontsize=14, ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.25, 'Arterial Stiffness Index', fontsize=10, 
                   ha='center', transform=ax.transAxes, style='italic', color='gray')
            ax.text(0.5, 0.15, '(Height / ΔT_dicrotic)', fontsize=9, 
                   ha='center', transform=ax.transAxes, color='gray')
        else:
            ax.text(0.5, 0.5, 'Age Index\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
        
        # PAT
        ax = fig.add_subplot(gs[1, 0])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Pulse Arrival Time', fontsize=12, fontweight='bold', 
               ha='center', transform=ax.transAxes)
        pat = ppg_result.get('pat')
        if pat:
            ax.text(0.5, 0.6, f'{pat:.1f}', fontsize=42, fontweight='bold', 
                   ha='center', transform=ax.transAxes, color='teal')
            ax.text(0.5, 0.4, 'ms', fontsize=14, ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.25, 'ECG R-peak to PPG peak', fontsize=10, 
                   ha='center', transform=ax.transAxes, style='italic', color='gray')
        else:
            ax.text(0.5, 0.5, 'PAT\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
        
        # Respiration Rate
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Respiration Rate', fontsize=12, fontweight='bold', 
               ha='center', transform=ax.transAxes)
        resp = ppg_result.get('respiration_rate')
        if resp:
            ax.text(0.5, 0.6, f'{resp:.1f}', fontsize=42, fontweight='bold', 
                   ha='center', transform=ax.transAxes, color='darkgreen')
            ax.text(0.5, 0.4, 'breaths/min', fontsize=14, ha='center', transform=ax.transAxes)
            ax.text(0.5, 0.25, 'From PPG amplitude modulation', fontsize=10, 
                   ha='center', transform=ax.transAxes, style='italic', color='gray')
        else:
            ax.text(0.5, 0.5, 'Respiration\nnot available\n(requires 30s+ signal)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_summary(self, pdf, plt, ecg_result, ppg_result):
        """Summary page"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.97, 'Analysis Summary', fontsize=18, fontweight='bold', 
               ha='center', transform=ax.transAxes)
        
        y = 0.88
        
        # ECG
        ax.text(0.05, y, 'ECG Analysis:', fontsize=13, fontweight='bold', 
               transform=ax.transAxes, color='#2060a0')
        y -= 0.03
        
        hr = ecg_result.get('heart_rate')
        if hr:
            ax.text(0.07, y, 'Heart Rate:', fontsize=10, transform=ax.transAxes)
            ax.text(0.35, y, f'{hr["mean"]:.1f} ± {hr["std"]:.1f} bpm', fontsize=10, 
                   fontfamily='monospace', transform=ax.transAxes)
            y -= 0.025
        
        intervals = ecg_result.get('intervals', {})
        for k in ['PR_Interval', 'QRS_Duration', 'QTc']:
            if k in intervals:
                ax.text(0.07, y, f'{k.replace("_", " ")}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{intervals[k]:.0f} ms', fontsize=10, 
                       fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
        
        hrv = ecg_result.get('hrv', {})
        for k in ['HRV_SDNN', 'HRV_RMSSD']:
            if k in hrv:
                ax.text(0.07, y, f'{k}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{hrv[k]:.1f}', fontsize=10, 
                       fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
        
        # PPG
        if ppg_result:
            y -= 0.02
            ax.text(0.05, y, 'PPG Analysis:', fontsize=13, fontweight='bold', 
                   transform=ax.transAxes, color='#a02060')
            y -= 0.03
            
            if ppg_result.get('spo2'):
                ax.text(0.07, y, 'SpO2:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{ppg_result["spo2"]}%', fontsize=10, 
                       fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
            
            if ppg_result.get('age_index'):
                ax.text(0.07, y, 'Age Index:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{ppg_result["age_index"]:.2f} m/s', fontsize=10, 
                       fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
            
            if ppg_result.get('pat'):
                ax.text(0.07, y, 'PAT:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{ppg_result["pat"]:.1f} ms', fontsize=10, 
                       fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
            
            if ppg_result.get('respiration_rate'):
                ax.text(0.07, y, 'Respiration:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{ppg_result["respiration_rate"]:.1f} br/min', fontsize=10, 
                       fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
        
        y -= 0.03
        ax.text(0.5, y, 'Note: For informational purposes only. Consult healthcare professional for medical advice.',
               fontsize=9, ha='center', transform=ax.transAxes, style='italic', color='#666666')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

# ============================================================================
# GUI
# ============================================================================

class PatientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Patient Information")
        self.setModal(True)
        self.setMinimumWidth(400)
        self.setStyleSheet(DARK_STYLE)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name = QLineEdit()
        self.name.setPlaceholderText("Enter patient name")
        self.age = QLineEdit()
        self.age.setPlaceholderText("Enter age")
        self.age.setValidator(QIntValidator(0, 150))
        self.gender = QComboBox()
        self.gender.addItems(["Male", "Female", "Other"])
        self.weight = QLineEdit()
        self.weight.setPlaceholderText("Optional (kg)")
        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("Optional (cm)")
        self.notes = QTextEdit()
        self.notes.setPlaceholderText("Notes...")
        self.notes.setMaximumHeight(80)

        form.addRow("Name *:", self.name)
        form.addRow("Age *:", self.age)
        form.addRow("Gender:", self.gender)
        form.addRow("Weight (kg):", self.weight)
        form.addRow("Height (cm):", self.height_edit)
        form.addRow("Notes:", self.notes)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._validate)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def _validate(self):
        if not self.name.text().strip() or not self.age.text().strip():
            QMessageBox.warning(self, "Error", "Name and Age required")
            return
        self.accept()

    def get_info(self):
        return {
            'Name': self.name.text().strip(),
            'Age': self.age.text().strip(),
            'Gender': self.gender.currentText(),
            'Weight': self.weight.text().strip() or 'N/A',
            'Height': self.height_edit.text().strip() or 'N/A',
            'Notes': self.notes.toPlainText().strip() or 'None'
        }

class SessionDialog(QDialog):
    def __init__(self, base_dir, parent=None):
        super().__init__(parent)
        self.base_dir = base_dir
        self.selected = None
        self.setWindowTitle("Select Session")
        self.setModal(True)
        self.setMinimumSize(450, 350)
        self.setStyleSheet(DARK_STYLE)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select session:"))
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._select)
        layout.addWidget(self.list)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._select)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(self.gen_btn)
        layout.addLayout(btn_layout)

        self.list.itemSelectionChanged.connect(lambda: self.gen_btn.setEnabled(bool(self.list.selectedItems())))
        self._load()

    def _load(self):
        if not os.path.exists(self.base_dir):
            return
        for item in sorted(os.listdir(self.base_dir), reverse=True):
            path = os.path.join(self.base_dir, item)
            if os.path.isdir(path):
                files = os.listdir(path)
                if any('_ecg.csv' in f for f in files):
                    li = QListWidgetItem(item)
                    li.setData(Qt.UserRole, path)
                    self.list.addItem(li)

    def _select(self):
        sel = self.list.selectedItems()
        if sel:
            self.selected = sel[0].data(Qt.UserRole)
            self.accept()

class NirogScanGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NirogScan v5.0 Fixed")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet(DARK_STYLE)

        self.signals = DataSignals()
        self.ble_manager = BLEManager(self.signals)

        self.ecg_buffer = CircularBuffer(PLOT_WINDOW, np.float32)
        self.ppg_red_buffer = CircularBuffer(PLOT_WINDOW, np.uint32)
        self.ppg_ir_buffer = CircularBuffer(PLOT_WINDOW, np.uint32)

        self.packet_count = 0
        self.last_packet = None
        self.data_lock = Lock()

        self.save_dir = None
        self.log_queue = queue.Queue(maxsize=1000)
        self.log_thread = None
        self.logging = False
        self.report_thread = None

        self._setup_ui()
        self._connect_signals()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(UPDATE_RATE_MS)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setSpacing(10)

        # Plots
        plots = QWidget()
        pl = QVBoxLayout(plots)
        pl.setSpacing(5)

        self.ecg_plot = pg.PlotWidget(title=f"ECG ({ECG_FS} Hz)")
        self.ecg_plot.setBackground('#1e1e1e')
        self.ecg_plot.setLabel('left', 'µV')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.setYRange(-1500, 1500)
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen('#00ff00', width=1))
        pl.addWidget(self.ecg_plot, stretch=2)

        self.ppg_red_plot = pg.PlotWidget(title=f"PPG Red ({PPG_FS} Hz)")
        self.ppg_red_plot.setBackground('#1e1e1e')
        self.ppg_red_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_red_curve = self.ppg_red_plot.plot(pen=pg.mkPen('#ff0000', width=1))
        pl.addWidget(self.ppg_red_plot, stretch=1)

        self.ppg_ir_plot = pg.PlotWidget(title=f"PPG IR ({PPG_FS} Hz)")
        self.ppg_ir_plot.setBackground('#1e1e1e')
        self.ppg_ir_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_ir_curve = self.ppg_ir_plot.plot(pen=pg.mkPen('#ff00ff', width=1))
        pl.addWidget(self.ppg_ir_plot, stretch=1)

        main.addWidget(plots, stretch=1)

        # Control Panel
        panel = QWidget()
        panel.setFixedWidth(280)
        panel_layout = QVBoxLayout(panel)

        # Connection
        conn = QGroupBox("Connection")
        cl = QVBoxLayout()
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self._on_scan)
        cl.addWidget(self.scan_btn)
        self.device_combo = QComboBox()
        cl.addWidget(self.device_combo)
        bl = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        self.connect_btn.setEnabled(False)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self._on_disconnect)
        self.disconnect_btn.setEnabled(False)
        bl.addWidget(self.connect_btn)
        bl.addWidget(self.disconnect_btn)
        cl.addLayout(bl)
        conn.setLayout(cl)
        panel_layout.addWidget(conn)

        # Status
        status = QGroupBox("Status")
        sl = QGridLayout()
        font = QFont()
        font.setBold(True)
        labels = [("Packets:", "pkt_lbl", "0"), ("Dropped:", "drop_lbl", "0"), 
                 ("CRC:", "crc_lbl", "0"), ("Battery:", "bat_lbl", "--"), 
                 ("Temp:", "temp_lbl", "--"), ("Leads:", "lead_lbl", "--")]
        for i, (txt, attr, val) in enumerate(labels):
            sl.addWidget(QLabel(txt), i, 0)
            lbl = QLabel(val)
            lbl.setFont(font)
            setattr(self, attr, lbl)
            sl.addWidget(lbl, i, 1)
        status.setLayout(sl)
        panel_layout.addWidget(status)

        # Logging
        log = QGroupBox("Logging")
        ll = QVBoxLayout()
        self.dir_lbl = QLabel("No directory")
        self.dir_lbl.setWordWrap(True)
        self.dir_lbl.setStyleSheet("color: #888; font-size: 9px;")
        ll.addWidget(self.dir_lbl)
        self.dir_btn = QPushButton("Select Directory")
        self.dir_btn.clicked.connect(self._on_dir)
        ll.addWidget(self.dir_btn)
        self.log_btn = QPushButton("Start Logging")
        self.log_btn.clicked.connect(self._on_log)
        self.log_btn.setEnabled(False)
        ll.addWidget(self.log_btn)
        log.setLayout(ll)
        panel_layout.addWidget(log)

        # Report
        rpt = QGroupBox("Report")
        rl = QVBoxLayout()
        self.rpt_btn = QPushButton("Generate Report")
        self.rpt_btn.clicked.connect(self._on_report)
        rl.addWidget(self.rpt_btn)
        nk_txt = f"NeuroKit2: {'v'+nk.__version__ if NEUROKIT_AVAILABLE else 'NOT INSTALLED'}"
        nk_lbl = QLabel(nk_txt)
        nk_lbl.setStyleSheet(f"color: {'#5f5' if NEUROKIT_AVAILABLE else '#f55'}; font-size: 9px;")
        rl.addWidget(nk_lbl)
        rpt.setLayout(rl)
        panel_layout.addWidget(rpt)

        panel_layout.addStretch()
        main.addWidget(panel)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def _connect_signals(self):
        self.signals.new_packet.connect(self._on_packet)
        self.signals.connection_changed.connect(self._on_conn_change)
        self.signals.status_message.connect(lambda m: self.statusBar.showMessage(m))

    @asyncSlot()
    async def _on_scan(self):
        self.scan_btn.setEnabled(False)
        self.device_combo.clear()
        try:
            devs = await self.ble_manager.scan_devices()
            for d in devs:
                self.device_combo.addItem(f"{d.name} ({d.address})", d)
            self.connect_btn.setEnabled(len(devs) > 0)
            self.statusBar.showMessage(f"Found {len(devs)} device(s)")
        except Exception as e:
            self.statusBar.showMessage(f"Scan error: {e}")
        self.scan_btn.setEnabled(True)

    @asyncSlot()
    async def _on_connect(self):
        if self.device_combo.currentIndex() >= 0:
            self.connect_btn.setEnabled(False)
            ok = await self.ble_manager.connect(self.device_combo.currentData())
            if not ok:
                self.connect_btn.setEnabled(True)
            else:
                self.packet_count = 0
                self.ecg_buffer.clear()
                self.ppg_red_buffer.clear()
                self.ppg_ir_buffer.clear()
                self.log_btn.setEnabled(True)

    @asyncSlot()
    async def _on_disconnect(self):
        if self.logging:
            self._stop_log()
        self.disconnect_btn.setEnabled(False)
        await self.ble_manager.disconnect()
        self.log_btn.setEnabled(False)

    def _on_conn_change(self, c):
        self.connect_btn.setEnabled(not c)
        self.disconnect_btn.setEnabled(c)
        self.scan_btn.setEnabled(not c)
        self.device_combo.setEnabled(not c)

    def _on_packet(self, p):
        self.packet_count += 1
        ecg_uv = [adc_to_uv(a) for a in p.ecg]
        self.ecg_buffer.extend(ecg_uv)
        
        # PPG: Only append first value for display (both values are logged)
        if p.red[0] > 0:
            self.ppg_red_buffer.append(p.red[0])
            self.ppg_ir_buffer.append(p.ir[0])
        
        with self.data_lock:
            self.last_packet = p
        
        if self.logging:
            try:
                self.log_queue.put_nowait({'type': 'ecg', 'samples': ecg_uv})
                # Log BOTH PPG values (not just first)
                if p.red[0] > 0 or p.red[1] > 0:
                    self.log_queue.put_nowait({'type': 'ppg_red', 'values': list(p.red)})
                    self.log_queue.put_nowait({'type': 'ppg_ir', 'values': list(p.ir)})
                self.log_queue.put_nowait({'type': 'vitals', 
                                          'battery_v': p.battery_v,
                                          'battery_pct': p.battery_pct,
                                          'temp': p.temp,
                                          'leads': p.leads})
            except queue.Full:
                pass

    def _update_plots(self):
        self.ecg_curve.setData(self.ecg_buffer.get_ordered())
        self.ppg_red_curve.setData(self.ppg_red_buffer.get_ordered())
        self.ppg_ir_curve.setData(self.ppg_ir_buffer.get_ordered())
        
        with self.data_lock:
            p = self.last_packet
        
        if p:
            self.pkt_lbl.setText(str(self.packet_count))
            self.drop_lbl.setText(str(self.ble_manager.dropped))
            self.crc_lbl.setText(str(self.ble_manager.crc_errors))
            self.bat_lbl.setText(f"{p.battery_v:.2f}V ({p.battery_pct:.0f}%)")
            self.temp_lbl.setText(f"{p.temp:.1f}°C")
            if p.leads == 0:
                self.lead_lbl.setText("OK")
                self.lead_lbl.setStyleSheet("color: #5f5; font-weight: bold;")
            else:
                st = []
                if p.leads & 1: st.append("LO+")
                if p.leads & 2: st.append("LO-")
                self.lead_lbl.setText(" ".join(st) or "ERR")
                self.lead_lbl.setStyleSheet("color: #f55; font-weight: bold;")

    def _on_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            self.save_dir = d
            self.dir_lbl.setText(d)
            self.dir_lbl.setStyleSheet("color: #5f5; font-size: 9px;")

    def _on_log(self):
        if self.logging:
            self._stop_log()
        else:
            self._start_log()

    def _start_log(self):
        if not self.save_dir:
            QMessageBox.warning(self, "Error", "Select directory")
            return
        if not self.ble_manager.connected:
            QMessageBox.warning(self, "Error", "Connect first")
            return
        
        dlg = PatientDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        
        info = dlg.get_info()
        folder = f"{info['Name']}_{info['Age']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        folder = "".join(c for c in folder if c.isalnum() or c in "._- ")
        path = os.path.join(self.save_dir, folder)
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "patient_info.txt"), 'w') as f:
            for k, v in info.items():
                f.write(f"{k}: {v}\n")
        
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break
        
        self.log_thread = LogThread(self.log_queue, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", path)
        self.log_thread.start()
        self.logging = True
        self.log_btn.setText("Stop Logging")
        self.log_btn.setStyleSheet("background-color: #a33;")

    def _stop_log(self):
        if self.log_thread:
            self.log_queue.put(None)
            self.log_thread.stop()
            self.log_thread.wait()
            self.log_thread = None
        self.logging = False
        self.log_btn.setText("Start Logging")
        self.log_btn.setStyleSheet("")

    def _on_report(self):
        if not self.save_dir:
            QMessageBox.warning(self, "Error", "Select directory")
            return
        
        dlg = SessionDialog(self.save_dir, self)
        if dlg.exec_() != QDialog.Accepted or not dlg.selected:
            return
        
        out, _ = QFileDialog.getSaveFileName(self, "Save Report",
            os.path.join(self.save_dir, os.path.basename(dlg.selected) + "_report.pdf"), 
            "PDF (*.pdf)")
        if not out:
            return
        
        self.rpt_btn.setEnabled(False)
        self.report_thread = ReportGenerator(dlg.selected, out)
        self.report_thread.progress.connect(lambda m: self.statusBar.showMessage(m))
        self.report_thread.finished.connect(self._on_report_done)
        self.report_thread.start()

    def _on_report_done(self, ok, msg):
        self.rpt_btn.setEnabled(True)
        if ok:
            QMessageBox.information(self, "Done", f"Report saved:\n{msg}")
        else:
            QMessageBox.critical(self, "Error", f"Failed:\n{msg}")

    def closeEvent(self, e):
        if self.logging:
            self._stop_log()
        if self.ble_manager.connected:
            asyncio.get_event_loop().run_until_complete(self.ble_manager.disconnect())
        if self.report_thread and self.report_thread.isRunning():
            self.report_thread.wait()
        e.accept()

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = NirogScanGUI()
    win.show()
    with loop:
        loop.run_forever()

if __name__ == '__main__':
    main()