#!/usr/bin/env python3
"""
NirogScan v4.3 - Enhanced Dual Sample Analysis
- Sample 1: Best quality 3-5 cardiac cycles (high SNR)
- Sample 2: Full recording analysis (if ≥60 seconds)
- Complete NeuroKit2 feature extraction for ECG and PPG
"""

import sys
import os
import asyncio
import struct
import time
import queue
import warnings
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

from scipy.signal import butter, filtfilt, find_peaks as scipy_find_peaks, detrend
from scipy.stats import zscore

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
    NK_VERSION = nk.__version__
    print(f"[INIT] NeuroKit2 v{NK_VERSION} loaded")
except ImportError as e:
    NEUROKIT_AVAILABLE = False
    NK_VERSION = "N/A"
    nk = None
    print(f"[INIT] WARNING: NeuroKit2 not available: {e}")

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
PPG_EFFECTIVE_FS = 100

MIN_RPEAKS_REQUIRED = 5
MIN_PPG_PEAKS_REQUIRED = 5
MIN_DURATION_FOR_SAMPLE2 = 60  # seconds

# Analysis configuration
# Sample 1: Short, high-quality segment for basic analysis (ECG only)
# Sample 2: Full recording for comprehensive long-term analysis (both ECG and PPG)

SPO2_LOOKUP = [
    95, 95, 95, 96, 96, 96, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98,
    99, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 99, 99, 99, 99,
    99, 99, 99, 99, 98, 98, 98, 98, 98, 98, 97, 97, 97, 97, 96, 96,
    96, 96, 95, 95, 95, 94, 94, 94, 93, 93, 93, 92, 92, 92, 91, 91,
    90, 90, 89, 89, 89, 88, 88, 87, 87, 86, 86, 85, 85, 84, 84, 83,
    82, 82, 81, 81, 80, 80, 79, 78, 78, 77, 76, 76, 75, 74, 74, 73,
    72, 72, 71, 70, 69, 69, 68, 67, 66, 66, 65, 64, 63, 62, 62, 61,
    60, 59, 58, 57, 56, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
    45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 31, 30, 29,
    28, 27, 26, 25, 23, 22, 21, 20, 19, 17, 16, 15, 14, 12, 11, 10,
    9, 7, 6, 5, 3, 2, 1
]

METRIC_EXPLANATIONS = {
    'HRV_MeanNN': ('Average RR Interval', 'Average time between heartbeats. Normal: 600-1000ms.'),
    'HRV_SDNN': ('Heart Rate Variability', 'How much heartbeat timing varies. Higher = healthier. Normal: 50-100ms.'),
    'HRV_RMSSD': ('Short-term HRV', 'Beat-to-beat changes. Normal: 20-50ms.'),
    'HRV_pNN50': ('Large Beat Changes %', 'Percentage of beats >50ms different. Normal: 3-25%.'),
    'HRV_pNN20': ('Small Beat Changes %', 'Percentage of beats >20ms different.'),
    'HRV_SDSD': ('Successive Diff SD', 'Standard deviation of differences.'),
    'HRV_MedianNN': ('Median RR', 'Middle RR value, robust to outliers.'),
    'HRV_VLF': ('Very Low Freq Power', 'Long-term regulation. Needs 5+ min.'),
    'HRV_LF': ('Low Freq Power', 'Mixed sympathetic/parasympathetic. Normal: 400-1500 ms².'),
    'HRV_HF': ('High Freq Power', 'Parasympathetic activity. Normal: 150-400 ms².'),
    'HRV_LFHF': ('LF/HF Ratio', 'Stress balance. Normal: 1.5-2.0.'),
    'HRV_SD1': ('Poincaré SD1', 'Short-term variability.'),
    'HRV_SD2': ('Poincaré SD2', 'Long-term variability.'),
    'HRV_ApEn': ('Approximate Entropy', 'Signal complexity.'),
    'HRV_SampEn': ('Sample Entropy', 'Rhythm complexity.'),
    'HRV_DFA_alpha1': ('DFA Short-term', 'Fractal scaling. Normal: 0.75-1.25.'),
    'PR_Interval': ('PR Interval', 'Atrial to ventricular time. Normal: 120-200ms.'),
    'QRS_Duration': ('QRS Duration', 'Ventricular contraction. Normal: 80-120ms.'),
    'QT_Interval': ('QT Interval', 'Total ventricular activity.'),
    'QTc': ('Corrected QT', 'HR-adjusted QT. Normal: <440ms.'),
    'SpO2': ('Oxygen Saturation', 'Blood oxygen. Normal: 95-100%.'),
    'Perfusion_Index': ('Perfusion Index', 'Blood flow strength.'),
    'PPG_HR': ('Pulse Rate', 'Heart rate from PPG.'),
    'PRV_SDNN': ('Pulse Rate Variability', 'PRV from PPG.'),
    'PRV_RMSSD': ('PRV Short-term', 'Beat-to-beat PPG variability.'),
}

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


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if low >= high:
        return signal
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, padlen=min(len(signal)-1, 3*max(len(a), len(b))))


def lowpass_filter(signal: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    normalized_cutoff = min(0.999, cutoff / nyq)
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, signal, padlen=min(len(signal)-1, 3*max(len(a), len(b))))


class HealthPacket:
    __slots__ = ['timestamp', 'seq', 'ecg', 'leads', 'red', 'ir',
                 'battery_v', 'battery_pct', 'temp', 'crc', 'crc_valid']

    def __init__(self, data: bytes):
        if len(data) < PACKET_SIZE:
            raise ValueError(f"Packet too short: {len(data)} bytes")
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
    def __init__(self, signals: DataSignals):
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
        except Exception:
            pass

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
    def __init__(self, size: int, dtype=np.float32):
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
                except Exception:
                    pass

    def stop(self):
        self.running = False


class SignalQualityAnalyzer:
    @staticmethod
    def calculate_snr(signal: np.ndarray, fs: int) -> float:
        if len(signal) < fs:
            return float('-inf')
        try:
            signal = np.array(signal, dtype=np.float64)
            signal = signal - np.mean(signal)
            filtered = bandpass_filter(signal, 0.5, 40.0, fs)
            signal_power = np.var(filtered)
            noise = signal - filtered
            noise_power = np.var(noise)
            if noise_power < 1e-10:
                return 50.0
            return 10 * np.log10(signal_power / noise_power)
        except Exception:
            return float('-inf')

    @staticmethod
    def count_rpeaks_in_segment(segment: np.ndarray, fs: int) -> int:
        if not NEUROKIT_AVAILABLE or len(segment) < fs:
            return 0
        try:
            cleaned = nk.ecg_clean(segment, sampling_rate=fs)
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
            peaks = rpeaks.get('ECG_R_Peaks', np.array([]))
            return len(peaks)
        except Exception:
            return 0

    @staticmethod
    def find_best_ecg_window(ecg_data: np.ndarray, fs: int, 
                             min_rpeaks: int = MIN_RPEAKS_REQUIRED,
                             window_sec: float = 5.0, 
                             step_sec: float = 0.25) -> Tuple[int, int, float, int]:
        """Find best quality ECG window with high SNR and sufficient R-peaks"""
        window_samples = int(window_sec * fs)
        step_samples = int(step_sec * fs)
        
        print(f"[ECG-WINDOW] Searching {len(ecg_data)} samples ({len(ecg_data)/fs:.1f}s)")
        
        if len(ecg_data) < window_samples:
            snr = SignalQualityAnalyzer.calculate_snr(ecg_data, fs)
            rpeaks = SignalQualityAnalyzer.count_rpeaks_in_segment(ecg_data, fs)
            return 0, len(ecg_data), snr, rpeaks

        candidates = []
        for start in range(0, len(ecg_data) - window_samples + 1, step_samples):
            end = start + window_samples
            segment = ecg_data[start:end]
            snr = SignalQualityAnalyzer.calculate_snr(segment, fs)
            rpeak_count = SignalQualityAnalyzer.count_rpeaks_in_segment(segment, fs)
            
            if rpeak_count >= min_rpeaks:
                candidates.append((start, end, snr, rpeak_count, snr))
        
        if not candidates:
            snr = SignalQualityAnalyzer.calculate_snr(ecg_data[:window_samples], fs)
            rpeaks = SignalQualityAnalyzer.count_rpeaks_in_segment(ecg_data[:window_samples], fs)
            return 0, window_samples, snr, rpeaks
        
        candidates.sort(key=lambda x: x[4], reverse=True)
        best = candidates[0]
        print(f"[ECG-WINDOW] BEST: {best[0]/fs:.2f}s-{best[1]/fs:.2f}s, SNR: {best[2]:.1f}dB, R-peaks: {best[3]}")
        return best[0], best[1], best[2], best[3]




class PPGPreprocessor:
    @staticmethod
    def preprocess(raw_signal: np.ndarray, fs: int = PPG_FS) -> Dict[str, np.ndarray]:
        print(f"[PPG-PRE] Input: {len(raw_signal)} samples")
        
        signal = np.array(raw_signal, dtype=np.float64)
        dc_removed = signal - np.mean(signal)
        
        try:
            detrended = detrend(dc_removed)
        except Exception:
            detrended = dc_removed
        
        filtered = bandpass_filter(detrended, 0.5, 8.0, fs, order=3)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        return {
            'raw': signal,
            'dc_removed': dc_removed,
            'detrended': detrended,
            'filtered': filtered,
            'normalized': normalized
        }

    @staticmethod
    def find_peaks_robust(signal: np.ndarray, fs: int) -> np.ndarray:
        min_peak_distance = int(0.4 * fs)
        methods_results = []
        
        # Method 1: Scipy inverted
        try:
            inverted = -signal
            height_threshold = np.percentile(inverted, 70)
            peaks, _ = scipy_find_peaks(
                inverted, 
                height=height_threshold,
                distance=min_peak_distance,
                prominence=np.std(signal) * 0.3
            )
            if len(peaks) >= 3:
                methods_results.append(('scipy_inverted', peaks, len(peaks)))
        except Exception:
            pass
        
        # Method 2: NeuroKit2
        if NEUROKIT_AVAILABLE:
            try:
                peaks_info = nk.ppg_findpeaks(signal, sampling_rate=fs)
                peaks = np.array(peaks_info.get('PPG_Peaks', []))
                if len(peaks) >= 3:
                    methods_results.append(('neurokit', peaks, len(peaks)))
            except Exception:
                pass
        
        if not methods_results:
            return np.array([])
        
        best_method = max(methods_results, key=lambda x: x[2])
        return best_method[1]


class SpO2Calculator:
    @staticmethod
    def calculate(red_signal: np.ndarray, ir_signal: np.ndarray, 
                  red_peaks: np.ndarray, ir_peaks: np.ndarray,
                  fs: int = PPG_FS) -> Dict[str, Any]:
        
        result = {
            'spo2': None,
            'r_ratios': [],
            'perfusion_index_red': None,
            'perfusion_index_ir': None,
            'valid': False
        }
        
        peaks = ir_peaks if len(ir_peaks) >= len(red_peaks) else red_peaks
        
        if len(peaks) < 3:
            return result
        
        r_ratios = []
        
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            
            if end_idx - start_idx < 3 or end_idx >= len(red_signal):
                continue
            
            red_segment = red_signal[start_idx:end_idx]
            ir_segment = ir_signal[start_idx:end_idx]
            
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
            result['r_ratios'] = r_ratios
            
            lookup_index = int(r_avg * 100)
            if 0 <= lookup_index < len(SPO2_LOOKUP):
                spo2 = SPO2_LOOKUP[lookup_index]
            else:
                spo2 = max(0, min(100, int(-45.060 * r_avg * r_avg + 30.354 * r_avg + 94.845)))
            
            result['spo2'] = spo2
            result['valid'] = True
        
        try:
            red_dc = np.mean(red_signal)
            red_ac = np.max(red_signal) - np.min(red_signal)
            if red_dc > 0:
                result['perfusion_index_red'] = (red_ac / red_dc) * 100
            
            ir_dc = np.mean(ir_signal)
            ir_ac = np.max(ir_signal) - np.min(ir_signal)
            if ir_dc > 0:
                result['perfusion_index_ir'] = (ir_ac / ir_dc) * 100
        except Exception:
            pass
        
        return result


class ECGProcessor:
    """Complete ECG analysis with all NeuroKit2 features"""
    def __init__(self, fs: int = ECG_FS):
        self.fs = fs

    def process(self, ecg_uv: np.ndarray, sample_name: str = "", long_term_analysis: bool = False) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"[ECG-{sample_name}] {len(ecg_uv)} samples @ {self.fs} Hz ({len(ecg_uv)/self.fs:.2f}s)")
        print(f"{'='*70}")
        
        ecg_signal = np.array(ecg_uv, dtype=np.float64)
        
        result = {
            'sample_name': sample_name,
            'duration_sec': len(ecg_uv) / self.fs,
            'raw_uv': ecg_signal,
            'raw_mv': ecg_signal / 1000.0,
            'cleaned_uv': None,
            'cleaned_mv': None,
            'r_peaks': np.array([]),
            'heart_rate': None,
            'rr_intervals_ms': np.array([]),
            'hrv_time': {},
            'hrv_freq': {},
            'hrv_nonlinear': {},
            'waves': {},
            'intervals': {},
            'quality': None,
            'analysis_method': 'none',
        }
        
        if not NEUROKIT_AVAILABLE:
            result['cleaned_uv'] = ecg_signal
            result['cleaned_mv'] = ecg_signal / 1000.0
            return result

        # Clean signal
        try:
            cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method='neurokit')
            result['cleaned_uv'] = cleaned
            result['cleaned_mv'] = cleaned / 1000.0
        except Exception as e:
            print(f"[ECG] Clean failed: {e}")
            cleaned = ecg_signal
            result['cleaned_uv'] = ecg_signal
            result['cleaned_mv'] = ecg_signal / 1000.0

        # Find R-peaks
        try:
            _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            r_peaks = np.array(rpeaks_info.get('ECG_R_Peaks', []))
            result['r_peaks'] = r_peaks
            print(f"[ECG] R-peaks: {len(r_peaks)}")
        except Exception as e:
            print(f"[ECG] Peak detection failed: {e}")
            r_peaks = np.array([])

        # Heart rate
        if len(r_peaks) >= 2:
            rr_samples = np.diff(r_peaks)
            rr_ms = rr_samples / self.fs * 1000
            rr_valid = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
            result['rr_intervals_ms'] = rr_valid
            
            if len(rr_valid) > 0:
                hr_bpm = 60000 / rr_valid
                result['heart_rate'] = {
                    'mean': float(np.mean(hr_bpm)),
                    'std': float(np.std(hr_bpm)),
                    'min': float(np.min(hr_bpm)),
                    'max': float(np.max(hr_bpm)),
                    'values': hr_bpm
                }

        # Signal quality
        try:
            signals_df, info = nk.ecg_process(cleaned, sampling_rate=self.fs)
            result['analysis_method'] = 'neurokit2_full'
            if 'ECG_Quality' in signals_df.columns:
                quality = signals_df['ECG_Quality'].dropna().values
                if len(quality) > 0:
                    result['quality'] = float(np.mean(quality))
        except Exception as e:
            print(f"[ECG] Process failed: {e}")
            result['analysis_method'] = 'neurokit2_partial'

        # Wave delineation
        if len(r_peaks) >= 3:
            try:
                _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=self.fs, method='dwt')
                result['waves'] = waves
                self._calculate_intervals(result, waves, r_peaks)
            except Exception as e:
                print(f"[ECG] Delineation failed: {e}")

        # HRV analysis
        if len(r_peaks) >= 4:
            self._calculate_hrv(result, r_peaks, long_term_analysis)

        print(f"[ECG-{sample_name}] Complete")
        return result

    def _calculate_intervals(self, result: Dict, waves: Dict, r_peaks: np.ndarray):
        try:
            intervals = {}
            p_onsets = waves.get('ECG_P_Onsets', [])
            q_peaks = waves.get('ECG_Q_Peaks', [])
            s_peaks = waves.get('ECG_S_Peaks', [])
            t_offsets = waves.get('ECG_T_Offsets', [])
            
            # PR Interval
            pr_vals = []
            for i, r in enumerate(r_peaks):
                if i < len(p_onsets) and p_onsets[i] is not None and not np.isnan(p_onsets[i]):
                    pr = (r - p_onsets[i]) / self.fs * 1000
                    if 80 < pr < 300:
                        pr_vals.append(pr)
            if pr_vals:
                intervals['PR_Interval'] = float(np.mean(pr_vals))
            
            # QRS Duration
            qrs_vals = []
            for i in range(len(r_peaks)):
                if i < len(q_peaks) and i < len(s_peaks):
                    q, s = q_peaks[i], s_peaks[i]
                    if q is not None and s is not None and not np.isnan(q) and not np.isnan(s):
                        qrs = (s - q) / self.fs * 1000
                        if 40 < qrs < 200:
                            qrs_vals.append(qrs)
            if qrs_vals:
                intervals['QRS_Duration'] = float(np.mean(qrs_vals))
            
            # QT Interval
            qt_vals = []
            for i in range(len(r_peaks)):
                if i < len(q_peaks) and i < len(t_offsets):
                    q, t = q_peaks[i], t_offsets[i]
                    if q is not None and t is not None and not np.isnan(q) and not np.isnan(t):
                        qt = (t - q) / self.fs * 1000
                        if 200 < qt < 600:
                            qt_vals.append(qt)
            if qt_vals:
                intervals['QT_Interval'] = float(np.mean(qt_vals))
                if result.get('heart_rate') and result['heart_rate'].get('mean'):
                    hr = result['heart_rate']['mean']
                    rr_sec = 60.0 / hr
                    intervals['QTc'] = float(intervals['QT_Interval'] / np.sqrt(rr_sec))
            
            result['intervals'] = intervals
        except Exception as e:
            print(f"[ECG] Interval calc error: {e}")

    def _calculate_hrv(self, result: Dict, r_peaks: np.ndarray, long_term_analysis: bool = False):
        print(f"[ECG] HRV Analysis with {len(r_peaks)} R-peaks... (Long-term: {long_term_analysis})")
        
        # Time domain
        try:
            hrv_time = nk.hrv_time(r_peaks, sampling_rate=self.fs, show=False)
            time_metrics = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20',
                          'HRV_SDSD', 'HRV_CVNN', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_CVSD']
            for m in time_metrics:
                if m in hrv_time.columns:
                    val = hrv_time[m].values[0]
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        result['hrv_time'][m] = float(val)
        except Exception as e:
            print(f"[ECG] HRV Time failed: {e}")
        
        # Frequency domain - ONLY for long-term analysis (Sample 2)
        if long_term_analysis:
            duration_sec = len(result['raw_uv']) / self.fs
            if duration_sec >= 60 and len(r_peaks) >= 30:
                try:
                    hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=self.fs, show=False)
                    freq_metrics = ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 
                                   'HRV_TP', 'HRV_LnHF']
                    for m in freq_metrics:
                        if m in hrv_freq.columns:
                            val = hrv_freq[m].values[0]
                            if val is not None and not np.isnan(val) and not np.isinf(val) and val > 0:
                                result['hrv_freq'][m] = float(val)
                    print(f"[ECG] HRV Freq: {len(result['hrv_freq'])} metrics")
                except Exception as e:
                    print(f"[ECG] HRV Freq failed: {e}")
            else:
                print(f"[ECG] HRV Freq skipped: need 60s+ ({duration_sec:.0f}s available)")
        else:
            print(f"[ECG] HRV Freq skipped: Short-term analysis only (Sample 1)")
        
        # Nonlinear - ONLY for long-term analysis (Sample 2)
        if long_term_analysis and len(r_peaks) >= 10:
            try:
                hrv_nl = nk.hrv_nonlinear(r_peaks, sampling_rate=self.fs, show=False)
                nl_metrics = ['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_ApEn', 'HRV_SampEn', 
                             'HRV_DFA_alpha1', 'HRV_DFA_alpha2', 'HRV_MFDFA_alpha1_Width',
                             'HRV_MSE']
                for m in nl_metrics:
                    if m in hrv_nl.columns:
                        val = hrv_nl[m].values[0]
                        if val is not None and not np.isnan(val) and not np.isinf(val):
                            result['hrv_nonlinear'][m] = float(val)
                print(f"[ECG] HRV Nonlinear: {len(result['hrv_nonlinear'])} metrics")
            except Exception as e:
                print(f"[ECG] HRV Nonlinear failed: {e}")
        else:
            if not long_term_analysis:
                print(f"[ECG] HRV Nonlinear skipped: Short-term analysis only (Sample 1)")
            else:
                print(f"[ECG] HRV Nonlinear skipped: need 10+ R-peaks ({len(r_peaks)} available)")


class PPGProcessor:
    """Complete PPG analysis with all NeuroKit2 features"""
    def __init__(self, fs: int = PPG_FS):
        self.fs = fs

    def process(self, red_data: np.ndarray, ir_data: np.ndarray, sample_name: str = "", long_term_analysis: bool = False) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"[PPG-{sample_name}] {len(ir_data)} samples @ {self.fs} Hz ({len(ir_data)/self.fs:.2f}s)")
        print(f"{'='*70}")
        
        result = {
            'sample_name': sample_name,
            'duration_sec': len(ir_data) / self.fs,
            'red_raw': red_data,
            'ir_raw': ir_data,
            'red_processed': None,
            'ir_processed': None,
            'red_peaks': np.array([]),
            'ir_peaks': np.array([]),
            'heart_rate': None,
            'pp_intervals_ms': np.array([]),
            'prv_time': {},
            'prv_freq': {},
            'prv_nonlinear': {},
            'spo2': None,
            'spo2_data': {},
            'perfusion_index': None,
            'signal_quality': None,
            'analysis_method': 'none',
        }
        
        # Preprocess both channels
        ir_prep = PPGPreprocessor.preprocess(ir_data, self.fs)
        result['ir_processed'] = ir_prep
        
        red_prep = PPGPreprocessor.preprocess(red_data, self.fs)
        result['red_processed'] = red_prep
        
        # Find peaks
        ir_peaks = PPGPreprocessor.find_peaks_robust(ir_prep['normalized'], self.fs)
        result['ir_peaks'] = ir_peaks
        
        red_peaks = PPGPreprocessor.find_peaks_robust(red_prep['normalized'], self.fs)
        result['red_peaks'] = red_peaks
        
        print(f"[PPG] Peaks - IR: {len(ir_peaks)}, Red: {len(red_peaks)}")
        
        peaks = ir_peaks if len(ir_peaks) >= len(red_peaks) else red_peaks
        
        # Heart rate from peaks
        if len(peaks) >= 2:
            pp_samples = np.diff(peaks)
            pp_ms = pp_samples / self.fs * 1000
            pp_valid = pp_ms[(pp_ms > 300) & (pp_ms < 2000)]
            result['pp_intervals_ms'] = pp_valid
            
            if len(pp_valid) > 0:
                hr_bpm = 60000 / pp_valid
                result['heart_rate'] = {
                    'mean': float(np.mean(hr_bpm)),
                    'std': float(np.std(hr_bpm)),
                    'min': float(np.min(hr_bpm)),
                    'max': float(np.max(hr_bpm)),
                    'values': hr_bpm
                }
                result['analysis_method'] = 'custom_preprocessing'
        
        # SpO2 calculation
        if len(ir_peaks) >= 3 and len(red_peaks) >= 3:
            spo2_result = SpO2Calculator.calculate(red_data, ir_data, red_peaks, ir_peaks, self.fs)
            result['spo2_data'] = spo2_result
            if spo2_result['valid']:
                result['spo2'] = spo2_result['spo2']
                result['perfusion_index'] = spo2_result.get('perfusion_index_ir')
        
        # PRV analysis (Pulse Rate Variability)
        if len(peaks) >= 5:
            self._calculate_prv(result, peaks, long_term_analysis)
        
        # Signal quality
        ac = np.std(ir_prep['filtered'])
        dc = np.mean(ir_data)
        if dc > 0:
            result['signal_quality'] = min(1.0, ac / (dc * 0.01))
        
        print(f"[PPG-{sample_name}] Complete")
        return result

    def _calculate_prv(self, result: Dict, peaks: np.ndarray, long_term_analysis: bool = False):
        print(f"[PPG] PRV Analysis with {len(peaks)} peaks... (Long-term: {long_term_analysis})")
        
        if not NEUROKIT_AVAILABLE:
            return
        
        # Time domain PRV - always calculated
        try:
            prv_time = nk.hrv_time(peaks, sampling_rate=self.fs, show=False)
            time_metrics = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20',
                          'HRV_SDSD', 'HRV_MedianNN']
            for m in time_metrics:
                if m in prv_time.columns:
                    val = prv_time[m].values[0]
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        result['prv_time'][m.replace('HRV', 'PRV')] = float(val)
            print(f"[PPG] PRV Time: {len(result['prv_time'])} metrics")
        except Exception as e:
            print(f"[PPG] PRV Time failed: {e}")
        
        # Frequency domain PRV - ONLY for long-term analysis (Sample 2)
        if long_term_analysis:
            duration_sec = len(result['ir_raw']) / self.fs
            if duration_sec >= 60 and len(peaks) >= 30:
                try:
                    prv_freq = nk.hrv_frequency(peaks, sampling_rate=self.fs, show=False)
                    freq_metrics = ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn']
                    for m in freq_metrics:
                        if m in prv_freq.columns:
                            val = prv_freq[m].values[0]
                            if val is not None and not np.isnan(val) and not np.isinf(val) and val > 0:
                                result['prv_freq'][m.replace('HRV', 'PRV')] = float(val)
                    print(f"[PPG] PRV Freq: {len(result['prv_freq'])} metrics")
                except Exception as e:
                    print(f"[PPG] PRV Freq failed: {e}")
            else:
                print(f"[PPG] PRV Freq skipped: need 60s+ ({duration_sec:.0f}s available)")
        else:
            print(f"[PPG] PRV Freq skipped: Short-term analysis only (Sample 1)")
        
        # Nonlinear PRV - ONLY for long-term analysis (Sample 2)
        if long_term_analysis and len(peaks) >= 10:
            try:
                prv_nl = nk.hrv_nonlinear(peaks, sampling_rate=self.fs, show=False)
                nl_metrics = ['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_ApEn', 'HRV_SampEn']
                for m in nl_metrics:
                    if m in prv_nl.columns:
                        val = prv_nl[m].values[0]
                        if val is not None and not np.isnan(val) and not np.isinf(val):
                            result['prv_nonlinear'][m.replace('HRV', 'PRV')] = float(val)
                print(f"[PPG] PRV Nonlinear: {len(result['prv_nonlinear'])} metrics")
            except Exception as e:
                print(f"[PPG] PRV Nonlinear failed: {e}")
        else:
            if not long_term_analysis:
                print(f"[PPG] PRV Nonlinear skipped: Short-term analysis only (Sample 1)")
            else:
                print(f"[PPG] PRV Nonlinear skipped: need 10+ peaks ({len(peaks)} available)")


class ReportGenerator(QThread):
    """Enhanced report with dual-sample analysis"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, session_dir: str, output_path: str):
        super().__init__()
        self.session_dir = session_dir
        self.output_path = output_path
        self.ecg_data = None
        self.ppg_red_data = None
        self.ppg_ir_data = None
        self.vitals_data = None
        self.patient_info = {}

    def run(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            self.plt = plt
            self.PdfPages = PdfPages

            self.progress.emit("Loading session data...")
            self._load_data()

            if self.ecg_data is None or len(self.ecg_data) < ECG_FS * 3:
                self.finished.emit(False, "Insufficient ECG data (need 3+ seconds)")
                return

            # ============== ECG ANALYSIS ==============
            self.progress.emit("Analyzing ECG Sample 1 (Best Quality)...")
            
            # Sample 1: Best quality short segment - BASIC ANALYSIS ONLY
            start1, end1, snr1, rpeak_cnt1 = SignalQualityAnalyzer.find_best_ecg_window(
                self.ecg_data, ECG_FS, min_rpeaks=MIN_RPEAKS_REQUIRED
            )
            ecg_sample1 = self.ecg_data[start1:end1]
            ecg_processor = ECGProcessor(ECG_FS)
            ecg_result1 = ecg_processor.process(ecg_sample1, "Sample1", long_term_analysis=False)
            
            # Sample 2: Full recording - COMPLETE LONG-TERM ANALYSIS
            ecg_result2 = None
            total_duration = len(self.ecg_data) / ECG_FS
            if total_duration >= MIN_DURATION_FOR_SAMPLE2:
                self.progress.emit("Analyzing ECG Sample 2 (Full Recording - Long-term)...")
                ecg_result2 = ecg_processor.process(self.ecg_data, "Sample2", long_term_analysis=True)
            
            # ============== PPG ANALYSIS ==============
            ppg_result1 = None
            ppg_result2 = None
            
            if self.ppg_ir_data is not None and self.ppg_red_data is not None:
                ppg_total_duration = len(self.ppg_ir_data) / PPG_FS
                
                # Sample 1: FULL PPG RECORDING - BASIC ANALYSIS ONLY (no window selection)
                if ppg_total_duration >= 5:
                    self.progress.emit("Analyzing PPG Sample 1 (Full Recording - Basic)...")
                    ppg_processor = PPGProcessor(PPG_FS)
                    ppg_result1 = ppg_processor.process(self.ppg_red_data, self.ppg_ir_data, "Sample1", long_term_analysis=False)
                    
                    # Sample 2: Full recording - COMPLETE LONG-TERM ANALYSIS (same data, different analysis depth)
                    if ppg_total_duration >= MIN_DURATION_FOR_SAMPLE2:
                        self.progress.emit("Analyzing PPG Sample 2 (Full Recording - Long-term)...")
                        ppg_result2 = ppg_processor.process(self.ppg_red_data, self.ppg_ir_data, "Sample2", long_term_analysis=True)

            self.progress.emit("Generating PDF report...")
            self._create_pdf(ecg_result1, ecg_result2, ppg_result1, ppg_result2,
                           start1, end1, snr1, rpeak_cnt1)
            
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
            elif filename.endswith('_vitals.csv'):
                try:
                    self.vitals_data = pd.read_csv(filepath)
                except Exception:
                    pass
            elif filename == 'patient_info.txt':
                self._load_patient_info(filepath)

    def _parse_waveform(self, filepath: str) -> Optional[np.ndarray]:
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
        except Exception as e:
            print(f"[LOAD] Parse error: {e}")
            return None

    def _load_patient_info(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        self.patient_info[key.strip()] = value.strip()
        except Exception:
            pass

    def _draw_ecg_grid(self, ax, duration, y_min=-2.0, y_max=2.0):
        ax.set_facecolor('white')
        for x in np.arange(0, duration + 0.04, 0.04):
            ax.axvline(x, color='#ffdddd', lw=0.3, zorder=0)
        for y in np.arange(y_min, y_max + 0.1, 0.1):
            ax.axhline(y, color='#ffdddd', lw=0.3, zorder=0)
        for x in np.arange(0, duration + 0.2, 0.2):
            ax.axvline(x, color='#ffaaaa', lw=0.5, zorder=0)
        for y in np.arange(y_min, y_max + 0.5, 0.5):
            ax.axhline(y, color='#ffaaaa', lw=0.5, zorder=0)
        ax.axhline(0, color='#ff8888', lw=0.8, zorder=0)

    def _create_pdf(self, ecg_s1, ecg_s2, ppg_s1, ppg_s2, start_idx, end_idx, snr, rpeak_count):
        plt = self.plt
        with self.PdfPages(self.output_path) as pdf:
            # Cover page
            self._page_cover(pdf, plt, ecg_s1, ecg_s2, ppg_s1, ppg_s2)
            
            # ECG Sample 1
            self._page_ecg_signal(pdf, plt, ecg_s1, "Sample 1: Best Quality", start_idx, end_idx, snr)
            if len(ecg_s1.get('r_peaks', [])) >= 3:
                self._page_ecg_morphology(pdf, plt, ecg_s1)
            if ecg_s1.get('hrv_time'):
                self._page_hrv(pdf, plt, ecg_s1)
            
            # ECG Sample 2
            if ecg_s2:
                self._page_ecg_signal(pdf, plt, ecg_s2, "Sample 2: Full Recording", 0, len(self.ecg_data), 0)
                if len(ecg_s2.get('r_peaks', [])) >= 3:
                    self._page_ecg_morphology(pdf, plt, ecg_s2)
                if ecg_s2.get('hrv_time'):
                    self._page_hrv(pdf, plt, ecg_s2)
                # Comparison
                self._page_ecg_comparison(pdf, plt, ecg_s1, ecg_s2)
            
            # PPG Sample 1
            if ppg_s1:
                self._page_ppg_signal(pdf, plt, ppg_s1)
                self._page_ppg_beat_segmentation(pdf, plt, ppg_s1)
                self._page_ppg_analysis(pdf, plt, ppg_s1)
            
            # PPG Sample 2
            if ppg_s2:
                self._page_ppg_signal(pdf, plt, ppg_s2)
                self._page_ppg_beat_segmentation(pdf, plt, ppg_s2)
                self._page_ppg_analysis(pdf, plt, ppg_s2)
                # Comparison
                self._page_ppg_comparison(pdf, plt, ppg_s1, ppg_s2)
            
            # Summary & explanations
            self._page_summary(pdf, plt, ecg_s1, ecg_s2, ppg_s1, ppg_s2)
            self._page_explanations(pdf, plt, ecg_s1, ppg_s1)

    def _page_cover(self, pdf, plt, ecg_s1, ecg_s2, ppg_s1, ppg_s2):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('NirogScan Health Report - Dual Sample Analysis', fontsize=20, fontweight='bold', y=0.95)
        ax = fig.add_subplot(111)
        ax.axis('off')

        y = 0.88
        ax.text(0.5, y, 'Patient Information', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        y -= 0.04
        for key, value in self.patient_info.items():
            ax.text(0.3, y, f'{key}:', ha='right', fontsize=10, transform=ax.transAxes)
            ax.text(0.32, y, str(value), ha='left', fontsize=10, transform=ax.transAxes)
            y -= 0.025

        y -= 0.02
        ax.text(0.5, y, 'Analysis Summary', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
        y -= 0.04

        items = [
            ('Total ECG Duration', f'{len(self.ecg_data)/ECG_FS:.1f}s'),
            ('ECG Sample 1', f'{ecg_s1["duration_sec"]:.1f}s (Best 5s Window - Basic Analysis)'),
        ]
        if ecg_s2:
            items.append(('ECG Sample 2', f'{ecg_s2["duration_sec"]:.1f}s (Full Recording - Complete Analysis)'))
        
        if ppg_s1:
            items.append(('Total PPG Duration', f'{ppg_s1["duration_sec"]:.1f}s'))
            items.append(('PPG Sample 1', 'Same as above (Basic Analysis)'))
        if ppg_s2:
            items.append(('PPG Sample 2', 'Same as above (Complete Long-term Analysis)'))

        for label, value in items:
            ax.text(0.3, y, f'{label}:', ha='right', fontsize=10, transform=ax.transAxes)
            ax.text(0.32, y, value, ha='left', fontsize=10, fontfamily='monospace', transform=ax.transAxes)
            y -= 0.025

        y -= 0.03
        ax.text(0.5, y, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=9, ha='center', transform=ax.transAxes, style='italic', color='gray')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ecg_signal(self, pdf, plt, ecg_result, title, start_idx, end_idx, snr):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle(f'ECG - {title}', fontsize=14, fontweight='bold')

        raw_mv = ecg_result['raw_mv']
        clean_mv = ecg_result['cleaned_mv']
        r_peaks = ecg_result.get('r_peaks', np.array([]))
        duration = len(raw_mv) / ECG_FS
        t = np.arange(len(raw_mv)) / ECG_FS

        ax = axes[0]
        self._draw_ecg_grid(ax, duration)
        ax.plot(t, raw_mv, 'k-', lw=0.8, zorder=5)
        if len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < len(raw_mv)]
            ax.scatter(valid_peaks/ECG_FS, raw_mv[valid_peaks], color='red', s=80, marker='v', zorder=10)
            # Add R-R interval timing annotations
            for i in range(len(valid_peaks) - 1):
                if i < 10:  # Limit annotations to avoid clutter
                    rr_ms = (valid_peaks[i+1] - valid_peaks[i]) / ECG_FS * 1000
                    mid_t = (valid_peaks[i] + valid_peaks[i+1]) / 2 / ECG_FS
                    mid_y = np.mean([raw_mv[valid_peaks[i]], raw_mv[valid_peaks[i+1]]])
                    ax.annotate(f'{rr_ms:.0f}ms', xy=(mid_t, mid_y - 0.3), fontsize=7, ha='center',
                               bbox=dict(boxstyle='round,pad=0.2', fc='yellow', ec='orange', alpha=0.8))
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('mV')
        ax.set_title(f'Raw ECG - {ecg_result["sample_name"]} - R-R Intervals Shown')

        ax = axes[1]
        self._draw_ecg_grid(ax, duration)
        ax.plot(t, clean_mv, 'b-', lw=0.8, zorder=5)
        if len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < len(clean_mv)]
            ax.scatter(valid_peaks/ECG_FS, clean_mv[valid_peaks], color='red', s=80, marker='v', zorder=10)
            # Add R-R interval timing on cleaned signal too
            for i in range(len(valid_peaks) - 1):
                if i < 10:  # Limit annotations
                    rr_ms = (valid_peaks[i+1] - valid_peaks[i]) / ECG_FS * 1000
                    mid_t = (valid_peaks[i] + valid_peaks[i+1]) / 2 / ECG_FS
                    ax.plot([valid_peaks[i]/ECG_FS, valid_peaks[i+1]/ECG_FS], 
                           [1.5, 1.5], 'g-', lw=2, alpha=0.6, zorder=8)
                    ax.text(mid_t, 1.6, f'{rr_ms:.0f}ms', fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.15', fc='lightgreen', alpha=0.9))
        hr = ecg_result.get('heart_rate')
        if hr:
            rr_info = f"HR: {hr['mean']:.1f}±{hr['std']:.1f} bpm\nR-peaks: {len(r_peaks)}"
            rr_intervals = ecg_result.get('rr_intervals_ms', np.array([]))
            if len(rr_intervals) > 0:
                rr_info += f"\nRR: {np.mean(rr_intervals):.0f}±{np.std(rr_intervals):.0f}ms"
            ax.text(0.98, 0.95, rr_info,
                   transform=ax.transAxes, fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.9))
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('mV')
        ax.set_xlabel('Time (s)')
        ax.set_title('Cleaned ECG')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # If this is a long recording (Sample 2), create an extra-wide scrollable view
        if duration > 20 and "Sample2" in title:
            self._page_ecg_signal_wide(pdf, plt, ecg_result, title)
    
    def _page_ecg_signal_wide(self, pdf, plt, ecg_result, title):
        """Create extra-wide landscape page for full ECG signal visualization"""
        raw_mv = ecg_result['raw_mv']
        clean_mv = ecg_result['cleaned_mv']
        r_peaks = ecg_result.get('r_peaks', np.array([]))
        duration = len(raw_mv) / ECG_FS
        t = np.arange(len(raw_mv)) / ECG_FS
        
        # Calculate width based on duration: 1 inch per 2 seconds
        fig_width = max(20, duration / 2)
        fig_width = min(fig_width, 200)  # Cap at 200 inches
        
        fig, axes = plt.subplots(2, 1, figsize=(fig_width, 11))
        fig.suptitle(f'ECG - {title} - FULL SCROLLABLE VIEW', fontsize=16, fontweight='bold')
        
        # Raw ECG
        ax = axes[0]
        self._draw_ecg_grid(ax, duration)
        ax.plot(t, raw_mv, 'k-', lw=1.0, zorder=5)
        if len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < len(raw_mv)]
            ax.scatter(valid_peaks/ECG_FS, raw_mv[valid_peaks], color='red', s=100, marker='v', zorder=10)
            # Add ALL R-R intervals on wide view
            for i in range(len(valid_peaks) - 1):
                rr_ms = (valid_peaks[i+1] - valid_peaks[i]) / ECG_FS * 1000
                mid_t = (valid_peaks[i] + valid_peaks[i+1]) / 2 / ECG_FS
                ax.text(mid_t, 1.7, f'{rr_ms:.0f}', fontsize=7, ha='center',
                       bbox=dict(boxstyle='round,pad=0.15', fc='yellow', alpha=0.7))
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('mV', fontsize=12, fontweight='bold')
        ax.set_title('Raw ECG - All R-R Intervals Shown (ms)', fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        
        # Cleaned ECG
        ax = axes[1]
        self._draw_ecg_grid(ax, duration)
        ax.plot(t, clean_mv, 'b-', lw=1.0, zorder=5)
        if len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < len(clean_mv)]
            ax.scatter(valid_peaks/ECG_FS, clean_mv[valid_peaks], color='red', s=100, marker='v', zorder=10)
            # Draw lines between R-peaks
            for i in range(len(valid_peaks) - 1):
                ax.plot([valid_peaks[i]/ECG_FS, valid_peaks[i+1]/ECG_FS], 
                       [1.5, 1.5], 'g-', lw=2, alpha=0.5, zorder=8)
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('mV', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_title('Cleaned ECG', fontsize=12)
        ax.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"[PDF] Created wide ECG page: {fig_width:.1f} inches x 11 inches")


    def _page_ecg_morphology(self, pdf, plt, ecg_result):
        # [Same as before - showing single beat, average beat, RR tachogram, Poincaré]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'ECG Morphology - {ecg_result["sample_name"]}', fontsize=14, fontweight='bold')

        clean_mv = ecg_result['cleaned_mv']
        r_peaks = ecg_result.get('r_peaks', np.array([]))
        waves = ecg_result.get('waves', {})

        # Single beat
        ax = axes[0, 0]
        if len(r_peaks) >= 2:
            idx = len(r_peaks) // 2
            r = r_peaks[idx]
            pre, post = int(0.3 * ECG_FS), int(0.4 * ECG_FS)
            start, end = max(0, r - pre), min(len(clean_mv), r + post)
            beat = clean_mv[start:end]
            t_beat = (np.arange(len(beat)) - (r - start)) / ECG_FS * 1000
            ax.plot(t_beat, beat, 'b-', lw=2)
            ax.axvline(0, color='red', linestyle='--', lw=1.5, alpha=0.7)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('mV')
            ax.set_title('Single Beat')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # Average beat
        ax = axes[0, 1]
        if len(r_peaks) >= 5:
            beats = []
            for rp in r_peaks[1:-1]:
                pre, post = int(0.25 * ECG_FS), int(0.35 * ECG_FS)
                if rp - pre >= 0 and rp + post < len(clean_mv):
                    beats.append(clean_mv[rp-pre:rp+post])
            if len(beats) >= 3:
                min_len = min(len(b) for b in beats)
                beats_arr = np.array([b[:min_len] for b in beats])
                mean_beat = np.mean(beats_arr, axis=0)
                std_beat = np.std(beats_arr, axis=0)
                t_avg = (np.arange(min_len) - int(0.25 * ECG_FS)) / ECG_FS * 1000
                ax.fill_between(t_avg, mean_beat-std_beat, mean_beat+std_beat, alpha=0.3, color='blue')
                ax.plot(t_avg, mean_beat, 'b-', lw=2)
                ax.axvline(0, color='red', linestyle='--', lw=1, alpha=0.7)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('mV')
                ax.set_title(f'Average Beat (n={len(beats)})')
                ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # RR tachogram
        ax = axes[1, 0]
        rr_ms = ecg_result.get('rr_intervals_ms', np.array([]))
        if len(rr_ms) >= 2:
            ax.plot(rr_ms, 'b-o', markersize=6, lw=1.5)
            ax.axhline(np.mean(rr_ms), color='red', linestyle='--', lw=2)
            ax.set_xlabel('Beat #')
            ax.set_ylabel('RR (ms)')
            ax.set_title('RR Tachogram')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # Poincaré plot
        ax = axes[1, 1]
        if len(rr_ms) >= 3:
            rr1, rr2 = rr_ms[:-1], rr_ms[1:]
            ax.scatter(rr1, rr2, alpha=0.7, s=50, c='blue')
            ax.set_xlabel('RR(n) ms')
            ax.set_ylabel('RR(n+1) ms')
            ax.set_title('Poincaré Plot')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_hrv(self, pdf, plt, ecg_result):
        # [Same as before - time domain, frequency domain, nonlinear]
        hrv_time = ecg_result.get('hrv_time', {})
        hrv_freq = ecg_result.get('hrv_freq', {})
        hrv_nl = ecg_result.get('hrv_nonlinear', {})

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'HRV Analysis - {ecg_result["sample_name"]}', fontsize=14, fontweight='bold')

        # Time domain
        ax = axes[0, 0]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Time Domain', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        y = 0.88
        for key in ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_MedianNN']:
            if key in hrv_time:
                name = METRIC_EXPLANATIONS.get(key, (key, ''))[0]
                ax.text(0.05, y, f'{name}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.65, y, f'{hrv_time[key]:.2f}', fontsize=10, fontfamily='monospace',
                       fontweight='bold', transform=ax.transAxes)
                y -= 0.1

        # Frequency domain
        ax = axes[0, 1]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Frequency Domain', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        y = 0.88
        if hrv_freq:
            for key in ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']:
                if key in hrv_freq:
                    name = METRIC_EXPLANATIONS.get(key, (key, ''))[0]
                    ax.text(0.05, y, f'{name}:', fontsize=10, transform=ax.transAxes)
                    ax.text(0.65, y, f'{hrv_freq[key]:.2f}', fontsize=10, fontfamily='monospace',
                           fontweight='bold', transform=ax.transAxes)
                    y -= 0.1
        else:
            ax.text(0.5, 0.5, 'Requires 60+ seconds', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11, color='gray')

        # RR histogram
        ax = axes[1, 0]
        rr_ms = ecg_result.get('rr_intervals_ms', np.array([]))
        if len(rr_ms) >= 3:
            ax.hist(rr_ms, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(rr_ms), color='red', linestyle='--', lw=2)
            ax.set_xlabel('RR Interval (ms)')
            ax.set_ylabel('Count')
            ax.set_title('RR Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # Nonlinear
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Nonlinear', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        y = 0.88
        if hrv_nl:
            for key in ['HRV_SD1', 'HRV_SD2', 'HRV_ApEn', 'HRV_DFA_alpha1']:
                if key in hrv_nl:
                    name = METRIC_EXPLANATIONS.get(key, (key, ''))[0]
                    ax.text(0.05, y, f'{name}:', fontsize=10, transform=ax.transAxes)
                    ax.text(0.65, y, f'{hrv_nl[key]:.4f}', fontsize=10, fontfamily='monospace',
                           fontweight='bold', transform=ax.transAxes)
                    y -= 0.1

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ecg_comparison(self, pdf, plt, ecg_s1, ecg_s2):
        """Compare ECG Sample 1 vs Sample 2"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('ECG Comparison: Sample 1 vs Sample 2', fontsize=14, fontweight='bold')

        # Compare heart rate
        ax = axes[0, 0]
        hr1 = ecg_s1.get('heart_rate', {})
        hr2 = ecg_s2.get('heart_rate', {})
        if hr1 and hr2:
            labels = ['Sample 1\n(Best)', 'Sample 2\n(Full)']
            means = [hr1['mean'], hr2['mean']]
            stds = [hr1['std'], hr2['std']]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=10, color=['#4CAF50', '#2196F3'], alpha=0.7)
            ax.set_ylabel('Heart Rate (bpm)')
            ax.set_title('Heart Rate Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')

        # Compare HRV metrics
        ax = axes[0, 1]
        hrv1 = ecg_s1.get('hrv_time', {})
        hrv2 = ecg_s2.get('hrv_time', {})
        if hrv1 and hrv2:
            metrics = ['HRV_SDNN', 'HRV_RMSSD']
            s1_vals = [hrv1.get(m, 0) for m in metrics]
            s2_vals = [hrv2.get(m, 0) for m in metrics]
            x = np.arange(len(metrics))
            width = 0.35
            ax.bar(x - width/2, s1_vals, width, label='Sample 1', color='#4CAF50', alpha=0.7)
            ax.bar(x + width/2, s2_vals, width, label='Sample 2', color='#2196F3', alpha=0.7)
            ax.set_ylabel('ms')
            ax.set_title('HRV Time Domain')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('HRV_', '') for m in metrics])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')

        # Compare intervals
        ax = axes[1, 0]
        int1 = ecg_s1.get('intervals', {})
        int2 = ecg_s2.get('intervals', {})
        if int1 and int2:
            metrics = ['PR_Interval', 'QRS_Duration', 'QTc']
            available = [m for m in metrics if m in int1 and m in int2]
            if available:
                s1_vals = [int1[m] for m in available]
                s2_vals = [int2[m] for m in available]
                x = np.arange(len(available))
                width = 0.35
                ax.bar(x - width/2, s1_vals, width, label='Sample 1', color='#4CAF50', alpha=0.7)
                ax.bar(x + width/2, s2_vals, width, label='Sample 2', color='#2196F3', alpha=0.7)
                ax.set_ylabel('ms')
                ax.set_title('ECG Intervals')
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('_', ' ') for m in available], rotation=15)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')

        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.5, 0.95, 'Key Differences', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        
        y = 0.85
        text_items = []
        
        if hr1 and hr2:
            hr_diff = abs(hr1['mean'] - hr2['mean'])
            text_items.append(f"HR Difference: {hr_diff:.1f} bpm")
        
        if hrv1 and hrv2:
            if 'HRV_SDNN' in hrv1 and 'HRV_SDNN' in hrv2:
                sdnn_diff = abs(hrv1['HRV_SDNN'] - hrv2['HRV_SDNN'])
                text_items.append(f"SDNN Difference: {sdnn_diff:.1f} ms")
        
        text_items.append(f"\nSample 1: {ecg_s1['duration_sec']:.1f}s")
        text_items.append(f"Sample 2: {ecg_s2['duration_sec']:.1f}s")
        text_items.append(f"\nR-peaks Sample 1: {len(ecg_s1.get('r_peaks', []))}")
        text_items.append(f"R-peaks Sample 2: {len(ecg_s2.get('r_peaks', []))}")
        
        for item in text_items:
            ax.text(0.1, y, item, fontsize=10, transform=ax.transAxes)
            y -= 0.08

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ppg_signal(self, pdf, plt, ppg_result):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'PPG Signals & Beat Segmentation - {ppg_result["sample_name"]}', fontsize=14, fontweight='bold')

        ir_raw = ppg_result['ir_raw']
        red_raw = ppg_result['red_raw']
        ir_peaks = ppg_result.get('ir_peaks', np.array([]))
        red_peaks = ppg_result.get('red_peaks', np.array([]))

        window = min(len(ir_raw), PPG_FS * 15)
        t = np.arange(window) / PPG_FS

        # IR channel with beat segmentation
        ax = axes[0, 0]
        ax.plot(t, ir_raw[:window], 'purple', lw=0.8)
        peaks_win = ir_peaks[ir_peaks < window]
        if len(peaks_win) > 0:
            ax.scatter(peaks_win/PPG_FS, ir_raw[peaks_win], color='green', s=60, marker='v', zorder=10)
            # Add beat segmentation lines and PP intervals
            for i in range(len(peaks_win) - 1):
                if i < 8:  # Limit annotations
                    # Draw vertical lines to show beat boundaries
                    ax.axvline(peaks_win[i]/PPG_FS, color='cyan', linestyle='--', lw=0.8, alpha=0.5)
                    # Show PP interval
                    pp_ms = (peaks_win[i+1] - peaks_win[i]) / PPG_FS * 1000
                    mid_t = (peaks_win[i] + peaks_win[i+1]) / 2 / PPG_FS
                    y_pos = np.max(ir_raw[:window]) * 0.9
                    ax.text(mid_t, y_pos, f'{pp_ms:.0f}ms', fontsize=7, ha='center',
                           bbox=dict(boxstyle='round,pad=0.15', fc='yellow', alpha=0.8))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ADC')
        ax.set_title(f'IR Channel - {len(ir_peaks)} peaks - Beat Segmentation')
        ax.grid(True, alpha=0.3)

        # Red channel with beat segmentation
        ax = axes[0, 1]
        ax.plot(t, red_raw[:window], 'red', lw=0.8)
        peaks_win = red_peaks[red_peaks < window]
        if len(peaks_win) > 0:
            ax.scatter(peaks_win/PPG_FS, red_raw[peaks_win], color='green', s=60, marker='v', zorder=10)
            # Add beat segmentation
            for i in range(len(peaks_win) - 1):
                if i < 8:
                    ax.axvline(peaks_win[i]/PPG_FS, color='cyan', linestyle='--', lw=0.8, alpha=0.5)
                    pp_ms = (peaks_win[i+1] - peaks_win[i]) / PPG_FS * 1000
                    mid_t = (peaks_win[i] + peaks_win[i+1]) / 2 / PPG_FS
                    y_pos = np.max(red_raw[:window]) * 0.9
                    ax.text(mid_t, y_pos, f'{pp_ms:.0f}ms', fontsize=7, ha='center',
                           bbox=dict(boxstyle='round,pad=0.15', fc='yellow', alpha=0.8))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ADC')
        ax.set_title(f'Red Channel - {len(red_peaks)} peaks - Beat Segmentation')
        ax.grid(True, alpha=0.3)

        # Pulse rate with statistics
        ax = axes[1, 0]
        hr = ppg_result.get('heart_rate')
        if hr and 'values' in hr and len(hr['values']) > 2:
            ax.plot(hr['values'], 'g-o', markersize=4, lw=1)
            ax.axhline(hr['mean'], color='red', linestyle='--', lw=2, label=f"Mean: {hr['mean']:.1f} bpm")
            ax.fill_between(range(len(hr['values'])), hr['mean']-hr['std'], hr['mean']+hr['std'],
                           alpha=0.2, color='green')
            ax.set_xlabel('Beat #')
            ax.set_ylabel('Pulse Rate (bpm)')
            ax.set_title('Pulse Rate from PPG')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        # PP intervals with timing info
        ax = axes[1, 1]
        pp_ms = ppg_result.get('pp_intervals_ms', np.array([]))
        if len(pp_ms) >= 3:
            ax.hist(pp_ms, bins=15, color='purple', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(pp_ms), color='red', linestyle='--', lw=2, 
                      label=f'Mean: {np.mean(pp_ms):.0f}ms')
            ax.axvline(np.median(pp_ms), color='green', linestyle=':', lw=2,
                      label=f'Median: {np.median(pp_ms):.0f}ms')
            stats_text = f'Std: {np.std(pp_ms):.0f}ms\nRange: {np.min(pp_ms):.0f}-{np.max(pp_ms):.0f}ms'
            ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
            ax.set_xlabel('PP Interval (ms)')
            ax.set_ylabel('Count')
            ax.set_title('Peak-to-Peak Interval Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # If this is a long recording (Sample 2), create an extra-wide scrollable view
        if ppg_result['duration_sec'] > 20 and "Sample2" in ppg_result["sample_name"]:
            self._page_ppg_signal_wide(pdf, plt, ppg_result)
    
    def _page_ppg_signal_wide(self, pdf, plt, ppg_result):
        """Create extra-wide landscape page for full PPG signal visualization"""
        ir_raw = ppg_result['ir_raw']
        red_raw = ppg_result['red_raw']
        ir_peaks = ppg_result.get('ir_peaks', np.array([]))
        red_peaks = ppg_result.get('red_peaks', np.array([]))
        duration = ppg_result['duration_sec']
        
        # Calculate width based on duration: 1 inch per 3 seconds for PPG
        fig_width = max(20, duration / 3)
        fig_width = min(fig_width, 200)  # Cap at 200 inches
        
        fig, axes = plt.subplots(2, 1, figsize=(fig_width, 11))
        fig.suptitle(f'PPG Signals - {ppg_result["sample_name"]} - FULL SCROLLABLE VIEW', 
                    fontsize=16, fontweight='bold')
        
        t = np.arange(len(ir_raw)) / PPG_FS
        
        # IR channel with ALL peak annotations
        ax = axes[0]
        ax.plot(t, ir_raw, 'purple', lw=1.0)
        if len(ir_peaks) > 0:
            ax.scatter(ir_peaks/PPG_FS, ir_raw[ir_peaks], color='green', s=100, marker='v', zorder=10)
            # Add ALL PP intervals
            for i in range(len(ir_peaks) - 1):
                pp_ms = (ir_peaks[i+1] - ir_peaks[i]) / PPG_FS * 1000
                mid_t = (ir_peaks[i] + ir_peaks[i+1]) / 2 / PPG_FS
                y_pos = np.max(ir_raw) * 0.95
                ax.text(mid_t, y_pos, f'{pp_ms:.0f}', fontsize=6, ha='center',
                       bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.6))
                # Draw beat segmentation lines
                ax.axvline(ir_peaks[i]/PPG_FS, color='cyan', linestyle='--', lw=0.8, alpha=0.4)
        ax.set_xlim(0, duration)
        ax.set_ylabel('ADC', fontsize=12, fontweight='bold')
        ax.set_title(f'IR Channel - {len(ir_peaks)} peaks - All PP Intervals (ms)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Red channel with peak annotations
        ax = axes[1]
        ax.plot(t, red_raw, 'red', lw=1.0)
        if len(red_peaks) > 0:
            ax.scatter(red_peaks/PPG_FS, red_raw[red_peaks], color='green', s=100, marker='v', zorder=10)
            # Add PP intervals
            for i in range(len(red_peaks) - 1):
                pp_ms = (red_peaks[i+1] - red_peaks[i]) / PPG_FS * 1000
                mid_t = (red_peaks[i] + red_peaks[i+1]) / 2 / PPG_FS
                y_pos = np.max(red_raw) * 0.95
                ax.text(mid_t, y_pos, f'{pp_ms:.0f}', fontsize=6, ha='center',
                       bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.6))
                # Draw beat segmentation lines
                ax.axvline(red_peaks[i]/PPG_FS, color='cyan', linestyle='--', lw=0.8, alpha=0.4)
        ax.set_xlim(0, duration)
        ax.set_ylabel('ADC', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_title(f'Red Channel - {len(red_peaks)} peaks - All PP Intervals (ms)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        print(f"[PDF] Created wide PPG page: {fig_width:.1f} inches x 11 inches")

    def _page_ppg_beat_segmentation(self, pdf, plt, ppg_result):
        """Show individual PPG beats and morphology with improved visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'PPG Beat Morphology & Segmentation - {ppg_result["sample_name"]}', fontsize=14, fontweight='bold')

        ir_raw = ppg_result['ir_raw']
        ir_proc = ppg_result.get('ir_processed', {})
        ir_peaks = ppg_result.get('ir_peaks', np.array([]))
        pp_ms = ppg_result.get('pp_intervals_ms', np.array([]))

        # Panel 1: ALL Individual Beats Overlapping (normalized and aligned)
        ax = axes[0, 0]
        if len(ir_peaks) >= 3:
            # Extract all beats
            beats = []
            for i in range(len(ir_peaks) - 1):
                start_idx = ir_peaks[i]
                end_idx = ir_peaks[i + 1]
                if end_idx - start_idx > 5:  # Valid beat
                    beat = ir_raw[start_idx:end_idx]
                    # Normalize each beat to 0-1 for comparison
                    beat_norm = (beat - np.min(beat)) / (np.max(beat) - np.min(beat) + 1e-10)
                    beats.append((beat_norm, len(beat)))
            
            if len(beats) >= 3:
                # Find median length for alignment
                median_len = int(np.median([b[1] for b in beats]))
                
                # Resample all beats to common length and plot WITH CLEAR OVERLAP
                from scipy.interpolate import interp1d
                n_beats_show = min(30, len(beats))  # Show up to 30 beats for better visualization
                
                for i, (beat_norm, beat_len) in enumerate(beats[:n_beats_show]):
                    # Resample to median length
                    if beat_len != median_len:
                        x_old = np.linspace(0, 1, beat_len)
                        x_new = np.linspace(0, 1, median_len)
                        try:
                            f = interp1d(x_old, beat_norm, kind='cubic')
                            beat_resampled = f(x_new)
                        except:
                            beat_resampled = beat_norm[:median_len] if beat_len >= median_len else np.pad(beat_norm, (0, median_len - beat_len))
                    else:
                        beat_resampled = beat_norm
                    
                    # Use rainbow colors with varying transparency
                    alpha = 0.5 if i < 10 else (0.3 if i < 20 else 0.2)
                    color = plt.cm.rainbow(i / n_beats_show)
                    t_beat = np.arange(median_len) / PPG_FS * 1000
                    ax.plot(t_beat, beat_resampled, color=color, lw=1.8, alpha=alpha, zorder=n_beats_show-i)
                
                ax.set_xlabel('Time (ms)', fontweight='bold')
                ax.set_ylabel('Normalized Amplitude (0-1)', fontweight='bold')
                ax.set_title(f'{n_beats_show} Beats Overlaid - Morphology Consistency', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
                ax.text(0.02, 0.98, f'Total Beats: {len(beats)}\nShowing: {n_beats_show}', 
                       transform=ax.transAxes, fontsize=8, ha='left', va='top',
                       bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'Insufficient beats', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Insufficient peaks', ha='center', va='center', transform=ax.transAxes)

        # Panel 2: Average Beat with ALL Individual Beats (semi-transparent) - ACTUAL ADC VALUES
        ax = axes[0, 1]
        if len(ir_peaks) >= 5:
            beats_raw = []
            for i in range(len(ir_peaks) - 1):
                start_idx = ir_peaks[i]
                end_idx = ir_peaks[i + 1]
                if end_idx - start_idx > 5:
                    beat = ir_raw[start_idx:end_idx]
                    beats_raw.append(beat)
            
            if len(beats_raw) >= 3:
                # Find minimum length
                min_len = min(len(b) for b in beats_raw)
                beats_arr = np.array([b[:min_len] for b in beats_raw])
                
                # Plot ALL individual beats in background (very transparent)
                t_beat = np.arange(min_len) / PPG_FS * 1000
                n_beats_show = min(50, len(beats_arr))
                for i in range(n_beats_show):
                    alpha = 0.08 if i >= 20 else 0.12
                    ax.plot(t_beat, beats_arr[i], color='gray', lw=0.6, alpha=alpha, zorder=1)
                
                # Calculate and plot mean ± SD (bold)
                mean_beat = np.mean(beats_arr, axis=0)
                std_beat = np.std(beats_arr, axis=0)
                
                ax.fill_between(t_beat, mean_beat - std_beat, mean_beat + std_beat, 
                               alpha=0.4, color='purple', label='±1 SD', zorder=2)
                ax.plot(t_beat, mean_beat, 'purple', lw=3.5, label='Mean Beat', zorder=3)
                
                # Mark systolic and diastolic points
                peak_idx = np.argmax(mean_beat)
                valley_idx = np.argmin(mean_beat)
                ax.plot(t_beat[peak_idx], mean_beat[peak_idx], 'ro', markersize=12, 
                       label='Systolic Peak', zorder=4)
                ax.plot(t_beat[valley_idx], mean_beat[valley_idx], 'bs', markersize=12,
                       label='Diastolic Valley', zorder=4)
                
                ax.set_xlabel('Time (ms)', fontweight='bold')
                ax.set_ylabel('ADC Value', fontweight='bold')
                ax.set_title(f'Mean Beat with All {len(beats_arr)} Beats Overlaid', fontweight='bold')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
            else:
                ax.axis('off')
        else:
            ax.axis('off')

        # Panel 3: Beat Morphology Metrics
        ax = axes[1, 0]
        if len(ir_peaks) >= 3:
            # Calculate beat characteristics
            beat_metrics = {
                'amplitudes': [],
                'widths_50': [],  # Width at 50% amplitude
                'rise_times': [],
                'fall_times': []
            }
            
            for i in range(len(ir_peaks) - 1):
                start_idx = ir_peaks[i]
                end_idx = ir_peaks[i + 1]
                if end_idx - start_idx > 5:
                    beat = ir_raw[start_idx:end_idx]
                    
                    # Amplitude
                    amplitude = np.max(beat) - np.min(beat)
                    beat_metrics['amplitudes'].append(amplitude)
                    
                    # Width at 50% (pulse width)
                    min_val = np.min(beat)
                    max_val = np.max(beat)
                    half_amp = min_val + (max_val - min_val) * 0.5
                    
                    # Find indices where signal crosses 50%
                    above_half = beat < half_amp  # Inverted PPG
                    if np.sum(above_half) > 0:
                        indices = np.where(above_half)[0]
                        if len(indices) > 1:
                            width = (indices[-1] - indices[0]) / PPG_FS * 1000  # ms
                            beat_metrics['widths_50'].append(width)
            
            # Plot metrics if available
            if beat_metrics['amplitudes']:
                metrics_to_plot = []
                labels = []
                
                if beat_metrics['amplitudes']:
                    metrics_to_plot.append(beat_metrics['amplitudes'])
                    labels.append(f'Amplitude\n{np.mean(beat_metrics["amplitudes"]):.0f}±{np.std(beat_metrics["amplitudes"]):.0f}')
                
                if beat_metrics['widths_50']:
                    metrics_to_plot.append(beat_metrics['widths_50'])
                    labels.append(f'Width@50%\n{np.mean(beat_metrics["widths_50"]):.0f}±{np.std(beat_metrics["widths_50"]):.0f}ms')
                
                if pp_ms is not None and len(pp_ms) > 0:
                    metrics_to_plot.append(pp_ms)
                    labels.append(f'PP Interval\n{np.mean(pp_ms):.0f}±{np.std(pp_ms):.0f}ms')
                
                # Create box plot
                bp = ax.boxplot(metrics_to_plot, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
                    patch.set_facecolor(color)
                
                ax.set_ylabel('Value')
                ax.set_title('Beat Morphology Metrics')
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(axis='x', labelsize=8)
            else:
                ax.axis('off')
        else:
            ax.axis('off')

        # Panel 4: PP Interval Variability with Quality Indicators
        ax = axes[1, 1]
        if len(pp_ms) >= 3:
            beat_numbers = np.arange(1, len(pp_ms) + 1)
            
            # Color code by variability
            colors = []
            threshold = np.std(pp_ms)
            mean_pp = np.mean(pp_ms)
            for val in pp_ms:
                if abs(val - mean_pp) < threshold * 0.5:
                    colors.append('green')  # Good consistency
                elif abs(val - mean_pp) < threshold:
                    colors.append('yellow')  # Moderate variation
                else:
                    colors.append('red')  # High variation
            
            ax.scatter(beat_numbers, pp_ms, c=colors, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
            ax.plot(beat_numbers, pp_ms, 'k-', lw=0.8, alpha=0.3)
            ax.axhline(mean_pp, color='blue', linestyle='--', lw=2, label=f'Mean: {mean_pp:.0f}ms')
            ax.fill_between(beat_numbers, mean_pp - np.std(pp_ms), mean_pp + np.std(pp_ms),
                           alpha=0.2, color='blue', label=f'±1 SD')
            
            # Add coefficient of variation
            cv = (np.std(pp_ms) / np.mean(pp_ms)) * 100
            ax.text(0.02, 0.98, f'CV: {cv:.1f}%\nGreen: Consistent\nYellow: Moderate\nRed: Variable', 
                   transform=ax.transAxes, fontsize=8, va='top',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.9))
            
            ax.set_xlabel('Beat Number')
            ax.set_ylabel('PP Interval (ms)')
            ax.set_title('Beat-to-Beat Interval Variability')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ppg_analysis(self, pdf, plt, ppg_result):
        # [SpO2, perfusion, PRV metrics]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(f'PPG Analysis - {ppg_result["sample_name"]}', fontsize=14, fontweight='bold')

        # SpO2
        ax = axes[0, 0]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Oxygen Saturation', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        spo2 = ppg_result.get('spo2')
        if spo2:
            color = 'green' if spo2 >= 95 else ('orange' if spo2 >= 90 else 'red')
            ax.text(0.5, 0.65, f'{spo2}%', fontsize=48, fontweight='bold', ha='center',
                   transform=ax.transAxes, color=color)
            ax.text(0.5, 0.4, 'SpO2', fontsize=12, ha='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')

        # Perfusion & Quality
        ax = axes[0, 1]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Signal Quality', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        y = 0.82
        items = []
        if ppg_result.get('perfusion_index'):
            items.append(('Perfusion Index', f'{ppg_result["perfusion_index"]:.2f}%'))
        if ppg_result.get('signal_quality'):
            items.append(('Signal Quality', f'{ppg_result["signal_quality"]:.3f}'))
        hr = ppg_result.get('heart_rate')
        if hr:
            items.append(('Pulse Rate', f'{hr["mean"]:.1f} ± {hr["std"]:.1f} bpm'))
        
        for label, value in items:
            ax.text(0.1, y, f'{label}:', fontsize=10, transform=ax.transAxes)
            ax.text(0.6, y, value, fontsize=10, fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
            y -= 0.08

        # PRV Time Domain
        ax = axes[1, 0]
        ax.axis('off')
        ax.text(0.5, 0.98, 'PRV Time Domain', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        prv_time = ppg_result.get('prv_time', {})
        y = 0.85
        if prv_time:
            for key, val in list(prv_time.items())[:5]:
                ax.text(0.1, y, f'{key}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.6, y, f'{val:.2f}', fontsize=10, fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
                y -= 0.1
        else:
            ax.text(0.5, 0.5, 'Requires 5+ peaks', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11, color='gray')

        # Poincaré plot
        ax = axes[1, 1]
        pp_ms = ppg_result.get('pp_intervals_ms', np.array([]))
        if len(pp_ms) >= 3:
            pp1, pp2 = pp_ms[:-1], pp_ms[1:]
            ax.scatter(pp1, pp2, alpha=0.7, s=50, c='purple')
            ax.set_xlabel('PP(n) ms')
            ax.set_ylabel('PP(n+1) ms')
            ax.set_title('PPG Poincaré Plot')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ppg_comparison(self, pdf, plt, ppg_s1, ppg_s2):
        """Compare PPG Sample 1 vs Sample 2"""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('PPG Comparison: Sample 1 vs Sample 2', fontsize=14, fontweight='bold')

        # Compare pulse rate
        ax = axes[0, 0]
        hr1 = ppg_s1.get('heart_rate', {})
        hr2 = ppg_s2.get('heart_rate', {})
        if hr1 and hr2:
            labels = ['Sample 1\n(Best)', 'Sample 2\n(Full)']
            means = [hr1['mean'], hr2['mean']]
            stds = [hr1['std'], hr2['std']]
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=stds, capsize=10, color=['#4CAF50', '#2196F3'], alpha=0.7)
            ax.set_ylabel('Pulse Rate (bpm)')
            ax.set_title('Pulse Rate Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')

        # Compare SpO2
        ax = axes[0, 1]
        spo2_1 = ppg_s1.get('spo2')
        spo2_2 = ppg_s2.get('spo2')
        if spo2_1 and spo2_2:
            labels = ['Sample 1', 'Sample 2']
            values = [spo2_1, spo2_2]
            x = np.arange(len(labels))
            colors = ['green' if v >= 95 else 'orange' if v >= 90 else 'red' for v in values]
            ax.bar(x, values, color=colors, alpha=0.7)
            ax.set_ylabel('SpO2 (%)')
            ax.set_title('SpO2 Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim([80, 100])
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')

        # Compare PRV metrics
        ax = axes[1, 0]
        prv1 = ppg_s1.get('prv_time', {})
        prv2 = ppg_s2.get('prv_time', {})
        if prv1 and prv2:
            metrics = ['PRV_SDNN', 'PRV_RMSSD']
            available = [m for m in metrics if m in prv1 and m in prv2]
            if available:
                s1_vals = [prv1[m] for m in available]
                s2_vals = [prv2[m] for m in available]
                x = np.arange(len(available))
                width = 0.35
                ax.bar(x - width/2, s1_vals, width, label='Sample 1', color='#4CAF50', alpha=0.7)
                ax.bar(x + width/2, s2_vals, width, label='Sample 2', color='#2196F3', alpha=0.7)
                ax.set_ylabel('ms')
                ax.set_title('PRV Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('PRV_', '') for m in available])
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')

        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.5, 0.95, 'Key Differences', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        
        y = 0.85
        text_items = []
        
        if hr1 and hr2:
            pr_diff = abs(hr1['mean'] - hr2['mean'])
            text_items.append(f"Pulse Rate Diff: {pr_diff:.1f} bpm")
        
        if spo2_1 and spo2_2:
            spo2_diff = abs(spo2_1 - spo2_2)
            text_items.append(f"SpO2 Difference: {spo2_diff:.0f}%")
        
        text_items.append(f"\nSample 1: {ppg_s1['duration_sec']:.1f}s")
        text_items.append(f"Sample 2: {ppg_s2['duration_sec']:.1f}s")
        text_items.append(f"\nIR peaks Sample 1: {len(ppg_s1.get('ir_peaks', []))}")
        text_items.append(f"IR peaks Sample 2: {len(ppg_s2.get('ir_peaks', []))}")
        
        for item in text_items:
            ax.text(0.1, y, item, fontsize=10, transform=ax.transAxes)
            y -= 0.08

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_summary(self, pdf, plt, ecg_s1, ecg_s2, ppg_s1, ppg_s2):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.97, 'Summary - All Samples', fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
        
        y = 0.90
        
        # ECG Sample 1
        ax.text(0.05, y, 'ECG Sample 1 (Best 5s Window - Basic Analysis):', fontsize=13, fontweight='bold', transform=ax.transAxes, color='#2060a0')
        y -= 0.03
        if ecg_s1:
            hr1 = ecg_s1.get('heart_rate', {})
            items = [
                ('Duration', f'{ecg_s1["duration_sec"]:.1f}s'),
                ('R-peaks', f'{len(ecg_s1.get("r_peaks", []))}'),
                ('Analysis Type', 'Time-domain only'),
            ]
            if hr1:
                items.append(('Heart Rate', f'{hr1["mean"]:.1f} ± {hr1["std"]:.1f} bpm'))
            for label, value in items:
                ax.text(0.07, y, f'{label}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.45, y, value, fontsize=10, fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
        
        # ECG Sample 2
        if ecg_s2:
            y -= 0.02
            ax.text(0.05, y, 'ECG Sample 2 (Full Recording - Complete Analysis):', fontsize=13, fontweight='bold', transform=ax.transAxes, color='#2060a0')
            y -= 0.03
            hr2 = ecg_s2.get('heart_rate', {})
            items = [
                ('Duration', f'{ecg_s2["duration_sec"]:.1f}s'),
                ('R-peaks', f'{len(ecg_s2.get("r_peaks", []))}'),
                ('Analysis Type', 'Time + Freq + Nonlinear'),
            ]
            if hr2:
                items.append(('Heart Rate', f'{hr2["mean"]:.1f} ± {hr2["std"]:.1f} bpm'))
            hrv_freq = ecg_s2.get('hrv_freq', {})
            if hrv_freq:
                items.append(('Long-term HRV', '✓ Available'))
            for label, value in items:
                ax.text(0.07, y, f'{label}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.45, y, value, fontsize=10, fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
        
        # PPG Sample 1
        if ppg_s1:
            y -= 0.02
            ax.text(0.05, y, 'PPG Sample 1 (Full Recording - Basic Analysis):', fontsize=13, fontweight='bold', transform=ax.transAxes, color='#a02060')
            y -= 0.03
            items = [
                ('Duration', f'{ppg_s1["duration_sec"]:.1f}s'),
                ('IR Peaks', f'{len(ppg_s1.get("ir_peaks", []))}'),
                ('Analysis Type', 'Time-domain only'),
            ]
            if ppg_s1.get('spo2'):
                items.append(('SpO2', f'{ppg_s1["spo2"]}%'))
            for label, value in items:
                ax.text(0.07, y, f'{label}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.45, y, value, fontsize=10, fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
        
        # PPG Sample 2
        if ppg_s2:
            y -= 0.02
            ax.text(0.05, y, 'PPG Sample 2 (Full Recording - Complete Analysis):', fontsize=13, fontweight='bold', transform=ax.transAxes, color='#a02060')
            y -= 0.03
            items = [
                ('Duration', f'{ppg_s2["duration_sec"]:.1f}s'),
                ('IR Peaks', f'{len(ppg_s2.get("ir_peaks", []))}'),
                ('Analysis Type', 'Time + Freq + Nonlinear'),
            ]
            if ppg_s2.get('spo2'):
                items.append(('SpO2', f'{ppg_s2["spo2"]}%'))
            prv_freq = ppg_s2.get('prv_freq', {})
            if prv_freq:
                items.append(('Long-term PRV', '✓ Available'))
            for label, value in items:
                ax.text(0.07, y, f'{label}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.45, y, value, fontsize=10, fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_explanations(self, pdf, plt, ecg_result, ppg_result):
        # [Same as before - metric explanations]
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.97, 'Understanding Your Results', fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)

        metrics_present = set()
        metrics_present.update(ecg_result.get('hrv_time', {}).keys())
        metrics_present.update(ecg_result.get('hrv_freq', {}).keys())
        if ppg_result:
            if ppg_result.get('spo2'):
                metrics_present.add('SpO2')
            metrics_present.update(ppg_result.get('prv_time', {}).keys())

        y = 0.88
        count = 0
        for key in ['SpO2', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_LFHF', 'HRV_DFA_alpha1', 'PRV_SDNN']:
            if key in metrics_present and key in METRIC_EXPLANATIONS and count < 12:
                name, explanation = METRIC_EXPLANATIONS[key]
                ax.text(0.03, y, f'• {name}:', fontsize=10, fontweight='bold', transform=ax.transAxes)
                y -= 0.022
                ax.text(0.05, y, explanation[:85], fontsize=9, transform=ax.transAxes, color='#333333')
                y -= 0.035
                count += 1

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


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
        self.weight.setPlaceholderText("Optional")
        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("Optional")
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
        self.setWindowTitle("NirogScan v4.3 - Dual Sample Analysis")
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

        plots = QWidget()
        plot_layout = QVBoxLayout(plots)
        plot_layout.setSpacing(5)

        self.ecg_plot = pg.PlotWidget(title=f"ECG ({ECG_FS} Hz)")
        self.ecg_plot.setBackground('#1e1e1e')
        self.ecg_plot.setLabel('left', 'µV')
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ecg_plot.setYRange(-1500, 1500)
        self.ecg_curve = self.ecg_plot.plot(pen=pg.mkPen('#00ff00', width=1))
        plot_layout.addWidget(self.ecg_plot, stretch=2)

        self.ppg_red_plot = pg.PlotWidget(title=f"PPG Red ({PPG_FS} Hz)")
        self.ppg_red_plot.setBackground('#1e1e1e')
        self.ppg_red_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_red_curve = self.ppg_red_plot.plot(pen=pg.mkPen('#ff0000', width=1))
        plot_layout.addWidget(self.ppg_red_plot, stretch=1)

        self.ppg_ir_plot = pg.PlotWidget(title=f"PPG IR ({PPG_FS} Hz)")
        self.ppg_ir_plot.setBackground('#1e1e1e')
        self.ppg_ir_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ppg_ir_curve = self.ppg_ir_plot.plot(pen=pg.mkPen('#ff00ff', width=1))
        plot_layout.addWidget(self.ppg_ir_plot, stretch=1)

        main.addWidget(plots, stretch=1)

        panel = QWidget()
        panel.setFixedWidth(280)
        panel_layout = QVBoxLayout(panel)

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

        status = QGroupBox("Status")
        sl = QGridLayout()
        font = QFont()
        font.setBold(True)
        labels = [("Packets:", "pkt_lbl", "0"), ("Dropped:", "drop_lbl", "0"), ("CRC:", "crc_lbl", "0"),
                 ("Battery:", "bat_lbl", "--"), ("Temp:", "temp_lbl", "--"), ("Leads:", "lead_lbl", "--")]
        for i, (txt, attr, val) in enumerate(labels):
            sl.addWidget(QLabel(txt), i, 0)
            lbl = QLabel(val)
            lbl.setFont(font)
            setattr(self, attr, lbl)
            sl.addWidget(lbl, i, 1)
        status.setLayout(sl)
        panel_layout.addWidget(status)

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

        rpt = QGroupBox("Report")
        rl = QVBoxLayout()
        self.rpt_btn = QPushButton("Generate Report")
        self.rpt_btn.clicked.connect(self._on_report)
        rl.addWidget(self.rpt_btn)
        nk_txt = f"NeuroKit2: v{NK_VERSION}" if NEUROKIT_AVAILABLE else "NeuroKit2: NOT INSTALLED"
        nk_lbl = QLabel(nk_txt)
        nk_lbl.setStyleSheet(f"color: {'#5f5' if NEUROKIT_AVAILABLE else '#f55'}; font-size: 9px;")
        rl.addWidget(nk_lbl)
        rpt.setLayout(rl)
        panel_layout.addWidget(rpt)

        panel_layout.addStretch()
        main.addWidget(panel)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Dual Sample Analysis Mode")

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
        if p.red[0] > 0:
            self.ppg_red_buffer.append(p.red[0])
            self.ppg_ir_buffer.append(p.ir[0])
        with self.data_lock:
            self.last_packet = p
        if self.logging:
            try:
                self.log_queue.put_nowait({'type': 'ecg', 'samples': ecg_uv})
                if p.red[0] > 0:
                    self.log_queue.put_nowait({'type': 'ppg_red', 'values': list(p.red)})
                    self.log_queue.put_nowait({'type': 'ppg_ir', 'values': list(p.ir)})
                self.log_queue.put_nowait({'type': 'vitals', 'battery_v': p.battery_v, 
                                          'battery_pct': p.battery_pct, 'temp': p.temp, 'leads': p.leads})
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
            os.path.join(self.save_dir, os.path.basename(dlg.selected) + "_report.pdf"), "PDF (*.pdf)")
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