#!/usr/bin/env python3
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

from scipy.signal import butter, filtfilt, find_peaks as scipy_find_peaks, detrend, hilbert
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

# ============================================================================
# NEW FLAG FOR PPG DELINEATION (v5.5 addition)
# ============================================================================
HAS_PPG_DELINEATE = hasattr(nk, 'ppg_delineate') if NEUROKIT_AVAILABLE else False

# ============================================================================
# CONSTANTS
# ============================================================================

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
MIN_RPEAKS = MIN_RPEAKS_REQUIRED  # alias for old code compatibility

# NEW CONSTANT FOR PPG PEAKS (v5.5 addition)
MIN_PPG_PEAKS = 5

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
    'HRV_SDNN': ('Heart Rate Variability', 'How much heartbeat timing varies. Higher = healthier heart. Normal: 50-100ms.'),
    'HRV_RMSSD': ('Short-term HRV', 'Beat-to-beat changes. Reflects relaxation nervous system. Normal: 20-50ms.'),
    'HRV_pNN50': ('Large Beat Changes %', 'Percentage of beats differing >50ms. Higher = good health. Normal: 3-25%.'),
    'HRV_pNN20': ('Small Beat Changes %', 'Percentage of beats differing >20ms. More sensitive measure.'),
    'HRV_SDSD': ('Successive Diff SD', 'Standard deviation of beat-to-beat differences.'),
    'HRV_MedianNN': ('Median RR Interval', 'Middle value of RR intervals, less affected by outliers.'),
    'HRV_VLF': ('Very Low Freq Power', 'Long-term regulation, thermoregulation. Needs 5+ min recording.'),
    'HRV_LF': ('Low Freq Power', 'Mix of stress and relaxation activity. Normal: 400-1500 ms2.'),
    'HRV_HF': ('High Freq Power', 'Relaxation/parasympathetic activity. Normal: 150-400 ms2.'),
    'HRV_LFHF': ('LF/HF Ratio', 'Stress vs relaxation balance. Normal: 1.5-2.0. Higher = more stress.'),
    'HRV_SD1': ('Poincare SD1', 'Short-term variability. Related to RMSSD.'),
    'HRV_SD2': ('Poincare SD2', 'Long-term variability. Related to SDNN.'),
    'HRV_ApEn': ('Approximate Entropy', 'Signal complexity. Lower = more regular heartbeat.'),
    'HRV_SampEn': ('Sample Entropy', 'Heart rhythm complexity measure.'),
    'HRV_DFA_alpha1': ('DFA Short-term', 'Fractal scaling. Normal: 0.75-1.25.'),
    'PR_Interval': ('PR Interval', 'Atrial to ventricular time. Normal: 120-200ms.'),
    'QRS_Duration': ('QRS Duration', 'Ventricular contraction time. Normal: 80-120ms.'),
    'QT_Interval': ('QT Interval', 'Total ventricular activity. Varies with heart rate.'),
    'QTc': ('Corrected QT', 'Heart rate adjusted QT. Normal: <440ms.'),
    'SpO2': ('Oxygen Saturation', 'Blood oxygen level. Normal: 95-100%.'),
    'Perfusion_Index': ('Perfusion Index', 'Blood flow strength. Higher = better circulation.'),
    'PPG_HR': ('Pulse Rate', 'Heart rate from blood flow. Should match ECG heart rate.'),
    'PRV_SDNN': ('Pulse Rate Variability', 'Variability from PPG. Similar to HRV from ECG.'),
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


# MODIFIED: lowpass filter now uses order 2 (from v5.5) instead of order 4
def lowpass_filter(signal: np.ndarray, cutoff: float, fs: int, order: int = 2) -> np.ndarray:
    nyq = fs / 2.0
    normalized_cutoff = min(0.999, cutoff / nyq)
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, signal, padlen=min(len(signal)-1, 3*max(len(a), len(b))))


# ============================================================================
# NEW ROBUST PPG PEAK DETECTION (from v5.5)
# ============================================================================
def robust_ppg_peaks(signal, fs):
    """Detect peaks in PPG signal, trying both polarities."""
    min_dist = int(0.3 * fs)
    prom = (np.percentile(signal, 95) - np.percentile(signal, 5)) * 0.1
    peaks, _ = scipy_find_peaks(signal, distance=min_dist, prominence=prom)
    if len(peaks) < 3:
        peaks, _ = scipy_find_peaks(-signal, distance=min_dist, prominence=prom)
    return np.array(peaks)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

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


# ============================================================================
# BLE MANAGER
# ============================================================================

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


# ============================================================================
# CIRCULAR BUFFER
# ============================================================================

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


# ============================================================================
# LOG THREAD
# ============================================================================

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


# ============================================================================
# SIGNAL QUALITY ANALYZER
# ============================================================================

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
    def find_best_window(ecg_data: np.ndarray, fs: int,
                         min_rpeaks: int = MIN_RPEAKS_REQUIRED,
                         window_sec: float = 5.0,
                         step_sec: float = 0.25) -> Tuple[int, int, float, int]:
        window_samples = int(window_sec * fs)
        step_samples = int(step_sec * fs)

        print(f"[WINDOW] Searching in {len(ecg_data)} samples ({len(ecg_data)/fs:.1f}s)")
        print(f"[WINDOW] Window: {window_sec}s, step: {step_sec}s, min R-peaks: {min_rpeaks}")

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
                rr_bonus = 0
                if NEUROKIT_AVAILABLE and rpeak_count >= 3:
                    try:
                        cleaned = nk.ecg_clean(segment, sampling_rate=fs)
                        _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
                        peaks = rpeaks_info.get('ECG_R_Peaks', np.array([]))
                        if len(peaks) >= 2:
                            rr = np.diff(peaks) / fs * 1000
                            rr_valid = rr[(rr > 300) & (rr < 2000)]
                            if len(rr_valid) >= 2:
                                cv = np.std(rr_valid) / np.mean(rr_valid)
                                rr_bonus = (1.0 - min(1.0, cv)) * 10
                    except Exception:
                        pass
                candidates.append((start, end, snr, rpeak_count, snr + rr_bonus))

        if not candidates:
            for min_req in range(min_rpeaks - 1, 1, -1):
                for start in range(0, len(ecg_data) - window_samples + 1, step_samples):
                    end = start + window_samples
                    segment = ecg_data[start:end]
                    rpeak_count = SignalQualityAnalyzer.count_rpeaks_in_segment(segment, fs)
                    if rpeak_count >= min_req:
                        snr = SignalQualityAnalyzer.calculate_snr(segment, fs)
                        candidates.append((start, end, snr, rpeak_count, snr))
                if candidates:
                    break

        if not candidates:
            snr = SignalQualityAnalyzer.calculate_snr(ecg_data[:window_samples], fs)
            rpeaks = SignalQualityAnalyzer.count_rpeaks_in_segment(ecg_data[:window_samples], fs)
            return 0, window_samples, snr, rpeaks

        candidates.sort(key=lambda x: x[4], reverse=True)
        best = candidates[0]
        print(f"[WINDOW] BEST: {best[0]/fs:.2f}s-{best[1]/fs:.2f}s, SNR: {best[2]:.1f}dB, R-peaks: {best[3]}")
        return best[0], best[1], best[2], best[3]

    @staticmethod
    def find_best_ecg_window(ecg_data: np.ndarray, fs: int,
                              min_rpeaks: int = MIN_RPEAKS_REQUIRED,
                              window_sec: float = 5.0,
                              step_sec: float = 0.25) -> Tuple[int, int, float, int]:
        return SignalQualityAnalyzer.find_best_window(
            ecg_data, fs, min_rpeaks, window_sec, step_sec)


# ============================================================================
# PPG PREPROCESSOR (unchanged – not used in report generation)
# ============================================================================

class PPGPreprocessor:
    @staticmethod
    def preprocess(raw_signal: np.ndarray, fs: int = PPG_FS) -> Dict[str, np.ndarray]:
        print(f"[PPG-PRE] Input: {len(raw_signal)} samples, range: {raw_signal.min():.0f}-{raw_signal.max():.0f}")
        signal = np.array(raw_signal, dtype=np.float64)
        dc_removed = signal - np.mean(signal)
        try:
            detrended = detrend(dc_removed)
        except Exception:
            detrended = dc_removed
        filtered = bandpass_filter(detrended, 0.5, 8.0, fs, order=3)
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        print(f"[PPG-PRE] Output range: {normalized.min():.2f} to {normalized.max():.2f}")
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
        max_peak_distance = int(2.0 * fs)
        methods_results = []

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
        except Exception as e:
            print(f"[PPG-PEAKS] scipy_inverted failed: {e}")

        try:
            height_threshold = np.percentile(signal, 70)
            peaks, _ = scipy_find_peaks(
                signal,
                height=height_threshold,
                distance=min_peak_distance,
                prominence=np.std(signal) * 0.3
            )
            if len(peaks) >= 3:
                methods_results.append(('scipy_direct', peaks, len(peaks)))
        except Exception as e:
            print(f"[PPG-PEAKS] scipy_direct failed: {e}")

        if NEUROKIT_AVAILABLE:
            try:
                peaks_info = nk.ppg_findpeaks(signal, sampling_rate=fs)
                peaks = np.array(peaks_info.get('PPG_Peaks', []))
                if len(peaks) >= 3:
                    methods_results.append(('neurokit', peaks, len(peaks)))
            except Exception as e:
                print(f"[PPG-PEAKS] neurokit failed: {e}")

            try:
                peaks_info = nk.ppg_findpeaks(signal, sampling_rate=fs, method='elgendi')
                peaks = np.array(peaks_info.get('PPG_Peaks', []))
                if len(peaks) >= 3:
                    methods_results.append(('elgendi', peaks, len(peaks)))
            except Exception as e:
                print(f"[PPG-PEAKS] elgendi failed: {e}")

        try:
            diff1 = np.diff(signal)
            zero_crossings = []
            for i in range(len(diff1) - 1):
                if diff1[i] > 0 and diff1[i+1] <= 0:
                    zero_crossings.append(i + 1)
            if len(zero_crossings) >= 3:
                peaks = np.array(zero_crossings)
                intervals = np.diff(peaks)
                valid_mask = (intervals >= min_peak_distance) & (intervals <= max_peak_distance)
                if np.sum(valid_mask) >= 2:
                    methods_results.append(('zero_crossing', peaks, len(peaks)))
        except Exception as e:
            print(f"[PPG-PEAKS] zero_crossing failed: {e}")

        if not methods_results:
            print("[PPG-PEAKS] WARNING: No peaks found by any method")
            return np.array([])

        best_method = max(methods_results, key=lambda x: x[2])
        print(f"[PPG-PEAKS] Using {best_method[0]} with {best_method[2]} peaks")
        peaks = best_method[1]

        if len(peaks) >= 2:
            intervals = np.diff(peaks)
            valid_intervals = (intervals >= min_peak_distance) & (intervals <= max_peak_distance)
            filtered_peaks = [peaks[0]]
            for i, is_valid in enumerate(valid_intervals):
                if is_valid:
                    filtered_peaks.append(peaks[i + 1])
            peaks = np.array(filtered_peaks)

        return peaks


# ============================================================================
# SPO2 CALCULATOR (unchanged – not used in report generation)
# ============================================================================

class SpO2Calculator:
    @staticmethod
    def calculate(red_signal: np.ndarray, ir_signal: np.ndarray,
                  red_peaks: np.ndarray, ir_peaks: np.ndarray,
                  fs: int = PPG_FS) -> Dict[str, Any]:
        print(f"[SPO2] Calculating from {len(red_peaks)} red peaks, {len(ir_peaks)} IR peaks")
        result = {
            'spo2': None, 'spo2_values': [], 'r_ratios': [],
            'perfusion_index_red': None, 'perfusion_index_ir': None, 'valid': False
        }
        peaks = ir_peaks if len(ir_peaks) >= len(red_peaks) else red_peaks
        if len(peaks) < 3:
            print("[SPO2] Insufficient peaks for SpO2 calculation")
            return result

        r_ratios = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            if end_idx - start_idx < 3:
                continue
            if end_idx >= len(red_signal) or end_idx >= len(ir_signal):
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

        print(f"[SPO2] Valid R-ratios: {len(r_ratios)}")
        if len(r_ratios) >= 2:
            r_ratios_sorted = sorted(r_ratios)
            trim_count = max(1, len(r_ratios) // 4)
            r_ratios_trimmed = r_ratios_sorted[trim_count:-trim_count] if len(r_ratios) > 4 else r_ratios_sorted
            r_avg = np.mean(r_ratios_trimmed)
            result['r_ratios'] = r_ratios
            lookup_index = int(r_avg * 100)
            if 0 <= lookup_index < len(SPO2_LOOKUP):
                spo2 = SPO2_LOOKUP[lookup_index]
            else:
                spo2 = max(0, min(100, int(-45.060 * r_avg * r_avg + 30.354 * r_avg + 94.845)))
            result['spo2'] = spo2
            result['valid'] = True
            print(f"[SPO2] R-ratio avg: {r_avg:.3f}, SpO2: {spo2}%")

        try:
            red_dc = np.mean(red_signal)
            red_ac = np.max(red_signal) - np.min(red_signal)
            if red_dc > 0:
                result['perfusion_index_red'] = (red_ac / red_dc) * 100
            ir_dc = np.mean(ir_signal)
            ir_ac = np.max(ir_signal) - np.min(ir_signal)
            if ir_dc > 0:
                result['perfusion_index_ir'] = (ir_ac / ir_dc) * 100
        except Exception as e:
            print(f"[SPO2] Perfusion index calculation failed: {e}")

        return result


# ============================================================================
# ECG PROCESSOR (unchanged)
# ============================================================================

class ECGProcessor:
    def __init__(self, fs: int = ECG_FS):
        self.fs = fs

    def process(self, ecg_uv: np.ndarray) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"[ECG] ANALYSIS: {len(ecg_uv)} samples @ {self.fs} Hz ({len(ecg_uv)/self.fs:.2f}s)")
        print(f"{'='*70}")

        ecg_signal = np.array(ecg_uv, dtype=np.float64)
        result = {
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
            'hrv': {},  # flat dict for backward compat
            'waves': {},
            'intervals': {},
            'quality': None,
            'analysis_method': 'none',
        }

        if not NEUROKIT_AVAILABLE:
            result['cleaned_uv'] = ecg_signal
            result['cleaned_mv'] = ecg_signal / 1000.0
            result['analysis_method'] = 'basic'
            return result

        try:
            cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method='neurokit')
            result['cleaned_uv'] = cleaned
            result['cleaned_mv'] = cleaned / 1000.0
            print(f"[ECG] Cleaned: {cleaned.min():.1f} to {cleaned.max():.1f} uV")
        except Exception as e:
            print(f"[ECG] Clean failed: {e}")
            cleaned = ecg_signal
            result['cleaned_uv'] = ecg_signal
            result['cleaned_mv'] = ecg_signal / 1000.0

        try:
            _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            r_peaks = np.array(rpeaks_info.get('ECG_R_Peaks', []))
            result['r_peaks'] = r_peaks
            print(f"[ECG] R-peaks: {len(r_peaks)} at {r_peaks[:8]}{'...' if len(r_peaks)>8 else ''}")
        except Exception as e:
            print(f"[ECG] Peak detection failed: {e}")
            r_peaks = np.array([])

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
                print(f"[ECG] HR: {result['heart_rate']['mean']:.1f} +/- {result['heart_rate']['std']:.1f} bpm")

        try:
            signals_df, info = nk.ecg_process(cleaned, sampling_rate=self.fs)
            result['analysis_method'] = 'neurokit2_full'
            if 'ECG_Quality' in signals_df.columns:
                quality = signals_df['ECG_Quality'].dropna().values
                if len(quality) > 0:
                    result['quality'] = float(np.mean(quality))
                    print(f"[ECG] Quality: {result['quality']:.3f}")
        except Exception as e:
            print(f"[ECG] Full process failed: {e}")
            result['analysis_method'] = 'neurokit2_partial'

        if len(r_peaks) >= 3:
            try:
                _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=self.fs, method='dwt')
                result['waves'] = waves
                self._calculate_intervals(result, waves, r_peaks)
                print("[ECG] Delineation complete")
            except Exception as e:
                print(f"[ECG] Delineation failed: {e}")

        if len(r_peaks) >= 4:
            self._calculate_hrv(result, r_peaks)

        print(f"[ECG] Complete: {result['analysis_method']}")
        return result

    def _calculate_intervals(self, result: Dict, waves: Dict, r_peaks: np.ndarray):
        try:
            intervals = {}
            p_onsets = waves.get('ECG_P_Onsets', [])
            q_peaks = waves.get('ECG_Q_Peaks', [])
            s_peaks = waves.get('ECG_S_Peaks', [])
            t_offsets = waves.get('ECG_T_Offsets', [])

            pr_vals = []
            for i, r in enumerate(r_peaks):
                if i < len(p_onsets) and p_onsets[i] is not None and not np.isnan(p_onsets[i]):
                    pr = (r - p_onsets[i]) / self.fs * 1000
                    if 80 < pr < 300:
                        pr_vals.append(pr)
            if pr_vals:
                intervals['PR_Interval'] = float(np.mean(pr_vals))

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
            for k, v in intervals.items():
                print(f"[ECG] {k}: {v:.1f}ms")
        except Exception as e:
            print(f"[ECG] Interval calc error: {e}")

    def _calculate_hrv(self, result: Dict, r_peaks: np.ndarray):
        print(f"[ECG] HRV Analysis with {len(r_peaks)} R-peaks...")

        try:
            hrv_time = nk.hrv_time(r_peaks, sampling_rate=self.fs, show=False)
            time_metrics = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20',
                            'HRV_SDSD', 'HRV_CVNN', 'HRV_MedianNN', 'HRV_MadNN']
            for m in time_metrics:
                if m in hrv_time.columns:
                    val = hrv_time[m].values[0]
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        result['hrv_time'][m] = float(val)
                        result['hrv'][m] = float(val)  # flat copy for old code
            print(f"[ECG] HRV Time: {len(result['hrv_time'])} metrics")
        except Exception as e:
            print(f"[ECG] HRV Time failed: {e}")

        duration_sec = len(result['raw_uv']) / self.fs
        if duration_sec >= 60 and len(r_peaks) >= 30:
            try:
                hrv_freq = nk.hrv_frequency(r_peaks, sampling_rate=self.fs, show=False)
                freq_metrics = ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_TP']
                for m in freq_metrics:
                    if m in hrv_freq.columns:
                        val = hrv_freq[m].values[0]
                        if val is not None and not np.isnan(val) and not np.isinf(val) and val > 0:
                            result['hrv_freq'][m] = float(val)
                            result['hrv'][m] = float(val)
                print(f"[ECG] HRV Freq: {len(result['hrv_freq'])} metrics")
            except Exception as e:
                print(f"[ECG] HRV Freq failed: {e}")
        else:
            print(f"[ECG] HRV Freq skipped: need 60s+ ({duration_sec:.0f}s available)")

        if len(r_peaks) >= 10:
            try:
                hrv_nl = nk.hrv_nonlinear(r_peaks, sampling_rate=self.fs, show=False)
                nl_metrics = ['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_ApEn', 'HRV_SampEn',
                              'HRV_DFA_alpha1', 'HRV_DFA_alpha2']
                for m in nl_metrics:
                    if m in hrv_nl.columns:
                        val = hrv_nl[m].values[0]
                        if val is not None and not np.isnan(val) and not np.isinf(val):
                            result['hrv_nonlinear'][m] = float(val)
                            result['hrv'][m] = float(val)
                print(f"[ECG] HRV Nonlinear: {len(result['hrv_nonlinear'])} metrics")
            except Exception as e:
                print(f"[ECG] HRV Nonlinear failed: {e}")
        else:
            print(f"[ECG] HRV Nonlinear skipped: need 10+ R-peaks ({len(r_peaks)} available)")


# ============================================================================
# PPG PROCESSOR (unchanged – not used in report generation)
# ============================================================================

class PPGProcessor:
    def __init__(self, fs: int = PPG_FS):
        self.fs = fs

    def process(self, red_data: np.ndarray, ir_data: np.ndarray) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"[PPG] ANALYSIS: {len(ir_data)} samples @ {self.fs} Hz ({len(ir_data)/self.fs:.2f}s)")
        print(f"{'='*70}")

        result = {
            'red_raw': red_data, 'ir_raw': ir_data,
            'red_processed': None, 'ir_processed': None,
            'red_peaks': np.array([]), 'ir_peaks': np.array([]),
            'heart_rate': None, 'pp_intervals_ms': np.array([]),
            'prv_time': {}, 'prv_freq': {}, 'prv_nonlinear': {},
            'spo2': None, 'spo2_data': {}, 'perfusion_index': None,
            'signal_quality': None, 'analysis_method': 'none',
        }

        ir_prep = PPGPreprocessor.preprocess(ir_data, self.fs)
        result['ir_processed'] = ir_prep
        red_prep = PPGPreprocessor.preprocess(red_data, self.fs)
        result['red_processed'] = red_prep

        ir_peaks = PPGPreprocessor.find_peaks_robust(ir_prep['normalized'], self.fs)
        result['ir_peaks'] = ir_peaks
        red_peaks = PPGPreprocessor.find_peaks_robust(red_prep['normalized'], self.fs)
        result['red_peaks'] = red_peaks
        print(f"[PPG] Peaks found - IR: {len(ir_peaks)}, Red: {len(red_peaks)}")

        peaks = ir_peaks if len(ir_peaks) >= len(red_peaks) else red_peaks
        if len(peaks) >= 2:
            pp_samples = np.diff(peaks)
            pp_ms = pp_samples / self.fs * 1000
            pp_valid = pp_ms[(pp_ms > 300) & (pp_ms < 2000)]
            result['pp_intervals_ms'] = pp_valid
            if len(pp_valid) > 0:
                hr_bpm = 60000 / pp_valid
                result['heart_rate'] = {
                    'mean': float(np.mean(hr_bpm)), 'std': float(np.std(hr_bpm)),
                    'min': float(np.min(hr_bpm)), 'max': float(np.max(hr_bpm)),
                    'values': hr_bpm
                }
                print(f"[PPG] HR: {result['heart_rate']['mean']:.1f} +/- {result['heart_rate']['std']:.1f} bpm")
                result['analysis_method'] = 'custom_preprocessing'

        if len(ir_peaks) >= 3 and len(red_peaks) >= 3:
            spo2_result = SpO2Calculator.calculate(red_data, ir_data, red_peaks, ir_peaks, self.fs)
            result['spo2_data'] = spo2_result
            if spo2_result['valid']:
                result['spo2'] = spo2_result['spo2']
                result['perfusion_index'] = spo2_result.get('perfusion_index_ir')

        if len(peaks) >= 5:
            self._calculate_prv(result, peaks)

        ac = np.std(ir_prep['filtered'])
        dc = np.mean(ir_data)
        if dc > 0:
            result['signal_quality'] = min(1.0, ac / (dc * 0.01))

        print(f"[PPG] Complete: {result['analysis_method']}")
        return result

    def _calculate_prv(self, result: Dict, peaks: np.ndarray):
        if not NEUROKIT_AVAILABLE:
            return
        try:
            prv_time = nk.hrv_time(peaks, sampling_rate=self.fs, show=False)
            for m in ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20']:
                if m in prv_time.columns:
                    val = prv_time[m].values[0]
                    if val is not None and not np.isnan(val) and not np.isinf(val):
                        result['prv_time'][m.replace('HRV', 'PRV')] = float(val)
        except Exception as e:
            print(f"[PPG] PRV Time failed: {e}")

        if len(peaks) >= 10:
            try:
                prv_nl = nk.hrv_nonlinear(peaks, sampling_rate=self.fs, show=False)
                for m in ['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2']:
                    if m in prv_nl.columns:
                        val = prv_nl[m].values[0]
                        if val is not None and not np.isnan(val) and not np.isinf(val):
                            result['prv_nonlinear'][m.replace('HRV', 'PRV')] = float(val)
            except Exception as e:
                print(f"[PPG] PRV Nonlinear failed: {e}")


# ============================================================================
# STANDALONE PPG PROCESSING (REPLACED WITH v5.5 VERSION)
# ============================================================================

def process_ppg(red_data, ir_data, fs, patient_height_m=1.70):
    """Enhanced PPG processing with light filtering and robust peak detection (v5.5 version)"""
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
        'red_features': {},
        'ir_features': {},
        'heart_rate': None,
        'spo2': None,
        'perfusion_index': None,
        'age_index': None,
        'respiration_rate': None,
        'baseline_shift_ir': {},
        'baseline_shift_red': {},
        'baseline_ir': None,
        'baseline_red': None,
        'pat': None,
    }

    # Light lowpass only (preserve baseline, remove high-frequency noise)
    ir_filtered = lowpass_filter(ir_data, 5.0, fs, order=2)
    red_filtered = lowpass_filter(red_data, 5.0, fs, order=2)
    result['ir_filtered'] = ir_filtered
    result['red_filtered'] = red_filtered

    # --- Robust peak detection ---
    ir_peaks = np.array([])
    red_peaks = np.array([])
    if NEUROKIT_AVAILABLE:
        try:
            peaks_info = nk.ppg_findpeaks(ir_filtered, sampling_rate=fs)
            ir_peaks = np.array(peaks_info.get('PPG_Peaks', []))
        except:
            pass
        try:
            peaks_info = nk.ppg_findpeaks(red_filtered, sampling_rate=fs)
            red_peaks = np.array(peaks_info.get('PPG_Peaks', []))
        except:
            pass

    if len(ir_peaks) < MIN_PPG_PEAKS:
        ir_peaks = robust_ppg_peaks(ir_filtered, fs)
    if len(red_peaks) < MIN_PPG_PEAKS:
        red_peaks = robust_ppg_peaks(red_filtered, fs)

    result['ir_peaks'] = ir_peaks
    result['red_peaks'] = red_peaks
    print(f"[PPG] IR peaks: {len(ir_peaks)}, Red peaks: {len(red_peaks)}")

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

    # SpO2 calculation
    if len(ir_peaks) >= 3 and len(red_peaks) >= 3:
        r_ratios = []
        for ir_p in ir_peaks:
            diffs = np.abs(red_peaks - ir_p)
            if len(diffs) == 0:
                continue
            nearest_idx = np.argmin(diffs)
            red_p = red_peaks[nearest_idx]
            if abs(red_p - ir_p) > 0.5 * fs:
                continue

            start = max(0, min(ir_p, red_p) - int(0.2*fs))
            end = min(len(ir_data), max(ir_p, red_p) + int(0.2*fs))
            if end - start < 3:
                continue

            red_seg = red_data[start:end]
            ir_seg = ir_data[start:end]

            red_dc = np.mean(red_seg)
            ir_dc = np.mean(ir_seg)
            if red_dc <= 0 or ir_dc <= 0:
                continue

            red_ac = np.max(red_seg) - np.min(red_seg)
            ir_ac = np.max(ir_seg) - np.min(ir_seg)
            if ir_ac <= 0 or ir_dc <= 0:
                continue

            r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
            if 0.2 < r_ratio < 2.0:
                r_ratios.append(r_ratio)

        if len(r_ratios) >= 3:
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

    # Age Index (requires NeuroKit2 delineation)
    if NEUROKIT_AVAILABLE and HAS_PPG_DELINEATE and len(ir_peaks) >= 3:
        try:
            _, waves_ir = nk.ppg_delineate(ir_filtered, ir_peaks, sampling_rate=fs)
            result['ir_features'] = waves_ir
            if 'PPG_Dicrotic_Notch' in waves_ir:
                dicrotic = np.array(waves_ir['PPG_Dicrotic_Notch'])
                valid_mask = ~np.isnan(dicrotic)
                dicrotic = dicrotic[valid_mask]
                peaks_valid = ir_peaks[valid_mask]
                delta_t_values = []
                for i in range(len(peaks_valid)):
                    if i < len(dicrotic) and not np.isnan(dicrotic[i]):
                        delta_t = (dicrotic[i] - peaks_valid[i]) / fs
                        if 0.1 < delta_t < 0.35:
                            delta_t_values.append(delta_t)
                if len(delta_t_values) >= 2:
                    avg_delta_t = np.mean(delta_t_values)
                    age_index = patient_height_m / avg_delta_t
                    result['age_index'] = age_index
                    print(f"[PPG] Age Index: {age_index:.2f} m/s")
        except Exception as e:
            print(f"[PPG] Age Index failed: {e}")
    else:
        if NEUROKIT_AVAILABLE and not HAS_PPG_DELINEATE:
            print("[PPG] Age Index: ppg_delineate not available in this NeuroKit2 version")

    # Respiration Rate
    if len(ir_data) >= fs * 30:
        try:
            analytic_signal = hilbert(ir_filtered)
            amplitude_envelope = np.abs(analytic_signal)
            resp_signal = lowpass_filter(amplitude_envelope, 0.5, fs, order=2)
            min_distance = int(2.0 * fs)
            peaks, _ = scipy_find_peaks(resp_signal, distance=min_distance)
            if len(peaks) >= 3:
                breath_intervals = np.diff(peaks) / fs
                breath_rate = 60.0 / np.mean(breath_intervals)
                if 5 < breath_rate < 40:
                    result['respiration_rate'] = breath_rate
                    print(f"[PPG] Respiration: {breath_rate:.1f} breaths/min")
        except Exception as e:
            print(f"[PPG] Respiration failed: {e}")

    # Baseline Shift (with robust polyfit)
    try:
        for channel, filt in [('ir', ir_filtered), ('red', red_filtered)]:
            x = np.arange(len(filt))
            if len(filt) >= 2:
                coeffs = np.polyfit(x, filt, 1)
                drift_per_sample = coeffs[0]
                drift_per_sec = drift_per_sample * fs
                trend_line = np.polyval(coeffs, x)
                detrended_sig = filt - trend_line
                std_dev = np.std(detrended_sig)
                drift_magnitude = np.max(filt) - np.min(filt)
                result[f'baseline_shift_{channel}'] = {
                    'drift_per_sec': drift_per_sec,
                    'std_deviation': std_dev,
                    'drift_magnitude': drift_magnitude,
                }
                # Store the trend line for plotting (used by report generator)
                result[f'baseline_{channel}'] = trend_line
            else:
                result[f'baseline_shift_{channel}'] = {
                    'drift_per_sec': 0.0,
                    'std_deviation': 0.0,
                    'drift_magnitude': 0.0,
                }
                result[f'baseline_{channel}'] = np.zeros_like(filt)
        print(f"[PPG] Baseline IR: drift={result['baseline_shift_ir']['drift_per_sec']:.3f}/s, std={result['baseline_shift_ir']['std_deviation']:.2f}")
        print(f"[PPG] Baseline Red: drift={result['baseline_shift_red']['drift_per_sec']:.3f}/s, std={result['baseline_shift_red']['std_deviation']:.2f}")
    except Exception as e:
        print(f"[PPG] Baseline analysis failed: {e}")

    return result


def calculate_pat(ecg_r_peaks, ppg_peaks, ecg_fs, ppg_fs):
    """Calculate Pulse Arrival Time"""
    if len(ecg_r_peaks) < 2 or len(ppg_peaks) < 2:
        return None

    ecg_times = ecg_r_peaks / ecg_fs
    ppg_times = ppg_peaks / ppg_fs

    pat_values = []
    ppg_idx = 0
    for r_time in ecg_times:
        while ppg_idx < len(ppg_times) and ppg_times[ppg_idx] < r_time:
            ppg_idx += 1
        if ppg_idx < len(ppg_times):
            pat = (ppg_times[ppg_idx] - r_time) * 1000.0
            if 50 < pat < 500:
                pat_values.append(pat)

    if len(pat_values) >= 2:
        avg_pat = np.mean(pat_values)
        print(f"[PAT] Pulse Arrival Time: {avg_pat:.1f}ms")
        return avg_pat

    return None


# ============================================================================
# REPORT GENERATOR (unchanged except it now uses the new process_ppg)
# ============================================================================

class ReportGenerator(QThread):

    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def _page_summary(self, pdf, plt, ecg_result, ecg_result2, ppg_result, ppg_result2, snr):
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
        if ecg_result is None:
            ax.text(0.07, y, 'No ECG data available.', fontsize=10, transform=ax.transAxes, color='red')
            y -= 0.03
        else:
            hr = ecg_result.get('heart_rate') if ecg_result is not None else None
            if hr:
                ax.text(0.07, y, 'Heart Rate:', fontsize=10, transform=ax.transAxes)
                ax.text(0.35, y, f'{hr["mean"]:.1f} ± {hr["std"]:.1f} bpm', fontsize=10, 
                    fontfamily='monospace', transform=ax.transAxes)
                y -= 0.025
            intervals = ecg_result.get('intervals', {}) if ecg_result is not None else {}
            for k in ['PR_Interval', 'QRS_Duration', 'QTc']:
                if k in intervals:
                    ax.text(0.07, y, f'{k.replace("_", " ")}:', fontsize=10, transform=ax.transAxes)
                    ax.text(0.35, y, f'{intervals[k]:.0f} ms', fontsize=10, 
                        fontfamily='monospace', transform=ax.transAxes)
                    y -= 0.025
            hrv = ecg_result.get('hrv', {}) if ecg_result is not None else {}
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
            import matplotlib.gridspec as gridspec
            from matplotlib.backends.backend_pdf import PdfPages
            self.plt = plt
            self.PdfPages = PdfPages
            self.gridspec = gridspec

            self.progress.emit("Loading session data...")
            self._load_data()

            if self.ecg_data is None or len(self.ecg_data) < ECG_FS * 3:
                self.finished.emit(False, "Insufficient ECG data (need 3+ seconds)")
                return

            # ECG Sample 1: Best quality window
            self.progress.emit("Analyzing ECG Sample 1 (Best Quality)...")
            start1, end1, snr1, rpeak_cnt1 = SignalQualityAnalyzer.find_best_ecg_window(
                self.ecg_data, ECG_FS, min_rpeaks=MIN_RPEAKS
            )
            ecg_sample1 = self.ecg_data[start1:end1]
            ecg_processor = ECGProcessor(ECG_FS)
            ecg_result1 = ecg_processor.process(ecg_sample1)

            # ECG Sample 2: Full recording for long-term HRV
            ecg_result2 = None
            total_duration = len(self.ecg_data) / ECG_FS
            if total_duration >= 60:
                self.progress.emit("Analyzing ECG Sample 2 (Full Recording - Long-term)...")
                ecg_result2 = ecg_processor.process(self.ecg_data)

            # PPG Analysis (uses the new process_ppg function)
            ppg_result1 = None
            ppg_result2 = None
            if self.ppg_ir_data is not None and self.ppg_red_data is not None:
                ppg_total_duration = len(self.ppg_ir_data) / PPG_FS
                if ppg_total_duration >= 5:
                    self.progress.emit("Analyzing PPG (Full Recording)...")
                    height_m = 1.70
                    if 'Height' in self.patient_info:
                        try:
                            height_cm = float(self.patient_info['Height'])
                            if 50 < height_cm < 250:
                                height_m = height_cm / 100.0
                        except Exception:
                            pass
                    ppg_result1 = process_ppg(self.ppg_red_data, self.ppg_ir_data, PPG_FS, height_m)

                    # Also compute PAT if both ECG and PPG available
                    if ecg_result1.get('r_peaks') is not None and len(ecg_result1['r_peaks']) >= 2:
                        ppg_peaks = ppg_result1.get('ir_peaks', np.array([]))
                        if len(ppg_peaks) >= 2:
                            r_peaks_full = ecg_result1['r_peaks'] + start1
                            ppg_result1['pat'] = calculate_pat(
                                r_peaks_full, ppg_peaks, ECG_FS, PPG_FS)

                    if ppg_total_duration >= 60:
                        self.progress.emit("Analyzing PPG Sample 2 (Long-term)...")
                        ppg_result2 = process_ppg(self.ppg_red_data, self.ppg_ir_data, PPG_FS, height_m)

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
                print(f"[LOAD] ECG: {len(self.ecg_data) if self.ecg_data is not None else 0}")
            elif filename.endswith('_ppg_red.csv'):
                self.ppg_red_data = self._parse_waveform(filepath)
                print(f"[LOAD] PPG Red: {len(self.ppg_red_data) if self.ppg_red_data is not None else 0}")
            elif filename.endswith('_ppg_ir.csv'):
                self.ppg_ir_data = self._parse_waveform(filepath)
                print(f"[LOAD] PPG IR: {len(self.ppg_ir_data) if self.ppg_ir_data is not None else 0}")
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
        """Draw medical ECG paper grid (25mm/s, 10mm/mV)"""
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

    def _create_pdf(self, ecg_result1, ecg_result2, ppg_result1, ppg_result2,
                    start_idx, end_idx, snr, rpeak_count):
        plt = self.plt
        with self.PdfPages(self.output_path) as pdf:
            # Page 1: Cover
            self._page_cover(pdf, plt, ecg_result1, ppg_result1, snr, start_idx, end_idx, rpeak_count)

            # Page 2: ECG Medical Grid
            self._page_ecg_medical(pdf, plt, ecg_result1)

            # Page 3: ECG PQRST Labeled Beats
            self._page_ecg_pqrst(pdf, plt, ecg_result1)

            # Page 4: HRV Analysis
            if ecg_result1.get('hrv_time') or ecg_result1.get('hrv'):
                self._page_ecg_hrv(pdf, plt, ecg_result1)

            # Page 5: Full recording overview
            self._page_full_recording(pdf, plt, start_idx, end_idx)

            # Page 6+7: PPG
            if ppg_result1:
                self._page_ppg_signals(pdf, plt, ppg_result1)
                self._page_ppg_analysis(pdf, plt, ppg_result1)

            # Page 8: Summary
            self._page_summary(pdf, plt, ecg_result1, ecg_result2, ppg_result1, ppg_result2, snr)

            # Page 9: Explanations
            self._page_explanations(pdf, plt, ecg_result1, ppg_result1)

    # ------------------------------------------------------------------
    # COVER PAGE
    # ------------------------------------------------------------------
    def _page_cover(self, pdf, plt, ecg_result, ppg_result, snr,
                    start_idx, end_idx, rpeak_count):
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('NirogScan Health Report', fontsize=24, fontweight='bold', y=0.95)
        ax = fig.add_subplot(111)
        ax.axis('off')

        y = 0.85
        ax.text(0.5, y, 'Patient Information', fontsize=14, fontweight='bold',
                ha='center', transform=ax.transAxes)
        y -= 0.04
        for key, value in self.patient_info.items():
            ax.text(0.3, y, f'{key}:', ha='right', fontsize=10, transform=ax.transAxes)
            ax.text(0.32, y, str(value), ha='left', fontsize=10, transform=ax.transAxes)
            y -= 0.025

        y -= 0.02
        ax.text(0.5, y, 'Recording & Vital Signs', fontsize=14, fontweight='bold',
                ha='center', transform=ax.transAxes)
        y -= 0.04

        items = [
            ('ECG Duration', f'{len(self.ecg_data)/ECG_FS:.1f}s'),
            ('ECG Sample Rate', f'{ECG_FS} Hz'),
            ('Analysis Window', f'{start_idx/ECG_FS:.2f}s - {end_idx/ECG_FS:.2f}s'),
            ('Window SNR', f'{snr:.1f} dB'),
            ('R-peaks Found', f'{rpeak_count}'),
        ]
        hr = ecg_result.get('heart_rate')
        if hr:
            items.append(('Heart Rate (ECG)', f'{hr["mean"]:.1f} +/- {hr["std"]:.1f} bpm'))
        if ppg_result:
            if ppg_result.get('spo2'):
                items.append(('SpO2', f'{ppg_result["spo2"]}%'))
            ppg_hr = ppg_result.get('heart_rate')
            if ppg_hr:
                items.append(('Pulse Rate (PPG)', f'{ppg_hr["mean"]:.1f} bpm'))
            if ppg_result.get('age_index'):
                items.append(('Age Index', f'{ppg_result["age_index"]:.2f} m/s'))
            if ppg_result.get('pat'):
                items.append(('PAT', f'{ppg_result["pat"]:.1f} ms'))
            if ppg_result.get('respiration_rate'):
                items.append(('Respiration Rate', f'{ppg_result["respiration_rate"]:.1f} breaths/min'))
            if ppg_result.get('perfusion_index'):
                items.append(('Perfusion Index', f'{ppg_result["perfusion_index"]:.2f}%'))

        for label, value in items:
            ax.text(0.3, y, f'{label}:', ha='right', fontsize=10, transform=ax.transAxes)
            ax.text(0.32, y, value, ha='left', fontsize=10, fontfamily='monospace',
                    transform=ax.transAxes)
            y -= 0.025

        y -= 0.03
        ax.text(0.5, y, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=9, ha='center', transform=ax.transAxes, style='italic', color='gray')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # ECG MEDICAL GRID PAGE
    # ------------------------------------------------------------------
    def _page_ecg_medical(self, pdf, plt, ecg_result):
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('ECG Medical Grid Analysis', fontsize=14, fontweight='bold')

        clean_mv = ecg_result['cleaned_mv']
        r_peaks = ecg_result.get('r_peaks', np.array([]))
        duration = len(clean_mv) / ECG_FS
        t = np.arange(len(clean_mv)) / ECG_FS

        # Top: full cleaned ECG with R-peak markers and RR labels
        ax = axes[0]
        self._draw_ecg_grid(ax, duration)
        ax.plot(t, clean_mv, 'k-', lw=0.8, zorder=5, label='Cleaned ECG')
        if len(r_peaks) > 0:
            valid_peaks = r_peaks[r_peaks < len(clean_mv)]
            ax.scatter(valid_peaks/ECG_FS, clean_mv[valid_peaks], color='red',
                       s=100, marker='v', zorder=10, label='R-peaks')
            for i in range(len(valid_peaks) - 1):
                rr_ms = (valid_peaks[i+1] - valid_peaks[i]) / ECG_FS * 1000
                mid_t = (valid_peaks[i] + valid_peaks[i+1]) / 2 / ECG_FS
                ax.annotate(f'RR={rr_ms:.0f}ms', xy=(mid_t, -1.7), fontsize=7,
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow',
                                      ec='orange', alpha=0.9))
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('Amplitude (mV)', fontsize=10)
        ax.set_title('Cleaned ECG with R-peaks')
        ax.legend(loc='upper right', fontsize=8)

        # Bottom: single beat zoomed with interval annotations
        ax = axes[1]
        if len(r_peaks) >= 2:
            idx = len(r_peaks) // 2
            r = r_peaks[idx]
            pre, post = int(0.3 * ECG_FS), int(0.4 * ECG_FS)
            start, end = max(0, r - pre), min(len(clean_mv), r + post)
            beat = clean_mv[start:end]
            t_beat = (np.arange(len(beat)) - (r - start)) / ECG_FS * 1000
            beat_duration_s = len(beat) / ECG_FS
            self._draw_ecg_grid(ax, beat_duration_s)
            ax.plot(t_beat, beat, 'b-', lw=2)
            ax.axvline(0, color='red', linestyle='--', lw=1.5, alpha=0.7, label='R-peak')
            intervals = ecg_result.get('intervals', {})
            info = ""
            for k in ['PR_Interval', 'QRS_Duration', 'QT_Interval', 'QTc']:
                if k in intervals:
                    info += f"{k.replace('_', ' ')}: {intervals[k]:.0f}ms\n"
            if info:
                ax.text(0.02, 0.98, info.strip(), transform=ax.transAxes, fontsize=9,
                        va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
            hr = ecg_result.get('heart_rate')
            if hr:
                ax.text(0.98, 0.95, f"HR: {hr['mean']:.1f}+/-{hr['std']:.1f} bpm",
                        transform=ax.transAxes, fontsize=10, ha='right', va='top',
                        bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.9))
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

    # ------------------------------------------------------------------
    # ECG PQRST PAGE
    # ------------------------------------------------------------------
    def _page_ecg_pqrst(self, pdf, plt, ecg_result):
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

        # Plot 1: PQRST points labeled
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
            markers = {
                'ECG_P_Peaks': ('P', 'green'), 'ECG_Q_Peaks': ('Q', 'orange'),
                'ECG_S_Peaks': ('S', 'purple'), 'ECG_T_Peaks': ('T', 'brown')
            }
            for wkey, (label, color) in markers.items():
                if wkey in waves and waves[wkey] is not None:
                    arr = np.array(waves[wkey])
                    if idx < len(arr) and not np.isnan(arr[idx]):
                        w_idx = int(arr[idx])
                        if start <= w_idx < end:
                            w_t = (w_idx - r) / ECG_FS * 1000
                            ax.scatter([w_t], [clean_mv[w_idx]], c=color, s=120, zorder=10)
                            ax.annotate(label, (w_t, clean_mv[w_idx]), xytext=(0, 15),
                                        textcoords='offset points', ha='center',
                                        fontsize=11, fontweight='bold', color=color)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('mV')
            ax.set_title('PQRST Points Labeled')
            ax.grid(True, alpha=0.3)

        # Plot 2: colored segments
        ax = fig.add_subplot(gs[0, 1])
        if len(r_peaks) >= 2:
            idx = len(r_peaks) // 2
            r = r_peaks[idx]
            pre, post = int(0.3 * ECG_FS), int(0.4 * ECG_FS)
            start, end = max(0, r - pre), min(len(clean_mv), r + post)
            beat = clean_mv[start:end]
            t_beat = (np.arange(len(beat)) - (r - start)) / ECG_FS * 1000
            ax.plot(t_beat, beat, 'gray', lw=1, alpha=0.5)

            def get_idx(key):
                if key in waves and waves[key] is not None:
                    arr = np.array(waves[key])
                    if idx < len(arr) and not np.isnan(arr[idx]):
                        return int(arr[idx])
                return None

            p_on = get_idx('ECG_P_Onsets')
            p_off = get_idx('ECG_P_Offsets')
            q_peak = get_idx('ECG_Q_Peaks')
            s_peak = get_idx('ECG_S_Peaks')
            t_on = get_idx('ECG_T_Onsets')
            t_off = get_idx('ECG_T_Offsets')

            if p_on and p_off and start <= p_on < p_off < end:
                p_t = t_beat[p_on-start:p_off-start+1]
                ax.plot(p_t, beat[p_on-start:p_off-start+1], 'green', lw=3, label='P-wave')
            if q_peak and s_peak and start <= q_peak < s_peak < end:
                qrs_t = t_beat[q_peak-start:s_peak-start+1]
                ax.plot(qrs_t, beat[q_peak-start:s_peak-start+1], 'red', lw=3, label='QRS')
            if t_on and t_off and start <= t_on < t_off < end:
                t_t = t_beat[t_on-start:t_off-start+1]
                ax.plot(t_t, beat[t_on-start:t_off-start+1], 'blue', lw=3, label='T-wave')

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
                            np.mean(rr_ms)+np.std(rr_ms), alpha=0.2, color='red')
            ax.set_xlabel('Beat #')
            ax.set_ylabel('RR Interval (ms)')
            ax.set_title('RR Tachogram')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient RR data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')

        # Plot 4: Poincare Plot
        ax = fig.add_subplot(gs[1, 1])
        if len(rr_ms) >= 3:
            rr1, rr2 = rr_ms[:-1], rr_ms[1:]
            ax.scatter(rr1, rr2, alpha=0.7, s=50, c='blue', edgecolors='darkblue')
            min_v = min(rr1.min(), rr2.min()) * 0.95
            max_v = max(rr1.max(), rr2.max()) * 1.05
            ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.5)
            sd1 = ecg_result.get('hrv_nonlinear', {}).get('HRV_SD1',
                  np.std(rr2 - rr1) / np.sqrt(2))
            sd2 = ecg_result.get('hrv_nonlinear', {}).get('HRV_SD2',
                  np.std(rr2 + rr1) / np.sqrt(2))
            ax.text(0.05, 0.95, f'SD1: {sd1:.1f}ms\nSD2: {sd2:.1f}ms',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
            ax.set_xlabel('RR(n) ms')
            ax.set_ylabel('RR(n+1) ms')
            ax.set_title('Poincare Plot')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # HRV PAGE
    # ------------------------------------------------------------------
    def _page_ecg_hrv(self, pdf, plt, ecg_result):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Heart Rate Variability (HRV) Analysis', fontsize=14, fontweight='bold')

        hrv_time = ecg_result.get('hrv_time', {})
        hrv_freq = ecg_result.get('hrv_freq', {})
        hrv_nl = ecg_result.get('hrv_nonlinear', {})

        # Time domain metrics
        ax = axes[0, 0]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Time Domain', fontsize=12, fontweight='bold',
                ha='center', transform=ax.transAxes)
        y = 0.85
        for key in ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MedianNN']:
            if key in hrv_time:
                name = METRIC_EXPLANATIONS.get(key, (key, ''))[0]
                ax.text(0.05, y, f'{name}:', fontsize=10, transform=ax.transAxes)
                ax.text(0.65, y, f'{hrv_time[key]:.2f}', fontsize=10,
                        fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
                y -= 0.1

        # Frequency domain metrics
        ax = axes[0, 1]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Frequency Domain', fontsize=12, fontweight='bold',
                ha='center', transform=ax.transAxes)
        y = 0.88
        if hrv_freq:
            for key in ['HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn']:
                if key in hrv_freq:
                    name = METRIC_EXPLANATIONS.get(key, (key, ''))[0]
                    ax.text(0.05, y, f'{name}:', fontsize=10, transform=ax.transAxes)
                    ax.text(0.65, y, f'{hrv_freq[key]:.2f}', fontsize=10,
                            fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
                    y -= 0.1
        else:
            ax.text(0.5, 0.5, 'Frequency analysis requires\n60+ seconds of recording',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray')

        # RR distribution histogram
        ax = axes[1, 0]
        rr_ms = ecg_result.get('rr_intervals_ms', np.array([]))
        if len(rr_ms) >= 3:
            ax.hist(rr_ms, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(rr_ms), color='red', linestyle='--', lw=2,
                       label=f'Mean: {np.mean(rr_ms):.0f}ms')
            ax.axvline(np.median(rr_ms), color='green', linestyle=':', lw=2,
                       label=f'Median: {np.median(rr_ms):.0f}ms')
            ax.set_xlabel('RR Interval (ms)')
            ax.set_ylabel('Count')
            ax.set_title('RR Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')

        # Nonlinear metrics
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.5, 0.98, 'Nonlinear', fontsize=12, fontweight='bold',
                ha='center', transform=ax.transAxes)
        y = 0.88
        if hrv_nl:
            for key in ['HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_ApEn', 'HRV_DFA_alpha1']:
                if key in hrv_nl:
                    name = METRIC_EXPLANATIONS.get(key, (key, ''))[0]
                    ax.text(0.05, y, f'{name}:', fontsize=10, transform=ax.transAxes)
                    ax.text(0.65, y, f'{hrv_nl[key]:.4f}', fontsize=10,
                            fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
                    y -= 0.1
        else:
            ax.text(0.5, 0.5, 'Nonlinear analysis requires\n10+ R-peaks',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # FULL RECORDING OVERVIEW PAGE
    # ------------------------------------------------------------------
    def _page_full_recording(self, pdf, plt, start_idx, end_idx):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ecg_mv = self.ecg_data / 1000.0
        max_pts = 8000
        if len(ecg_mv) > max_pts:
            factor = len(ecg_mv) // max_pts
            ecg_ds = ecg_mv[::factor]
            t = np.arange(len(ecg_ds)) * factor / ECG_FS
        else:
            ecg_ds = ecg_mv
            t = np.arange(len(ecg_ds)) / ECG_FS
        ax.plot(t, ecg_ds, 'b-', lw=0.5, alpha=0.8)
        ax.axvspan(start_idx/ECG_FS, end_idx/ECG_FS, alpha=0.3, color='green',
                   label='Analysis Window')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title(f'Full ECG Recording ({len(self.ecg_data)/ECG_FS:.1f}s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # PPG SIGNALS PAGE (unchanged – uses baseline_ir and baseline_red)
    # ------------------------------------------------------------------
    def _page_ppg_signals(self, pdf, plt, ppg_result):
        """PPG signals with peaks labeled and baseline tracking"""
        fig, axes = plt.subplots(4, 1, figsize=(11, 11))
        fig.suptitle('PPG Signal Analysis', fontsize=14, fontweight='bold')

        ir_raw = ppg_result['ir_raw']
        red_raw = ppg_result['red_raw']
        ir_filtered = ppg_result.get('ir_filtered', ir_raw)
        red_filtered = ppg_result.get('red_filtered', red_raw)
        ir_peaks = ppg_result.get('ir_peaks', np.array([]))
        red_peaks = ppg_result.get('red_peaks', np.array([]))
        baseline_ir = ppg_result.get('baseline_ir', None)
        baseline_red = ppg_result.get('baseline_red', None)

        window = min(len(ir_raw), PPG_FS * 15)
        t = np.arange(window) / PPG_FS

        # IR Channel
        ax = axes[0]
        ax.plot(t, ir_raw[:window], 'purple', lw=0.8, alpha=0.4, label='Raw IR')
        ax.plot(t, ir_filtered[:window], 'b-', lw=1.2, label='Filtered IR')
        peaks_win = ir_peaks[ir_peaks < window]
        if len(peaks_win) > 0:
            ax.plot(t[peaks_win], ir_filtered[peaks_win], 'ro', label='IR Peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('IR Amplitude')
        ax.legend()
        ax.set_title('IR Channel')

        # Red Channel
        ax = axes[1]
        ax.plot(t, red_raw[:window], 'darkred', lw=0.8, alpha=0.4, label='Raw Red')
        ax.plot(t, red_filtered[:window], 'r-', lw=1.2, label='Filtered Red')
        peaks_win = red_peaks[red_peaks < window]
        if len(peaks_win) > 0:
            ax.plot(t[peaks_win], red_filtered[peaks_win], 'go', label='Red Peaks')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Red Amplitude')
        ax.legend()
        ax.set_title('Red Channel')

        # Baseline Tracking IR
        ax = axes[2]
        ax.plot(t, ir_filtered[:window], 'b-', lw=1, alpha=0.5, label='Filtered IR')
        if baseline_ir is not None:
            ax.plot(t, baseline_ir[:window], 'k--', lw=1.2, label='IR Baseline')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('IR Baseline')
        ax.legend()
        ax.set_title('IR Baseline Tracking')

        # Baseline Tracking Red
        ax = axes[3]
        ax.plot(t, red_filtered[:window], 'r-', lw=1, alpha=0.5, label='Filtered Red')
        if baseline_red is not None:
            ax.plot(t, baseline_red[:window], 'k--', lw=1.2, label='Red Baseline')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Red Baseline')
        ax.legend()
        ax.set_title('Red Baseline Tracking')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _page_ppg_analysis(self, pdf, plt, ppg_result):
        """PPG analysis: SpO2, Age Index, PAT, Respiration, Frequency, Segmentation"""
        import neurokit2 as nk
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('PPG Advanced Analysis', fontsize=16, fontweight='bold')
        gs = self.gridspec.GridSpec(3, 3, figure=fig)

        # 1. SpO2
        ax = fig.add_subplot(gs[0, 0])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Oxygen Saturation', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        spo2 = ppg_result.get('spo2')
        if spo2:
            color = 'green' if spo2 >= 95 else ('orange' if spo2 >= 90 else 'red')
            ax.text(0.5, 0.6, f'{spo2:.1f}%', fontsize=48, fontweight='bold', ha='center', transform=ax.transAxes, color=color)
            ax.text(0.5, 0.35, 'SpO2', fontsize=14, ha='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'SpO2\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')

        # 2. Age Index
        ax = fig.add_subplot(gs[0, 1])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Age Index (Stiffness)', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        age_idx = ppg_result.get('age_index')
        if age_idx:
            ax.text(0.5, 0.6, f'{age_idx:.2f}', fontsize=42, fontweight='bold', ha='center', transform=ax.transAxes, color='purple')
            ax.text(0.5, 0.4, 'm/s', fontsize=14, ha='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Age Index\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')

        # 3. PAT
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Pulse Arrival Time', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        pat = ppg_result.get('pat')
        if pat:
            ax.text(0.5, 0.6, f'{pat:.1f}', fontsize=42, fontweight='bold', ha='center', transform=ax.transAxes, color='teal')
            ax.text(0.5, 0.4, 'ms', fontsize=14, ha='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'PAT\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')

        # 4. Frequency change plot (IR)
        ir_peaks = ppg_result.get('ir_peaks', np.array([]))
        ax = fig.add_subplot(gs[1, 0])
        ax.set_title('IR Instantaneous Frequency')
        if len(ir_peaks) > 1:
            ir_pp_intervals = np.diff(ir_peaks) / PPG_FS
            ir_freq = 1.0 / ir_pp_intervals
            ax.plot(ir_peaks[1:] / PPG_FS, ir_freq, 'b.-', label='IR Freq (Hz)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Not enough IR peaks', ha='center', va='center', transform=ax.transAxes)

        # 5. Frequency change plot (Red)
        red_peaks = ppg_result.get('red_peaks', np.array([]))
        ax = fig.add_subplot(gs[1, 1])
        ax.set_title('Red Instantaneous Frequency')
        if len(red_peaks) > 1:
            red_pp_intervals = np.diff(red_peaks) / PPG_FS
            red_freq = 1.0 / red_pp_intervals
            ax.plot(red_peaks[1:] / PPG_FS, red_freq, 'r.-', label='Red Freq (Hz)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Not enough Red peaks', ha='center', va='center', transform=ax.transAxes)

        # 6. Segmentation and labeling of 5 PPG beats (IR)
        ir_filtered = ppg_result.get('ir_filtered', None)
        ax = fig.add_subplot(gs[1, 2])
        ax.set_title('IR Beat Segmentation (5 beats)')
        if ir_filtered is not None and len(ir_peaks) >= 6:
            for i in range(5):
                start = int(ir_peaks[i])
                end = int(ir_peaks[i+1])
                beat = ir_filtered[start:end]
                t_beat = np.arange(len(beat)) / PPG_FS
                ax.plot(t_beat, beat, label=f'Beat {i+1}')
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        else:
            ax.text(0.5, 0.5, 'Not enough IR beats', ha='center', va='center', transform=ax.transAxes)

        # 7. Segmentation and labeling of 5 PPG beats (Red)
        red_filtered = ppg_result.get('red_filtered', None)
        ax = fig.add_subplot(gs[2, 0])
        ax.set_title('Red Beat Segmentation (5 beats)')
        if red_filtered is not None and len(red_peaks) >= 6:
            for i in range(5):
                start = int(red_peaks[i])
                end = int(red_peaks[i+1])
                beat = red_filtered[start:end]
                t_beat = np.arange(len(beat)) / PPG_FS
                ax.plot(t_beat, beat, label=f'Beat {i+1}')
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
        else:
            ax.text(0.5, 0.5, 'Not enough Red beats', ha='center', va='center', transform=ax.transAxes)

        # 8. Respiration Rate
        ax = fig.add_subplot(gs[2, 1])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Respiration Rate', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        resp = ppg_result.get('respiration_rate')
        if resp:
            ax.text(0.5, 0.6, f'{resp:.1f}', fontsize=42, fontweight='bold', ha='center', transform=ax.transAxes, color='darkgreen')
            ax.text(0.5, 0.4, 'breaths/min', fontsize=14, ha='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Respiration\nnot available', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')

        # 9. Baseline tracking summary
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        ax.text(0.5, 0.95, 'Baseline Tracking', fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
        bs = ppg_result.get('baseline_shift_ir', {})
        txt = ''
        if bs:
            txt += f"IR Drift: {bs.get('drift_per_sec', 0):.3f}/s\nIR Std: {bs.get('std_deviation', 0):.2f}\n"
            bs = ppg_result.get('baseline_shift_red', {})
            txt += f"Red Drift: {bs.get('drift_per_sec', 0):.3f}/s\nRed Std: {bs.get('std_deviation', 0):.2f}"
        else:
            txt = 'Not available'
        ax.text(0.5, 0.5, txt, ha='center', va='center', transform=ax.transAxes, fontsize=12)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # EXPLANATIONS PAGE
    # ------------------------------------------------------------------
    def _page_explanations(self, pdf, plt, ecg_result, ppg_result):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.5, 0.97, 'Understanding Your Results', fontsize=18,
                fontweight='bold', ha='center', transform=ax.transAxes)
        ax.text(0.5, 0.94, '(Simple explanations of each measurement)', fontsize=11,
                ha='center', transform=ax.transAxes, style='italic', color='gray')

        metrics_present = set()
        metrics_present.update(ecg_result.get('hrv_time', {}).keys())
        metrics_present.update(ecg_result.get('hrv_freq', {}).keys())
        metrics_present.update(ecg_result.get('hrv_nonlinear', {}).keys())
        metrics_present.update(ecg_result.get('intervals', {}).keys())
        if ppg_result:
            if ppg_result.get('spo2'):
                metrics_present.add('SpO2')
            if ppg_result.get('perfusion_index'):
                metrics_present.add('Perfusion_Index')
            if ppg_result.get('heart_rate'):
                metrics_present.add('PPG_HR')

        priority = ['SpO2', 'Perfusion_Index', 'PPG_HR', 'HRV_MeanNN', 'HRV_SDNN',
                    'HRV_RMSSD', 'HRV_pNN50', 'HRV_LF', 'HRV_HF', 'HRV_LFHF',
                    'HRV_SD1', 'HRV_SD2', 'HRV_ApEn', 'HRV_DFA_alpha1',
                    'PR_Interval', 'QRS_Duration', 'QTc', 'PRV_SDNN', 'PRV_RMSSD']

        y = 0.88
        count = 0
        for key in priority:
            if key in metrics_present and key in METRIC_EXPLANATIONS and count < 16:
                name, explanation = METRIC_EXPLANATIONS[key]
                ax.text(0.03, y, f'  {name}:', fontsize=10, fontweight='bold',
                        transform=ax.transAxes)
                y -= 0.022
                if len(explanation) > 90:
                    explanation = explanation[:87] + '...'
                ax.text(0.05, y, explanation, fontsize=9, transform=ax.transAxes,
                        color='#333333')
                y -= 0.035
                count += 1
                if y < 0.08:
                    break

        y -= 0.02
        ax.text(0.5, y,
                'Note: These are for informational purposes only. '
                'Consult a healthcare professional for medical advice.',
                fontsize=9, ha='center', transform=ax.transAxes,
                style='italic', color='#666666')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


# ============================================================================
# DIALOGS
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
        self.weight.setPlaceholderText("Optional")
        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("Optional (cm) - used for Age Index")
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

        self.list.itemSelectionChanged.connect(
            lambda: self.gen_btn.setEnabled(bool(self.list.selectedItems())))
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


# ============================================================================
# MAIN GUI
# ============================================================================

class NirogScanGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NirogScan v4.3 (PPG Updated)")
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
        self.ecg_plot.setLabel('left', 'uV')
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
                self.log_queue.put_nowait({
                    'type': 'vitals', 'battery_v': p.battery_v,
                    'battery_pct': p.battery_pct, 'temp': p.temp, 'leads': p.leads
                })
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
            self.temp_lbl.setText(f"{p.temp:.1f}C")
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
        self.log_thread = LogThread(
            self.log_queue, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", path)
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
        out, _ = QFileDialog.getSaveFileName(
            self, "Save Report",
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


# ============================================================================
# ENTRY POINT
# ============================================================================

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