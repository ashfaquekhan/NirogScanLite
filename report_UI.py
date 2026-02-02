#!/usr/bin/env python3
"""
Y3X Data Logger - PyQt5 Version (Nuitka Compatible)
Optimized for performance with auto-scaling layouts
"""

import sys
import os
import struct
import asyncio
import queue
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Lock
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QLabel, QPushButton, QComboBox, QMessageBox, QFileDialog,
    QDialog, QFormLayout, QLineEdit, QTextEdit, QStatusBar, QSplitter,
    QListWidget, QListWidgetItem, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox,
    QSizePolicy
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

import pyqtgraph as pg
from bleak import BleakClient, BleakScanner

from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import pywt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

# Optimize pyqtgraph
pg.setConfigOptions(
    antialias=False,  # Faster rendering
    background='#1a1a1a',
    foreground='#ffffff',
    useOpenGL=False,  # More compatible
    enableExperimental=False
)

# Configuration
PLOT_BUFFER_SIZE = 500
PLOT_UPDATE_MS = 33  # ~30 FPS
DATA_QUEUE_SIZE = 2000
ECG_SAMPLING_RATE = 128
PPG_SAMPLING_RATE = 64

DEFAULT_FILTER_SETTINGS = {
    'bandpass_enabled': True,
    'bandpass_low': 0.5,
    'bandpass_high': 45.0,
    'notch_enabled': True,
    'notch_freq': 50.0,
    'wavelet_enabled': True,
    'wavelet_level': 4
}

# ============================================================================
# FAST CIRCULAR BUFFER
# ============================================================================
class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.lock = Lock()
        self._cache = None
        self._dirty = True
    
    def extend(self, values):
        with self.lock:
            self.buffer.extend(values)
            self._dirty = True
    
    def append(self, v):
        with self.lock:
            self.buffer.append(v)
            self._dirty = True
    
    def get_data(self):
        with self.lock:
            if self._dirty or self._cache is None:
                self._cache = np.array(self.buffer, dtype=np.float32)
                self._dirty = False
            return self._cache
    
    def clear(self):
        with self.lock:
            self.buffer.clear()
            self._cache = None
            self._dirty = True

# ============================================================================
# ECG PROCESSING
# ============================================================================
class ECGProcessor:
    @staticmethod
    def bandpass(data, low, high, fs, order=4):
        nyq = fs / 2
        if low / nyq <= 0 or high / nyq >= 1:
            return data
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, data)
    
    @staticmethod
    def notch(data, freq, fs, q=30):
        if freq >= fs / 2:
            return data
        b, a = iirnotch(freq / (fs / 2), q)
        return filtfilt(b, a, data)
    
    @staticmethod
    def wavelet_denoise(signal, level=4):
        try:
            coeffs = pywt.wavedec(signal, 'db4', level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(signal)))
            coeffs = [pywt.threshold(c, thresh, 'soft') for c in coeffs]
            rec = pywt.waverec(coeffs, 'db4')
            return rec[:len(signal)] if len(rec) > len(signal) else rec
        except:
            return signal
    
    @staticmethod
    def filter_ecg(signal, fs, settings=None):
        if settings is None:
            settings = DEFAULT_FILTER_SETTINGS
        filtered = signal.copy()
        if settings.get('bandpass_enabled', True):
            filtered = ECGProcessor.bandpass(filtered, settings.get('bandpass_low', 0.5), 
                                            settings.get('bandpass_high', 45.0), fs)
        if settings.get('notch_enabled', True):
            filtered = ECGProcessor.notch(filtered, settings.get('notch_freq', 50.0), fs)
        if settings.get('wavelet_enabled', True):
            filtered = ECGProcessor.wavelet_denoise(filtered, settings.get('wavelet_level', 4))
        return filtered
    
    @staticmethod
    def detect_r_peaks(signal, fs):
        if len(signal) < fs:
            return np.array([])
        try:
            nyq = fs / 2
            b, a = butter(2, [5 / nyq, min(15, nyq * 0.9) / nyq], btype='band')
            filtered = filtfilt(b, a, signal - np.mean(signal))
            diff = np.diff(filtered)
            diff = np.append(diff, diff[-1])
            squared = diff ** 2
            win = max(3, int(0.08 * fs))
            if win % 2 == 0:
                win += 1
            ma = np.convolve(squared, np.ones(win) / win, 'same')
            thresh = np.mean(ma) + 0.3 * np.std(ma)
            peaks, _ = find_peaks(ma, height=thresh, distance=int(0.3 * fs))
            refined = []
            win_search = int(0.05 * fs)
            for p in peaks:
                start = max(0, p - win_search)
                end = min(len(signal), p + win_search + 1)
                refined.append(start + np.argmax(signal[start:end]))
            return np.array(refined)
        except:
            return np.array([])
    
    @staticmethod
    def get_trimmed_rr_indices(peaks, trim_count=1):
        if len(peaks) <= 2 * trim_count:
            return []
        return list(range(trim_count, len(peaks) - trim_count))
    
    @staticmethod
    def scale_ecg(data):
        if len(data) == 0:
            return data, 1.0
        centered = data - np.mean(data)
        rng = np.ptp(centered)
        if rng == 0:
            return centered, 1.0
        scale = 3.0 / rng
        return centered * scale, scale
    
    @staticmethod
    def calculate_snr(signal, fs):
        try:
            if len(signal) < fs:
                return float('nan')
            centered = signal - np.mean(signal)
            nyq = fs / 2
            lo, hi = 0.5 / nyq, min(40, nyq * 0.8) / nyq
            if lo >= hi:
                return float('nan')
            b, a = butter(3, [lo, hi], btype='band')
            sig_band = filtfilt(b, a, centered)
            sig_pow = np.var(sig_band)
            noise_pow = np.var(np.diff(centered))
            return 10 * np.log10(sig_pow / noise_pow) if noise_pow > 0 else float('inf')
        except:
            return float('nan')

# ============================================================================
# REPORT GENERATOR
# ============================================================================
class ReportThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)
    
    def __init__(self, session_dir, output_path, filter_settings):
        super().__init__()
        self.session_dir = session_dir
        self.output_path = output_path
        self.filter_settings = filter_settings
        self.ecg = None
        self.ppg = None
        self.bioz = None
        self.patient = {}
        self.vitals = {}
    
    def run(self):
        try:
            self.progress.emit("Loading data...")
            self._load()
            self.progress.emit("Generating PDF...")
            self._generate()
            self.finished.emit(True, self.output_path)
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def _load(self):
        for f in os.listdir(self.session_dir):
            path = os.path.join(self.session_dir, f)
            if f.endswith('_ecg.csv'):
                self.ecg = self._load_csv(path)
            elif f.endswith('_ppg.csv'):
                self.ppg = self._load_csv(path)
            elif f.endswith('_bioz.csv'):
                self.bioz = self._load_csv(path)
            elif f.endswith('_hr.csv'):
                self.vitals['hr'] = self._load_csv(path)
            elif f.endswith('_spo2.csv'):
                self.vitals['spo2'] = self._load_csv(path)
            elif f.endswith('_temp.csv'):
                self.vitals['temp'] = self._load_csv(path)
            elif f.endswith('_rr.csv'):
                self.vitals['rr'] = self._load_csv(path)
            elif f == 'patient_info.txt':
                self.patient = self._load_patient(path)
    
    def _load_csv(self, path):
        try:
            df = pd.read_csv(path)
            if 'Data' in df.columns:
                return df['Data'].values.astype(np.float64)
            return None
        except:
            return None
    
    def _load_patient(self, path):
        info = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line and '=' not in line:
                        k, v = line.split(':', 1)
                        info[k.strip()] = v.strip()
        except:
            pass
        return info
    
    def _find_best_window(self, data, fs, dur=4):
        if data is None or len(data) == 0:
            return None, None, float('nan')
        win = int(dur * fs)
        if len(data) < win:
            return 0, len(data), ECGProcessor.calculate_snr(data, fs)
        best_snr, best_start = float('-inf'), 0
        step = max(1, win // 4)
        for i in range(0, len(data) - win, step):
            snr = ECGProcessor.calculate_snr(data[i:i + win], fs)
            if not np.isnan(snr) and snr > best_snr:
                best_snr, best_start = snr, i
        return best_start, best_start + win, best_snr
    
    def _draw_grid(self, ax, duration):
        ax.set_facecolor('white')
        for x in np.arange(0, duration + 0.04, 0.04):
            ax.axvline(x, color='#ffcccc', lw=0.3)
        for y in np.arange(-2.0, 2.1, 0.1):
            ax.axhline(y, color='#ffcccc', lw=0.3)
        for x in np.arange(0, duration + 0.2, 0.2):
            ax.axvline(x, color='#ff6666', lw=0.5)
        for y in np.arange(-2.0, 2.1, 0.5):
            ax.axhline(y, color='#ff6666', lw=0.5)
        ax.axhline(0, color='#ff6666', lw=0.8)
    
    def _get_filter_description(self):
        parts = []
        if self.filter_settings.get('bandpass_enabled'):
            parts.append(f"Bandpass({self.filter_settings.get('bandpass_low')}-{self.filter_settings.get('bandpass_high')}Hz)")
        if self.filter_settings.get('notch_enabled'):
            parts.append(f"Notch({self.filter_settings.get('notch_freq')}Hz)")
        if self.filter_settings.get('wavelet_enabled'):
            parts.append(f"Wavelet(db4,L{self.filter_settings.get('wavelet_level')})")
        return " + ".join(parts) if parts else "No filtering"
    
    def _plot_ecg_with_rr(self, ax, ecg_data, fs, title, color='k', show_filter_info=False):
        duration = len(ecg_data) / fs
        t = np.arange(len(ecg_data)) / fs
        ecg_scaled, scale = ECGProcessor.scale_ecg(ecg_data)
        if np.max(np.abs(ecg_scaled)) > 1.8:
            ecg_scaled *= 1.8 / np.max(np.abs(ecg_scaled))
        self._draw_grid(ax, duration)
        ax.plot(t, ecg_scaled, f'{color}-', lw=0.8, label='ECG')
        peaks = ECGProcessor.detect_r_peaks(ecg_data, fs)
        if len(peaks) > 0:
            valid = peaks[peaks < len(ecg_scaled)]
            pt = valid / fs
            pv = ecg_scaled[valid]
            ax.scatter(pt, pv, color='red', s=40, marker='v', label='R-peaks', zorder=10)
            trimmed = ECGProcessor.get_trimmed_rr_indices(valid, trim_count=1)
            for i in trimmed:
                if i > 0:
                    rr_ms = (valid[i] - valid[i-1]) / fs * 1000
                    mid = (pt[i-1] + pt[i]) / 2
                    ax.text(mid, -1.7, f'{rr_ms:.0f}ms', fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='lightgray', alpha=0.9))
        ax.text(0.01, 0.97, f'Scale: {scale:.2e}', transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
        if show_filter_info:
            ax.text(0.01, 0.03, f'Filters: {self._get_filter_description()}', transform=ax.transAxes, 
                   fontsize=7, va='bottom', bbox=dict(boxstyle='round', fc='lightcyan', alpha=0.9))
        ax.set_xlim(0, duration)
        ax.set_ylim(-2.0, 2.0)
        ax.set_ylabel('Amplitude (mV)')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title(title, fontsize=11)
        return peaks, scale
    
    def _generate(self):
        start, end, snr = self._find_best_window(self.ecg, ECG_SAMPLING_RATE, 4)
        
        with PdfPages(self.output_path) as pdf:
            # Page 1: Summary
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Y3X Health Monitor Report', fontsize=18, fontweight='bold', y=0.95)
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            y = 0.85
            ax.text(0.5, y, 'Patient Information', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
            y -= 0.05
            for k, v in self.patient.items():
                ax.text(0.3, y, f'{k}:', ha='right', transform=ax.transAxes)
                ax.text(0.32, y, str(v), ha='left', transform=ax.transAxes)
                y -= 0.03
            
            y -= 0.03
            ax.text(0.5, y, 'Recording Summary', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
            y -= 0.05
            
            if self.ecg is not None:
                ax.text(0.3, y, 'ECG Duration:', ha='right', transform=ax.transAxes)
                ax.text(0.32, y, f'{len(self.ecg)/ECG_SAMPLING_RATE:.1f}s ({len(self.ecg)} samples)', ha='left', transform=ax.transAxes)
                y -= 0.03
            
            for name, key in [('Heart Rate', 'hr'), ('SpO2', 'spo2'), ('Temperature', 'temp'), ('Resp Rate', 'rr')]:
                if key in self.vitals and self.vitals[key] is not None and len(self.vitals[key]) > 0:
                    v = self.vitals[key]
                    ranges = {'hr': (30, 200), 'spo2': (80, 100), 'temp': (30, 45), 'rr': (5, 40)}
                    v = v[(v > ranges[key][0]) & (v < ranges[key][1])]
                    if len(v) > 0:
                        units = {'hr': 'bpm', 'spo2': '%', 'temp': 'C', 'rr': '/min'}
                        ax.text(0.3, y, f'{name}:', ha='right', transform=ax.transAxes)
                        ax.text(0.32, y, f'{np.mean(v):.1f} {units[key]}', ha='left', transform=ax.transAxes)
                        y -= 0.03
            
            y -= 0.03
            ax.text(0.5, y, 'Filter Settings', fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)
            y -= 0.05
            for fname, fkey, fval in [('Bandpass', 'bandpass_enabled', f"{self.filter_settings.get('bandpass_low')}-{self.filter_settings.get('bandpass_high')} Hz"),
                                      ('Notch', 'notch_enabled', f"{self.filter_settings.get('notch_freq')} Hz"),
                                      ('Wavelet', 'wavelet_enabled', f"db4, Level {self.filter_settings.get('wavelet_level')}")]:
                ax.text(0.3, y, f'{fname}:', ha='right', transform=ax.transAxes)
                ax.text(0.32, y, f"Enabled ({fval})" if self.filter_settings.get(fkey) else "Disabled", 
                       ha='left', transform=ax.transAxes, color='black' if self.filter_settings.get(fkey) else 'gray')
                y -= 0.03
            
            y -= 0.04
            ax.text(0.5, y, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                   fontsize=9, ha='center', transform=ax.transAxes, style='italic', color='gray')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: ECG
            if self.ecg is not None and start is not None:
                fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
                fig.suptitle('ECG Analysis', fontsize=14, fontweight='bold')
                ecg_win = self.ecg[start:end]
                self._plot_ecg_with_rr(axes[0], ecg_win, ECG_SAMPLING_RATE, 'Raw ECG', 'k')
                filt = ECGProcessor.filter_ecg(ecg_win, ECG_SAMPLING_RATE, self.filter_settings)
                peaks_f, _ = self._plot_ecg_with_rr(axes[1], filt, ECG_SAMPLING_RATE, 'Filtered ECG', 'b', True)
                if len(peaks_f) > 2:
                    valid_f = peaks_f[peaks_f < len(filt)]
                    trimmed = ECGProcessor.get_trimmed_rr_indices(valid_f, trim_count=1)
                    rr_vals = [(valid_f[i] - valid_f[i-1]) / ECG_SAMPLING_RATE * 1000 for i in trimmed if i > 0]
                    rr_vals = [r for r in rr_vals if 300 < r < 2000]
                    if rr_vals:
                        hr = 60000 / np.mean(rr_vals)
                        axes[1].text(0.99, 0.97, f'HR: {hr:.1f} bpm\nRR: {np.mean(rr_vals):.0f}+/-{np.std(rr_vals):.0f}ms',
                               transform=axes[1].transAxes, fontsize=9, ha='right', va='top',
                               bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.9))
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 3: PPG and BioZ
            plots = []
            if self.ppg is not None and len(self.ppg) > 0:
                plots.append(('PPG', self.ppg, PPG_SAMPLING_RATE, True))
            if self.bioz is not None and len(self.bioz) > 0:
                plots.append(('BioZ', self.bioz, None, False))
            
            if plots:
                fig, axes = plt.subplots(len(plots), 1, figsize=(11, 8.5))
                if len(plots) == 1:
                    axes = [axes]
                fig.suptitle('PPG and Respiration Data', fontsize=14, fontweight='bold')
                for ax, (name, data, fs, show_fs) in zip(axes, plots):
                    samples = min(len(data), (fs or 128) * 10)
                    seg = data[-samples:] - np.mean(data[-samples:])
                    x = np.arange(len(seg)) / fs if fs else np.arange(len(seg))
                    ax.plot(x, seg, 'r-' if 'PPG' in name else 'g-', lw=0.8, label=name)
                    ax.set_title(f'{name} Signal' + (f' ({fs} Hz)' if show_fs else ''), fontsize=11)
                    ax.set_xlabel('Time (s)' if fs else 'Samples')
                    ax.set_ylabel('Amplitude')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 4: Full ECG
            if self.ecg is not None and len(self.ecg) > 0:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                max_pts = 5000
                factor = max(1, len(self.ecg) // max_pts)
                ecg_ds = self.ecg[::factor]
                t = np.arange(len(ecg_ds)) * factor / ECG_SAMPLING_RATE
                ecg_norm, _ = ECGProcessor.scale_ecg(ecg_ds)
                ax.plot(t, ecg_norm, 'b-', lw=0.5, alpha=0.8)
                if start is not None:
                    ax.axvspan(start / ECG_SAMPLING_RATE, end / ECG_SAMPLING_RATE, alpha=0.2, color='green', label='Best SNR Window')
                    ax.legend(loc='upper right')
                ax.set_title(f'Full ECG Recording ({len(self.ecg)/ECG_SAMPLING_RATE:.1f}s)', fontsize=14)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude (normalized)')
                ax.grid(True, alpha=0.3)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 5: Analysis
            if self.ecg is not None:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.text(0.5, 0.95, 'ECG Analysis Report', fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
                peaks_full = ECGProcessor.detect_r_peaks(self.ecg, ECG_SAMPLING_RATE)
                y = 0.85
                if len(peaks_full) > 2:
                    trimmed = ECGProcessor.get_trimmed_rr_indices(peaks_full, trim_count=1)
                    rr_vals = [(peaks_full[i] - peaks_full[i-1]) / ECG_SAMPLING_RATE * 1000 for i in trimmed if i > 0]
                    rr = np.array([r for r in rr_vals if 300 < r < 2000])
                    if len(rr) > 0:
                        hr = 60000 / np.mean(rr)
                        _, scale = ECGProcessor.scale_ecg(self.ecg)
                        ax.text(0.1, y, 'ECG Analysis:', fontsize=12, fontweight='bold', transform=ax.transAxes)
                        y -= 0.05
                        for line in [f'  Heart Rate: {hr:.1f} bpm', f'  RR Interval: {np.mean(rr):.0f} +/- {np.std(rr):.1f} ms',
                                    f'  RR Range: {np.min(rr):.0f} - {np.max(rr):.0f} ms', f'  Total Beats: {len(peaks_full)}',
                                    f'  Scale Factor: {scale:.2e}']:
                            ax.text(0.1, y, line, fontsize=11, fontfamily='monospace', transform=ax.transAxes)
                            y -= 0.04
                        y -= 0.03
                        ax.text(0.1, y, 'Filter Settings Used:', fontsize=12, fontweight='bold', transform=ax.transAxes)
                        y -= 0.05
                        ax.text(0.1, y, f'  {self._get_filter_description()}', fontsize=11, fontfamily='monospace', transform=ax.transAxes)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            d = pdf.infodict()
            d['Title'] = 'Y3X Health Report'
            d['CreationDate'] = datetime.now()

# ============================================================================
# BLE
# ============================================================================
class BLEClient:
    UUID = "0000{}-0000-1000-8000-00805f9b34fb"
    CHARS = {
        'spo2': UUID.format("2A5E"), 'temp': UUID.format("2A6E"),
        'hr': UUID.format("2A37"), 'bat': UUID.format("2A19"),
        'ecg': "00001424-0000-1000-8000-00805f9b34fb",
        'resp': "babe4a4c-7789-11ed-a1eb-0242ac120002",
        'ppg': "cd5c1525-4448-7db8-ae4c-d1da8cba36d0",
        'rr': "cd5ca86f-4448-7db8-ae4c-d1da8cba36d0"
    }
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.address = None
        self.callback = None
    
    async def connect(self):
        devices = await BleakScanner.discover()
        dev = next((d for d in devices if d.address.upper() == self.address.upper()), None)
        if not dev:
            raise Exception(f"Device {self.address} not found")
        self.client = BleakClient(dev.address)
        await self.client.connect()
        self.connected = True
    
    def _emit(self, data):
        if self.callback:
            self.callback(data)
    
    async def start(self):
        if not self.connected:
            raise Exception("Not connected")
        
        def make_cb(dtype):
            if dtype == 'spo2':
                return lambda s, d: self._emit({'type': 'SPO2', 'spo2': struct.unpack('<H', d[1:3])[0]})
            elif dtype == 'temp':
                return lambda s, d: self._emit({'type': 'TEMP', 'temperature': ((d[1] << 8) | d[0]) / 100.0})
            elif dtype == 'ecg':
                return lambda s, d: self._emit({'type': 'ECG', 'samples': [struct.unpack('<i', d[i:i+4])[0] for i in range(0, len(d), 4)]})
            elif dtype == 'resp':
                return lambda s, d: self._emit({'type': 'RESP', 'samples': [struct.unpack('<i', d[i:i+4])[0] for i in range(0, len(d), 4)]})
            elif dtype == 'ppg':
                return lambda s, d: self._emit({'type': 'PPG', 'value': struct.unpack('<h', d)[0]})
            elif dtype == 'rr':
                return lambda s, d: self._emit({'type': 'RR', 'resp_rate': struct.unpack('<H', d)[0]})
            elif dtype == 'hr':
                return lambda s, d: self._emit({'type': 'HR', 'heart_rate': struct.unpack('<H', d[1:3])[0] if d[0] & 0x01 else d[1]})
            elif dtype == 'bat':
                return lambda s, d: self._emit({'type': 'BAT', 'level': d[0]})
        
        for service in self.client.services:
            for char in service.characteristics:
                for name, uuid in self.CHARS.items():
                    if char.uuid.lower() == uuid.lower():
                        try:
                            await self.client.start_notify(char, make_cb(name))
                        except:
                            pass
                        break
    
    async def disconnect(self):
        if self.connected:
            await self.client.disconnect()
            self.connected = False


class BLEScanner(QThread):
    found = pyqtSignal(list)
    
    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            devices = loop.run_until_complete(BleakScanner.discover())
            self.found.emit([(d.address, d.name or "Unknown") for d in devices])
        except:
            self.found.emit([])
        finally:
            loop.close()


class BLEThread(QThread):
    data = pyqtSignal(dict)
    status = pyqtSignal(str)
    
    def __init__(self, address):
        super().__init__()
        self.client = BLEClient()
        self.client.address = address
        self.client.callback = lambda d: self.data.emit(d)
        self.running = False
    
    def run(self):
        self.running = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            loop.close()
    
    async def _run(self):
        try:
            await self.client.connect()
            await self.client.start()
            self.status.emit(f"Connected to {self.client.address}")
            while self.running:
                await asyncio.sleep(0.1)
        except Exception as e:
            self.status.emit(f"Error: {e}")
        finally:
            if self.client.connected:
                await self.client.disconnect()
    
    def stop(self):
        self.running = False

# ============================================================================
# LOGGING THREAD - Serial format, txt then rename to csv
# ============================================================================
class LogThread(QThread):
    def __init__(self, q, name, path):
        super().__init__()
        self.q = q
        self.name = name
        self.path = path
        self.running = False
        self.files = {}
        self.file_paths = {}
    
    def run(self):
        self.running = True
        os.makedirs(self.path, exist_ok=True)
        data_types = ['ecg', 'bioz', 'ppg', 'temp', 'hr', 'spo2', 'rr']
        
        try:
            for dtype in data_types:
                txt_path = os.path.join(self.path, f"{self.name}_{dtype}.txt")
                self.file_paths[dtype] = txt_path
                self.files[dtype] = open(txt_path, 'w', buffering=8192)
                self.files[dtype].write("Timestamp,Data\n")
            
            while self.running:
                try:
                    d = self.q.get(timeout=0.1)
                    if d is None:
                        break
                    self._write(d)
                except:
                    continue
        finally:
            for f in self.files.values():
                if f and not f.closed:
                    f.close()
            self._convert_to_csv()
    
    def _write(self, d):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        t = d.get('type')
        
        if t == 'ECG':
            samples = d.get('samples', [])
            f = self.files.get('ecg')
            if f and samples:
                for s in samples:
                    f.write(f"{ts},{s}\n")
        elif t == 'RESP':
            samples = d.get('samples', [])
            f = self.files.get('bioz')
            if f and samples:
                for s in samples:
                    f.write(f"{ts},{s}\n")
        elif t == 'PPG':
            v = d.get('value')
            f = self.files.get('ppg')
            if f and v is not None:
                f.write(f"{ts},{v}\n")
        elif t == 'TEMP':
            v = d.get('temperature')
            f = self.files.get('temp')
            if f and v is not None:
                f.write(f"{ts},{v:.2f}\n")
        elif t == 'HR':
            v = d.get('heart_rate')
            f = self.files.get('hr')
            if f and v is not None:
                f.write(f"{ts},{v}\n")
        elif t == 'SPO2':
            v = d.get('spo2')
            f = self.files.get('spo2')
            if f and v is not None:
                f.write(f"{ts},{v}\n")
        elif t == 'RR':
            v = d.get('resp_rate')
            f = self.files.get('rr')
            if f and v is not None:
                f.write(f"{ts},{v}\n")
    
    def _convert_to_csv(self):
        for dtype, txt_path in self.file_paths.items():
            if os.path.exists(txt_path):
                csv_path = txt_path.replace('.txt', '.csv')
                try:
                    os.rename(txt_path, csv_path)
                except:
                    pass
    
    def stop(self):
        self.running = False

# ============================================================================
# OPTIMIZED PLOT WIDGET
# ============================================================================
class PlotWidget(QFrame):
    def __init__(self, title, color='#00ff00'):
        super().__init__()
        self.setFrameStyle(QFrame.NoFrame)
        self.buf = None
        self.title = title
        self.color = color
        self.initialized = False
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumSize(200, 100)
        self.plot_widget.setBackground('#1a1a1a')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.hideButtons()
        self.plot_widget.getPlotItem().setClipToView(True)
        self.plot_widget.setDownsampling(auto=True, mode='peak')
        
        self.curve = None
        layout.addWidget(self.plot_widget)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 100)
    
    def showEvent(self, event):
        """Initialize plot after widget is shown to avoid QPainter errors"""
        super().showEvent(event)
        if not self.initialized:
            self.initialized = True
            self.plot_widget.setTitle(self.title, color='#aaa', size='9pt')
            self.curve = self.plot_widget.plot(pen=pg.mkPen(self.color, width=1))
            self.plot_widget.enableAutoRange(axis='y')
            self.plot_widget.setAutoVisible(y=True)
    
    def set_buffer(self, b):
        self.buf = b
    
    def update_plot(self):
        if not self.buf or not self.initialized or not self.curve:
            return
        data = self.buf.get_data()
        if len(data) == 0:
            return
        self.curve.setData(data)
    
    def clear(self):
        if self.curve:
            self.curve.setData([])
        if self.buf:
            self.buf.clear()

# ============================================================================
# UI WIDGETS
# ============================================================================
class VitalWidget(QFrame):
    def __init__(self, label):
        super().__init__()
        self.setStyleSheet("QFrame{background:#252525;border:1px solid #333;border-radius:6px}")
        self.setMinimumSize(100, 70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        
        title = QLabel(label)
        title.setStyleSheet("color:#777;font-size:10px;font-weight:bold;border:none")
        title.setAlignment(Qt.AlignCenter)
        
        self.value = QLabel("--")
        self.value.setStyleSheet("color:#0f0;font-size:24px;font-weight:bold;border:none")
        self.value.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(self.value)
    
    def set(self, v, color=None):
        self.value.setText(f"{v:.1f}" if isinstance(v, float) else str(v))
        if color:
            self.value.setStyleSheet(f"color:{color};font-size:24px;font-weight:bold;border:none")
    
    def clear(self):
        self.value.setText("--")


class BatteryWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        lbl = QLabel("Battery:")
        lbl.setStyleSheet("color:#777")
        self.level = QLabel("--%")
        self.level.setStyleSheet("color:#777;font-weight:bold")
        layout.addWidget(lbl)
        layout.addWidget(self.level)
        layout.addStretch()
    
    def set(self, v):
        c = "#f44" if v <= 20 else "#fa0" if v <= 50 else "#4f4"
        self.level.setText(f"{v}%")
        self.level.setStyleSheet(f"color:{c};font-weight:bold")


class FilterWidget(QGroupBox):
    def __init__(self):
        super().__init__("Filters")
        self.setStyleSheet("""
            QGroupBox{background:#252525;border:1px solid #333;border-radius:6px;color:#aaa;font-weight:bold;padding-top:14px;margin-top:6px}
            QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px}
            QCheckBox{color:#888}
            QCheckBox::indicator{width:14px;height:14px}
            QCheckBox::indicator:checked{background:#4a4;border:1px solid #333;border-radius:2px}
            QCheckBox::indicator:unchecked{background:#333;border:1px solid #444;border-radius:2px}
            QSpinBox,QDoubleSpinBox{background:#333;color:#ccc;border:1px solid #444;border-radius:3px;padding:2px;max-width:50px}
            QLabel{color:#666;border:none}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 16, 8, 8)
        
        # Bandpass
        h1 = QHBoxLayout()
        self.bp_check = QCheckBox("BP")
        self.bp_check.setChecked(True)
        self.bp_low = QDoubleSpinBox()
        self.bp_low.setRange(0.1, 10)
        self.bp_low.setValue(0.5)
        self.bp_low.setDecimals(1)
        self.bp_high = QDoubleSpinBox()
        self.bp_high.setRange(10, 100)
        self.bp_high.setValue(45)
        self.bp_high.setDecimals(0)
        h1.addWidget(self.bp_check)
        h1.addWidget(self.bp_low)
        h1.addWidget(QLabel("-"))
        h1.addWidget(self.bp_high)
        h1.addWidget(QLabel("Hz"))
        h1.addStretch()
        layout.addLayout(h1)
        
        # Notch
        h2 = QHBoxLayout()
        self.notch_check = QCheckBox("Notch")
        self.notch_check.setChecked(True)
        self.notch_freq = QDoubleSpinBox()
        self.notch_freq.setRange(40, 60)
        self.notch_freq.setValue(50)
        self.notch_freq.setDecimals(0)
        h2.addWidget(self.notch_check)
        h2.addWidget(self.notch_freq)
        h2.addWidget(QLabel("Hz"))
        h2.addStretch()
        layout.addLayout(h2)
        
        # Wavelet
        h3 = QHBoxLayout()
        self.wav_check = QCheckBox("Wavelet")
        self.wav_check.setChecked(True)
        self.wav_level = QSpinBox()
        self.wav_level.setRange(1, 8)
        self.wav_level.setValue(4)
        h3.addWidget(self.wav_check)
        h3.addWidget(QLabel("L:"))
        h3.addWidget(self.wav_level)
        h3.addStretch()
        layout.addLayout(h3)
    
    def get_settings(self):
        return {
            'bandpass_enabled': self.bp_check.isChecked(),
            'bandpass_low': self.bp_low.value(),
            'bandpass_high': self.bp_high.value(),
            'notch_enabled': self.notch_check.isChecked(),
            'notch_freq': self.notch_freq.value(),
            'wavelet_enabled': self.wav_check.isChecked(),
            'wavelet_level': self.wav_level.value()
        }


class ControlWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QFrame{background:#252525;border:1px solid #333;border-radius:6px}
            QPushButton{background:#333;color:#ccc;border:1px solid #444;border-radius:4px;padding:6px 10px;font-weight:bold}
            QPushButton:hover{background:#444}
            QPushButton:disabled{background:#222;color:#555}
            QComboBox{background:#333;color:#ccc;border:1px solid #444;border-radius:4px;padding:4px 8px}
            QLabel{color:#777;border:none}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.dir_label = QLabel("No directory")
        self.dir_label.setWordWrap(True)
        self.dir_label.setStyleSheet("color:#666;font-size:9px;border:none")
        layout.addWidget(self.dir_label)
        
        self.dir_btn = QPushButton("Select Directory")
        layout.addWidget(self.dir_btn)
        
        layout.addWidget(QLabel("BLE Device"))
        self.scan_btn = QPushButton("Scan")
        layout.addWidget(self.scan_btn)
        
        self.device_combo = QComboBox()
        self.device_combo.addItem("Select device...")
        layout.addWidget(self.device_combo)
        
        h = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setEnabled(False)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        h.addWidget(self.connect_btn)
        h.addWidget(self.disconnect_btn)
        layout.addLayout(h)
        
        self.log_btn = QPushButton("Start Logging")
        layout.addWidget(self.log_btn)
        
        self.report_btn = QPushButton("Generate Report")
        layout.addWidget(self.report_btn)
        
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color:#666;font-size:9px;border:none")
        layout.addWidget(self.status)
        
        layout.addStretch()
    
    def set_status(self, msg, color="#666"):
        self.status.setText(msg)
        self.status.setStyleSheet(f"color:{color};font-size:9px;border:none")


class PatientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Patient Info")
        self.setModal(True)
        self.setMinimumWidth(350)
        self.setStyleSheet("""
            QDialog{background:#2a2a2a}
            QLabel{color:#ccc;font-weight:bold}
            QLineEdit,QComboBox,QTextEdit{background:#333;color:#ccc;border:1px solid #444;border-radius:4px;padding:6px}
            QPushButton{background:#333;color:#ccc;border:1px solid #444;border-radius:4px;padding:8px 14px;font-weight:bold}
            QPushButton:hover{background:#444}
            QPushButton#dummy{background:#443;color:#fa0}
            QPushButton#dummy:hover{background:#554}
        """)
        
        layout = QVBoxLayout(self)
        
        h = QHBoxLayout()
        h.addStretch()
        self.dummy_btn = QPushButton("Fill Dummy")
        self.dummy_btn.setObjectName("dummy")
        self.dummy_btn.clicked.connect(self._fill_dummy)
        h.addWidget(self.dummy_btn)
        layout.addLayout(h)
        
        form = QFormLayout()
        self.name = QLineEdit()
        self.age = QLineEdit()
        self.age.setValidator(QIntValidator(0, 150))
        self.gender = QComboBox()
        self.gender.addItems(["Male", "Female", "Other"])
        self.weight = QLineEdit()
        self.weight.setValidator(QDoubleValidator(0, 500, 2))
        self.height = QLineEdit()
        self.height.setValidator(QDoubleValidator(0, 300, 2))
        self.diseases = QTextEdit()
        self.diseases.setMaximumHeight(50)
        
        form.addRow("Name*:", self.name)
        form.addRow("Age*:", self.age)
        form.addRow("Gender:", self.gender)
        form.addRow("Weight:", self.weight)
        form.addRow("Height:", self.height)
        form.addRow("Diseases:", self.diseases)
        layout.addLayout(form)
        
        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("OK")
        ok.clicked.connect(self._ok)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        layout.addLayout(btns)
    
    def _fill_dummy(self):
        import random
        self.name.setText(f"Test_{random.randint(1000,9999)}")
        self.age.setText(str(random.randint(20, 60)))
        self.gender.setCurrentIndex(random.randint(0, 1))
        self.weight.setText(str(random.randint(50, 90)))
        self.height.setText(str(random.randint(150, 185)))
        self.diseases.setText("None (Test)")
    
    def _ok(self):
        if not self.name.text().strip() or not self.age.text().strip():
            QMessageBox.warning(self, "Error", "Name and Age required")
            return
        self.accept()
    
    def get(self):
        return {
            'name': self.name.text().strip(),
            'age': self.age.text().strip(),
            'gender': self.gender.currentText(),
            'weight': self.weight.text().strip() or 'N/A',
            'height': self.height.text().strip() or 'N/A',
            'diseases': self.diseases.toPlainText().strip() or 'None'
        }


class SessionDialog(QDialog):
    def __init__(self, base_dir, parent=None):
        super().__init__(parent)
        self.base_dir = base_dir
        self.selected = None
        self.setWindowTitle("Select Session")
        self.setModal(True)
        self.setMinimumSize(380, 280)
        self.setStyleSheet("""
            QDialog{background:#2a2a2a}
            QLabel{color:#ccc;font-weight:bold}
            QListWidget{background:#333;color:#ccc;border:1px solid #444;border-radius:4px}
            QListWidget::item{padding:6px}
            QListWidget::item:selected{background:#4a6a8a}
            QPushButton{background:#333;color:#ccc;border:1px solid #444;border-radius:4px;padding:8px 14px;font-weight:bold}
            QPushButton:hover{background:#444}
        """)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select session:"))
        
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._sel)
        layout.addWidget(self.list)
        
        btns = QHBoxLayout()
        btns.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._sel)
        btns.addWidget(cancel)
        btns.addWidget(self.gen_btn)
        layout.addLayout(btns)
        
        self.list.itemSelectionChanged.connect(lambda: self.gen_btn.setEnabled(bool(self.list.selectedItems())))
        self._load()
    
    def _load(self):
        if not os.path.exists(self.base_dir):
            return
        for item in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, item)
            if os.path.isdir(path):
                files = os.listdir(path)
                if any('_ecg.csv' in f for f in files) or 'patient_info.txt' in files:
                    li = QListWidgetItem(item)
                    li.setData(Qt.UserRole, path)
                    self.list.addItem(li)
    
    def _sel(self):
        sel = self.list.selectedItems()
        if sel:
            self.selected = sel[0].data(Qt.UserRole)
            self.accept()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Y3X Data Logger")
        self.setGeometry(100, 100, 1100, 750)
        self.setMinimumSize(800, 500)
        self.setStyleSheet("QMainWindow{background:#1a1a1a}QWidget{background:transparent;color:#ccc}")
        
        # Buffers
        self.ecg_buf = CircularBuffer(PLOT_BUFFER_SIZE)
        self.bioz_buf = CircularBuffer(PLOT_BUFFER_SIZE)
        self.ppg_buf = CircularBuffer(PLOT_BUFFER_SIZE)
        
        # State
        self.save_dir = None
        self.q = queue.Queue(maxsize=DATA_QUEUE_SIZE)
        self.ble = None
        self.log = None
        self.scan = None
        self.report = None
        self.logging = False
        
        self._setup_ui()
        self._connect_signals()
        
        # Timer for plot updates - start after window is shown
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_plots)
    
    def showEvent(self, event):
        """Start timer after window is shown to avoid QPainter errors"""
        super().showEvent(event)
        if not self.timer.isActive():
            QTimer.singleShot(100, lambda: self.timer.start(PLOT_UPDATE_MS))
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Left: Plots (using splitter for resizable)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        self.ecg_plot = PlotWidget(f"ECG ({ECG_SAMPLING_RATE} Hz)", '#00ff00')
        self.ecg_plot.set_buffer(self.ecg_buf)
        
        self.bioz_plot = PlotWidget("Respiration (BioZ)", '#ff6666')
        self.bioz_plot.set_buffer(self.bioz_buf)
        
        self.ppg_plot = PlotWidget(f"PPG ({PPG_SAMPLING_RATE} Hz)", '#66ffff')
        self.ppg_plot.set_buffer(self.ppg_buf)
        
        left_layout.addWidget(self.ecg_plot, 1)
        left_layout.addWidget(self.bioz_plot, 1)
        left_layout.addWidget(self.ppg_plot, 1)
        
        # Right: Controls
        right_widget = QWidget()
        right_widget.setFixedWidth(240)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        
        # Vitals grid
        vitals = QWidget()
        vgrid = QGridLayout(vitals)
        vgrid.setSpacing(6)
        vgrid.setContentsMargins(0, 0, 0, 0)
        
        self.hr_widget = VitalWidget("HR (bpm)")
        self.spo2_widget = VitalWidget("SpO2 (%)")
        self.temp_widget = VitalWidget("Temp (C)")
        self.rr_widget = VitalWidget("RR (/min)")
        
        vgrid.addWidget(self.hr_widget, 0, 0)
        vgrid.addWidget(self.spo2_widget, 0, 1)
        vgrid.addWidget(self.temp_widget, 1, 0)
        vgrid.addWidget(self.rr_widget, 1, 1)
        
        self.battery = BatteryWidget()
        self.filters = FilterWidget()
        self.controls = ControlWidget()
        
        right_layout.addWidget(vitals)
        right_layout.addWidget(self.battery)
        right_layout.addWidget(self.filters)
        right_layout.addWidget(self.controls, 1)
        
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 0)
        
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("background:#252525;color:#777")
    
    def _connect_signals(self):
        self.controls.dir_btn.clicked.connect(self._select_dir)
        self.controls.scan_btn.clicked.connect(self._scan)
        self.controls.connect_btn.clicked.connect(self._connect)
        self.controls.disconnect_btn.clicked.connect(self._disconnect)
        self.controls.log_btn.clicked.connect(self._toggle_log)
        self.controls.report_btn.clicked.connect(self._gen_report)
        self.controls.device_combo.currentIndexChanged.connect(
            lambda i: self.controls.connect_btn.setEnabled(i > 0))
    
    def _update_plots(self):
        self.ecg_plot.update_plot()
        self.bioz_plot.update_plot()
        self.ppg_plot.update_plot()
    
    def _select_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        if d:
            self.save_dir = d
            self.controls.dir_label.setText(d)
            self.statusBar().showMessage(f"Directory: {d}")
    
    def _scan(self):
        self.controls.scan_btn.setEnabled(False)
        self.controls.device_combo.clear()
        self.controls.device_combo.addItem("Scanning...")
        self.controls.set_status("Scanning...", "#fa0")
        self.scan = BLEScanner()
        self.scan.found.connect(self._on_devices)
        self.scan.start()
    
    def _on_devices(self, devices):
        self.controls.device_combo.clear()
        self.controls.device_combo.addItem("Select device...")
        for addr, name in devices:
            self.controls.device_combo.addItem(f"{name} ({addr})", addr)
        self.controls.scan_btn.setEnabled(True)
        self.controls.set_status(f"Found {len(devices)}", "#4f4")
    
    def _connect(self):
        if self.controls.device_combo.currentIndex() <= 0:
            return
        addr = self.controls.device_combo.currentData()
        self.controls.set_status("Connecting...", "#fa0")
        self.ble = BLEThread(addr)
        self.ble.data.connect(self._on_data, Qt.QueuedConnection)
        self.ble.status.connect(self._on_status)
        self.ble.start()
        self.controls.scan_btn.setEnabled(False)
        self.controls.device_combo.setEnabled(False)
        self.controls.connect_btn.setEnabled(False)
        self.controls.disconnect_btn.setEnabled(True)
    
    def _disconnect(self):
        if self.ble:
            if self.logging:
                self._stop_log()
            self.ble.stop()
            self.ble.wait()
            self.ble = None
            self._clear()
            self.controls.scan_btn.setEnabled(True)
            self.controls.device_combo.setEnabled(True)
            self.controls.disconnect_btn.setEnabled(False)
            self.controls.set_status("Disconnected", "#777")
    
    def _clear(self):
        self.hr_widget.clear()
        self.spo2_widget.clear()
        self.temp_widget.clear()
        self.rr_widget.clear()
        self.ecg_plot.clear()
        self.bioz_plot.clear()
        self.ppg_plot.clear()
    
    def _toggle_log(self):
        if self.logging:
            self._stop_log()
        else:
            self._start_log()
    
    def _start_log(self):
        if not self.save_dir:
            QMessageBox.warning(self, "Error", "Select directory first")
            return
        if not self.ble or not self.ble.running:
            QMessageBox.warning(self, "Error", "Connect to device first")
            return
        
        dlg = PatientDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        
        info = dlg.get()
        name = f"{info['name']}_{info['age']}_{info['weight']}kg"
        name = "".join(c for c in name if c.isalnum() or c in "._- ")
        path = os.path.join(self.save_dir, name)
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "patient_info.txt"), 'w') as f:
            for k, v in info.items():
                f.write(f"{k.title()}: {v}\n")
            f.write(f"Recording Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except:
                break
        
        session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log = LogThread(self.q, session, path)
        self.log.start()
        
        self.logging = True
        self.controls.log_btn.setText("Stop Logging")
        self.controls.set_status("Logging...", "#4f4")
        self.statusBar().showMessage(f"Logging: {path}")
    
    def _stop_log(self):
        if self.log:
            self.q.put(None)
            self.log.stop()
            self.log.wait()
            self.log = None
            self.logging = False
            self.controls.log_btn.setText("Start Logging")
            self.controls.set_status("Stopped", "#777")
            self.statusBar().showMessage("Logging stopped - CSV files created")
    
    def _gen_report(self):
        if not self.save_dir:
            QMessageBox.warning(self, "Error", "Select directory first")
            return
        
        dlg = SessionDialog(self.save_dir, self)
        if dlg.exec_() != QDialog.Accepted or not dlg.selected:
            return
        
        out, _ = QFileDialog.getSaveFileName(self, "Save Report",
            os.path.join(self.save_dir, os.path.basename(dlg.selected) + "_report.pdf"), "PDF (*.pdf)")
        if not out:
            return
        
        self.controls.set_status("Generating...", "#fa0")
        self.controls.report_btn.setEnabled(False)
        
        self.report = ReportThread(dlg.selected, out, self.filters.get_settings())
        self.report.progress.connect(lambda m: self.statusBar().showMessage(m))
        self.report.finished.connect(self._on_report_done)
        self.report.start()
    
    def _on_report_done(self, ok, msg):
        self.controls.report_btn.setEnabled(True)
        if ok:
            self.controls.set_status("Done", "#4f4")
            self.statusBar().showMessage(f"Saved: {msg}")
            QMessageBox.information(self, "Success", f"Report saved:\n{msg}")
        else:
            self.controls.set_status("Failed", "#f44")
            QMessageBox.critical(self, "Error", f"Failed:\n{msg}")
    
    def _on_data(self, d):
        if not self.q.full():
            self.q.put(d.copy())
        
        t = d.get('type')
        if t == 'ECG':
            samples = d.get('samples', [])
            if samples:
                mx = max(abs(x) for x in samples) if samples else 1
                sc = 1000.0 if mx > 100000 else 100.0
                self.ecg_buf.extend([x / sc for x in samples])
        elif t == 'RESP':
            samples = d.get('samples', [])
            if samples:
                mx = max(abs(x) for x in samples) if samples else 1
                sc = 1000.0 if mx > 100000 else 100.0
                self.bioz_buf.extend([x / sc for x in samples])
        elif t == 'PPG':
            v = d.get('value')
            if v is not None:
                self.ppg_buf.append(v / 100.0)
        elif t == 'HR':
            v = d.get('heart_rate')
            if v:
                self.hr_widget.set(v, "#fa0" if v < 60 else "#f66" if v > 100 else "#4f4")
        elif t == 'SPO2':
            v = d.get('spo2')
            if v:
                self.spo2_widget.set(v, "#f66" if v < 95 else "#4f4")
        elif t == 'TEMP':
            v = d.get('temperature')
            if v:
                self.temp_widget.set(v, "#f66" if v > 37.5 else "#fa0" if v < 36 else "#4f4")
        elif t == 'RR':
            v = d.get('resp_rate')
            if v:
                self.rr_widget.set(v, "#fa0" if v < 12 or v > 20 else "#4f4")
        elif t == 'BAT':
            v = d.get('level')
            if v is not None:
                self.battery.set(v)
    
    def _on_status(self, msg):
        if "Connected" in msg:
            self.controls.set_status("Connected", "#4f4")
        elif "Error" in msg:
            self.controls.set_status("Error", "#f44")
        self.statusBar().showMessage(msg)
    
    def closeEvent(self, e):
        self.timer.stop()
        if self.log and self.log.isRunning():
            self.q.put(None)
            self.log.stop()
            self.log.wait()
        if self.ble and self.ble.isRunning():
            self.ble.stop()
            self.ble.wait()
        if self.report and self.report.isRunning():
            self.report.wait()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark palette
    from PyQt5.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(26, 26, 26))
    palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
    palette.setColor(QPalette.Base, QColor(40, 40, 40))
    palette.setColor(QPalette.Text, QColor(200, 200, 200))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    palette.setColor(QPalette.Highlight, QColor(70, 100, 130))
    app.setPalette(palette)
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()