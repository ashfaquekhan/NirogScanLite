#!/usr/bin/env python3
import sys
import serial
import numpy as np
import time
import threading
from collections import deque

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                QWidget, QLabel, QTextEdit, QSplitter, QFrame, QPushButton)
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    import OpenGL.GL as gl
    AVAILABLE_QT = True
except ImportError:
    AVAILABLE_QT = False
    print("Install PyQt6: pip install PyQt6 PyOpenGL")

try:
    from pyPPG import PPG, Fiducials, Biomarkers
    import pyPPG.preproc as PP
    import pyPPG.fiducials as FP
    import pyPPG.biomarkers as BM
    import pyPPG.ppg_sqi as SQI
    AVAILABLE_PYPPG = True
except ImportError:
    AVAILABLE_PYPPG = False
    print("Install pyPPG: pip install pyPPG")

class PPGProcessor:
    """PPG processor using pyPPG library"""
    
    def __init__(self, fs=100):
        self.fs = fs
        self.setup_preprocessing()
        
    def setup_preprocessing(self):
        """Setup pyPPG preprocessing"""
        self.preprocessor = PP.Preprocess(
            fL=0.5000001,
            fH=12,
            order=4,
            sm_wins={'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10}
        )
        
    def analyze_signal(self, signal_data):
        """Analyze PPG signal using pyPPG"""
        if len(signal_data) < 200:
            return None, None, None
            
        try:
            # Create PPG signal object
            ppg_signal = PPG(v=signal_data, fs=self.fs, filtering=True)
            
            # Preprocess signal
            s = self.preprocessor.get_signals(s=ppg_signal)
            
            # Extract fiducial points
            fp_detector = FP.FiducialPoints(s=s)
            fiducials = fp_detector.get_fp()
            fp = Fiducials(fp=fiducials)
            
            # Calculate biomarkers
            bmex = BM.BmCollection(s=s, fp=fp)
            bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
            bm = Biomarkers(bm_defs=bm_defs, bm_vals=bm_vals, bm_stats=bm_stats)
            
            # Get signal quality index
            ppg_sqi = None
            try:
                if len(fp.sp) > 0:
                    sqi_scores = SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)
                    ppg_sqi = np.mean(sqi_scores) * 100
            except:
                ppg_sqi = 0
            
            return s, fp, bm, ppg_sqi
            
        except Exception as e:
            print(f"pyPPG analysis error: {e}")
            return None, None, None, None

class BiomarkerExtractor:
    """Extract comprehensive biomarkers from pyPPG results"""
    
    def __init__(self):
        self.ir_history = deque(maxlen=100)
        self.red_history = deque(maxlen=100)
        
    def update_data(self, ir_results, red_results):
        """Update analysis history"""
        if ir_results:
            self.ir_history.append(ir_results)
        if red_results:
            self.red_history.append(red_results)
    
    def calculate_spo2(self):
        """Calculate SpO2 from dual wavelength"""
        if len(self.ir_history) < 3 or len(self.red_history) < 3:
            return 0
            
        try:
            # Get recent biomarker values
            ir_recent = list(self.ir_history)[-3:]
            red_recent = list(self.red_history)[-3:]
            
            # Extract AC and DC components from biomarkers
            ir_amps = []
            ir_areas = []
            red_amps = []
            red_areas = []
            
            for result in ir_recent:
                s, fp, bm, sqi = result
                if bm and hasattr(bm, 'bm_vals'):
                    # Get systolic amplitude and pulse area
                    vals = bm.bm_vals
                    if 'Sp' in vals:  # Systolic peak amplitude
                        ir_amps.append(vals['Sp'])
                    if 'PA' in vals:  # Pulse area (represents DC)
                        ir_areas.append(vals['PA'])
            
            for result in red_recent:
                s, fp, bm, sqi = result
                if bm and hasattr(bm, 'bm_vals'):
                    vals = bm.bm_vals
                    if 'Sp' in vals:
                        red_amps.append(vals['Sp'])
                    if 'PA' in vals:
                        red_areas.append(vals['PA'])
            
            if len(ir_amps) > 0 and len(red_amps) > 0 and len(ir_areas) > 0 and len(red_areas) > 0:
                # Calculate R ratio
                ir_ac = np.mean(ir_amps)
                ir_dc = np.mean(ir_areas)
                red_ac = np.mean(red_amps)
                red_dc = np.mean(red_areas)
                
                if ir_dc > 0 and red_dc > 0:
                    R = (red_ac / red_dc) / (ir_ac / ir_dc)
                    
                    # Empirical calibration
                    if R < 0.4:
                        spo2 = 100 - 25 * R
                    elif R < 3.4:
                        spo2 = 110 - 25 * R
                    else:
                        spo2 = 85 - 15 * R
                        
                    return max(70, min(100, spo2))
        except:
            pass
        
        return 0
    
    def get_comprehensive_biomarkers(self):
        """Get all biomarkers"""
        if len(self.ir_history) == 0:
            return {}
            
        # Get latest IR analysis
        latest_ir = self.ir_history[-1] if self.ir_history else None
        latest_red = self.red_history[-1] if self.red_history else None
        
        biomarkers = {}
        
        # Basic vital signs
        biomarkers['spo2'] = self.calculate_spo2()
        biomarkers['heart_rate'] = 0
        biomarkers['signal_quality_ir'] = 0
        biomarkers['signal_quality_red'] = 0
        
        # Extract from IR channel
        if latest_ir:
            s, fp, bm, sqi = latest_ir
            biomarkers['signal_quality_ir'] = sqi if sqi else 0
            
            if bm and hasattr(bm, 'bm_vals'):
                vals = bm.bm_vals
                
                # Heart rate from RR intervals
                if 'RR' in vals:
                    biomarkers['heart_rate'] = 60 / (vals['RR'] / 1000) if vals['RR'] > 0 else 0
                
                # Pulse characteristics
                biomarkers['systolic_amplitude'] = vals.get('Sp', 0)
                biomarkers['pulse_width'] = vals.get('PW', 0)
                biomarkers['pulse_area'] = vals.get('PA', 0)
                biomarkers['dicrotic_notch_amplitude'] = vals.get('Dn', 0)
                
                # Timing features
                biomarkers['pulse_onset_time'] = vals.get('PPG_y_on', 0)
                biomarkers['systolic_peak_time'] = vals.get('PPG_y_sp', 0)
                biomarkers['dicrotic_notch_time'] = vals.get('PPG_y_dn', 0)
                
                # Advanced features
                biomarkers['reflection_index'] = vals.get('RI', 0)
                biomarkers['stiffness_index'] = vals.get('SI', 0)
                biomarkers['augmentation_index'] = vals.get('AI', 0)
                
                # HRV features
                if bm.bm_stats:
                    stats = bm.bm_stats
                    if 'RR' in stats:
                        rr_stats = stats['RR']
                        biomarkers['hrv_mean'] = rr_stats.get('AVG', 0)
                        biomarkers['hrv_std'] = rr_stats.get('STD', 0)
                        biomarkers['hrv_rmssd'] = rr_stats.get('MAD', 0)
        
        # Extract from RED channel
        if latest_red:
            s, fp, bm, sqi = latest_red
            biomarkers['signal_quality_red'] = sqi if sqi else 0
            
            if bm and hasattr(bm, 'bm_vals'):
                vals = bm.bm_vals
                biomarkers['red_systolic_amplitude'] = vals.get('Sp', 0)
                biomarkers['red_pulse_area'] = vals.get('PA', 0)
        
        return biomarkers

class SerialCollector:
    """Serial data collection"""
    
    def __init__(self, port="COM10"):
        self.port = port
        self.serial_conn = None
        self.running = False
        
        self.ir_buffer = deque(maxlen=2000)
        self.red_buffer = deque(maxlen=2000)
        
    def connect(self):
        """Connect to device"""
        try:
            if self.serial_conn:
                self.serial_conn.close()
            self.serial_conn = serial.Serial(self.port, 115200, timeout=1)
            time.sleep(2)
            self.serial_conn.flushInput()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect device"""
        self.running = False
        if self.serial_conn:
            try:
                self.serial_conn.close()
                self.serial_conn = None
            except:
                pass
    
    def start_collection(self):
        """Start data collection"""
        if not self.connect():
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        return True
    
    def _collect_loop(self):
        """Collection loop"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    parts = line.split(',')
                    
                    if len(parts) >= 3:
                        try:
                            ir_val = int(parts[1])
                            red_val = int(parts[2])
                            self.ir_buffer.append(ir_val)
                            self.red_buffer.append(red_val)
                        except:
                            continue
                
                time.sleep(0.01)
            except:
                break
    
    def get_data(self, samples=None):
        """Get recent data"""
        n = samples if samples else len(self.ir_buffer)
        n = min(n, len(self.ir_buffer))
        
        if n == 0:
            return [], []
            
        return list(self.ir_buffer)[-n:], list(self.red_buffer)[-n:]

class RealTimePlot(QOpenGLWidget):
    """Real-time signal plot"""
    
    def __init__(self, title, color):
        super().__init__()
        self.title = title
        self.color = color
        self.data = []
        self.setMinimumSize(600, 200)
        
    def initializeGL(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        
    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
    def update_data(self, data):
        self.data = data[-800:] if len(data) > 800 else data
        self.update()
        
    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()
        
        # Grid
        gl.glColor3f(0.2, 0.2, 0.2)
        gl.glLineWidth(1.0)
        gl.glBegin(gl.GL_LINES)
        for i in range(11):
            x = i * 0.1
            gl.glVertex2f(x, 0)
            gl.glVertex2f(x, 1)
        for i in range(6):
            y = i * 0.2
            gl.glVertex2f(0, y)
            gl.glVertex2f(1, y)
        gl.glEnd()
        
        # Signal
        if len(self.data) > 1:
            data_arr = np.array(self.data, dtype=float)
            x = np.linspace(0, 1, len(data_arr))
            
            min_val, max_val = data_arr.min(), data_arr.max()
            if max_val > min_val:
                y = 0.1 + (data_arr - min_val) / (max_val - min_val) * 0.8
            else:
                y = np.full_like(data_arr, 0.5)
            
            gl.glColor3f(*self.color)
            gl.glLineWidth(2.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for i in range(len(x)):
                gl.glVertex2f(x[i], y[i])
            gl.glEnd()

class CycleAnalysisPlot(QOpenGLWidget):
    """Cycle analysis with fiducial points"""
    
    def __init__(self, title, color):
        super().__init__()
        self.title = title
        self.color = color
        self.cycle_data = None
        self.fiducials = None
        self.setMinimumSize(400, 300)
        
    def initializeGL(self):
        gl.glClearColor(0.02, 0.02, 0.05, 1.0)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        
    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, 1, 0, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
    def update_analysis(self, s, fp):
        if s and fp:
            self.cycle_data = s.ppg
            self.fiducials = fp
        self.update()
        
    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glLoadIdentity()
        
        # Grid
        gl.glColor3f(0.15, 0.15, 0.15)
        gl.glLineWidth(1.0)
        gl.glBegin(gl.GL_LINES)
        for i in range(21):
            x = i * 0.05
            gl.glVertex2f(x, 0)
            gl.glVertex2f(x, 1)
        for i in range(11):
            y = i * 0.1
            gl.glVertex2f(0, y)
            gl.glVertex2f(1, y)
        gl.glEnd()
        
        if self.cycle_data is None or self.fiducials is None:
            return
            
        # Show single cycle around first systolic peak
        if len(self.fiducials.sp) > 0:
            peak_idx = self.fiducials.sp[0]
            start_idx = max(0, peak_idx - 100)
            end_idx = min(len(self.cycle_data), peak_idx + 200)
            
            cycle_segment = self.cycle_data[start_idx:end_idx]
            x = np.linspace(0, 1, len(cycle_segment))
            
            # Normalize
            min_val, max_val = cycle_segment.min(), cycle_segment.max()
            if max_val > min_val:
                y = 0.1 + (cycle_segment - min_val) / (max_val - min_val) * 0.8
            else:
                y = np.full_like(cycle_segment, 0.5)
            
            # Draw signal
            gl.glColor3f(*self.color)
            gl.glLineWidth(3.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for i in range(len(x)):
                gl.glVertex2f(x[i], y[i])
            gl.glEnd()
            
            # Draw fiducial points
            points = [
                ('sp', (1.0, 0.0, 0.0), 'Systolic Peak'),
                ('on', (0.0, 1.0, 0.0), 'Onset'),
                ('dn', (0.0, 0.8, 1.0), 'Dicrotic Notch'),
                ('off', (0.8, 0.0, 1.0), 'Offset')
            ]
            
            gl.glPointSize(12.0)
            for point_type, color, label in points:
                if hasattr(self.fiducials, point_type):
                    point_list = getattr(self.fiducials, point_type)
                    if len(point_list) > 0:
                        # Find points in current segment
                        for p_idx in point_list:
                            if start_idx <= p_idx < end_idx:
                                rel_idx = p_idx - start_idx
                                if 0 <= rel_idx < len(x):
                                    gl.glColor3f(*color)
                                    gl.glBegin(gl.GL_POINTS)
                                    gl.glVertex2f(x[rel_idx], y[rel_idx])
                                    gl.glEnd()

class BiomarkerDisplay(QTextEdit):
    """Comprehensive biomarker display"""
    
    def __init__(self):
        super().__init__()
        self.setMaximumHeight(250)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #444;
                padding: 10px;
            }
        """)
        
    def update_biomarkers(self, biomarkers, ir_fp, red_fp):
        """Update biomarker display"""
        
        # Feature detection status
        ir_features = ""
        red_features = ""
        
        if ir_fp:
            ir_features = f"Onset:{len(ir_fp.on)}  Sys:{len(ir_fp.sp)}  Notch:{len(ir_fp.dn)}  Off:{len(ir_fp.off)}"
        
        if red_fp:
            red_features = f"Onset:{len(red_fp.on)}  Sys:{len(red_fp.sp)}  Notch:{len(red_fp.dn)}  Off:{len(red_fp.off)}"
        
        text = f"""COMPREHENSIVE PPG BIOMARKER ANALYSIS (pyPPG Library)
══════════════════════════════════════════════════════════════════════════════════════════════

VITAL SIGNS & QUALITY
Heart Rate: {biomarkers.get('heart_rate', 0):.1f} BPM                SpO2: {biomarkers.get('spo2', 0):.1f} %
IR Signal Quality: {biomarkers.get('signal_quality_ir', 0):.1f} %    RED Signal Quality: {biomarkers.get('signal_quality_red', 0):.1f} %

PULSE MORPHOLOGY
Systolic Amplitude: {biomarkers.get('systolic_amplitude', 0):.1f}     Pulse Width: {biomarkers.get('pulse_width', 0):.1f} ms
Pulse Area: {biomarkers.get('pulse_area', 0):.1f}                     Dicrotic Notch: {biomarkers.get('dicrotic_notch_amplitude', 0):.1f}

CARDIOVASCULAR INDICES  
Reflection Index: {biomarkers.get('reflection_index', 0):.3f}         Stiffness Index: {biomarkers.get('stiffness_index', 0):.3f}
Augmentation Index: {biomarkers.get('augmentation_index', 0):.3f}

HEART RATE VARIABILITY
Mean RR: {biomarkers.get('hrv_mean', 0):.1f} ms                       STD RR: {biomarkers.get('hrv_std', 0):.1f} ms
RMSSD: {biomarkers.get('hrv_rmssd', 0):.1f} ms

FIDUCIAL POINT DETECTION
IR Channel:  {ir_features}
RED Channel: {red_features}
══════════════════════════════════════════════════════════════════════════════════════════════"""
        
        self.setPlainText(text)

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.collector = SerialCollector()
        self.processor = PPGProcessor()
        self.biomarker_extractor = BiomarkerExtractor()
        
        self.ir_results = None
        self.red_results = None
        
        self.setWindowTitle("PPG Analysis System - pyPPG Library")
        self.setGeometry(100, 100, 1400, 900)
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Setup user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Controls
        controls = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.connect_btn.clicked.connect(self.connect_device)
        self.disconnect_btn.clicked.connect(self.disconnect_device)
        
        controls.addWidget(self.connect_btn)
        controls.addWidget(self.disconnect_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Main content
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Real-time plots
        plot_frame = QFrame()
        plot_layout = QHBoxLayout(plot_frame)
        
        # IR Plot
        ir_group = QFrame()
        ir_group.setFrameStyle(QFrame.Shape.Box)
        ir_layout = QVBoxLayout(ir_group)
        ir_title = QLabel("IR Signal")
        ir_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ir_title.setStyleSheet("font-weight: bold; color: #ff4444; padding: 5px;")
        self.ir_plot = RealTimePlot("IR Signal", (1.0, 0.2, 0.2))
        ir_layout.addWidget(ir_title)
        ir_layout.addWidget(self.ir_plot)
        
        # RED Plot
        red_group = QFrame()
        red_group.setFrameStyle(QFrame.Shape.Box)
        red_layout = QVBoxLayout(red_group)
        red_title = QLabel("RED Signal")
        red_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        red_title.setStyleSheet("font-weight: bold; color: #4488ff; padding: 5px;")
        self.red_plot = RealTimePlot("RED Signal", (0.2, 0.6, 1.0))
        red_layout.addWidget(red_title)
        red_layout.addWidget(self.red_plot)
        
        plot_layout.addWidget(ir_group)
        plot_layout.addWidget(red_group)
        
        # Analysis section
        analysis_frame = QFrame()
        analysis_layout = QHBoxLayout(analysis_frame)
        
        # Cycle analysis
        cycle_frame = QFrame()
        cycle_layout = QHBoxLayout(cycle_frame)
        
        # IR Cycle
        ir_cycle_group = QFrame()
        ir_cycle_group.setFrameStyle(QFrame.Shape.Box)
        ir_cycle_layout = QVBoxLayout(ir_cycle_group)
        ir_cycle_title = QLabel("IR Cycle Analysis")
        ir_cycle_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ir_cycle_title.setStyleSheet("font-weight: bold; color: #ff4444; padding: 5px;")
        self.ir_cycle = CycleAnalysisPlot("IR Cycle", (1.0, 0.2, 0.2))
        ir_cycle_layout.addWidget(ir_cycle_title)
        ir_cycle_layout.addWidget(self.ir_cycle)
        
        # RED Cycle
        red_cycle_group = QFrame()
        red_cycle_group.setFrameStyle(QFrame.Shape.Box)
        red_cycle_layout = QVBoxLayout(red_cycle_group)
        red_cycle_title = QLabel("RED Cycle Analysis")
        red_cycle_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        red_cycle_title.setStyleSheet("font-weight: bold; color: #4488ff; padding: 5px;")
        self.red_cycle = CycleAnalysisPlot("RED Cycle", (0.2, 0.6, 1.0))
        red_cycle_layout.addWidget(red_cycle_title)
        red_cycle_layout.addWidget(self.red_cycle)
        
        cycle_layout.addWidget(ir_cycle_group)
        cycle_layout.addWidget(red_cycle_group)
        
        # Biomarker display
        self.biomarker_display = BiomarkerDisplay()
        
        analysis_layout.addWidget(cycle_frame, 2)
        analysis_layout.addWidget(self.biomarker_display, 1)
        
        main_splitter.addWidget(plot_frame)
        main_splitter.addWidget(analysis_frame)
        main_splitter.setSizes([350, 550])
        
        layout.addWidget(main_splitter)
        
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(200)  # 5 FPS for analysis
        
    def connect_device(self):
        """Connect device"""
        if self.collector.start_collection():
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            
    def disconnect_device(self):
        """Disconnect device"""
        self.collector.disconnect()
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        
    def update_display(self):
        """Update all displays"""
        ir_data, red_data = self.collector.get_data()
        
        if len(ir_data) < 10:
            return
            
        # Update real-time plots
        self.ir_plot.update_data(ir_data)
        self.red_plot.update_data(red_data)
        
        # Analyze signals with pyPPG
        if len(ir_data) >= 500:
            self.ir_results = self.processor.analyze_signal(ir_data[-1000:])
            if self.ir_results and self.ir_results[0]:
                s, fp, bm, sqi = self.ir_results
                self.ir_cycle.update_analysis(s, fp)
                
        if len(red_data) >= 500:
            self.red_results = self.processor.analyze_signal(red_data[-1000:])
            if self.red_results and self.red_results[0]:
                s, fp, bm, sqi = self.red_results
                self.red_cycle.update_analysis(s, fp)
        
        # Update biomarkers
        self.biomarker_extractor.update_data(self.ir_results, self.red_results)
        biomarkers = self.biomarker_extractor.get_comprehensive_biomarkers()
        
        # Get fiducial points for display
        ir_fp = self.ir_results[1] if self.ir_results else None
        red_fp = self.red_results[1] if self.red_results else None
        
        self.biomarker_display.update_biomarkers(biomarkers, ir_fp, red_fp)
        
    def closeEvent(self, event):
        """Handle close"""
        self.collector.disconnect()
        event.accept()

def main():
    """Main entry point"""
    print("PPG Analysis System using pyPPG Library")
    print("Comprehensive biomarker extraction with validated algorithms")
    print("=" * 60)
    
    if not AVAILABLE_QT:
        print("PyQt6 not available! Install: pip install PyQt6 PyOpenGL")
        return
        
    if not AVAILABLE_PYPPG:
        print("pyPPG not available! Install: pip install pyPPG")
        return
        
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()