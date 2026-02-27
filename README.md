# NirogScan BLE Application - Setup Instructions

## System Requirements
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.10.x (via Anaconda)
- **RAM**: 8GB minimum (16GB recommended for compilation)
- **Disk**: 5GB free space

## Step 1: Create Conda Environment
```bash
# Create new environment
conda create -n nirogscan_py310 python=3.10 -y

# Activate environment
conda activate nirogscan_py310
```

## Step 2: Install Dependencies
```bash
# Install from requirements.txt (includes biosppy/peakutils for ECG processing)
pip install -r requirements.txt

# Verify critical packages
python -c "import PyQt5; print('PyQt5:', PyQt5.Qt.PYQT_VERSION_STR)"
python -c "import bleak; print('Bleak:', bleak.__version__)"
python -c "import winrt.windows.foundation.collections; print('WinRT: OK')"
```

## Step 3: Test Application
```bash
# Test BLE application
python ble.py

# Test Report application
python report_UI.py
```

## Step 4: Build with Nuitka

### Option A: Using Build Script (Recommended)
```bash
# Run the build script
build_nuitka.bat
```

### Option B: Manual Build
```bash
# Build ble.py
python -m nuitka --onefile --windows-console-mode=force --enable-plugin=pyqt5 --follow-imports --include-package=winrt --include-package=winrt.windows --include-package=winrt.windows.foundation --include-package=winrt.windows.foundation.collections --include-package=winrt.windows.devices --include-package=winrt.windows.devices.bluetooth --include-package=winrt.windows.devices.bluetooth.advertisement --include-package=winrt.windows.storage.streams --include-package=bleak --include-package=bleak.backends.winrt ble.py

# Build report_UI.py
python -m nuitka --onefile --windows-console-mode=force --enable-plugin=pyqt5 --follow-imports --include-package=winrt --include-package=winrt.windows --include-package=winrt.windows.foundation --include-package=winrt.windows.foundation.collections --include-package=winrt.windows.devices --include-package=winrt.windows.devices.bluetooth --include-package=winrt.windows.devices.bluetooth.advertisement --include-package=winrt.windows.storage.streams --include-package=bleak --include-package=bleak.backends.winrt report_UI.py
```

## Step 5: Alternative - PyInstaller (Easier)
```bash
# Build ble.py
pyinstaller --onefile --console --name=ble_app --collect-all=bleak --collect-all=PyQt5 --collect-all=pyqtgraph --collect-all=scipy --collect-all=numpy --collect-all=pandas ble.py

# Build report_UI.py
pyinstaller --onefile --console --name=report_app --collect-all=bleak --collect-all=PyQt5 --collect-all=matplotlib --collect-all=pyqtgraph --collect-all=scipy --collect-all=numpy --collect-all=pandas report_UI.py

# Executables will be in: dist/ble_app.exe and dist/report_app.exe
```

## Troubleshooting

### Issue: "ModuleNotFoundError: winrt.windows.foundation.collections"
**Solution**: 
```bash
pip install bleak-winrt --force-reinstall
python -c "import winrt.windows.foundation.collections; print('OK')"
```

### Issue: "ModuleNotFoundError: peakutils" or biosppy import failure
**Solution**: 
```bash
pip install biosppy peakutils
# or rerun
pip install -r requirements.txt
```

### Issue: "IMPORT_HARD_UNITTEST" error
**Solution**: Don't use `--nofollow-import-to=unittest` flag

### Issue: Executable doesn't start
**Solution**: Build with `--windows-console-mode=force` to see errors

### Issue: Bluetooth not working in executable
**Solution**: Ensure all `winrt.*` packages are included in build command

## Notes
- Build time: 20-40 minutes per application
- Final executable size: ~160-180 MB
- Keep console enabled for debugging
- Test executable on a clean Windows system before deployment