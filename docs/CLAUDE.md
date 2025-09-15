# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

This project uses Python 3.13 with a virtual environment located in `.venv/`.

### Virtual Environment Setup

```bash
source .venv/bin/activate  # Activate virtual environment
deactivate                 # Deactivate when done
```

### Dependencies

The project uses these key packages (installed in virtual environment):

- `matplotlib 3.10.6` - Plotting and visualization
- `numpy 2.3.3` - Numerical computations and coordinate transformations
- `pandas 2.3.2` - CSV data loading (optional, falls back to numpy)
- `psutil` - System monitoring (for Pi deployment)
- `rplidar` - RPLidar SDK (optional, for hardware integration)

### Running the Application

#### New Modular System (Recommended)

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Single scan processing
python main.py single --input data/test_onRug.csv

# Continuous scanning with CSV (development)
python main.py continuous --sensor csv

# Continuous scanning with RPLidar (Pi deployment)
python main.py continuous --sensor rplidar_sdk --device /dev/ttyUSB0

# Performance benchmark
python main.py benchmark --count 50

# Different visualization modes
python main.py single --visualization headless
python main.py single --visualization opencv
```

#### Legacy Compatibility

```bash
# Original interface (backwards compatible)
python viewer_logger.py
python viewer_logger_legacy.py
```

## Project Architecture

This is a modular LIDAR data visualization system designed for autonomous vehicle development. The system supports both development (full GUI) and Raspberry Pi deployment (headless/lightweight modes).

### Core Modules

**`src/lidar_system/lidar_processor.py`** - Data Processing Engine

- Handles CSV file loading and real-time sensor data processing
- Dual backend support (pandas/numpy) with automatic fallback
- Quality filtering and coordinate transformations
- Configurable thresholds for different sensor types

**`src/visualization/lidar_visualizer.py`** - Multi-Backend Visualization

- Full GUI mode (matplotlib with interactive display)
- Headless mode (matplotlib without display - Pi compatible)
- OpenCV mode (lightweight, fast rendering)
- PIL mode (ultra-lightweight for minimal systems)
- Text-only mode (terminal output with ASCII visualization)

**`src/sensors/sensor_interface.py`** - Sensor Abstraction Layer

- CSV file interface (development/testing)
- RPLidar SDK interface (hardware integration)
- Buffered sensor wrapper for continuous operation
- Auto-reconnection and error handling

**`src/config/config.py`** - Configuration Management

- Platform detection (automatic Pi vs development settings)
- JSON-based configuration with intelligent defaults
- Performance optimization settings
- Visualization backend selection

**`src/lidar_system/pi_monitor.py`** - System Monitoring & Optimization

- Real-time performance monitoring (CPU, memory, temperature)
- Pi-specific optimizations and health checks
- Performance metrics tracking for LIDAR processing
- Alert system for resource issues

**`main.py`** - Command-Line Interface

- Single scan processing mode
- Continuous scanning mode
- Performance benchmarking
- Configuration management commands

### Project Structure

```
Za_making/
├── data/                           # LIDAR CSV data files
│   ├── test_onRug.csv             # Default test data
│   ├── onTheFloor.csv             # Floor scanning data
│   ├── onWorkBench.csv            # Workbench scanning data
│   ├── Test_higherup.csv          # Elevated position data
│   └── test_ontable.csv           # Table scanning data
├── src/                           # Source code modules
│   ├── config/
│   │   └── config.py              # Configuration management
│   ├── lidar_system/
│   │   ├── lidar_processor.py     # Data processing engine
│   │   └── pi_monitor.py          # Pi monitoring and optimization
│   ├── sensors/
│   │   └── sensor_interface.py    # Sensor abstraction layer
│   └── visualization/
│       └── lidar_visualizer.py    # Multi-backend visualization
├── scripts/                       # Utility scripts
│   └── viewer_logger_legacy.py    # Legacy compatibility wrapper
├── deployment/                    # Deployment files
│   ├── requirements.txt           # Python dependencies
│   └── lidar-system.service       # Systemd service file
├── docs/                          # Documentation
│   └── CLAUDE.md                  # This documentation
├── tests/                         # Test files (future)
├── main.py                        # Main application entry point
└── viewer_logger.py               # Original legacy viewer
```

### Input Data Format

CSV files with columns: `angle, distance_mm, quality`

- Lines starting with `#` are treated as comments
- All data files are stored in the `data/` folder

### Output

- Generates `scan.png` visualization at 200 DPI
- Creates `lidar_system.log` for system diagnostics
- Interactive matplotlib display window

## Configuration System

### Configuration Files

- `lidar_config.json` - Main configuration (auto-created with platform-specific defaults)
- `lidar-system.service` - Systemd service for Pi auto-startup

### Configuration Management

```bash
# Show current configuration
python main.py config --show

# Create default configuration file
python main.py config --create-default

# Override settings via command line
python main.py single --visualization headless --log-level DEBUG
```

### Platform-Specific Defaults

**Development Platform:**

- Full GUI visualization with matplotlib
- Higher resource limits and processing quality
- Debug logging enabled
- CSV file interface for testing

**Raspberry Pi:**

- Headless/lightweight visualization modes
- Memory and CPU optimizations
- RPLidar SDK interface for real sensors
- System monitoring and health checks enabled
- Web interface for remote monitoring

### Available Test Data Files (in `data/` folder)

- `data/test_onRug.csv` (default)
- `data/onTheFloor.csv`
- `data/onWorkBench.csv`
- `data/Test_higherup.csv`
- `data/test_ontable.csv`

## Raspberry Pi Deployment

### Installation

```bash
# Copy project to Pi
scp -r Za_making/ pi@raspberrypi:~/lidar-system/

# SSH to Pi and setup
ssh pi@raspberrypi
cd ~/lidar-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Create this from your dev environment

# Install systemd service
sudo cp lidar-system.service /etc/systemd/system/
sudo systemctl enable lidar-system
sudo systemctl start lidar-system
```

### Monitoring

```bash
# Check service status
sudo systemctl status lidar-system

# View logs
sudo journalctl -u lidar-system -f

# System monitoring
python main.py continuous --sensor rplidar_sdk  # With built-in monitoring
```
