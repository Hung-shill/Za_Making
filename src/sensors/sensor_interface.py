"""
Sensor Interface Module
Handles different LIDAR sensor interfaces including RPLidar SDK and CSV files.
"""

import time
import threading
import queue
import logging
from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from src.lidar_system.lidar_processor import LidarScanData, LidarProcessor

# Try to import RPLidar SDK
try:
    from rplidar import RPLidar, RPLidarException
    RPLIDAR_SDK_AVAILABLE = True
except ImportError:
    RPLIDAR_SDK_AVAILABLE = False

# Try to import serial for custom protocols
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


@dataclass
class ScanPoint:
    """Single LIDAR measurement point"""
    angle_deg: float
    distance_mm: float
    quality: int
    timestamp: Optional[float] = None


class SensorInterface(ABC):
    """Abstract base class for LIDAR sensor interfaces"""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to sensor"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from sensor"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if sensor is connected"""
        pass

    @abstractmethod
    def get_scan(self) -> Optional[List[ScanPoint]]:
        """Get a single scan from sensor"""
        pass

    @abstractmethod
    def start_continuous_scan(self) -> bool:
        """Start continuous scanning mode"""
        pass

    @abstractmethod
    def stop_continuous_scan(self) -> None:
        """Stop continuous scanning mode"""
        pass


class CSVSensorInterface(SensorInterface):
    """Interface for reading CSV files (for testing/development)"""

    def __init__(self, csv_file: str, lidar_processor: LidarProcessor, loop: bool = True):
        self.csv_file = csv_file
        self.processor = lidar_processor
        self.loop = loop
        self.logger = logging.getLogger("CSVSensorInterface")
        self._connected = False
        self._scan_data = None
        self._current_index = 0

    def connect(self) -> bool:
        """Load CSV file"""
        try:
            self._scan_data = self.processor.process_csv_file(self.csv_file)
            self._connected = True
            self.logger.info(f"Loaded CSV file: {self.csv_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load CSV file: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect (no-op for CSV)"""
        self._connected = False
        self.logger.info("Disconnected from CSV file")

    def is_connected(self) -> bool:
        return self._connected

    def get_scan(self) -> Optional[List[ScanPoint]]:
        """Get scan data from loaded CSV"""
        if not self._connected or self._scan_data is None:
            return None

        # Convert processed data back to scan points
        points = []
        for i in range(len(self._scan_data.x_coords)):
            angle_deg = np.rad2deg(self._scan_data.angles_rad[i])
            distance_mm = self._scan_data.distances_m[i] * 1000
            quality = int(self._scan_data.qualities[i])

            points.append(ScanPoint(
                angle_deg=angle_deg,
                distance_mm=distance_mm,
                quality=quality,
                timestamp=time.time()
            ))

        return points

    def start_continuous_scan(self) -> bool:
        """Start continuous mode (returns same data repeatedly for CSV)"""
        return self.is_connected()

    def stop_continuous_scan(self) -> None:
        """Stop continuous mode (no-op for CSV)"""
        pass


class RPLidarInterface(SensorInterface):
    """Interface for RPLidar sensors using the RPLidar SDK"""

    def __init__(self, device_path: str, baudrate: int = 115200, timeout: float = 1.0):
        if not RPLIDAR_SDK_AVAILABLE:
            raise ImportError("RPLidar SDK not available. Install with: pip install rplidar")

        self.device_path = device_path
        self.baudrate = baudrate
        self.timeout = timeout
        self.logger = logging.getLogger("RPLidarInterface")
        self._lidar = None
        self._connected = False
        self._scanning = False

    def connect(self) -> bool:
        """Connect to RPLidar sensor"""
        try:
            self._lidar = RPLidar(self.device_path, baudrate=self.baudrate, timeout=self.timeout)

            # Test connection
            info = self._lidar.get_info()
            health = self._lidar.get_health()

            self.logger.info(f"Connected to RPLidar: {info}")
            self.logger.info(f"Health status: {health}")

            if health[0] != "Good":
                self.logger.warning(f"RPLidar health status: {health}")

            self._connected = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to RPLidar: {e}")
            self._cleanup_lidar()
            return False

    def disconnect(self) -> None:
        """Disconnect from RPLidar"""
        self.stop_continuous_scan()
        self._cleanup_lidar()
        self._connected = False
        self.logger.info("Disconnected from RPLidar")

    def is_connected(self) -> bool:
        return self._connected and self._lidar is not None

    def get_scan(self) -> Optional[List[ScanPoint]]:
        """Get a single scan from RPLidar"""
        if not self.is_connected():
            return None

        try:
            # Get one complete scan
            scan_data = []
            for _, angle, distance, quality in self._lidar.iter_scans():
                # Convert to our format
                points = []
                for i in range(len(angle)):
                    points.append(ScanPoint(
                        angle_deg=angle[i],
                        distance_mm=distance[i],
                        quality=quality[i],
                        timestamp=time.time()
                    ))
                return points

            return None

        except RPLidarException as e:
            self.logger.error(f"RPLidar scan error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during scan: {e}")
            return None

    def start_continuous_scan(self) -> bool:
        """Start continuous scanning mode"""
        if not self.is_connected():
            return False

        try:
            self._lidar.start_motor()
            time.sleep(1)  # Let motor spin up
            self._scanning = True
            self.logger.info("Started continuous scanning")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start continuous scanning: {e}")
            return False

    def stop_continuous_scan(self) -> None:
        """Stop continuous scanning mode"""
        if self._lidar and self._scanning:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._scanning = False
                self.logger.info("Stopped continuous scanning")
            except Exception as e:
                self.logger.error(f"Error stopping scan: {e}")

    def _cleanup_lidar(self):
        """Clean up RPLidar connection"""
        if self._lidar:
            try:
                if self._scanning:
                    self._lidar.stop()
                    self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
            finally:
                self._lidar = None
                self._scanning = False


class BufferedSensorInterface:
    """Buffered wrapper for sensor interfaces with background data collection"""

    def __init__(self, sensor_interface: SensorInterface, buffer_size: int = 10):
        self.sensor = sensor_interface
        self.buffer_size = buffer_size
        self.logger = logging.getLogger("BufferedSensorInterface")

        self._scan_queue = queue.Queue(maxsize=buffer_size)
        self._collection_thread = None
        self._stop_event = threading.Event()
        self._collecting = False

    def connect(self) -> bool:
        """Connect to underlying sensor"""
        return self.sensor.connect()

    def disconnect(self) -> None:
        """Disconnect and stop collection"""
        self.stop_collection()
        self.sensor.disconnect()

    def is_connected(self) -> bool:
        return self.sensor.is_connected()

    def start_collection(self, scan_frequency_hz: float = 10.0) -> bool:
        """Start background scan collection"""
        if not self.is_connected():
            return False

        if self._collecting:
            self.logger.warning("Collection already running")
            return True

        if not self.sensor.start_continuous_scan():
            return False

        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_worker,
            args=(scan_frequency_hz,),
            daemon=True
        )
        self._collection_thread.start()
        self._collecting = True

        self.logger.info(f"Started scan collection at {scan_frequency_hz} Hz")
        return True

    def stop_collection(self) -> None:
        """Stop background collection"""
        if not self._collecting:
            return

        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=2.0)

        self.sensor.stop_continuous_scan()
        self._collecting = False
        self.logger.info("Stopped scan collection")

    def get_latest_scan(self, timeout: float = 1.0) -> Optional[List[ScanPoint]]:
        """Get the most recent scan from buffer"""
        try:
            # Get the most recent scan, discarding older ones
            latest_scan = None
            while True:
                try:
                    latest_scan = self._scan_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_scan is None and timeout > 0:
                # Wait for new scan if none available
                latest_scan = self._scan_queue.get(timeout=timeout)

            return latest_scan

        except queue.Empty:
            self.logger.warning("No scan data available")
            return None

    def _collection_worker(self, scan_frequency_hz: float) -> None:
        """Background worker for collecting scans"""
        scan_interval = 1.0 / scan_frequency_hz
        last_scan_time = 0

        while not self._stop_event.is_set():
            current_time = time.time()

            # Rate limiting
            if current_time - last_scan_time < scan_interval:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue

            # Get scan
            scan_data = self.sensor.get_scan()
            if scan_data:
                try:
                    # Add to queue (drop oldest if full)
                    if self._scan_queue.full():
                        try:
                            self._scan_queue.get_nowait()  # Drop oldest
                        except queue.Empty:
                            pass

                    self._scan_queue.put_nowait(scan_data)
                    last_scan_time = current_time

                except queue.Full:
                    self.logger.warning("Scan buffer full, dropping data")

            time.sleep(0.001)  # Small yield


def create_sensor_interface(interface_type: str, **kwargs) -> SensorInterface:
    """Factory function to create sensor interfaces"""
    if interface_type.lower() == "csv":
        csv_file = kwargs.get("csv_file", "Test_onRug.csv")
        processor = kwargs.get("lidar_processor")
        if not processor:
            from src.lidar_system.lidar_processor import LidarProcessor
            processor = LidarProcessor()
        return CSVSensorInterface(csv_file, processor)

    elif interface_type.lower() == "rplidar_sdk":
        device_path = kwargs.get("device_path", "/dev/ttyUSB0")
        baudrate = kwargs.get("baudrate", 115200)
        timeout = kwargs.get("timeout", 1.0)
        return RPLidarInterface(device_path, baudrate, timeout)

    else:
        raise ValueError(f"Unsupported sensor interface type: {interface_type}")