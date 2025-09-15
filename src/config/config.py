"""
Configuration Management for LIDAR System
Handles platform-specific settings for development and Raspberry Pi deployment.
"""

import json
import os
import platform
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from src.lidar_system.lidar_processor import LidarConfig
from src.visualization.lidar_visualizer import VisualizationConfig, VisualizationMode


@dataclass
class SystemConfig:
    """System-level configuration"""
    platform: str = platform.system().lower()
    is_raspberry_pi: bool = False
    log_level: str = "INFO"
    log_file: str = "lidar_system.log"
    max_log_size_mb: int = 10
    log_backup_count: int = 5
    enable_performance_monitoring: bool = False
    memory_limit_mb: Optional[int] = None


@dataclass
class SensorConfig:
    """Sensor interface configuration"""
    interface_type: str = "csv"  # csv, rplidar_sdk, serial
    device_path: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout_seconds: float = 1.0
    csv_file: str = "Test_onRug.csv"
    auto_reconnect: bool = True
    scan_frequency_hz: float = 10.0


@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    max_points_per_scan: int = 2000
    processing_threads: int = 1
    enable_data_buffering: bool = True
    buffer_size: int = 10
    enable_gpu_acceleration: bool = False
    memory_cleanup_interval: int = 100  # scans


@dataclass
class WebConfig:
    """Web interface configuration for remote monitoring"""
    enable_web_interface: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    enable_streaming: bool = False
    stream_fps: int = 5


@dataclass
class AppConfig:
    """Complete application configuration"""
    system: SystemConfig
    sensor: SensorConfig
    lidar_processor: LidarConfig
    visualizer: VisualizationConfig
    performance: PerformanceConfig
    web: WebConfig


class ConfigManager:
    """Manages application configuration with platform detection and overrides"""

    def __init__(self, config_file: str = "lidar_config.json"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger("ConfigManager")
        self._detect_platform()

    def load_config(self) -> AppConfig:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            return self._load_from_file()
        else:
            config = self._create_default_config()
            self.save_config(config)
            return config

    def save_config(self, config: AppConfig) -> None:
        """Save configuration to file"""
        try:
            config_dict = self._config_to_dict(config)
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")

    def _load_from_file(self) -> AppConfig:
        """Load configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
            return self._dict_to_config(config_dict)
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {self.config_file}: {e}")
            self.logger.info("Using default configuration")
            return self._create_default_config()

    def _create_default_config(self) -> AppConfig:
        """Create default configuration based on platform"""
        is_pi = self._is_raspberry_pi()

        # System configuration
        system = SystemConfig(
            is_raspberry_pi=is_pi,
            log_level="DEBUG" if not is_pi else "INFO",
            enable_performance_monitoring=is_pi,
            memory_limit_mb=500 if is_pi else None
        )

        # Sensor configuration
        sensor = SensorConfig(
            interface_type="rplidar_sdk" if is_pi else "csv",
            device_path="/dev/ttyUSB0" if is_pi else "",
            csv_file="data/test_onRug.csv"
        )

        # LIDAR processor configuration
        lidar_processor = LidarConfig()

        # Visualization configuration
        if is_pi:
            # Pi: Use headless or lightweight modes
            visualizer = VisualizationConfig(
                mode=VisualizationMode.HEADLESS,
                dpi=100,  # Lower DPI for Pi
                figure_size=(400, 400)  # Smaller images
            )
        else:
            # Development: Full GUI
            visualizer = VisualizationConfig(
                mode=VisualizationMode.FULL_GUI
            )

        # Performance configuration
        performance = PerformanceConfig(
            max_points_per_scan=1000 if is_pi else 2000,
            processing_threads=1 if is_pi else 2,
            enable_data_buffering=True,
            buffer_size=5 if is_pi else 10
        )

        # Web interface configuration
        web = WebConfig(
            enable_web_interface=is_pi,  # Enable on Pi for remote monitoring
            enable_streaming=is_pi
        )

        return AppConfig(
            system=system,
            sensor=sensor,
            lidar_processor=lidar_processor,
            visualizer=visualizer,
            performance=performance,
            web=web
        )

    def _detect_platform(self) -> None:
        """Detect platform and log information"""
        system = platform.system()
        machine = platform.machine()
        is_pi = self._is_raspberry_pi()

        self.logger.info(f"Platform detected: {system} ({machine})")
        if is_pi:
            self.logger.info("Raspberry Pi detected - using Pi-optimized settings")
        else:
            self.logger.info("Development platform detected - using full features")

    def _is_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            # Check for Pi-specific files
            if Path("/proc/device-tree/model").exists():
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    return "raspberry pi" in model

            # Check for Pi-specific hardware
            if Path("/proc/cpuinfo").exists():
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read().lower()
                    return "bcm" in cpuinfo and "arm" in cpuinfo

            # Check architecture
            machine = platform.machine().lower()
            return machine.startswith("arm") and platform.system().lower() == "linux"

        except Exception:
            return False

    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization"""
        return {
            "system": asdict(config.system),
            "sensor": asdict(config.sensor),
            "lidar_processor": asdict(config.lidar_processor),
            "visualizer": {
                **asdict(config.visualizer),
                "mode": config.visualizer.mode.value  # Convert enum to string
            },
            "performance": asdict(config.performance),
            "web": asdict(config.web)
        }

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to configuration objects"""
        # Handle visualization mode enum
        viz_dict = config_dict.get("visualizer", {})
        if "mode" in viz_dict:
            viz_dict["mode"] = VisualizationMode(viz_dict["mode"])

        return AppConfig(
            system=SystemConfig(**config_dict.get("system", {})),
            sensor=SensorConfig(**config_dict.get("sensor", {})),
            lidar_processor=LidarConfig(**config_dict.get("lidar_processor", {})),
            visualizer=VisualizationConfig(**viz_dict),
            performance=PerformanceConfig(**config_dict.get("performance", {})),
            web=WebConfig(**config_dict.get("web", {}))
        )

    def update_config(self, config: AppConfig, **kwargs) -> AppConfig:
        """Update configuration with new values"""
        config_dict = self._config_to_dict(config)

        # Update nested configuration
        for key, value in kwargs.items():
            if "." in key:
                section, param = key.split(".", 1)
                if section in config_dict:
                    config_dict[section][param] = value
            else:
                # Top-level parameter
                if hasattr(config, key):
                    setattr(config, key, value)

        return self._dict_to_config(config_dict)

    def get_visualization_backend_recommendation(self, config: AppConfig) -> VisualizationMode:
        """Recommend visualization backend based on platform and available libraries"""
        if config.system.is_raspberry_pi:
            # Try lightweight modes first on Pi
            try:
                import cv2
                return VisualizationMode.OPENCV
            except ImportError:
                pass

            try:
                from PIL import Image
                return VisualizationMode.PIL
            except ImportError:
                pass

            # Fall back to headless matplotlib
            try:
                import matplotlib
                return VisualizationMode.HEADLESS
            except ImportError:
                return VisualizationMode.TEXT_ONLY

        else:
            # Development platform - prefer full GUI
            try:
                import matplotlib
                return VisualizationMode.FULL_GUI
            except ImportError:
                return VisualizationMode.TEXT_ONLY