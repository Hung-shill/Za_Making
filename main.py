#!/usr/bin/env python3
"""
LIDAR System Main Application
Command-line interface for LIDAR data processing and visualization.
Supports multiple operation modes for development and Raspberry Pi deployment.
"""

import argparse
import logging
import time
import signal
import sys
import numpy as np
from pathlib import Path
from typing import Optional

from src.config.config import ConfigManager, AppConfig
from src.lidar_system.lidar_processor import LidarProcessor
from src.visualization.lidar_visualizer import LidarVisualizer, VisualizationMode
from src.sensors.sensor_interface import create_sensor_interface, BufferedSensorInterface


class LidarSystem:
    """Main LIDAR system controller"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.running = False

        # Initialize components
        self.processor = LidarProcessor(config.lidar_processor)
        self.visualizer = LidarVisualizer(config.visualizer)
        self.sensor = None
        self.buffered_sensor = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.config.system.log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("LidarSystem")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def run_single_scan(self, input_file: Optional[str] = None) -> bool:
        """Process and visualize a single scan"""
        try:
            # Use provided file or config default
            filename = input_file or self.config.sensor.csv_file

            self.logger.info(f"Processing single scan from {filename}")

            # Process the scan
            scan_data = self.processor.process_csv_file(filename)

            # Visualize
            output_path = self.visualizer.visualize(scan_data, f"LIDAR scan from {filename}")

            if output_path:
                self.logger.info(f"Scan visualization saved to {output_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to process single scan: {e}")
            return False

    def run_continuous(self) -> bool:
        """Run continuous scanning mode"""
        try:
            # Create sensor interface
            self.sensor = create_sensor_interface(
                self.config.sensor.interface_type,
                csv_file=self.config.sensor.csv_file,
                device_path=self.config.sensor.device_path,
                baudrate=self.config.sensor.baudrate,
                timeout=self.config.sensor.timeout_seconds,
                lidar_processor=self.processor
            )

            # Connect to sensor
            if not self.sensor.connect():
                self.logger.error("Failed to connect to sensor")
                return False

            # Setup buffered interface if enabled
            if self.config.performance.enable_data_buffering:
                self.buffered_sensor = BufferedSensorInterface(
                    self.sensor,
                    self.config.performance.buffer_size
                )

                if not self.buffered_sensor.start_collection(self.config.sensor.scan_frequency_hz):
                    self.logger.error("Failed to start buffered collection")
                    return False

                scan_source = self.buffered_sensor
            else:
                if not self.sensor.start_continuous_scan():
                    self.logger.error("Failed to start continuous scanning")
                    return False
                scan_source = self.sensor

            self.logger.info("Started continuous scanning mode")
            self.running = True
            scan_count = 0

            while self.running:
                try:
                    # Get scan data
                    if self.config.performance.enable_data_buffering:
                        scan_points = scan_source.get_latest_scan(timeout=1.0)
                    else:
                        scan_points = scan_source.get_scan()

                    if scan_points:
                        # Convert scan points to arrays for processing
                        angles_deg = [p.angle_deg for p in scan_points]
                        distances_mm = [p.distance_mm for p in scan_points]
                        qualities = [p.quality for p in scan_points]

                        # Process the data
                        scan_data = self.processor.process_raw_data(
                            np.array(angles_deg),
                            np.array(distances_mm),
                            np.array(qualities)
                        )

                        # Visualize
                        title = f"Live LIDAR Scan #{scan_count}"
                        output_path = self.visualizer.visualize(scan_data, title)

                        scan_count += 1

                        if scan_count % 10 == 0:
                            self.logger.info(f"Processed {scan_count} scans")

                        # Memory cleanup
                        if (self.config.performance.memory_cleanup_interval > 0 and
                            scan_count % self.config.performance.memory_cleanup_interval == 0):
                            self._cleanup_memory()

                    # Rate limiting
                    time.sleep(1.0 / self.config.sensor.scan_frequency_hz)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    self.logger.error(f"Error during continuous scanning: {e}")
                    if not self.config.sensor.auto_reconnect:
                        break

                    # Try to reconnect
                    self.logger.info("Attempting to reconnect...")
                    self._cleanup_sensor()
                    time.sleep(2)
                    continue

            self.logger.info(f"Continuous scanning stopped after {scan_count} scans")
            return True

        except Exception as e:
            self.logger.error(f"Failed to run continuous scanning: {e}")
            return False
        finally:
            self._cleanup_sensor()

    def run_benchmark(self, num_scans: int = 100) -> bool:
        """Run performance benchmark"""
        self.logger.info(f"Starting benchmark with {num_scans} scans")

        start_time = time.time()

        try:
            for i in range(num_scans):
                if not self.run_single_scan():
                    self.logger.error(f"Benchmark failed at scan {i}")
                    return False

                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    self.logger.info(f"Benchmark progress: {i+1}/{num_scans} ({rate:.1f} scans/sec)")

            total_time = time.time() - start_time
            avg_rate = num_scans / total_time

            self.logger.info(f"Benchmark completed: {num_scans} scans in {total_time:.2f}s")
            self.logger.info(f"Average rate: {avg_rate:.2f} scans/sec")

            return True

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return False

    def _cleanup_sensor(self):
        """Clean up sensor connections"""
        if self.buffered_sensor:
            self.buffered_sensor.stop_collection()
            self.buffered_sensor = None

        if self.sensor:
            self.sensor.disconnect()
            self.sensor = None

    def _cleanup_memory(self):
        """Perform memory cleanup"""
        import gc
        gc.collect()
        self.logger.debug("Memory cleanup performed")

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down LIDAR system")
        self.running = False
        self._cleanup_sensor()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LIDAR Data Processing and Visualization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s single --input scan.csv              # Process single CSV file
  %(prog)s continuous --sensor rplidar_sdk      # Continuous with RPLidar
  %(prog)s continuous --sensor csv --loop       # Continuous with CSV (loop)
  %(prog)s benchmark --count 50                 # Performance benchmark
  %(prog)s --config custom.json single          # Use custom config
        """
    )

    # Global options
    parser.add_argument("--config", "-c",
                       help="Configuration file path (default: lidar_config.json)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Override log level")
    parser.add_argument("--visualization", "-v",
                       choices=[mode.value for mode in VisualizationMode],
                       help="Override visualization mode")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single scan command
    single_parser = subparsers.add_parser("single", help="Process a single scan")
    single_parser.add_argument("--input", "-i", help="Input CSV file")

    # Continuous scan command
    continuous_parser = subparsers.add_parser("continuous", help="Continuous scanning")
    continuous_parser.add_argument("--sensor", choices=["csv", "rplidar_sdk"],
                                  help="Sensor interface type")
    continuous_parser.add_argument("--device", help="Device path (for hardware sensors)")
    continuous_parser.add_argument("--frequency", "-f", type=float,
                                  help="Scan frequency in Hz")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    benchmark_parser.add_argument("--count", "-n", type=int, default=100,
                                 help="Number of scans to process")

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--create-default", action="store_true",
                              help="Create default configuration file")

    args = parser.parse_args()

    # Load configuration
    config_file = args.config or "lidar_config.json"
    config_manager = ConfigManager(config_file)

    if args.command == "config":
        if args.show:
            config = config_manager.load_config()
            print("Current configuration:")
            import json
            print(json.dumps(config_manager._config_to_dict(config), indent=2))
            return 0
        elif args.create_default:
            config = config_manager._create_default_config()
            config_manager.save_config(config)
            print(f"Default configuration saved to {config_file}")
            return 0
        else:
            config_parser.print_help()
            return 1

    config = config_manager.load_config()

    # Apply command-line overrides
    if args.log_level:
        config.system.log_level = args.log_level

    if args.visualization:
        config.visualizer.mode = VisualizationMode(args.visualization)

    # Initialize system
    system = LidarSystem(config)

    try:
        if args.command == "single":
            success = system.run_single_scan(args.input)
        elif args.command == "continuous":
            if args.sensor:
                config.sensor.interface_type = args.sensor
            if args.device:
                config.sensor.device_path = args.device
            if args.frequency:
                config.sensor.scan_frequency_hz = args.frequency
            success = system.run_continuous()
        elif args.command == "benchmark":
            success = system.run_benchmark(args.count)
        else:
            # Default to single scan for backwards compatibility
            success = system.run_single_scan()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    finally:
        system.shutdown()


if __name__ == "__main__":
    sys.exit(main())