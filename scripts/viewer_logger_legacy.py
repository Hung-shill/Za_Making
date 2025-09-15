"""
Legacy LIDAR Viewer - Backwards Compatibility Wrapper
This file maintains compatibility with the original viewer_logger.py interface
while using the new modular system underneath.
"""

import logging
from pathlib import Path

# Import new modular components
from src.config.config import ConfigManager
from src.lidar_system.lidar_processor import LidarProcessor
from src.visualization.lidar_visualizer import LidarVisualizer, VisualizationMode
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import LidarSystem

# Configure logging (maintain original format)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lidar_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RPLidar_C1")

# Configuration for backwards compatibility
filename = "data/test_onRug.csv"

def main():
    """Legacy main function - maintains original behavior"""
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()

        # Force legacy settings for compatibility
        config.sensor.csv_file = filename
        config.visualizer.mode = VisualizationMode.FULL_GUI

        # Initialize system
        system = LidarSystem(config)

        # Run single scan (original behavior)
        logger.info(f"Processing {filename} using new modular system")
        success = system.run_single_scan(filename)

        if success:
            logger.info("Scan processing completed successfully")
        else:
            logger.error("Scan processing failed")

    except Exception as e:
        logger.error(f"Legacy viewer failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()