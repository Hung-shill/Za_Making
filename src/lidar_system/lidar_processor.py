"""
LIDAR Data Processing Module
Handles data loading, filtering, and coordinate transformations for autonomous vehicle systems.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class LidarConfig:
    """Configuration for LIDAR data processing"""
    quality_threshold_pandas: int = 120
    quality_threshold_numpy: int = 100
    min_distance_mm_pandas: int = 80
    max_distance_mm_pandas: int = 8000
    min_distance_mm_numpy: int = 50
    max_distance_mm_numpy: int = 8000


@dataclass
class LidarScanData:
    """Container for processed LIDAR scan data"""
    x_coords: np.ndarray
    y_coords: np.ndarray
    distances_m: np.ndarray
    angles_rad: np.ndarray
    qualities: np.ndarray
    timestamp: Optional[float] = None
    frame_id: Optional[str] = None


class LidarProcessor:
    """Processes LIDAR data from various sources with quality filtering"""

    def __init__(self, config: Optional[LidarConfig] = None):
        self.config = config or LidarConfig()
        self.logger = logging.getLogger("LidarProcessor")

    def process_csv_file(self, filename: str) -> LidarScanData:
        """Load and process LIDAR data from CSV file"""
        self.logger.info(f"Loading lidar data from {filename}")

        try:
            # Try pandas first, fall back to numpy
            if self._pandas_available():
                return self._process_with_pandas(filename)
            else:
                return self._process_with_numpy(filename)

        except FileNotFoundError:
            self.logger.error(f"Lidar data file not found: {filename}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to process lidar data: {e}", exc_info=True)
            raise

    def process_raw_data(self, angles_deg: np.ndarray, distances_mm: np.ndarray,
                        qualities: np.ndarray) -> LidarScanData:
        """Process raw LIDAR data arrays directly"""
        self.logger.debug(f"Processing {len(angles_deg)} raw lidar points")

        # Apply quality filtering
        if len(qualities) > 0:
            mask = (qualities > self.config.quality_threshold_numpy) & \
                   (distances_mm > self.config.min_distance_mm_numpy) & \
                   (distances_mm < self.config.max_distance_mm_numpy)

            angles_deg = angles_deg[mask]
            distances_mm = distances_mm[mask]
            qualities = qualities[mask]

            self._log_filtering_results(len(mask), np.sum(mask))

        # Convert to standard units
        distances_m = distances_mm.astype(float) / 1000.0
        angles_rad = np.deg2rad(angles_deg.astype(float))

        # Polar to Cartesian transformation
        x_coords = distances_m * np.cos(angles_rad)
        y_coords = distances_m * np.sin(angles_rad)

        return LidarScanData(
            x_coords=x_coords,
            y_coords=y_coords,
            distances_m=distances_m,
            angles_rad=angles_rad,
            qualities=qualities.astype(float)
        )

    def _pandas_available(self) -> bool:
        """Check if pandas is available"""
        try:
            import pandas as pd
            return pd is not None
        except ImportError:
            return False

    def _process_with_pandas(self, filename: str) -> LidarScanData:
        """Process CSV file using pandas"""
        df = pd.read_csv(
            filename,
            comment="#",
            sep=r"[,\s]+",
            names=["angle", "distance_mm", "quality"],
            engine="python",
            header=None,
        )

        # Convert to numeric and clean
        for col in ["angle", "distance_mm", "quality"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["angle", "distance_mm"])

        # Quality filtering
        initial_rows = len(df)
        df = df[
            (df["quality"] > self.config.quality_threshold_pandas) &
            (df["distance_mm"].between(
                self.config.min_distance_mm_pandas,
                self.config.max_distance_mm_pandas
            ))
        ]
        filtered_rows = len(df)

        self._validate_filtered_data(filtered_rows, initial_rows)
        self._log_filtering_results(initial_rows, filtered_rows)

        # Extract data
        qualities = df["quality"].to_numpy(dtype=float)
        distances_m = df["distance_mm"].to_numpy(dtype=float) / 1000.0
        angles_rad = np.deg2rad(df["angle"].to_numpy(dtype=float))

        # Polar to Cartesian
        x_coords = distances_m * np.cos(angles_rad)
        y_coords = distances_m * np.sin(angles_rad)

        self.logger.info(f"Processed {filtered_rows} valid lidar points using pandas")

        return LidarScanData(
            x_coords=x_coords,
            y_coords=y_coords,
            distances_m=distances_m,
            angles_rad=angles_rad,
            qualities=qualities
        )

    def _process_with_numpy(self, filename: str) -> LidarScanData:
        """Process CSV file using numpy only"""
        data = np.genfromtxt(filename, comments="#", dtype=float)

        if data.ndim == 1:
            data = np.atleast_2d(data)
        if data.shape[1] < 3:
            self.logger.error(f"Invalid CSV format: expected 3 columns, got {data.shape[1]}")
            raise ValueError("Expected at least 3 columns: angle, distance_mm, quality")

        angle = data[:, 0]
        distance_mm = data[:, 1]
        quality = data[:, 2]

        # Quality filtering
        initial_points = len(angle)
        mask = (quality > self.config.quality_threshold_numpy) & \
               (distance_mm > self.config.min_distance_mm_numpy) & \
               (distance_mm < self.config.max_distance_mm_numpy)

        angle = angle[mask]
        distance_mm = distance_mm[mask]
        quality = quality[mask]

        filtered_points = len(angle)
        self._validate_filtered_data(filtered_points, initial_points)
        self._log_filtering_results(initial_points, filtered_points)

        # Convert units and coordinates
        distances_m = distance_mm.astype(float) / 1000.0
        angles_rad = np.deg2rad(angle.astype(float))

        x_coords = distances_m * np.cos(angles_rad)
        y_coords = distances_m * np.sin(angles_rad)

        self.logger.info(f"Processed {filtered_points} valid lidar points using NumPy")

        return LidarScanData(
            x_coords=x_coords,
            y_coords=y_coords,
            distances_m=distances_m,
            angles_rad=angles_rad,
            qualities=quality.astype(float)
        )

    def _validate_filtered_data(self, filtered_count: int, initial_count: int):
        """Validate that filtering didn't remove all data"""
        if filtered_count == 0:
            self.logger.error("No valid lidar data after filtering - sensor may be malfunctioning")
            raise ValueError("No valid lidar data available")

        if filtered_count < initial_count * 0.5:
            self.logger.warning(f"Low quality data: {filtered_count}/{initial_count} points passed filter")

    def _log_filtering_results(self, initial: int, filtered: int):
        """Log the results of data filtering"""
        self.logger.debug(f"Data filtering: {filtered}/{initial} points retained ({100*filtered/initial:.1f}%)")