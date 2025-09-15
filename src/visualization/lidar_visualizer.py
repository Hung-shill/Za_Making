"""
LIDAR Visualization Module
Handles different visualization modes including headless operation for Raspberry Pi.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from src.lidar_system.lidar_processor import LidarScanData

# Try imports for different visualization backends
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Circle, FancyArrow, Wedge, Polygon
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VisualizationMode(Enum):
    """Available visualization modes"""
    FULL_GUI = "full_gui"          # Full matplotlib with GUI
    HEADLESS = "headless"          # Matplotlib without display
    OPENCV = "opencv"              # OpenCV-based rendering
    PIL = "pil"                    # PIL-based rendering
    TEXT_ONLY = "text"             # Terminal text output only


@dataclass
class VisualizationConfig:
    """Configuration for LIDAR visualization"""
    mode: VisualizationMode = VisualizationMode.FULL_GUI
    output_file: str = "scan.png"
    figure_size: Tuple[int, int] = (600, 600)
    dpi: int = 200
    xlim: Tuple[float, float] = (-3.0, 3.0)
    ylim: Tuple[float, float] = (-3.0, 3.0)
    range_rings_max: float = 6.0
    range_rings_step: float = 1.0
    lidar_icon_size: float = 0.12
    lidar_fov_deg: float = 270.0
    point_size: int = 4


class LidarVisualizer:
    """Handles LIDAR data visualization with multiple backends"""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger("LidarVisualizer")
        self._validate_backend()

    def visualize(self, scan_data: LidarScanData, title: Optional[str] = None) -> Optional[str]:
        """
        Visualize LIDAR scan data using configured mode
        Returns: Path to output file if saved, None otherwise
        """
        title = title or f"LIDAR Scan ({len(scan_data.x_coords)} points)"

        if self.config.mode == VisualizationMode.FULL_GUI:
            return self._visualize_matplotlib(scan_data, title, show=True)
        elif self.config.mode == VisualizationMode.HEADLESS:
            return self._visualize_matplotlib(scan_data, title, show=False)
        elif self.config.mode == VisualizationMode.OPENCV:
            return self._visualize_opencv(scan_data, title)
        elif self.config.mode == VisualizationMode.PIL:
            return self._visualize_pil(scan_data, title)
        elif self.config.mode == VisualizationMode.TEXT_ONLY:
            return self._visualize_text(scan_data, title)
        else:
            raise ValueError(f"Unsupported visualization mode: {self.config.mode}")

    def _validate_backend(self):
        """Validate that required backend is available"""
        if self.config.mode in [VisualizationMode.FULL_GUI, VisualizationMode.HEADLESS]:
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib not available for GUI/headless mode")
        elif self.config.mode == VisualizationMode.OPENCV:
            if not OPENCV_AVAILABLE:
                raise ImportError("OpenCV not available for OpenCV mode")
        elif self.config.mode == VisualizationMode.PIL:
            if not PIL_AVAILABLE:
                raise ImportError("PIL not available for PIL mode")

    def _visualize_matplotlib(self, scan_data: LidarScanData, title: str, show: bool = True) -> str:
        """Visualize using matplotlib (original implementation)"""
        # Set backend for headless operation
        if not show:
            plt.switch_backend('Agg')

        fig, ax = plt.subplots(figsize=(6, 6))

        # Create scatter plot
        sc = ax.scatter(
            scan_data.x_coords,
            scan_data.y_coords,
            s=self.config.point_size,
            c=scan_data.qualities,
            cmap="viridis"
        )

        # Draw LIDAR icon and range rings
        self._draw_lidar_icon_matplotlib(ax, label="Pookie")
        self._draw_range_rings_matplotlib(ax)

        # Set up plot
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(title)
        ax.set_aspect("equal", "box")
        ax.set_xlim(self.config.xlim)
        ax.set_ylim(self.config.ylim)

        plt.colorbar(sc, ax=ax, label="Quality")
        plt.grid(True, which="both", linestyle="dotted", alpha=0.5)
        ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.6, alpha=0.4)

        # Save and optionally show
        plt.savefig(self.config.output_file, dpi=self.config.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        self.logger.info(f"Saved visualization to {self.config.output_file}")
        return self.config.output_file

    def _visualize_opencv(self, scan_data: LidarScanData, title: str) -> str:
        """Lightweight visualization using OpenCV"""
        # Create blank image
        img = np.zeros((self.config.figure_size[1], self.config.figure_size[0], 3), dtype=np.uint8)

        # Calculate scaling
        x_range = self.config.xlim[1] - self.config.xlim[0]
        y_range = self.config.ylim[1] - self.config.ylim[0]
        scale_x = self.config.figure_size[0] / x_range
        scale_y = self.config.figure_size[1] / y_range

        center_x = self.config.figure_size[0] // 2
        center_y = self.config.figure_size[1] // 2

        # Draw range rings
        for r in np.arange(self.config.range_rings_step, self.config.range_rings_max + 0.001, self.config.range_rings_step):
            radius = int(r * scale_x)
            cv2.circle(img, (center_x, center_y), radius, (50, 50, 50), 1)

        # Draw coordinate axes
        cv2.line(img, (0, center_y), (self.config.figure_size[0], center_y), (100, 100, 100), 1)
        cv2.line(img, (center_x, 0), (center_x, self.config.figure_size[1]), (100, 100, 100), 1)

        # Draw LIDAR points
        for i in range(len(scan_data.x_coords)):
            x = int(center_x + scan_data.x_coords[i] * scale_x)
            y = int(center_y - scan_data.y_coords[i] * scale_y)  # Flip Y

            if 0 <= x < self.config.figure_size[0] and 0 <= y < self.config.figure_size[1]:
                # Color based on quality (simple mapping)
                quality_norm = min(scan_data.qualities[i] / 300.0, 1.0)
                color = (
                    int(255 * quality_norm),  # B
                    int(255 * (1 - quality_norm)),  # G
                    0  # R
                )
                cv2.circle(img, (x, y), 2, color, -1)

        # Draw LIDAR icon
        cv2.circle(img, (center_x, center_y), 8, (255, 255, 255), -1)
        cv2.circle(img, (center_x, center_y), 8, (0, 0, 255), 2)

        # Add title
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save image
        cv2.imwrite(self.config.output_file, img)
        self.logger.info(f"Saved OpenCV visualization to {self.config.output_file}")
        return self.config.output_file

    def _visualize_pil(self, scan_data: LidarScanData, title: str) -> str:
        """Ultra-lightweight visualization using PIL"""
        # Create image
        img = Image.new('RGB', self.config.figure_size, 'black')
        draw = ImageDraw.Draw(img)

        # Calculate scaling
        x_range = self.config.xlim[1] - self.config.xlim[0]
        y_range = self.config.ylim[1] - self.config.ylim[0]
        scale_x = self.config.figure_size[0] / x_range
        scale_y = self.config.figure_size[1] / y_range

        center_x = self.config.figure_size[0] // 2
        center_y = self.config.figure_size[1] // 2

        # Draw range rings
        for r in np.arange(self.config.range_rings_step, self.config.range_rings_max + 0.001, self.config.range_rings_step):
            radius = int(r * scale_x)
            draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius],
                        outline='gray', width=1)

        # Draw LIDAR points
        for i in range(len(scan_data.x_coords)):
            x = int(center_x + scan_data.x_coords[i] * scale_x)
            y = int(center_y - scan_data.y_coords[i] * scale_y)  # Flip Y

            if 0 <= x < self.config.figure_size[0] and 0 <= y < self.config.figure_size[1]:
                quality_norm = min(scan_data.qualities[i] / 300.0, 1.0)
                color = (int(255 * quality_norm), int(255 * (1 - quality_norm)), 0)
                draw.ellipse([x-2, y-2, x+2, y+2], fill=color)

        # Draw LIDAR icon
        draw.ellipse([center_x-8, center_y-8, center_x+8, center_y+8], fill='white', outline='red', width=2)

        # Save image
        img.save(self.config.output_file)
        self.logger.info(f"Saved PIL visualization to {self.config.output_file}")
        return self.config.output_file

    def _visualize_text(self, scan_data: LidarScanData, title: str) -> None:
        """Text-only visualization for terminal output"""
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Points: {len(scan_data.x_coords)}")
        print(f"Distance range: {scan_data.distances_m.min():.2f}m - {scan_data.distances_m.max():.2f}m")
        print(f"Quality range: {scan_data.qualities.min():.0f} - {scan_data.qualities.max():.0f}")
        print(f"X range: {scan_data.x_coords.min():.2f}m - {scan_data.x_coords.max():.2f}m")
        print(f"Y range: {scan_data.y_coords.min():.2f}m - {scan_data.y_coords.max():.2f}m")

        # Simple ASCII visualization (optional)
        if len(scan_data.x_coords) > 0:
            print("\nSimple ASCII representation:")
            grid_size = 20
            grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

            for i in range(len(scan_data.x_coords)):
                x_idx = int((scan_data.x_coords[i] - self.config.xlim[0]) /
                           (self.config.xlim[1] - self.config.xlim[0]) * (grid_size - 1))
                y_idx = int((scan_data.y_coords[i] - self.config.ylim[0]) /
                           (self.config.ylim[1] - self.config.ylim[0]) * (grid_size - 1))

                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    grid[grid_size - 1 - y_idx][x_idx] = '*'

            # Mark center
            grid[grid_size//2][grid_size//2] = 'L'

            for row in grid:
                print(''.join(row))

        return None

    def _draw_lidar_icon_matplotlib(self, ax, x0=0.0, y0=0.0, yaw_deg=0.0, label="Pookie"):
        """Draw stylized LIDAR icon using matplotlib"""
        yaw = np.deg2rad(yaw_deg)
        size = self.config.lidar_icon_size
        color = "royalblue"

        # Field of view wedge
        wedge_r = size * 6.5
        ax.add_patch(Wedge(
            (x0, y0), wedge_r,
            yaw_deg - self.config.lidar_fov_deg / 2.0,
            yaw_deg + self.config.lidar_fov_deg / 2.0,
            facecolor=color, alpha=0.06, edgecolor="none", zorder=2
        ))

        # Base
        base_L = size * 0.55
        base_W = size * 0.32
        base_local = np.array([
            [-base_L * 0.5, -base_W * 0.5],
            [-base_L * 0.5, +base_W * 0.5],
            [+base_L * 0.35, +base_W * 0.5],
            [+base_L * 0.55, 0.0],
            [+base_L * 0.35, -base_W * 0.5],
        ])
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        base = (base_local @ R.T) + np.array([x0, y0])
        ax.add_patch(Polygon(base, closed=True, facecolor=color, edgecolor="black",
                           linewidth=0.6, alpha=0.85, zorder=5))

        # Head
        head_offset = size * 0.22
        head_x = x0 + head_offset * np.cos(yaw)
        head_y = y0 + head_offset * np.sin(yaw)
        ax.add_patch(Circle((head_x, head_y), radius=size * 0.14, facecolor="white",
                          edgecolor=color, linewidth=1.0, zorder=6))

        # Heading arrow
        ax.add_patch(FancyArrow(
            x0, y0, size * 0.9 * np.cos(yaw), size * 0.9 * np.sin(yaw),
            width=size * 0.08, head_width=size * 0.25, head_length=size * 0.25,
            length_includes_head=True, color=color, alpha=0.9, zorder=7
        ))

        # Label
        if label:
            ax.text(x0 + size * 0.7 * np.cos(yaw), y0 + size * 0.7 * np.sin(yaw),
                   str(label), fontsize=8, color=color, va="center", ha="left", zorder=8)

    def _draw_range_rings_matplotlib(self, ax):
        """Draw range rings using matplotlib"""
        for r in np.arange(self.config.range_rings_step, self.config.range_rings_max + 0.000001,
                          self.config.range_rings_step):
            ax.add_patch(Circle((0, 0), r, fill=False, color="black", alpha=0.2,
                              linestyle="dotted", lw=0.7))