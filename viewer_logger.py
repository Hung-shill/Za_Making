import pandas as pd  # optional; code falls back to NumPy if unavailable
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.patches import Circle, FancyArrow, Wedge, Polygon, Rectangle
import math
import logging

# Configure logging for autonomous vehicle system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("lidar_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger("RPLidar_C1")


"""
x = r * cos(theta), y = r * sin(theta). Formulas for converting polar to Cartesian coordinates.
have to be in radians for numpy trig functions

Also need to activate the virtual environment in terminal:
source venv/bin/activate

to deactivate:
deactivate

"""
# Change this to your CSV filename
filename = "data/test_onRug.csv"


def symbolic_lidar(ax, x0=0.0, y0=0.0, yaw_deg=0.0, label="Pookie"):
    """Draw pookie lidar at (x0,y0)"""
    yaw = np.deg2rad(yaw_deg)
    L = 0.06
    tri_local = np.array([[+L, 0.0], [-0.06 * L, +0.5 * L], [-0.06 * L, -0.5 * L]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    tri = (tri_local @ R.T) + np.array([x0, y0])
    ax.add_patch(Polygon(tri, closed=True, fill=True, color="blue", alpha=0.9))

    ax.add_patch(
        FancyArrow(
            x0,
            y0,
            0.18 * np.cos(yaw),
            0.18 * np.sin(yaw),
            width=0.01,
            length_includes_head=True,
        )
    )


def draw_lidar_icon(
    ax,
    x0=0.0,
    y0=0.0,
    yaw_deg=0.0,
    size=0.12,
    color="royalblue",
    fov_deg=270,
    label=None,
):
    """Draw a stylized lidar icon with base, head, FOV and heading.

    - size: overall icon scale in meters
    - fov_deg: field of view wedge angle
    """
    yaw = np.deg2rad(yaw_deg)

    # Field of view wedge (subtle, behind icon)
    wedge_r = size * 6.5
    ax.add_patch(
        Wedge(
            (x0, y0),
            wedge_r,
            yaw_deg - fov_deg / 2.0,
            yaw_deg + fov_deg / 2.0,
            facecolor=color,
            alpha=0.06,
            edgecolor="none",
            zorder=2,
        )
    )

    # Simple base (small rounded rectangle made from a polygon)
    base_L = size * 0.55
    base_W = size * 0.32
    base_local = np.array(
        [
            [-base_L * 0.5, -base_W * 0.5],
            [-base_L * 0.5, +base_W * 0.5],
            [+base_L * 0.35, +base_W * 0.5],
            [+base_L * 0.55, 0.0],  # slight nose
            [+base_L * 0.35, -base_W * 0.5],
        ]
    )
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    base = (base_local @ R.T) + np.array([x0, y0])
    ax.add_patch(
        Polygon(
            base,
            closed=True,
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            zorder=5,
        )
    )

    # Lidar head (small circle slightly forward)
    head_offset = size * 0.22
    head_x = x0 + head_offset * np.cos(yaw)
    head_y = y0 + head_offset * np.sin(yaw)
    ax.add_patch(
        Circle(
            (head_x, head_y),
            radius=size * 0.14,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=6,
        )
    )

    # Heading arrow
    ax.add_patch(
        FancyArrow(
            x0,
            y0,
            size * 0.9 * np.cos(yaw),
            size * 0.9 * np.sin(yaw),
            width=size * 0.08,
            head_width=size * 0.25,
            head_length=size * 0.25,
            length_includes_head=True,
            color=color,
            alpha=0.9,
            zorder=7,
        )
    )

    # Label
    if label is not None:
        ax.text(
            x0 + size * 0.7 * np.cos(yaw),
            y0 + size * 0.7 * np.sin(yaw),
            str(label),
            fontsize=8,
            color=color,
            va="center",
            ha="left",
            zorder=8,
        )


def draw_range_rings(ax, rmax=6.0, step=1.0):
    """Draw range rings every step meters up to rmax"""
    for r in np.arange(step, rmax + 0.000001, step):
        ax.add_patch(
            Circle(
                (0, 0),
                r,
                fill=False,
                color="black",
                alpha=0.2,
                linestyle="dotted",
                lw=0.7,
            )
        )
        # ax.text(
        #     r * 0.7,
        #     0.1,
        #     f"{r:g}m",
        #     color="black",
        #     alpha=0.3,
        #     va="bottom",
        #     ha="left",
        #     fontsize=8,
        # )


# def draw_footprint(ax, length=0.3, width=0.2, x0= 0.0, y0 =-0.09):

"""Load file, skipping header lines that start with #.
Supports pandas if installed; otherwise falls back to NumPy.
"""
logger.info(f"Loading lidar data from {filename}")

try:
    if pd is not None:
        df = pd.read_csv(
            filename,
            comment="#",
            sep=r"[,\s]+",
            names=["angle", "distance_mm", "quality"],
            engine="python",
            header=None,
        )

        for col in ["angle", "distance_mm", "quality"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["angle", "distance_mm"])

        # Check for data quality issues
        initial_rows = len(df)
        df = df[(df["quality"] > 120) & (df["distance_mm"].between(80, 8000))]
        filtered_rows = len(df)

        if filtered_rows == 0:
            logger.error(
                "No valid lidar data after quality filtering - sensor may be malfunctioning"
            )
            raise ValueError("No valid lidar data available")

        if filtered_rows < initial_rows * 0.5:
            logger.warning(
                f"Low quality data: {filtered_rows}/{initial_rows} points passed filter"
            )

        logger.info(f"Processed {filtered_rows} valid lidar points")
        qualities = df["quality"].to_numpy(dtype=float)
        distances_m = (df["distance_mm"].to_numpy(dtype=float)) / 1000.0
        angles_rad = np.deg2rad(df["angle"].to_numpy(dtype=float))
    else:
        data = np.genfromtxt(
            fname=filename,
            comments="#",
            dtype=float,
        )
        if data.ndim == 1:
            data = np.atleast_2d(data)
        if data.shape[1] < 3:
            logger.error(f"Invalid CSV format: expected 3 columns, got {data.shape[1]}")
            raise ValueError("Expected at least 3 columns: angle, distance_mm, quality")

        angle = data[:, 0]
        distance_mm = data[:, 1]
        quality = data[:, 2]

        initial_points = len(angle)
        mask = (quality > 100.0) & (distance_mm > 50.0)
        angle = angle[mask]
        distance_mm = distance_mm[mask]
        quality = quality[mask]

        filtered_points = len(angle)
        if filtered_points == 0:
            logger.error(
                "No valid lidar data after NumPy filtering - sensor issue detected"
            )
            raise ValueError("No valid lidar data available")

        if filtered_points < initial_points * 0.5:
            logger.warning(
                f"Low quality data: {filtered_points}/{initial_points} points passed filter"
            )

        logger.info(f"Processed {filtered_points} valid lidar points using NumPy")
        qualities = quality.astype(float)
        distances_m = distance_mm.astype(float) / 1000.0
        angles_rad = np.deg2rad(angle.astype(float))

except FileNotFoundError:
    logger.error(f"Lidar data file not found: {filename}")
    raise
except Exception as e:
    logger.error(f"Failed to process lidar data: {e}", exc_info=True)
    raise

# Polar → Cartesian
xs = distances_m * np.cos(angles_rad)
ys = distances_m * np.sin(angles_rad)

# Plot
fig, ax = plot.subplots(figsize=(6, 6))
sc = ax.scatter(xs, ys, s=4, c=qualities, cmap="viridis")

# Draw the lidar icon at the origin
draw_lidar_icon(ax, x0=0.0, y0=0.0, yaw_deg=0.0, label="Pookie")
draw_range_rings(ax, rmax=6.0, step=1.0)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title(f"LIDAR scan from {filename}")
ax.set_aspect("equal", "box")  # equal scaling
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
plot.colorbar(sc, ax=ax, label="Quality")
plot.grid(True, which="both", linestyle="dotted", alpha=0.5)
ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
ax.axvline(0, color="gray", lw=0.6, alpha=0.4)
plot.savefig("scan.png", dpi=200, bbox_inches="tight")
plot.show()
