"""
Obstacle Detection and Safety Zone Module
Provides obstacle analysis and safety zone management for autonomous vehicle navigation.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from src.lidar_system.lidar_processor import LidarScanData


class ThreatLevel(Enum):
    """Threat levels for obstacle detection"""
    IMMEDIATE_DANGER = "immediate_danger"  # Red zone - immediate stop/turn
    CAUTION = "caution"                    # Yellow zone - slow down
    SAFE = "safe"                          # Green zone - normal operation
    CLEAR = "clear"                        # No obstacles detected


@dataclass
class SafetyZone:
    """Configuration for safety zones around the vehicle"""
    immediate_danger_radius: float = 0.5   # meters
    caution_radius: float = 1.5            # meters
    safe_radius: float = 3.0               # meters

    # Colors for visualization
    immediate_color: str = "red"
    caution_color: str = "orange"
    safe_color: str = "green"


@dataclass
class ObstacleInfo:
    """Information about a detected obstacle"""
    distance: float
    angle_deg: float
    angle_rad: float
    x_coord: float
    y_coord: float
    threat_level: ThreatLevel
    quality: float


@dataclass
class NavigationAnalysis:
    """Complete navigation analysis results"""
    closest_obstacle: Optional[ObstacleInfo]
    obstacles_by_zone: Dict[ThreatLevel, List[ObstacleInfo]]
    safe_corridors: List[Tuple[float, float]]  # (start_angle, end_angle) in degrees
    overall_threat: ThreatLevel
    recommended_action: str


class ObstacleDetector:
    """Detects obstacles and analyzes safety zones for autonomous navigation"""

    def __init__(self, safety_zones: Optional[SafetyZone] = None):
        self.safety_zones = safety_zones or SafetyZone()
        self.logger = logging.getLogger("ObstacleDetector")

    def analyze_scan(self, scan_data: LidarScanData) -> NavigationAnalysis:
        """Perform complete obstacle analysis of a LIDAR scan"""

        # Find all obstacles with threat levels
        obstacles = self._detect_obstacles(scan_data)

        # Group obstacles by threat level
        obstacles_by_zone = self._group_by_threat_level(obstacles)

        # Find closest obstacle
        closest_obstacle = self._find_closest_obstacle(obstacles)

        # Find safe navigation corridors
        safe_corridors = self._find_safe_corridors(scan_data)

        # Determine overall threat level
        overall_threat = self._determine_overall_threat(obstacles_by_zone)

        # Generate navigation recommendation
        recommended_action = self._generate_recommendation(overall_threat, closest_obstacle)

        return NavigationAnalysis(
            closest_obstacle=closest_obstacle,
            obstacles_by_zone=obstacles_by_zone,
            safe_corridors=safe_corridors,
            overall_threat=overall_threat,
            recommended_action=recommended_action
        )

    def _detect_obstacles(self, scan_data: LidarScanData) -> List[ObstacleInfo]:
        """Detect obstacles and classify by threat level"""
        obstacles = []

        for i, (x, y, distance, angle_rad, quality) in enumerate(zip(
            scan_data.x_coords, scan_data.y_coords, scan_data.distances_m,
            scan_data.angles_rad, scan_data.qualities
        )):
            angle_deg = np.degrees(angle_rad)

            # Determine threat level based on distance
            if distance <= self.safety_zones.immediate_danger_radius:
                threat_level = ThreatLevel.IMMEDIATE_DANGER
            elif distance <= self.safety_zones.caution_radius:
                threat_level = ThreatLevel.CAUTION
            elif distance <= self.safety_zones.safe_radius:
                threat_level = ThreatLevel.SAFE
            else:
                threat_level = ThreatLevel.CLEAR

            obstacle = ObstacleInfo(
                distance=distance,
                angle_deg=angle_deg,
                angle_rad=angle_rad,
                x_coord=x,
                y_coord=y,
                threat_level=threat_level,
                quality=quality
            )
            obstacles.append(obstacle)

        return obstacles

    def _group_by_threat_level(self, obstacles: List[ObstacleInfo]) -> Dict[ThreatLevel, List[ObstacleInfo]]:
        """Group obstacles by their threat level"""
        grouped = {level: [] for level in ThreatLevel}

        for obstacle in obstacles:
            grouped[obstacle.threat_level].append(obstacle)

        return grouped

    def _find_closest_obstacle(self, obstacles: List[ObstacleInfo]) -> Optional[ObstacleInfo]:
        """Find the closest obstacle to the vehicle"""
        if not obstacles:
            return None

        return min(obstacles, key=lambda obs: obs.distance)

    def _find_safe_corridors(self, scan_data: LidarScanData, min_corridor_width: float = 30.0) -> List[Tuple[float, float]]:
        """Find safe navigation corridors (areas with no immediate dangers)"""
        corridors = []

        # Convert angles to degrees and sort by angle
        angles_deg = np.degrees(scan_data.angles_rad)
        distances = scan_data.distances_m

        # Find continuous segments with safe distances
        safe_mask = distances > self.safety_zones.immediate_danger_radius

        # Find transitions from unsafe to safe and back
        safe_start = None

        for i, (angle, is_safe) in enumerate(zip(angles_deg, safe_mask)):
            if is_safe and safe_start is None:
                # Start of safe corridor
                safe_start = angle
            elif not is_safe and safe_start is not None:
                # End of safe corridor
                corridor_width = angle - safe_start
                if corridor_width >= min_corridor_width:
                    corridors.append((safe_start, angle))
                safe_start = None

        # Handle case where scan ends in safe zone
        if safe_start is not None:
            corridor_width = 360 - safe_start  # Assuming 360-degree scan
            if corridor_width >= min_corridor_width:
                corridors.append((safe_start, 360))

        return corridors

    def _determine_overall_threat(self, obstacles_by_zone: Dict[ThreatLevel, List[ObstacleInfo]]) -> ThreatLevel:
        """Determine the overall threat level"""
        if obstacles_by_zone[ThreatLevel.IMMEDIATE_DANGER]:
            return ThreatLevel.IMMEDIATE_DANGER
        elif obstacles_by_zone[ThreatLevel.CAUTION]:
            return ThreatLevel.CAUTION
        elif obstacles_by_zone[ThreatLevel.SAFE]:
            return ThreatLevel.SAFE
        else:
            return ThreatLevel.CLEAR

    def _generate_recommendation(self, overall_threat: ThreatLevel, closest_obstacle: Optional[ObstacleInfo]) -> str:
        """Generate navigation recommendation based on analysis"""
        if overall_threat == ThreatLevel.IMMEDIATE_DANGER:
            if closest_obstacle:
                return f"STOP! Immediate danger at {closest_obstacle.distance:.2f}m, {closest_obstacle.angle_deg:.0f}°"
            else:
                return "STOP! Immediate danger detected"
        elif overall_threat == ThreatLevel.CAUTION:
            if closest_obstacle:
                return f"SLOW DOWN: Obstacle at {closest_obstacle.distance:.2f}m, {closest_obstacle.angle_deg:.0f}°"
            else:
                return "SLOW DOWN: Obstacles in caution zone"
        elif overall_threat == ThreatLevel.SAFE:
            return "PROCEED WITH CAUTION: Obstacles detected in safe zone"
        else:
            return "ALL CLEAR: No obstacles detected"

    def get_safety_zone_info(self) -> Dict[str, Any]:
        """Get safety zone configuration for visualization"""
        return {
            "immediate_danger": {
                "radius": self.safety_zones.immediate_danger_radius,
                "color": self.safety_zones.immediate_color,
                "alpha": 0.3
            },
            "caution": {
                "radius": self.safety_zones.caution_radius,
                "color": self.safety_zones.caution_color,
                "alpha": 0.2
            },
            "safe": {
                "radius": self.safety_zones.safe_radius,
                "color": self.safety_zones.safe_color,
                "alpha": 0.1
            }
        }