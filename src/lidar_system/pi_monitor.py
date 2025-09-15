"""
Raspberry Pi Monitoring and Optimization Module
Provides system monitoring, performance optimization, and Pi-specific utilities.
"""

import time
import threading
import logging
import psutil
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    temperature_celsius: Optional[float] = None
    disk_usage_percent: float = 0.0
    timestamp: float = 0.0


@dataclass
class LidarMetrics:
    """LIDAR system performance metrics"""
    scans_per_second: float = 0.0
    points_per_scan: int = 0
    processing_time_ms: float = 0.0
    visualization_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    dropped_scans: int = 0
    timestamp: float = 0.0


class PiMonitor:
    """System monitoring for Raspberry Pi"""

    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger("PiMonitor")
        self.is_pi = self._detect_raspberry_pi()
        self._monitoring = False
        self._monitor_thread = None
        self._metrics_history = []
        self._max_history = 100

        # Callbacks for alerts
        self.alert_callbacks: Dict[str, Callable] = {}

    def start_monitoring(self) -> bool:
        """Start background monitoring"""
        if self._monitoring:
            self.logger.warning("Monitoring already running")
            return True

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self._monitor_thread.start()

        self.logger.info(f"Started system monitoring (interval: {self.monitoring_interval}s)")
        return True

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        self.logger.info("Stopped system monitoring")

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        temperature = None
        if self.is_pi:
            temperature = self._get_pi_temperature()

        disk = psutil.disk_usage('/')

        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            temperature_celsius=temperature,
            disk_usage_percent=disk.percent,
            timestamp=time.time()
        )

    def get_metrics_history(self, last_n: Optional[int] = None) -> list:
        """Get historical metrics"""
        if last_n:
            return self._metrics_history[-last_n:]
        return self._metrics_history.copy()

    def register_alert_callback(self, alert_type: str, callback: Callable[[SystemMetrics], None]) -> None:
        """Register callback for system alerts"""
        self.alert_callbacks[alert_type] = callback
        self.logger.info(f"Registered alert callback for {alert_type}")

    def check_system_health(self, metrics: Optional[SystemMetrics] = None) -> Dict[str, Any]:
        """Check system health and return status"""
        if metrics is None:
            metrics = self.get_current_metrics()

        health = {
            "status": "healthy",
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        # CPU check
        if metrics.cpu_percent > 85:
            health["errors"].append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            health["status"] = "critical"
        elif metrics.cpu_percent > 70:
            health["warnings"].append(f"Elevated CPU usage: {metrics.cpu_percent:.1f}%")
            if health["status"] == "healthy":
                health["status"] = "warning"

        # Memory check
        if metrics.memory_percent > 90:
            health["errors"].append(f"High memory usage: {metrics.memory_percent:.1f}%")
            health["status"] = "critical"
            health["recommendations"].append("Consider reducing buffer sizes or scan frequency")
        elif metrics.memory_percent > 75:
            health["warnings"].append(f"Elevated memory usage: {metrics.memory_percent:.1f}%")
            if health["status"] == "healthy":
                health["status"] = "warning"

        # Temperature check (Pi only)
        if metrics.temperature_celsius:
            if metrics.temperature_celsius > 80:
                health["errors"].append(f"High temperature: {metrics.temperature_celsius:.1f}°C")
                health["status"] = "critical"
                health["recommendations"].append("Check cooling and reduce processing load")
            elif metrics.temperature_celsius > 70:
                health["warnings"].append(f"Elevated temperature: {metrics.temperature_celsius:.1f}°C")
                if health["status"] == "healthy":
                    health["status"] = "warning"

        # Disk space check
        if metrics.disk_usage_percent > 95:
            health["errors"].append(f"Disk space critical: {metrics.disk_usage_percent:.1f}%")
            health["status"] = "critical"
            health["recommendations"].append("Clean up log files and old scan data")
        elif metrics.disk_usage_percent > 85:
            health["warnings"].append(f"Disk space low: {metrics.disk_usage_percent:.1f}%")
            if health["status"] == "healthy":
                health["status"] = "warning"

        return health

    def optimize_for_pi(self) -> Dict[str, Any]:
        """Apply Pi-specific optimizations"""
        optimizations = {
            "applied": [],
            "failed": [],
            "recommendations": []
        }

        if not self.is_pi:
            optimizations["recommendations"].append("Not running on Raspberry Pi - optimizations skipped")
            return optimizations

        # GPU memory split optimization
        try:
            gpu_mem = self._get_gpu_memory_split()
            if gpu_mem and gpu_mem > 16:
                optimizations["recommendations"].append(
                    f"Consider reducing GPU memory split from {gpu_mem}MB to 16MB for headless operation"
                )
        except Exception as e:
            self.logger.debug(f"Could not check GPU memory: {e}")

        # Swap file check
        try:
            swap_info = psutil.swap_memory()
            if swap_info.total == 0:
                optimizations["recommendations"].append(
                    "Consider enabling swap file for memory-intensive operations"
                )
        except Exception as e:
            self.logger.debug(f"Could not check swap: {e}")

        # CPU governor optimization
        try:
            governor = self._get_cpu_governor()
            if governor and governor != "performance":
                optimizations["recommendations"].append(
                    f"Consider setting CPU governor to 'performance' (current: {governor})"
                )
        except Exception as e:
            self.logger.debug(f"Could not check CPU governor: {e}")

        return optimizations

    def _monitor_worker(self) -> None:
        """Background monitoring worker"""
        while self._monitoring:
            try:
                metrics = self.get_current_metrics()

                # Store metrics
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history.pop(0)

                # Check for alerts
                health = self.check_system_health(metrics)
                if health["status"] != "healthy":
                    for alert_type, callback in self.alert_callbacks.items():
                        try:
                            callback(metrics)
                        except Exception as e:
                            self.logger.error(f"Alert callback {alert_type} failed: {e}")

                # Log critical issues
                if health["status"] == "critical":
                    self.logger.error(f"System critical: {', '.join(health['errors'])}")
                elif health["status"] == "warning":
                    self.logger.warning(f"System warnings: {', '.join(health['warnings'])}")

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            if Path("/proc/device-tree/model").exists():
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    return "raspberry pi" in model
            return False
        except Exception:
            return False

    def _get_pi_temperature(self) -> Optional[float]:
        """Get Raspberry Pi CPU temperature"""
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                with open(temp_path, "r") as f:
                    temp_millidegrees = int(f.read().strip())
                    return temp_millidegrees / 1000.0
        except Exception as e:
            self.logger.debug(f"Could not read temperature: {e}")
        return None

    def _get_gpu_memory_split(self) -> Optional[int]:
        """Get GPU memory split setting"""
        try:
            import subprocess
            result = subprocess.run(
                ["vcgencmd", "get_mem", "gpu"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Output format: "gpu=64M"
                output = result.stdout.strip()
                if "=" in output:
                    mem_str = output.split("=")[1].rstrip("M")
                    return int(mem_str)
        except Exception as e:
            self.logger.debug(f"Could not get GPU memory: {e}")
        return None

    def _get_cpu_governor(self) -> Optional[str]:
        """Get CPU frequency governor"""
        try:
            governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            if governor_path.exists():
                with open(governor_path, "r") as f:
                    return f.read().strip()
        except Exception as e:
            self.logger.debug(f"Could not read CPU governor: {e}")
        return None


class LidarPerformanceTracker:
    """Tracks LIDAR system performance metrics"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.logger = logging.getLogger("LidarPerformanceTracker")
        self._metrics_history = []
        self._scan_times = []
        self._last_scan_time = 0
        self._total_scans = 0
        self._dropped_scans = 0

    def record_scan(self, points_count: int, processing_time_ms: float,
                   visualization_time_ms: float = 0.0) -> None:
        """Record metrics for a completed scan"""
        current_time = time.time()

        # Calculate scan rate
        if self._last_scan_time > 0:
            scan_interval = current_time - self._last_scan_time
            self._scan_times.append(scan_interval)
            if len(self._scan_times) > 100:  # Keep last 100 scan intervals
                self._scan_times.pop(0)

        self._last_scan_time = current_time
        self._total_scans += 1

        # Calculate average scan rate
        if len(self._scan_times) > 0:
            avg_interval = sum(self._scan_times) / len(self._scan_times)
            scans_per_second = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            scans_per_second = 0.0

        # Get memory usage
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)

        # Create metrics record
        metrics = LidarMetrics(
            scans_per_second=scans_per_second,
            points_per_scan=points_count,
            processing_time_ms=processing_time_ms,
            visualization_time_ms=visualization_time_ms,
            memory_usage_mb=memory_usage_mb,
            dropped_scans=self._dropped_scans,
            timestamp=current_time
        )

        # Store metrics
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self.max_history:
            self._metrics_history.pop(0)

        # Log performance warnings
        if processing_time_ms > 100:  # More than 100ms processing time
            self.logger.warning(f"Slow processing: {processing_time_ms:.1f}ms for {points_count} points")

        if scans_per_second < 5 and len(self._scan_times) > 10:  # Less than 5 Hz
            self.logger.warning(f"Low scan rate: {scans_per_second:.1f} Hz")

    def record_dropped_scan(self) -> None:
        """Record a dropped/failed scan"""
        self._dropped_scans += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self._metrics_history:
            return {"status": "no_data"}

        recent_metrics = self._metrics_history[-10:]  # Last 10 scans

        avg_rate = sum(m.scans_per_second for m in recent_metrics) / len(recent_metrics)
        avg_processing_time = sum(m.processing_time_ms for m in recent_metrics) / len(recent_metrics)
        avg_points = sum(m.points_per_scan for m in recent_metrics) / len(recent_metrics)
        current_memory = recent_metrics[-1].memory_usage_mb if recent_metrics else 0

        return {
            "total_scans": self._total_scans,
            "dropped_scans": self._dropped_scans,
            "drop_rate_percent": (self._dropped_scans / max(self._total_scans, 1)) * 100,
            "avg_scan_rate_hz": avg_rate,
            "avg_processing_time_ms": avg_processing_time,
            "avg_points_per_scan": int(avg_points),
            "current_memory_mb": current_memory,
            "status": "healthy" if avg_rate > 5 and avg_processing_time < 200 else "degraded"
        }

    def get_metrics_history(self, last_n: Optional[int] = None) -> list:
        """Get historical performance metrics"""
        if last_n:
            return self._metrics_history[-last_n:]
        return self._metrics_history.copy()