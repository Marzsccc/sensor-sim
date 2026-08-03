"""
Sensor Simulator: Generate realistic multi-sensor measurements for SLAM/state-estimation.

Supports:
  - Configurable IMU (consumer/tactical/navigation grade)
  - GNSS with multipath, signal loss, DOP scaling
  - Arbitrary 6-DoF trajectories
"""

from .trajectory import Trajectory
from .imu import IMUSensor
from .gnss import GNSSSensor

__version__ = "0.1.0"
