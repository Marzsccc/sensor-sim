"""
Sensor Simulator: Generate realistic multi-sensor measurements for SLAM/state-estimation.

Supports:
  - Configurable IMU (consumer/tactical/navigation grade)
  - GNSS with multipath, signal loss, DOP scaling
  - Wheel odometry (Ackermann or differential)
  - Arbitrary 6-DoF trajectories
"""

from .trajectory import Trajectory
from .imu import IMUSensor, IMUSpec, SensorGrade
from .gnss import GNSSSensor, GNSSSpec, GNSSGrade
from .wheel import WheelOdometry, WheelSpec, WheelGrade

__version__ = "0.2.0"
