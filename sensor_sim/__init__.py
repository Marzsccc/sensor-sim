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
from .allan import (
    overlapping_allan_deviation,
    allan_variance,
    extract_noise_parameters,
    NoiseParams,
)
from .gnss import GNSSSensor, GNSSSpec, GNSSGrade
from .wheel import WheelOdometry, WheelSpec, WheelGrade
from .latency import LatencyModel, SensorClock, TimeSync, LatencyScenario
from .predict import PosePredictor, PredictionConfig, DisplayPipeline

__version__ = "0.5.0"
