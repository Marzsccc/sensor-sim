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
from .predict import (
    PosePredictor,
    PredictionConfig,
    DisplayPipeline,
    PredictorUncertainty,
    UncertaintyConfig,
)
from .adas import (
    AdasMarker,
    MarkerProjector,
    AdasPipeline,
    lead_vehicle_marker,
    hazard_marker,
    lane_line_marker,
)
from .visibility import (
    Frustum,
    Occluder,
    MarkerVisibilityPolicy,
    VisibilityStatus,
    angular_size,
    evaluate_markers,
)
from .hazard import (
    WarningLevel,
    TTCModel,
    HazardAssessment,
    WarningArbitrator,
    ThreatPipeline,
    level_from_score,
)
from .lidar import (
    Object,
    GroundPlane,
    Box,
    Sphere,
    LidarConfig,
    LidarFrame,
    LidarSensor,
)
from .camera import (
    CameraConfig,
    CameraFrame,
    FlowFrame,
    FeaturePoint,
    CameraSensor,
)
from .eskf import ESKF, ESKFConfig, ESKFState

__version__ = "0.13.0"
