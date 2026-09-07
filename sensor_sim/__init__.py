"""
Sensor Simulator: Generate realistic multi-sensor measurements for SLAM/state-estimation.

Supports:
  - Configurable IMU (consumer/tactical/navigation grade)
  - GNSS with multipath, signal loss, DOP scaling
  - Wheel odometry (Ackermann or differential)
  - Arbitrary 6-DoF trajectories
"""

from .adas import (
    AdasMarker,
    AdasPipeline,
    MarkerProjector,
    hazard_marker,
    lane_line_marker,
    lead_vehicle_marker,
)
from .allan import (
    NoiseParams,
    allan_variance,
    extract_noise_parameters,
    overlapping_allan_deviation,
)
from .camera import (
    CameraConfig,
    CameraFrame,
    CameraSensor,
    FeaturePoint,
    FlowFrame,
)
from .eskf import ESKF, ESKFConfig, ESKFState
from .evaluation import (
    EvalReport,
    GateResult,
    MetricStream,
    MonteCarloGate,
    NeesAccumulator,
    TrajectoryEvaluator,
    add_gate,
    compare_reports,
)
from .gnss import GNSSGrade, GNSSSensor, GNSSSpec
from .hazard import (
    HazardAssessment,
    ThreatPipeline,
    TTCModel,
    WarningArbitrator,
    WarningLevel,
    level_from_score,
)
from .imu import IMUSensor, IMUSpec, SensorGrade
from .latency import LatencyModel, LatencyScenario, SensorClock, TimeSync
from .lidar import (
    Box,
    GroundPlane,
    LidarConfig,
    LidarFrame,
    LidarSensor,
    Object,
    Sphere,
)
from .predict import (
    DisplayPipeline,
    PosePredictor,
    PredictionConfig,
    PredictorUncertainty,
    UncertaintyConfig,
)
from .radar import RadarConfig, RadarFrame, RadarSensor
from .scenarios import (
    Scenario,
    batch_summary,
    default_library,
    run_scenario,
    run_scenario_mc,
)
from .tracking import RadarTracker, Track, TrackConfig, TrackingFrame
from .trajectory import Trajectory
from .visibility import (
    Frustum,
    MarkerVisibilityPolicy,
    Occluder,
    VisibilityStatus,
    angular_size,
    evaluate_markers,
)
from .wheel import WheelGrade, WheelOdometry, WheelSpec

__version__ = "0.18.0"

__all__ = [
    "ESKF",
    "AdasMarker",
    "AdasPipeline",
    "Box",
    "CameraConfig",
    "CameraFrame",
    "CameraSensor",
    "DisplayPipeline",
    "ESKFConfig",
    "ESKFState",
    "EvalReport",
    "FeaturePoint",
    "FlowFrame",
    "Frustum",
    "GNSSGrade",
    "GNSSSensor",
    "GNSSSpec",
    "GateResult",
    "GroundPlane",
    "HazardAssessment",
    "IMUSensor",
    "IMUSpec",
    "LatencyModel",
    "LatencyScenario",
    "LidarConfig",
    "LidarFrame",
    "LidarSensor",
    "MarkerProjector",
    "MarkerVisibilityPolicy",
    "MetricStream",
    "MonteCarloGate",
    "NeesAccumulator",
    "NoiseParams",
    "Object",
    "Occluder",
    "PosePredictor",
    "PredictionConfig",
    "PredictorUncertainty",
    "RadarConfig",
    "RadarFrame",
    "RadarSensor",
    "RadarTracker",
    "Scenario",
    "SensorClock",
    "SensorGrade",
    "Sphere",
    "TTCModel",
    "ThreatPipeline",
    "TimeSync",
    "Track",
    "TrackConfig",
    "TrackingFrame",
    "Trajectory",
    "TrajectoryEvaluator",
    "UncertaintyConfig",
    "VisibilityStatus",
    "WarningArbitrator",
    "WarningLevel",
    "WheelGrade",
    "WheelOdometry",
    "WheelSpec",
    "add_gate",
    "allan_variance",
    "angular_size",
    "batch_summary",
    "compare_reports",
    "default_library",
    "evaluate_markers",
    "extract_noise_parameters",
    "hazard_marker",
    "lane_line_marker",
    "lead_vehicle_marker",
    "level_from_score",
    "overlapping_allan_deviation",
    "run_scenario",
    "run_scenario_mc",
]
