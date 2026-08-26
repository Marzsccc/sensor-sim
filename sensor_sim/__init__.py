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
from .evaluation import (
    NeesAccumulator,
    MetricStream,
    GateResult,
    EvalReport,
    TrajectoryEvaluator,
    MonteCarloGate,
    add_gate,
    compare_reports,
)
from .scenarios import (
    Scenario,
    run_scenario,
    run_scenario_mc,
    default_library,
    batch_summary,
)

__version__ = "0.15.1"

__all__ = [
    "AdasMarker",
    "AdasPipeline",
    "Box",
    "CameraConfig",
    "CameraFrame",
    "CameraSensor",
    "DisplayPipeline",
    "ESKF",
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
    "Scenario",
    "SensorClock",
    "SensorGrade",
    "Sphere",
    "TTCModel",
    "ThreatPipeline",
    "TimeSync",
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
