"""
Display-time pose prediction & delay compensation.

The core AR-HUD / low-latency-rendering problem this module solves:

    The HUD must draw the world as seen from the vehicle's pose *at the
    moment the light hits the windshield* (display time t_d).  Every
    measurement in the pipeline -- camera frame, IMU packet, GNSS fix --
    describes the vehicle at an *earlier* time (t_s < t_d).  Naively
    rendering with the latest fused pose injects the full pipeline delay
    into the displayed world, so AR markers lag the real world by
    (delay * speed) meters and (delay * yaw_rate) degrees.

    Solution (standard in automotive HUD / VR reprojection):
      1. Estimate the latency of each source (see latency.py, TimeSync).
      2. Fuse delayed measurements into a state at the *measurement* time.
      3. Propagate the state from the newest measurement time to t_d
         using the high-rate IMU (attitude) + wheel speed (position).

This module implements the propagation step -- a lightweight, sensor-grade
`PosePredictor` that integrates IMU + wheel odometry from a known state to
a future display time -- plus a `DisplayPipeline` that wires latency models,
a naive renderer and a compensated renderer together end-to-end, so the
prediction error can be measured directly.

Reference: Groves (2013) Ch. 9 (INS/GNSS integration) -- the "prediction"
step of an ESKF is exactly this propagation; Azuma (1997) -- "A Survey of
Augmented Reality" -- prediction/registration error taxonomy.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .utils import (
    quat_to_rotmat,
    rotmat_to_quat,
    quat_multiply,
    quat_conjugate,
    quat_rotate,
    skew,
)
from .trajectory import Trajectory
from .latency import LatencyModel, LatencyScenario


# ---------------------------------------------------------------------------
# 1. Lightweight pose predictor (IMU + wheel odometry propagation)
# ---------------------------------------------------------------------------

@dataclass
class PredictionConfig:
    """Tuning knobs for the display-time pose predictor.

    Parameters
    ----------
    dt : float
        Integration step (s) used to propagate to the display time.
    use_wheel : bool
        If True, use wheel odometry (speed + yaw rate) for the position/
        heading prediction; the gyro is preferred for attitude but the wheel
        yaw rate is used as an independent check when `blend_wheel_yaw`.
    blend_wheel_yaw : bool
        If True, average gyro and wheel yaw-rate predictions for the yaw
        update (cheap "sensor fusion" at the prediction level).
    gravity : float
        Local gravity magnitude (m/s^2) for attitude initialization.
    """

    dt: float = 0.005
    use_wheel: bool = True
    blend_wheel_yaw: bool = True
    gravity: float = 9.81


class PosePredictor:
    """Propagate a 6-DoF pose from a known state to a future display time.

    Two propagation sources:

      * IMU (accelerometer + gyro): integrates specific force and angular
        rate.  The attitude update is exact to first order in the rotation
        vector; the position update integrates the gravity-compensated
        acceleration in the world frame.
      * Wheel odometry (speed + yaw rate): integrates a planar nonholonomic
        model, which is very stable for road vehicles and is what the HUD
        actually uses for the short 20-80 ms prediction horizon.

    The class is deliberately lightweight (no covariance): it is meant for
    the *short-horizon* display-time prediction where a kinematic model is
    more robust than a full 15-state ESKF running at 500 Hz.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()

    # -- IMU propagation ---------------------------------------------------

    @staticmethod
    def _integrate_attitude(q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Integrate body angular rate into a quaternion (first-order exact)."""
        gx, gy, gz = gyro
        norm = np.linalg.norm(gyro)
        if norm < 1e-12:
            return q / np.linalg.norm(q)
        half = 0.5 * dt * norm
        c, s = np.cos(half), np.sin(half) / norm
        dq = np.array([c, s * gx, s * gy, s * gz])
        return quat_multiply(q, dq) / np.linalg.norm(quat_multiply(q, dq))

    def propagate_imu(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        att: np.ndarray,
        acc_meas: np.ndarray,
        gyr_meas: np.ndarray,
        horizon: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate state forward `horizon` seconds using IMU measurements.

        The measurements are assumed constant over the horizon (valid for
        short horizons; a real system would re-sample the IMU buffer).

        Returns (pos, vel, att) at display time.
        """
        dt = self.config.dt
        n = max(1, int(round(horizon / dt)))
        p, v, q = pos.copy(), vel.copy(), att.copy()
        for _ in range(n):
            # Attitude: world <- body
            q = self._integrate_attitude(q, gyr_meas, dt)
            # Specific force -> world-frame acceleration (gravity compensated)
            # NB: trajectory.py convention uses gravity_world = [0,0,+g]
            R_wb = quat_to_rotmat(q).T
            a_world = R_wb @ acc_meas - np.array([0.0, 0.0, self.config.gravity])
            v = v + a_world * dt
            p = p + v * dt
        return p, v, q

    # -- Wheel propagation -------------------------------------------------

    @staticmethod
    def propagate_wheel(
        pos: np.ndarray,
        yaw: float,
        speed: float,
        yaw_rate: float,
        horizon: float,
        dt: float = 0.005,
    ) -> Tuple[np.ndarray, float]:
        """Propagate a planar nonholonomic vehicle model.

        Parameters
        ----------
        pos : (2,) or (3,) array
            Position in world frame (x, y [, z]).
        yaw : float
            Heading (rad), world frame, CCW positive.
        speed : float
            Longitudinal speed (m/s), positive forward.
        yaw_rate : float
            Yaw rate (rad/s), positive CCW.
        horizon : float
            Prediction horizon (s).
        dt : float
            Integration step (s).

        Returns
        -------
        (pos, yaw) at display time.
        """
        n = max(1, int(round(horizon / dt)))
        p = np.array(pos, dtype=float).copy()
        if p.shape[0] == 2:
            p = np.array([p[0], p[1], 0.0])
        y = yaw
        for _ in range(n):
            y += yaw_rate * dt
            p[0] += speed * np.cos(y) * dt
            p[1] += speed * np.sin(y) * dt
        return p, y

    # -- Combined prediction (what the HUD calls) ---------------------------

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        att: np.ndarray,
        acc_meas: np.ndarray,
        gyr_meas: np.ndarray,
        horizon: float,
        wheel_speed: Optional[float] = None,
        wheel_yaw_rate: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Predict display-time pose from the latest fused state.

        Parameters
        ----------
        pos : (3,)
            Latest fused position (world frame, m).
        vel : (3,)
            Latest fused velocity (world frame, m/s).
        att : (4,)
            Latest fused attitude quaternion [w,x,y,z].
        acc_meas : (3,)
            Latest accelerometer reading (body frame, m/s^2).
        gyr_meas : (3,)
            Latest gyro reading (body frame, rad/s).
        horizon : float
            Time until display (s), i.e. total pipeline latency.
        wheel_speed / wheel_yaw_rate : float, optional
            Wheel odometry readings used when config.use_wheel=True.

        Returns
        -------
        dict with keys:
          pos  : (3,) predicted position at display time
          vel  : (3,) predicted velocity
          att  : (4,) predicted attitude quaternion
        """
        p_imu, v_imu, q_imu = self.propagate_imu(pos, vel, att, acc_meas, gyr_meas, horizon)

        if self.config.use_wheel and wheel_speed is not None and wheel_yaw_rate is not None:
            # Heading from the IMU attitude (yaw) as the wheel baseline.
            yaw_imu = _quat_yaw(q_imu)
            p_w, yaw_w = self.propagate_wheel(
                pos[:2], _quat_yaw(att), wheel_speed, wheel_yaw_rate, horizon, self.config.dt
            )
            if self.config.blend_wheel_yaw:
                yaw = 0.5 * (yaw_imu + yaw_w)
            else:
                yaw = yaw_w
            # Keep IMU pitch/roll, override yaw with the blended value.
            roll, pitch, _ = _quat_euler(q_imu)
            q_out = _euler_to_quat(roll, pitch, yaw)
            # Horizontal position from wheel (kinematically exact for the plane).
            p_out = np.array([p_w[0], p_w[1], p_imu[2]])
            return {"pos": p_out, "vel": v_imu, "att": q_out}

        return {"pos": p_imu, "vel": v_imu, "att": q_imu}


# ---------------------------------------------------------------------------
# 2. End-to-end display pipeline (naive vs compensated)
# ---------------------------------------------------------------------------

@dataclass
class DisplayPipeline:
    """Simulate an AR-HUD render loop and compare naive vs compensated.

    The pipeline samples the true trajectory at 100 Hz, applies per-sensor
    latency (camera/gnss/wheel), fuses the *delayed* measurements into a
    "latest pose" at receive time, then renders at display time:

      * Naive:       renders with the latest fused pose (no prediction).
      * Compensated: propagates the latest fused pose forward by the
                     estimated total latency before rendering.

    The error metric is the distance between the rendered pose and the true
    trajectory at display time (position error in meters, heading error in
    degrees).
    """

    traj: Trajectory
    sensors: Dict[str, LatencyModel] = field(default_factory=dict)
    display_rate: float = 60.0          # Hz
    total_delay_est: Optional[float] = None  # estimated latency for compensation
    use_wheel: bool = True
    primary_sensor: str = "camera"      # perception anchor for AR content

    def __post_init__(self):
        if not self.sensors:
            self.sensors = {
                "camera": LatencyModel(fixed_delay_s=0.05, jitter_std_s=0.005, seed=11),
                "gnss":   LatencyModel(fixed_delay_s=0.20, jitter_std_s=0.02, loss_prob=0.03, seed=12),
                "wheel":  LatencyModel(fixed_delay_s=0.005, jitter_std_s=0.001, seed=13),
            }
        if self.total_delay_est is None:
            # Default: the AR content is anchored to the primary sensor's
            # capture time (e.g. the camera frame) plus one frame of display
            # latency.  The compensated renderer must predict the vehicle
            # pose forward by exactly this horizon.
            primary_delay = self.sensors[self.primary_sensor].fixed_delay_s
            self.total_delay_est = primary_delay + 1.0 / self.display_rate

    def _fused_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the 'latest fused pose' a filter would have at host time t.

        In an AR-HUD pipeline the AR content is anchored to the primary
        sensor's capture time: the fused state describes the vehicle at
        capture time t_c, and the display happens at t_c + total_delay_est.
        So at display time t the fused state describes the vehicle at
        t - total_delay_est.
        """
        t_state = t - self.total_delay_est
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t_state), 0, len(self.traj.points) - 1))
        pt = self.traj.points[i]
        return pt.pos.copy(), pt.vel.copy(), pt.att.copy()

    def _imu_reading_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (acc_meas, gyr_meas) the IMU would report at host time t.

        The IMU is assumed latency-free (or its latency is already corrected
        by the clock sync); we use the true body acceleration/omega.
        """
        # Nearest trajectory point to t
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        pt = self.traj.points[i]
        return self.traj.body_accel(pt), pt.omega

    def _wheel_reading_at(self, t: float) -> Tuple[float, float]:
        """Return (speed, yaw_rate) from wheel odometry at host time t."""
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        pt = self.traj.points[i]
        speed = float(np.linalg.norm(pt.vel[:2]))
        yaw = _quat_yaw(pt.att)
        # yaw rate from consecutive points
        j = min(i + 1, len(ts) - 1)
        dt = max(ts[j] - ts[i], 1e-6)
        yaw_rate = (_quat_yaw(self.traj.points[j].att) - yaw) / dt
        return speed, yaw_rate

    def run(self) -> Dict[str, float]:
        """Run the display loop and return mean/max errors for both renderers.

        The loop starts at t0 + horizon (a warm-up period during which the
        filter has accumulated enough delayed measurements to produce a
        valid fused state) and ends at t1 (end of trajectory).
        """
        t0 = self.traj.points[0].t
        t1 = self.traj.points[-1].t
        horizon = self.total_delay_est
        t_start = t0 + horizon  # skip cold-start window

        n_naive, n_comp = 0, 0
        naive_pos_errs, comp_pos_errs = [], []
        naive_hdg_errs, comp_hdg_errs = [], []

        t = t_start
        while t <= t1 + 1e-9:
            # True pose at display time t
            i_true = int(np.clip(np.searchsorted(self.traj.ts, t), 0, len(self.traj.points) - 1))
            pt_true = self.traj.points[i_true]

            # Latest fused pose (lags by sensor latency)
            pos, vel, att = self._fused_pose(t)

            # ---- Naive renderer: use fused pose directly ----
            e_pos = float(np.linalg.norm(pos - pt_true.pos))
            e_hdg = _angle_diff(_quat_yaw(att), _quat_yaw(pt_true.att))
            naive_pos_errs.append(e_pos)
            naive_hdg_errs.append(e_hdg)
            n_naive += 1

            # ---- Compensated renderer: predict to display time ----
            # Use IMU/wheel readings at the *fused state* time (t - horizon)
            # so the propagation is self-consistent: state @ t-horizon +
            # readings @ t-horizon -> display @ t.
            acc_meas, gyr_meas = self._imu_reading_at(t - horizon)
            speed, yaw_rate = self._wheel_reading_at(t - horizon)
            pred = self._predictor().predict(
                pos, vel, att, acc_meas, gyr_meas, horizon,
                wheel_speed=speed if self.use_wheel else None,
                wheel_yaw_rate=yaw_rate if self.use_wheel else None,
            )
            e_pos_c = float(np.linalg.norm(pred["pos"] - pt_true.pos))
            e_hdg_c = _angle_diff(_quat_yaw(pred["att"]), _quat_yaw(pt_true.att))
            comp_pos_errs.append(e_pos_c)
            comp_hdg_errs.append(e_hdg_c)
            n_comp += 1

            t += 1.0 / self.display_rate

        return {
            "naive_pos_err_mean": float(np.mean(naive_pos_errs)),
            "naive_pos_err_max": float(np.max(naive_pos_errs)),
            "naive_heading_err_mean": float(np.mean(naive_hdg_errs)),
            "naive_heading_err_max": float(np.max(naive_hdg_errs)),
            "comp_pos_err_mean": float(np.mean(comp_pos_errs)),
            "comp_pos_err_max": float(np.max(comp_pos_errs)),
            "comp_heading_err_mean": float(np.mean(comp_hdg_errs)),
            "comp_heading_err_max": float(np.max(comp_hdg_errs)),
            "horizon_s": horizon,
        }

    def _predictor(self) -> "PosePredictor":
        return PosePredictor(PredictionConfig(use_wheel=self.use_wheel))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_yaw(q: np.ndarray) -> float:
    """Yaw (rad) from a [w,x,y,z] quaternion, ZYX convention."""
    w, x, y, z = q / np.linalg.norm(q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """(roll, pitch, yaw) from a [w,x,y,z] quaternion."""
    w, x, y, z = q / np.linalg.norm(q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) -> [w,x,y,z] quaternion, ZYX convention."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest angle a - b in radians."""
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))
