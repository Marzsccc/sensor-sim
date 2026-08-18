"""
ADAS marker projection & display-time latency compensation (v0.9.0).

The AR-HUD draws ADAS markers -- a lead-vehicle bbox, an FCW/AEB hazard
wedge, the lane boundaries -- anchored to *world* objects that the onboard
perception pipeline observed at an earlier capture time.  If the HUD
projects those markers using the latest fused ego pose, the marker lags the
real world by (pipeline delay * speed) meters, so a braking FCW alert sits
several meters behind the vehicle it is warning about, and lane lines no
longer align with the road.

This module completes the chain built in v0.5.0 (PosePredictor) and
v0.7.0 (PredictorUncertainty):

  1. `MarkerProjector` -- projects a world-frame ADAS marker (points or a
     single reference) into the HUD body frame.  It accepts either a *raw*
     ego pose or a *display-time compensated* ego pose, so both the naive
     and compensated render paths share one projection routine.

  2. `AdasMarker` / maker helpers -- typed markers matching the user's AR-HUD
     stack (ACC lead-vehicle, FCW/AEB hazard, lane lines), each carrying a
     world-frame anchor and a few helper accessors.

  3. `AdasPipeline` -- end-to-end loop that, at every display tick:
       - takes the latest fused ego pose (lags by the pipeline latency),
       - renders the marker with the naive pose,
       - renders the marker with the display-time predicted pose,
       - fades the marker alpha by the display-time uncertainty (bigger
         95%-ellipse -> more transparent / clamped), and
       - reports the *marker-on-screen* error (pixel / deg) for each path.

The key metric is marker reprojection error in the HUD body frame, which is
exactly what a driver perceives as "the box is 30 cm behind the car".

Reference: Azuma (1997) "A Survey of Augmented Reality" -- registration error
taxonomy (static, dynamic, temporal); Groves (2013) Ch.9 -- the display-time
prediction is the INS/GNSS integration prediction step specialized for the
short AR-HUD horizon.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .utils import (
    quat_to_rotmat,
    quat_conjugate,
    quat_rotate,
    quat_multiply,
    rotmat_to_quat,
)
from .trajectory import Trajectory
from .latency import LatencyModel
from .predict import PosePredictor, PredictionConfig, PredictorUncertainty
from .predict import _quat_yaw, _angle_diff


# ---------------------------------------------------------------------------
# ADAS marker types
# ---------------------------------------------------------------------------

@dataclass
class AdasMarker:
    """A world-anchored ADAS marker to be drawn on the HUD.

    Parameters
    ----------
    kind : str
        One of "lead_vehicle" (ACC), "hazard" (FCW/AEB), "lane_left",
        "lane_right", "lane_center".
    ref_world : (3,)
        A single representative world point (m) used for range/azimuth
        and for the aggregate error (e.g. the lead vehicle's bumper center,
        the hazard's closest point, a lane midpoint at range).
    points_world : (N,3), optional
        Full corner/edge points (e.g. a lead bbox) to be projected and drawn.
    label : str, optional
        Display label (e.g. "ACC target", "FCW").
    """
    kind: str
    ref_world: np.ndarray
    points_world: Optional[np.ndarray] = None
    label: str = ""

    def __post_init__(self):
        self.ref_world = np.asarray(self.ref_world, dtype=float)
        if self.points_world is not None:
            self.points_world = np.asarray(self.points_world, dtype=float)


def lead_vehicle_marker(
    center_world: np.ndarray,
    half_extent: Tuple[float, float, float] = (0.9, 0.45, 0.7),
) -> AdasMarker:
    """Bounding-box marker for an ACC lead vehicle.

    The box is axis-aligned in the world frame (sufficient for the demo;
    a real system uses the lead's yaw).  `ref_world` is the box center.
    """
    center = np.asarray(center_world, dtype=float)
    u, v, w = half_extent
    corners = np.array([
        [-u, 0.0, -w], [-u, 0.0,  w], [ u, 0.0,  w], [ u, 0.0, -w],
        [-u, 0.0, -w], [-u, v,   -w], [ u, v,   -w], [ u, 0.0, -w],
        [-u, 0.0,  w], [-u, v,    w], [ u, v,    w], [ u, 0.0,  w],
        [-u, v, -w], [-u, v, w], [u, v, w], [u, v, -w],
    ]) + center
    return AdasMarker(kind="lead_vehicle", ref_world=center,
                      points_world=corners, label="ACC target")


def hazard_marker(
    point_world: np.ndarray,
    half_angle_deg: float = 20.0,
    max_range: float = 30.0,
    label: str = "FCW/AEB",
) -> AdasMarker:
    """FCW/AEB hazard: a point plus a warning-wedge apex angle/range.

    The apex is the ego position (computed at projection time); the marker
    itself just carries the hazard world point so the wedge can be placed
    toward it on the HUD.
    """
    p = np.asarray(point_world, dtype=float)
    m = AdasMarker(kind="hazard", ref_world=p, label=label)
    object.__setattr__(m, "half_angle_deg", half_angle_deg)
    object.__setattr__(m, "max_range", max_range)
    return m


def lane_line_marker(
    start_world: np.ndarray,
    end_world: np.ndarray,
    which: str = "lane_center",
    label: str = "",
) -> AdasMarker:
    """A straight lane-boundary segment between two world points."""
    start = np.asarray(start_world, dtype=float)
    end = np.asarray(end_world, dtype=float)
    pts = np.vstack([start, end])
    ref = 0.5 * (start + end)
    return AdasMarker(kind=which, ref_world=ref, points_world=pts,
                      label=label or which)

# ---------------------------------------------------------------------------
# Marker projection into the HUD body frame
# ---------------------------------------------------------------------------

class MarkerProjector:
    """Project world-frame ADAS markers into the HUD body frame.

    The "HUD body frame" follows the sensor-sim convention: x forward,
    y right, z down (a right-handed, forward-right-down frame).  A marker
    point p_world is transformed by the ego pose T_wb = [R_wb | t_wb]:

        p_body = R_wb @ (p_world - t_wb)

    Because a pose quaternion `att` is world<-body, R_wb = quat_to_rotmat(att)
    already maps world -> body as required (see docstring of utils.py).

    The projector is deliberately pose-agnostic: pass the *raw* fused pose
    for the naive path or the *predicted* display-time pose for the
    compensated path.
    """

    @staticmethod
    def to_body(p_world: np.ndarray, pos: np.ndarray, att: np.ndarray) -> np.ndarray:
        """Map a world point to the HUD body frame given an ego pose.

        Returns the body-frame coordinates (x forward, y right, z down).
        """
        p_world = np.asarray(p_world, dtype=float)
        R_wb = quat_to_rotmat(att)
        return R_wb @ (p_world - np.asarray(pos, dtype=float))

    def project(
        self,
        marker: AdasMarker,
        pos: np.ndarray,
        att: np.ndarray,
    ) -> Dict[str, object]:
        """Project a marker under a given ego pose.

        Returns
        -------
        dict with keys:
          ref_body   : (3,) reference point in body frame
          range      : horizontal range to the ref point (m)
          azimuth    : atan2(y, x) of the ref in body frame (rad, + = right)
          points_body: (N,3) projected corners/edges, or None
          in_front   : bool, True if the ref lies forward of the HUD
          behind     : bool, True if it is behind the HUD origin
        """
        ref_body = self.to_body(marker.ref_world, pos, att)
        # horizontal (ignore the down/up z component for range/azimuth)
        x, y = ref_body[0], ref_body[1]
        rng = float(np.hypot(x, y))
        az = float(np.arctan2(y, x))
        in_front = x > 0.0
        pts = None
        if marker.points_world is not None:
            pts = np.stack([self.to_body(p, pos, att) for p in marker.points_world])
        return {
            "ref_body": ref_body,
            "range": rng,
            "azimuth": az,
            "points_body": pts,
            "in_front": in_front,
            "behind": not in_front,
        }

    @staticmethod
    def screen_xy(
        ref_body: np.ndarray,
        f_px_x: float = 1000.0,
        f_px_y: float = 1000.0,
        cx: float = 0.0,
        cy: float = 0.0,
    ) -> Tuple[np.ndarray, bool]:
        """Perspective-project a body point to HUD pixel coords (pinhole).

        The HUD looks along +x (forward).  The body frame has y right and
        z down, so a point at (x, y, z) maps as

            u = cx + f_px_x * y / x        (right positive)
            v = cy + f_px_y * z / x        (down positive)

        Points with x <= 0 (behind or at the HUD plane) are marked invalid
        and should be culled by the caller.

        Returns ((u, v), valid).
        """
        x = float(ref_body[0])
        if x <= 1e-6:
            return np.array([np.nan, np.nan]), False
        u = cx + f_px_x * float(ref_body[1]) / x
        v = cy + f_px_y * float(ref_body[2]) / x
        return np.array([u, v]), True


# ---------------------------------------------------------------------------
# End-to-end ADAS pipeline (naive vs compensated, uncertainty fade)
# ---------------------------------------------------------------------------

@dataclass
class AdasPipeline:
    """Simulate ADAS marker rendering under pipeline latency.

    Mirrors `DisplayPipeline` but focuses on *marker* (not ego-pose) error.
    At every display tick the pipeline:

      1. Samples the true ego pose at display time t.
      2. Builds the latest fused ego pose (lags by `total_delay_est`).
      3. Renders each marker with:

           naive      : fused pose (no prediction)     -> marker lags.
           compensated: PosePredictor-predicted pose   -> marker tracks.

      4. Computes the *marker reprojection error* = difference (in meters of
         horizontal radial offset and in screen pixels) between the naive /
         compensated body-frame marker position and the true one.

      5. Computes the display-time uncertainty ellipse and a fade alpha in
         [0,1] (1 = fully opaque) for the compensated render, so the HUD can
         be drawn more transparently when pose/uncertainty grows.

    Parameters
    ----------
    traj : Trajectory
        True ego trajectory.
    markers : List[AdasMarker]
        World-anchored ADAS markers to render.
    sensor_delay_s : float
        Primary (camera) pipeline delay (s).  The HUD anchors markers to the
        camera capture time, so the compensation horizon equals this + one
        display frame.
    display_rate : float
        Display refresh rate (Hz).
    use_wheel : bool
        Use wheel odometry in the pose predictor.
    """
    traj: Trajectory
    markers: List[AdasMarker] = field(default_factory=list)
    sensor_delay_s: float = 0.05
    display_rate: float = 60.0
    use_wheel: bool = True

    def __post_init__(self):
        self.total_delay_est = self.sensor_delay_s + 1.0 / self.display_rate
        self._predictor = PosePredictor(PredictionConfig(use_wheel=self.use_wheel))
        self._unc = PredictorUncertainty()

    # -- ego pose sources ------------------------------------------------

    def _fused_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latest fused ego pose at host time t (lags by the pipeline delay)."""
        t_state = t - self.total_delay_est
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t_state), 0, len(self.traj.points) - 1))
        pt = self.traj.points[i]
        return pt.pos.copy(), pt.vel.copy(), pt.att.copy()

    def _true_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t), 0, len(self.traj.points) - 1))
        pt = self.traj.points[i]
        return pt.pos.copy(), pt.vel.copy(), pt.att.copy()

    def _imu_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        pt = self.traj.points[i]
        return self.traj.body_accel(pt), pt.omega

    def _wheel_at(self, t: float) -> Tuple[float, float]:
        ts = self.traj.ts
        i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        pt = self.traj.points[i]
        speed = float(np.linalg.norm(pt.vel[:2]))
        yaw = _quat_yaw(pt.att)
        j = min(i + 1, len(ts) - 1)
        dt = max(ts[j] - ts[i], 1e-6)
        yaw_rate = (_quat_yaw(self.traj.points[j].att) - yaw) / dt
        return speed, yaw_rate

    def _compensated_pose(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted display-time ego pose at time t."""
        pos, vel, att = self._fused_pose(t)
        h = self.total_delay_est
        acc_meas, gyr_meas = self._imu_at(t - h)
        speed, yaw_rate = self._wheel_at(t - h)
        pred = self._predictor.predict(
            pos, vel, att, acc_meas, gyr_meas, h,
            wheel_speed=speed if self.use_wheel else None,
            wheel_yaw_rate=yaw_rate if self.use_wheel else None,
        )
        return pred["pos"], pred["vel"], pred["att"]

    # -- uncertainty fade ------------------------------------------------

    def _fade_alpha(self, t: float, P_init: np.ndarray,
                    sigma_scale: float = 1.0) -> Tuple[float, Dict[str, float]]:
        """Display-time fade alpha from the propagated pose covariance.

        The marker alpha is driven by how much the pose uncertainty grows
        over the compensation horizon: alpha shrinks as the 95% horizontal
        radius grows.  `sigma_scale` tunes how aggressive the fade is.

        Returns (alpha, ellipse_dict).
        alpha = clip( 1 - sigma_scale * (r95 / r_ref - 1), 0, 1 ), with
        r_ref a nominal "opaque" radius taken as the initial ellipse radius
        (so steady state is ~1.0 and outages/cold-start fade it down).
        """
        pos, vel, att = self._fused_pose(t)
        acc_meas, gyr_meas = self._imu_at(t - self.total_delay_est)
        res = self._unc.predict_with_covariance(
            self._predictor, pos, vel, att, acc_meas, gyr_meas,
            self.total_delay_est, P_init,
            wheel_speed=None, wheel_yaw_rate=None,
        )
        ellipse = res["ellipse"]
        r95 = ellipse["radius_95"]
        # nominal opaque radius: initial (pre-propagation) horizontal 95%
        p0 = np.asarray(P_init, dtype=float)[0:2, 0:2]
        lam = np.linalg.eigvalsh(0.5 * (p0 + p0.T))
        r_ref = 2.4477 * np.sqrt(0.5 * float(np.sum(lam)))
        r_ref = max(r_ref, 1e-6)
        alpha = float(np.clip(1.0 - sigma_scale * (r95 / r_ref - 1.0), 0.0, 1.0))
        return alpha, ellipse

    # -- main loop -------------------------------------------------------

    def run(self, P_init: np.ndarray,
            sigma_scale: float = 1.0,
            f_px: float = 1000.0) -> Dict[str, object]:
        """Run the marker rendering loop; aggregate naive vs compensated error.

        P_init is the 9x9 ESKF [p, v, theta] covariance describing the fused
        ego-state uncertainty (see PredictorUncertainty.propagate_covariance).

        Returns a dict with per-marker and aggregate stats:

          markers            : per-marker results
          horizon_s          : compensation horizon
          fade_alpha_mean    : mean marker alpha over the loop (0-1)
          summary            : aggregate naive/comp errors (radial m + px)
        """
        t0 = self.traj.points[0].t
        t1 = self.traj.points[-1].t
        h = self.total_delay_est
        t_start = t0 + h
        dt = 1.0 / self.display_rate

        # accumulate per marker
        per_marker = {}
        for m in self.markers:
            key = m.kind
            acc = {
                "naive_radial": [], "comp_radial": [],
                "naive_px": [], "comp_px": [],
                "range": [],
            }
            per_marker[key] = acc

        alphas = []

        t = t_start
        while t <= t1 + 1e-9:
            pos_t, vel_t, att_t = self._true_pose(t)
            pos_f, vel_f, att_f = self._fused_pose(t)
            pos_c, vel_c, att_c = self._compensated_pose(t)

            alpha, _ = self._fade_alpha(t, P_init, sigma_scale=sigma_scale)
            alphas.append(alpha)

            proj = MarkerProjector()
            for m in self.markers:
                acc = per_marker[m.kind]
                # naive marker body pose (lagged ego)
                n = proj.project(m, pos_f, att_f)
                # compensated marker body pose
                c = proj.project(m, pos_c, att_c)
                # true marker body pose (what the driver should see)
                tr = proj.project(m, pos_t, att_t)

                acc["range"].append(tr["range"])

                # radial reprojection error in the horizontal body plane
                n_rad = float(np.hypot(n["ref_body"][0] - tr["ref_body"][0],
                                       n["ref_body"][1] - tr["ref_body"][1]))
                c_rad = float(np.hypot(c["ref_body"][0] - tr["ref_body"][0],
                                       c["ref_body"][1] - tr["ref_body"][1]))
                acc["naive_radial"].append(n_rad)
                acc["comp_radial"].append(c_rad)

                # pixel error (project each to screen; skip invalid)
                (nu, nv), nvalid = MarkerProjector.screen_xy(n["ref_body"], f_px, f_px)
                (cu, cv), cvalid = MarkerProjector.screen_xy(c["ref_body"], f_px, f_px)
                (tu, tv), tvalid = MarkerProjector.screen_xy(tr["ref_body"], f_px, f_px)
                if nvalid and tvalid:
                    acc["naive_px"].append(float(np.hypot(nu - tu, nv - tv)))
                if cvalid and tvalid:
                    acc["comp_px"].append(float(np.hypot(cu - tu, cv - tv)))

            t += dt

        # aggregate
        markers_out = {}
        summary = {"naive_radial_mean": [], "comp_radial_mean": [],
                   "naive_px_mean": [], "comp_px_mean": [],
                   "naive_radial_max": [], "comp_radial_max": []}
        for key, acc in per_marker.items():
            m_out = {
                "count": len(acc["range"]),
                "range_mean": float(np.mean(acc["range"])),
                "naive_radial_mean": float(np.mean(acc["naive_radial"])),
                "comp_radial_mean": float(np.mean(acc["comp_radial"])),
                "naive_radial_max": float(np.max(acc["naive_radial"])),
                "comp_radial_max": float(np.max(acc["comp_radial"])),
            }
            if acc["naive_px"]:
                m_out["naive_px_mean"] = float(np.mean(acc["naive_px"]))
                m_out["naive_px_max"] = float(np.max(acc["naive_px"]))
            if acc["comp_px"]:
                m_out["comp_px_mean"] = float(np.mean(acc["comp_px"]))
                m_out["comp_px_max"] = float(np.max(acc["comp_px"]))
            m_out["fade_alpha_mean"] = float(np.mean(alphas)) if alphas else 1.0
            markers_out[key] = m_out

            summary["naive_radial_mean"].append(m_out["naive_radial_mean"])
            summary["comp_radial_mean"].append(m_out["comp_radial_mean"])
            summary["naive_px_mean"].append(m_out.get("naive_px_mean", np.nan))
            summary["comp_px_mean"].append(m_out.get("comp_px_mean", np.nan))
            summary["naive_radial_max"].append(m_out["naive_radial_max"])
            summary["comp_radial_max"].append(m_out["comp_radial_max"])

        agg = {
            "naive_radial_mean": float(np.mean(summary["naive_radial_mean"])),
            "comp_radial_mean": float(np.mean(summary["comp_radial_mean"])),
            "naive_radial_max": float(np.max(summary["naive_radial_max"])),
            "comp_radial_max": float(np.max(summary["comp_radial_max"])),
            "naive_px_mean": float(np.nanmean(summary["naive_px_mean"])),
            "comp_px_mean": float(np.nanmean(summary["comp_px_mean"])),
            "horizon_s": h,
            "fade_alpha_mean": float(np.mean(alphas)) if alphas else 1.0,
        }

        return {
            "markers": markers_out,
            "horizon_s": h,
            "fade_alpha_mean": agg["fade_alpha_mean"],
            "summary": agg,
        }
