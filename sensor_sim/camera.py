"""Camera image / feature & optical-flow simulation (v0.13.0).

A forward-facing pinhole camera that projects analytic 3D world features into
pixels and tracks them across frames to produce per-feature optical flow -- the
roadmap item that pairs with the existing IMU/GNSS/wheel/outage/LiDAR modules
and is directly relevant to monocular/VIO and AR-HUD work.

Design goals (mirror the rest of :mod:`sensor_sim`):

- **Single source of truth for the sensor pose.**  The camera sits on a rigid
  mount (``mount_t_body``) on the vehicle body; world features are transformed
  into the sensor frame exactly like ``MarkerProjector`` / ``LidarSensor``
  compose their mounts, using ``_quat_to_rotmat``.
- **Analytic projection** (no rasterization): a pinhole model with focal
  length, principal point and (optional) barrel distortion.  Features are
  3D points (or small spheres) that we transform and project; deterministic
  and easy to validate by hand.
- **Optical flow / feature tracking:** given the same world feature observed
  from two host poses, we compute its pixel position in each frame, its
  displacement (the true optical flow), and its per-frame point velocity
  (px/s).  This is the measurement a VO / feature-tracker consumes.
- **Realistic measurement effects:** Gaussian pixel noise, detection dropout
  (low contrast / far / motion blur), and front/back-hemisphere culling via an
  optional frustum so occluded / behind-camera features are dropped.

Coordinate conventions (kept identical to the rest of sensor-sim):
- World is **z-up**; body frame **+x forward, +y left, +z up**.
- Camera optical axis is **+x** in the sensor/body frame (forward-looking),
  with image x to the right (+y) and image y down (-z), a standard right-handed
  forward camera.
- ``R_bw = _quat_to_rotmat(att)`` (world <- body); body->world is ``R_bw.T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .trajectory import _quat_to_rotmat
from .lidar import Object, GroundPlane, Box, Sphere  # reuse world primitives


# --------------------------------------------------------------------------
# Camera intrinsics & config
# --------------------------------------------------------------------------
@dataclass
class CameraConfig:
    """Configuration for a forward-facing pinhole camera.

    Parameters
    ----------
    fx, fy : float
        Focal length in pixels (px).  ``fy`` defaults to ``fx``.
    cx, cy : float
        Principal point in pixels (default image centre).
    width, height : int
        Image resolution in pixels.
    k1 : float
        Radial distortion coefficient (barrel for k1 < 0, pincushion > 0).
        ``0`` disables distortion.
    min_z : float
        Features closer than this (in sensor +x) are dropped (near clip).
    max_z : float
        Features farther than this are dropped (far clip / detection range).
    pixel_noise_sigma : float
        Std dev of additive Gaussian pixel noise (px).
    dropout_prob : float
        Probability any *visible* feature is missed (contrast/motion-blur/far).
    """
    fx: float = 600.0
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 1280
    height: int = 720
    k1: float = 0.0
    min_z: float = 0.1
    max_z: float = 150.0
    pixel_noise_sigma: float = 0.5
    dropout_prob: float = 0.02

    def __post_init__(self):
        if self.fy is None:
            self.fy = self.fx
        if self.cx is None:
            self.cx = self.width / 2.0
        if self.cy is None:
            self.cy = self.height / 2.0
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx/fy must be > 0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be > 0")
        if self.min_z < 0 or self.max_z <= self.min_z:
            raise ValueError("need 0 <= min_z < max_z")


# --------------------------------------------------------------------------
# Output frame
# --------------------------------------------------------------------------
@dataclass
class CameraFrame:
    """One feature observation frame.

    Parameters
    ----------
    t : float
        Host time (s).
    pts : np.ndarray
        ``(N, 2)`` observed pixel coordinates (u = column, v = row).
    depths : np.ndarray
        ``(N,)`` depth (sensor +x distance, m) of each feature.
    feature_ids : np.ndarray
        ``(N,)`` stable id of each feature (for cross-frame tracking).
    object_ids : np.ndarray
        ``(N,)`` id of the object each feature belongs to (-1 = free point).
    """
    t: float
    pts: np.ndarray
    depths: np.ndarray
    feature_ids: np.ndarray
    object_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.pts.shape[0])


@dataclass
class FlowFrame:
    """Dense feature *tracking* result between two host poses.

    For every feature observed in *both* frames we store its pixel position in
    the previous and current frame plus the resulting displacement.

    Parameters
    ----------
    t0, t1 : float
        Previous and current host times (s).
    prev_pts : np.ndarray
        ``(M, 2)`` pixel positions in the previous frame.
    cur_pts : np.ndarray
        ``(M, 2)`` pixel positions in the current frame.
    flow : np.ndarray
        ``(M, 2)`` pixel displacement ``cur_pts - prev_pts`` (the optical flow).
    feature_ids : np.ndarray
        ``(M,)`` feature ids tracked across both frames.
    depths : np.ndarray
        ``(M,)`` current-frame depth (m) of each tracked feature.
    vel_px : np.ndarray
        ``(M, 2)`` per-feature pixel velocity (px/s) = flow / dt.
    """
    t0: float
    t1: float
    prev_pts: np.ndarray
    cur_pts: np.ndarray
    flow: np.ndarray
    feature_ids: np.ndarray
    depths: np.ndarray
    vel_px: np.ndarray

    def __len__(self) -> int:
        return int(self.flow.shape[0])


# --------------------------------------------------------------------------
# Feature
# --------------------------------------------------------------------------
@dataclass
class FeaturePoint:
    """A tracked 3D landmark with a stable id for cross-frame association.

    Parameters
    ----------
    feature_id : int
        Stable id used to match the same landmark across frames.
    pos_w : np.ndarray
        Resting world position ``[3]`` (m).
    object_id : int
        Owning object id (-1 for free-standing points), e.g. for an object a
        feature is attached to.
    velocity : np.ndarray
        Constant world velocity (m/s) added to ``pos_w`` for dynamic objects.
    """
    feature_id: int
    pos_w: np.ndarray
    object_id: int = -1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def world_pos(self, t: float) -> np.ndarray:
        """World position at host time ``t`` (dynamic features move)."""
        return self.pos_w + self.velocity * t


# --------------------------------------------------------------------------
# Camera sensor
# --------------------------------------------------------------------------
class CameraSensor:
    """Simulated forward-facing pinhole camera mounted on the vehicle body.

    Parameters
    ----------
    config : CameraConfig
        Intrinsics, resolution and noise.
    mount_t_body : np.ndarray, optional
        Translation of the camera in body frame (``[3]``), default origin.
    """

    def __init__(self, config: CameraConfig,
                 mount_t_body: Optional[np.ndarray] = None):
        self.config = config
        self.mount_t_body = (
            np.asarray(mount_t_body, dtype=float)
            if mount_t_body is not None else np.zeros(3)
        )

    # -- rigid transform helpers -------------------------------------------
    def _pose_world_to_sensor(self, point):
        """Compose host pose + mount into world->sensor rotation and origin."""
        pos_w = np.asarray(point.pos, dtype=float)
        R_bw = _quat_to_rotmat(point.att)     # world -> body
        R_wb = R_bw.T                         # body -> world
        origin_w = pos_w + R_wb @ self.mount_t_body
        return R_bw, origin_w

    # -- projection ---------------------------------------------------------
    def project(self, p_sensor: np.ndarray):
        """Project sensor-frame points ``(...,3)`` to pixel coords ``(...,2)``.

        Sensor frame is +x forward / +y right / -z down (image axes), so we map
        ``(x, y, z) -> (u, v) = (fx*x/z + cx, fy*(-z)/z + cy)``.  Only points
        with ``x > 0`` (in front) are projected; returns NaN otherwise.
        """
        cfg = self.config
        p = np.asarray(p_sensor, dtype=float)
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        valid = x > 0.0
        out = np.full(p.shape[:-1] + (2,), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_z = np.where(valid, 1.0 / np.where(x > 0, x, 1.0), 0.0)
            u = cfg.fx * y * inv_z + cfg.cx
            v = cfg.fy * (-z) * inv_z + cfg.cy
        # radial distortion (barrel/pincushion) on the normalized radius.
        if cfg.k1 != 0.0:
            xn = y * inv_z        # normalized image x
            yn = (-z) * inv_z     # normalized image y
            r2 = xn * xn + yn * yn
            du = cfg.fx * xn * (cfg.k1 * r2)
            dv = cfg.fy * yn * (cfg.k1 * r2)
            u = u + du
            v = v + dv
        out[..., 0] = np.where(valid, u, np.nan)
        out[..., 1] = np.where(valid, v, np.nan)
        return out

    # -- single observation -------------------------------------------------
    def observe(self, point, features: List[FeaturePoint], t: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> CameraFrame:
        """Project and observe all features from the given host pose.

        Parameters
        ----------
        point : TrajPoint
            Ground-truth host state (position + ``att`` quaternion).
        features : list of FeaturePoint
            World landmarks to observe.
        t : float
            Host time (s) passed to dynamic features.
        rng : np.random.Generator, optional
            RNG for repeatable noise.

        Returns
        -------
        CameraFrame
            Features that are in front (``min_z <= x <= max_z``), inside the
            image and not dropped.
        """
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng()
        R_bw, origin_w = self._pose_world_to_sensor(point)

        seen = []
        for f in features:
            pw = f.world_pos(t)
            p_cam = R_bw @ (pw - origin_w)      # world -> sensor frame
            x = p_cam[0]
            if x < cfg.min_z or x > cfg.max_z:
                continue                          # outside depth range
            px = self.project(p_cam.reshape(1, 3))[0]
            if not np.isfinite(px).all():
                continue
            u, v = px
            if u < 0 or u >= cfg.width or v < 0 or v >= cfg.height:
                continue                          # outside image
            if rng.random() < cfg.dropout_prob:
                continue                          # missed detection
            # add pixel noise
            noise = rng.normal(0.0, cfg.pixel_noise_sigma, size=2)
            seen.append((u + noise[0], v + noise[1], x,
                         f.feature_id, f.object_id))

        if not seen:
            return CameraFrame(t=t, pts=np.empty((0, 2)), depths=np.empty(0),
                               feature_ids=np.empty(0, dtype=int),
                               object_ids=np.empty(0, dtype=int))
        arr = np.array(seen, dtype=float)
        return CameraFrame(
            t=t,
            pts=arr[:, :2],
            depths=arr[:, 2],
            feature_ids=arr[:, 3].astype(int),
            object_ids=arr[:, 4].astype(int),
        )

    # -- two-frame tracking / optical flow ----------------------------------
    def track(self, p0, p1, features: List[FeaturePoint], t0: float,
              t1: float, rng: Optional[np.random.Generator] = None) -> FlowFrame:
        """Track features across two host poses and compute optical flow.

        Parameters
        ----------
        p0, p1 : TrajPoint
            Previous and current host states.
        features : list of FeaturePoint
            World landmarks (dynamic features move between ``t0`` and ``t1``).
        t0, t1 : float
            Previous and current host times (s).
        rng : np.random.Generator, optional
            RNG (shared so both frames get the same dropout pattern).

        Returns
        -------
        FlowFrame
            Per-feature previous/current pixel position, displacement, depth
            and pixel velocity.
        """
        f0 = self.observe(p0, features, t=t0, rng=rng)
        f1 = self.observe(p1, features, t=t1, rng=rng)
        if len(f1) == 0:
            return FlowFrame(t0=t0, t1=t1, prev_pts=np.empty((0, 2)),
                             cur_pts=np.empty((0, 2)), flow=np.empty((0, 2)),
                             feature_ids=np.empty(0, dtype=int),
                             depths=np.empty(0), vel_px=np.empty((0, 2)))

        # associate by feature id (present in both frames)
        id_to_cur = {fid: i for i, fid in enumerate(f1.feature_ids)}
        prev = []
        cur = []
        ids = []
        depths = []
        for i, fid in enumerate(f0.feature_ids):
            j = id_to_cur.get(fid)
            if j is not None:
                prev.append(f0.pts[i])
                cur.append(f1.pts[j])
                ids.append(fid)
                depths.append(f1.depths[j])

        if not prev:
            return FlowFrame(t0=t0, t1=t1, prev_pts=np.empty((0, 2)),
                             cur_pts=np.empty((0, 2)), flow=np.empty((0, 2)),
                             feature_ids=np.empty(0, dtype=int),
                             depths=np.empty(0), vel_px=np.empty((0, 2)))
        prev = np.array(prev, dtype=float)
        cur = np.array(cur, dtype=float)
        ids = np.array(ids, dtype=int)
        depths = np.array(depths, dtype=float)
        flow = cur - prev
        dt = t1 - t0
        vel_px = flow / dt if dt > 0 else np.zeros_like(flow)
        return FlowFrame(t0=t0, t1=t1, prev_pts=prev, cur_pts=cur, flow=flow,
                         feature_ids=ids, depths=depths, vel_px=vel_px)


# Re-export world primitives so camera users can build worlds with one import.
__all__ = [
    "CameraConfig",
    "CameraFrame",
    "FlowFrame",
    "FeaturePoint",
    "CameraSensor",
    "Object",
    "GroundPlane",
    "Box",
    "Sphere",
]
