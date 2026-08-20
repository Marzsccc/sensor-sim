"""LiDAR point-cloud simulation (v0.12.0).

Raycast a configurable LiDAR against a simple analytic world (ground plane,
axis-aligned boxes, spheres, optionally dynamic objects moving at constant
velocity) to produce realistic per-scan point clouds for SLAM / ADAS /
sensor-fusion development -- the roadmap item that pairs with the existing
IMU/GNSS/wheel/outage modules.

Design goals (mirroring the rest of :mod:`sensor_sim`):
- **Single source of truth for the sensor pose.**  The LiDAR sits on a rigid
  mount (``mount_t_body``) attached to the vehicle body.  Given a host
  trajectory point (position + quaternion), we compose the mount pose into
  the world frame exactly like ``MarkerProjector`` composes the camera mount.
- **Analytic raycasting** (no mesh rasterization): every ray is solved for the
  nearest intersection among ground-plane / AABB / sphere primitives, then
  clamped to ``[range_min, range_max]``.  Deterministic, fast, and easy to
  validate by hand.
- **Two scan geometries:**
    * ``spinning`` -- full 360° horizontal sweep, one fixed elevation per
      beam (classic mechanical VLP-16 style).  Useful for surround SLAM.
    * ``solid``  -- rectangular FOV (azimuth × elevation grid), the pattern
      of a forward-facing automotive unit (e.g. for the AR-HUD/ADAS stack).
- **Realistic noise & dropout:**
    * Gaussian range noise (std configurable per grade),
    * per-point dropout probability (finite reflectivity / far-range loss),
    * a ``reflectivity`` field so downstream modules can threshold dim returns,
    * dynamic objects move at constant velocity between ticks, so a target
      that crosses the beam leaves a smeared, moving cluster in the cloud.

Coordinate conventions (kept identical to :mod:`sensor_sim.trajectory`):
- World is **z-up**; ground plane at ``z = ground_z``.
- Body frame has **+x forward, +y left, +z up** (the sensor-sim convention).
- A ray in sensor/body frame ``d_body`` is rotated to world with
  ``R_bw.T @ d_body`` where ``R_bw = _quat_to_rotmat(att)`` (world <- body).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .trajectory import _quat_to_rotmat


# --------------------------------------------------------------------------
# World primitives
# --------------------------------------------------------------------------
@dataclass
class Object:
    """Abstract analytic world object. Subclasses implement ``_hits``.

    Parameters
    ----------
    object_id : int
        Stable id so downstream code can track which object produced a point.
    """
    object_id: int

    def world_pos(self, t: float) -> np.ndarray:
        """Center/position of the object at host time ``t`` (world, [3])."""
        raise NotImplementedError

    def _hits(self, origin: np.ndarray, direction: np.ndarray,
              center: np.ndarray) -> Optional[float]:
        """Return the nearest positive intersection distance (m) or None."""
        raise NotImplementedError

    def intersect(self, origin: np.ndarray, direction: np.ndarray,
                  t: float) -> Optional[float]:
        """Raycast in world frame at host time ``t``."""
        return self._hits(origin, direction, self.world_pos(t))


@dataclass
class GroundPlane(Object):
    """Infinite horizontal plane (default z = 0).

    Parameters
    ----------
    ground_z : float
        Plane elevation in world z (default 0).
    """
    ground_z: float = 0.0

    def world_pos(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, self.ground_z])

    def _hits(self, origin, direction, center):
        z0 = center[2]
        dz = direction[2]
        if abs(dz) < 1e-12:          # ray parallel to plane
            return None
        t_hit = (z0 - origin[2]) / dz
        return t_hit if t_hit > 0 else None


@dataclass
class Box(Object):
    """Axis-aligned box defined by center + half extents. Axis-aligned in the
    *world* frame, so it represents a stationary landmark by default; make it
    dynamic by giving it a nonzero ``velocity``.

    Parameters
    ----------
    center : np.ndarray
        Resting center (world, [3]).
    half_extents : np.ndarray
        Half-size along x/y/z (world, [3]).
    velocity : np.ndarray
        Constant velocity (m/s) added to ``center`` for ``world_pos``.
    reflectivity : float
        Material reflectivity in [0, 1]; scales the returned intensity.
    """
    center: np.ndarray
    half_extents: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reflectivity: float = 0.5

    def world_pos(self, t: float) -> np.ndarray:
        return self.center + self.velocity * t

    def _hits(self, origin, direction, center):
        # Slab method. AABB corners: center +/- half_extents.
        hmin = center - self.half_extents
        hmax = center + self.half_extents
        tmin = -np.inf
        tmax = np.inf
        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < hmin[i] or origin[i] > hmax[i]:
                    return None            # ray parallel, misses slab
            else:
                inv = 1.0 / direction[i]
                t1 = (hmin[i] - origin[i]) * inv
                t2 = (hmax[i] - origin[i]) * inv
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        return tmin if tmin > 0 else (tmax if tmax > 0 else None)


@dataclass
class Sphere(Object):
    """Sphere primitive (dynamic velocity supported)."""
    center: np.ndarray
    radius: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reflectivity: float = 0.6

    def world_pos(self, t: float) -> np.ndarray:
        return self.center + self.velocity * t

    def _hits(self, origin, direction, center):
        oc = origin - center
        a = float(np.dot(direction, direction))
        if a < 1e-12:
            return None
        b = 2.0 * float(np.dot(oc, direction))
        c = float(np.dot(oc, oc)) - self.radius ** 2
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return None
        sqrt = np.sqrt(disc)
        t0 = (-b - sqrt) / (2.0 * a)
        t1 = (-b + sqrt) / (2.0 * a)
        if t0 > 0:
            return t0
        if t1 > 0:
            return t1
        return None


# --------------------------------------------------------------------------
# LiDAR configuration
# --------------------------------------------------------------------------
@dataclass
class LidarConfig:
    """Configuration for a simulated LiDAR unit.

    Parameters
    ----------
    mode : {"spinning", "solid"}
        ``spinning`` sweeps azimuth across a full circle; ``solid`` scans a
        rectangular azimuth × elevation FOV (automotive forward unit).
    h_fov_deg : float
        Horizontal FOV span in degrees. For ``spinning`` this is fixed at
        360°; for ``solid`` it is the azimuth half-width applied on both
        sides (i.e. total span = ``2 * h_fov_deg``).
    v_fov_deg : float
        Vertical FOV *half*-span in degrees (total vertical span is
        ``2 * v_fov_deg``), with beams spread from -span..+span about the
        mount optical axis. For ``spinning`` this defines the per-beam
        elevation spread.
    beams : int
        Number of vertical beams (rows).
    columns : int
        Azimuth samples per scan (columns).
    range_min : float
        Minimum reported range (m); nearer hits are dropped.
    range_max : float
        Maximum reported range (m); farther returns are dropped.
    range_noise_sigma_m : float
        Std dev of additive Gaussian range noise (m).
    dropout_prob : float
        Probability any given valid ray is dropped (reflectivity / return loss).
    """
    mode: str = "spinning"
    h_fov_deg: float = 180.0          # half FOV for solid; ignored for spinning
    v_fov_deg: float = 15.0           # half vertical span for both
    beams: int = 16
    columns: int = 1800
    range_min: float = 0.5
    range_max: float = 100.0
    range_noise_sigma_m: float = 0.02
    dropout_prob: float = 0.05

    def __post_init__(self):
        if self.mode not in ("spinning", "solid"):
            raise ValueError(f"mode must be 'spinning' or 'solid', got {self.mode!r}")
        if self.beams < 1 or self.columns < 1:
            raise ValueError("beams and columns must be >= 1")


# --------------------------------------------------------------------------
# Scan frame
# --------------------------------------------------------------------------
@dataclass
class LidarFrame:
    """One LiDAR scan.

    Points are in the **sensor/body frame** (+x forward, +y left, +z up), i.e.
    the raw measurement the downstream module would consume.

    Parameters
    ----------
    t : float
        Host time of the scan.
    points : np.ndarray
        ``(N, 3)`` valid points in sensor frame (m).
    ranges : np.ndarray
        ``(N,)`` measured range (m) *after* noise is added.
    intensities : np.ndarray
        ``(N,)`` reflectivity of the hit object in [0, 1].
    object_ids : np.ndarray
        ``(N,)`` id of the object each point hit (-1 = ground, else object id).
    azimuths : np.ndarray
        ``(N,)`` azimuth of each point (rad).
    elevations : np.ndarray
        ``(N,)`` elevation of each point (rad).
    """
    t: float
    points: np.ndarray
    ranges: np.ndarray
    intensities: np.ndarray
    object_ids: np.ndarray
    azimuths: np.ndarray
    elevations: np.ndarray

    def __len__(self) -> int:
        return int(self.points.shape[0])


# --------------------------------------------------------------------------
# LiDAR sensor
# --------------------------------------------------------------------------
class LidarSensor:
    """Simulated LiDAR sensor mounted rigidly on the vehicle body.

    Parameters
    ----------
    config : LidarConfig
        Scan geometry and noise parameters.
    mount_t_body : np.ndarray, optional
        Translation of the sensor in body frame (``[3]``), default origin.
    """

    def __init__(self, config: LidarConfig,
                 mount_t_body: Optional[np.ndarray] = None):
        self.config = config
        self.mount_t_body = (
            np.asarray(mount_t_body, dtype=float)
            if mount_t_body is not None else np.zeros(3)
        )
        self._az, self._el = self._build_ray_grid(config)

    # -- ray grid ----------------------------------------------------------
    def _build_ray_grid(self, cfg: LidarConfig):
        el_span = np.deg2rad(cfg.v_fov_deg)
        if cfg.beams > 1:
            elevations = np.linspace(-el_span, el_span, cfg.beams)
        else:
            elevations = np.array([0.0])
        if cfg.mode == "spinning":
            azimuths = np.linspace(-np.pi, np.pi, cfg.columns, endpoint=False)
        else:  # solid
            az_span = np.deg2rad(cfg.h_fov_deg)
            if cfg.columns > 1:
                azimuths = np.linspace(-az_span, az_span, cfg.columns)
            else:
                azimuths = np.array([0.0])
        return azimuths, elevations

    # -- per-direction unit vector in sensor/body frame ---------------------
    @staticmethod
    def _dir_body(az, el):
        return np.array([
            float(np.cos(el) * np.cos(az)),
            float(np.cos(el) * np.sin(az)),
            float(np.sin(el)),
        ])

    # -- one full scan -------------------------------------------------------
    def scan(self, point, world: List[Object], t: float = 0.0,
             rng: Optional[np.random.Generator] = None) -> LidarFrame:
        """Cast a full scan from the given host pose.

        Parameters
        ----------
        point : TrajPoint
            Ground-truth host state (position + ``att`` quaternion).
        world : list of Object
            Static + dynamic world primitives.
        t : float
            Host time (passed to dynamic objects).
        rng : np.random.Generator, optional
            RNG for repeatable noise. Defaults to a fresh default_rng.

        Returns
        -------
        LidarFrame
            Valid points after clamping, noise and dropout.
        """
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng()

        pos_w = np.asarray(point.pos, dtype=float)
        R_bw = _quat_to_rotmat(point.att)          # world -> body
        R_wb = R_bw.T                              # body -> world
        sensor_origin_w = pos_w + R_wb @ self.mount_t_body

        # Pre-allocate full-grid outputs (dropouts leave holes).
        N = cfg.beams * cfg.columns
        ranges = np.full(N, np.inf)
        azs = np.zeros(N)
        els = np.zeros(N)
        obj_ids = np.full(N, -2, dtype=int)
        intensity = np.zeros(N)

        idx = 0
        for el in self._el:
            for az in self._az:
                d_body = self._dir_body(az, el)
                d_w = R_wb @ d_body
                azs[idx] = az
                els[idx] = el
                best = np.inf
                best_obj = -2
                refl = 0.0
                for obj in world:
                    hit = obj.intersect(sensor_origin_w, d_w, t)
                    if hit is not None and hit < best:
                        best = hit
                        best_obj = obj.object_id
                        refl = getattr(obj, "reflectivity", 0.5)
                ranges[idx] = best
                obj_ids[idx] = best_obj
                intensity[idx] = refl
                idx += 1

        # Clamp to [range_min, range_max].
        valid = (ranges >= cfg.range_min) & (ranges <= cfg.range_max)
        ranges = ranges.astype(float)

        # Range noise (Gaussian) on valid returns.
        noise = rng.normal(0.0, cfg.range_noise_sigma_m, size=N)
        ranges = ranges + noise

        # Dropout: randomly mask some valid rays.
        drop = rng.random(N) < cfg.dropout_prob
        valid = valid & ~drop

        # Recover valid points.
        idxs = np.nonzero(valid)[0]
        r = ranges[idxs]
        az = azs[idxs]
        el = els[idxs]

        # Build points in sensor frame: p = r * dir(az, el).
        pts = np.stack([
            r * np.cos(el) * np.cos(az),
            r * np.cos(el) * np.sin(az),
            r * np.sin(el),
        ], axis=-1)

        return LidarFrame(
            t=t,
            points=pts,
            ranges=r,
            intensities=intensity[idxs],
            object_ids=obj_ids[idxs],
            azimuths=az,
            elevations=el,
        )
