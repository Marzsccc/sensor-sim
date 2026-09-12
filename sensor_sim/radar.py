"""Automotive radar detection simulation (v0.17.0).

A forward-looking 77 GHz radar that measures analytic world objects -- one
detection per object -- in spherical coordinates: range (to the *near surface*
via the same analytic raycast as :mod:`sensor_sim.lidar`), azimuth/elevation
(to the object centre) and the radial Doppler range-rate.  This is the sensor
class that pairs with the existing IMU/GNSS/wheel/outage/LiDAR/camera modules
and is the measurement layer behind ACC / FCW / AEB target lists.

Design goals (mirror :mod:`sensor_sim.lidar` / :mod:`sensor_sim.camera`):

- **One rigid mount, one frame convention.**  The radar sits at
  ``mount_t_body`` on the vehicle body; world objects are transformed into the
  sensor frame with the same ``quat_to_rotmat`` plumbing as ``LidarSensor``
  and ``MarkerProjector`` (+x forward, +y left, +z up; world z-up).
- **Analytic detections** (no rasterization): every world ``Object`` inside
  the radar FOV and range gate is a candidate.  Range is measured to the near
  surface along the centre bearing, azimuth/elevation to the centre -- so an
  extended lead vehicle reads a few metres *closer* than its centre, exactly
  what a real radar reports for a resolved target.
- **Radar equation & detection probability.**  Per-object RCS drives
  ``SNR = snr_ref * (rcs/rcs_ref) * (range_ref/R)**4``; detection is a
  Bernoulli draw with ``P_d = SNR / (SNR + detect_threshold)``, so far / dim
  targets drop out -- the radar analogue of LiDAR dropout probability.
- **Doppler range-rate, closing positive.**  ``v_r = (v_sensor - v_obj) . u``
  along the centre bearing (u = unit vector sensor -> object).  Positive means
  closing, which is the ACC/FCW convention (a receding target reads negative).
  Kinematics are translation-only: rigid-body rotation of the mount and target
  micro-Doppler (wheel rotation, pedestrian limb motion) are out of scope.

Radar-specific measurement effects captured here:

- spherical measurement space (range / azimuth / elevation) with per-axis
  Gaussian noise,
- finite FOV (azimuth & elevation half-spans) plus range gate
  ``[range_min, range_max]``,
- far-range / small-RCS detection dropout via the radar equation,
- range-rate (Doppler) noise -- the velocity resolution cell.

Documented limitations (out of scope for this layer):

- ground / static clutter (guardrail, road) is *not* generated: this is the
  object-target detection layer, matching how the ADAS stack consumes targets;
- one detection per object (no glint / multipath ghosting, no range-Doppler
  cell quantization);
- no elevation rate / no micro-Doppler.

Coordinate conventions (kept identical to the rest of sensor-sim): world is
**z-up**; body frame **+x forward, +y left, +z up**; the radar optical axis is
+x (forward-looking), azimuth measured from +x toward +y, elevation from the
x-y plane toward +z.  ``R_bw = quat_to_rotmat(att)`` (world <- body);
body->world is ``R_bw.T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lidar import GroundPlane, Object  # reuse world primitives + raycast
from .utils import quat_to_rotmat


# --------------------------------------------------------------------------
# Radar configuration
# --------------------------------------------------------------------------
@dataclass
class RadarConfig:
    """Configuration for a forward-looking automotive radar unit.

    Parameters
    ----------
    h_fov_deg : float
        Azimuth half-span in degrees (total FOV = ``2 * h_fov_deg``).
        A typical long-range radar sweeps roughly +-60 deg.
    v_fov_deg : float
        Elevation half-span in degrees (total = ``2 * v_fov_deg``), ~+-10 deg.
    range_min : float
        Minimum reported range (m); nearer objects are dropped.
    range_max : float
        Maximum reported range (m); farther objects are dropped.
    range_noise_sigma_m : float
        Std dev of Gaussian range noise (m) -- modern LR radar ~0.1 m.
    angle_noise_sigma_deg : float
        Std dev of Gaussian azimuth/elevation noise (deg), ~0.5 deg.
    range_rate_noise_sigma_mps : float
        Std dev of Doppler range-rate noise (m/s), ~0.15 m/s.
    rcs_ref_sqm : float
        Reference radar cross section (m^2) at which ``snr_ref_db`` is
        quoted (a passenger car is roughly 10 m^2).
    range_ref_m : float
        Reference range (m) for the radar equation.
    snr_ref_db : float
        SNR (dB) received from an ``rcs_ref_sqm`` target at ``range_ref_m``.
    detect_threshold_linear : float
        Detection threshold in *linear* SNR.  Per-object detection
        probability is ``P_d = SNR/(SNR + detect_threshold_linear)``; at the
        reference point ``SNR >> threshold`` so P_d ~ 1, and P_d falls toward
        0 as the target recedes (SNR ~ R^-4).
    """

    h_fov_deg: float = 60.0
    v_fov_deg: float = 10.0
    range_min: float = 1.0
    range_max: float = 200.0
    range_noise_sigma_m: float = 0.10
    angle_noise_sigma_deg: float = 0.5
    range_rate_noise_sigma_mps: float = 0.15
    rcs_ref_sqm: float = 10.0
    range_ref_m: float = 50.0
    snr_ref_db: float = 40.0
    detect_threshold_linear: float = 100.0

    def __post_init__(self):
        if self.h_fov_deg <= 0.0 or self.v_fov_deg <= 0.0:
            raise ValueError("h_fov_deg and v_fov_deg must be > 0")
        if self.range_max <= self.range_min:
            raise ValueError("range_max must be > range_min")
        for name in (
            "range_noise_sigma_m",
            "angle_noise_sigma_deg",
            "range_rate_noise_sigma_mps",
            "rcs_ref_sqm",
            "range_ref_m",
            "detect_threshold_linear",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.range_ref_m <= 0.0:
            raise ValueError("range_ref_m must be > 0")
        if self.rcs_ref_sqm <= 0.0:
            raise ValueError("rcs_ref_sqm must be > 0")

    # -- derived -----------------------------------------------------------
    @property
    def _snr_ref_linear(self) -> float:
        return 10.0 ** (self.snr_ref_db / 10.0)

    def detection_probability(self, r_m: float, rcs_sqm: float) -> float:
        """P_det for one object at slant range ``r_m`` with RCS ``rcs_sqm``.

        Radar equation: SNR scales as ``rcs / R**4`` relative to the
        reference point; detection probability is the saturating curve
        ``SNR / (SNR + detect_threshold_linear)``.
        """
        snr = (
            self._snr_ref_linear
            * (rcs_sqm / self.rcs_ref_sqm)
            * (self.range_ref_m / r_m) ** 4
        )
        return float(snr / (snr + self.detect_threshold_linear))


# --------------------------------------------------------------------------
# Scan frame
# --------------------------------------------------------------------------
@dataclass
class RadarFrame:
    """One radar scan: a sparse list of object detections.

    All per-detection fields are parallel arrays of length ``N`` (the number
    of detections this scan; N = 0 when nothing passed the gate).  ``points``
    are in the **sensor/body frame** (+x forward, +y left, +z up),
    reconstructed from the *noisy* polar measurement -- i.e. the raw target
    list a fusion / tracking layer would consume.

    Parameters
    ----------
    t : float
        Host time of the scan.
    ranges : np.ndarray
        ``(N,)`` measured range (m).
    range_rates : np.ndarray
        ``(N,)`` measured radial range-rate (m/s); **positive = closing**.
    azimuths : np.ndarray
        ``(N,)`` measured azimuth (rad, +x forward, +y left).
    elevations : np.ndarray
        ``(N,)`` measured elevation (rad).
    points : np.ndarray
        ``(N, 3)`` sensor-frame position from the noisy polar measurement.
    snr_db : np.ndarray
        ``(N,)`` detection SNR (dB).
    object_ids : np.ndarray
        ``(N,)`` id of the object that produced each detection.
    range_true : np.ndarray
        ``(N,)`` truth range (m) before noise (surface range).
    range_rate_true : np.ndarray
        ``(N,)`` truth closing speed (m/s) before noise.
    """

    t: float
    ranges: np.ndarray
    range_rates: np.ndarray
    azimuths: np.ndarray
    elevations: np.ndarray
    points: np.ndarray
    snr_db: np.ndarray
    object_ids: np.ndarray
    range_true: np.ndarray
    range_rate_true: np.ndarray


# --------------------------------------------------------------------------
# Radar sensor
# --------------------------------------------------------------------------
class RadarSensor:
    """Simulated automotive radar mounted rigidly on the vehicle body.

    Parameters
    ----------
    config : RadarConfig
        FOV, range gate and noise / detection parameters.
    mount_t_body : np.ndarray, optional
        Translation of the radar in body frame (``[3]``), default origin.
    """

    def __init__(self, config: RadarConfig, mount_t_body: np.ndarray | None = None):
        self.config = config
        self.mount_t_body = (
            np.asarray(mount_t_body, dtype=float)
            if mount_t_body is not None
            else np.zeros(3)
        )

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _rcs_of(obj: Object, fallback_sqm: float) -> float:
        """RCS of a world primitive.

        An explicit ``rcs_sqm`` attribute wins (users modelling a van vs a
        bicycle); otherwise map the LiDAR ``reflectivity`` in [0, 1] onto the
        reference RCS so a full-reflector object equals ``rcs_ref_sqm``.
        """
        rcs = getattr(obj, "rcs_sqm", None)
        if rcs is not None:
            return float(rcs)
        return float(fallback_sqm * getattr(obj, "reflectivity", 0.5))

    # -- one full scan -------------------------------------------------------
    def scan(
        self,
        point,
        world: list[Object],
        t: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> RadarFrame:
        """Measure every object in the radar's FOV and range gate.

        Parameters
        ----------
        point : TrajPoint-like
            Ground-truth host state: ``pos`` (world [3]), ``att`` (wxyz
            quaternion) and optionally ``vel`` (world [3]) for the Doppler
            term.  If ``vel`` is absent a stationary host is assumed.
        world : list of Object
            Static + dynamic world primitives (``GroundPlane`` is skipped:
            this layer reports object targets, not clutter).
        t : float
            Host time (passed to dynamic objects).
        rng : np.random.Generator, optional
            RNG for repeatable noise / dropout.  Defaults to a fresh one.

        Returns
        -------
        RadarFrame
            Sparse detection list (possibly empty).
        """
        cfg = self.config
        if rng is None:
            rng = np.random.default_rng()

        pos_w = np.asarray(point.pos, dtype=float)
        _v = getattr(point, "vel", None)
        vel_w = np.zeros(3) if _v is None else np.asarray(_v, dtype=float)
        R_bw = quat_to_rotmat(point.att)  # world -> body
        R_wb = R_bw.T  # body -> world
        origin_w = pos_w + R_wb @ self.mount_t_body
        # Rigid mount, translation-only: v_sensor = v_host.  The rotational
        # term (omega x mount offset) is a sub-m/s correction at car speeds
        # and is documented as out of scope.

        hf = np.deg2rad(cfg.h_fov_deg)
        vf = np.deg2rad(cfg.v_fov_deg)

        rows = []
        for obj in world:
            if isinstance(obj, GroundPlane):
                continue  # no ground clutter here
            center_w = np.asarray(obj.world_pos(t), dtype=float)
            delta = center_w - origin_w
            r_center = float(np.linalg.norm(delta))
            if r_center < 1e-9:
                continue
            u_c = delta / r_center  # centre bearing (sensor->obj)

            # Bearing in sensor/body frame -> azimuth / elevation gate.
            rel_b = R_bw @ delta
            az = float(np.arctan2(rel_b[1], rel_b[0]))
            el = float(np.arcsin(np.clip(rel_b[2] / r_center, -1.0, 1.0)))
            if abs(az) > hf or abs(el) > vf:
                continue

            # Range to the NEAR SURFACE along the centre bearing (extended
            # targets read closer than their centre -- like a real radar).
            hit = obj.intersect(origin_w, u_c, t)
            r_true = float(hit) if hit is not None else r_center
            if r_true < cfg.range_min or r_true > cfg.range_max:
                continue

            # Radar equation -> detection probability -> Bernoulli draw.
            rcs = self._rcs_of(obj, cfg.rcs_ref_sqm)
            snr_lin = (
                cfg._snr_ref_linear
                * (rcs / cfg.rcs_ref_sqm)
                * (cfg.range_ref_m / r_true) ** 4
            )
            p_d = snr_lin / (snr_lin + cfg.detect_threshold_linear)
            if rng.random() >= p_d:
                continue

            # Truth Doppler (closing positive) from translation only.
            obj_v = np.asarray(getattr(obj, "velocity", np.zeros(3)), dtype=float)
            vr_true = float(np.dot(vel_w - obj_v, u_c))

            # Measurement noise (range / angles / range-rate).
            r_m = r_true + rng.normal(0.0, cfg.range_noise_sigma_m)
            az_m = az + np.deg2rad(cfg.angle_noise_sigma_deg) * rng.standard_normal()
            el_m = el + np.deg2rad(cfg.angle_noise_sigma_deg) * rng.standard_normal()
            vr_m = vr_true + rng.normal(0.0, cfg.range_rate_noise_sigma_mps)

            rows.append(
                (
                    r_m,
                    vr_m,
                    az_m,
                    el_m,
                    10.0 * np.log10(snr_lin),
                    obj.object_id,
                    r_true,
                    vr_true,
                )
            )

        if not rows:
            z = np.zeros(0)
            return RadarFrame(
                t=t,
                ranges=z,
                range_rates=z,
                azimuths=z,
                elevations=z,
                points=np.zeros((0, 3)),
                snr_db=z,
                object_ids=z.astype(int),
                range_true=z,
                range_rate_true=z,
            )

        arr = np.asarray(rows, dtype=float)
        r_m, vr_m, az_m, el_m = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        snr_db = arr[:, 4]
        obj_ids = arr[:, 5].astype(int)
        r_true = arr[:, 6]
        vr_true = arr[:, 7]

        # Sensor-frame position from the NOISY polar measurement.
        pts = np.stack(
            [
                r_m * np.cos(el_m) * np.cos(az_m),
                r_m * np.cos(el_m) * np.sin(az_m),
                r_m * np.sin(el_m),
            ],
            axis=-1,
        )

        return RadarFrame(
            t=t,
            ranges=r_m,
            range_rates=vr_m,
            azimuths=az_m,
            elevations=el_m,
            points=pts,
            snr_db=snr_db,
            object_ids=obj_ids,
            range_true=r_true,
            range_rate_true=vr_true,
        )
