"""Constant-velocity EKF object tracking over radar detections (v0.18.0).

Single-frame automotive radar is *structurally blind to cross-range motion*:
a crossing pedestrian produces a Doppler range-rate near zero until it is
almost abeam, because the radar only measures the radial component of the
relative velocity (the v0.17.0 ``RadarSensor`` demo documents exactly this).
The missing piece between "detections" and an ACC / FCW / AEB / RCTA target
list is a **tracker**: fuse a temporal sequence of sparse detections into a
smooth kinematic state (position + full 2-D velocity) so that lateral motion
-- invisible in any single scan -- becomes observable from the azimuth
history.

This module implements the classic production pattern:

- **Constant-velocity model in the world frame.**  The track state is
  ``x = [px, py, vx, vy]`` -- the horizontal-plane position and *absolute*
  velocity of the radar reflection point.  Because the model is in the world
  frame, the predict step is exactly linear for any constant-velocity target
  regardless of what the host is doing; the host motion only enters through
  the *measurement* equation (each scan is transformed through the host pose
  at that scan's time, the same ``+x forward / +y left / +z up`` convention as
  the rest of sensor-sim).  No per-scan host-motion compensation bookkeeping.
- **EKF in spherical measurement space.**  The radar measures
  ``z = [range, azimuth, range-rate]`` (closing positive).  The measurement
  model ``h(x)`` projects the state through the current host pose and returns
  exactly the three quantities ``RadarFrame`` reports; the Jacobian is
  analytic (validated numerically in the tests).
- **Nearest-neighbour association with a Mahalanobis gate.**  Every track
  predicts its measurement with covariance ``S = HPH' + R``; a detection is a
  candidate for a track when its innovation squared ``d2 < gate_chi2``
  (default the chi-square 3-dof 99% threshold 11.345).  Candidate pairs are
  assigned greedily by smallest ``d2``.
- **Two-class track management (confirmed vs tentative).**  A track must be
  updated on ``confirm_min`` scans (default 3, ~0.3 s at 10 Hz) before it is
  promoted from tentative to confirmed.  Confirmed tracks claim detections
  first; tentative tracks only take what no confirmed track wants, and a
  tentative track that misses a single scan is dropped immediately.  This
  stops the textbook ``newborn death spiral`` of pure greedy NN: a freshly
  birthed track sits on its birth detection with a huge covariance, its
  innovation is ~0, and it would otherwise starve well-converged tracks after
  a single statistical gate fluke.
- **Lifecycle.**  Unassociated detections birth new tracks (position from the
  converted measurement, radial velocity from the Doppler term with a
  deliberately *large* cross-range velocity prior -- the tracker admits it
  cannot know lateral speed at birth); confirmed tracks that miss
  ``coast_max`` consecutive scans are deleted.  Re-acquired targets start a
  fresh track id.

Design notes / documented limitations (kept honest, mirroring the suite):

- **Planar (ground-plane) tracker.**  State is 2-D; elevation is ignored.
  Results are exact when targets sit in the sensor elevation plane (tests and
  the demo use z = 0 targets and a z = 0 mount) and are an excellent
  approximation at car-scale elevations, which is the standard ACC assumption
  -- the ADAS stack plans in the ground plane.
- **One detection per object per scan** (inherited from ``RadarSensor``): no
  glint / multipath ghosting, so no track-to-track fusion of multiple
  reflections is modelled.
- **Point-target reflection.**  For an extended ``Box`` the radar reports the
  range of the *near surface*; the tracker therefore tracks the reflection
  point on that surface, which moves at the object's true velocity -- range
  estimates match ``range_true`` (the surface range), not the box centre.
- **Nearest-neighbour association, not JPDA / MHT.**  When two targets cross
  in measurement space a greedy NN tracker can momentarily swap identities;
  this is a documented, real production behaviour (mitigated in practice by
  motion models and track scores, both out of scope here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .radar import RadarFrame  # type hint only (no circular import: radar does not import tracking)
from .trajectory import _quat_to_rotmat

# Default radar measurement noise (matches RadarConfig typical values so a
# tracker configured with defaults can consume a default RadarSensor).
DEF_RANGE_SIGMA_M = 0.10
DEF_ANGLE_SIGMA_DEG = 0.5
DEF_RATE_SIGMA_MPS = 0.15

# chi-square, 3 dof, 99% -- the default association gate.
DEF_GATE_CHI2 = 11.345


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class TrackConfig:
    """Tracker tuning: motion model, measurement noise, gate & lifecycle.

    Parameters
    ----------
    process_noise_accel_mps2 : float
        Std dev of the continuous white-noise acceleration driving the
        constant-velocity model (m/s^2, per axis).  Governs how freely tracks
        may deviate from straight-line motion between scans (a crossing
        pedestrian needs a larger value than a highway lead vehicle).
    range_noise_sigma_m : float
        Std dev of the range measurement noise used to build ``R`` (m).
        Keep in sync with the ``RadarConfig`` of the radar under test.
    angle_noise_sigma_deg : float
        Std dev of the azimuth measurement noise for ``R`` (deg).
    range_rate_noise_sigma_mps : float
        Std dev of the Doppler range-rate measurement noise for ``R`` (m/s).
    gate_chi2 : float
        Mahalanobis association gate (innovation squared).  Default is the
        chi-square 3-dof 99% quantile: a detection whose innovation exceeds
        this is treated as a *different* object.
    init_radial_vel_sigma_mps : float
        Std dev of the birth velocity prior along the measured bearing (m/s).
        The Doppler term makes the radial component partially observable at
        birth, so this is only a little above the range-rate noise.
    init_lateral_vel_sigma_mps : float
        Std dev of the birth velocity prior across the bearing (m/s).  Large
        on purpose: a single scan cannot see cross-range motion, so new
        tracks must admit total ignorance of lateral speed (a person walks at
        ~2 m/s, a crossing car can do 15).
    coast_max : int
        Consecutive scans a *confirmed* track may miss a detection before it
        is deleted.  At 10 Hz with the default 8 this is 0.8 s of coasting.
    confirm_min : int
        Number of scans on which a new (tentative) track must be updated
        before it is promoted to *confirmed* (birth counts as the first).
        Tentative tracks may only claim detections that no confirmed track
        wants, and are deleted the first time they miss a scan -- both rules
        stop the classic ``newborn-with-wide-prior steals the detection``
        death spiral (a freshly birthed track sits on its birth detection
        with a huge covariance, so its innovation is ~0 and pure
        nearest-neighbour assignment lets it starve well-converged tracks
        after a single statistical gate fluke).
    """

    process_noise_accel_mps2: float = 1.0
    range_noise_sigma_m: float = DEF_RANGE_SIGMA_M
    angle_noise_sigma_deg: float = DEF_ANGLE_SIGMA_DEG
    range_rate_noise_sigma_mps: float = DEF_RATE_SIGMA_MPS
    gate_chi2: float = DEF_GATE_CHI2
    init_radial_vel_sigma_mps: float = 2.0
    init_lateral_vel_sigma_mps: float = 10.0
    coast_max: int = 8
    confirm_min: int = 3

    def __post_init__(self):
        if self.process_noise_accel_mps2 < 0.0:
            raise ValueError("process_noise_accel_mps2 must be >= 0")
        for name in (
            "range_noise_sigma_m",
            "angle_noise_sigma_deg",
            "range_rate_noise_sigma_mps",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.range_noise_sigma_m <= 0.0 and (
            self.angle_noise_sigma_deg <= 0.0 and self.range_rate_noise_sigma_mps <= 0.0
        ):
            # Allow any *one* channel to be ideal, but R must stay positive
            # definite overall (all three zero => singular S).  Tests that
            # want a "quiet" tracker pass ~1e-6 on every channel instead.
            raise ValueError("measurement noise may not be zero on every channel")

    # -- derived ------------------------------------------------------------
    @property
    def r_matrix(self) -> np.ndarray:
        """Measurement covariance ``R`` (3x3) in [m^2, rad^2, (m/s)^2]."""
        return np.diag(
            [
                self.range_noise_sigma_m**2,
                np.deg2rad(self.angle_noise_sigma_deg) ** 2,
                self.range_rate_noise_sigma_mps**2,
            ]
        )


# --------------------------------------------------------------------------
# Track state
# --------------------------------------------------------------------------
@dataclass
class Track:
    """One tracked object: state, covariance and lifecycle counters.

    Parameters
    ----------
    track_id : int
        Monotonically increasing id; never reused within a tracker instance.
    x : np.ndarray
        ``(4,)`` state ``[px, py, vx, vy]`` -- horizontal position (m) and
        absolute velocity (m/s) of the reflection point, world frame.
    P : np.ndarray
        ``(4, 4)`` state covariance.
    born_t : float
        Host time of the birth scan.
    """

    track_id: int
    x: np.ndarray
    P: np.ndarray
    born_t: float
    last_t: float | None = None
    age: int = 0  # scans processed since birth (updates + coasts)
    missed: int = 0  # consecutive scans without an assigned detection
    hits: int = 0  # scans on which a detection was assigned (birth counts)
    confirmed: bool = False  # hits >= confirm_min (promoted by the tracker)

    # -- convenience accessors ---------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        """Estimated horizontal position ``[px, py]`` (m, world frame)."""
        return self.x[:2].copy()

    @property
    def vel(self) -> np.ndarray:
        """Estimated absolute velocity ``[vx, vy]`` (m/s, world frame)."""
        return self.x[2:].copy()

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))

    def velocity_relative_to(self, vel_w: np.ndarray) -> np.ndarray:
        """Velocity of the target relative to a host moving at ``vel_w``."""
        return self.x[2:] - np.asarray(vel_w, dtype=float)[:2]


# --------------------------------------------------------------------------
# Tracking frame (one processed scan)
# --------------------------------------------------------------------------
@dataclass
class TrackingFrame:
    """Result of feeding one radar scan to the tracker.

    Parameters
    ----------
    t : float
        Host time of the scan.
    n_detections : int
        Detections the radar reported this scan.
    tracks : tuple of Track
        Surviving tracks after association / update / birth / deletion,
        ordered by track id.  **The Track objects are live**: the tracker
        owns them and mutates them in place on every subsequent scan, so
        frames from earlier scans keep pointing at the same (now updated)
        objects.  Snapshot ``x``/``P``/counters inside your per-scan loop if
        you need a history.
    """

    t: float
    n_detections: int
    tracks: tuple[Track, ...] = ()

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)


# --------------------------------------------------------------------------
# EKF primitives
# --------------------------------------------------------------------------
def _predict(track: Track, dt: float, cfg: TrackConfig) -> None:
    """CV predict in place.  ``dt <= 0`` (first update) is a no-op."""
    if dt <= 0.0:
        return
    x, P = track.x, track.P
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = cfg.process_noise_accel_mps2**2
    d3, d2, d1 = dt**3 / 3.0, dt**2 / 2.0, dt
    Q = q * np.array(
        [
            [d3, 0.0, d2, 0.0],
            [0.0, d3, 0.0, d2],
            [d2, 0.0, d1, 0.0],
            [0.0, d2, 0.0, d1],
        ]
    )
    track.x = F @ x
    track.P = F @ P @ F.T + Q


def _bearing_unit(xy: np.ndarray) -> np.ndarray:
    """Horizontal unit vector from sensor origin to a state position."""
    r = float(np.hypot(xy[0], xy[1]))
    if r < 1e-12:
        return np.array([1.0, 0.0])
    return xy / r


def _measurement_jacobian(
    x: np.ndarray,
    pos_w: np.ndarray,
    R_bw: np.ndarray,
    mount_t_body: np.ndarray,
    vel_w: np.ndarray,
):
    """Analytic 3x4 Jacobian of ``h`` at state ``x`` (world frame).

    ``h`` maps the state to ``[range, azimuth, range-rate]`` as seen by a
    radar at ``pos_w`` with body->world attitude ``R_bw`` (i.e. world->body
    ``R_bw``) and body-frame mount ``mount_t_body``, on a host moving at
    ``vel_w``.  Returns ``(zhat, H)``.
    """
    p_w = x[:2] - pos_w[:2]  # horizontal offset, world
    p_b = R_bw[:2, :2] @ p_w - mount_t_body[:2]  # sensor frame (+x fwd, +y left)
    r = float(np.hypot(p_b[0], p_b[1]))
    if r < 1e-12:
        r = 1e-12
    # azimuth in the sensor frame (atan2 of the sensor-frame offset).
    az = float(np.arctan2(p_b[1], p_b[0]))
    # world unit bearing sensor -> target (used by the Doppler term).
    origin_w = pos_w + R_bw.T @ mount_t_body
    u_w = (x[:2] - origin_w[:2]) / r
    v_rel = vel_w[:2] - x[2:]  # host - target (closing positive)
    vr = float(np.dot(v_rel, u_w))

    zhat = np.array([r, az, vr])

    # -- Jacobian -----------------------------------------------------------
    # dr/dx_pos = u_w' (horizontal plane; r is the horizontal range).
    H = np.zeros((3, 4))
    H[0, :2] = u_w

    # daz/dx_pos: azimuth depends on the sensor-frame offset.
    #   az = atan2(y_b, x_b),  daz = [-y_b, x_b] / (x_b^2 + y_b^2)
    #   x_b = R_bw_xy (x_pos - pos_w) - mount_xy
    daz_dp_b = np.array([-p_b[1], p_b[0]]) / (r * r)
    H[1, :2] = daz_dp_b @ R_bw[:2, :2]

    # dvr/dx_pos = v_rel' * d(u_w)/dx_pos,  d(u_w)/dx_pos = (I - u u') / r
    dudx = (np.eye(2) - np.outer(u_w, u_w)) / r
    H[2, :2] = v_rel @ dudx
    # dvr/dv_obj = -u_w'
    H[2, 2:] = -u_w

    return zhat, H


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------
class RadarTracker:
    """Constant-velocity EKF tracker over ``RadarFrame`` detections.

    Parameters
    ----------
    config : TrackConfig, optional
        Model / gate / lifecycle tuning (defaults are sane for a default
        ``RadarSensor`` at 10 Hz).
    mount_t_body : np.ndarray, optional
        Radar mount translation in the body frame ``[3]`` -- must match the
        ``RadarSensor`` whose frames are being processed (default origin).
    """

    def __init__(
        self,
        config: TrackConfig | None = None,
        mount_t_body: np.ndarray | None = None,
    ):
        self.config = config if config is not None else TrackConfig()
        self.mount_t_body = (
            np.asarray(mount_t_body, dtype=float)
            if mount_t_body is not None
            else np.zeros(3)
        )
        self.tracks: list[Track] = []
        self._next_id = 1

    # -- lifecycle ----------------------------------------------------------
    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def reset(self) -> None:
        """Drop all tracks and reset the id counter (fresh start)."""
        self.tracks.clear()
        self._next_id = 1

    # -- one scan -----------------------------------------------------------
    def process(
        self,
        frame: RadarFrame,
        pos_w: np.ndarray,
        att: np.ndarray,
        vel_w: np.ndarray | None = None,
    ) -> TrackingFrame:
        """Feed one radar scan into the tracker.

        Parameters
        ----------
        frame : RadarFrame
            Scan from ``RadarSensor.scan``.
        pos_w : np.ndarray
            Host position ``[3]`` (world) at the scan time.
        att : np.ndarray
            Host attitude (wxyz quaternion) at the scan time.
        vel_w : np.ndarray, optional
            Host velocity ``[3]`` (world); a stationary host is assumed when
            omitted.

        Returns
        -------
        TrackingFrame
            Surviving tracks plus scan bookkeeping.
        """
        cfg = self.config
        pos_w = np.asarray(pos_w, dtype=float)
        R_bw = _quat_to_rotmat(np.asarray(att, dtype=float))
        vel_w = (
            np.zeros(3)
            if vel_w is None
            else np.asarray(vel_w, dtype=float)
        )
        origin_w = pos_w + R_bw.T @ self.mount_t_body
        t = float(frame.t)
        R = cfg.r_matrix

        # 1) predict every track to the current scan time.
        for tr in self.tracks:
            dt = 0.0 if tr.last_t is None else t - tr.last_t
            _predict(tr, dt, cfg)

        # 2) per-track predicted measurements & innovation covariances.
        zhats = []
        Ss = []
        Sinvs = []
        for tr in self.tracks:
            zhat, H = _measurement_jacobian(
                tr.x, pos_w, R_bw, self.mount_t_body, vel_w
            )
            S = H @ tr.P @ H.T + R
            zhats.append(zhat)
            Ss.append(S)
            Sinvs.append(np.linalg.inv(S))

        # 3) detection -> track association.  Two passes with greedy NN in
        #    each: *confirmed* tracks claim detections first (a tentative
        #    track's wide birth prior makes its innovation ~0, so without
        #    this ordering a newborn would starve a well-converged track
        #    after a single statistical gate fluke), then tentative tracks
        #    compete for the leftovers, then anything unclaimed is birthed.
        n_det = len(frame.ranges)
        assigned_det: set[int] = set()
        assigned_tr: set[int] = set()

        def _greedy_pass(indices: list[int]) -> None:
            cand: list[tuple[float, int, int]] = []
            for j in indices:
                for k in range(n_det):
                    if k in assigned_det:
                        continue
                    z = np.array(
                        [
                            frame.ranges[k],
                            frame.azimuths[k],
                            frame.range_rates[k],
                        ]
                    )
                    nu = z - zhats[j]
                    d2 = float(nu @ Sinvs[j] @ nu)
                    if d2 < cfg.gate_chi2:
                        cand.append((d2, j, k))
            cand.sort(key=lambda c: c[0])
            for d2, j, k in cand:
                if j in assigned_tr or k in assigned_det:
                    continue
                assigned_tr.add(j)
                assigned_det.add(k)
                self._update_track(self.tracks[j], k, frame, R_bw, pos_w, vel_w, R)

        confirmed_idx = [j for j, tr in enumerate(self.tracks) if tr.confirmed]
        tentative_idx = [j for j, tr in enumerate(self.tracks) if not tr.confirmed]
        _greedy_pass(confirmed_idx)
        _greedy_pass(tentative_idx)

        # 4) coast unassigned tracks; delete the stale ones.  A tentative
        #    track that misses a single scan is dropped immediately (it never
        #    confirmed, so it is indistinguishable from a spurious blip); a
        #    confirmed track may coast up to ``coast_max`` scans.
        survivors: list[Track] = []
        for j, tr in enumerate(self.tracks):
            tr.last_t = t
            tr.age += 1
            if j in assigned_tr:
                survivors.append(tr)
                continue
            tr.missed += 1
            if tr.confirmed and tr.missed <= cfg.coast_max:
                survivors.append(tr)
        self.tracks = survivors

        # 5) birth new tracks for unassigned detections.
        for k in range(n_det):
            if k in assigned_det:
                continue
            self._birth(k, frame, origin_w, R_bw, vel_w, t, cfg)

        self.tracks.sort(key=lambda tr: tr.track_id)
        return TrackingFrame(
            t=t,
            n_detections=n_det,
            tracks=tuple(self.tracks),
        )

    # -- helpers ------------------------------------------------------------
    def _update_track(
        self,
        tr: Track,
        k: int,
        frame: RadarFrame,
        R_bw: np.ndarray,
        pos_w: np.ndarray,
        vel_w: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """EKF update of ``tr`` with detection ``k`` (Joseph form)."""
        z = np.array([frame.ranges[k], frame.azimuths[k], frame.range_rates[k]])
        zhat, H = _measurement_jacobian(
            tr.x, pos_w, R_bw, self.mount_t_body, vel_w
        )
        nu = z - zhat
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ nu
        I_KH = np.eye(4) - K @ H
        tr.P = I_KH @ tr.P @ I_KH.T + K @ R @ K.T
        tr.missed = 0
        tr.hits += 1
        tr.confirmed = tr.hits >= self.config.confirm_min

    def _birth(
        self,
        k: int,
        frame: RadarFrame,
        origin_w: np.ndarray,
        R_bw: np.ndarray,
        vel_w: np.ndarray,
        t: float,
        cfg: TrackConfig,
    ) -> None:
        """Spawn a track from detection ``k``.

        Position comes from the converted measurement
        (``RadarFrame.points``, sensor frame -> world); the velocity prior is
        radial-only (from the Doppler term) with a deliberately large
        cross-range sigma -- a single scan cannot see lateral motion.
        """
        R_wb = R_bw.T
        p_sensor = frame.points[k]
        xy = origin_w[:2] + (R_wb @ p_sensor)[:2]

        u = _bearing_unit(xy - origin_w[:2])  # horizontal bearing, world
        perp = np.array([-u[1], u[0]])
        vr_m = float(frame.range_rates[k])
        # Absolute velocity prior: radial part from Doppler, lateral = 0.
        v_abs = vel_w[:2] - vr_m * u

        # Position covariance: along bearing = range noise, across = angular.
        sig_r = cfg.range_noise_sigma_m
        sig_cross = float(np.hypot(xy[0] - origin_w[0], xy[1] - origin_w[1])) * np.deg2rad(
            cfg.angle_noise_sigma_deg
        )
        P_pos = sig_r**2 * np.outer(u, u) + sig_cross**2 * np.outer(perp, perp)
        # Velocity covariance: tight radial, loose lateral.
        P_vel = (
            cfg.init_radial_vel_sigma_mps**2 * np.outer(u, u)
            + cfg.init_lateral_vel_sigma_mps**2 * np.outer(perp, perp)
        )
        x0 = np.concatenate([xy, v_abs])
        P0 = np.zeros((4, 4))
        P0[:2, :2] = P_pos
        P0[2:, 2:] = P_vel

        tr = Track(
            track_id=self._next_id,
            x=x0,
            P=P0,
            born_t=t,
            last_t=t,
            age=1,
            missed=0,
            hits=1,
            confirmed=self.config.confirm_min <= 1,
        )
        self._next_id += 1
        self.tracks.append(tr)
