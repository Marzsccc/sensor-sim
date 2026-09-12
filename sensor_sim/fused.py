"""Radar + camera fused tracking (v0.19.0).

v0.18.0's ``RadarTracker`` is the industry-standard single-sensor baseline:
a radar sees range / azimuth / Doppler extremely well, but a *single scan is
structurally blind to cross-range motion* -- a crossing pedestrian or a
lane-tangling cut-in vehicle produces near-zero Doppler until it is almost
abeam.  The tracker recovers lateral speed from the azimuth *history*, but it
needs several scans and its lateral-velocity prior starts deliberately wide.

The classic production answer is **sensor fusion**: a camera adds an
azimuthal measurement that is far more precise per scan (the pixel column of
a 1080p frame resolves ~0.05° at 50 px/deg versus ~0.1-0.5° for automotive
radar), and -- critically -- the *azimuth history across frames* carries
information about lateral motion *within one camera update*.  Merging the two
into a single EKF

- **sharpens lateral kinematics** (velocity in particular): the camera's
  azimuthal measurement pulls the lateral velocity observable, which is
  exactly the axis the radar alone must integrate several scans to see;
- **keeps the radar's strengths**: stable range / Doppler, all-weather
  operation, and a measurement that survives when the camera drops the target
  (night / glare / occlusion);
- **gives graceful degradation**: when the camera loses a target the filter
  runs radar-only (the radar's measurement update still fires); when the radar
  misses, a camera-only update keeps the track alive.  One fused state, one
  lifecycle, no mode switches.

This module implements the production pattern:

- **One EKF, one 4-state CV track, two measurement models.**  The fused track
  state is ``x = [px, py, vx, vy]`` (horizontal plane, world frame), exactly
  as in ``RadarTracker``.  The radar update consumes ``[range, azimuth,
  range-rate]`` from ``RadarFrame``; the camera update consumes ``[azimuth]``
  from ``CameraFrame`` (``u`` pixels converted with the camera intrinsics).
  The host-motion bookkeeping is identical to v0.18 (mount offsets carried in
  the measurement model, no per-scan host compensation).
- **Per-sensor Mahalanobis association with a shared confirmation policy.**
  Each update is gated by its own innovation covariance ``S = HPH' + R`` and
  chi-square threshold (3 dof for radar, 1 dof for camera); a track is
  confirmed after ``confirm_min`` *total* hits from any sensor, and a
  confirmed track may coast up to ``coast_max`` *consecutive* scans without
  *any* update (in sensor time, not per-sensor).  Unassociated detections
  birth new tracks -- the largest innovation-squared candidate wins, exactly
  as in v0.18.
- **Sensor dropout orchestrates itself.**  When the camera's ``observe()``
  drops a feature (night / occlusion) or the radar misses a scan, the update
  simply does not happen; there is no explicit mode switch.  During a camera
  dropout the fused track degrades to radar-only behaviour (and vice versa),
  and the covariance honestly reflects both the missing measurement and the
  added process noise.

Design notes / documented limitations (mirroring the suite's honesty):

- **Planar fusion.**  State is 2-D as in ``RadarTracker``; the camera azimuth
  measurement also ignores elevation (matches the flat-ground ACC model).
- **One detection per object per sensor per scan.**  Inherited from both
  sensors; we do not model ghosting / multi-reflection.
- **Camera measurements are closest-object features.**  ``CameraSensor``
  observes world ``FeaturePoints``; we assume each fused target owns at least
  one visible feature and use the geometrically closest feature as its
  measurement.  No appearance-based re-identification (a real perception
  stack would feed an appearance embedding into the association gate).
- **Greedy NN association, not JPDA / MHT** -- the same documented
  limitation as v0.18.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraFrame
from .radar import RadarFrame
from .tracking import (
    DEF_GATE_CHI2,
    DEF_ANGLE_SIGMA_DEG,
    DEF_RANGE_SIGMA_M,
    DEF_RATE_SIGMA_MPS,
    Track,
    TrackConfig,
    _bearing_unit,
    _measurement_jacobian,
    _predict,
)
from .utils import quat_to_rotmat

# chi-square, 1 dof, 95% -- the camera azimuth gate (a 1-D innovation).
DEF_CAM_GATE_CHI2 = 3.841


# --------------------------------------------------------------------------
# Fused configuration
# --------------------------------------------------------------------------
@dataclass
class FusedConfig:
    """Tuning for the fused radar + camera tracker.

    Parameters
    ----------
    track_config : TrackConfig
        Motion model / lifecycle / gate tuning for the shared EKF.  Reused for
        the radar measurement noise (``r_matrix``) exactly as in v0.18 and for
        the birth-prior policy.
    cam_azimuth_sigma_deg : float
        Std dev of the camera azimuth measurement used for ``R`` (deg).
        Automotive camera features are typically 1-5 px; with a 30 px/deg
        lens this is ~0.03-0.17°.  Keep in sync with the ``CameraConfig``
        under test.
    cam_gate_chi2 : float
        Mahalanobis gate for the camera azimuth update (innovation squared).
        Default is the chi-square 1-dof 95% quantile 3.841.
    """

    track_config: TrackConfig = field(default_factory=TrackConfig)
    cam_azimuth_sigma_deg: float = 0.1
    cam_gate_chi2: float = DEF_CAM_GATE_CHI2

    def __post_init__(self) -> None:
        if self.cam_azimuth_sigma_deg <= 0:
            raise ValueError("cam_azimuth_sigma_deg must be > 0")
        if self.cam_gate_chi2 <= 0:
            raise ValueError("cam_gate_chi2 must be > 0")
        # the radar R used by the fused tracker is exactly the radar one.
        self._radar_R = self.track_config.r_matrix
        self._cam_R = np.array(
            [[np.deg2rad(self.cam_azimuth_sigma_deg) ** 2]]
        )

    # -- noise matrices ------------------------------------------------------
    @property
    def radar_r(self) -> np.ndarray:
        """Radar measurement noise ``(3,3)`` (range/azimuth/range-rate)."""
        return self._radar_R

    @property
    def cam_r(self) -> np.ndarray:
        """Camera azimuth measurement noise ``(1,1)``."""
        return self._cam_R


# --------------------------------------------------------------------------
# Fused tracker
# --------------------------------------------------------------------------
class FusedTracker:
    """Single EKF fusing radar detections and camera azimuths onto shared tracks.

    Parameters
    ----------
    config : FusedConfig, optional
        Fusion tuning (defaults are sane for a default ``RadarSensor`` at
        10 Hz plus a ~30 px/deg camera).
    radar_mount_t_body : np.ndarray, optional
        Radar mount translation in the body frame ``[3]`` -- must match the
        ``RadarSensor`` whose frames are processed (default origin).
    cam_mount_t_body : np.ndarray, optional
        Camera mount translation in the body frame ``[3]`` -- must match the
        ``CameraSensor`` whose frames are processed (default origin).
    cam_cx : float, optional
        Camera principal-point column (px).  Needed to convert a pixel column
        ``u`` into an azimuth; if omitted, ``cam_fx`` must be supplied and
        ``cx = width/2`` is assumed (a centered lens).
    cam_fx : float, optional
        Camera focal length (px) -- the columns-per-radian scale.  If omitted
        as well, ``cam_fx`` defaults to ``width`` (a 90°-ish lens over
        ``width`` pixels), so a full ``FusedTracker`` can be built with only
        the two mounts.
    cam_width : int, optional
        Camera image width (px), used for the centered-principal-point
        default and the pixel-vs-azimuth conversion.  Default 1920.
    """

    def __init__(
        self,
        config: FusedConfig | None = None,
        radar_mount_t_body: np.ndarray | None = None,
        cam_mount_t_body: np.ndarray | None = None,
        cam_cx: float | None = None,
        cam_fx: float | None = None,
        cam_width: int = 1920,
    ):
        self.config = config if config is not None else FusedConfig()
        self.radar_mount = np.asarray(
            radar_mount_t_body if radar_mount_t_body is not None else np.zeros(3),
            dtype=float,
        )
        self.cam_mount = np.asarray(
            cam_mount_t_body if cam_mount_t_body is not None else np.zeros(3),
            dtype=float,
        )
        if cam_fx is None:
            cam_fx = float(cam_width)
        self.cam_fx = float(cam_fx)
        self.cam_cx = float(cam_cx) if cam_cx is not None else cam_width / 2.0
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

    # -- helpers ------------------------------------------------------------
    def _pixel_to_azimuth(self, u: float) -> float:
        """Camera pixel column -> body azimuth (rad, +x forward / +y left).

        The camera's ``project`` maps ``(x, y, z) -> (u, v) = (fx*y/x + cx,
        fy*(-z)/x + cy)`` with the SAME +y-left convention as the radar (the
        world frame's lateral axis passes straight through), so a world
        target at lateral offset ``y`` sits at pixel ``u = fx*y/x + cx`` and
        the horizontal bearing is ``atan2(y/x, 1) = atan2(u - cx, fx)`` -- a
        target to the right (``y < 0``) lands at ``u < cx`` and maps back to
        a negative azimuth, matching the body-frame ``+y left`` sign.
        """
        return float(np.arctan2(u - self.cam_cx, self.cam_fx))

    # -- one fusion cycle ----------------------------------------------------
    def process(
        self,
        frame: RadarFrame,
        cam: CameraFrame | None,
        pos_w: np.ndarray,
        att: np.ndarray,
        vel_w: np.ndarray | None = None,
    ) -> "FusedFrame":
        """Feed one radar scan plus an optional camera observation into the
        tracker.

        The radar scan is the primary clock (each call advances sensor time
        by the radar period); the camera observation is optional -- omit or
        pass an empty ``CameraFrame`` when the camera did not produce one at
        this tick and the update is simply skipped (a graceful degradation
        path identical to a real dropout).

        Parameters
        ----------
        frame : RadarFrame
            Scan from ``RadarSensor.scan`` (the time clock).
        cam : CameraFrame, optional
            Feature observation from ``CameraSensor.observe`` at the *same*
            host time.  ``None`` (or an empty frame) skips the camera update.
        pos_w : np.ndarray
            Host position ``[3]`` (world) at the scan time.
        att : np.ndarray
            Host attitude (wxyz quaternion) at the scan time.
        vel_w : np.ndarray, optional
            Host velocity ``[3]`` (world); a stationary host is assumed when
            omitted.

        Returns
        -------
        FusedFrame
            Surviving tracks plus per-sensor bookkeeping.
        """
        cfg = self.config
        pos_w = np.asarray(pos_w, dtype=float)
        R_bw = quat_to_rotmat(np.asarray(att, dtype=float))
        vel_w = np.zeros(3) if vel_w is None else np.asarray(vel_w, dtype=float)
        origin_w = pos_w + R_bw.T @ self.radar_mount
        cam_origin_w = pos_w + R_bw.T @ self.cam_mount
        t = float(frame.t)
        Rr = cfg.radar_r
        Rc = cfg.cam_r
        cam_ok = cam is not None and len(cam) > 0

        # 1) predict every track to the current scan time.
        for tr in self.tracks:
            dt = 0.0 if tr.last_t is None else t - tr.last_t
            _predict(tr, dt, cfg.track_config)

        # 2) per-track predicted measurements & innovation covariances
        #    (radar 3-D + camera 1-D arrays kept parallel).
        zhats, Hs = [], []
        for tr in self.tracks:
            zhat_r, H_r = _measurement_jacobian(
                tr.x, pos_w, R_bw, self.radar_mount, vel_w
            )
            S_r = H_r @ tr.P @ H_r.T + Rr
            # camera azimuth predicted directly in the body frame.
            p_w = tr.x[:2] - cam_origin_w[:2]
            p_b = R_bw[:2, :2] @ p_w - self.cam_mount[:2]
            az_c = float(np.arctan2(p_b[1], p_b[0]))
            r_xy = float(np.hypot(p_b[0], p_b[1])) or 1e-12
            H_c = np.zeros((1, 4))
            daz_dp_b = np.array([-p_b[1], p_b[0]]) / (r_xy * r_xy)
            H_c[0, :2] = daz_dp_b @ R_bw[:2, :2]
            S_c = H_c @ tr.P @ H_c.T + Rc
            zhats.append((zhat_r, H_r, az_c, H_c, S_r, S_c))

        # 3) association: radar detections first (they birth/update shared
        #    tracks), then camera features assigned greedily to the same
        #    tracks by azimuth distance.
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
                    nu = z - zhats[j][0]
                    d2 = float(nu @ np.linalg.inv(zhats[j][4]) @ nu)
                    if d2 < cfg.track_config.gate_chi2:
                        cand.append((d2, j, k))
            cand.sort(key=lambda c: c[0])
            for d2, j, k in cand:
                if j in assigned_tr or k in assigned_det:
                    continue
                assigned_tr.add(j)
                assigned_det.add(k)
                self._radar_update(self.tracks[j], k, frame, R_bw, pos_w, vel_w, Rr)

        confirmed_idx = [j for j, tr in enumerate(self.tracks) if tr.confirmed]
        tentative_idx = [j for j, tr in enumerate(self.tracks) if not tr.confirmed]
        _greedy_pass(confirmed_idx)
        _greedy_pass(tentative_idx)

        # camera features assigned to the tracks; a track updated by radar
        # this scan still gets a camera update below -- two sequential
        # updates per tick is the whole point.  Association uses the
        # *pre-update* predictions; the update itself recomputes them from
        # the post-radar state (see ``_cam_update``).
        if cam_ok:
            n_feat = len(cam)
            assigned_feat: set[int] = set()
            for j in sorted(
                range(len(self.tracks)),
                key=lambda j: (j not in assigned_tr, j),
            ):
                azhat, Hc, Sc = zhats[j][2], zhats[j][3], zhats[j][5]
                best_k, best_d2 = -1, float("inf")
                for k in range(n_feat):
                    if k in assigned_feat:
                        continue
                    az_m = self._pixel_to_azimuth(float(cam.pts[k, 0]))
                    nu = az_m - azhat
                    d2 = float(nu * nu / Sc[0, 0])
                    if d2 < cfg.cam_gate_chi2 and d2 < best_d2:
                        best_k, best_d2 = k, d2
                if best_k >= 0:
                    assigned_feat.add(best_k)
                    self._cam_update(
                        self.tracks[j], best_k, cam, R_bw, cam_origin_w, Rc
                    )

        # 4) coast unassigned tracks; delete the stale ones (in sensor time,
        #    exactly as RadarTracker).
        survivors: list[Track] = []
        for j, tr in enumerate(self.tracks):
            tr.last_t = t
            tr.age += 1
            if j in assigned_tr or (cam_ok and j in assigned_feat):
                survivors.append(tr)
                continue
            tr.missed += 1
            if tr.confirmed and tr.missed <= cfg.track_config.coast_max:
                survivors.append(tr)
        self.tracks = survivors

        # 5) birth new tracks for unassigned radar detections (the camera
        #    alone does not birth: without a range it cannot place a state).
        for k in range(n_det):
            if k in assigned_det:
                continue
            self._birth(k, frame, origin_w, R_bw, vel_w, t, cfg.track_config)

        self.tracks.sort(key=lambda tr: tr.track_id)
        return FusedFrame(
            t=t,
            n_detections=n_det,
            n_features=0 if not cam_ok else len(cam),
            n_radar_updates=len(assigned_det),
            n_cam_updates=0 if not cam_ok else len(assigned_feat),
            tracks=tuple(self.tracks),
        )

    # -- updates ------------------------------------------------------------
    def _radar_update(
        self,
        tr: Track,
        k: int,
        frame: RadarFrame,
        R_bw: np.ndarray,
        pos_w: np.ndarray,
        vel_w: np.ndarray,
        R: np.ndarray,
    ) -> None:
        """EKF update of ``tr`` with radar detection ``k`` (Joseph form)."""
        z = np.array([frame.ranges[k], frame.azimuths[k], frame.range_rates[k]])
        zhat, H = _measurement_jacobian(
            tr.x, pos_w, R_bw, self.radar_mount, vel_w
        )
        nu = z - zhat
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ nu
        I_KH = np.eye(4) - K @ H
        tr.P = I_KH @ tr.P @ I_KH.T + K @ R @ K.T
        tr.missed = 0
        tr.hits += 1
        tr.confirmed = tr.hits >= self.config.track_config.confirm_min

    def _cam_update(
        self,
        tr: Track,
        k: int,
        cam: CameraFrame,
        R_bw: np.ndarray,
        cam_origin_w: np.ndarray,
        Rc: np.ndarray,
    ) -> None:
        """EKF update of ``tr`` with camera feature ``k`` (Joseph form).

        The prediction (``azhat``/``Hc``) is recomputed from the *current*
        state -- this track may already have received a radar update this
        scan, so reusing the pre-update Jacobian would over-correct.
        """
        az_m = self._pixel_to_azimuth(float(cam.pts[k, 0]))
        p_w = tr.x[:2] - cam_origin_w[:2]
        p_b = R_bw[:2, :2] @ p_w - self.cam_mount[:2]
        azhat = float(np.arctan2(p_b[1], p_b[0]))
        r_xy = float(np.hypot(p_b[0], p_b[1])) or 1e-12
        Hc = np.zeros((1, 4))
        daz_dp_b = np.array([-p_b[1], p_b[0]]) / (r_xy * r_xy)
        Hc[0, :2] = daz_dp_b @ R_bw[:2, :2]
        nu = np.array([az_m - azhat])
        S = Hc @ tr.P @ Hc.T + Rc
        K = tr.P @ Hc.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ nu
        I_KH = np.eye(4) - K @ Hc
        tr.P = I_KH @ tr.P @ I_KH.T + K @ Rc @ K.T
        tr.missed = 0
        tr.hits += 1
        tr.confirmed = tr.hits >= self.config.track_config.confirm_min

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
        """Spawn a track from radar detection ``k`` (identical policy to
        v0.18 -- the camera does not birth, it refines)."""
        R_wb = R_bw.T
        p_sensor = frame.points[k]
        xy = origin_w[:2] + (R_wb @ p_sensor)[:2]

        u = _bearing_unit(xy - origin_w[:2])
        perp = np.array([-u[1], u[0]])
        vr_m = float(frame.range_rates[k])
        v_abs = vel_w[:2] - vr_m * u

        sig_r = cfg.range_noise_sigma_m
        sig_cross = float(np.hypot(xy[0] - origin_w[0], xy[1] - origin_w[1])) * np.deg2rad(
            cfg.angle_noise_sigma_deg
        )
        P_pos = sig_r**2 * np.outer(u, u) + sig_cross**2 * np.outer(perp, perp)
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
            confirmed=cfg.confirm_min <= 1,
        )
        self._next_id += 1
        self.tracks.append(tr)


# --------------------------------------------------------------------------
# Fused frame
# --------------------------------------------------------------------------
@dataclass
class FusedFrame:
    """One fusion cycle result.

    Parameters
    ----------
    t : float
        Host time of the cycle (the radar scan time).
    n_detections : int
        Radar detections this cycle.
    n_features : int
        Camera features observed this cycle (0 when the camera was skipped).
    n_radar_updates : int
        Tracks that received a radar update this cycle.
    n_cam_updates : int
        Tracks that received a camera update this cycle.
    tracks : tuple of Track
        Surviving tracks (live objects, same mutation semantics as v0.18).
    """

    t: float
    n_detections: int
    n_features: int
    n_radar_updates: int
    n_cam_updates: int
    tracks: tuple[Track, ...] = ()

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)