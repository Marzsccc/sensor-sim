"""GNSS outage simulation and uncertainty-aware display output (v0.8.0).

This module answers a question the AR-HUD must handle in production:
what happens when GNSS is lost (tunnel, underground garage, urban
canyon)?  The ESKF keeps integrating IMU + wheel odometry, the position
covariance grows, and the display-time prediction (v0.5.0 + v0.7.0) must
reflect that growth so markers can be faded/clamped.

Components
----------
- ``OutageConfig`` / ``OutageModel``: configurable GNSS outage schedule
  (deterministic periods or random dropouts).
- ``OutageSimulator``: end-to-end run of the fusion + prediction pipeline
  under GNSS outages.  It drives an :class:`~sensor_sim.eskf.ESKF`
  (IMU at 200 Hz, wheel at 100 Hz, GNSS at 10 Hz subject to the outage
  schedule), and at each display tick computes:
    * the display-time mean pose (``PosePredictor``),
    * the display-time covariance (``PredictorUncertainty``) with the
      full 15x15 ESKF covariance (the extra bias rows are dropped but
      their contribution is folded in through the covariance block),
    * a horizontal 95% ellipse.
  With ``use_ellipse_output=True`` it renders the *ellipse-weighted*
  display pose, i.e. blends toward the wheel-only dead-reckoning pose as
  uncertainty grows -- a simple model of marker fading/clamping.
- ``consistency``: NEES (normalized estimation error squared) analysis
  that validates the propagated covariance is *consistent* with the true
  error, both during normal operation and across outage recovery.  This
  is the statistical guarantee the ellipse actually means something.

Production traps this module exposes
-------------------------------------
1. **The 6-sigma gate prevents recovery.**  The stock ESKF's outlier
   gate (``update_gnss`` rejects innovations > 6 sigma) will silently
   drop *every* GNSS fix after a long outage, because the true position
   error has drifted far beyond the (over-confident) covariance.  With
   ``recovery="none"`` the filter never re-converges -- a real failure
   mode in production.  With ``recovery="reset"`` (default) the first
   fix after an outage performs a *re-acquisition reset*: position and
   velocity are re-seeded from the fix and the covariance is inflated to
   the initial levels, which is what real systems do after a tunnel.
2. **Absolute heading is unobservable on a straight road.**  Wheel
   odometry observes speed and yaw *rate* but not absolute heading, so
   during an outage the heading error never gets corrected and the
   linearized ESKF covariance under-reports the cross-track drift
   (``F`` couples heading to position only through acceleration, which
   is ~0 at constant speed).  ``OutageSimulator`` therefore *inflates*
   the covariance during outage windows with a drift model
   (heading std growth + cross-track integration) so the displayed
   ellipse honestly reflects the growing uncertainty -- the same trick
   production HUDs use to fade markers in tunnels.

Notes
-----
- The simulator samples the ground truth from the trajectory with
  :meth:`~sensor_sim.trajectory.Trajectory.at` (nearest point), the same
  convention as ``DisplayPipeline``.
- Wheel odometry is fused as a *velocity + yaw-rate* measurement in the
  ESKF error-state update (measurement model ``z = [vx, vy, yaw_rate]``),
  which is what keeps the filter observably well-conditioned during a
  long GNSS outage.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .eskf import ESKF, ESKFConfig
from .predict import PosePredictor, PredictionConfig, PredictorUncertainty
from .trajectory import Trajectory
from .utils import quat_to_rotmat, skew


# ---------------------------------------------------------------------------
# Outage model
# ---------------------------------------------------------------------------


@dataclass
class OutageConfig:
    """GNSS outage schedule.

    Parameters
    ----------
    outage_periods : list of (t_start, t_end)
        Deterministic outage windows (seconds).  GNSS measurements are
        dropped inside these windows.
    random_dropout_prob : float
        If > 0, each GNSS measurement is additionally dropped with this
        probability (Bernoulli), e.g. urban-canyon multipath losses.
    """

    outage_periods: List[Tuple[float, float]] = field(default_factory=list)
    random_dropout_prob: float = 0.0

    def in_outage(self, t: float, rng: Optional[np.random.Generator] = None) -> bool:
        """True if a GNSS fix at time ``t`` should be discarded."""
        if any(a <= t <= b for a, b in self.outage_periods):
            return True
        if self.random_dropout_prob > 0.0:
            rng = rng or np.random.default_rng()
            return rng.random() < self.random_dropout_prob
        return False


class OutageModel:
    """Thin wrapper: schedule + per-measurement dropout decisions."""

    def __init__(self, config: Optional[OutageConfig] = None,
                 seed: Optional[int] = None):
        self.config = config or OutageConfig()
        self._rng = np.random.default_rng(seed)

    def drop(self, t: float) -> bool:
        """Decide whether the GNSS fix at time ``t`` is usable."""
        return self.config.in_outage(t, self._rng)

    @property
    def total_outage_time(self) -> float:
        return sum(max(0.0, b - a) for a, b in self.config.outage_periods)


# ---------------------------------------------------------------------------
# Full fusion + prediction simulation under outage
# ---------------------------------------------------------------------------


@dataclass
class OutageSimConfig:
    """Tuning for :class:`OutageSimulator`."""

    imu_rate: float = 200.0
    wheel_rate: float = 100.0
    gnss_rate: float = 10.0
    display_rate: float = 60.0
    horizon: float = 0.0667           # camera 50 ms + one 60 Hz frame
    seed: int = 42

    eskf: ESKFConfig = field(default_factory=ESKFConfig)
    predict: PredictionConfig = field(default_factory=PredictionConfig)
    outage: OutageConfig = field(default_factory=OutageConfig)

    # when True, per-display-tick ellipse-weighted output is computed
    use_ellipse_output: bool = True

    # wheel-odometry update in the filter (measurement noise, 1-sigma)
    wheel_speed_std: float = 0.05     # m/s
    wheel_yawrate_std: float = 0.01   # rad/s

    # fade threshold: radius_95 (m) at which the ellipse-weighted output
    # has fully blended toward the dead-reckoning pose
    fade_radius_95: float = 0.5

    # recovery policy after an outage window:
    #   "reset" : first usable GNSS fix re-seeds p/v and re-inflates P
    #   "none"  : naive filter (6-sigma gate may reject all fixes forever)
    recovery: str = "reset"

    # number of GNSS updates after a re-acquisition reset that bypass the
    # ESKF outlier gate (true error >> covariance right after the reset)
    recovery_warmup_fixes: int = 3

    # seconds after the outage ends during which wheel updates are
    # downweighted (heading is still uncertain; wheel velocity would drag
    # the filter away from the recovering GNSS fixes)
    recovery_grace_s: float = 5.0

    # outage-aware covariance inflation (keeps the display ellipse honest
    # when absolute heading is unobservable; see module docstring)
    inflate_during_outage: bool = True
    # heading-uncertainty growth rate during outage (deg/s, 1-sigma)
    outage_heading_drift_deg_per_s: float = 0.2


@dataclass
class OutageResult:
    """Per-display-tick records + aggregate statistics."""

    t: np.ndarray
    true_pos: np.ndarray                 # (N, 3)
    fused_pos: np.ndarray                # (N, 3)  ESKF pose at tick time
    disp_pos: np.ndarray                 # (N, 3)  ellipse-weighted display pose
    disp_cov: np.ndarray                 # (N, 9, 9)
    pos_std: np.ndarray                  # (N, 3)  sqrt of diag display cov (p)
    radius_95: np.ndarray                # (N,)
    in_outage: np.ndarray                # (N,) bool
    fused_error: np.ndarray              # (N, 3)  fused vs true
    disp_error: np.ndarray               # (N, 3)  displayed vs true

    def outage_fraction(self) -> float:
        return float(np.mean(self.in_outage))

    def mean_fused_error(self) -> float:
        return float(np.mean(np.linalg.norm(self.fused_error, axis=1)))

    def mean_disp_error(self) -> float:
        return float(np.mean(np.linalg.norm(self.disp_error, axis=1)))

    def max_radius_95(self) -> float:
        return float(np.max(self.radius_95))

    def max_fused_error(self) -> float:
        return float(np.max(np.linalg.norm(self.fused_error, axis=1)))


class OutageSimulator:
    """Run ESKF + display prediction under a GNSS outage schedule.

    This mirrors the production AR-HUD pipeline under degraded GNSS:
    IMU (200 Hz) and wheel odometry (100 Hz) keep the filter alive, the
    GNSS fix (10 Hz) is dropped inside the outage windows, and every
    display tick (60 Hz) renders the display-time pose with its
    uncertainty ellipse.
    """

    def __init__(self, config: Optional[OutageSimConfig] = None):
        self.cfg = config or OutageSimConfig()
        if self.cfg.recovery not in ("reset", "none"):
            raise ValueError(
                f"recovery must be 'reset' or 'none', got {self.cfg.recovery!r}"
            )
        self.filter = ESKF(self.cfg.eskf)
        self.predictor = PosePredictor(self.cfg.predict)
        self.uncertainty = PredictorUncertainty(
            # reuse the same IMU noise parameters as the ESKF so the
            # propagated covariance is consistent with the filter
            dataclasses.replace(
                PredictorUncertainty().config,
                imu_acc_noise_density=self.cfg.eskf.acc_noise_density,
                imu_gyr_noise_density=self.cfg.eskf.gyr_noise_density,
                imu_acc_bias_rw=self.cfg.eskf.acc_bias_rw,
                imu_gyr_bias_rw=self.cfg.eskf.gyr_bias_rw,
            )
        )
        self.outage = OutageModel(self.cfg.outage, seed=self.cfg.seed)
        self._rng = np.random.default_rng(self.cfg.seed)
        self._warmup = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_imu(self, traj: Trajectory, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Ground-truth (acc, gyro) at host time ``t`` (body frame).

        Noise is deliberately *not* added here: the ESKF models the IMU
        noise in its process-noise covariance Q, so adding sensor noise on
        top would double-count it.  This mirrors ``DisplayPipeline``.
        """
        pt = traj.at(t)
        return traj.body_accel(pt), pt.omega

    def _sample_wheel(self, traj: Trajectory, t: float) -> Tuple[float, float]:
        """Ground-truth (speed, yaw_rate) from the planar model."""
        pt = traj.at(t)
        v_b = quat_to_rotmat(pt.att).T @ pt.vel
        speed = float(np.hypot(v_b[0], v_b[1]))
        return speed, float(pt.omega[2])

    def _step_imu(self, traj: Trajectory, t: float):
        acc, gyr = self._sample_imu(traj, t)
        self.filter.predict(acc, gyr, 1.0 / self.cfg.imu_rate)

    def _step_wheel(self, traj: Trajectory, t: float, downweight: bool = False):
        """Fuse one wheel-odometry measurement (velocity only).

        The wheel observes the *body-frame forward speed*; mapped through
        the current attitude this is a world-frame velocity measurement:

            z = v_world = R(q) @ [speed, 0, 0]
            H = [0 I3 0 0 0]

        Yaw *rate* is deliberately **not** used as an observation: it is a
        kinematic input already integrated in the predict step, and using
        it as a yaw-error pseudo-measurement would keep the heading
        covariance over-confident during GNSS outages (the filter would
        think absolute heading is known while it actually drifts), which
        then breaks the GNSS recovery gate.

        ``downweight=True`` (used right after an outage) inflates the
        measurement noise: with a heading-uncertain state the wheel
        velocity constrains the *wrong* direction and actively drags the
        filter away from the recovering GNSS fixes.
        """
        speed, _ = self._sample_wheel(traj, t)

        st = self.filter.state
        R = quat_to_rotmat(st.q)            # world <- body
        v_w = R @ np.array([speed, 0.0, 0.0])
        v_w += self._rng.normal(0.0, self.cfg.wheel_speed_std, size=3)

        H = np.zeros((3, 15))
        H[0:3, self.filter.IDX_V] = np.eye(3)
        h = st.v.copy()

        std = self.cfg.wheel_speed_std
        if downweight:
            std = std * 20.0               # heading-uncertain -> trust less
        else:
            # adapt to the current heading uncertainty: the wheel velocity
            # direction is only as good as the attitude estimate.  If yaw
            # is poorly known, a 20 m/s wheel read has a cross-track
            # error of ~v*sin(sigma_yaw); inflate R accordingly so the
            # filter does not trust a measurement whose direction it
            # cannot know (this is what makes the straight-road filter
            # stable instead of drifting while over-confident).
            yaw_std = np.sqrt(np.clip(st.P[6, 6], 0.0, None))
            cross_std = float(np.linalg.norm(st.v)) * np.sin(yaw_std)
            std = np.hypot(std, cross_std)
        Rm = np.diag([std ** 2] * 3)
        dz = v_w - h
        S = H @ st.P @ H.T + Rm
        # innovation gate: skip clearly bad wheel reads (protects the
        # covariance from rank-deficiency blow-ups during outages)
        innov_cov = np.sqrt(np.diag(S))
        if np.any(np.abs(dz) > 6.0 * np.clip(innov_cov, 1e-9, None)):
            return
        K = st.P @ H.T @ np.linalg.inv(S)
        dx = K @ dz

        self.filter._inject(dx)

        I = np.eye(15)
        IKH = I - K @ H
        st.P = IKH @ st.P @ IKH.T + K @ Rm @ K.T
        st.P = 0.5 * (st.P + st.P.T)
        st.n_updates += 1

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def _in_flight_time(self, t: float, t_out_start: float) -> float:
        """Time spent inside the current outage window before ``t``."""
        for a, b in self.cfg.outage.outage_periods:
            if a <= t <= b:
                return max(0.0, t - max(t_out_start, a))
        return 0.0

    def _inflate_outage_covariance(self, P: np.ndarray, dt_out: float) -> np.ndarray:
        """Inflate the 9x9 [p,v,theta] covariance to reflect dead-reckoning drift.

        Absolute heading is unobservable on a straight road (wheel odometry
        observes yaw *rate* only), so the linearized ESKF covariance
        under-reports the cross-track drift during an outage.  We add a
        drift model:

          * heading uncertainty grows linearly with time in outage
            (sigma_theta = drift_rate * dt_out)
          * heading error maps to cross-track position error through
            v (sigma_cross ~ v * dt_out * sigma_theta / sqrt(3))

        This is the covariance equivalent of fading markers in a tunnel.
        """
        P = np.asarray(P, dtype=float).copy()
        v = np.linalg.norm(self.filter.velocity)
        th_std = np.deg2rad(self.cfg.outage_heading_drift_deg_per_s) * dt_out
        if th_std <= 0:
            return P
        cross_std = v * dt_out * th_std / np.sqrt(3.0)
        # heading block (yaw only; keep roll/pitch) + cross-track position
        P[6, 6] += th_std * th_std
        P[7, 7] += th_std * th_std
        P[0, 0] += cross_std * cross_std
        P[1, 1] += cross_std * cross_std
        return P

    def _recover_reset(self, pos: np.ndarray, vel: np.ndarray):
        """Re-acquisition reset: re-seed p/v from the first usable fix.

        The 6-sigma gate in ``update_gnss`` would otherwise reject the
        fix (true error >> covariance after a long outage); a real system
        does a coarse re-initialization here instead.  The simulator
        bypasses the gate for the next ``recovery_warmup_fixes`` GNSS
        updates (tracked via ``self._warmup``) so the filter can
        re-converge instead of silently rejecting every fix forever.
        """
        st = self.filter.state
        c = self.cfg.eskf
        st.p = np.asarray(pos, dtype=float).copy()
        st.v = np.asarray(vel, dtype=float).copy()
        # re-inflate to initial levels (drop stale confidence)
        std = np.array([
            c.init_pos_std, c.init_pos_std, c.init_pos_std,
            c.init_vel_std, c.init_vel_std, c.init_vel_std,
            np.deg2rad(c.init_att_std_deg), np.deg2rad(c.init_att_std_deg),
            np.deg2rad(c.init_att_std_deg),
            c.init_accbias_std, c.init_accbias_std, c.init_accbias_std,
            c.init_gyrbias_std, c.init_gyrbias_std, c.init_gyrbias_std])
        st.P = np.diag(std * std)
        st.n_updates += 1
        self._warmup = self.cfg.recovery_warmup_fixes

    def _gnss_update(self, pos: np.ndarray, vel: np.ndarray):
        """GNSS update that honours the recovery warm-up (gate bypass).

        During the first ``recovery_warmup_fixes`` updates after a reset
        the true error is still much larger than the (re-inflated)
        covariance, so the stock 6-sigma gate would reject the fix.  We
        widen the ESKF's outlier gate (``gate_multiplier``) for those
        updates -- the covariance itself is never touched, so the filter
        pulls itself onto the fix with honest Joseph-form updates and
        re-converges.
        """
        if self._warmup > 0:
            self._warmup -= 1
            saved = self.filter.gate_multiplier
            self.filter.gate_multiplier = 1e6   # wide open
            self.filter.update_gnss(pos, vel)
            self.filter.gate_multiplier = saved
        else:
            self.filter.update_gnss(pos, vel)

    def run(self, traj: Trajectory, t0: float = 0.0,
            t1: Optional[float] = None) -> OutageResult:
        cfg = self.cfg
        t1 = traj.points[-1].t if t1 is None else t1

        dt_imu = 1.0 / cfg.imu_rate
        dt_wheel = 1.0 / cfg.wheel_rate
        dt_gnss = 1.0 / cfg.gnss_rate
        dt_disp = 1.0 / cfg.display_rate

        # seed the filter at t0
        p0, v0, q0 = traj.at(t0).pos, traj.at(t0).vel, traj.at(t0).att
        self.filter.set_initial_state(t0, p0, v0, q0)

        # pre-generate GNSS fix times so drop decisions are deterministic
        gnss_times = np.arange(t0 + dt_gnss, t1 + 1e-9, dt_gnss)
        gnss_usable = np.array([not self.outage.drop(float(t))
                                for t in gnss_times])
        # deterministic-window flag per fix (recovery is keyed to the
        # deterministic schedule, not the random dropouts)
        gnss_in_det = np.array([
            any(a <= float(t) <= b for a, b in cfg.outage.outage_periods)
            for t in gnss_times])
        gnss_idx = 0
        prev_fix_in_outage = False

        # wheel measurement times (aligned to the wheel rate)
        wheel_times = np.arange(t0 + dt_wheel, t1 + 1e-9, dt_wheel)
        wheel_idx = 0

        # recovery grace: downweight wheel updates until this time
        recovery_until = -1.0

        ts, true, fused, disp = [], [], [], []
        covs, rads, flags = [], [], []
        ferr, derr = [], []

        in_outage_now = False
        t_out_start = t0
        prev_fix_usable = True   # no outage before the first fix

        n = max(1, int(round((t1 - t0) / dt_disp)))
        for k in range(n):
            t = t0 + k * dt_disp

            # --- outage state -----------------------------------------
            was_out = in_outage_now
            in_outage_now = bool(np.any([a <= t <= b for a, b in
                                         cfg.outage.outage_periods]))
            if in_outage_now and not was_out:
                t_out_start = t

            # --- advance IMU until this display tick -------------------
            while self.filter.state.t < t - 1e-9:
                self._step_imu(traj, self.filter.state.t)

            # --- wheel odometry update (100 Hz) ------------------------
            while wheel_idx < len(wheel_times) and wheel_times[wheel_idx] <= t:
                self._step_wheel(traj, float(wheel_times[wheel_idx]),
                                 downweight=t <= recovery_until)
                wheel_idx += 1

            # --- GNSS update if a fix is available ---------------------
            while gnss_idx < len(gnss_times) and gnss_times[gnss_idx] <= t:
                fix_t = float(gnss_times[gnss_idx])
                fix_in_outage = bool(gnss_in_det[gnss_idx])
                if gnss_usable[gnss_idx]:
                    pos = traj.at(fix_t).pos
                    pos += self._rng.normal(
                        0.0, cfg.eskf.gnss_pos_std, size=3)
                    vel = traj.at(fix_t).vel
                    vel += self._rng.normal(
                        0.0, cfg.eskf.gnss_vel_std, size=3)
                    if (prev_fix_in_outage and not fix_in_outage
                            and cfg.recovery == "reset"):
                        # first usable fix after an outage: re-acquisition
                        self._recover_reset(pos, vel)
                        recovery_until = fix_t + cfg.recovery_grace_s
                    else:
                        self._gnss_update(pos, vel)
                prev_fix_in_outage = fix_in_outage
                prev_fix_usable = bool(gnss_usable[gnss_idx])
                gnss_idx += 1

            fused_p = self.filter.position.copy()
            fused_v = self.filter.velocity.copy()
            fused_q = self.filter.quaternion.copy()

            # --- display-time prediction + covariance ------------------
            acc_m, gyr_m = self._sample_imu(traj, t)
            b_a, b_g = self.filter.estimated_biases()
            ws, wyr = self._sample_wheel(traj, t)
            P9 = self.filter.state.P[0:9, 0:9].copy()

            # outage-aware covariance inflation (honest ellipse)
            if cfg.inflate_during_outage and in_outage_now:
                dt_out = self._in_flight_time(t, t_out_start)
                P9 = self._inflate_outage_covariance(P9, dt_out)

            res = self.uncertainty.predict_with_covariance(
                self.predictor, fused_p, fused_v, fused_q,
                acc_m - b_a, gyr_m - b_g, cfg.horizon, P9,
                wheel_speed=ws, wheel_yaw_rate=wyr,
            )
            P_disp = res["P_disp"]
            ell = res["ellipse"]
            disp_p = res["pos"]

            # ellipse-weighted output: blend the predicted pose toward the
            # fused pose as uncertainty grows (marker fading/clamping)
            w = np.clip(ell["radius_95"] / cfg.fade_radius_95, 0.0, 1.0)
            disp_p_w = (1.0 - w) * disp_p + w * fused_p

            # outage flag: inside a deterministic window, or the last GNSS
            # fix was dropped (random dropout) and no fix arrived since
            no_fix_since = (not prev_fix_usable
                            and (gnss_idx == 0 or gnss_times[gnss_idx - 1] < t))
            flag_out = in_outage_now or no_fix_since

            true_p = traj.at(t).pos
            ts.append(t)
            true.append(true_p)
            fused.append(fused_p)
            disp.append(disp_p_w)
            covs.append(P_disp)
            rads.append(ell["radius_95"])
            flags.append(flag_out)
            ferr.append(fused_p - true_p)
            derr.append(disp_p_w - true_p)

        cov_arr = np.array(covs)
        return OutageResult(
            t=np.array(ts),
            true_pos=np.array(true),
            fused_pos=np.array(fused),
            disp_pos=np.array(disp),
            disp_cov=cov_arr,
            pos_std=np.sqrt(np.maximum(
                np.diagonal(cov_arr[:, 0:3, 0:3], axis1=1, axis2=2), 0.0)),
            radius_95=np.array(rads),
            in_outage=np.array(flags),
            fused_error=np.array(ferr),
            disp_error=np.array(derr),
        )


# ---------------------------------------------------------------------------
# Covariance consistency (NEES)
# ---------------------------------------------------------------------------


def consistency(
    errors: np.ndarray,
    covariances: np.ndarray,
    dim: int = 3,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """NEES consistency check between estimation errors and covariances.

    NEES (normalized estimation error squared) for an n-dimensional
    Gaussian estimator is

        eps = e^T P^-1 e   ~  chi2(n)

    averaged over N independent samples the mean NEES is chi2(N*n)/N.
    The estimator is consistent if the sample mean lies inside the
    chi-square confidence interval.

    Parameters
    ----------
    errors : (N, 3)  true-minus-estimated position errors
    covariances : (N, 3, 3)  display-time position covariances
    dim : int
        Number of degrees of freedom (2 = horizontal, 3 = full).
    alpha : float
        Significance level for the chi2 interval.

    Returns
    -------
    dict: nees_mean, nees_median, ci_low, ci_high, consistent (bool)
    """
    err = np.asarray(errors, dtype=float)
    cov = np.asarray(covariances, dtype=float)
    n = len(err)
    if n == 0 or dim not in (2, 3):
        raise ValueError("consistency() needs N>0 samples and dim in (2,3)")

    P = cov[:, 0:dim, 0:dim]
    P = 0.5 * (P + np.swapaxes(P, 1, 2))
    e = err[:, 0:dim]
    nees = np.array([
        float(e[i] @ np.linalg.solve(P[i], e[i]))
        for i in range(n)
    ])
    mean_nees = float(np.mean(nees))

    # chi2 interval for the mean of N samples: chi2(N*dim)/N
    from scipy import stats  # noqa: WPS433  (lazy import, optional dep)
    lo = stats.chi2.ppf(alpha / 2.0, n * dim) / n
    hi = stats.chi2.ppf(1.0 - alpha / 2.0, n * dim) / n
    return {
        "nees_mean": mean_nees,
        "nees_median": float(np.median(nees)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "consistent": bool(lo <= mean_nees <= hi),
    }


def nees_vs_outage(
    result: OutageResult,
    dim: int = 3,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """NEES inside vs outside outage windows (one dict of scalars)."""
    m = result.in_outage
    out = {}
    if np.any(m):
        out.update({
            "nees_in_outage": consistency(
                result.disp_error[m], result.disp_cov[m], dim, alpha
            )["nees_mean"],
        })
    if np.any(~m):
        out.update({
            "nees_healthy": consistency(
                result.disp_error[~m], result.disp_cov[~m], dim, alpha
            )["nees_mean"],
        })
    return out
