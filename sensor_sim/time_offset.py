"""Online sensor clock-offset estimation, joint with the kinematic state -- v0.24.0.

Motivation
----------
v0.22.0 and v0.23.0 assumed the timestamps were *trustworthy*: a sample's
capture time was known, and the only question was *when* to fuse it (measurement
time vs receive time).  In a real vehicle that assumption is usually false.  The
GNSS receiver, the camera and the wheel-encoder ECU each run their own
oscillator; the host reads their stamps as if they were host time.  A sensor
clock that is ``b`` seconds *fast* makes every sample look ``b`` seconds
*newer* than it is -- the data is stale by ``b``, and the pose built from it
trails reality by roughly ``speed x b`` metres.  No amount of rewind/replay
fixes this, because the input *timestamps themselves* are wrong.

``sensor_sim.latency`` already models this (``SensorClock``) and offers an
offline fix (``TimeSync``): linear regression on (host, sensor) timestamp
*pairs* -- IEEE 1588 / PTP style.  That works when you have a paired reference
(e.g. PPS, or a bidirectional link).  A vehicle has neither: there is no
second clock to compare against, only a moving body and a stream of position
fixes.

This module estimates the offset **online, jointly with the state**.  The
measurement model is one line of algebra: if a fix's true capture time is
``t_rep - b`` and the filter's state is stamped at ``t_rep``, the fix predicts
the *future* position

    h(x) = p + v * (t_true - t_rep) = p - v * b

so the Jacobian is ``dh/db = -v``: the offset enters the innovation **through
velocity and only through velocity**.  Everything below follows from that one
term.

What is -- and is not -- observable
-----------------------------------
The tempting conclusion "no motion, no information; motion, information" is
*not* the whole story, and getting it wrong is the classic calibration trap:

1. **Standstill (v == 0): unobservable, and harmless.**  ``dh/db = 0``, so no
   number of fixes shrinks ``P_bb``; and the induced error ``v x b`` is zero,
   so nothing is broken either.
2. **Constant speed: still unobservable.**  Shifting the clock by ``b`` and
the trajectory start by ``v * b`` produces *bit-identical* measurements -- a
   time offset is indistinguishable from a time-shifted trajectory.  A
   constant-speed highway drive therefore **cannot** calibrate a clock offset,
   however long it is.
3. **Varying speed, with an independent motion reference on the reference
   clock** (wheel odometry / vehicle speed / IMU mechanisation): **observable**.
   Marginalising the free initial position, the Fisher information from
   position fixes is ``sum((v_k - mean(v))^2) / sigma^2``, so the achievable
   std is ``sigma / sqrt(sum((v_k - mean(v))^2))`` -- it is the *variation* of
   speed that carries the information, not its magnitude.  At 20 +- 10 m/s with
   1 m GNSS noise, 30 s of driving pins the offset to ~8 ms.
   Without such a reference (velocity inferred from the same position stream),
   no trajectory shape helps: the offset stays degenerate.

The two-sided punchline that makes this usable: *the state in which the offset
cannot be learned is also the state in which it does no harm* -- the induced
error is ``v x b``.  A filter can start with a vague prior and let the estimate
converge as the vehicle pulls away, decelerates, or changes speed at all.

State layout
------------
``x = [p_1..p_n, v_1..v_n, b, (d)]`` -- the kinematic block is the same
constant-velocity model used by ``delay_fusion.CVModel``; ``b`` is the constant
(slowly wandering) clock offset in seconds and ``d`` an optional unitless rate
error (``d = ppm x 1e-6``), each as a random walk.

Two observation channels:

* :meth:`OffsetAugmentedFilter.update` -- a position fix whose stamp is on the
  *skewed* sensor clock (this is what the offset model must explain).
* :meth:`OffsetAugmentedFilter.update_velocity` -- a velocity/speed fix on the
  *reference* clock (wheel odometry, vehicle CAN speed, IMU-derived velocity).
  This is the independent motion reference that breaks the degeneracy of
  case 2 above; it is what a real vehicle always has and a bare GNSS stream
  never has.

Clock convention (fixed by tests)
---------------------------------
``ClockOffset(bias_s=b, rate_ppm=r, t_ref)`` maps

    t_reported = t_true + b + d * (t_true - t_ref),      d = r * 1e-6

so a **positive** offset means the sensor clock *reads ahead*: stamps are later
than reality and the sample is **stale** by ``b``.  The filter's ``t`` is the
**reference (host) clock** -- the clock the display runs on -- and the state it
carries is the pose at that reference time.

Who drives that clock matters, and getting it wrong is a trap this module has
already fallen into once (it is why ``update`` contains no ``predict_to``):

* **Right:** advance the clock with the *reference-clock* stream --
  :meth:`OffsetAugmentedFilter.update_velocity` (wheel odometry / IMU), or an
  explicit ``predict_to`` from the caller.  A fix stamped ``t_rep`` is then
  evaluated at ``tau_hat = t_rep - offset_hat`` and inserted with a linear
  motion correction relative to the clock.
* **Wrong:** let the *skewed* stream advance the clock (``predict_to`` to each
  ``t_rep``).  The estimator then becomes self-deceiving: a ``b_hat`` that is
  too large makes every reference-clock velocity fix land *behind* the state's
  clock, and a lagged velocity stream reconstructs a lagged trajectory -- so
  the wrong ``b_hat`` fits the data perfectly.  The measured consequence was a
  flat valley of equally-good solutions, with a strong **sign asymmetry**
  (positive offsets converged, negative ones slid to ``b_hat ~ -0.25 s`` while
  reporting a 5 ms ``P_bb``: confidently wrong).  Driving the clock from the
  reference stream removes the valley and the estimator becomes symmetric in
  ``b``.  Lesson: *clock semantics belong to the estimator, not to the fusion
  trick* -- the same lesson v0.23.0 learned about rewinding.

A and B tolerance
-----------------
* ``t`` is advanced only by the reference-clock channel, so the propagation
  interval is exact to ``O(d * dt)`` -- with ``d`` in the ppm range and ``dt`` a
  sensor period this is sub-millimetre per step and is left out of the model
  (the ``d`` estimate itself is carried in the measurement model, where it
  matters).
* A position fix is inserted with a linear motion correction relative to the
  current clock; there is no rewind/replay here.  ``n_behind`` counts the fixes
  whose estimated true time precedes the clock (the usual case for a clock that
  reads ahead), ``n_ahead`` the ones that follow it.  Compose with
  ``delay_fusion`` / ``nonlinear_delay`` when *both* an unknown offset and an
  unknown transport delay are present (see docs/v0.24.0).
* The insertion is a first-order (EKF) treatment: the motion correction is
  linearised at the current estimate, and the second-order effect of ``P_bb``
  on the correction is dropped -- the same approximation every EKF makes about
  a parameter it linearises.

Reference: Bar-Shalom, "Update with out-of-sequence measurements in tracking:
exact solution" (IEEE TAES 2002); IEEE 1588 / PTP clock synchronisation;
Nilsson, "Real-time control systems with delays" (PhD thesis, Lund, 1998),
ch. 3 (timestamps as a measurement of a stochastic time offset).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .delay_fusion import CVModel

__all__ = [
    "ClockOffset",
    "OffsetAugmentedFilter",
    "OffsetDiagnostics",
    "bias_fisher_information",
    "crlb_bias_std",
]

# Speeds above this (m/s) count as "informative" for the diagnostics counter:
# below it the induced error v*b stays under ~25 mm even for a 0.5 s offset.
_INFORMATIVE_SPEED_EPS = 0.05


# ---------------------------------------------------------------------------
# Ground-truth clock model (simulation side)
# ---------------------------------------------------------------------------

@dataclass
class ClockOffset:
    """A sensor clock that is offset (and optionally rate-erroneous) vs the host.

    ``t_reported = t_true + bias_s + rate * (t_true - t_ref)`` with
    ``rate = rate_ppm * 1e-6``.  A positive ``bias_s`` means the sensor clock
    reads *ahead*: the sample is stamped later than it happened, i.e. it
    reaches the filter looking newer (and staler) than it is.

    Parameters
    ----------
    bias_s : float
        Constant offset in seconds.
    rate_ppm : float
        Constant rate error in parts per million (20 ppm is a plain crystal).
    t_ref : float
        Reference instant at which ``bias_s`` is exact.
    """

    bias_s: float = 0.0
    rate_ppm: float = 0.0
    t_ref: float = 0.0

    def __post_init__(self) -> None:
        for name in ("bias_s", "rate_ppm", "t_ref"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            setattr(self, name, value)

    @property
    def rate(self) -> float:
        """Unitless rate error ``d`` (d = 1 means the clock runs 2x fast)."""
        return self.rate_ppm * 1e-6

    def to_reported(self, t_true):
        """True time -> what the sensor stamps on the sample."""
        t_true = np.asarray(t_true, dtype=float)
        return t_true + self.bias_s + self.rate * (t_true - self.t_ref)

    def to_true(self, t_reported):
        """Reported stamp -> when the sample was really captured (exact inverse)."""
        t_reported = np.asarray(t_reported, dtype=float)
        return (t_reported - self.bias_s + self.rate * self.t_ref) / (1.0 + self.rate)

    def residual_s(self, t_true):
        """``t_reported - t_true`` at ``t_true`` (the staleness induced)."""
        t_true = np.asarray(t_true, dtype=float)
        return self.to_reported(t_true) - t_true

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"ClockOffset(bias={self.bias_s * 1e3:+.4f} ms, "
                f"rate={self.rate_ppm:+.1f} ppm)")


# ---------------------------------------------------------------------------
# Observability / performance prediction
# ---------------------------------------------------------------------------

def bias_fisher_information(speeds: Sequence[float], sigma: float) -> float:
    """Fisher information on the clock offset from position fixes (units 1/s^2).

    For a position fix of the (moving) body, ``dh/db = -v``.  With the initial
    position marginalised out (the usual case: nothing pins where the body
    started), the *constant* part of ``v`` is absorbed by the initial-position
    trade-off and only the centred speed carries information:

        I(b) = sum((v_k - mean(v))^2) / sigma^2

    Zero for a standstill **and** for constant speed -- see the module
    docstring.  ``v`` is the speed on the *reference* clock (odometry / IMU),
    not the speed inferred from the skewed stream.
    """
    v = np.asarray(speeds, dtype=float)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be a positive finite number")
    if v.ndim != 1:
        raise ValueError("speeds must be a 1-D sequence")
    if not np.all(np.isfinite(v)):
        raise ValueError("speeds must be finite")
    if v.size == 0:
        return 0.0
    centred = v - float(np.mean(v))
    return float(np.sum(centred * centred) / (sigma * sigma))


def crlb_bias_std(speeds: Sequence[float], sigma: float) -> float:
    """Achievable 1-sigma on the clock offset, ``sigma / sqrt(sum d_v^2)``.

    ``d_v = v - mean(v)`` are the centred reference-clock speeds.  Returns
    ``inf`` for a standstill **or for constant speed** (offset degenerate with
    a time-shifted trajectory).
    """
    info = bias_fisher_information(speeds, sigma)
    if info <= 0.0:
        return float("inf")
    return float(1.0 / np.sqrt(info))


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class OffsetDiagnostics:
    """Counters describing what the filter did."""

    n_updates: int = 0
    n_predicts: int = 0
    n_velocity_updates: int = 0
    n_ahead: int = 0
    n_behind: int = 0
    n_informative: int = 0

    def as_dict(self) -> dict:
        return dict(
            n_updates=self.n_updates,
            n_predicts=self.n_predicts,
            n_velocity_updates=self.n_velocity_updates,
            n_ahead=self.n_ahead,
            n_behind=self.n_behind,
            n_informative=self.n_informative,
        )


class OffsetAugmentedFilter:
    """CV Kalman filter with an augmented, jointly estimated clock offset.

    Parameters
    ----------
    model : CVModel
        Kinematic model for the ``2n`` position/velocity block.
    x0, P0 : np.ndarray
        Initial kinematic state ``(2n,)`` and covariance ``(2n, 2n)``.
    t0 : float
        Initial (reference) time.
    obs_sigma : float
        Default isotropic position measurement sigma (m); override per update.
    estimate_rate : bool
        Also estimate the unitless rate error ``d`` (needs a long baseline).
    bias0, bias_std0 : float
        Prior mean (s) and std (s) of the clock offset.  ``bias_std0`` encodes
        how large an offset you are willing to believe in -- 0.5 s covers any
        sane un-synchronised stream.
    rate0, rate_std0 : float
        Prior mean and std of the unitless rate error (``rate_std0=2e-4``
        corresponds to 200 ppm).
    bias_rw_std : float
        Random-walk intensity of the offset (s per sqrt(s)) -- how fast the
        offset is allowed to wander.
    rate_rw_std : float
        Random-walk intensity of the rate error (1 per sqrt(s)).
    record_history : bool
        Store ``(t, bias, bias_std, rate, rate_std)`` per update for plotting.
    """

    def __init__(
        self,
        model: CVModel,
        x0: np.ndarray,
        P0: np.ndarray,
        t0: float = 0.0,
        obs_sigma: float = 1.0,
        estimate_rate: bool = True,
        bias0: float = 0.0,
        bias_std0: float = 0.5,
        rate0: float = 0.0,
        rate_std0: float = 2e-4,
        bias_rw_std: float = 1e-4,
        rate_rw_std: float = 1e-6,
        record_history: bool = True,
    ) -> None:
        if not hasattr(model, "state_dim") or not hasattr(model, "F"):
            raise TypeError("model must be a CVModel-like object (F/Q/state_dim)")
        n = int(model.n_dim)
        self.model = model
        self.n_dim = n
        self.obs_sigma = float(obs_sigma)
        if not np.isfinite(self.obs_sigma) or self.obs_sigma <= 0.0:
            raise ValueError("obs_sigma must be positive and finite")
        self.estimate_rate = bool(estimate_rate)

        self._dim = 2 * n + (2 if self.estimate_rate else 1)
        self._i_b = 2 * n
        self._i_d = 2 * n + 1

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        P0 = np.asarray(P0, dtype=float)
        if x0.shape != (2 * n,):
            raise ValueError(f"x0 must have shape ({2 * n},), got {x0.shape}")
        if P0.shape != (2 * n, 2 * n):
            raise ValueError(f"P0 must have shape ({2 * n}, {2 * n}), got {P0.shape}")

        for name, value in (("bias_std0", bias_std0), ("rate_std0", rate_std0),
                            ("bias_rw_std", bias_rw_std), ("rate_rw_std", rate_rw_std)):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        self.x = np.zeros(self._dim)
        self.x[: 2 * n] = x0
        self.x[self._i_b] = float(bias0)
        if self.estimate_rate:
            self.x[self._i_d] = float(rate0)

        self.P = np.zeros((self._dim, self._dim))
        self.P[: 2 * n, : 2 * n] = P0
        self.P[self._i_b, self._i_b] = float(bias_std0) ** 2
        if self.estimate_rate:
            self.P[self._i_d, self._i_d] = float(rate_std0) ** 2

        self.bias_rw_std = float(bias_rw_std)
        self.rate_rw_std = float(rate_rw_std)
        self.t = float(t0)
        self.t_ref = float(t0)
        self.diag = OffsetDiagnostics()
        self.record_history = bool(record_history)
        self.history: list[tuple[float, float, float, float, float]] = []
        if self.record_history:
            self._record()

    # -- state accessors ---------------------------------------------------
    @property
    def state_dim(self) -> int:
        return self._dim

    @property
    def position(self) -> np.ndarray:
        return self.x[: self.n_dim].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.n_dim: 2 * self.n_dim].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[self.n_dim: 2 * self.n_dim]))

    @property
    def bias(self) -> float:
        """Estimated clock offset (s); positive = sensor stamp reads ahead."""
        return float(self.x[self._i_b])

    @property
    def bias_std(self) -> float:
        return float(np.sqrt(max(self.P[self._i_b, self._i_b], 0.0)))

    @property
    def rate(self) -> float:
        return float(self.x[self._i_d]) if self.estimate_rate else 0.0

    @property
    def rate_std(self) -> float:
        if not self.estimate_rate:
            return 0.0
        return float(np.sqrt(max(self.P[self._i_d, self._i_d], 0.0)))

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[: self.n_dim, : self.n_dim].copy()

    # -- clock helpers -----------------------------------------------------
    def true_time(self, t_rep: float) -> float:
        """Estimated true capture time of a sample stamped ``t_rep``."""
        b, d = self.bias, self.rate
        return (t_rep - b + d * self.t_ref) / (1.0 + d)

    def reported_time(self, t_true: float) -> float:
        """Inverse of :meth:`true_time` (what the sensor would have stamped)."""
        b, d = self.bias, self.rate
        return t_true + b + d * (t_true - self.t_ref)

    def offset_at(self, t: float | None = None) -> float:
        """Estimated staleness (s) of a sample stamped at ``t`` (default now)."""
        if t is None:
            t = self.t
        b, d = self.bias, self.rate
        return float((b + d * (t - self.t_ref)) / (1.0 + d))

    # -- propagation -------------------------------------------------------
    def predict_to(self, t: float) -> None:
        """Propagate the estimate to reference time ``t`` (no-op if equal)."""
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        if t < self.t - 1e-12:
            raise ValueError(f"cannot predict backwards ({self.t} -> {t})")
        if abs(t - self.t) <= 1e-12:
            return
        dt = t - self.t
        n = self.n_dim
        F = np.eye(self._dim)
        F[: 2 * n, : 2 * n] = self.model.F(dt)
        if self.estimate_rate:
            F[self._i_b, self._i_d] = dt  # b(t) = b + d * dt

        Q = np.zeros((self._dim, self._dim))
        Q[: 2 * n, : 2 * n] = self.model.Q(dt)
        q_b, q_d = self.bias_rw_std ** 2, self.rate_rw_std ** 2
        Q[self._i_b, self._i_b] = q_b * dt
        if self.estimate_rate:
            # offset integrates the (random-walk) rate error
            Q[self._i_b, self._i_b] += q_d * dt ** 3 / 3.0
            Q[self._i_b, self._i_d] = q_d * dt ** 2 / 2.0
            Q[self._i_d, self._i_b] = q_d * dt ** 2 / 2.0
            Q[self._i_d, self._i_d] = q_d * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)
        self.diag.n_predicts += 1

    def predict_position_to(self, t_future: float) -> tuple[np.ndarray, np.ndarray]:
        """CV-extrapolate the position to ``t_future`` without mutating the filter."""
        if not np.isfinite(t_future):
            raise ValueError("t_future must be finite")
        if t_future < self.t - 1e-12:
            raise ValueError("t_future must be >= current filter time")
        dt = t_future - self.t
        n = self.n_dim
        p = self.x[:n] + self.x[n: 2 * n] * dt
        Pp, Pv = self.P[:n, :n], self.P[n: 2 * n, n: 2 * n]
        Ppv = self.P[:n, n: 2 * n]
        cov = Pp + dt * (Ppv + Ppv.T) + dt * dt * Pv
        return p, cov

    # -- measurement ingestion --------------------------------------------
    def update(self, t_rep: float, z: np.ndarray, sigma: float | None = None) -> None:
        """Insert a position fix stamped ``t_rep`` on the *sensor* clock.

        ``z`` is the measured position (shape ``(n,)``); ``sigma`` overrides
        ``obs_sigma`` for this fix.

        The filter's clock is **not** advanced here.  The fix is evaluated at
        its estimated true time ``tau_hat = t_rep - offset_hat`` and inserted
        into the estimate with a linear motion correction; the clock is driven
        by the reference-clock channel (:meth:`update_velocity` / ``predict_to``).
        Letting the skewed stream advance the clock is a model error that makes
        the estimator self-deceiving -- see the module docstring.
        """
        if not np.isfinite(t_rep):
            raise ValueError("t_rep must be finite")
        z = np.atleast_1d(np.asarray(z, dtype=float))
        n = self.n_dim
        if z.shape != (n,):
            raise ValueError(f"z must have shape ({n},), got {z.shape}")
        if not np.all(np.isfinite(z)):
            raise ValueError("z must be finite")
        sig = self.obs_sigma if sigma is None else float(sigma)
        if not np.isfinite(sig) or sig <= 0.0:
            raise ValueError("sigma must be positive and finite")

        # Estimated *reference-clock* time of the fix: this is where the model
        # must be evaluated.  Parameter-dependent -- the Jacobians below carry
        # the dependency, which is the whole point of the estimator.
        tau = self.true_time(t_rep)
        if tau > self.t:
            self.diag.n_ahead += 1
        elif tau < self.t:
            self.diag.n_behind += 1

        b, d = self.bias, self.rate
        inv = 1.0 / (1.0 + d)
        dt = tau - self.t  # < 0 for a fix older than the reference clock
        p, v = self.x[:n], self.x[n: 2 * n]
        speed = float(np.linalg.norm(v))

        H = np.zeros((n, self._dim))
        H[:, :n] = np.eye(n)
        H[:, n: 2 * n] = np.eye(n) * dt
        # d(tau)/db = -1/(1+d), d(tau)/dd = (t_ref - t_rep + b)/(1+d)^2
        H[:, self._i_b] = -v * inv
        if self.estimate_rate:
            H[:, self._i_d] = v * (self.t_ref - t_rep + b) * inv * inv

        h = p + v * dt
        R = np.eye(n) * sig * sig
        y = z - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self._dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form

        self.diag.n_updates += 1
        if speed > _INFORMATIVE_SPEED_EPS:
            self.diag.n_informative += 1
        if self.record_history:
            self._record()

    def update_velocity(self, t_ref: float, z_v: np.ndarray,
                        sigma_v: float) -> None:
        """Fuse a velocity/speed fix on the **reference** clock.

        This is the independent motion reference (wheel odometry, vehicle CAN
        speed, IMU-derived velocity).  It carries no clock skew -- which is
        exactly why it is what makes the skewed position stream's offset
        observable (see the module docstring).

        Parameters
        ----------
        t_ref : float
            Reference-clock time of the fix.
        z_v : np.ndarray
            Measured velocity, shape ``(n,)``.
        sigma_v : float
            Isotropic velocity sigma (m/s).
        """
        if not np.isfinite(t_ref):
            raise ValueError("t_ref must be finite")
        n = self.n_dim
        z_v = np.atleast_1d(np.asarray(z_v, dtype=float))
        if z_v.shape != (n,):
            raise ValueError(f"z_v must have shape ({n},), got {z_v.shape}")
        if not np.all(np.isfinite(z_v)):
            raise ValueError("z_v must be finite")
        sig = float(sigma_v)
        if not np.isfinite(sig) or sig <= 0.0:
            raise ValueError("sigma_v must be positive and finite")

        if t_ref > self.t + 1e-12:
            self.predict_to(t_ref)
        elif t_ref < self.t - 1e-12:
            self.diag.n_behind += 1

        H = np.zeros((n, self._dim))
        H[:, n: 2 * n] = np.eye(n)
        h = self.x[n: 2 * n].copy()
        R = np.eye(n) * sig * sig
        y = z_v - h
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self._dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.diag.n_velocity_updates += 1
        if self.record_history:
            self._record()

    # -- internals ---------------------------------------------------------
    def _record(self) -> None:
        self.history.append((self.t, self.bias, self.bias_std, self.rate, self.rate_std))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"OffsetAugmentedFilter(t={self.t:.3f}s, "
                f"bias={self.bias * 1e3:+.2f}+-{self.bias_std * 1e3:.2f} ms, "
                f"rate={self.rate * 1e6:+.2f} ppm)")
