"""Out-of-sequence (delayed) measurement fusion -- v0.22.0.

Real perception stacks deliver measurements *later* than they were captured
and sometimes *out of order* (camera 60 ms, GNSS 100 ms, radar 30 ms, wheel
odometry 5 ms).  A filter that fuses every measurement at its *receive* time
silently injects that latency into the state estimate -- the classic AR-HUD
"marker lags the world" symptom.

This module implements the principled fix (measurement-time fusion with
out-of-order reprocessing):

  1. ``DelayedMeasurement`` -- a measurement with both a *measurement* time
     and a *receive* time, plus its linear observation model ``(H, R)``.
  2. ``CVModel`` -- a time-invariant constant-velocity Gaussian model
     (arbitrary position dimension, scalar acceleration PSD), providing
     ``F(dt)`` / ``Q(dt)``.
  3. ``DelayedFusionFilter`` -- a linear Kalman filter that keeps a bounded
     history of states + measurements.  A delayed / out-of-order measurement
     is *inserted into the past*: the filter rewinds to the nearest stored
     state at or before ``t_meas``, applies the measurement, and replays every
     later measurement forward.  For a linear time-invariant system this
     reproduces exactly the estimate that an in-order filter would have had
     (see ``test_delay_fusion.py::TestExactness``).
  4. ``predict_to(t_future)`` -- propagate the corrected estimate to a future
     (display) time, i.e. the forward half of AR-HUD delay compensation.

Clock semantics (important)
---------------------------
``DelayedFusionFilter.t`` is the **newest fused measurement (validity) time**,
not the host wall clock.  Consequences:

* A stream that is *delayed but monotone* in validity time is **in sequence**
  -- it is fused forward as usual and never rewinds.  The state then honestly
  carries the sensor timestamp, and the residual latency is removed at the
  display instant by ``predict_to_time`` (this is the AR-HUD split: fuse at
  measurement time, render at display time).
* A sample is **out of sequence** only when its validity time precedes the
  newest fused time (typical for interleaved multi-rate sensors: a 60 ms
  camera frame lands after newer 30 ms radar / 5 ms odometry frames).
* After a rewind, ``t`` is restored to where it was -- a late sample corrects
  the past, it does not drag the clock back.

Two modes are supported so a pipeline can be A/B measured:
  - ``mode="oosm"``        : fuse at measurement time, reprocess out-of-order.
  - ``mode="receive_time"``: naive baseline -- fuse at receive time (the
    state is stamped *now*, but describes a scene that is already one sensor
    latency old; measured as a large NEES / RMSE penalty in the demo).

Reference: Bar-Shalom, "Update with out-of-sequence measurements in tracking:
exact solution" (IEEE TAES 2002); Larsen et al., "Incorporating time-delayed
measurements in the Kalman filter" (IEEE TAES 2000).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class DelayedMeasurement:
    """A measurement stamped with both measurement time and receive time.

    Parameters
    ----------
    t_meas : float
        Time the quantity was *true* (capture / validity time).
    t_recv : float
        Host time the sample reached the filter.
    z : np.ndarray
        Observation vector, shape ``(m,)``.
    H : np.ndarray
        Linear observation matrix, shape ``(m, 2n)``.
    R : np.ndarray
        Observation noise covariance, shape ``(m, m)``.
    sensor : str
        Optional source label (for diagnostics).
    """

    t_meas: float
    t_recv: float
    z: np.ndarray
    H: np.ndarray
    R: np.ndarray
    sensor: str = ""

    def __post_init__(self) -> None:
        self.z = np.atleast_1d(np.asarray(self.z, dtype=float))
        self.H = np.asarray(self.H, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        if self.t_recv < self.t_meas:
            raise ValueError(
                f"receive time {self.t_recv} precedes measurement time {self.t_meas}"
            )


# ---------------------------------------------------------------------------
# Time-invariant linear model
# ---------------------------------------------------------------------------

class CVModel:
    """Constant-velocity model over an ``n``-dim position.

    State ``x = [p_1..p_n, v_1..v_n]`` (dimension ``2n``).  Dynamics
    ``F(dt)`` and process noise ``Q(dt)`` for a continuous white-acceleration
    model with spectral density ``q`` (per axis).

    ``F_fn`` / ``Q_fn`` may be supplied to swap in a different (still
    time-invariant) model; ``dt`` is passed through unchanged.
    """

    def __init__(
        self,
        n_dim: int = 2,
        accel_psd: float = 1.0,
        F_fn: Optional[Callable[[float], np.ndarray]] = None,
        Q_fn: Optional[Callable[[float], np.ndarray]] = None,
    ) -> None:
        if n_dim < 1:
            raise ValueError("n_dim must be >= 1")
        self.n_dim = int(n_dim)
        self.accel_psd = float(accel_psd)
        self._F_fn = F_fn
        self._Q_fn = Q_fn

    @property
    def state_dim(self) -> int:
        return 2 * self.n_dim

    def F(self, dt: float) -> np.ndarray:
        if self._F_fn is not None:
            return np.asarray(self._F_fn(dt), dtype=float)
        n = self.n_dim
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt
        return F

    def Q(self, dt: float) -> np.ndarray:
        if self._Q_fn is not None:
            return np.asarray(self._Q_fn(dt), dtype=float)
        n, q, dt2 = self.n_dim, self.accel_psd, dt * dt
        Q = np.zeros((2 * n, 2 * n))
        blk_pp = q * dt2 * dt / 3.0
        blk_pv = q * dt2 / 2.0
        blk_vv = q * dt
        Q[:n, :n] = np.eye(n) * blk_pp
        Q[:n, n:] = np.eye(n) * blk_pv
        Q[n:, :n] = np.eye(n) * blk_pv
        Q[n:, n:] = np.eye(n) * blk_vv
        return Q

    def observe_position(self, sigma: float, n_obs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Position-only observation model with isotropic ``sigma`` (m)."""
        k = self.n_dim if n_obs is None else int(n_obs)
        H = np.zeros((k, self.state_dim))
        H[:k, :k] = np.eye(k)
        R = np.eye(k) * sigma * sigma
        return H, R


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    t: float
    x: np.ndarray
    P: np.ndarray


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

@dataclass
class FusionDiagnostics:
    """Counters describing what the filter did -- handy for tests / reports."""

    n_accepted: int = 0
    n_received: int = 0
    n_out_of_order: int = 0
    n_reprocessed: int = 0
    n_pruned: int = 0
    max_rewind_s: float = 0.0

    def as_dict(self) -> dict:
        return dict(
            n_accepted=self.n_accepted,
            n_received=self.n_received,
            n_out_of_order=self.n_out_of_order,
            n_reprocessed=self.n_reprocessed,
            n_pruned=self.n_pruned,
            max_rewind_s=self.max_rewind_s,
        )


class DelayedFusionFilter:
    """Linear KF with measurement-time fusion and out-of-order reprocessing.

    Parameters
    ----------
    model : CVModel
        Time-invariant linear model providing ``F`` / ``Q``.
    x0, P0 : np.ndarray
        Initial state and covariance (at ``t0``).
    t0 : float
        Initial time.
    mode : {"oosm", "receive_time"}
        ``"oosm"`` fuses at measurement time (correct); ``"receive_time"``
        fuses on arrival (naive baseline for comparison).
    history_s : float
        How far back the state/measurement history is kept for reprocessing.
        Measurements older than ``t_now - history_s`` are pruned.
    """

    def __init__(
        self,
        model: CVModel,
        x0: np.ndarray,
        P0: np.ndarray,
        t0: float = 0.0,
        mode: str = "oosm",
        history_s: float = 5.0,
    ) -> None:
        if mode not in ("oosm", "receive_time"):
            raise ValueError("mode must be 'oosm' or 'receive_time'")
        self.model = model
        self.mode = mode
        self.history_s = float(history_s)

        self.t = float(t0)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        if self.x.shape != (model.state_dim,):
            raise ValueError(f"x0 must have shape ({model.state_dim},)")
        if self.P.shape != (model.state_dim, model.state_dim):
            raise ValueError(f"P0 must have shape ({model.state_dim}, {model.state_dim})")

        self._snapshots: List[_Snapshot] = [_Snapshot(self.t, self.x.copy(), self.P.copy())]
        self._measurements: List[DelayedMeasurement] = []
        self.diag = FusionDiagnostics()

    # -- propagation -------------------------------------------------------
    def predict_to(self, t: float) -> None:
        """Propagate the estimate forward to time ``t`` (no-op if not later)."""
        if t < self.t - 1e-12:
            raise ValueError(f"cannot predict backwards ({self.t} -> {t}); "
                             "delayed measurements must go through add_measurement")
        if abs(t - self.t) <= 1e-12:
            return
        dt = t - self.t
        F, Q = self.model.F(dt), self.model.Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t)
        self._snapshots.append(_Snapshot(self.t, self.x.copy(), self.P.copy()))
        self._prune()

    def predict_to_time(self, t_future: float) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate at a *future* time without mutating the filter (display time)."""
        if t_future < self.t:
            raise ValueError("t_future must be >= current filter time")
        dt = t_future - self.t
        F, Q = self.model.F(dt), self.model.Q(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    # -- measurement handling ---------------------------------------------
    def add_measurement(self, m: DelayedMeasurement) -> None:
        """Ingest a measurement, reprocessing history if it is out of order."""
        self.diag.n_received += 1

        if self.mode == "receive_time":
            # Naive baseline: pretend the value is valid now.
            if m.t_recv > self.t + 1e-12:
                self.predict_to(m.t_recv)
            self._update(m)
            self.diag.n_accepted += 1
            self._prune()
            return

        t_eff = m.t_meas
        if t_eff < self.t - 1e-9:
            # Out of sequence: rewind & replay.
            self.diag.n_out_of_order += 1
            self.diag.max_rewind_s = max(self.diag.max_rewind_s, self.t - t_eff)
            self._reprocess(m)
            self.diag.n_accepted += 1
        else:
            if t_eff > self.t + 1e-12:
                self.predict_to(t_eff)
            self._update(m)
            self.diag.n_accepted += 1
            self._prune()
        self._measurements.append(m)
        self._prune()

    # -- internals ---------------------------------------------------------
    def _update(self, m: DelayedMeasurement) -> None:
        S = m.H @ self.P @ m.H.T + m.R
        K = self.P @ m.H.T @ np.linalg.pinv(S)
        innovation = m.z - m.H @ self.x
        self.x = self.x + K @ innovation
        I_KH = np.eye(self.model.state_dim) - K @ m.H
        # Joseph form for numerical symmetry / PSD safety.
        self.P = I_KH @ self.P @ I_KH.T + K @ m.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        if self._snapshots and abs(self._snapshots[-1].t - self.t) <= 1e-12:
            self._snapshots[-1] = _Snapshot(self.t, self.x.copy(), self.P.copy())
        else:
            self._snapshots.append(_Snapshot(self.t, self.x.copy(), self.P.copy()))

    def _reprocess(self, new_m: DelayedMeasurement) -> None:
        """Rewind to ``<= new_m.t_meas`` and replay all later measurements."""
        anchor = self._snapshot_at_or_before(new_m.t_meas)
        if anchor is None:
            raise RuntimeError("no state history far enough back to reprocess")
        t_now = self.t

        pending = [m for m in self._measurements if m.t_meas > anchor.t + 1e-12]
        pending.append(new_m)
        pending.sort(key=lambda mm: mm.t_meas)

        # Snapshots strictly *before* the anchor are unaffected by the replay
        # and must be carried over.  Dropping them would amputate the history
        # on every out-of-order sample, so a later rewind would eventually
        # find no state to rewind to.
        prefix = [s for s in self._snapshots if s.t < anchor.t - 1e-9]

        # Rewind working state to the anchor.
        t_w, x_w, P_w = anchor.t, anchor.x.copy(), anchor.P.copy()
        snapshots: List[_Snapshot] = [_Snapshot(t_w, x_w.copy(), P_w.copy())]

        def _step(t_target: float) -> None:
            nonlocal t_w, x_w, P_w
            if t_target > t_w + 1e-12:
                dt = t_target - t_w
                F, Q = self.model.F(dt), self.model.Q(dt)
                x_w = F @ x_w
                P_w = F @ P_w @ F.T + Q
                t_w = t_target
                snapshots.append(_Snapshot(t_w, x_w.copy(), P_w.copy()))

        for mm in pending:
            _step(mm.t_meas)
            self.t, self.x, self.P = t_w, x_w, P_w
            self._snapshots = prefix + snapshots
            self._update(mm)
            x_w, P_w = self.x.copy(), self.P.copy()
            snapshots[-1] = _Snapshot(t_w, x_w.copy(), P_w.copy())
            self.diag.n_reprocessed += 1

        # Re-advance to where the filter previously was.
        _step(t_now)
        self.t, self.x, self.P = t_w, x_w, P_w
        self._snapshots = prefix + snapshots

    def _snapshot_at_or_before(self, t: float) -> Optional[_Snapshot]:
        best: Optional[_Snapshot] = None
        for s in self._snapshots:
            if s.t <= t + 1e-9:
                best = s
        return best

    def _prune(self) -> None:
        horizon = self.t - self.history_s
        keep = [s for s in self._snapshots if s.t >= horizon - 1e-9]
        removed = len(self._snapshots) - len(keep)
        if removed > 0:
            # Never drop the very latest snapshot.
            if keep and keep[-1].t < self._snapshots[-1].t:
                keep.append(self._snapshots[-1])
            if not keep:
                keep = [self._snapshots[-1]]
            self._snapshots = keep
            self.diag.n_pruned += removed
        n_before = len(self._measurements)
        self._measurements = [m for m in self._measurements if m.t_meas >= horizon - 1e-9]
        self.diag.n_pruned += n_before - len(self._measurements)

    # -- read-outs ---------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[: self.model.n_dim]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.model.n_dim :]

    def estimate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.t, self.x.copy(), self.P.copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def position_covariance(P: np.ndarray, n_dim: int) -> np.ndarray:
    """Extract the position block of a covariance matrix."""
    return np.asarray(P, dtype=float)[:n_dim, :n_dim]


def nees_position(x: np.ndarray, P: np.ndarray, truth_p: np.ndarray, n_dim: int) -> float:
    """Position-only NEES for one sample (``2n`` dof if measured each step)."""
    err = np.asarray(x[:n_dim], dtype=float) - np.asarray(truth_p, dtype=float)
    S = position_covariance(P, n_dim)
    return float(err @ np.linalg.pinv(S) @ err)
