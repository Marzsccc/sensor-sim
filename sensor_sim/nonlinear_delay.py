"""Nonlinear (ESKF) out-of-sequence / delayed measurement fusion -- v0.23.0.

v0.22.0 solved *delayed measurements* for a **linear time-invariant** filter:
keep a state history, and when a late sample arrives, rewind to the state at or
before its measurement time, apply it, and replay every later measurement.
That trick leans on linearity (``F`` / ``H`` are matrices, updates compose).

A GNSS+IMU filter is not linear: the mechanisation involves quaternion
products and a body/world rotation, and the covariance propagates through a
*linearised* error-state model.  The same idea still works, but the replay
needs one extra ingredient: **the raw inertial samples**.

Why it works
------------
The ESKF's nominal-state propagation is a *deterministic function* of the IMU
sample sequence.  Snapshot the full state (nominal + error covariance) at an
IMU boundary, and re-feeding the identical IMU samples in the identical order
reproduces the identical trajectory of states -- down to floating-point
round-off, not merely "to first order".  Late GNSS measurements are then
inserted in validity-time order along that replayed timeline, which gives
exactly the estimate an in-order pipeline would have produced.

So the design is:

    snapshot  = (t, p, v, q, b_a, b_g, P, n_updates)   at every IMU boundary
    buffers   = IMU samples + GNSS measurements, bounded by ``history_s``

    late GNSS  -> anchor = newest snapshot with t <= t_meas
              -> replay IMU (mechanisation) interleaved with the pending
                 GNSS updates, in validity-time order
              -> state time is restored to where it was

Only the *covariance* linearisation is approximate (it always was); the
mechanisation itself is replayed exactly.

Clock semantics (same contract as v0.22.0)
------------------------------------------
``t`` is the newest integrated (validity) time, not the wall clock:

* A GNSS fix whose validity time is **ahead** of the current state is applied
  immediately (nothing can be integrated without an IMU sample, and it is
  early rather than late).
* A GNSS fix whose validity time is **behind** the current state is out of
  sequence: rewind + replay.  It corrects the past without dragging the
  clock back.
* Display time is handled separately by ``extrapolate_to`` -- the AR-HUD
  "render at t_display" half of delay compensation.  A constantly delayed but
  monotone stream therefore never rewinds at all; the residual latency is
  removed at the render instant.

Two modes for A/B measurement:
  - ``mode="oosm"``         : measurement-time fusion with rewind + replay.
  - ``mode="receive_time"`` : naive baseline, fuse at arrival.  The state is
    stamped with the IMU clock but describes a scene one GNSS latency old --
    the "marker trails the world" bug, in its GNSS-aided flavour.

Reference: Bar-Shalom, "Update with out-of-sequence measurements in tracking:
exact solution" (IEEE TAES 2002); Larsen et al. (IEEE TAES 2000); Solà,
"Quaternion kinematics for the error-state Kalman filter" (arXiv 1711.02508).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .eskf import ESKF, ESKFConfig


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ImuSample:
    """One inertial sample in the body frame."""

    t: float
    acc: np.ndarray
    gyro: np.ndarray

    def __post_init__(self) -> None:
        self.t = float(self.t)
        self.acc = np.asarray(self.acc, dtype=float)
        self.gyro = np.asarray(self.gyro, dtype=float)
        if self.acc.shape != (3,) or self.gyro.shape != (3,):
            raise ValueError("acc and gyro must have shape (3,)")


@dataclass
class GnssMeasurement:
    """A GNSS fix with a measurement (validity) time and an arrival time."""

    t_meas: float
    pos: np.ndarray
    vel: Optional[np.ndarray] = None
    pos_std: Optional[float] = None
    vel_std: Optional[float] = None
    t_recv: Optional[float] = None

    def __post_init__(self) -> None:
        self.t_meas = float(self.t_meas)
        self.pos = np.asarray(self.pos, dtype=float)
        if self.pos.shape != (3,):
            raise ValueError("pos must have shape (3,)")
        if self.vel is not None:
            self.vel = np.asarray(self.vel, dtype=float)
            if self.vel.shape != (3,):
                raise ValueError("vel must have shape (3,)")
        if self.t_recv is not None:
            self.t_recv = float(self.t_recv)
            if self.t_recv < self.t_meas:
                raise ValueError(
                    f"receive time {self.t_recv} precedes measurement time {self.t_meas}"
                )


@dataclass
class EskfSnapshot:
    """A complete ESKF state captured at an IMU boundary."""

    t: float
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    b_a: np.ndarray
    b_g: np.ndarray
    P: np.ndarray
    n_updates: int


@dataclass
class RewindDiagnostics:
    """Counters describing what the filter did -- handy for tests / reports."""

    n_imu: int = 0
    n_gnss: int = 0
    n_applied: int = 0
    n_out_of_order: int = 0
    n_rewinds: int = 0
    n_reprocessed: int = 0
    n_imu_replayed: int = 0
    max_rewind_s: float = 0.0
    n_pruned: int = 0
    n_clamped: int = 0

    def as_dict(self) -> dict:
        return dict(
            n_imu=self.n_imu,
            n_gnss=self.n_gnss,
            n_applied=self.n_applied,
            n_out_of_order=self.n_out_of_order,
            n_rewinds=self.n_rewinds,
            n_reprocessed=self.n_reprocessed,
            n_imu_replayed=self.n_imu_replayed,
            max_rewind_s=self.max_rewind_s,
            n_pruned=self.n_pruned,
            n_clamped=self.n_clamped,
        )


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def capture_state(filt: ESKF, t: Optional[float] = None) -> EskfSnapshot:
    """Copy the full filter state into a plain (restorable) snapshot."""
    st = filt.state
    return EskfSnapshot(
        t=float(st.t if t is None else t),
        p=st.p.copy(),
        v=st.v.copy(),
        q=st.q.copy(),
        b_a=st.b_a.copy(),
        b_g=st.b_g.copy(),
        P=st.P.copy(),
        n_updates=int(st.n_updates),
    )


def restore_state(filt: ESKF, snap: EskfSnapshot) -> None:
    """Write a snapshot back into the filter (in place)."""
    st = filt.state
    st.t = snap.t
    st.p = snap.p.copy()
    st.v = snap.v.copy()
    st.q = snap.q.copy()
    st.b_a = snap.b_a.copy()
    st.b_g = snap.b_g.copy()
    st.P = snap.P.copy()
    st.n_updates = snap.n_updates


def position_nees(err: np.ndarray, P_pos: np.ndarray) -> float:
    """Position NEES for a 3-dof error and its 3x3 covariance."""
    e = np.asarray(err, dtype=float)
    S = 0.5 * (np.asarray(P_pos, dtype=float) + np.asarray(P_pos, dtype=float).T)
    return float(e @ np.linalg.pinv(S) @ e)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class RewindReplayESKF:
    """ESKF wrapper adding measurement-time fusion with rewind + replay.

    Parameters
    ----------
    cfg : ESKFConfig, optional
        Passed straight to the underlying :class:`~sensor_sim.eskf.ESKF`.
    mode : {"oosm", "receive_time"}
        ``"oosm"`` = measurement-time fusion (correct);
        ``"receive_time"`` = fuse on arrival (naive baseline).
    history_s : float
        How far back snapshots / IMU samples / GNSS records are kept for
        reprocessing.  A fix older than ``t_now - history_s`` cannot be
        honoured exactly; it is clamped to the oldest retained state and
        counted in ``diag.n_clamped``.
    """

    def __init__(
        self,
        cfg: Optional[ESKFConfig] = None,
        mode: str = "oosm",
        history_s: float = 5.0,
    ) -> None:
        if mode not in ("oosm", "receive_time"):
            raise ValueError("mode must be 'oosm' or 'receive_time'")
        self.filt = ESKF(cfg)
        self.mode = mode
        self.history_s = float(history_s)

        self._snaps: List[EskfSnapshot] = []
        self._imu: List[ImuSample] = []
        self._gnss: List[GnssMeasurement] = []
        self.diag = RewindDiagnostics()

    # -- setup -------------------------------------------------------------
    def initialize(
        self,
        t: float,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        b_a: Optional[np.ndarray] = None,
        b_g: Optional[np.ndarray] = None,
    ) -> "RewindReplayESKF":
        """Seed the filter and open the history with a first snapshot."""
        self.filt.set_initial_state(t, p, v, q, b_a=b_a, b_g=b_g)
        self._snaps = [capture_state(self.filt, t)]
        return self

    # -- inertial input ----------------------------------------------------
    def add_imu(self, t: float, acc: np.ndarray, gyro: np.ndarray) -> "RewindReplayESKF":
        """Integrate one IMU sample and snapshot the resulting state."""
        t = float(t)
        dt = t - self.filt.state.t
        if dt < -1e-12:
            raise ValueError(
                f"IMU sample at {t} precedes current state time {self.filt.state.t}; "
                "inertial samples must be monotone"
            )
        if dt > 1e-12:
            self.filt.predict(acc, gyro, dt)
        self._imu.append(ImuSample(t, acc, gyro))
        if dt > 1e-12:
            self._snaps.append(capture_state(self.filt, t))
        self.diag.n_imu += 1
        self._prune()
        return self

    # -- aiding input ------------------------------------------------------
    def add_gnss(
        self,
        t_meas: float,
        pos: np.ndarray,
        vel: Optional[np.ndarray] = None,
        t_recv: Optional[float] = None,
        pos_std: Optional[float] = None,
        vel_std: Optional[float] = None,
    ) -> "RewindReplayESKF":
        """Ingest a GNSS fix, rewinding + replaying if it is out of sequence."""
        m = GnssMeasurement(t_meas, pos, vel, pos_std, vel_std, t_recv)
        self.diag.n_gnss += 1
        t_now = self.filt.state.t

        if self.mode == "receive_time":
            self.filt.update_gnss(m.pos, m.vel, m.pos_std, m.vel_std)
            self.diag.n_applied += 1
        elif m.t_meas < t_now - 1e-9:
            self.diag.n_out_of_order += 1
            self.diag.max_rewind_s = max(self.diag.max_rewind_s, t_now - m.t_meas)
            self._rewind_and_replay(m)
        else:
            self.filt.update_gnss(m.pos, m.vel, m.pos_std, m.vel_std)
            self.diag.n_applied += 1

        self._gnss.append(m)
        self._prune()
        return self

    # -- display-time extrapolation ----------------------------------------
    def extrapolate_to(
        self, t_future: float, accel_psd: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constant-velocity extrapolation to a future display instant.

        Returns ``(p, P_pos)`` -- the pose to render and its 3x3 position
        covariance, grown by a white-acceleration process of density
        ``accel_psd`` (m^2/s^5).  Does not mutate the filter.

        ``t_future`` must be finite: a display instant of ``inf`` (a caller
        that forgot to check whether more data was coming) would otherwise
        turn into a NaN pose through ``0 * inf`` in the velocity term.
        """
        t_future = float(t_future)
        if not np.isfinite(t_future):
            raise ValueError(f"t_future must be finite, got {t_future}")
        if not np.isfinite(self.filt.state.t):
            raise ValueError("filter state time is not finite")
        dt = t_future - self.filt.state.t
        if dt < -1e-12:
            raise ValueError("t_future must be >= current filter time")
        st = self.filt.state
        p = st.p + st.v * max(dt, 0.0)

        P6 = st.P[:6, :6]
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * max(dt, 0.0)
        q = float(accel_psd)
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt ** 3 / 3.0
        Q[:3, 3:] = np.eye(3) * q * dt ** 2 / 2.0
        Q[3:, :3] = np.eye(3) * q * dt ** 2 / 2.0
        Q[3:, 3:] = np.eye(3) * q * dt
        P6n = F @ P6 @ F.T + Q
        return p, 0.5 * (P6n[:3, :3] + P6n[:3, :3].T)

    # -- internals ---------------------------------------------------------
    def _rewind_and_replay(self, new_m: GnssMeasurement) -> None:
        """Anchor at ``<= new_m.t_meas``, replay IMU + pending GNSS forward."""
        anchor = None
        for s in self._snaps:
            if s.t <= new_m.t_meas + 1e-9:
                anchor = s
        if anchor is None:
            # History does not reach that far back: clamp to the oldest state
            # we still have and record it (better a slightly late correction
            # than dropping the fix or corrupting the timeline).
            anchor = self._snaps[0]
            self.diag.n_clamped += 1

        t_now = self.filt.state.t
        prefix = [s for s in self._snaps if s.t < anchor.t - 1e-9]

        pending_gnss = [m for m in self._gnss if m.t_meas > anchor.t + 1e-9]
        pending_gnss.append(new_m)
        pending_gnss.sort(key=lambda mm: mm.t_meas)

        pending_imu = [s for s in self._imu if s.t > anchor.t + 1e-9]

        restore_state(self.filt, anchor)
        snaps: List[EskfSnapshot] = [capture_state(self.filt, anchor.t)]

        i = j = 0
        while i < len(pending_imu) or j < len(pending_gnss):
            t_imu = pending_imu[i].t if i < len(pending_imu) else np.inf
            t_g = pending_gnss[j].t_meas if j < len(pending_gnss) else np.inf
            # IMU first on ties: a fix stamped exactly at an IMU boundary is
            # applied *after* that boundary's mechanisation, which is what an
            # in-order pipeline does.
            if t_imu <= t_g + 1e-12:
                s = pending_imu[i]
                self.filt.predict(s.acc, s.gyro, s.t - self.filt.state.t)
                snaps.append(capture_state(self.filt, s.t))
                self.diag.n_imu_replayed += 1
                i += 1
            else:
                g = pending_gnss[j]
                self.filt.update_gnss(g.pos, g.vel, g.pos_std, g.vel_std)
                self.diag.n_reprocessed += 1
                j += 1

        # The replayed timeline must land exactly where the filter already was.
        if abs(self.filt.state.t - t_now) > 1e-9:
            raise RuntimeError(
                f"replay ended at t={self.filt.state.t}, expected {t_now}"
            )

        self.diag.n_rewinds += 1
        self._snaps = prefix + snaps

    def _prune(self) -> None:
        t_now = self.filt.state.t
        horizon = t_now - self.history_s

        keep_s = [s for s in self._snaps if s.t >= horizon - 1e-9]
        if not keep_s:
            keep_s = [self._snaps[-1]]
        removed = len(self._snaps) - len(keep_s)
        self.diag.n_pruned += max(removed, 0)
        self._snaps = keep_s

        keep_i = [s for s in self._imu if s.t >= horizon - 1e-9]
        self._imu = keep_i

        keep_g = [m for m in self._gnss if m.t_meas >= horizon - 1e-9]
        self._gnss = keep_g

    # -- read-outs ---------------------------------------------------------
    @property
    def time(self) -> float:
        return self.filt.state.t

    @property
    def position(self) -> np.ndarray:
        return self.filt.state.p

    @property
    def velocity(self) -> np.ndarray:
        return self.filt.state.v

    @property
    def quaternion(self) -> np.ndarray:
        return self.filt.state.q

    @property
    def rotation_matrix(self) -> np.ndarray:
        from .utils import quat_to_rotmat

        return quat_to_rotmat(self.filt.state.q)

    @property
    def position_covariance(self) -> np.ndarray:
        P = self.filt.state.P
        return 0.5 * (P[:3, :3] + P[:3, :3].T)

    def estimate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.filt.state.t, self.filt.state.p.copy(), self.filt.state.P.copy()

    def history(self) -> Tuple[int, int, int]:
        """``(n_snapshots, n_imu_buffered, n_gnss_buffered)``."""
        return len(self._snaps), len(self._imu), len(self._gnss)
