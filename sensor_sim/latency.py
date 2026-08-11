"""
Sensor latency & time synchronization model.

Models the timing imperfections found in real perception/state-estimation
pipelines (directly relevant to AR-HUD delay compensation):

  1. LatencyModel  -- fixed + jittered transport/computation delay per sensor,
                     with optional packet loss (dropout) and stale-timestamping.
  2. SensorClock   -- each sensor runs on its own clock with a constant rate
                     error (ppm) plus random walk; models oscillator drift.
  3. TimeSync      -- align a drifting sensor clock to the reference clock using
                     a simple offset + rate (clock skew) estimate, and compute
                     the *effective* delay seen by a fusion filter that fuses
                     measurements by receive-time.

Why this matters:
  - A camera at 30 Hz with 50 ms pipeline delay reports the scene as it was
    1.5 frames ago; fusing it at receive-time without compensation injects
    exactly the delay into the state estimate.
  - GNSS has a similar (larger) latency; wheel odometry is usually the fastest.
  - AR-HUD systems must predict the vehicle pose at *display time*, so
    estimating & compensating sensor latency is a core calibration step.

Reference: Groves (2013) ch. 9, "INS/GNSS integration"; IEEE 1588 clock sync.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Latency model
# ---------------------------------------------------------------------------

@dataclass
class LatencyModel:
    """Per-sensor delay: fixed + jitter, optional loss and stale timestamps.

    Parameters
    ----------
    fixed_delay_s : float
        Constant pipeline delay (capture -> output), e.g. 0.05 s camera.
    jitter_std_s : float
        Std dev of zero-mean Gaussian jitter added on top.
    loss_prob : float
        Probability that a measurement is dropped entirely (0..1).
    stale_ts_prob : float
        Probability that the sensor stamps the sample with the *previous*
        sample time (models poorly implemented drivers that reuse old
        timestamps).
    seed : int | None
        RNG seed for reproducibility.
    """

    fixed_delay_s: float = 0.0
    jitter_std_s: float = 0.0
    loss_prob: float = 0.0
    stale_ts_prob: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def deliver(self, sample_time: float) -> Tuple[float, float, bool]:
        """Return (receive_time, timestamp_used_by_filter, valid).

        - `receive_time` is the host-clock time the sample arrives.
        - `timestamp_used_by_filter` is what a naive filter would use: either
          the true sample time or a stale (previous) one.
        - `valid=False` means the sample was dropped (loss).
        """
        if self.loss_prob > 0.0 and self._rng.random() < self.loss_prob:
            return sample_time, sample_time, False

        jitter = self._rng.normal(0.0, self.jitter_std_s) if self.jitter_std_s > 0 else 0.0
        receive_time = sample_time + self.fixed_delay_s + jitter

        if self.stale_ts_prob > 0.0 and self._rng.random() < self.stale_ts_prob:
            ts = sample_time - self._sample_interval  # previous sample time
        else:
            ts = sample_time
        return receive_time, ts, True

    @property
    def _sample_interval(self) -> float:
        # Default 1/30 s (camera rate); callers may override via `sample_interval`.
        return getattr(self, "sample_interval", 1.0 / 30.0)

    def expected_delay(self) -> float:
        """Mean delay a measurement experiences (excluding loss)."""
        return self.fixed_delay_s

    def __repr__(self) -> str:
        return (f"LatencyModel(fixed={self.fixed_delay_s*1e3:.1f}ms, "
                f"jitter={self.jitter_std_s*1e3:.1f}ms, "
                f"loss={self.loss_prob:.0%})")


# ---------------------------------------------------------------------------
# 2. Sensor clock (oscillator drift)
# ---------------------------------------------------------------------------

@dataclass
class SensorClock:
    """A clock that drifts relative to the reference (host) clock.

    Models: constant rate error (ppm, e.g. a 20 ppm crystal) + a slowly
    wandering rate (random walk). Converts between host time and sensor time.
    """

    rate_ppm: float = 20.0          # steady rate error in parts-per-million
    wander_ppm_s: float = 1.0       # random-walk intensity of rate (ppm per sqrt(s))
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._rate = self.rate_ppm * 1e-6

    def sensor_time(self, host_t: float) -> float:
        """Host time -> sensor clock reading (drifting)."""
        # Rate error accumulates; wander is a Brownian motion on the rate.
        # For short horizons treat rate as piecewise-constant per call.
        wander = self._rng.normal(0.0, self.wander_ppm_s * 1e-6)
        return host_t * (1.0 + self._rate + wander)

    def host_time(self, sensor_t: float) -> float:
        """Sensor clock reading -> approximate host time (inverse)."""
        # Same inverse model; used for coarse alignment before TimeSync.
        return sensor_t / (1.0 + self._rate)


# ---------------------------------------------------------------------------
# 3. Time synchronization / effective delay estimation
# ---------------------------------------------------------------------------

@dataclass
class TimeSync:
    """Estimate offset & skew of a sensor clock w.r.t. the host clock.

    Uses a simple linear regression on (host_time, sensor_time) pairs, which
    is the offline equivalent of IEEE 1588 / PTP's clock servo (offset +
    rate). From the fitted line we can:

      - convert sensor timestamps to host time,
      - compute the *effective* receive-time delay seen by a filter that
        fuses on arrival.
    """

    samples: list = field(default_factory=list)  # list of (host_t, sensor_t)

    def add_pair(self, host_t: float, sensor_t: float) -> None:
        self.samples.append((host_t, sensor_t))

    def fit(self) -> Tuple[float, float, float]:
        """Return (offset, rate, r2).

        offset = sensor_time - host_time at host_t=0 (s)
        rate   = d(sensor_time)/d(host_time) - 1  (unitless, ppm*1e-6)
        """
        if len(self.samples) < 2:
            raise ValueError("Need >= 2 (host, sensor) pairs to fit clock model.")
        h = np.array([s[0] for s in self.samples])
        s = np.array([s[1] for s in self.samples])
        A = np.vstack([np.ones_like(h), h]).T
        (offset, slope), *_ = np.linalg.lstsq(A, s, rcond=None)
        # slope = 1 + rate
        pred = offset + slope * h
        ss_res = float(np.sum((s - pred) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return float(offset), float(slope - 1.0), r2

    def to_host(self, sensor_t: float) -> float:
        """Convert a sensor timestamp to host time using the fitted model."""
        offset, rate, _ = self.fit()
        return (sensor_t - offset) / (1.0 + rate)

    def effective_delay(self, receive_time: float, sensor_ts: float) -> float:
        """Delay a filter *actually* sees when fusing on receive-time.

        If the filter uses the sensor timestamp it can compensate; if it
        fuses on arrival the age of the data is `receive_time - true_sample_time`.
        """
        host_sample_time = self.to_host(sensor_ts)
        return receive_time - host_sample_time


# ---------------------------------------------------------------------------
# Convenience: run a full latency scenario
# ---------------------------------------------------------------------------

@dataclass
class LatencyScenario:
    """Bundle of sensors with per-sensor latency models + a shared scenario."""

    sensors: dict = field(default_factory=dict)   # name -> LatencyModel
    dt: float = 0.01                              # simulation step (s)

    def simulate(self, traj, host_clock: Optional[SensorClock] = None,
                 sensor_clocks: Optional[dict] = None):
        """Yield per-sensor measurement streams over a trajectory.

        For each sensor model, iterate the trajectory points (sampled at
        `self.dt`), apply the latency model, and collect
        (receive_time, true_sample_time, timestamp_used, valid).

        Returns dict: name -> list of tuples.
        """
        out: dict = {}
        for name, model in self.sensors.items():
            model.sample_interval = self.dt
            recs = []
            for pt in traj.points:
                t = pt.t
                # If the sensor has its own clock, stamp in sensor time.
                if sensor_clocks and name in sensor_clocks:
                    ts = sensor_clocks[name].sensor_time(t)
                else:
                    ts = t
                recv, ts_used, valid = model.deliver(ts)
                recs.append((recv, ts, ts_used, valid))
            out[name] = recs
        return out

    def summary(self, streams: dict) -> dict:
        """Summarize per-sensor: mean effective delay, loss rate."""
        summ = {}
        for name, recs in streams.items():
            valid = [r for r in recs if r[3]]
            delays = [r[0] - r[1] for r in valid]  # recv - true sample time
            summ[name] = {
                "mean_delay_s": float(np.mean(delays)) if delays else 0.0,
                "max_delay_s": float(np.max(delays)) if delays else 0.0,
                "loss_rate": 1.0 - len(valid) / len(recs) if recs else 0.0,
                "n_samples": len(valid),
            }
        return summ
