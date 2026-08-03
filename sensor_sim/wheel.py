"""
Wheel odometry (轮速计) sensor model.

Models a differential-drive / Ackermann vehicle's wheel encoders:

  - Wheel radius error (per wheel, correlated)
  - Encoder quantization / resolution
  - Slip (correlated Gauss-Markov, higher when turning)
  - White noise
  - Optional: left/right wheel speed output (for differential models)
    or rear-axle speed + yaw rate output (for Ackermann models)

Reference: Groves (2013) Ch. 6, "Wheel Speed" aiding; typical encoder
datasheets (e.g. incremental encoders 1024-4096 PPR).
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class WheelGrade(Enum):
    CONSUMER = "consumer"       # e.g. hobby RC, low-cost EV (errors ~2-5%)
    AUTOMOTIVE = "automotive"   # production car ABS sensors (~0.5-1%)
    PRECISION = "precision"     # lab / measurement vehicle (<0.2%)


@dataclass
class WheelSpec:
    """Wheel odometry error specification."""

    # Wheel radius error (1σ, % of nominal radius)
    radius_error_pct: float

    # Encoder resolution: meters of travel per tick (quantization)
    tick_resolution_m: float

    # Slip model
    slip_sigma_pct: float       # 1σ slip as % of speed (steady state)
    slip_tau_s: float           # slip correlation time (s)

    # Output rate (Hz)
    output_rate_hz: float

    # Model type
    ackermann: bool = True      # True: speed+yaw_rate; False: left/right wheel


PRESETS_WHEEL = {
    WheelGrade.CONSUMER: WheelSpec(
        radius_error_pct=3.0,
        tick_resolution_m=0.02,
        slip_sigma_pct=2.0,
        slip_tau_s=5.0,
        output_rate_hz=10.0,
        ackermann=True,
    ),
    WheelGrade.AUTOMOTIVE: WheelSpec(
        radius_error_pct=0.8,
        tick_resolution_m=0.005,
        slip_sigma_pct=0.6,
        slip_tau_s=10.0,
        output_rate_hz=50.0,
        ackermann=True,
    ),
    WheelGrade.PRECISION: WheelSpec(
        radius_error_pct=0.15,
        tick_resolution_m=0.001,
        slip_sigma_pct=0.15,
        slip_tau_s=20.0,
        output_rate_hz=100.0,
        ackermann=True,
    ),
}


class WheelOdometry:
    """
    Wheel speed sensor simulator.

    Generates either:
      - Ackermann output: [v_meas (m/s), yaw_rate_meas (rad/s)]  — typical
        for production vehicles (ABS wheel speeds + steering model)
      - Differential output: [v_left, v_right] (m/s) — for robots / EVs
        with individual wheel encoders

    Error model:
        v_meas = (1 + r_err) * v_true * (1 + slip) + quantization + noise
    """

    def __init__(self, spec, dt: float = 0.01, seed: int = None):
        if isinstance(spec, WheelGrade):
            spec = PRESETS_WHEEL[spec]
        self.spec = spec
        self.dt = dt
        self._rng = np.random.RandomState(seed)

        # Static radius error (constant per run, per wheel for differential)
        self._radius_err = self._rng.randn() * spec.radius_error_pct * 0.01
        self._radius_err_l = self._rng.randn() * spec.radius_error_pct * 0.01
        self._radius_err_r = self._rng.randn() * spec.radius_error_pct * 0.01

        # Slip (correlated, Gauss-Markov)
        tau = spec.slip_tau_s
        self._slip_alpha = np.exp(-dt / tau) if tau > 0 else 0.0
        self._slip_sigma_drive = (
            spec.slip_sigma_pct * 0.01 * np.sqrt(1 - self._slip_alpha**2)
            if tau > 0 else 0.0
        )
        self._slip = 0.0

        # Quantization half-step
        self._quant_half = spec.tick_resolution_m / 2.0

        # Output rate
        self._output_interval_steps = max(1, int(round(1.0 / spec.output_rate_hz / dt)))
        self._steps_since_output = 0

    def _evolve_slip(self):
        self._slip = self._slip_alpha * self._slip + self._rng.randn() * self._slip_sigma_drive

    def _quantize(self, v: float) -> float:
        """Simulate encoder quantization: round to nearest tick."""
        if self._quant_half <= 0:
            return v
        return np.floor(v / self._quant_half + 0.5) * self._quant_half

    def measure(self, v_body: np.ndarray, omega_body: np.ndarray) -> Tuple[float, float]:
        """
        Generate wheel odometry measurement.

        Parameters
        ----------
        v_body : (3,) array
            True velocity in body frame (m/s). Forward = +x.
        omega_body : (3,) array
            True angular velocity in body frame (rad/s). Yaw = +z.

        Returns
        -------
        v_meas : float
            Measured forward speed (m/s), or NaN if not output step.
        yaw_rate_meas : float or None
            Measured yaw rate (rad/s). For differential model, returns
            (v_left, v_right) instead.
        valid : bool
            Whether this step produced a measurement.
        """
        self._evolve_slip()

        if not self._should_output():
            return np.nan, np.nan, False

        v_forward = float(v_body[0])
        yaw_rate = float(omega_body[2])

        # Noise
        noise = self._rng.randn() * 0.001  # small additive noise

        if self.spec.ackermann:
            # Radius error scales speed; slip scales speed
            v_meas = v_forward * (1 + self._radius_err) * (1 + self._slip) + noise
            v_meas = self._quantize(v_meas)
            # Yaw rate: mostly from gyro-like estimate, add small noise
            yr_noise = self._rng.randn() * 0.001
            yaw_rate_meas = yaw_rate + yr_noise
            return v_meas, yaw_rate_meas, True
        else:
            # Differential: left/right wheel speeds
            # half_track assumed 0.8m; yaw contribution = omega * half_track
            half_track = 0.8
            v_l_true = v_forward - yaw_rate * half_track
            v_r_true = v_forward + yaw_rate * half_track
            v_l = v_l_true * (1 + self._radius_err_l) * (1 + self._slip) + noise
            v_r = v_r_true * (1 + self._radius_err_r) * (1 + self._slip) + noise
            v_l = self._quantize(v_l)
            v_r = self._quantize(v_r)
            return v_l, v_r, True

    def _should_output(self) -> bool:
        self._steps_since_output += 1
        if self._steps_since_output >= self._output_interval_steps:
            self._steps_since_output = 0
            return True
        return False

    def measure_vectorized(self, v_body, omega_body):
        """Measure over full trajectory. Returns (v_meas, yr_meas, valid)."""
        n = len(v_body)
        v_out, yr_out, valid = [], [], []
        for i in range(n):
            a, b, ok = self.measure(v_body[i], omega_body[i])
            v_out.append(a)
            yr_out.append(b)
            valid.append(ok)
        return np.array(v_out), np.array(yr_out), np.array(valid)

    def get_errors(self) -> dict:
        return {
            "radius_error_pct": self._radius_err * 100,
            "slip_sigma_pct": self.spec.slip_sigma_pct,
            "tick_resolution_m": self.spec.tick_resolution_m,
        }

    def reset(self):
        self._slip = 0.0
        self._steps_since_output = 0

    def __repr__(self):
        if isinstance(self.spec, WheelGrade):
            return f"WheelOdometry(grade={self.spec.value})"
        return f"WheelOdometry(spec={self.spec!r})"
