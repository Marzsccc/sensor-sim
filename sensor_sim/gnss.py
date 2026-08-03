"""
GNSS sensor model with configurable error sources.

Error sources modeled:
  1. White noise (position + velocity), scaled by DOP
  2. Multipath (correlated noise, 1st-order Gauss-Markov)
  3. Signal loss / dropout
  4. Satellite geometry degradation (DOP variation)

Receiver grades:
  Consumer:  phone-grade, ~2-5m CEP
  Automotive:  u-blox F9, ~1-2m CEP
  RTK:         cm-level with base station
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class GNSSGrade(Enum):
    CONSUMER = "consumer"       # Phone, basic L1
    AUTOMOTIVE = "automotive"   # u-blox F9, dual-frequency
    RTK = "rtk"                 # Real-time kinematic, cm-level
    SURVEY = "survey"           # Survey-grade post-processing


@dataclass
class GNSSSpec:
    """GNSS receiver error specification."""

    # Position noise (1σ, meters), per horizontal (H) and vertical (V)
    pos_noise_h_m: float
    pos_noise_v_m: float

    # Velocity noise (1σ, m/s)
    vel_noise_ms: float

    # Multipath: correlation time and amplitude
    multipath_tau_s: float      # correlation time constant (s)
    multipath_sigma_h_m: float  # steady-state std horizontal (m)
    multipath_sigma_v_m: float  # steady-state std vertical (m)

    # Signal loss
    dropout_prob: float          # per-measurement dropout probability
    mean_loss_duration_s: float  # mean outage duration (s)

    # Rate
    output_rate_hz: float


PRESETS_GNSS = {
    GNSSGrade.CONSUMER: GNSSSpec(
        pos_noise_h_m=3.0,
        pos_noise_v_m=6.0,
        vel_noise_ms=0.1,
        multipath_tau_s=30.0,
        multipath_sigma_h_m=5.0,
        multipath_sigma_v_m=10.0,
        dropout_prob=0.01,
        mean_loss_duration_s=2.0,
        output_rate_hz=1.0,
    ),
    GNSSGrade.AUTOMOTIVE: GNSSSpec(
        pos_noise_h_m=1.0,
        pos_noise_v_m=2.0,
        vel_noise_ms=0.05,
        multipath_tau_s=20.0,
        multipath_sigma_h_m=2.0,
        multipath_sigma_v_m=4.0,
        dropout_prob=0.005,
        mean_loss_duration_s=1.0,
        output_rate_hz=5.0,
    ),
    GNSSGrade.RTK: GNSSSpec(
        pos_noise_h_m=0.02,
        pos_noise_v_m=0.04,
        vel_noise_ms=0.01,
        multipath_tau_s=10.0,
        multipath_sigma_h_m=0.05,
        multipath_sigma_v_m=0.10,
        dropout_prob=0.001,
        mean_loss_duration_s=0.5,
        output_rate_hz=10.0,
    ),
    GNSSGrade.SURVEY: GNSSSpec(
        pos_noise_h_m=0.005,
        pos_noise_v_m=0.010,
        vel_noise_ms=0.005,
        multipath_tau_s=5.0,
        multipath_sigma_h_m=0.02,
        multipath_sigma_v_m=0.04,
        dropout_prob=0.0005,
        mean_loss_duration_s=0.3,
        output_rate_hz=20.0,
    ),
}


class GNSSSensor:
    """
    GNSS sensor simulator.

    Generates position and velocity measurements with:
      - White Gaussian noise scaled by DOP
      - Multipath as correlated (Gauss-Markov) noise
      - Random signal dropouts with configurable mean duration

    Parameters
    ----------
    spec : GNSSSpec or GNSSGrade
        Receiver specification or grade preset.
    dt : float
        Simulation timestep (s). Measurements are generated at spec.output_rate_hz,
        but the internal multipath state evolves at dt resolution.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, spec, dt: float = 0.01, seed: int = None):
        if isinstance(spec, GNSSGrade):
            spec = PRESETS_GNSS[spec]
        self.spec = spec
        self.dt = dt
        self._rng = np.random.RandomState(seed)

        # Multipath state (first-order Gauss-Markov)
        self._mp_h = np.zeros(2)   # horizontal multipath (E, N)
        self._mp_v = 0.0            # vertical multipath

        # Gauss-Markov update coefficients
        tau = spec.multipath_tau_s
        self._mp_alpha = np.exp(-dt / tau) if tau > 0 else 0.0
        self._mp_sigma_drive_h = (
            spec.multipath_sigma_h_m * np.sqrt(1 - self._mp_alpha**2)
            if tau > 0 else 0.0
        )
        self._mp_sigma_drive_v = (
            spec.multipath_sigma_v_m * np.sqrt(1 - self._mp_alpha**2)
            if tau > 0 else 0.0
        )

        # Dropout state machine
        self._in_dropout = False
        self._dropout_remaining = 0.0
        self._dropout_prob_per_step = spec.dropout_prob

        # Output rate control
        self._output_interval_steps = max(1, int(round(1.0 / spec.output_rate_hz / dt)))
        self._steps_since_output = 0

        # Last valid measurement (used during dropouts to return NaN or None)
        self._last_position = None
        self._last_velocity = None

        # Debug: count stats
        self.num_dropouts = 0
        self.total_dropout_time = 0.0

    def _evolve_multipath(self):
        """Evolve correlated multipath error (first-order Gauss-Markov)."""
        drive_h = self._rng.randn(2) * self._mp_sigma_drive_h
        drive_v = self._rng.randn() * self._mp_sigma_drive_v
        self._mp_h = self._mp_alpha * self._mp_h + drive_h
        self._mp_v = self._mp_alpha * self._mp_v + drive_v

    def _should_output(self) -> bool:
        """Check if it's time for a GNSS output sample."""
        self._steps_since_output += 1
        if self._steps_since_output >= self._output_interval_steps:
            self._steps_since_output = 0
            return True
        return False

    def measure(
        self, pos_true: np.ndarray, vel_true: np.ndarray = None
    ) -> tuple:
        """
        Generate GNSS measurement.

        Parameters
        ----------
        pos_true : (3,) array
            True position in world frame (m). [E, N, U] or [x, y, z].
        vel_true : (3,) array or None
            True velocity in world frame (m/s). If None, velocity is not measured.

        Returns
        -------
        pos_meas : (3,) array or None
            Measured position, or None during dropout / non-output step.
        vel_meas : (3,) array or None
            Measured velocity (same semantics as pos_meas).
        valid : bool
            Whether the measurement is valid (not in dropout, at output rate).
        """
        self._evolve_multipath()

        # Handle dropout
        if self._in_dropout:
            self._dropout_remaining -= self.dt
            self.total_dropout_time += self.dt
            if self._dropout_remaining <= 0:
                self._in_dropout = False
            return None, None, False

        # Random dropout onset
        if self._rng.rand() < self._dropout_prob_per_step:
            self._in_dropout = True
            self._dropout_remaining = self._rng.exponential(
                self.spec.mean_loss_duration_s
            )
            self.num_dropouts += 1
            return None, None, False

        # Check output rate
        if not self._should_output():
            return None, None, False

        # White noise (position)
        noise_h = self._rng.randn(2) * self.spec.pos_noise_h_m
        noise_v = self._rng.randn() * self.spec.pos_noise_v_m

        # Total position error = white noise + multipath
        pos_meas = pos_true.copy()
        pos_meas[:2] += noise_h + self._mp_h
        pos_meas[2] += noise_v + self._mp_v

        # Velocity (if requested)
        vel_meas = None
        if vel_true is not None:
            vel_noise = self._rng.randn(3) * self.spec.vel_noise_ms
            vel_meas = vel_true + vel_noise

        # Store last valid
        self._last_position = pos_meas.copy()
        self._last_velocity = vel_meas.copy() if vel_meas is not None else None

        return pos_meas.copy(), vel_meas, True

    def measure_vectorized(self, positions: np.ndarray, velocities=None):
        """
        Generate GNSS measurements for an entire trajectory.

        Parameters
        ----------
        positions : (N, 3) array
            True positions.
        velocities : (N, 3) array or None

        Returns
        -------
        pos_meas_list : list of (3,) arrays or None
        vel_meas_list : list of (3,) arrays or None
        valid_mask : (N,) bool array
        """
        n = len(positions)
        pos_out, vel_out, valid = [], [], []
        for i in range(n):
            vel_i = velocities[i] if velocities is not None else None
            p, v, ok = self.measure(positions[i], vel_i)
            pos_out.append(p)
            vel_out.append(v)
            valid.append(ok)
        return pos_out, vel_out, np.array(valid)

    def get_stats(self) -> dict:
        """Return runtime statistics."""
        return {
            "num_dropouts": self.num_dropouts,
            "total_dropout_time_s": round(self.total_dropout_time, 2),
        }

    def reset(self):
        """Reset all state."""
        self._mp_h = np.zeros(2)
        self._mp_v = 0.0
        self._in_dropout = False
        self._dropout_remaining = 0.0
        self._steps_since_output = 0
        self.num_dropouts = 0
        self.total_dropout_time = 0.0
        self._last_position = None
        self._last_velocity = None

    def __repr__(self):
        if isinstance(self.spec, GNSSGrade):
            return f"GNSSSensor(grade={self.spec.value})"
        return f"GNSSSensor(spec={self.spec!r})"
