"""
IMU sensor model with realistic error characteristics.

Implements Allan variance / IEEE Std 952-1997 style error models:
  - Accelerometer: bias, scale factor, misalignment, noise, random walk
  - Gyroscope:    bias, scale factor, misalignment, noise, random walk

Sensor grades reference:
  Consumer:  phones, drones (e.g. BMI160, ICM-20948)
  Tactical:  industrial robots, mid-range AHRS (e.g. ADIS16490)
  Navigation: military, aerospace (e.g. Honeywell HG9900)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

from .utils import skew


class SensorGrade(Enum):
    CONSUMER = "consumer"
    TACTICAL = "tactical"
    NAVIGATION = "navigation"


# ---------------------------------------------------------------------------
# Preset specs for each sensor grade
# Reference: Groves (2013), IEEE Std 952, manufacturer datasheets
# ---------------------------------------------------------------------------

@dataclass
class IMUSpec:
    """IMU error specification for one grade."""

    # Accelerometer
    acc_bias_mg: float          # bias repeatability (mg = milli-g)
    acc_noise_ug_hz: float      # velocity random walk (ug/sqrt(Hz))
    acc_bias_instability_ug: float  # in-run bias instability (ug)
    acc_scale_ppm: float        # scale factor error (ppm)

    # Gyroscope
    gyr_bias_deg_h: float       # bias repeatability (deg/hr)
    gyr_noise_deg_h_hz: float   # angular random walk (deg/sqrt(hr))
    gyr_bias_instability_deg_h: float  # in-run bias instability (deg/hr)
    gyr_scale_ppm: float        # scale factor error (ppm)

    # Common
    misalignment_arcmin: float  # axis misalignment (arcmin)

    @property
    def acc_bias_ms2(self) -> float:
        return self.acc_bias_mg * 9.81e-3

    @property
    def acc_noise_ms2_hz(self) -> float:
        return self.acc_noise_ug_hz * 9.81e-6

    @property
    def acc_bias_instability_ms2(self) -> float:
        return self.acc_bias_instability_ug * 9.81e-6

    @property
    def gyr_bias_rad_s(self) -> float:
        return self.gyr_bias_deg_h * np.pi / 180 / 3600

    @property
    def gyr_noise_rad_s_hz(self) -> float:
        return self.gyr_noise_deg_h_hz * np.pi / 180 / 3600

    @property
    def gyr_bias_instability_rad_s(self) -> float:
        return self.gyr_bias_instability_deg_h * np.pi / 180 / 3600

    @property
    def misalignment_rad(self) -> float:
        return self.misalignment_arcmin * np.pi / 180 / 60


# ---------------------------------------------------------------------------
# Preset sensor grades
# ---------------------------------------------------------------------------

PRESETS = {
    SensorGrade.CONSUMER: IMUSpec(
        acc_bias_mg=50.0,
        acc_noise_ug_hz=200.0,
        acc_bias_instability_ug=100.0,
        acc_scale_ppm=10000,
        gyr_bias_deg_h=10.0,
        gyr_noise_deg_h_hz=0.5,
        gyr_bias_instability_deg_h=10.0,
        gyr_scale_ppm=10000,
        misalignment_arcmin=30.0,
    ),
    SensorGrade.TACTICAL: IMUSpec(
        acc_bias_mg=1.0,
        acc_noise_ug_hz=50.0,
        acc_bias_instability_ug=10.0,
        acc_scale_ppm=1000,
        gyr_bias_deg_h=1.0,
        gyr_noise_deg_h_hz=0.05,
        gyr_bias_instability_deg_h=0.5,
        gyr_scale_ppm=1000,
        misalignment_arcmin=5.0,
    ),
    SensorGrade.NAVIGATION: IMUSpec(
        acc_bias_mg=0.025,
        acc_noise_ug_hz=5.0,
        acc_bias_instability_ug=1.0,
        acc_scale_ppm=100,
        gyr_bias_deg_h=0.001,
        gyr_noise_deg_h_hz=0.001,
        gyr_bias_instability_deg_h=0.001,
        gyr_scale_ppm=10,
        misalignment_arcmin=0.5,
    ),
}


# ---------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------

def _random_orthogonal_small(angle_rad: float) -> np.ndarray:
    """
    Random small-angle rotation matrix.
    T ≈ I + [φ]× where φ ~ N(0, angle_rad) per axis.
    """
    phi = np.random.randn(3) * angle_rad
    return skew(phi)



class IMUSensor:
    """
    IMU sensor simulator.

    Error model per axis (x, y, z independently for noise/bias,
    combined for misalignment):

        a_meas = (I + M_a) * S_a * a_true + b_a + w_a
        ω_meas = (I + M_g) * S_g * ω_true + b_g + w_g

    where:
        M = misalignment skew matrix
        S = scale factor diagonal + cross-axis coupling
        b = bias (static + random walk)
        w = white noise

    Parameters
    ----------
    spec : IMUSpec or SensorGrade
        Sensor error specification or grade preset.
    dt : float
        Sensor output rate (s), typically 0.005-0.01.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, spec, dt: float = 0.01, seed: int = None):
        if isinstance(spec, SensorGrade):
            spec = PRESETS[spec]
        self.spec = spec
        self.dt = dt
        self._rng = np.random.RandomState(seed)

        # Draw static components (constant across run)
        self._bias_acc = self._draw_acc_bias()
        self._bias_gyr = self._draw_gyr_bias()
        self._scale_acc = self._draw_scale_matrix(self.spec.acc_scale_ppm)
        self._scale_gyr = self._draw_scale_matrix(self.spec.gyr_scale_ppm)
        self._mis_acc = self._draw_misalignment()
        self._mis_gyr = self._draw_misalignment()

        # Bias instability random walk state
        self._rw_acc = np.zeros(3)
        self._rw_gyr = np.zeros(3)
        self._rw_sigma_acc = self.spec.acc_bias_instability_ms2
        self._rw_sigma_gyr = self.spec.gyr_bias_instability_rad_s

        # Noise standard deviations
        self._noise_sigma_acc = self.spec.acc_noise_ms2_hz * np.sqrt(1.0 / dt)
        self._noise_sigma_gyr = self.spec.gyr_noise_rad_s_hz * np.sqrt(1.0 / dt)

    def _draw_acc_bias(self) -> np.ndarray:
        return self._rng.randn(3) * self.spec.acc_bias_ms2

    def _draw_gyr_bias(self) -> np.ndarray:
        return self._rng.randn(3) * self.spec.gyr_bias_rad_s

    def _draw_scale_matrix(self, scale_ppm: float) -> np.ndarray:
        """Scale factor error: diagonal 1 + ε, off-diagonal ~cross-axis error."""
        scale_frac = scale_ppm * 1e-6
        # Diagonal: scale error per axis
        diag = 1.0 + self._rng.randn(3) * scale_frac / np.sqrt(3)
        # Off-diagonal: cross-axis sensitivity
        off = self._rng.randn(3, 3) * scale_frac * 0.1  # 10% of scale error
        np.fill_diagonal(off, 0)
        return np.diag(diag) + off

    def _draw_misalignment(self) -> np.ndarray:
        return skew(self._rng.randn(3) * self.spec.misalignment_rad / np.sqrt(3))

    def _evolve_bias(self):
        """Evolve bias instability via random walk (first-order Gauss-Markov)."""
        self._rw_acc += self._rng.randn(3) * self._rw_sigma_acc * np.sqrt(self.dt)
        self._rw_gyr += self._rng.randn(3) * self._rw_sigma_gyr * np.sqrt(self.dt)

    def measure(
        self, acc_true: np.ndarray, omega_true: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate noisy IMU measurement.

        Parameters
        ----------
        acc_true : (3,) array
            True specific force in sensor frame (m/s²).
        omega_true : (3,) array
            True angular velocity in sensor frame (rad/s).

        Returns
        -------
        acc_meas : (3,) array
            Measured acceleration (m/s²).
        gyr_meas : (3,) array
            Measured angular velocity (rad/s).
        """
        self._evolve_bias()

        # Apply error model: meas = (I+M)*S*true + bias_static + bias_rw + noise
        bias_acc_total = self._bias_acc + self._rw_acc
        bias_gyr_total = self._bias_gyr + self._rw_gyr

        noise_acc = self._rng.randn(3) * self._noise_sigma_acc
        noise_gyr = self._rng.randn(3) * self._noise_sigma_gyr

        acc_meas = (
            (np.eye(3) + self._mis_acc) @ (self._scale_acc @ acc_true)
            + bias_acc_total + noise_acc
        )
        gyr_meas = (
            (np.eye(3) + self._mis_gyr) @ (self._scale_gyr @ omega_true)
            + bias_gyr_total + noise_gyr
        )

        return acc_meas, gyr_meas

    def get_errors(self) -> dict:
        """Return the static error components for inspection."""
        return {
            "bias_acc": self._bias_acc,
            "bias_gyr": self._bias_gyr,
            "scale_acc": self._scale_acc,
            "scale_gyr": self._scale_gyr,
            "misalign_acc": self._mis_acc,
            "misalign_gyr": self._mis_gyr,
            "noise_sigma_acc": self._noise_sigma_acc,
            "noise_sigma_gyr": self._noise_sigma_gyr,
        }

    def reset_bias_walk(self):
        """Reset the bias random walk to zero."""
        self._rw_acc = np.zeros(3)
        self._rw_gyr = np.zeros(3)

    def __repr__(self):
        if isinstance(self.spec, SensorGrade):
            return f"IMUSensor(grade={self.spec})"
        return f"IMUSensor(spec={self.spec!r})"
