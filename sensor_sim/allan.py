"""
Overlapping Allan variance analysis for IMU noise characterization.

Implements the classic overlapping Allan variance (IEEE Std 952-1997 style,
as used in gyroscope / accelerometer datasheets). Given a stationary sample
sequence (e.g. a static IMU run), it produces the Allan-deviation vs
averaging-time curve on a log-spaced (power-of-two) tau grid, then extracts
the dominant noise coefficients from the characteristic slopes of the
log-log plot:

    slope -1    ->  quantization noise       Q = q / sqrt(12)
    slope -1/2  ->  white noise (ARW/VRW)    N = adev * sqrt(tau)
    slope  0    ->  bias instability         B = adev_min / 0.664
    slope +1/2  ->  random walk (RRW)        K = adev * sqrt(3/tau)

All quantities are in the *data units* of the input sequence
(e.g. rad/s for a gyro, m/s^2 for an accelerometer); see the example
``examples/allan_example.py`` for conversions to the datasheet units
(deg/sqrt(h), ug/sqrt(Hz), deg/h ...).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

__all__ = [
    "overlapping_allan_deviation",
    "allan_variance",
    "extract_noise_parameters",
    "NoiseParams",
]


def _binary_tau_grid(max_m: int) -> np.ndarray:
    """Power-of-two cluster sizes m = 2^k <= max_m (log-spaced tau grid)."""
    if max_m < 1:
        raise ValueError(f"max_m must be >= 1, got {max_m}")
    k_max = int(np.floor(np.log2(max_m)))
    ms = 2 ** np.arange(0, k_max + 1)
    return ms[ms <= max_m]


def _cluster_means(data: np.ndarray, m: int) -> np.ndarray:
    """Mean of every non-overlapping-free sliding cluster of size m (O(N))."""
    cs = np.concatenate(([0.0], np.cumsum(data)))
    sums = cs[m:] - cs[:-m]
    return sums / m


def overlapping_allan_deviation(
    data: np.ndarray, fs: float = 1.0, max_m: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlapping Allan deviation of a 1-D stationary sample sequence.

    Parameters
    ----------
    data : (N,) array_like
        Sample sequence (one axis of gyro/accel at rest).
    fs : float
        Sampling rate in Hz. tau = m / fs.
    max_m : int or None
        Largest cluster size (in samples). Default: (N-1)//2, the maximum
        that still yields >= 1 overlapping pair. The tau grid uses powers of
        two up to max_m (log-spaced, ~12-20 points for typical run lengths).

    Returns
    -------
    taus : (K,) ndarray
        Averaging times tau = m / fs (s).
    adev : (K,) ndarray
        Overlapping Allan deviation sigma(tau) in data units
        (sqrt of the overlapping Allan variance).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError(
            f"data must be 1-D (pass one sensor axis at a time), got shape {data.shape}"
        )
    if data.size < 4:
        raise ValueError(f"data too short for Allan variance: {data.size} samples")
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")

    N = data.size
    max_m = (N - 1) // 2 if max_m is None else int(max_m)
    max_m = min(max_m, (N - 1) // 2)
    ms = _binary_tau_grid(max_m)

    taus = ms / fs
    adev = np.empty(ms.size)
    for k, m in enumerate(ms):
        avg = _cluster_means(data, m)          # (N - m) cluster means
        diff = avg[m:] - avg[:-m]              # overlapping pairs
        # Overlapping Allan variance = mean(diff^2) / 2
        adev[k] = np.sqrt(np.mean(diff ** 2) / 2.0)
    return taus, adev


def allan_variance(
    data: np.ndarray, fs: float = 1.0, max_m: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan variance (square of the deviation). See
    :func:`overlapping_allan_deviation`."""
    taus, adev = overlapping_allan_deviation(data, fs=fs, max_m=max_m)
    return taus, adev ** 2


@dataclass
class NoiseParams:
    """
    Noise coefficients extracted from an Allan deviation curve.

    All values are in the data units of the analyzed sequence, e.g.
    rad/s for a gyro or m/s^2 for an accelerometer:

        white_noise       N   white-noise coefficient: ARW for a gyro
                            (rad/s * sqrt(s) == rad/sqrt(s)),
                            VRW for an accelerometer (m/s^2 * sqrt(s)).
        bias_instability  B   minimum Allan deviation / 0.664
                            (rate units, e.g. rad/s).
        random_walk       K   rate random walk (RRW) coefficient
                            (rate units / sqrt(s), e.g. rad/s / sqrt(s)).
        quantization      Q   quantization noise coefficient, Q = q/sqrt(12)
                            (data units).
        tau_min, adev_min   location of the curve minimum (bias-instability
                            corner), useful for sanity checks.

    Aliases: ``arw``/``vrw`` -> ``white_noise``, ``rrw`` -> ``random_walk``.
    """

    white_noise: float
    bias_instability: float
    random_walk: float
    quantization: float
    tau_min: float
    adev_min: float

    @property
    def arw(self) -> float:
        """Angle random walk coefficient (gyro white noise), data units."""
        return self.white_noise

    @property
    def vrw(self) -> float:
        """Velocity random walk coefficient (accel white noise), data units."""
        return self.white_noise

    @property
    def rrw(self) -> float:
        """Rate random walk coefficient, data units."""
        return self.random_walk

    def summary(self, unit: str = "") -> str:
        """Compact multi-line summary for printing."""
        u = f" {unit}" if unit else ""
        return (
            f"  white noise       N = {self.white_noise:.4e}{u}*sqrt(s)\n"
            f"  bias instability  B = {self.bias_instability:.4e}{u}\n"
            f"  random walk       K = {self.random_walk:.4e}{u}/sqrt(s)\n"
            f"  quantization      Q = {self.quantization:.4e}{u}\n"
            f"  curve minimum: adev_min = {self.adev_min:.4e}{u} at tau = {self.tau_min:.3f} s"
        )


def extract_noise_parameters(
    taus: np.ndarray, adev: np.ndarray, slope_band: float = 0.25
) -> NoiseParams:
    """
    Extract noise coefficients from an (overlapping) Allan deviation curve.

    The log-log slope of the curve is measured locally; each coefficient is
    the median of the point-wise estimate over the samples whose slope falls
    in the characteristic band:

        -1      quantization:   Q = adev * tau / sqrt(3)
        -0.5    white noise:    N = adev * sqrt(tau)
        0       (minimum)       B = min(adev) / 0.664
        +0.5    random walk:    K = adev * sqrt(3 / tau)

    Parameters
    ----------
    taus : (K,) ndarray
        Averaging times (s), as returned by :func:`overlapping_allan_deviation`.
    adev : (K,) ndarray
        Allan deviation in data units.
    slope_band : float
        Half-width of the slope acceptance band (default 0.25 covers
        +/-0.25 around the nominal slope; robust on log-spaced grids).

    Returns
    -------
    NoiseParams
    """
    taus = np.asarray(taus, dtype=float)
    adev = np.asarray(adev, dtype=float)
    if taus.shape != adev.shape or taus.size < 4:
        raise ValueError("taus and adev must be equal-length arrays with >= 4 points")

    lt = np.log10(taus)
    la = np.log10(adev)
    slopes = np.gradient(la, lt)

    def _estimate(mask, values):
        if mask.sum() < 2:
            return float("nan")
        return float(np.median(values[mask]))

    w_mask = np.abs(slopes + 0.5) <= slope_band
    r_mask = np.abs(slopes - 0.5) <= slope_band
    q_mask = np.abs(slopes + 1.0) <= slope_band

    white = _estimate(w_mask, adev * np.sqrt(taus))
    rrw = _estimate(r_mask, adev * np.sqrt(3.0 / taus))
    quant = _estimate(q_mask, adev * taus) / np.sqrt(3.0)

    imin = int(np.argmin(adev))
    bias = float(adev[imin] / 0.664)

    return NoiseParams(
        white_noise=white,
        bias_instability=bias,
        random_walk=rrw,
        quantization=quant,
        tau_min=float(taus[imin]),
        adev_min=float(adev[imin]),
    )
