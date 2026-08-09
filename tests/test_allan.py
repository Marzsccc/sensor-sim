"""
Tests for the overlapping Allan variance analysis (sensor_sim/allan.py).

Run: .venv/bin/python -m pytest tests/test_allan.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sensor_sim.allan import (
    overlapping_allan_deviation,
    allan_variance,
    extract_noise_parameters,
    NoiseParams,
)
from sensor_sim.imu import IMUSensor, SensorGrade

# 2^20 samples @ 100 Hz -> ~3 h of data; keeps the tau grid rich and the
# slope-based extraction reliable while keeping the test suite fast.
N = 1 << 20
FS = 100.0
DT = 1.0 / FS


# ---------------------------------------------------------------------------
# 1. Shape of the curve for a constant-noise signal
# ---------------------------------------------------------------------------

def test_constant_noise_curve_shape():
    """Constant white noise: taus strictly increasing, adev strictly
    positive and monotonically decreasing (white-noise dominated)."""
    rng = np.random.RandomState(42)
    data = rng.randn(N) * 1e-3
    taus, adev = overlapping_allan_deviation(data, fs=FS)
    assert taus.ndim == 1 and taus.size >= 12, "tau grid too coarse"
    assert np.all(np.diff(taus) > 0), "taus must be strictly increasing"
    assert np.all(adev > 0), "Allan deviation must be positive"
    assert np.all(np.diff(adev) < 0), \
        "white-noise-dominated curve must decrease monotonically"


# ---------------------------------------------------------------------------
# 2. White noise: slope -1/2 and ARW extraction
# ---------------------------------------------------------------------------

def test_white_noise_slope_minus_half():
    """Pure white noise -> Allan deviation ~ tau^-0.5 (log-log slope -0.5),
    and N = adev * sqrt(tau) recovers sigma * sqrt(tau0)."""
    sigma = 2.5e-3
    rng = np.random.RandomState(0)
    data = rng.randn(N) * sigma
    taus, adev = overlapping_allan_deviation(data, fs=FS)

    lt = np.log10(taus)
    slopes = np.gradient(np.log10(adev), lt)
    mid = slice(2, -4)  # avoid short-tau & long-tau edge effects
    assert abs(np.mean(slopes[mid]) + 0.5) < 0.06, \
        f"mean slope {np.mean(slopes[mid]):.3f}, expected ~-0.5"

    n_est = np.median(adev * np.sqrt(taus))
    n_true = sigma * np.sqrt(DT)  # ARW/VRW coefficient
    assert abs(n_est - n_true) / n_true < 0.03, \
        f"N extracted {n_est:.4e}, expected {n_true:.4e}"


# ---------------------------------------------------------------------------
# 3. Random walk: slope +1/2 and RRW extraction
# ---------------------------------------------------------------------------

def test_random_walk_slope_plus_half():
    """Integrated white noise (rate random walk) -> Allan deviation ~ tau^+0.5
    (log-log slope +0.5), and K = adev * sqrt(3/tau) recovers the driving
    step coefficient."""
    sigma_eps = 1e-5  # per-step std of the random-walk increments
    rng = np.random.RandomState(1)
    eps = rng.randn(N) * sigma_eps
    data = np.cumsum(eps)
    taus, adev = overlapping_allan_deviation(data, fs=FS)

    lt = np.log10(taus)
    slopes = np.gradient(np.log10(adev), lt)
    mid = slice(3, -2)
    assert abs(np.mean(slopes[mid]) - 0.5) < 0.07, \
        f"mean slope {np.mean(slopes[mid]):.3f}, expected ~+0.5"

    k_est = np.median(adev * np.sqrt(3.0 / taus))
    # Discrete rw with step std sigma_eps at period tau0 has diffusion rate
    # q_b = sigma_eps^2 / tau0, hence the Allan RRW coefficient is
    # K = sqrt(q_b) = sigma_eps / sqrt(tau0) = sigma_eps * sqrt(fs).
    k_true = sigma_eps * np.sqrt(FS)
    assert abs(k_est - k_true) / k_true < 0.06, \
        f"K extracted {k_est:.4e}, expected {k_true:.4e}"


# ---------------------------------------------------------------------------
# 4. End-to-end: parameter extraction close to the IMU model's settings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [7, 11])
def test_extraction_matches_imu_model(seed):
    """Static tactical-grade gyro run: extracted ARW/VRW and RRW must land
    within 25% of the model's true settings."""
    imu = IMUSensor(SensorGrade.TACTICAL, dt=DT, seed=seed)
    acc_true = np.array([0.0, 0.0, 9.81])
    gyr_true = np.zeros(3)

    data = np.empty(N)
    for i in range(N):
        _, g = imu.measure(acc_true, gyr_true)
        data[i] = g[0]

    taus, adev = overlapping_allan_deviation(data, fs=FS)
    params = extract_noise_parameters(taus, adev)

    spec = imu.spec
    # ARW: white-noise coefficient in rad/s * sqrt(s)
    n_true = spec.gyr_noise_rad_s_hz
    assert np.isfinite(params.white_noise)
    assert abs(params.white_noise - n_true) / n_true < 0.25, \
        f"ARW extracted {params.white_noise:.4e}, expected {n_true:.4e}"

    # RRW: the bias-instability random walk has per-step std
    # sigma = BI_rad_s * sqrt(dt) -> Allan RRW coefficient K = BI_rad_s.
    k_true = spec.gyr_bias_instability_rad_s
    assert np.isfinite(params.random_walk)
    assert abs(params.random_walk - k_true) / k_true < 0.25, \
        f"RRW extracted {params.random_walk:.4e}, expected {k_true:.4e}"

    # Bias instability: this model implements "bias instability" as a
    # random walk (no true 1/f plateau), so min(adev)/0.664 lands at the
    # ARW<->RRW crossover and can exceed the BI level by a few x. Keep an
    # order-of-magnitude sanity bound; the tight checks are ARW and RRW.
    assert np.isfinite(params.bias_instability)
    assert 0.5 < params.bias_instability / k_true < 10.0, \
        f"BI extracted {params.bias_instability:.4e}, expected ~{k_true:.4e}"


# ---------------------------------------------------------------------------
# 5. Accel VRW extraction (bonus: white-noise coefficient on accelerometer)
# ---------------------------------------------------------------------------

def test_vrw_extraction_accelerometer():
    """Static tactical-grade accelerometer: VRW extracted within 25%."""
    imu = IMUSensor(SensorGrade.TACTICAL, dt=DT, seed=3)
    acc_true = np.array([0.0, 0.0, 9.81])
    gyr_true = np.zeros(3)

    data = np.empty(N)
    for i in range(N):
        a, _ = imu.measure(acc_true, gyr_true)
        data[i] = a[0]

    taus, adev = overlapping_allan_deviation(data, fs=FS)
    params = extract_noise_parameters(taus, adev)
    vrw_true = imu.spec.acc_noise_ms2_hz  # m/s^2 * sqrt(s)
    assert np.isfinite(params.white_noise)
    assert abs(params.white_noise - vrw_true) / vrw_true < 0.25, \
        f"VRW extracted {params.white_noise:.4e}, expected {vrw_true:.4e}"


# ---------------------------------------------------------------------------
# 6. Misc: variance == deviation^2, quantization slope, input validation
# ---------------------------------------------------------------------------

def test_allan_variance_is_deviation_squared():
    rng = np.random.RandomState(5)
    data = rng.randn(1 << 16) * 1e-3
    taus_v, avar = allan_variance(data, fs=FS)
    taus_d, adev = overlapping_allan_deviation(data, fs=FS)
    assert np.array_equal(taus_v, taus_d)
    assert np.allclose(avar, adev ** 2, rtol=1e-12)


def test_quantization_noise_slope_minus_one():
    """Quantization-dominated noise (slope -1): Q = adev*tau/sqrt(3)
    recovers q/sqrt(12) with the q = uniform(w_k - w_{k-1}) model."""
    q = 1e-4
    rng = np.random.RandomState(2)
    w = rng.uniform(-q / 2, q / 2, N + 1)
    # Angle-quantization model: quantizing the integrated signal adds iid
    # U(-q/2, q/2) error to the angle; the resulting rate sample is
    # fs * (w[k] - w[k-1]), which has slope -1 in the Allan domain.
    data = FS * np.diff(w)
    taus, adev = overlapping_allan_deviation(data, fs=FS)

    lt = np.log10(taus)
    slopes = np.gradient(np.log10(adev), lt)
    assert abs(np.mean(slopes[:6]) + 1.0) < 0.1, \
        f"short-tau mean slope {np.mean(slopes[:6]):.3f}, expected ~-1"

    q_est = np.median(adev * taus) / np.sqrt(3.0)
    q_true = q / np.sqrt(12.0)
    assert abs(q_est - q_true) / q_true < 0.02, \
        f"Q extracted {q_est:.4e}, expected {q_true:.4e}"


def test_input_validation():
    with pytest.raises(ValueError):
        overlapping_allan_deviation(np.zeros((2, 2)), fs=FS)   # not 1-D
    with pytest.raises(ValueError):
        overlapping_allan_deviation(np.zeros(3), fs=FS)        # too short
    with pytest.raises(ValueError):
        overlapping_allan_deviation(np.zeros(64), fs=0.0)      # bad fs
    with pytest.raises(ValueError):
        extract_noise_parameters(np.ones(3), np.ones(3))       # too few points


def test_noise_params_aliases():
    p = NoiseParams(1.0, 2.0, 3.0, 4.0, 0.1, 0.5)
    assert p.arw == p.white_noise == 1.0
    assert p.vrw == p.white_noise
    assert p.rrw == p.random_walk == 3.0
    assert "white noise" in p.summary()
