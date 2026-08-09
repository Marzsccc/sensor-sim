"""
Allan variance demo: characterize a tactical-grade gyro from a static run.

Generates 1 hour of static (zero-rate) gyro data with the library's IMU
model (known white noise + bias random walk + static bias), runs the
overlapping Allan variance analysis, and compares the extracted noise
coefficients against the model's true settings.

Units handled here (datasheet-style):
    ARW  [deg/sqrt(h)]  = N_rad * 180/pi * sqrt(3600)
    BI   [deg/h]        = B_rad * 180/pi * 3600
    RRW  [deg/h/sqrt(h)]= K_rad * 180/pi * 3600
where N, B, K are the raw values in rad/s units returned by the analysis.

Run:
    python3 examples/allan_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.allan import (
    overlapping_allan_deviation,
    extract_noise_parameters,
)

D2R = np.pi / 180.0
SQRT_H = np.sqrt(3600.0)


def main():
    # --- Generate a static (zero-rate) gyro run ---------------------------
    dt = 0.01          # 100 Hz
    fs = 1.0 / dt
    duration_h = 1.0   # 1 hour
    n = int(duration_h * 3600.0 / dt)

    imu = IMUSensor(SensorGrade.TACTICAL, dt=dt, seed=7)
    acc_true = np.array([0.0, 0.0, 9.81])
    gyr_true = np.array([0.0, 0.0, 0.0])

    gx = np.empty(n)
    for i in range(n):
        _, g = imu.measure(acc_true, gyr_true)
        gx[i] = g[0]   # x-axis gyro (rad/s)

    # --- Allan analysis ---------------------------------------------------
    taus, adev = overlapping_allan_deviation(gx, fs=fs)
    params = extract_noise_parameters(taus, adev)

    print("=" * 68)
    print("Overlapping Allan variance analysis — tactical gyro (x-axis)")
    print(f"  {duration_h:.0f} h static run @ {fs:.0f} Hz  ({n} samples)")
    print("=" * 68)
    print("\nExtracted noise coefficients (raw, rad/s units):")
    print(params.summary(unit="rad/s"))

    # --- Convert to datasheet units --------------------------------------
    arw_dph = params.white_noise * (180.0 / np.pi) * SQRT_H      # deg/sqrt(h)
    bi_dph = params.bias_instability * (180.0 / np.pi) * 3600.0  # deg/h
    rrw_dph = params.random_walk * (180.0 / np.pi) * 3600.0      # deg/h/sqrt(h)

    # --- True values from the model spec ---------------------------------
    spec = imu.spec
    # RRW driving term: the model's bias-instability random walk has per-step
    # sigma = BI_rad_s * sqrt(dt), whose Allan RRW coefficient is K = BI_rad_s
    # (see tests for the derivation).

    def rel(a, b):
        return 100.0 * abs(a - b) / b

    print("=" * 68)
    print("Extracted vs. model settings")
    print("=" * 68)

    # Raw comparison in rad/s units (the model's native units)
    arw_true_rad = spec.gyr_noise_rad_s_hz
    rrw_true_rad = spec.gyr_bias_instability_rad_s
    print(f"  ARW (rad/sqrt(s)): est {params.white_noise:.4e}  "
          f"true {arw_true_rad:.4e}  rel err {rel(params.white_noise, arw_true_rad):5.1f}%")
    print(f"  BI  (rad/s)      : est {params.bias_instability:.4e}  "
          f"true {spec.gyr_bias_instability_rad_s:.4e}  "
          f"(curve minimum / 0.664)")
    print(f"  RRW (rad/s/sqrt(s)): est {params.random_walk:.4e}  "
          f"true {rrw_true_rad:.4e}  rel err {rel(params.random_walk, rrw_true_rad):5.1f}%")

    # Datasheet-style conversions of the extracted values (informational)
    print(f"\n  Datasheet units: ARW {arw_dph:.4f} deg/sqrt(h) | BI {bi_dph:.4f} deg/h "
          f"| RRW {rrw_dph:.4f} deg/h/sqrt(h)")

    ok = rel(params.white_noise, arw_true_rad) < 15.0 and rel(params.random_walk, rrw_true_rad) < 25.0
    print("\n  " + ("PASS: extracted coefficients match model settings"
                    if ok else "NOTE: large deviation from model settings"))

    # --- Save the curve for plotting (no matplotlib dependency) -----------
    out = os.path.join(os.path.dirname(__file__), "allan_curve.npz")
    np.savez(out, taus=taus, adev=adev, fs=fs)
    print(f"\n  Allan curve saved to {out}")


if __name__ == "__main__":
    main()
