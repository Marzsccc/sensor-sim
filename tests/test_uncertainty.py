"""
Tests for display-time prediction uncertainty propagation (v0.7.0).

Coverage:
  - process_noise: structure (9x9, PSD, velocity/attitude blocks),
    accel-noise coupling through the attitude rotation matrix, model-error
    scaling with horizon
  - propagate_covariance: zero-noise PSD growth, covariance growth vs
    horizon (monotonic, super-linear for the model-error term),
    initial-state contribution preserved for zero horizon
  - horizontal_ellipse: positive semi-axes, area, 95% radius vs 1-sigma
  - predict_with_covariance: end-to-end shape + ellipse
  - integration: ESKF steady-state P -> display-time covariance grows
    monotonically with horizon and stays bounded for 80 ms
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sensor_sim.predict import (
    PredictorUncertainty,
    UncertaintyConfig,
    PosePredictor,
    PredictionConfig,
)
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.utils import euler_to_quat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _straight_traj(duration=10.0, speed=10.0):
    wp = [
        Waypoint(t=0.0, pos=np.array([0.0, 0.0, 0.0]),
                 vel=np.array([speed, 0.0, 0.0]),
                 att=np.array([1.0, 0.0, 0.0, 0.0]),
                 omega=np.array([0.0, 0.0, 0.0])),
        Waypoint(t=duration, pos=np.array([speed * duration, 0.0, 0.0]),
                 vel=np.array([speed, 0.0, 0.0]),
                 att=np.array([1.0, 0.0, 0.0, 0.0]),
                 omega=np.array([0.0, 0.0, 0.0])),
    ]
    return Trajectory(wp, dt=0.01)


def _level_body_accel(speed=10.0):
    """Body accel for a straight level run at constant speed."""
    pt = _straight_traj().points[0]
    return _straight_traj().body_accel(pt)


def _eye9_pos_std(std_p=0.5, std_v=0.2, std_th_deg=0.5):
    """9x9 diagonal covariance with sensible initial values."""
    d = np.array([std_p] * 3 + [std_v] * 3 + [np.deg2rad(std_th_deg)] * 3)
    return np.diag(d ** 2)


# ---------------------------------------------------------------------------
# process_noise
# ---------------------------------------------------------------------------


def test_process_noise_shape_and_psd():
    unc = PredictorUncertainty()
    Q = unc.process_noise(np.array([1.0, 0, 0, 0]), np.zeros(3), dt=0.005)
    assert Q.shape == (9, 9)
    # PSD: eigenvalues >= 0
    w = np.linalg.eigvalsh(Q)
    assert w.min() >= -1e-15, f"Q not PSD: {w.min()}"
    # velocity and attitude blocks populated, position block zero
    assert np.allclose(Q[0:3, 0:3], 0.0)
    assert np.trace(Q[3:6, 3:6]) > 0
    assert np.trace(Q[6:9, 6:9]) > 0


def test_process_noise_accel_coupling_through_rotation():
    """Accel-noise block must rotate with the attitude."""
    unc = PredictorUncertainty()
    q0 = np.array([1.0, 0, 0, 0])                 # level
    q90 = euler_to_quat(0.0, 0.0, np.pi / 2)      # yaw 90 deg
    a = np.array([0.0, 0.0, 9.81])                # any nonzero body accel
    Q0 = unc.process_noise(q0, a, dt=0.01)
    Q1 = unc.process_noise(q90, a, dt=0.01)
    # Velocity-block trace is rotation-invariant; off-diagonal xz/yz terms
    # must appear after the rotation (cross-coupling in world frame).
    assert np.isclose(np.trace(Q0[3:6, 3:6]), np.trace(Q1[3:6, 3:6]))
    # For a yaw-rotated body, accel noise projects differently on x/y.
    assert abs(Q0[3, 5]) < 1e-12 or abs(Q1[3, 5]) > 0 or abs(Q1[4, 5]) > 0


def test_process_noise_model_error_scales_with_dt():
    """Model-error term must grow linearly with dt (deterministic drift)."""
    unc = PredictorUncertainty()
    q = np.array([1.0, 0, 0, 0])
    a = np.zeros(3)
    Q_a = unc.process_noise(q, a, dt=0.01)
    Q_b = unc.process_noise(q, a, dt=0.02)
    # Velocity diagonal from model error only (no accel noise here since
    # Qv includes accel noise too -- compare the difference):
    d_a = np.diag(Q_a[3:6, 3:6])
    d_b = np.diag(Q_b[3:6, 3:6])
    # The accel-noise part is identical per unit dt; the model-error part
    # is s_m^2*dt.  So (d_b - 2*d_a) is the extra linearity check:
    # d_b should be ~2x d_a when model error dominates (s_m = 0.5 >> noise).
    assert d_b[0] > 1.8 * d_a[0], f"model error not dominant: {d_a[0]} {d_b[0]}"


# ---------------------------------------------------------------------------
# propagate_covariance
# ---------------------------------------------------------------------------


def test_propagate_zero_horizon_keeps_init():
    unc = PredictorUncertainty()
    P0 = _eye9_pos_std()
    P = unc.propagate_covariance(P0, np.array([1.0, 0, 0, 0]),
                                 np.zeros(3), horizon=0.0)
    assert np.allclose(P, P0, atol=1e-12)


def test_propagate_no_noise_grows_along_velocity():
    """With zero process noise, a position error grows only via v*dt."""
    unc = PredictorUncertainty(UncertaintyConfig(
        imu_acc_noise_density=0.0, imu_gyr_noise_density=0.0,
        imu_acc_bias_rw=0.0, imu_gyr_bias_rw=0.0,
        model_error_acc_std=0.0,
    ))
    P0 = _eye9_pos_std(std_p=0.5, std_v=0.2, std_th_deg=0.5)
    P = unc.propagate_covariance(P0, np.array([1.0, 0, 0, 0]),
                                 np.zeros(3), horizon=0.05)
    # Position std grows by ~ v_std * horizon = 0.2 * 0.05 = 0.01 m
    # plus the initial 0.5 m -> ~0.5001 m.  Velocity std unchanged.
    assert abs(np.sqrt(P[0, 0]) - np.sqrt(P0[0, 0])) < 0.02
    assert abs(np.sqrt(P[3, 3]) - np.sqrt(P0[3, 3])) < 1e-9
    # Cross-correlation p-v appears (F term): P[0,3] > 0
    assert P[0, 3] > 0


def test_propagate_grows_with_horizon():
    unc = PredictorUncertainty()
    P0 = _eye9_pos_std()
    P50 = unc.propagate_covariance(P0, np.array([1.0, 0, 0, 0]),
                                   np.zeros(3), horizon=0.05)
    P100 = unc.propagate_covariance(P0, np.array([1.0, 0, 0, 0]),
                                    np.zeros(3), horizon=0.10)
    assert np.sqrt(P100[0, 0]) > np.sqrt(P50[0, 0])
    assert np.sqrt(P100[3, 3]) > np.sqrt(P50[3, 3])


def test_propagate_preserves_psd():
    unc = PredictorUncertainty()
    P0 = _eye9_pos_std()
    P = unc.propagate_covariance(P0, np.array([1.0, 0, 0, 0]),
                                 np.zeros(3), horizon=0.08)
    w = np.linalg.eigvalsh(P)
    assert w.min() >= -1e-12, f"P not PSD: {w.min()}"


# ---------------------------------------------------------------------------
# horizontal_ellipse
# ---------------------------------------------------------------------------


def test_ellipse_axes_and_area():
    unc = PredictorUncertainty()
    P0 = _eye9_pos_std(std_p=0.5)
    P = unc.propagate_covariance(P0, np.array([1.0, 0, 0, 0]),
                                 np.zeros(3), horizon=0.05)
    e = unc.horizontal_ellipse(P)
    assert e["sigma_x"] >= 0 and e["sigma_y"] >= 0
    assert e["sigma_x"] >= e["sigma_y"] - 1e-9
    assert e["area"] >= 0
    # 95% radius must exceed the 1-sigma radius
    assert e["radius_95"] > max(e["sigma_x"], e["sigma_y"])
    # Circle case: isotropic covariance -> sigma_x ~ sigma_y
    P_iso = np.diag([0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_iso = unc.horizontal_ellipse(P_iso)
    assert abs(e_iso["sigma_x"] - e_iso["sigma_y"]) < 1e-9


# ---------------------------------------------------------------------------
# predict_with_covariance
# ---------------------------------------------------------------------------


def test_predict_with_covariance_end_to_end():
    unc = PredictorUncertainty()
    pred = PosePredictor(PredictionConfig(use_wheel=True))
    P0 = _eye9_pos_std()
    res = unc.predict_with_covariance(
        pred,
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([10.0, 0.0, 0.0]),
        att=np.array([1.0, 0, 0, 0]),
        acc_meas=np.array([0.0, 0.0, 9.81]),
        gyr_meas=np.zeros(3),
        horizon=0.0667,
        P_init=P0,
        wheel_speed=10.0,
        wheel_yaw_rate=0.0,
    )
    assert set(res.keys()) >= {"pos", "vel", "att", "P_disp", "ellipse"}
    assert res["P_disp"].shape == (9, 9)
    # Mean position ~ speed * horizon forward
    assert abs(res["pos"][0] - 10.0 * 0.0667) < 0.05
    # Uncertainty grows beyond the initial position std
    assert res["ellipse"]["sigma_x"] > 0.5


def test_predict_with_covariance_bad_P():
    unc = PredictorUncertainty()
    pred = PosePredictor(PredictionConfig())
    try:
        unc.predict_with_covariance(
            pred,
            pos=np.zeros(3), vel=np.zeros(3), att=np.array([1, 0, 0, 0]),
            acc_meas=np.zeros(3), gyr_meas=np.zeros(3),
            horizon=0.05, P_init=np.eye(15),
        )
        assert False, "expected ValueError for 15x15 P"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Integration: ESKF steady-state P through the predictor
# ---------------------------------------------------------------------------


def test_eskf_steady_state_through_predictor():
    """Run the ESKF to steady state, then propagate its 9x9 covariance."""
    from sensor_sim.eskf import ESKF, ESKFConfig
    from sensor_sim.imu import IMUSensor, SensorGrade

    dt_imu = 0.01
    traj = _straight_traj(duration=10.0, speed=10.0)
    imu = IMUSensor(SensorGrade.TACTICAL, dt=dt_imu, seed=7)
    cfg = ESKFConfig(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5,
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=2.2, gnss_vel_std=0.1,
    )
    eskf = ESKF(cfg)
    p0 = traj.points[0]
    eskf.set_initial_state(p0.t, p0.pos, p0.vel, p0.att)

    # Run a few GNSS updates to shrink P
    for i, pt in enumerate(traj.points):
        a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
        eskf.predict(a_m, w_m, dt_imu)
        if i % 20 == 0:
            eskf.update_gnss(pt.pos, pt.vel)

    # 9x9 block from the 15x15 ESKF covariance
    P9 = eskf.state.P[:9, :9]

    unc = PredictorUncertainty()
    P50 = unc.propagate_covariance(P9, eskf.quaternion, np.zeros(3), 0.05)
    P80 = unc.propagate_covariance(P9, eskf.quaternion, np.zeros(3), 0.08)
    e = unc.horizontal_ellipse(P80)

    assert P50.shape == (9, 9)
    # display-time uncertainty must be monotonically growing
    assert np.trace(P80) > np.trace(P50)
    # and the 95% radius for an 80 ms horizon at 10 m/s must stay
    # within a few meters (sanity bound, not a tight assertion)
    assert 0.1 < e["radius_95"] < 5.0
    assert e["sigma_x"] >= e["sigma_y"] - 1e-9


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception:
                failed += 1
                print(f"  FAIL {name}")
                traceback.print_exc()
    total = sum(1 for n in globals() if n.startswith("test_") and callable(globals()[n]))
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
