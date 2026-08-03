"""
Basic tests for sensor-sim modules.
Run: python3 tests/test_basic.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade
from sensor_sim.utils import euler_to_quat, quat_to_euler, quat_rotate


def _build_straight_traj(duration=10.0, speed=10.0):
    """Helper: straight line trajectory eastward."""
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    p1 = p0 + v0 * duration
    waypoints = [
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=duration, pos=p1, vel=v0, att=q0, omega=np.zeros(3)),
    ]
    return Trajectory(waypoints, dt=0.01)


def test_trajectory_length():
    """Trajectory produces correct number of points."""
    traj = _build_straight_traj(10.0)
    assert len(traj) == 1001, f"Expected 1001 points, got {len(traj)}"


def test_trajectory_position():
    """Trajectory position is correct at endpoints."""
    traj = _build_straight_traj(10.0, 10.0)
    pts = traj.points
    assert np.allclose(pts[0].pos, [0, 0, 0]), f"Start pos wrong: {pts[0].pos}"
    assert np.allclose(pts[-1].pos, [100, 0, 0], atol=1e-6), f"End pos wrong: {pts[-1].pos}"


def test_trajectory_body_accel_static():
    """Stationary trajectory yields accel = [0, 0, g] in body frame."""
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.zeros(3)
    q0 = euler_to_quat(0, 0, 0)
    waypoints = [
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=1, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
    ]
    traj = Trajectory(waypoints, dt=0.01)
    body_acc = traj.body_accels()
    # Level attitude: body accel ≈ [0, 0, g] (minus gravity reaction)
    assert np.allclose(body_acc[50], [0, 0, 9.81], atol=0.01), \
        f"Body accel wrong: {body_acc[50]}"


def test_imu_noise():
    """IMU produces non-identical measurements."""
    imu = IMUSensor(SensorGrade.CONSUMER, dt=0.01, seed=0)
    acc_true = np.array([0.0, 0.0, 9.81])
    gyr_true = np.array([0.0, 0.0, 0.0])
    m1, _ = imu.measure(acc_true, gyr_true)
    m2, _ = imu.measure(acc_true, gyr_true)
    assert not np.allclose(m1, m2), "IMU measurements should have noise"


def test_imu_bias_constant():
    """IMU bias is constant across a single run."""
    imu = IMUSensor(SensorGrade.CONSUMER, dt=0.01, seed=42)
    acc_true = np.array([0.0, 0.0, 9.81])
    gyr_true = np.array([0.0, 0.0, 0.0])
    biases = []
    for _ in range(100):
        imu.measure(acc_true, gyr_true)
        biases.append(imu._bias_acc.copy())
    biases = np.array(biases)
    assert np.allclose(biases.std(axis=0), 0, atol=1e-10), "Static bias should not change"


def test_imu_bias_random_walk():
    """IMU bias instability evolves (small random walk)."""
    imu = IMUSensor(SensorGrade.CONSUMER, dt=0.01, seed=42)
    imu.reset_bias_walk()
    acc_true = np.array([0.0, 0.0, 9.81])
    gyr_true = np.array([0.0, 0.0, 0.0])
    rw_vals = []
    for _ in range(1000):
        imu.measure(acc_true, gyr_true)
        rw_vals.append(imu._rw_acc.copy())
    rw_vals = np.array(rw_vals)
    # Should have non-zero variance (random walk is active)
    assert rw_vals.std(axis=0).max() > 0, "Bias random walk should produce variation"


def test_gnss_output_rate():
    """GNSS outputs at the correct rate."""
    gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=0.01, seed=42)
    pos = np.array([0.0, 0.0, 0.0])
    outputs = 0
    for _ in range(1000):  # 10 seconds
        p, _, valid = gnss.measure(pos)
        if valid:
            outputs += 1
    # 5 Hz output rate, but dropouts reduce actual outputs
    # Expected: ~20-50 outputs in 10s (depending on dropout luck)
    assert 15 <= outputs <= 55, f"Expected 15-55 GNSS outputs in 10s, got {outputs}"


def test_gnss_noise():
    """GNSS measurements are noisy."""
    gnss = GNSSSensor(GNSSGrade.CONSUMER, dt=0.01, seed=0)
    pos = np.array([100.0, 200.0, 50.0])
    pos_vals = []
    for _ in range(10000):
        p, _, valid = gnss.measure(pos)
        if valid:
            pos_vals.append(p[:2])  # horizontal
    pos_vals = np.array(pos_vals)
    # Consumer GNSS: horizontal error std ~3m
    errors = np.linalg.norm(pos_vals - pos[:2], axis=1)
    assert 2.0 < errors.std() < 8.0, f"GNSS error std {errors.std():.2f} out of expected range"


def test_utils_quat_roundtrip():
    """Euler → quat → Euler roundtrip."""
    rpy = np.array([0.1, -0.2, 1.5])
    q = euler_to_quat(*rpy)
    rpy2 = quat_to_euler(q)
    assert np.allclose(rpy, rpy2, atol=1e-6), f"Roundtrip failed: {rpy} vs {rpy2}"


def test_utils_rotate():
    """Quaternion rotation: rotate z-axis by 90° yaw → x-axis."""
    q = euler_to_quat(0, 0, np.pi/2)  # 90° yaw
    v = np.array([0, 0, 1])  # z-axis in world
    result = quat_rotate(q, v)
    # z-axis rotated 90° around z → still z-axis. Wait, rotation by yaw around z
    # doesn't change z-vector. Let me use a better test.
    # Rotate x-axis by 90° yaw → y-axis
    v2 = np.array([1, 0, 0])
    result2 = quat_rotate(q, v2)
    assert np.allclose(result2, [0, 1, 0], atol=1e-6), f"Rotation wrong: {result2}"


def test_wheel_ackermann_straight():
    """Wheel odometry on straight road: measured speed ≈ true speed."""
    from sensor_sim.wheel import WheelOdometry, WheelGrade
    from sensor_sim.utils import quat_rotate
    traj = _build_straight_traj(10.0, 20.0)
    wo = WheelOdometry(WheelGrade.PRECISION, dt=0.01, seed=1)
    # Body-frame velocity: rotate world velocity by conjugate quat
    v_world = traj.velocities
    v_body = np.array([quat_rotate(q, v) for q, v in zip(traj.attitudes, v_world)])
    v, yr, valid = wo.measure_vectorized(v_body, traj.body_omegas())
    assert valid.sum() > 0
    v_ok = v[valid]
    # Precision grade: <0.5% error
    err = np.abs(v_ok - 20.0) / 20.0
    assert err.mean() < 0.005, f"Wheel speed error too large: {err.mean()*100:.2f}%"


def test_wheel_differential_yaw():
    """Differential model: turning produces left≠right wheel speeds."""
    from sensor_sim.wheel import WheelOdometry, WheelSpec
    spec = WheelSpec(
        radius_error_pct=0.1, tick_resolution_m=0.0005,
        slip_sigma_pct=0.05, slip_tau_s=20.0,
        output_rate_hz=100.0, ackermann=False,
    )
    # Turning right (positive yaw): right wheel must be faster than left.
    # Repeat with several seeds to be robust against random radius errors.
    lags = []
    for seed in range(10):
        wo = WheelOdometry(spec, dt=0.01, seed=seed)
        v_l, v_r, ok = wo.measure(np.array([10.0, 0, 0]), np.array([0, 0, 0.5]))
        lags.append(v_r - v_l)
    assert np.median(lags) > 0, f"Turning: right wheel should exceed left, lags={lags}"


def test_wheel_output_rate():
    """Wheel odometry output rate matches spec."""
    from sensor_sim.wheel import WheelOdometry, WheelGrade
    traj = _build_straight_traj(10.0)
    wo = WheelOdometry(WheelGrade.AUTOMOTIVE, dt=0.01, seed=4)  # 50 Hz
    v_body = np.tile(np.array([10.0, 0, 0]), (len(traj), 1))
    v, yr, valid = wo.measure_vectorized(v_body, traj.body_omegas())
    n_out = valid.sum()
    expected = 10.0 * 50.0  # 10s @ 50Hz = 500 samples
    assert abs(n_out - expected) / expected < 0.05, \
        f"Output count {n_out}, expected ~{expected}"


def test_trajectory_const_speed_turn():
    """const_speed keeps speed magnitude AND body-x velocity constant in turns."""
    from sensor_sim.utils import quat_conjugate
    speed = 20.0
    wps = [
        Waypoint(t=0.0, pos=np.array([0, 0, 0]), vel=np.array([speed, 0, 0]),
                 att=euler_to_quat(0, 0, 0), omega=np.zeros(3)),
        Waypoint(t=5.0, pos=np.array([speed*5, 0, 0]), vel=np.array([speed, 0, 0]),
                 att=euler_to_quat(0, 0, 0), omega=np.zeros(3)),
        Waypoint(t=10.0, pos=np.array([speed*5 + 50, 50, 0]),
                 vel=np.array([speed*np.cos(0.5), speed*np.sin(0.5), 0]),
                 att=euler_to_quat(0, 0, 0.5), omega=np.zeros(3)),
    ]
    traj = Trajectory(wps, dt=0.01, const_speed=speed)
    v_world = traj.velocities
    speed_mag = np.linalg.norm(v_world, axis=1)
    assert np.allclose(speed_mag, speed, atol=1e-6), \
        f"Speed magnitude varies: {speed_mag.min():.3f}-{speed_mag.max():.3f}"
    # Nonholonomic: body-x velocity must equal speed everywhere
    v_body = np.array([quat_rotate(quat_conjugate(q), v)
                       for q, v in zip(traj.attitudes, v_world)])
    bx = v_body[:, 0]
    assert np.allclose(bx, speed, atol=0.05), \
        f"Body-x velocity varies: {bx.min():.3f}-{bx.max():.3f}"


if __name__ == "__main__":
    tests = [
        test_trajectory_length,
        test_trajectory_position,
        test_trajectory_body_accel_static,
        test_imu_noise,
        test_imu_bias_constant,
        test_imu_bias_random_walk,
        test_gnss_output_rate,
        test_gnss_noise,
        test_utils_quat_roundtrip,
        test_utils_rotate,
        test_wheel_ackermann_straight,
        test_wheel_differential_yaw,
        test_wheel_output_rate,
        test_trajectory_const_speed_turn,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
        except Exception as e:
            print(f"  ✗ {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
