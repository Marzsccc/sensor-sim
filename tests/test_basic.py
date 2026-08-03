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
