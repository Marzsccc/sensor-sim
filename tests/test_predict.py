"""
Tests for sensor-sim display-time pose prediction & compensation.

Run: python3 tests/test_predict.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.predict import (
    PosePredictor,
    PredictionConfig,
    DisplayPipeline,
    _quat_yaw,
)
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.utils import euler_to_quat, quat_to_euler, quat_rotate
from sensor_sim.latency import LatencyModel


def _build_straight_traj(duration=10.0, speed=10.0, accel=0.0):
    """Helper: straight-line trajectory along +x."""
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    # With acceleration the endpoint velocity changes
    p1 = p0 + v0 * duration + 0.5 * accel * duration**2 * np.array([1, 0, 0])
    v1 = v0 + accel * duration * np.array([1, 0, 0])
    waypoints = [
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=duration, pos=p1, vel=v1, att=q0, omega=np.zeros(3)),
    ]
    return Trajectory(waypoints, dt=0.01)


def _build_turning_traj(duration=6.0, speed=10.0, yaw_rate=0.3):
    """Helper: constant-yaw-rate turn (arc) with velocity along the tangent."""
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    q1 = euler_to_quat(0, 0, yaw_rate * duration)
    # Velocity at each endpoint follows the arc tangent.
    v1 = np.array([speed * np.cos(yaw_rate * duration),
                   speed * np.sin(yaw_rate * duration), 0.0])
    # Endpoint position on the arc (integration of the tangent).
    n = 400
    yaws = np.linspace(0, yaw_rate * duration, n)
    p1 = np.array([0.0, 0.0, 0.0])
    for i in range(1, n):
        p1 += np.array([speed * np.cos(yaws[i]), speed * np.sin(yaws[i]), 0]) * (duration / n)
    waypoints = [
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.array([0, 0, yaw_rate])),
        Waypoint(t=duration, pos=p1, vel=v1, att=q1, omega=np.array([0, 0, yaw_rate])),
    ]
    return Trajectory(waypoints, dt=0.01)


def test_static_pose_imu_propagation():
    """Static pose: IMU with zero accel/gyro must not move the state."""
    pred = PosePredictor(PredictionConfig(dt=0.01))
    pos = np.zeros(3)
    vel = np.zeros(3)
    att = euler_to_quat(0, 0, 0)
    # IMU measures specific force = gravity in body frame
    acc_meas = np.array([0.0, 0.0, 9.81])
    gyr = np.zeros(3)
    p, v, q = pred.propagate_imu(pos, vel, att, acc_meas, gyr, horizon=0.1)
    assert np.allclose(p, pos, atol=1e-6), f"Static pos moved: {p}"
    assert np.allclose(v, vel, atol=1e-6), f"Static vel changed: {v}"
    assert np.allclose(q, att, atol=1e-6), f"Static att changed: {q}"


def test_constant_velocity_imu_propagation():
    """Const-velocity straight motion: IMU propagates position correctly."""
    pred = PosePredictor(PredictionConfig(dt=0.005))
    speed = 10.0
    horizon = 0.1
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([speed, 0.0, 0.0])
    att = euler_to_quat(0, 0, 0)
    # Body accel = gravity only (no acceleration)
    acc_meas = np.array([0.0, 0.0, 9.81])
    gyr = np.zeros(3)
    p, v, q = pred.propagate_imu(pos, vel, att, acc_meas, gyr, horizon=horizon)
    assert np.allclose(p, [speed * horizon, 0, 0], atol=1e-3), f"Pos wrong: {p}"
    assert np.allclose(v, vel, atol=1e-3), f"Vel wrong: {v}"
    assert np.allclose(q, att, atol=1e-6), f"Att wrong: {q}"


def test_accelerating_imu_propagation():
    """Accelerating straight motion: IMU captures the extra displacement."""
    pred = PosePredictor(PredictionConfig(dt=0.005))
    speed, accel, horizon = 10.0, 2.0, 0.1
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([speed, 0.0, 0.0])
    att = euler_to_quat(0, 0, 0)
    # Body accel = gravity + forward accel
    acc_meas = np.array([accel, 0.0, 9.81])
    gyr = np.zeros(3)
    p, v, q = pred.propagate_imu(pos, vel, att, acc_meas, gyr, horizon=horizon)
    expected_disp = speed * horizon + 0.5 * accel * horizon**2
    assert np.allclose(p, [expected_disp, 0, 0], atol=5e-3), f"Pos wrong: {p}"
    assert np.allclose(v, [speed + accel * horizon, 0, 0], atol=5e-3), f"Vel wrong: {v}"


def test_wheel_propagation_straight():
    """Wheel model: constant speed/yaw_rate integrates to arc endpoint."""
    pred = PosePredictor(PredictionConfig(dt=0.005))
    speed, yaw_rate, horizon = 10.0, 0.0, 0.2
    p, yaw = pred.propagate_wheel(np.zeros(2), 0.0, speed, yaw_rate, horizon)
    assert np.allclose(p, [speed * horizon, 0, 0], atol=1e-3), f"Pos wrong: {p}"
    assert abs(yaw) < 1e-9


def test_wheel_propagation_turn():
    """Wheel model: turning at constant yaw rate traces an arc."""
    pred = PosePredictor(PredictionConfig(dt=0.005))
    speed, yaw_rate, horizon = 10.0, 0.5, 0.4
    p, yaw = pred.propagate_wheel(np.zeros(2), 0.0, speed, yaw_rate, horizon)
    # Radius = speed / yaw_rate
    R = speed / yaw_rate
    theta = yaw_rate * horizon
    expected = np.array([R * np.sin(theta), R * (1 - np.cos(theta)), 0])
    assert np.allclose(p, expected, atol=1e-2), f"Arc pos wrong: {p} vs {expected}"
    assert abs(yaw - theta) < 1e-6, f"Yaw wrong: {yaw} vs {theta}"


def test_predict_uses_wheel_when_enabled():
    """predict() blends wheel yaw/position when enabled."""
    pred = PosePredictor(PredictionConfig(dt=0.005, use_wheel=True))
    pos = np.zeros(3)
    vel = np.array([10.0, 0.0, 0.0])
    att = euler_to_quat(0, 0, 0)
    acc_meas = np.array([0.0, 0.0, 9.81])
    gyr = np.zeros(3)
    out = pred.predict(pos, vel, att, acc_meas, gyr, horizon=0.1,
                       wheel_speed=10.0, wheel_yaw_rate=0.0)
    # Wheel drives position exactly
    assert np.allclose(out["pos"], [1.0, 0, 0], atol=1e-2), f"Pos wrong: {out['pos']}"


def test_display_pipeline_compensation_beats_naive():
    """End-to-end: compensated renderer must beat naive renderer.

    The naive renderer uses the latest fused pose which lags by the total
    latency; the compensated renderer propagates it forward.  For a 50ms
    camera + 60Hz display (~66ms total) at 20 m/s, naive error is ~1.3 m
    while compensated error should drop well below 0.5 m.
    """
    traj = _build_straight_traj(duration=10.0, speed=20.0)
    pipe = DisplayPipeline(
        traj=traj,
        sensors={
            "camera": LatencyModel(fixed_delay_s=0.05, jitter_std_s=0.005, seed=11),
            "gnss":   LatencyModel(fixed_delay_s=0.20, jitter_std_s=0.02, loss_prob=0.03, seed=12),
            "wheel":  LatencyModel(fixed_delay_s=0.005, jitter_std_s=0.001, seed=13),
        },
        display_rate=60.0,
        total_delay_est=None,   # auto: camera delay + 1 frame
    )
    res = pipe.run()
    # Sanity: naive error ~ speed * horizon
    assert res["naive_pos_err_mean"] > 0.5, f"Naive error too small: {res['naive_pos_err_mean']:.3f}"
    # Compensation must be significantly better
    assert res["comp_pos_err_mean"] < 0.5 * res["naive_pos_err_mean"], (
        f"Compensation not better: naive={res['naive_pos_err_mean']:.3f} "
        f"comp={res['comp_pos_err_mean']:.3f}"
    )
    # Heading errors must also improve
    assert res["comp_heading_err_mean"] <= res["naive_heading_err_mean"] + 1e-6, (
        f"Heading not better: naive={res['naive_heading_err_mean']:.3f} "
        f"comp={res['comp_heading_err_mean']:.3f}"
    )


def test_display_pipeline_zero_delay():
    """With zero latency there is nothing to compensate: errors ~ 0."""
    traj = _build_straight_traj(duration=5.0, speed=10.0)
    pipe = DisplayPipeline(
        traj=traj,
        sensors={
            "camera": LatencyModel(fixed_delay_s=0.0, seed=1),
            "gnss":   LatencyModel(fixed_delay_s=0.0, seed=2),
            "wheel":  LatencyModel(fixed_delay_s=0.0, seed=3),
        },
        display_rate=60.0,
        total_delay_est=0.0,
    )
    res = pipe.run()
    assert res["naive_pos_err_mean"] < 0.05, f"Zero-delay naive err too high: {res['naive_pos_err_mean']}"
    assert res["comp_pos_err_mean"] < 0.05, f"Zero-delay comp err too high: {res['comp_pos_err_mean']}"


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
    print(f"\n{sum(1 for n in globals() if n.startswith('test_') and callable(globals()[n])) - failed}/{sum(1 for n in globals() if n.startswith('test_') and callable(globals()[n]))} passed")
    sys.exit(1 if failed else 0)
