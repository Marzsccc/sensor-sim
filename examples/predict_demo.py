"""
Display-time pose prediction demo for sensor-sim.

Demonstrates the AR-HUD delay-compensation problem end-to-end:

  * A camera (50 ms) + GNSS (200 ms) + wheel (5 ms) pipeline with realistic
    latencies.
  * The AR content is anchored to the camera capture time; the display
    happens ~67 ms later (camera latency + one 60 Hz frame).
  * Naive renderer uses the latest fused pose -> AR markers lag the real
    world by ~speed * 67 ms (1.3 m at 20 m/s).
  * Compensated renderer predicts the pose forward to display time using
    IMU + wheel odometry -> error drops to ~0.1 m.

Two scenarios: straight-line and a constant-yaw-rate turn.

Run: python3 examples/predict_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.predict import DisplayPipeline
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.utils import euler_to_quat
from sensor_sim.latency import LatencyModel


def _build_straight(duration=10.0, speed=20.0):
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    return Trajectory([
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=duration, pos=p0 + v0 * duration, vel=v0, att=q0, omega=np.zeros(3)),
    ], dt=0.01)


def _build_turn(duration=6.0, speed=15.0, yaw_rate=0.35):
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    q1 = euler_to_quat(0, 0, yaw_rate * duration)
    # Velocity follows the arc tangent at the endpoint.
    v1 = np.array([speed * np.cos(yaw_rate * duration),
                   speed * np.sin(yaw_rate * duration), 0.0])
    n = 400
    yaws = np.linspace(0, yaw_rate * duration, n)
    p1 = np.array([0.0, 0.0, 0.0])
    for i in range(1, n):
        p1 += np.array([speed * np.cos(yaws[i]), speed * np.sin(yaws[i]), 0]) * (duration / n)
    return Trajectory([
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.array([0, 0, yaw_rate])),
        Waypoint(t=duration, pos=p1, vel=v1, att=q1, omega=np.array([0, 0, yaw_rate])),
    ], dt=0.01)


def _run(name, traj):
    pipe = DisplayPipeline(
        traj=traj,
        sensors={
            "camera": LatencyModel(fixed_delay_s=0.05, jitter_std_s=0.005, seed=11),
            "gnss":   LatencyModel(fixed_delay_s=0.20, jitter_std_s=0.02, loss_prob=0.03, seed=12),
            "wheel":  LatencyModel(fixed_delay_s=0.005, jitter_std_s=0.001, seed=13),
        },
        display_rate=60.0,
    )
    res = pipe.run()
    h = res["horizon_s"] * 1e3
    print(f"=== {name} (horizon {h:.0f} ms) ===")
    print(f"  naive renderer     pos err mean {res['naive_pos_err_mean']*100:6.1f} cm  "
          f"max {res['naive_pos_err_max']*100:6.1f} cm")
    print(f"  compensated        pos err mean {res['comp_pos_err_mean']*100:6.1f} cm  "
          f"max {res['comp_pos_err_max']*100:6.1f} cm")
    print(f"  naive heading err  {res['naive_heading_err_mean']:7.3f} deg  "
          f"comp {res['comp_heading_err_mean']:7.3f} deg")
    impr = (1 - res["comp_pos_err_mean"] / res["naive_pos_err_mean"]) * 100
    print(f"  -> position error reduced by {impr:.1f}%\n")
    return res


if __name__ == "__main__":
    print("AR-HUD display-time pose prediction\n" + "=" * 40)
    _run("Straight line @ 20 m/s", _build_straight())
    _run("Turn @ 15 m/s, 0.35 rad/s", _build_turn())
