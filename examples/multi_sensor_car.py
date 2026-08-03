#!/usr/bin/env python3
"""
Demo: S-curve vehicle with IMU + GNSS + Wheel odometry.
Shows how to build a multi-sensor dataset for state estimation.

Run: python3 examples/multi_sensor_car.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade
from sensor_sim.wheel import WheelOdometry, WheelGrade
from sensor_sim.utils import euler_to_quat, quat_rotate, quat_conjugate

def main():
    # ---- 1. S-curve trajectory: 90 km/h (25 m/s), 60 s ----
    # NOTE: velocity direction must follow the body yaw (holonomic consistency)
    wps = [
        Waypoint(t=0.0,  pos=np.array([0, 0, 0]),     vel=np.array([25, 0, 0]),
                 att=euler_to_quat(0, 0, 0),    omega=np.zeros(3)),
        Waypoint(t=10.0, pos=np.array([250, 0, 0]),   vel=np.array([25, 0, 0]),
                 att=euler_to_quat(0, 0, 0),    omega=np.zeros(3)),
        Waypoint(t=30.0, pos=np.array([500, 100, 0]),
                 vel=np.array([25 * np.cos(0.3), 25 * np.sin(0.3), 0]),
                 att=euler_to_quat(0, 0, 0.3),  omega=np.zeros(3)),
        Waypoint(t=50.0, pos=np.array([750, 100, 0]), vel=np.array([25, 0, 0]),
                 att=euler_to_quat(0, 0, 0),    omega=np.zeros(3)),
        Waypoint(t=60.0, pos=np.array([1000, 0, 0]),  vel=np.array([25, 0, 0]),
                 att=euler_to_quat(0, 0, 0),    omega=np.zeros(3)),
    ]
    traj = Trajectory(wps, dt=0.01, const_speed=25.0)
    t = traj.ts

    # ---- 2. IMU (tactical, 100 Hz) ----
    imu = IMUSensor(SensorGrade.TACTICAL, dt=0.01, seed=42)
    acc_true = traj.body_accels()
    gyr_true = traj.body_omegas()
    acc_meas = np.zeros_like(acc_true)
    gyr_meas = np.zeros_like(gyr_true)
    for i in range(len(acc_true)):
        acc_meas[i], gyr_meas[i] = imu.measure(acc_true[i], gyr_true[i])

    # ---- 3. GNSS (automotive, 5 Hz) ----
    gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=0.01, seed=7)
    gnss_pos, gnss_vel, gnss_valid = gnss.measure_vectorized(traj.positions, traj.velocities)

    # ---- 4. Wheel odometry (automotive, 50 Hz) ----
    wheel = WheelOdometry(WheelGrade.AUTOMOTIVE, dt=0.01, seed=11)
    # att is world<-body; world→body needs conjugate rotation
    v_world = traj.velocities
    v_body = np.array([quat_rotate(quat_conjugate(q), v)
                       for q, v in zip(traj.attitudes, v_world)])
    wheel_v, wheel_yr, wheel_valid = wheel.measure_vectorized(v_body, gyr_true)

    # ---- Summary ----
    print("=" * 55)
    print("多传感器数据集生成完成")
    print("=" * 55)
    print(f"轨迹: 60 s @ 100 Hz, {len(t)} 点, S 弯 90 km/h")
    print(f"  - IMU:   {len(acc_meas)} 样本 @ 100 Hz (战术级)")
    print(f"  - GNSS:  {gnss_valid.sum()} 样本 @ 5 Hz (车载级, 丢星 {gnss.num_dropouts} 次)")
    print(f"  - 轮速计: {wheel_valid.sum()} 样本 @ 50 Hz (车载级)")
    print(f"IMU 静态段加计: {acc_meas[:100].mean(axis=0).round(3)} (应≈[0,0,9.81])")
    print(f"轮速计速度: 均值 {wheel_v[wheel_valid].mean():.2f} m/s (真值 25)")

    # ---- 5. (可选) 保存为 npz ----
    out = "s_curve_dataset.npz"
    np.savez(out,
             t=t, pos=traj.positions, vel=traj.velocities, att=traj.attitudes,
             acc_meas=acc_meas, gyr_meas=gyr_meas,
             gnss_pos=np.array(gnss_pos, dtype=object), gnss_valid=gnss_valid,
             wheel_v=wheel_v, wheel_yr=wheel_yr, wheel_valid=wheel_valid)
    print(f"数据集已保存: {out}")

if __name__ == "__main__":
    main()
