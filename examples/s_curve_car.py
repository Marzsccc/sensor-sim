"""
Example: Car driving an S-curve at highway speed.
Simulates consumer IMU + automotive GNSS + plots everything.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade


def quat_from_yaw(yaw_deg: float) -> np.ndarray:
    """Quaternion from yaw-only rotation (z-axis)."""
    yaw = np.deg2rad(yaw_deg)
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def build_s_curve() -> Trajectory:
    """
    Build an S-curve trajectory:
      - Start at origin, heading east
      - Turn left, then right (S-shape)
      - Constant speed 25 m/s (~90 km/h)
      - Total duration: 20 seconds
    """
    speed = 25.0  # m/s
    T_turn = 5.0  # seconds per turn

    # Waypoint 0: origin, heading east
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = quat_from_yaw(0)

    # Waypoint 1: end of first turn (left), heading north-east
    # Circle of radius R = speed * T / (π/2) → R = 25*5/1.57 ≈ 79.6m
    R = speed * T_turn / (np.pi / 2)
    cx = 0.0
    cy = -R  # center of left turn circle (south of trajectory)
    angle1 = np.pi / 2  # 90° turn
    p1 = np.array([cx + R * np.sin(angle1), cy + R * np.cos(angle1), 0.0])
    v1 = np.array([speed * np.cos(np.pi/4), speed * np.sin(np.pi/4), 0.0])
    q1 = quat_from_yaw(45)

    # Waypoint 2: end of second turn (right), heading east again
    cx2 = p1[0] + R * np.cos(np.pi/4)
    cy2 = p1[1] - R * np.sin(np.pi/4) + R
    angle2 = np.pi / 2
    p2 = np.array([cx2 + R * np.sin(angle2), cy2 - R * np.cos(angle2), 0.0])
    v2 = np.array([speed, 0.0, 0.0])
    q2 = quat_from_yaw(0)

    # Waypoint 3: straight line, continue east
    p3 = p2 + v2 * 5.0
    v3 = v2.copy()
    q3 = q2.copy()

    waypoints = [
        Waypoint(t=0.0,  pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=5.0,  pos=p1, vel=v1, att=q1, omega=np.zeros(3)),
        Waypoint(t=10.0, pos=p2, vel=v2, att=q2, omega=np.zeros(3)),
        Waypoint(t=15.0, pos=p3, vel=v3, att=q3, omega=np.zeros(3)),
    ]

    return Trajectory(waypoints, dt=0.01)


def main():
    print("=" * 60)
    print("Sensor Simulator — S-Curve Car Example")
    print("=" * 60)

    # Build trajectory
    traj = build_s_curve()
    print(f"\n{traj}")
    print(f"Duration: {traj.waypoints[-1].t - traj.waypoints[0].t:.1f}s")
    print(f"Distance: {np.sum(np.linalg.norm(np.diff(traj.positions, axis=0), axis=1)):.0f}m")

    # Create sensors
    imu = IMUSensor(SensorGrade.CONSUMER, dt=traj.dt, seed=42)
    gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=traj.dt, seed=123)

    print(f"\nIMU: {imu}")
    print(f"  Acc noise σ: {imu._noise_sigma_acc:.4f} m/s²")
    print(f"  Gyr noise σ: {imu._noise_sigma_gyr:.6f} rad/s")
    print(f"  Acc bias:    {imu._bias_acc}")
    print(f"  Gyr bias:    {imu._bias_gyr}")

    print(f"\nGNSS: {gnss}")
    print(f"  Pos noise H:  {gnss.spec.pos_noise_h_m:.2f} m")
    print(f"  Pos noise V:  {gnss.spec.pos_noise_v_m:.2f} m")
    print(f"  Output rate:  {gnss.spec.output_rate_hz} Hz")

    # Simulate
    print("\nSimulating...")
    t = traj.ts
    body_acc = traj.body_accels()
    body_omega = traj.body_omegas()

    imu_acc = np.zeros((len(traj), 3))
    imu_gyr = np.zeros((len(traj), 3))
    gnss_pos_list = []
    gnss_valid = []

    for i, pt in enumerate(traj.points):
        a, w = imu.measure(body_acc[i], body_omega[i])
        imu_acc[i] = a
        imu_gyr[i] = w

    gnss_pos_list, _, gnss_valid = gnss.measure_vectorized(traj.positions)

    gnss_stats = gnss.get_stats()
    print(f"  GNSS dropouts: {gnss_stats['num_dropouts']}")
    print(f"  Total dropout time: {gnss_stats['total_dropout_time_s']:.1f}s")

    # ----- Plotting -----
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Sensor Simulator — Consumer IMU + Automotive GNSS on S-Curve", fontsize=14)

    # Row 1: Trajectory overview
    ax = axes[0, 0]
    ax.plot(traj.positions[:, 0], traj.positions[:, 1], "k-", linewidth=1.5, label="Ground truth")
    mask = gnss_valid
    gnss_arr = np.array([p for p, v in zip(gnss_pos_list, mask) if v and p is not None])
    if len(gnss_arr) > 0:
        ax.scatter(gnss_arr[:, 0], gnss_arr[:, 1], c="red", s=8, alpha=0.6, label="GNSS")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Trajectory (top-down)")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # IMU accelerometer
    ax = axes[0, 1]
    ax.plot(t, imu_acc[:, 0], linewidth=0.5, alpha=0.7, label="X")
    ax.plot(t, imu_acc[:, 1], linewidth=0.5, alpha=0.7, label="Y")
    ax.plot(t, imu_acc[:, 2], linewidth=0.5, alpha=0.7, label="Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s²")
    ax.set_title("IMU Accelerometer (body frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IMU gyroscope (zoom on turn region)
    ax = axes[0, 2]
    ax.plot(t, imu_gyr[:, 2], linewidth=0.5, label="ω_z (yaw rate)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("rad/s")
    ax.set_title("IMU Gyro Z-axis (yaw rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: error analysis
    # Position error (when GNSS is valid)
    ax = axes[1, 0]
    gnss_err = []
    gnss_t = []
    for i, (valid, pos_m) in enumerate(zip(gnss_valid, gnss_pos_list)):
        if valid and pos_m is not None:
            err = np.linalg.norm(pos_m[:2] - traj.positions[i, :2])  # horizontal error
            gnss_err.append(err)
            gnss_t.append(t[i])
    ax.plot(gnss_t, gnss_err, "r.-", markersize=3, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal error (m)")
    ax.set_title("GNSS Position Error (horizontal)")
    ax.grid(True, alpha=0.3)

    # Accel error (measured - true)
    ax = axes[1, 1]
    acc_err = np.linalg.norm(imu_acc - body_acc, axis=1)
    ax.plot(t, acc_err, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s²")
    ax.set_title("IMU Accelerometer Error Norm")
    ax.grid(True, alpha=0.3)

    # Velocity from GNSS vs ground truth
    ax = axes[1, 2]
    gnss_vel_list = []
    gnss_tv = []
    for i, (valid, vel_m) in enumerate(zip(gnss_valid, gnss_pos_list)):
        if valid and vel_m is not None:
            gnss_vel_list.append(np.linalg.norm(traj.velocities[i]))
            gnss_tv.append(t[i])
    if gnss_vel_list:
        ax.plot(t, np.linalg.norm(traj.velocities, axis=1), "k-", linewidth=1, label="True speed")
        ax.scatter(gnss_tv, gnss_vel_list, c="red", s=10, alpha=0.6, label="GNSS")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Speed: Truth vs GNSS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), "s_curve_output.png")
    plt.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")
    plt.close()
    print("\nDone! All modules working correctly.")


if __name__ == "__main__":
    main()
