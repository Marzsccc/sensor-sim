"""
RTK-grade GNSS + IMU fusion demo (shows the ESKF accuracy ceiling).

Same trajectory as eskf_demo.py but with RTK-grade GNSS (cm-level).
With a tactical IMU + 10 Hz RTK, the fused solution stays within ~10 cm
of ground truth, dominated by RTK noise rather than IMU drift.

Usage:  python examples/eskf_demo_rtk.py [--seed N]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade
from sensor_sim.eskf import ESKF, ESKFConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dt_imu = 0.01
    T = 60.0

    wp = [
        Waypoint(t=0.0, pos=np.array([0, 0, 0]), vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
        Waypoint(t=15.0, pos=np.array([150, 0, 0]), vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
        Waypoint(t=20.0, pos=np.array([150, 50, 0]), vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                 omega=np.array([0, 0, np.pi/10])),
        Waypoint(t=40.0, pos=np.array([150, 250, 0]), vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                 omega=np.array([0, 0, 0])),
        Waypoint(t=60.0, pos=np.array([150, 250, 0]), vel=np.array([0, 10, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
    ]
    traj = Trajectory(wp, dt=dt_imu)
    imu = IMUSensor(SensorGrade.TACTICAL, dt=dt_imu, seed=args.seed)
    gnss = GNSSSensor(GNSSGrade.RTK, dt=dt_imu, seed=args.seed + 1)

    cfg = ESKFConfig(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5,
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=0.3, gnss_vel_std=0.05,
    )
    eskf = ESKF(cfg)
    pts = traj.points
    p0 = pts[0]
    eskf.set_initial_state(p0.t, p0.pos, p0.vel, p0.att)

    errs = []
    for pt in pts:
        a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
        eskf.predict(a_m, w_m, dt_imu)
        pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
        if valid:
            eskf.update_gnss(pos_m, vel_m)
        if pt.t > 10.0:      # settled region only
            errs.append(np.linalg.norm(eskf.position - pt.pos))
    errs = np.array(errs)

    print(f"=== RTK GNSS+IMU Fusion (seed={args.seed}) ===")
    print(f"Settled position error (t>10 s): RMS {np.sqrt((errs**2).mean()):.3f} m  "
          f"max {errs.max():.3f} m")
    print(f"Updates: {eskf.state.n_updates}")


if __name__ == "__main__":
    main()
