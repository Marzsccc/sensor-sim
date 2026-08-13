"""
End-to-end GNSS+IMU loose-coupling fusion demo using ESKF.

Pipeline:
  1. Generate a ground-truth trajectory (constant speed + turn).
  2. Synthesize noisy IMU (tactical grade) and GNSS (automotive grade).
  3. Run ESKF: IMU mechanization at 100 Hz, GNSS position+velocity updates at 5 Hz.
  4. Compare:
     - pure IMU dead-reckoning (drifts without updates)
     - ESKF fused solution vs ground truth
  5. Report position/velocity/attitude errors, bias estimation quality.

Usage:  python examples/eskf_demo.py [--seed N]
"""

import argparse
import time

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade
from sensor_sim.eskf import ESKF, ESKFConfig
from sensor_sim.utils import quat_to_euler


def build_trajectory(dt: float) -> Trajectory:
    """Constant-speed 8-shape-ish trajectory with a turn, 60 s."""
    wp = [
        Waypoint(t=0.0,  pos=np.array([0, 0, 0]),      vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),           omega=np.array([0, 0, 0])),
        Waypoint(t=15.0, pos=np.array([150, 0, 0]),    vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),           omega=np.array([0, 0, 0])),
        # Right turn: 90 deg over 5 s at 10 m/s (arc radius ~31.8 m)
        Waypoint(t=20.0, pos=np.array([150, 50, 0]),   vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                 omega=np.array([0, 0, np.pi/10])),
        Waypoint(t=40.0, pos=np.array([150, 250, 0]),  vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                 omega=np.array([0, 0, 0])),
        # Left turn back
        Waypoint(t=50.0, pos=np.array([50, 250, 0]),   vel=np.array([-10, 0, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, -np.sin(np.pi/4)]),
                 omega=np.array([0, 0, -np.pi/10])),
        Waypoint(t=60.0, pos=np.array([-50, 250, 0]),  vel=np.array([-10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),           omega=np.array([0, 0, 0])),
    ]
    return Trajectory(wp, dt=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dt_imu = 0.01        # 100 Hz
    dt_gnss = 0.2        # 5 Hz (GNSS sensor output rate)
    T = 60.0

    traj = build_trajectory(dt_imu)
    imu = IMUSensor(SensorGrade.TACTICAL, dt=dt_imu, seed=args.seed)
    gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=dt_imu, seed=args.seed + 1)

    # --- ESKF setup ---------------------------------------------------
    # Tune measurement noise to match the automotive-grade sensor:
    # pos noise 1 m + multipath 2 m (Gauss-Markov, tau=20 s) -> use ~2.2 m
    # vel noise 0.05 m/s -> use 0.1 m/s to stay conservative.
    cfg = ESKFConfig(
        acc_noise_density=1e-2,      # m/s^2/sqrt(Hz)
        gyr_noise_density=1e-3,      # rad/s/sqrt(Hz)
        acc_bias_rw=1e-4,
        gyr_bias_rw=1e-5,
        init_pos_std=5.0,
        init_att_std_deg=5.0,
        gnss_pos_std=2.2,            # automotive-grade ~2.2 m incl. multipath
        gnss_vel_std=0.1,
    )
    eskf = ESKF(cfg)

    pts = traj.points
    p0 = pts[0]
    eskf.set_initial_state(p0.t, p0.pos + np.array([3, -2, 1.0]), p0.vel,
                           p0.att)   # 3 m initial position error

    # Pure dead-reckoning copy for comparison
    dr_p = p0.pos.copy(); dr_v = p0.vel.copy(); dr_q = p0.att.copy()

    errs_pos = []; errs_vel = []; errs_yaw = []
    dr_errs = []
    bias_est = []; bias_true = []

    gnss_next = dt_gnss
    t_start = time.time()

    for pt in pts:
        t = pt.t
        # IMU measurement
        a_true = traj.body_accel(pt)
        w_true = pt.omega
        a_m, w_m = imu.measure(a_true, w_true)

        # ESKF predict
        eskf.predict(a_m, w_m, dt_imu)

        # Dead reckoning (same mechanization, no updates)
        dr_q = _q_integrate(dr_q, w_m, dt_imu)
        R = _q_to_R(dr_q)
        dr_a_w = R @ (a_m - np.zeros(3)) + np.array([0, 0, -9.80665])
        dr_v += dr_a_w * dt_imu
        dr_p += dr_v * dt_imu

        # GNSS update (internal output-rate control)
        pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
        if valid:
            eskf.update_gnss(pos_m, vel_m)

        # Record errors at 2 Hz
        if abs(t - round(t / 0.5) * 0.5) < dt_imu * 0.5:
            e_pos = np.linalg.norm(eskf.position - pt.pos)
            e_vel = np.linalg.norm(eskf.velocity - pt.vel)
            e_yaw = abs(quat_to_euler(eskf.quaternion)[2] - quat_to_euler(pt.att)[2])
            errs_pos.append((t, e_pos)); errs_vel.append((t, e_vel)); errs_yaw.append((t, e_yaw))
            dr_errs.append((t, np.linalg.norm(dr_p - pt.pos)))
            bias_est.append((t, eskf.estimated_biases()[0].copy()))
            bias_true.append((t, imu.get_errors()["bias_acc"]))

    t_el = time.time() - t_start

    # --- Report --------------------------------------------------------
    print(f"=== ESKF GNSS+IMU Loose Coupling (seed={args.seed}, {t_el:.1f}s sim) ===")
    t = np.array([x[0] for x in errs_pos])
    ep = np.array([x[1] for x in errs_pos])
    ev = np.array([x[1] for x in errs_vel])
    ey = np.array([x[1] for x in errs_yaw])
    dr = np.array([x[1] for x in dr_errs])

    print(f"\nIMU-only dead reckoning position error:  mean {dr.mean():6.2f} m  max {dr.max():6.2f} m")
    print(f"ESKF position error:                    mean {ep.mean():6.3f} m  max {ep.max():6.3f} m   (RMS {np.sqrt((ep**2).mean()):.3f} m)")
    print(f"ESKF velocity error:                    mean {ev.mean():6.3f} m/s max {ev.max():6.3f} m/s")
    print(f"ESKF yaw error:                         mean {np.rad2deg(ey.mean()):6.3f} deg  max {np.rad2deg(ey.max()):6.3f} deg")

    # Bias estimation convergence (last 10 s)
    bt = np.array([x[0] for x in bias_est]); be = np.array([x[1] for x in bias_est]); btrue = np.array([x[1] for x in bias_true])
    mask = bt > T - 10
    if mask.any():
        print(f"\nAccel bias estimate (last 10 s):  est {be[mask][-1].round(4)}  true {btrue[mask][-1].round(4)}")
        print(f"Gyro bias estimate:               est {eskf.estimated_biases()[1].round(5)}  "
              f"true {imu.get_errors()['bias_gyr'].round(5)}")
    return ep.max()


def _q_integrate(q, w, dt):
    """Quaternion integration (wxyz), w = body angular rate."""
    from sensor_sim.eskf import _quat_exp
    from sensor_sim.utils import quat_multiply
    qn = quat_multiply(q, _quat_exp(w * dt))
    return qn / np.linalg.norm(qn)


def _q_to_R(q):
    from sensor_sim.utils import quat_to_rotmat
    return quat_to_rotmat(q)


if __name__ == "__main__":
    main()


def main_rtk():
    """RTK-grade variant: shows the fusion accuracy ceiling."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dt_imu = 0.01
    T = 60.0
    traj = build_trajectory(dt_imu)
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
    print(f"\n=== RTK variant (seed={args.seed}) ===")
    print(f"Settled position error (t>10s): RMS {np.sqrt((errs**2).mean()):.3f} m  max {errs.max():.3f} m")
    print(f"Updates: {eskf.state.n_updates}")


if __name__ == "__main__":
    main()
