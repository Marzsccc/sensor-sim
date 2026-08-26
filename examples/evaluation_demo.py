"""
Unified evaluation demo (v0.14.0).

Runs the classic eskf_demo pipeline ONCE but scores BOTH estimators
(ESKF fusion vs pure IMU dead reckoning) through the new
TrajectoryEvaluator, then:

  - prints per-estimator RMSE / NEES reports
  - attaches regression gates (pos RMSE < 2 m, yaw RMSE < 2 deg,
    covariance consistency)
  - renders a side-by-side comparison table

This is the "one number tells me if the system is healthy today"
workflow: swap any estimator in, keep the evaluator + gates fixed.

Usage:  .venv/bin/python examples/evaluation_demo.py [--seed N] [--plot]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade
from sensor_sim.eskf import ESKF, ESKFConfig
from sensor_sim.utils import quat_to_euler
from sensor_sim.evaluation import (
    TrajectoryEvaluator,
    add_gate,
    compare_reports,
)


def build_trajectory(dt: float) -> Trajectory:
    """Same shape as eskf_demo: straights + right/left 90-deg turns."""
    wp = [
        Waypoint(t=0.0,  pos=np.array([0, 0, 0]),   vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),        omega=np.array([0, 0, 0])),
        Waypoint(t=15.0, pos=np.array([150, 0, 0]), vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),        omega=np.array([0, 0, 0])),
        Waypoint(t=20.0, pos=np.array([150, 50, 0]), vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]),
                 omega=np.array([0, 0, np.pi / 10])),
        Waypoint(t=40.0, pos=np.array([150, 250, 0]), vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]),
                 omega=np.array([0, 0, 0])),
        Waypoint(t=50.0, pos=np.array([50, 250, 0]), vel=np.array([-10, 0, 0]),
                 att=np.array([np.cos(np.pi / 4), 0, 0, -np.sin(np.pi / 4)]),
                 omega=np.array([0, 0, -np.pi / 10])),
        Waypoint(t=60.0, pos=np.array([-50, 250, 0]), vel=np.array([-10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),        omega=np.array([0, 0, 0])),
    ]
    return Trajectory(wp, dt=dt)


def _quat_mul(a, b):
    """Hamilton product of two [w,x,y,z] quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _q_integrate(q, w, dt):
    """Small-angle quaternion propagation (exact axis-angle increment)."""
    angle = float(np.linalg.norm(w) * dt)
    if angle < 1e-12:
        return q / np.linalg.norm(q)
    axis = w / np.linalg.norm(w)
    dq = np.concatenate([[np.cos(angle / 2)],
                         axis * np.sin(angle / 2)])
    qn = _quat_mul(q, dq)
    return qn / np.linalg.norm(qn)


def _q_to_R(q):
    from sensor_sim.utils import quat_to_rotmat
    return quat_to_rotmat(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    dt_imu = 0.01
    dt_gnss = 0.2
    G = 9.80665

    traj = build_trajectory(dt_imu)
    imu = IMUSensor(SensorGrade.TACTICAL, dt=dt_imu, seed=args.seed)
    gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=dt_imu, seed=args.seed + 1)

    cfg = ESKFConfig(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5,
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=2.2, gnss_vel_std=0.1,
    )
    eskf = ESKF(cfg)

    pts = traj.points
    p0 = pts[0]
    eskf.set_initial_state(p0.t, p0.pos + np.array([3, -2, 1.0]), p0.vel,
                           p0.att)

    # dead-reckoning twin (no GNSS updates)
    dr_p = p0.pos.copy()
    dr_v = p0.vel.copy()
    dr_q = p0.att.copy()

    # ---- evaluators --------------------------------------------------
    ev_eskf = TrajectoryEvaluator("eskf-tac-auto")
    ev_dr = TrajectoryEvaluator("dead-reckoning")

    gnss_next = dt_gnss
    for pt in pts:
        t = pt.t
        a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
        eskf.predict(a_m, w_m, dt_imu)

        # DR mechanization (no updates)
        dr_q = _q_integrate(dr_q, w_m, dt_imu)
        R_dr = _q_to_R(dr_q)
        dr_v += (R_dr @ a_m + np.array([0, 0, -G])) * dt_imu
        dr_p += dr_v * dt_imu

        pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
        if valid and t >= gnss_next:
            eskf.update_gnss(pos_m, vel_m)
            gnss_next += dt_gnss

        # --- score both estimators at every IMU tick ---------------
        st = eskf.state

        # ESKF: position (+NEES from the reported covariance block)
        ev_eskf.add_pose(t, st.p, pt.pos,
                         P=st.P[np.ix_([0, 1], [0, 1])], dims=(0, 1))
        ev_eskf.add_velocity(t, st.v, pt.vel)
        yaw_est_e = float(quat_to_euler(st.q)[2])
        yaw_true = float(quat_to_euler(pt.att)[2])
        ev_eskf.add_yaw(t, yaw_est_e, yaw_true)

        # DR: same quantities, no covariance available -> NEES skipped
        ev_dr.add_pose(t, dr_p, pt.pos, dims=(0, 1))
        ev_dr.add_velocity(t, dr_v, pt.vel)
        ev_dr.add_yaw(t, float(quat_to_euler(dr_q)[2]), yaw_true)

    rep_eskf = ev_eskf.finalize()
    rep_dr = ev_dr.finalize()

    # ---- regression gates ---------------------------------------------
    # The gates encode "this estimator must stay useful for HUD work":
    add_gate(rep_eskf, "pos_err_m", "<", 2.0)       # m, fused solution
    add_gate(rep_eskf, "vel_err_ms", "<", 0.5)      # m/s
    add_gate(rep_eskf, "yaw_err_deg", "<", 2.0)     # deg
    # covariance sanity: NEES mean must stay bounded (loose gate; strict
    # chi2 consistency is reported separately in the summary)
    add_gate(rep_eskf, "nees:nees_hpos@nees_mean", "<", 100.0)

    print("=" * 64)
    print(" Unified ground-truth evaluation demo (v0.14.0)")
    print("=" * 64)
    print(f"Trajectory : {traj}")
    print(f"IMU        : tactical @ {1/dt_imu:.0f} Hz")
    print(f"GNSS       : automotive @ {1/dt_gnss:.0f} Hz\n")

    print(rep_eskf.summary())
    print()
    print(rep_dr.summary())
    print()
    print("-- side by side --")
    print(compare_reports(rep_eskf, rep_dr))

    ok = rep_eskf.all_gates_passed()
    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
