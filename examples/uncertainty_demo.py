"""
Display-time uncertainty propagation demo (v0.7.0).

Shows how the ESKF steady-state covariance grows through the display-time
prediction, and how to render an uncertainty ellipse on the AR-HUD.

Pipeline:
  1. Run the ESKF (GNSS+IMU loose coupling) to steady state on a straight
     run + a turn.
  2. At the end, take the ESKF 9x9 [p, v, theta] covariance block.
  3. Propagate it forward by an 80 ms display horizon with
     PredictorUncertainty, including process noise + model error.
  4. Print the display-time ellipse (1-sigma axes, 95% radius).

Usage:  .venv/bin/python examples/uncertainty_demo.py [--horizon 0.08]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.eskf import ESKF, ESKFConfig
from sensor_sim.predict import (
    PredictorUncertainty,
    UncertaintyConfig,
    PosePredictor,
    PredictionConfig,
)
from sensor_sim.utils import quat_to_euler


def build_trajectory(dt: float) -> Trajectory:
    """Straight + 90-deg right turn + straight (60 s)."""
    wp = [
        Waypoint(t=0.0,  pos=np.array([0, 0, 0]),      vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),           omega=np.array([0, 0, 0])),
        Waypoint(t=15.0, pos=np.array([150, 0, 0]),    vel=np.array([10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),           omega=np.array([0, 0, 0])),
        Waypoint(t=20.0, pos=np.array([150, 50, 0]),   vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                 omega=np.array([0, 0, np.pi/10])),
        Waypoint(t=40.0, pos=np.array([150, 250, 0]),  vel=np.array([0, 10, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                 omega=np.array([0, 0, 0])),
        Waypoint(t=50.0, pos=np.array([50, 250, 0]),   vel=np.array([-10, 0, 0]),
                 att=np.array([np.cos(np.pi/4), 0, 0, -np.sin(np.pi/4)]),
                 omega=np.array([0, 0, -np.pi/10])),
        Waypoint(t=60.0, pos=np.array([-50, 250, 0]),  vel=np.array([-10, 0, 0]),
                 att=np.array([1, 0, 0, 0]),           omega=np.array([0, 0, 0])),
    ]
    return Trajectory(wp, dt=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dt_imu = 0.01
    traj = build_trajectory(dt_imu)
    imu = IMUSensor(SensorGrade.TACTICAL, dt=dt_imu, seed=args.seed)
    cfg = ESKFConfig(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5,
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=2.2, gnss_vel_std=0.1,
    )
    eskf = ESKF(cfg)
    p0 = traj.points[0]
    eskf.set_initial_state(p0.t, p0.pos + np.array([3, -2, 1.0]), p0.vel, p0.att)

    # Run ESKF to steady state
    for i, pt in enumerate(traj.points):
        a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
        eskf.predict(a_m, w_m, dt_imu)
        if i % 20 == 0:
            eskf.update_gnss(pt.pos, pt.vel)

    # 9x9 [p, v, theta] covariance block from the ESKF
    P9 = eskf.state.P[:9, :9]

    unc = PredictorUncertainty(UncertaintyConfig())
    pred = PosePredictor(PredictionConfig(use_wheel=True))

    # Latest IMU reading at the final state
    pt_last = traj.points[-1]
    acc_m, gyr_m = imu.measure(traj.body_accel(pt_last), pt_last.omega)

    res = unc.predict_with_covariance(
        pred,
        pos=eskf.position, vel=eskf.velocity, att=eskf.quaternion,
        acc_meas=acc_m, gyr_meas=gyr_m,
        horizon=args.horizon, P_init=P9,
        wheel_speed=10.0, wheel_yaw_rate=0.0,
    )

    print(f"=== Display-time uncertainty (horizon={args.horizon*1000:.0f} ms) ===")
    print(f"ESKF settled pos std:      {np.sqrt(P9[0,0]):.3f} m   vel std {np.sqrt(P9[3,3]):.3f} m/s")
    print(f"ESKF heading std:          {np.rad2deg(np.sqrt(P9[6,6])):.3f} deg")
    print(f"\nDisplay-time covariance (from ESKF P -> predictor):")
    print(f"  pos std   : {np.sqrt(res['P_disp'][0,0]):.3f} / {np.sqrt(res['P_disp'][1,1]):.3f} / {np.sqrt(res['P_disp'][2,2]):.3f} m")
    print(f"  vel std   : {np.sqrt(res['P_disp'][3,3]):.3f} / {np.sqrt(res['P_disp'][4,4]):.3f} / {np.sqrt(res['P_disp'][5,5]):.3f} m/s")
    print(f"  heading std: {np.rad2deg(np.sqrt(res['P_disp'][6,6])):.3f} deg")
    print(f"\nHorizontal uncertainty ellipse (1-sigma):")
    e = res["ellipse"]
    print(f"  sigma_x   : {e['sigma_x']:.3f} m")
    print(f"  sigma_y   : {e['sigma_y']:.3f} m")
    print(f"  rotation  : {e['rotation_deg']:.1f} deg")
    print(f"  area      : {e['area']:.3f} m^2")
    print(f"  95% radius: {e['radius_95']:.3f} m")
    print(f"\nMean display-time pos: {res['pos'].round(3)}")


if __name__ == "__main__":
    main()
