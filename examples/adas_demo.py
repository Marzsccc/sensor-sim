"""Example: ADAS marker projection under display-time latency (v0.9.0).

Shows why AR-HUD markers must be rendered with the display-time predicted
ego pose, not the latest fused pose: at 20 m/s with a 50 ms camera delay the
naive markers lag over 1.3 m behind the real objects, while the compensated
render collapses the error to ~0.1 m. Also prints the uncertainty fade alpha.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.adas import (
    AdasPipeline,
    lead_vehicle_marker,
    hazard_marker,
    lane_line_marker,
)


def main():
    # Straight road, 20 m/s, 10 s ---------------------------------------
    speed, dur, dt = 20.0, 10.0, 0.01
    ts = np.arange(0.0, dur, dt)
    wps = [
        Waypoint(
            t=t,
            pos=np.array([speed * t, 0.0, 0.0]),
            vel=np.array([speed, 0.0, 0.0]),
            att=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
        )
        for t in ts
    ]
    traj = Trajectory(wps, dt=dt)

    # World-anchored ADAS markers ---------------------------------------
    markers = [
        lead_vehicle_marker(np.array([25.0, 2.0, 0.0])),          # ACC target
        hazard_marker(np.array([15.0, 0.0, 0.0])),                # FCW/AEB
        lane_line_marker(np.array([0.0, -1.75, 0.0]),
                         np.array([80.0, -1.75, 0.0]), "lane_right"),
        lane_line_marker(np.array([0.0, 1.75, 0.0]),
                         np.array([80.0, 1.75, 0.0]), "lane_left"),
    ]

    pipe = AdasPipeline(traj=traj, markers=markers,
                        sensor_delay_s=0.05, display_rate=60.0, use_wheel=True)

    # 9x9 ESKF [p, v, theta] covariance (position ~0.22 m 1-sigma)
    P = np.eye(9)
    P[0:3, 0:3] *= 0.05
    P[3:6, 3:6] *= 0.1
    P[6:9, 6:9] *= 1e-4

    res = pipe.run(P, sigma_scale=1.0, f_px=1500.0)

    print(f"Compensation horizon      : {res['horizon_s']*1000:.1f} ms "
          f"({pipe.sensor_delay_s*1000:.0f} ms camera + 1 display frame)")
    print(f"Mean marker fade alpha    : {res['fade_alpha_mean']:.3f} (1 = opaque)\n")
    print(f"{'marker':<13}{'range':>7}{'naive(m)':>10}{'comp(m)':>9}"
          f"{'naive(px)':>10}{'comp(px)':>9}")
    for key, m in res["markers"].items():
        npx = f"{m.get('naive_px_mean', float('nan')):8.1f}" if m.get("naive_px_mean") is not None else "     n/a"
        cpx = f"{m.get('comp_px_mean', float('nan')):8.1f}" if m.get("comp_px_mean") is not None else "     n/a"
        print(f"{key:<13}{m['range_mean']:7.1f}{m['naive_radial_mean']:10.3f}"
              f"{m['comp_radial_mean']:9.3f}{npx:>10}{cpx:>9}")
    print("\nNaive lag ~= delay*speed = %.2f m; compensated collapses to ~%.2f m"
          % (res["horizon_s"] * 20.0, res["summary"]["comp_radial_mean"]))


if __name__ == "__main__":
    main()
