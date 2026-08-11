"""
Latency & time-sync demo for sensor-sim.

Shows how measurement delay + clock drift corrupt a naive fusion filter,
and how clock compensation recovers the true delay -- the core idea behind
AR-HUD display-time prediction.

Run: python3 examples/latency_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.latency import LatencyModel, SensorClock, TimeSync, LatencyScenario
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.utils import euler_to_quat


def main():
    # --- Build a simple accelerating trajectory ---------------------------
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    waypoints = [
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=10, pos=np.array([250.0, 0.0, 0.0]),
                 vel=np.array([25.0, 0.0, 0.0]), att=q0, omega=np.zeros(3)),
    ]
    traj = Trajectory(waypoints, dt=0.01)

    # --- Sensors with realistic latencies ---------------------------------
    scenario = LatencyScenario(sensors={
        "camera": LatencyModel(fixed_delay_s=0.05, jitter_std_s=0.005,
                               seed=11),                      # 30Hz camera
        "gnss":   LatencyModel(fixed_delay_s=0.20, jitter_std_s=0.02,
                               loss_prob=0.03, seed=12),      # GPS
        "wheel":  LatencyModel(fixed_delay_s=0.005, jitter_std_s=0.001,
                               seed=13),                      # wheel odom
    }, dt=0.01)

    streams = scenario.simulate(traj)
    summ = scenario.summary(streams)
    print("=== Per-sensor latency summary ===")
    for name, s in summ.items():
        print(f"  {name:8s} mean={s['mean_delay_s']*1e3:6.1f} ms  "
              f"max={s['max_delay_s']*1e3:6.1f} ms  loss={s['loss_rate']:.1%}")

    # --- Clock drift: sensor clock 50ppm ahead of host ---------------------
    print("\n=== Clock drift & compensation ===")
    clock = SensorClock(rate_ppm=50.0, wander_ppm_s=0.5, seed=2)
    model = LatencyModel(fixed_delay_s=0.05, jitter_std_s=0.002, seed=4)
    model.sample_interval = 0.01

    # Naive filter: fuses on receive time, ignores sensor clock
    naive_delays, comp_delays = [], []
    tsync = TimeSync()
    for pt in traj.points[::20]:          # sample every 0.2 s
        ts = clock.sensor_time(pt.t)
        tsync.add_pair(pt.t, ts)          # calibrate clock (offline / PTP)

    for pt in traj.points[::20]:
        ts = clock.sensor_time(pt.t)
        recv, ts_used, valid = model.deliver(ts)
        if not valid:
            continue
        # Naive: assume sensor timestamp is host time -> sees full latency
        naive_delays.append(recv - ts_used)
        # Compensated: convert sensor ts to host time first
        comp_delays.append(tsync.effective_delay(recv, ts_used))

    print(f"  Naive filter      sees mean delay: {np.mean(naive_delays)*1e3:6.1f} ms")
    print(f"  Compensated filter sees mean delay: {np.mean(comp_delays)*1e3:6.1f} ms")
    print(f"  (true fixed delay = {model.fixed_delay_s*1e3:.0f} ms)")


if __name__ == "__main__":
    main()
