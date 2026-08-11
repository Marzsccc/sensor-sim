"""
Tests for sensor-sim latency / time-sync module.

Run: python3 tests/test_latency.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sensor_sim.latency import (
    LatencyModel,
    SensorClock,
    TimeSync,
    LatencyScenario,
)
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.utils import euler_to_quat


def _build_traj(duration=10.0, speed=10.0):
    p0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([speed, 0.0, 0.0])
    q0 = euler_to_quat(0, 0, 0)
    waypoints = [
        Waypoint(t=0, pos=p0, vel=v0, att=q0, omega=np.zeros(3)),
        Waypoint(t=duration, pos=p0 + v0 * duration, vel=v0, att=q0, omega=np.zeros(3)),
    ]
    return Trajectory(waypoints, dt=0.01)


def test_latency_fixed():
    """Fixed delay shifts receive time by exactly the delay."""
    m = LatencyModel(fixed_delay_s=0.05, seed=1)
    recv, ts, valid = m.deliver(1.0)
    assert valid
    assert abs(recv - (1.0 + 0.05)) < 1e-6
    assert ts == 1.0  # true sample time preserved


def test_latency_loss():
    """High loss probability drops most samples."""
    m = LatencyModel(loss_prob=0.99, seed=7)
    valid_count = sum(1 for _ in range(200) if m.deliver(float(_))[2])
    assert valid_count <= 5, f"Expected almost all dropped, got {valid_count} valid"


def test_latency_stale_ts():
    """Stale timestamps are earlier than the true sample time."""
    m = LatencyModel(stale_ts_prob=1.0, seed=3)
    m.sample_interval = 1.0 / 30.0
    recv, ts, valid = m.deliver(2.0)
    assert valid
    assert ts < 2.0


def test_sensor_clock_drift():
    """A 20 ppm clock drifts ~2 ms over 100 s."""
    c = SensorClock(rate_ppm=20.0, wander_ppm_s=0.0, seed=1)
    t_sensor = c.sensor_time(100.0)
    # 20 ppm * 100 s = 2 ms
    assert abs(t_sensor - 100.0) < 0.01
    assert t_sensor > 100.0


def test_time_sync_fit():
    """Linear fit recovers offset and rate for a drifting clock."""
    host = np.linspace(0.0, 100.0, 500)
    rate = 30e-6   # 30 ppm
    offset = 0.25  # 250 ms
    sensor = offset + (1.0 + rate) * host
    ts = TimeSync()
    for h, s in zip(host, sensor):
        ts.add_pair(h, s)
    est_offset, est_rate, r2 = ts.fit()
    assert abs(est_offset - offset) < 1e-6
    assert abs(est_rate - rate) < 1e-9
    assert r2 > 0.999


def test_effective_delay():
    """Effective delay = receive_time - host-converted sample time."""
    m = LatencyModel(fixed_delay_s=0.05, seed=5)
    recv, ts, valid = m.deliver(1.0)
    assert valid
    tsync = TimeSync()
    # Perfect clock: host == sensor
    tsync.add_pair(0.0, 0.0)
    tsync.add_pair(100.0, 100.0)
    delay = tsync.effective_delay(recv, ts)
    assert abs(delay - 0.05) < 1e-6


def test_scenario_summary():
    """Scenario simulates multi-sensor streams and summarizes delays."""
    traj = _build_traj(5.0)
    sc = LatencyScenario(sensors={
        "camera": LatencyModel(fixed_delay_s=0.05, jitter_std_s=0.005, seed=11),
        "gnss":   LatencyModel(fixed_delay_s=0.20, jitter_std_s=0.02, loss_prob=0.05, seed=12),
        "wheel":  LatencyModel(fixed_delay_s=0.005, seed=13),
    }, dt=0.01)
    streams = sc.simulate(traj)
    summ = sc.summary(streams)
    assert abs(summ["camera"]["mean_delay_s"] - 0.05) < 0.01
    assert abs(summ["gnss"]["mean_delay_s"] - 0.20) < 0.02
    assert summ["wheel"]["mean_delay_s"] < 0.01
    assert summ["gnss"]["loss_rate"] > 0.0


def test_clock_compensation_recovers_delay():
    """With a fitted clock, effective delay equals the true fixed delay."""
    traj = _build_traj(5.0)
    clock = SensorClock(rate_ppm=50.0, wander_ppm_s=0.0, seed=2)
    model = LatencyModel(fixed_delay_s=0.05, seed=4)
    model.sample_interval = 0.01
    tsync = TimeSync()
    # Build clock pairs over the trajectory
    for pt in traj.points[::50]:
        tsync.add_pair(pt.t, clock.sensor_time(pt.t))
    # Simulate a measurement and check effective delay after compensation
    pt = traj.points[200]
    ts = clock.sensor_time(pt.t)
    recv, ts_used, valid = model.deliver(ts)
    assert valid
    delay = tsync.effective_delay(recv, ts_used)
    assert abs(delay - 0.05) < 0.02, f"Effective delay {delay:.4f} != 0.05"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
