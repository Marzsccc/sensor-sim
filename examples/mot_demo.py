"""Multi-object tracking (MOT) metrics demo (v0.21.0).

Runs the v0.18 radar tracker on a small multi-target scene and grades the
resulting track sets with the new ``sensor_sim.mot`` module -- the "how good
are the tracks?" question the v0.17-v0.20 stack never answered.

Two competing front-ends are scored against the *same* radar ground truth
(``RadarFrame`` truth-bearing channels -> ``truth_world_points``):

1. **naive** -- every raw detection is treated as a track (a single-scan
   "tracker"): it inherits all measurement noise, drops the target on a missed
   detection and cannot coast.  It is the honest floor.
2. **ekf** -- the v0.18 ``RadarTracker``: constant-velocity EKF per target,
   with confirmation / coasting, so it smooths noise and rides through
   dropouts.

Per scan the demo records OSPA and GOSPA for both, plus the CLEAR-MOT
counters (FP / FN / ID-switches) for the tracker, then prints a summary table
and saves ``examples/mot_demo.png`` (error curves + score comparison).

Run:
    python3 examples/mot_demo.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sensor_sim.lidar import Box, Sphere
from sensor_sim.mot import MotAccumulator, truth_world_points
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.tracking import RadarTracker, TrackConfig
from sensor_sim.trajectory import _euler_to_quat
from sensor_sim.utils import quat_to_rotmat

DT = 0.1  # s -- 10 Hz radar
T_END = 8.0  # s
MOUNT = np.array([0.5, 0.0, 0.0])  # radar on the bonnet

# Scene: one manifestly closing lead, one slower car slightly off-axis, and a
# crossing pedestrian (the radar cross-range blind spot from v0.18).
LEAD = Box(
    object_id=1,
    center=np.array([70.0, 0.0, 0.0]),
    half_extents=np.array([2.2, 1.0, 0.7]),
    velocity=np.array([-4.0, 0.0, 0.0]),
    reflectivity=1.0,
)
CAR2 = Box(
    object_id=2,
    center=np.array([45.0, 6.0, 0.0]),
    half_extents=np.array([2.0, 0.9, 0.7]),
    velocity=np.array([-2.0, 0.0, 0.0]),
    reflectivity=0.8,
)
PED = Sphere(
    object_id=3,
    center=np.array([30.0, -6.0, 0.0]),
    radius=0.35,
    velocity=np.array([0.0, 1.5, 0.0]),
    reflectivity=0.1,
)
# A stopped vehicle far down the road: only ~60% detection probability per
# scan at this range, so the naive front-end "loses" it on every miss while
# the EKF tracker coasts through the gaps.
PARKED = Box(
    object_id=4,
    center=np.array([100.0, 0.0, 0.0]),
    half_extents=np.array([2.2, 1.0, 0.7]),
    velocity=np.array([0.0, 0.0, 0.0]),
    reflectivity=0.25,
)
WORLD = [LEAD, CAR2, PED, PARKED]


def _pose(pos, q):
    class _P:
        pass

    p = _P()
    p.pos = np.asarray(pos, dtype=float)
    p.att = np.asarray(q, dtype=float)
    p.vel = np.zeros(3)
    return p


def main():
    rs = RadarSensor(RadarConfig(), mount_t_body=MOUNT)
    trk = RadarTracker(TrackConfig(process_noise_accel_mps2=1.0), mount_t_body=MOUNT)
    q = _euler_to_quat(0.0, 0.0, 0.0)
    host = np.zeros(3)
    R_wb = quat_to_rotmat(q).T
    origin = host + R_wb @ MOUNT

    acc_ekf = MotAccumulator(gate_m=3.0, cutoff_m=8.0)
    acc_naive = MotAccumulator(gate_m=3.0, cutoff_m=8.0)

    ts, ospa_ekf, ospa_naive, gospa_ekf = [], [], [], []

    print(f"{'t':>4} | {'det':>3} {'trk':>3} | {'OSPA naive':>10} {'OSPA ekf':>10} "
          f"| {'FP':>3} {'FN':>3} {'IDS':>3}")
    for i in range(round(T_END / DT) + 1):
        t = i * DT
        fr = rs.scan(_pose(host, q), WORLD, t=t, rng=np.random.default_rng(777 + i))
        tf = trk.process(fr, host, q, np.zeros(3))

        # --- ground truth (radar near-surface points, world frame) -------
        gpt = truth_world_points(fr, host, q, mount_t_body=MOUNT)
        gids = list(fr.truth_ids)

        # --- naive front-end: raw detections as singleton tracks ---------
        if fr.points.shape[0]:
            ept_naive = (origin + fr.points @ R_wb.T)[:, :2]
        else:
            ept_naive = np.zeros((0, 2))
        r_naive = acc_naive.add(
            ept_naive, gpt, est_ids=list(fr.object_ids), gt_ids=gids, t=t
        )

        # --- EKF tracker -------------------------------------------------
        trks = list(tf.tracks)
        ept_ekf = np.array([tr.pos for tr in trks]) if trks else np.zeros((0, 2))
        eids = [tr.track_id for tr in trks]
        r_ekf = acc_ekf.add(ept_ekf, gpt, est_ids=eids, gt_ids=gids, t=t)

        ts.append(t)
        ospa_naive.append(r_naive.ospa)
        ospa_ekf.append(r_ekf.ospa)
        gospa_ekf.append(r_ekf.gospa)
        print(f"{t:4.1f} | {fr.points.shape[0]:3d} {tf.n_tracks:3d} | "
              f"{r_naive.ospa:10.3f} {r_ekf.ospa:10.3f} | "
              f"{r_ekf.false_positives:3d} {r_ekf.false_negatives:3d} {r_ekf.id_switches:3d}")

    s_naive = acc_naive.summary()
    s_ekf = acc_ekf.summary()

    print("\n=== CLEAR-MOT summary ==============================================")
    print(f"{'metric':<14}{'naive':>10}{'ekf':>10}")
    for k in ("mota", "motp", "false_positives", "false_negatives",
              "id_switches", "mean_ospa", "mean_gospa"):
        print(f"{k:<14}{s_naive[k]:>10.3f}{s_ekf[k]:>10.3f}")
    print(f"{'mostly_tracked':<14}{s_naive.get('mostly_tracked', 0):>10}"
          f"{s_ekf.get('mostly_tracked', 0):>10}")
    print(f"{'mostly_lost':<14}{s_naive.get('mostly_lost', 0):>10}"
          f"{s_ekf.get('mostly_lost', 0):>10}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    ts = np.array(ts)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.0), sharex=True)

    ax = axes[0]
    ax.plot(ts, ospa_naive, color="#94a3b8", lw=1.3, label=f"naive (mean {s_naive['mean_ospa']:.2f})")
    ax.plot(ts, ospa_ekf, color="#e11d48", lw=1.6, label=f"EKF tracker (mean {s_ekf['mean_ospa']:.2f})")
    ax.set_ylabel("OSPA (m, c = 8)")
    ax.set_title("MOT metrics (v0.21.0): grading the track set, not the pose\n"
                 "naive = raw detections as tracks; EKF = v0.18 tracker")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    loc = np.array([g["localisation"] for g in gospa_ekf])
    mis = np.array([g["missed"] for g in gospa_ekf])
    fal = np.array([g["false"] for g in gospa_ekf])
    ax.stackplot(ts, loc, mis, fal,
                 labels=["localisation", "missed", "false"],
                 colors=["#0ea5e9", "#f59e0b", "#a855f7"], alpha=0.85)
    ax.set_ylabel("GOSPA components (m)")
    ax.set_title("GOSPA decomposition of the EKF tracker (c = 8, alpha = 2)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    keys = ["mota", "motp", "mean_ospa", "mean_gospa"]
    label = ["MOTA", "MOTP (m)", "mean OSPA (m)", "mean GOSPA (m)"]
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, [s_naive[k] for k in keys], w, label="naive", color="#94a3b8")
    ax.bar(x + w / 2, [s_ekf[k] for k in keys], w, label="EKF tracker", color="#e11d48")
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_title("Headline scores (MOTA: higher is better; others: lower is better)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = "examples/mot_demo.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
