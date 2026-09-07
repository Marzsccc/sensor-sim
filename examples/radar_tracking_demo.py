"""Constant-velocity radar tracking demo (v0.18.0).

The v0.17.0 ``radar_demo`` showed the radar's structural blind spot: a single
scan measures range / azimuth / Doppler range-rate, and a crossing pedestrian
produces almost no range-rate until it is nearly abeam -- the *cross-range*
(lateral) velocity is invisible in any one frame.  This demo adds the layer
that fixes it: a constant-velocity EKF tracker (``RadarTracker``) that fuses
a temporal sequence of detections into smooth 2-D tracks, recovering the
lateral velocity from the azimuth history.

Scene (stationary host at the origin -- the cleanest form of the blind spot,
the geometry behind rear / side cross-traffic alert):

1. a lead vehicle 150 m ahead approaching at 20 m/s -- a classic closing
   target: strong Doppler (+20 m/s), no lateral motion;
2. a pedestrian crossing left-to-right at x = 35 m, 2.2 m/s -- true radial
   range-rate stays below ~0.4 m/s the whole time, yet the tracker must
   estimate vy = 2.2 m/s from the drifting azimuth.

Per scan the demo prints the track table; the figure (saved to
``examples/radar_tracking_demo.png``) overlays measured vs tracked vs truth
range / range-rate / lateral velocity, so the Doppler blind spot and its
recovery by temporal fusion are visible in one picture.

Run:
    python3 examples/radar_tracking_demo.py
"""

import matplotlib

import numpy as np

from sensor_sim.lidar import Box, Sphere
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.trajectory import _euler_to_quat
from sensor_sim.tracking import RadarTracker, TrackConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.1  # s -- 10 Hz scan cadence
T_END = 5.0  # s

# World objects (analytic truth, as in radar_demo).
LEAD = Box(
    object_id=1,
    center=np.array([150.0, 0.0, 0.0]),
    half_extents=np.array([2.2, 1.0, 0.7]),
    velocity=np.array([-20.0, 0.0, 0.0]),  # approaching at 20 m/s
    reflectivity=1.0,
)  # ~10 m^2 car
PED = Sphere(
    object_id=2,
    center=np.array([35.0, -6.0, 0.0]),
    radius=0.35,
    velocity=np.array([0.0, 2.2, 0.0]),  # 2.2 m/s lateral walk
    reflectivity=0.1,
)  # ~1 m^2 person
WORLD = [LEAD, PED]

# Truth kinematics (analytic).
def _truth(t):
    lead_r = 150.0 - 20.0 * t - 2.2  # rear-surface range
    ped_y = -6.0 + 2.2 * t
    ped_r = np.hypot(35.0, ped_y)
    ped_vr = -2.2 * ped_y / ped_r  # host static: closing = -(v_ped . u)
    return lead_r, ped_y, ped_r, ped_vr


def main():
    rs = RadarSensor(RadarConfig(), mount_t_body=np.array([0.5, 0.0, 0.0]))
    trk = RadarTracker(
        TrackConfig(process_noise_accel_mps2=1.0),  # R matches RadarConfig()
        mount_t_body=np.array([0.5, 0.0, 0.0]),
    )
    q_id = _euler_to_quat(0.0, 0.0, 0.0)
    host = np.zeros(3)

    print(f"{'t':>4} {'ntr':>3} {'det':>3} | {'lead est r':>10} {'ped est vy':>10}")
    rows = []  # (t, object, kind, value) for plotting
    n = round(T_END / DT)
    for i in range(n + 1):
        t = i * DT
        fr = rs.scan(_pose(host, q_id), WORLD, t=t, rng=np.random.default_rng(20260907 + i))
        tf = trk.process(fr, host, q_id, np.zeros(3))
        lead_r, ped_y, ped_r, ped_vr = _truth(t)

        # Raw detections (measured range) as scatter points.
        for k in range(len(fr.ranges)):
            oid = fr.object_ids[k]
            rows.append((t, "lead" if oid == 1 else "ped", "r_meas", fr.ranges[k]))

        # Assign each track to the truth object it is closest to in range.
        lead_line = ped_line = "--"
        for tr in tf.tracks:
            est_r = float(np.hypot(tr.pos[0] - 0.5, tr.pos[1]))  # radar at +0.5 m
            u = (tr.pos - np.array([0.5, 0.0])) / max(est_r, 1e-9)
            est_vr = -float(tr.vel @ u)  # host static: closing = -(v . u)
            if abs(est_r - lead_r) <= abs(est_r - ped_r):
                rows.append((t, "lead", "r_est", est_r))
                rows.append((t, "lead", "vr_est", est_vr))
                rows.append((t, "lead", "vy_est", tr.vel[1]))
                lead_line = f"{est_r:9.1f}"
            else:
                rows.append((t, "ped", "r_est", est_r))
                rows.append((t, "ped", "vr_est", est_vr))
                rows.append((t, "ped", "vy_est", tr.vel[1]))
                ped_line = f"{tr.vel[1]:9.2f}"
        print(f"{t:5.1f} {tf.n_tracks:3d} {tf.n_detections:3d} | "
              f"{lead_line:>10} {ped_line:>10}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    ts = np.arange(0.0, T_END + DT, DT)
    lead_r_t = np.array([_truth(t)[0] for t in ts])
    ped_y_t = np.array([_truth(t)[1] for t in ts])
    ped_r_t = np.array([_truth(t)[2] for t in ts])
    ped_vr_t = np.array([_truth(t)[3] for t in ts])

    def _series(obj, kind):
        return np.array([r[3] for r in rows if r[1] == obj and r[2] == kind])

    def _time(obj, kind):
        return np.array([r[0] for r in rows if r[1] == obj and r[2] == kind])

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    lw = 1.0

    # 1) range ------------------------------------------------------------
    ax = axes[0]
    ax.plot(ts, lead_r_t, color="#e11d48", lw=lw, ls="--", label="lead truth")
    ax.plot(ts, ped_r_t, color="#0ea5e9", lw=lw, ls="--", label="ped truth")
    for obj, c in (("lead", "#e11d48"), ("ped", "#0ea5e9")):
        ax.plot(_time(obj, "r_est"), _series(obj, "r_est"), color=c, lw=lw, alpha=0.85)
        ax.plot(_time(obj, "r_meas"), _series(obj, "r_meas"), ".", color=c, ms=4, alpha=0.5)
    ax.set_ylabel("range (m)")
    ax.set_title("Radar detection tracking (v0.18.0): closing lead + crossing pedestrian\n"
                 "line = track estimate, dots = raw detections, dashed = truth")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 2) range-rate -------------------------------------------------------
    ax = axes[1]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.plot(ts, np.full_like(ts, 20.0), color="#e11d48", lw=lw, ls="--", label="lead truth +20")
    ax.plot(ts, ped_vr_t, color="#0ea5e9", lw=lw, ls="--", label="ped truth (|.|<0.4)")
    for obj, c in (("lead", "#e11d48"), ("ped", "#0ea5e9")):
        ax.plot(_time(obj, "vr_est"), _series(obj, "vr_est"), color=c, lw=lw, alpha=0.85)
    ax.set_ylabel("range-rate (m/s)")
    ax.set_ylim(-1.5, 23)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 3) lateral velocity -------------------------------------------------
    ax = axes[2]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.plot(ts, np.full_like(ts, 2.2), color="#0ea5e9", lw=lw, ls="--", label="ped truth vy = 2.2 m/s")
    ax.plot(_time("ped", "vy_est"), _series("ped", "vy_est"), color="#0ea5e9", lw=1.4,
            label="ped track vy estimate")
    ax.plot(_time("lead", "vy_est"), _series("lead", "vy_est"), color="#e11d48", lw=1.2,
            label="lead track vy estimate")
    ax.annotate("single-frame Doppler says ~0;\n"
                "temporal fusion recovers vy = 2.2 m/s", xy=(1.6, 1.1), fontsize=8,
                color="0.25", arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lateral velocity est (m/s)")
    ax.set_ylim(-1.0, 3.2)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out = "examples/radar_tracking_demo.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


def _pose(pos, q):
    class _P:
        pass
    p = _P()
    p.pos = np.asarray(pos, dtype=float)
    p.att = np.asarray(q, dtype=float)
    p.vel = np.zeros(3)
    return p


if __name__ == "__main__":
    main()
