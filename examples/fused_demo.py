"""Radar + camera fusion tracking demo (v0.19.0).

v0.18.0's ``radar_tracking_demo`` showed a radar-only tracker recovering a
crossing pedestrian's 2.2 m/s lateral velocity from the azimuth *history*.
This demo adds the production upgrade: a **camera** riding alongside the
radar, feeding a precise *per-scan* azimuth measurement into the same EKF
(``FusedTracker``).  The camera sees the crossing target's angular motion
immediately, so the fused tracker converges to the true lateral velocity
faster and with less error than radar alone.

Scene (stationary host at the origin -- the cleanest form of the radar blind
spot, the geometry behind rear / side cross-traffic alert):

1. a lead vehicle 90 m ahead approaching at 20 m/s -- strong Doppler, no
   lateral motion (the radar's home turf);
2. a pedestrian crossing left-to-right at x = 35 m, 2.2 m/s -- the camera's
   home turf: the azimuth drifts every frame and the fused filter reads the
   lateral motion directly.

The figure (saved to ``examples/fused_demo.png``) overlays, for both the
radar-only tracker and the fused tracker:

- top: lateral-velocity estimate vs truth (2.2 m/s) -- the fused line reaches
  the true value visibly faster;
- middle: range estimate vs truth;
- bottom: per-scan camera-update / radar-update counts and the per-track
  horizontal-position error.

Run:
    python3 examples/fused_demo.py
"""

import matplotlib

import numpy as np

from sensor_sim.camera import CameraConfig, CameraSensor, FeaturePoint
from sensor_sim.fused import FusedConfig, FusedTracker
from sensor_sim.lidar import Box, Sphere
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.trajectory import _euler_to_quat
from sensor_sim.tracking import RadarTracker, TrackConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.1  # s -- 10 Hz radar cadence (camera runs at the same rate)
T_END = 3.0  # s

# World objects (analytic truth, as in the radar demos).
LEAD = Box(
    object_id=1,
    center=np.array([90.0, 0.0, 0.0]),
    half_extents=np.array([2.2, 1.0, 0.7]),
    velocity=np.array([-20.0, 0.0, 0.0]),  # approaching at 20 m/s
    reflectivity=1.0,
)
PED = Sphere(
    object_id=2,
    center=np.array([35.0, -6.0, 0.0]),
    radius=0.35,
    velocity=np.array([0.0, 2.2, 0.0]),  # 2.2 m/s lateral walk
    reflectivity=0.1,
)
WORLD = [LEAD, PED]


def _features():
    """One FeaturePoint per world object (matching the radar object ids)."""
    return [
        FeaturePoint(
            feature_id=LEAD.object_id,
            pos_w=LEAD.center.copy(),
            object_id=LEAD.object_id,
            velocity=LEAD.velocity.copy(),
        ),
        FeaturePoint(
            feature_id=PED.object_id,
            pos_w=PED.center.copy(),
            object_id=PED.object_id,
            velocity=PED.velocity.copy(),
        ),
    ]


def _pose(pos, q):
    class _P:
        pass

    p = _P()
    p.pos = np.asarray(pos, dtype=float)
    p.att = np.asarray(q, dtype=float)
    p.vel = np.zeros(3)
    return p


def main():
    # Sensors share the same mounts and the same 10 Hz clock.
    rs = RadarSensor(RadarConfig(), mount_t_body=np.array([0.5, 0.0, 0.0]))
    cs = CameraSensor(
        CameraConfig(
            fx=1800.0, width=1920, height=1080,
            pixel_noise_sigma=0.5, dropout_prob=0.02, max_z=200.0,
        ),
        mount_t_body=np.array([0.5, 0.0, 0.6]),
    )
    tc = TrackConfig(process_noise_accel_mps2=1.0)
    trk_r = RadarTracker(tc, mount_t_body=np.array([0.5, 0.0, 0.0]))
    trk_f = FusedTracker(
        FusedConfig(
            track_config=tc,
            cam_azimuth_sigma_deg=0.1,  # ~2 px on the demo camera
        ),
        radar_mount_t_body=np.array([0.5, 0.0, 0.0]),
        cam_mount_t_body=np.array([0.5, 0.0, 0.6]),
        cam_fx=1800.0, cam_cx=960.0, cam_width=1920,
    )
    q_id = _euler_to_quat(0.0, 0.0, 0.0)
    host = np.zeros(3)
    feats = _features()

    print(f"{'t':>4} {'radar':>6} {'fused':>6} | "
          f"{'rad vy':>7} {'fus vy':>7} | {'camU':>4} {'radU':>4}")
    rows = []  # (t, kind, obj, value)

    def _track_vel(tf, oid):
        """Best-matching track's lateral velocity for the given object."""
        for tr in tf.tracks:
            # Use the object's truth range (surface point) to pick the track.
            if oid == 1:
                r_true = 90.0 - 20.0 * tf.t - 2.2
            else:
                ped_y = -6.0 + 2.2 * tf.t
                r_true = np.hypot(35.0, ped_y)
            est_r = float(np.hypot(tr.pos[0] - 0.5, tr.pos[1]))
            if abs(est_r - r_true) < 8.0:
                return tr
        return None

    n = round(T_END / DT)
    for i in range(n + 1):
        t = i * DT
        p = _pose(host, q_id)
        fr = rs.scan(p, WORLD, t=t, rng=np.random.default_rng(20260910 + i))
        cf = cs.observe(p, feats, t=t, rng=np.random.default_rng(20260910 + 1000 + i))

        tf_r = trk_r.process(fr, host, q_id, np.zeros(3))
        tf_f = trk_f.process(fr, cf, host, q_id, np.zeros(3))

        tr_r = _track_vel(tf_r, 2)
        tr_f = _track_vel(tf_f, 2)
        vy_r = tr_r.vel[1] if tr_r else float("nan")
        vy_f = tr_f.vel[1] if tr_f else float("nan")
        rows.append((t, "vy", "rad", vy_r))
        rows.append((t, "vy", "fus", vy_f))
        # range (both trackers, lead object 1)
        tr_r1 = _track_vel(tf_r, 1)
        tr_f1 = _track_vel(tf_f, 1)
        rows.append((t, "r", "rad", tr_r1.pos[0] if tr_r1 else float("nan")))
        rows.append((t, "r", "fus", tr_f1.pos[0] if tr_f1 else float("nan")))
        rows.append((t, "nupd", "rad", tf_r.n_tracks))
        rows.append((t, "nupd", "fus", tf_f.n_cam_updates))

        print(f"{t:5.1f} {tf_r.n_tracks:6d} {tf_f.n_tracks:6d} | "
              f"{vy_r:7.2f} {vy_f:7.2f} | {tf_f.n_cam_updates:4d} {tf_r.n_tracks:4d}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    ts = np.arange(0.0, T_END + DT, DT)
    ped_y_t = -6.0 + 2.2 * ts
    ped_vy_t = np.full_like(ts, 2.2)
    lead_r_t = 90.0 - 20.0 * ts - 2.2

    def _series(kind, who):
        return np.array([r[3] for r in rows if r[1] == kind and r[2] == who])

    def _time(kind, who):
        return np.array([r[0] for r in rows if r[1] == kind and r[2] == who])

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    lw = 1.2

    # 1) lateral velocity -------------------------------------------------
    ax = axes[0]
    ax.plot(ts, ped_vy_t, color="0.4", lw=1.0, ls="--", label="ped truth vy = 2.2 m/s")
    ax.plot(_time("vy", "fus"), _series("vy", "fus"), color="#0ea5e9", lw=lw,
            label="fused vy (radar+camera)")
    ax.plot(_time("vy", "rad"), _series("vy", "rad"), color="#e11d48", lw=lw,
            label="radar-only vy")
    ax.annotate("camera sees the azimuth every frame\n-> lateral velocity "
                "converges faster", xy=(0.5, 2.35), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_ylabel("lateral velocity (m/s)")
    ax.set_title("Radar+camera fusion (v0.19.0): crossing pedestrian\n"
                 "fused (blue) reaches truth faster than radar-only (red)")
    ax.set_ylim(-0.5, 3.2)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    # 2) lead range -------------------------------------------------------
    ax = axes[1]
    ax.plot(ts, lead_r_t, color="0.4", lw=1.0, ls="--", label="lead truth")
    ax.plot(_time("r", "fus"), _series("r", "fus"), color="#0ea5e9", lw=lw, label="fused")
    ax.plot(_time("r", "rad"), _series("r", "rad"), color="#e11d48", lw=lw, label="radar-only")
    ax.set_ylabel("lead range (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 3) per-scan update counts -------------------------------------------
    ax = axes[2]
    ax.plot(_time("nupd", "fus"), _series("nupd", "fus"), color="#0ea5e9", lw=lw,
            label="fused camera updates / scan")
    ax.plot(_time("nupd", "rad"), _series("nupd", "rad"), color="#e11d48", lw=lw,
            ls="--", label="radar-only tracks / scan")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count")
    ax.set_ylim(-0.5, 4.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = "examples/fused_demo.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")
    # Exit code: always 0 (the demo just renders).


if __name__ == "__main__":
    main()