"""Camera feature & optical-flow simulation demo (v0.13.0).

Builds a roadside scene of analytic features (a lane of static landmarks plus
one dynamic pedestrian walking across), then drives the ego camera forward
along +x.  For each frame pair it projects all visible features and reports the
per-feature optical flow (px displacement and px/s).  Saves the previous-frame
feature positions with current-frame flow vectors overlaid onto the image plane
to ``examples/camera_demo.png`` so you can eyeball the motion field.

Run:
    python3 examples/camera_demo.py
"""

import numpy as np

from sensor_sim.camera import CameraConfig, CameraSensor, FeaturePoint
from sensor_sim.trajectory import _euler_to_quat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # -- world: static landmarks + one dynamic pedestrian -------------------
    features = [
        FeaturePoint(fid, np.array([x, y, 0.0]), object_id=0)
        for fid, (x, y) in enumerate([
            (10.0, -2.0), (10.0, 2.0),     # near row
            (15.0, -2.5), (15.0, 2.5),     # mid row
            (22.0, -3.0), (22.0, 3.0),     # far row
            (30.0, -3.5), (30.0, 3.5),
        ])
    ]
    # a dynamic pedestrian walking across in front, +y direction
    features.append(FeaturePoint(
        100, np.array([8.0, -3.0, 1.0]), object_id=1,
        velocity=np.array([0.0, 1.2, 0.0])))

    cam = CameraSensor(CameraConfig(fx=800.0, width=1280, height=720,
                                    pixel_noise_sigma=0.4, dropout_prob=0.03))
    q_id = _euler_to_quat(0, 0, 0)

    # -- ego drives forward along +x at 10 m/s ------------------------------
    dt = 0.05
    x0 = 0.0
    poses = []
    n = 12
    for i in range(n):
        t = i * dt
        poses.append((t, np.array([x0 + 10.0 * t, 0.0, 1.5]), q_id))

    print(f"{'feat':>5} {'u0':>8} {'v0':>8} {'u1':>8} {'v1':>8} "
          f"{'flow_u':>8} {'flow_v':>8} {'depth':>7}")
    all_prev = []
    all_flow = []
    for i in range(n - 1):
        t0, p0, q0 = poses[i]
        t1, p1, q1 = poses[i + 1]
        fl = cam.track(
            _P(t0, p0, q0), _P(t1, p1, q1), features, t0=t0, t1=t1,
            rng=np.random.default_rng(1234))
        for k in range(len(fl)):
            print(f"{fl.feature_ids[k]:>5} "
                  f"{fl.prev_pts[k,0]:>8.2f} {fl.prev_pts[k,1]:>8.2f} "
                  f"{fl.cur_pts[k,0]:>8.2f} {fl.cur_pts[k,1]:>8.2f} "
                  f"{fl.flow[k,0]:>8.2f} {fl.flow[k,1]:>8.2f} "
                  f"{fl.depths[k]:>7.2f}")
            all_prev.append(fl.prev_pts[k])
            all_flow.append(fl.flow[k])

    all_prev = np.array(all_prev)
    all_flow = np.array(all_flow)

    # -- plot image-plane flow field ----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(all_prev[:, 0], all_prev[:, 1], s=14, c="#e11d48", zorder=3,
               label="prev-frame features")
    ax.quiver(all_prev[:, 0], all_prev[:, 1],
              all_flow[:, 0], all_flow[:, 1],
              color="#0ea5e9", width=0.004, scale=1.0, scale_units="xy",
              angles="xy", label="optical flow (px)")
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.set_title("Camera feature tracking & optical flow (v0.13.0)")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/camera_demo.png", dpi=130)
    print(f"\nSaved flow-field plot -> examples/camera_demo.png "
          f"({len(all_prev)} feature observations)")


class _P:
    """Minimal pose stand-in (keeps this demo independent of the trajectory)."""
    def __init__(self, t, pos, att):
        self.t = t
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)


if __name__ == "__main__":
    main()
