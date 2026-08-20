"""LiDAR point-cloud simulation demo (v0.12.0).

Builds a realistic roadside scene (ground + three static obstacles + one
dynamic vehicle crossing the beam) and casts a full ``spinning`` LiDAR scan
from a host trajectory pose.  Prints per-object point counts, min/mean range
and intensity, then saves a top-down scatter plot to ``examples/lidar_demo.png``
so you can eyeball the scan geometry (ground ring, box faces, dynamic smear).

Run:
    python3 examples/lidar_demo.py
"""

import numpy as np

from sensor_sim.lidar import LidarConfig, LidarSensor, GroundPlane, Box, Sphere
from sensor_sim.trajectory import Trajectory, Waypoint, _euler_to_quat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # -- world -------------------------------------------------------------
    world = [
        GroundPlane(0.0),                                # ground (id 0)
        Box(1, np.array([8.0, -2.5, 0.0]), np.array([0.5, 2.5, 1.0]),
            reflectivity=0.6),                           # curb/barrier
        Box(2, np.array([12.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0]),
            reflectivity=0.9),                           # roadside box
        Sphere(3, np.array([6.0, 4.0, 0.5]), 0.5,        # pole base
               reflectivity=0.4),
        Box(4, np.array([14.0, -1.0, 0.0]), np.array([1.5, 0.8, 1.2]),
            velocity=np.array([-4.0, 2.0, 0.0]),         # dynamic cross traffic
            reflectivity=0.7),
    ]

    # -- host pose (identity attitude, z=1.5 m) ----------------------------
    z = np.zeros(3)
    q = _euler_to_quat(0, 0, 0)
    host = Trajectory(
        [Waypoint(0.0, np.array([0.0, 0.0, 1.5]), z, q, z),
         Waypoint(0.1, np.array([0.0, 0.0, 1.5]), z, q, z)],
        dt=0.1,
    )
    point = host.at(0.0)

    # -- LiDAR -------------------------------------------------------------
    cfg = LidarConfig(mode="spinning", beams=16, columns=1800,
                      range_min=0.5, range_max=60.0,
                      range_noise_sigma_m=0.02, dropout_prob=0.05)
    lidar = LidarSensor(cfg, mount_t_body=np.array([1.0, 0.0, 0.0]))
    frame = lidar.scan(point, world, t=0.0, rng=np.random.default_rng(42))

    print("=" * 68)
    print("LiDAR point-cloud demo (v0.12.0)")
    print("=" * 68)
    print(f"total valid points : {len(frame)}")
    print(f"host pos           : {point.pos}")
    print(f"mount offset (body): {lidar.mount_t_body}")

    print("\nper-object hit counts:")
    print(f"  {'object':<12}{'points':>8}{'min_r':>9}{'mean_int':>10}")
    for oid in sorted(set(frame.object_ids.tolist())):
        sel = frame.object_ids == oid
        name = {0: "ground", 1: "barrier", 2: "box",
                3: "pole", 4: "dynamic"}.get(oid, f"obj{oid}")
        print(f"  {name:<12}{int(sel.sum()):>8}"
              f"{np.min(frame.ranges[sel]):>9.2f}"
              f"{np.mean(frame.intensities[sel]):>10.2f}")

    # -- top-down plot ------------------------------------------------------
    p = frame.points
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(p[:, 1], p[:, 0], c=frame.object_ids, cmap="tab10",
                    s=1.5, alpha=0.7)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("LiDAR spinning scan (top-down)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, lw=0.4)
    cb = fig.colorbar(sc, ax=ax, label="object id")
    fig.tight_layout()
    out = "examples/lidar_demo.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
