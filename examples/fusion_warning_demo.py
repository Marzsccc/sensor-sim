#!/usr/bin/env python3
"""v0.20.0 demo: fused tracks -> ADAS markers -> HUD warning arbitration.

Builds the v0.19 crossing-pedestrian scene, runs the fused radar+camera
tracker, then feeds the confirmed tracks through ``FusionThreatPipeline``
to derive kinematic closing rates and pick the HUD headline warning.

The point of the demo is the *coupling*: the threat level of the crossing
pedestrian cannot climb until the camera-aided fusion has resolved its
lateral velocity (the v0.18/0.19 lesson), while the approaching lead
vehicle warns immediately from its strong Doppler closing rate.
"""

import numpy as np

from sensor_sim.camera import CameraConfig, CameraSensor
from sensor_sim.fused import FusedConfig, FusedTracker
from sensor_sim.hazard import WarningLevel
from sensor_sim.lidar import Box, Sphere
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.track_warn import FusionThreatPipeline
from sensor_sim.tracking import TrackConfig
from sensor_sim.utils import quat_to_rotmat

DT = 0.1          # radar period (10 Hz)
T_END = 6.0       # 6 s of fusion
LEAD_SPEED = 15.0 # approaching lead vehicle (m/s)
PED_SPEED = 2.2   # crossing pedestrian lateral speed (m/s)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _pose(pos: np.ndarray, att: np.ndarray):
    return type("Pose", (), {"pos": pos, "att": att, "vel": np.zeros(3)})()


def _features():
    """One FeaturePoint per world object (matching the radar object ids)."""
    from sensor_sim.camera import FeaturePoint
    return [
        FeaturePoint(feature_id=1, pos_w=np.array([100.0, 0.6, 0.8]),
                     velocity=np.array([-LEAD_SPEED, 0.0, 0.0])),
        FeaturePoint(feature_id=2, pos_w=np.array([35.0, -6.0, 1.2]),
                     velocity=np.array([0.0, PED_SPEED, 0.0])),
    ]


def main() -> None:
    WORLD = [
        Box(center=np.array([100.0, 0.0, 0.0]), half_extents=np.array([2.0, 1.0, 1.0]),
            velocity=np.array([-LEAD_SPEED, 0.0, 0.0]), object_id=1),
        Sphere(center=np.array([35.0, -6.0, 0.0]), radius=0.4,
               velocity=np.array([0.0, PED_SPEED, 0.0]), object_id=2),
    ]

    rs = RadarSensor(
        RadarConfig(range_noise_sigma_m=0.1, angle_noise_sigma_deg=0.2,
                    range_rate_noise_sigma_mps=0.1),
        mount_t_body=np.array([0.5, 0.0, 0.0]),
    )
    cs = CameraSensor(
        CameraConfig(fx=1800.0, width=1920, height=1080,
                     pixel_noise_sigma=0.5, dropout_prob=0.02, max_z=250.0),
        mount_t_body=np.array([0.5, 0.0, 0.6]),
    )
    tc = TrackConfig(process_noise_accel_mps2=1.0, confirm_min=3)
    trk = FusedTracker(
        FusedConfig(track_config=tc, cam_azimuth_sigma_deg=0.1),
        radar_mount_t_body=np.array([0.5, 0.0, 0.0]),
        cam_mount_t_body=np.array([0.5, 0.0, 0.6]),
        cam_fx=1800.0, cam_cx=960.0, cam_width=1920,
    )
    pipe = FusionThreatPipeline()

    host = np.zeros(3)
    q_id = _euler_to_quat(0.0, 0.0, 0.0)
    feats = _features()
    rng = np.random.default_rng(20260912)

    print(f"{'t(s)':>5} {'nTrk':>4} {'lead cr':>8} {'ped cr':>8} {'level':>10}")
    rows = []
    for i in range(round(T_END / DT) + 1):
        t = i * DT
        point = _pose(host, q_id)
        fr = rs.scan(point, WORLD, t=t, rng=rng)
        cf = cs.observe(point, feats, t=t, rng=np.random.default_rng(20260912 + i))
        tf = trk.process(fr, cf, host, q_id, np.zeros(3))
        res = pipe.decide(list(tf.tracks), host, q_id)

        lead_cr = ped_cr = float("nan")
        for m in res["markers"]:
            # Identify by distance-to-truth (same trick as fused_demo).
            # LEAD truth: 100 - 15*t
            if m.label.startswith("track#"):
                p = m.ref_world[:2]
                if abs(np.hypot(p[0], p[1]) - (100.0 - LEAD_SPEED * t)) < 12.0:
                    lead_cr = -m.closing_rate_mps  # back to kinematic sign
                elif abs(np.hypot(p[0] - 35.0, p[1] + 6.0 - PED_SPEED * t)) < 3.0:
                    ped_cr = -m.closing_rate_mps
        rows.append((t, tf.n_tracks, lead_cr, ped_cr, res["level"].label))
        if i % 5 == 0 or i == round(T_END / DT):
            print(f"{t:5.1f} {tf.n_tracks:4d} {lead_cr:8.2f} {ped_cr:8.2f} {res['level'].label:>10}")

    levels = [r[4] for r in rows]
    labels = [WarningLevel._value2member_map_[0].label, "CAUTION", "WARN",
              "CRITICAL", "EMERGENCY"]
    print("\nFinal headline warning:", rows[-1][4])
    print("Max warning reached:", max(labels.index(l) for l in levels if l in labels))
    print("(lead vehicle closing rate = +; pedestrian crossing = ~0 until "
          "fusion resolves lateral velocity)")


if __name__ == "__main__":
    main()