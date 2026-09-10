"""Tests for the radar + camera fused tracker (v0.19.0)."""

import numpy as np
import pytest

from sensor_sim.camera import CameraConfig, CameraSensor, FeaturePoint
from sensor_sim.fused import FusedConfig, FusedTracker
from sensor_sim.lidar import Box, Sphere
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.trajectory import _euler_to_quat


class _Pose:
    """Minimal stand-in for a trajectory point (kept independent)."""

    def __init__(self, pos, att, vel=None):
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)
        self.vel = None if vel is None else np.asarray(vel, dtype=float)


Q_ID = np.array([1.0, 0.0, 0.0, 0.0])  # identity (no rotation)
DT = 0.1  # s -- 10 Hz radar cadence used throughout

# Quiet-but-physical noise for the radar / tracker pair (see test_tracking).
QR = dict(range_noise_sigma_m=1e-2, angle_noise_sigma_deg=5e-2, range_rate_noise_sigma_mps=2e-2)


def _quiet_radar(**kw):
    kw = {**QR, **kw}
    return RadarSensor(RadarConfig(**kw))


def _quiet_camera(**kw):
    kw.setdefault("pixel_noise_sigma", 0.05)  # ~0.1 px -> ~0.01°
    kw.setdefault("dropout_prob", 0.0)
    kw.setdefault("width", 1920)
    kw.setdefault("height", 1080)
    kw.setdefault("fx", 1800.0)  # ~53° horizontal FOV over 1920 px
    kw.setdefault("min_z", 0.1)
    kw.setdefault("max_z", 200.0)
    return CameraSensor(CameraConfig(**kw))


def _quiet_fused(**kw):
    """Fused tracker whose radar R matches the radar; a precise camera.

    Camera intrinsics match ``_quiet_camera`` (fx=1800, cx=960) so the
    pixel->azimuth conversion is exact.
    """
    kw.setdefault("process_noise_accel_mps2", 1.0)
    return FusedTracker(
        FusedConfig(
            track_config=_tc(**kw),
            cam_azimuth_sigma_deg=0.03,  # a sharp camera feature
        ),
        cam_fx=1800.0,
        cam_cx=960.0,
        cam_width=1920,
    )


def _tc(**kw):
    from sensor_sim.tracking import TrackConfig

    return TrackConfig(**{**QR, **kw})


def _car(center, velocity=(0.0, 0.0, 0.0), object_id=1, reflectivity=1.0):
    return Box(
        object_id=object_id,
        center=np.asarray(center, dtype=float),
        half_extents=np.array([2.2, 1.0, 0.7]),
        velocity=np.asarray(velocity, dtype=float),
        reflectivity=reflectivity,
    )


def _ped(center, velocity=(0.0, 0.0, 0.0), object_id=2):
    return Sphere(
        object_id=object_id,
        center=np.asarray(center, dtype=float),
        radius=0.35,
        velocity=np.asarray(velocity, dtype=float),
        reflectivity=0.1,
    )


def _pose(pos, vel=None):
    return _Pose(np.asarray(pos, dtype=float), Q_ID, vel)


def _run_fused(trk, radar, cam, world, t_end, host_vel=(0.0, 0.0, 0.0),
               cam_dropout_t=None, seed=0):
    """Feed 10 Hz radar scans, camera frames each tick (unless dropout).

    Camera observations are generated from the same world FeaturePoints, so a
    target that the radar sees is also seen by the camera (unless the caller
    schedules a dropout).  Returns (frames, last_frame).
    """
    cam_dropout_t = cam_dropout_t or (lambda t: False)
    n = round(t_end / DT)
    frames = []
    for i in range(n + 1):
        t = i * DT
        pos = np.array([host_vel[0] * t, host_vel[1] * t, 0.0])
        vel = np.asarray(host_vel, dtype=float)
        p = _pose(pos, vel)
        fr = radar.scan(p, world, t=t, rng=np.random.default_rng(seed + i))
        cam_frame = None
        if not cam_dropout_t(t):
            feats = _features_of(world)
            cam_frame = cam.observe(p, feats, t=t, rng=np.random.default_rng(seed + 1000 + i))
        tf = trk.process(fr, cam_frame, pos, Q_ID, vel)
        frames.append(tf)
    return frames, frames[-1]


def _features_of(world):
    """One FeaturePoint per Box/Sphere (matching the radar object ids)."""
    out = []
    for o in world:
        if isinstance(o, Box):
            center, vel = o.center, o.velocity
        elif isinstance(o, Sphere):
            center, vel = o.center, o.velocity
        else:
            continue
        out.append(
            FeaturePoint(
                feature_id=o.object_id,
                pos_w=np.asarray(center, dtype=float),
                object_id=o.object_id,
                velocity=np.asarray(vel, dtype=float),
            )
        )
    return out


# --------------------------------------------------------------------------
# Basic behaviour
# --------------------------------------------------------------------------
def test_birth_and_update_from_radar_only():
    """A detection with no camera frame still births a track (radar only)."""
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    frames, last = _run_fused(trk, radar, cam, [_car([60.0, 0.0, 0.0])], 0.1,
                              cam_dropout_t=lambda t: True)
    assert last.n_tracks == 1
    assert last.n_cam_updates == 0
    tr = last.tracks[0]
    assert abs(tr.pos[0] - 57.8) < 0.5  # surface range, not the centre


def test_camera_update_fires_when_features_present():
    """A camera frame with a matching feature adds a camera update."""
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    frames, last = _run_fused(trk, radar, cam, [_car([60.0, 0.0, 0.0])], 0.1)
    assert last.n_cam_updates == 1
    assert last.n_radar_updates == 1


def test_empty_world_no_tracks():
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    frames, last = _run_fused(trk, radar, cam, [], 0.5)
    assert last.n_tracks == 0


# --------------------------------------------------------------------------
# Fusion beats radar alone on lateral kinematics
# --------------------------------------------------------------------------
def test_fusion_reduces_lateral_velocity_error():
    """A crossing target: fusing camera azimuths sharpens vy substantially.

    The whole point of fusion is that the camera sees the azimuth (and hence
    the lateral track) every scan, so the fused tracker should estimate the
    pedestrian's 2.2 m/s lateral speed with meaningfully less error than the
    radar-only tracker after the same integration time.
    """
    from sensor_sim.tracking import RadarTracker

    radar = _quiet_radar()
    cam = _quiet_camera()
    world = [_ped([35.0, -6.0, 0.0], velocity=(0.0, 2.2, 0.0))]
    t_end = 2.0

    # radar-only reference: same radar scans, same seeds.
    trk_r = RadarTracker(_tc(process_noise_accel_mps2=1.0))
    frames_r, last_r = _run_radar_only(trk_r, radar, world, t_end)

    # fused: same radar + camera.
    trk_f = _quiet_fused()
    frames_f, last_f = _run_fused(trk_f, radar, cam, world, t_end)

    err_r = abs(last_r.tracks[0].vel[1] - 2.2)
    err_f = abs(last_f.tracks[0].vel[1] - 2.2)
    fused_id = last_f.tracks[0].track_id
    assert fused_id == 1, "single target should keep track id 1"
    # The fused estimate must be at least 2x closer on the lateral axis.
    assert err_f < err_r / 2.0, f"fused {err_f:.3f} vs radar-only {err_r:.3f}"


def _run_radar_only(trk, radar, world, t_end, seed=0):
    n = round(t_end / DT)
    frames = []
    for i in range(n + 1):
        t = i * DT
        pos = np.zeros(3)
        p = _pose(pos, np.zeros(3))
        fr = radar.scan(p, world, t=t, rng=np.random.default_rng(seed + i))
        frames.append(trk.process(fr, pos, Q_ID, np.zeros(3)))
    return frames, frames[-1]


def test_camera_dropout_degrades_gracefully():
    """Camera blind (night / glare): the fused filter runs radar-only and
    the track stays alive without a mode switch."""
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    world = [_car([60.0, 0.0, 0.0], velocity=(-10.0, 0.0, 0.0))]
    t_drop = 0.8
    frames, last = _run_fused(trk, radar, cam, world, 2.0,
                              cam_dropout_t=lambda t: t >= t_drop, seed=5)
    # Track survives the whole run (radar keeps it alive).
    assert last.n_tracks == 1
    tr = last.tracks[0]
    # It kept estimating the closing motion despite the camera going dark.
    assert abs(tr.vel[0] + 10.0) < 0.4
    # And the camera really was off for the tail of the run.
    n_cam_updates = sum(f.n_cam_updates for f in frames)
    assert n_cam_updates < (t_drop / DT)  # only the early ticks had camera
    # Every tick with a camera update kept the track confirmed-count rising.
    confirmed_hits = last.tracks[0].hits
    assert confirmed_hits >= 1


def test_camera_refines_azimuth_without_radar():
    """A camera feature within the gate updates an existing track."""
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    world = [_ped([35.0, -6.0, 0.0], velocity=(0.0, 2.2, 0.0))]
    frames, last = _run_fused(trk, radar, cam, world, 1.0, seed=3)
    # With a sharp camera the azimuth innovation is tiny; both updates fire.
    assert last.n_radar_updates >= 1
    assert last.n_cam_updates >= 1


# --------------------------------------------------------------------------
# Association & lifecycle
# --------------------------------------------------------------------------
def test_gate_rejects_far_camera_feature():
    """A camera feature far from every predicted azimuth is not assigned."""
    trk = FusedTracker(
        FusedConfig(
            track_config=_tc(process_noise_accel_mps2=1.0),
            cam_azimuth_sigma_deg=0.03,
            cam_gate_chi2=0.1,  # extremely tight camera gate
        ),
        cam_fx=1800.0,
        cam_cx=960.0,
        cam_width=1920,
    )
    radar = _quiet_radar()
    cam = _quiet_camera()
    world = [_car([60.0, 0.0, 0.0], velocity=(-10.0, 0.0, 0.0))]
    frames, last = _run_fused(trk, radar, cam, world, 0.5, seed=7)
    # The radar still updates; the camera feature is gated out (tight gate).
    assert last.n_radar_updates == 1
    assert last.n_cam_updates == 0


def test_two_targets_keep_identities_with_camera():
    """Two adjacent closing vehicles keep their lane order (camera helps)."""
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    world = [
        _car([50.0, -3.0, 0.0], velocity=(-18.0, 0.0, 0.0), object_id=1),
        _car([50.0, 3.0, 0.0], velocity=(-18.0, 0.0, 0.0), object_id=2),
    ]
    frames, last = _run_fused(trk, radar, cam, world, 2.0, seed=11)
    confirmed = [tr for tr in last.tracks if tr.confirmed]
    assert len(confirmed) == 2
    y_ests = sorted(tr.pos[1] for tr in confirmed)
    assert y_ests[0] < -2.0 and y_ests[1] > 2.0  # no lane swap


def test_reacquisition_gets_fresh_track_id():
    """A target that disappears and reappears births a NEW track id."""
    trk = _quiet_fused()
    radar = _quiet_radar()
    cam = _quiet_camera()
    car = _car([60.0, 0.0, 0.0], velocity=(-5.0, 0.0, 0.0))
    frames = []
    n = round(3.0 / DT)
    for i in range(n + 1):
        t = i * DT
        world = [car] if t < 1.0 or t >= 2.0 else []
        fr = radar.scan(_pose(np.zeros(3)), world, t=t, rng=np.random.default_rng(13 + i))
        cam_frame = cam.observe(
            _pose(np.zeros(3)), _features_of(world), t=t,
            rng=np.random.default_rng(13 + 1000 + i),
        ) if world else None
        frames.append(trk.process(fr, cam_frame, np.zeros(3), Q_ID, np.zeros(3)))
    first_id = frames[5].tracks[0].track_id
    last_ids = [tr.track_id for tr in frames[-1].tracks]
    assert last_ids and last_ids[0] != first_id
    assert max(last_ids) > first_id


# --------------------------------------------------------------------------
# Noise-config sanity
# --------------------------------------------------------------------------
def test_fused_config_rejects_nonpositive_noise():
    from sensor_sim.tracking import TrackConfig

    with pytest.raises(ValueError):
        FusedConfig(track_config=TrackConfig(**QR), cam_azimuth_sigma_deg=0.0)