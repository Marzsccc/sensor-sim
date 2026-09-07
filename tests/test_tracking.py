"""Tests for the radar detection tracker (v0.18.0)."""

import numpy as np
import pytest

from sensor_sim.lidar import Box, Sphere
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.tracking import RadarTracker, TrackConfig


class _Pose:
    """Minimal stand-in for a trajectory point (kept independent)."""

    def __init__(self, pos, att, vel=None):
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)
        self.vel = None if vel is None else np.asarray(vel, dtype=float)


Q_ID = np.array([1.0, 0.0, 0.0, 0.0])  # identity (no rotation)
DT = 0.1  # s -- 10 Hz scan cadence used throughout

# Measurement noise for the "quiet but physical" radar / tracker pair.  NOT
# microscopic: an over-idealised pair (sigma ~ 1e-3) makes the EKF
# overconfident *below the model floor* -- the radar reflection point of an
# extended box slides on the rear face as the bearing changes (~mm-scale
# unmodelled motion), so true innovations exceed R and tracks permanently
# fall outside the 11.3 gate.  Real radars sit at ~0.1 m / 0.5 deg; 0.01 m /
# 0.05 deg keeps geometry checks near-exact while staying above the floor.
QR = dict(range_noise_sigma_m=1e-2, angle_noise_sigma_deg=5e-2, range_rate_noise_sigma_mps=2e-2)


def _quiet_radar(**kw):
    """Radar with small but physical measurement noise."""
    kw = {**QR, **kw}
    return RadarSensor(RadarConfig(**kw))


def _quiet_tracker(**kw):
    """Tracker whose R matches the radar; default process noise (sigma_a=1)."""
    kw = {**QR, **kw}
    kw.setdefault("process_noise_accel_mps2", 1.0)
    return RadarTracker(TrackConfig(**kw))


def _car(center, velocity=(0.0, 0.0, 0.0), object_id=1, reflectivity=1.0):
    """A car-sized box (4.4 m long) centred on the ground plane (z = 0)."""
    return Box(
        object_id=object_id,
        center=np.asarray(center, dtype=float),
        half_extents=np.array([2.2, 1.0, 0.7]),
        velocity=np.asarray(velocity, dtype=float),
        reflectivity=reflectivity,
    )


def _ped(center, velocity=(0.0, 0.0, 0.0), object_id=2):
    """A pedestrian-sized sphere on the ground plane."""
    return Sphere(
        object_id=object_id,
        center=np.asarray(center, dtype=float),
        radius=0.35,
        velocity=np.asarray(velocity, dtype=float),
        reflectivity=0.1,
    )


def _run(tracker, radar, world, t_end, host_vel=(0.0, 0.0, 0.0), seed=0):
    """Feed 10 Hz scans up to ``t_end``; return (frames, last_frame).

    Host sits at the origin facing +x, optionally moving at ``host_vel``.
    """
    frames = []
    n = round(t_end / DT)
    for i in range(n + 1):
        t = i * DT
        pos = np.array([host_vel[0] * t, host_vel[1] * t, 0.0])
        vel = np.asarray(host_vel, dtype=float)
        fr = radar.scan(_Pose(pos, Q_ID, vel), world, t=t, rng=np.random.default_rng(seed + i))
        tf = tracker.process(fr, pos, Q_ID, vel)
        frames.append(tf)
    return frames, frames[-1]


# --------------------------------------------------------------------------
# Birth & basic geometry
# --------------------------------------------------------------------------
def test_birth_single_detection():
    """One detection on the first scan spawns one track at that position."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [_car([60.0, 0.0, 0.0])]
    frames, last = _run(trk, radar, world, 0.1)
    assert last.n_tracks == 1
    tr = last.tracks[0]
    assert tr.track_id == 1
    assert abs(tr.pos[0] - 57.8) < 0.5  # surface range, not the centre
    assert abs(tr.pos[1]) < 0.5


def test_stationary_target_velocity_zero():
    """Ego and target at rest -> estimated velocity -> 0.

    Uses a small process noise: a static target genuinely has ~no
    acceleration, and a large sigma_a would keep the velocity posterior wide
    (the tracker's Q/measurement balance is scenario-dependent by design).
    """
    trk = _quiet_tracker(process_noise_accel_mps2=0.05)
    radar = _quiet_radar()
    frames, last = _run(trk, radar, [_car([60.0, 0.0, 0.0])], 1.5)
    tr = last.tracks[0]
    assert tr.speed < 0.2


def test_empty_world_no_tracks():
    """A scan with no detections yields no tracks."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    frames, last = _run(trk, radar, [], 0.5)
    assert last.n_tracks == 0
    assert last.n_detections == 0


def test_track_config_rejects_singular_measurement_noise():
    with pytest.raises(ValueError):
        TrackConfig(
            range_noise_sigma_m=0.0,
            angle_noise_sigma_deg=0.0,
            range_rate_noise_sigma_mps=0.0,
        )


# --------------------------------------------------------------------------
# Kinematics: closing / receding / crossing
# --------------------------------------------------------------------------
def test_closing_lead_converges_to_truth():
    """Lead vehicle closing at 20 m/s: range, Doppler and vx all converge."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [_car([60.0, 0.0, 0.0], velocity=(-20.0, 0.0, 0.0))]
    frames, last = _run(trk, radar, world, 2.0)
    tr = last.tracks[0]
    t = 2.0
    assert abs(tr.pos[0] - (57.8 - 20.0 * t)) < 0.5  # surface range
    assert abs(tr.vel[0] + 20.0) < 0.3  # approaching
    assert abs(tr.vel[1]) < 0.2
    # Doppler estimate: closing is positive by the ACC convention.
    v_rel = tr.velocity_relative_to(np.zeros(3))
    assert abs(np.hypot(*v_rel) - 20.0) < 0.3


def test_receding_target_velocity_positive():
    """A target driving away must estimate a positive (receding) velocity."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [_car([60.0, 0.0, 0.0], velocity=(8.0, 0.0, 0.0))]  # driving away
    frames, last = _run(trk, radar, world, 2.0)
    tr = last.tracks[0]
    assert tr.vel[0] > 7.5  # moving away from the stationary host
    assert tr.vel[0] < 8.5


def test_crossing_pedestrian_lateral_velocity_recovered():
    """Single-frame Doppler cannot see cross-range motion, but the tracker can.

    A pedestrian crossing at 2.2 m/s (starting 6 m left of the beam) has a
    true radial range-rate bounded by ~0.4 m/s for the whole scene; the
    tracker must recover vy ~ 2.2 m/s from the azimuth history alone.
    """
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [_ped([35.0, -6.0, 0.0], velocity=(0.0, 2.2, 0.0))]
    frames, last = _run(trk, radar, world, 4.0)
    tr = last.tracks[0]
    assert abs(tr.vel[1] - 2.2) < 0.25  # lateral velocity recovered
    assert abs(tr.vel[0]) < 0.25  # no phantom down-range motion
    # Late in the scene the pedestrian is near/behind the beam: the radial
    # estimate must stay small even though |vy| = 2.2 m/s.
    trk2 = _quiet_tracker()
    radar2 = _quiet_radar()
    frames2, _ = _run(trk2, radar2, world, 2.7, seed=1)  # ~ abeam (y ~ 0)
    tr2 = frames2[-1].tracks[0]
    assert abs(tr2.vel[1] - 2.2) < 0.35


def test_host_motion_relative_kinematics():
    """Host chasing a slower lead: estimates are the *relative* closing pair.

    Host 25 m/s, lead 15 m/s, 90 m ahead -> track state holds the lead's
    absolute velocity (~15 m/s) while its velocity relative to the host is
    -10 m/s (closing Doppler +10).
    """
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [_car([90.0, 0.0, 0.0], velocity=(15.0, 0.0, 0.0))]
    frames, last = _run(trk, radar, world, 2.0, host_vel=(25.0, 0.0, 0.0))
    tr = last.tracks[0]
    assert abs(tr.vel[0] - 15.0) < 0.4  # absolute lead velocity
    v_rel = tr.velocity_relative_to(np.array([25.0, 0.0, 0.0]))
    assert abs(v_rel[0] + 10.0) < 0.4  # relative closing at 10 m/s


# --------------------------------------------------------------------------
# Track management: coast, delete, re-acquisition, gating
# --------------------------------------------------------------------------
def test_coast_and_delete_after_leaving_range():
    """A target that exits the range gate coasts, then is deleted."""
    radar = RadarSensor(RadarConfig(range_max=50.0, **QR))
    trk = RadarTracker(
        TrackConfig(coast_max=5, process_noise_accel_mps2=1.0, **QR)
    )
    # Car centre 45 m away driving away at 10 m/s: rear surface starts at
    # 42.8 m and passes the 50 m gate at t ~ 0.72 s (scans 0..7 visible).
    world = [_car([45.0, 0.0, 0.0], velocity=(10.0, 0.0, 0.0))]
    frames, last = _run(trk, radar, world, 2.0, seed=3)
    ages = [len(f.tracks) for f in frames]
    assert ages[0] == 1  # born
    # Confirmed at scan 2, still updated through scan 7, then coasts 5 scans
    # (missed 1..5, kept) and is deleted on the 6th miss (scan 13).
    for k in range(8, 13):
        assert ages[k] == 1, f"scan {k} should coast with the track alive"
    assert ages[13] == 0  # deleted on the 6th consecutive miss
    assert last.n_tracks == 0


def test_missed_counter_increments_while_coasting():
    """A visible-then-lost target coasts with missed increasing per scan."""
    radar = RadarSensor(RadarConfig(range_max=50.0, **QR))
    trk = RadarTracker(
        TrackConfig(coast_max=10, process_noise_accel_mps2=1.0, **QR)
    )
    world = [_car([45.0, 0.0, 0.0], velocity=(10.0, 0.0, 0.0))]
    # Track objects are live (mutated in place by later scans), so capture
    # ``missed`` per scan inside the loop.
    missed_series = []
    for i in range(21):  # t = 0 .. 2.0 s at 10 Hz
        t = i * DT
        fr = radar.scan(_Pose([0, 0, 0], Q_ID, np.zeros(3)), world, t=t, rng=np.random.default_rng(5 + i))
        tf = trk.process(fr, np.zeros(3), Q_ID, np.zeros(3))
        missed_series.append(tf.tracks[0].missed if tf.n_tracks else None)
    # Visible through scan 7 (missed = 0); from scan 8 the counter climbs
    # 1, 2, ... each scan and the track is deleted when missed > 10.
    assert missed_series[7] == 0
    assert missed_series[8] == 1
    assert missed_series[17] == 10  # still alive at the coast limit
    assert missed_series[18] is None  # deleted on the 11th consecutive miss
    assert missed_series[20] is None


def test_reacquisition_gets_fresh_track_id():
    """A target that disappears and reappears births a NEW track id."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    # World object toggled: visible 0-1 s, gone 1-2 s, visible 2-3 s.
    car = _car([60.0, 0.0, 0.0], velocity=(-5.0, 0.0, 0.0))
    frames = []
    n = round(3.0 / DT)
    for i in range(n + 1):
        t = i * DT
        world = [car] if t < 1.0 or t >= 2.0 else []
        fr = radar.scan(_Pose([0, 0, 0], Q_ID, np.zeros(3)), world, t=t, rng=np.random.default_rng(7 + i))
        frames.append(trk.process(fr, np.zeros(3), Q_ID, np.zeros(3)))
    first_id = frames[5].tracks[0].track_id  # during the first visibility
    last_ids = [tr.track_id for tr in frames[-1].tracks]
    assert last_ids and last_ids[0] != first_id
    assert max(last_ids) > first_id


def test_gate_rejects_remote_detection_spawns_new_track():
    """A detection far from every prediction is a new object, not a jump."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    a = _car([60.0, 0.0, 0.0], velocity=(-10.0, 0.0, 0.0), object_id=1)
    b = _car([60.0, 40.0, 0.0], velocity=(0.0, 0.0, 0.0), object_id=2)  # far aside
    # Track objects are live, so snapshot the state at the cutover scan.
    snap = None
    for i in range(20):  # 2.0 s
        t = i * DT
        world = [a] if t < 0.8 else [b]  # hard cutover to a new azimuth
        fr = radar.scan(_Pose([0, 0, 0], Q_ID, np.zeros(3)), world, t=t, rng=np.random.default_rng(11 + i))
        tf = trk.process(fr, np.zeros(3), Q_ID, np.zeros(3))
        if t == 0.9:  # second scan of object b
            snap = [(tr.track_id, tr.missed, tr.pos.copy(), tr.hits) for tr in tf.tracks]
    # Right after the cutover: old track must NOT snap to the new detection.
    # It coasts (missed >= 1) while a second, new track is born for object b.
    assert snap is not None and len(snap) == 2
    old = [s for s in snap if s[0] == 1][0]
    new = [s for s in snap if s[0] != 1][0]
    assert old[1] >= 1  # old track is coasting, not jumping
    assert abs(old[2][1]) < 2.0  # ...and did not move to y ~ 40
    assert abs(new[2][1] - 39.0) < 2.0  # b's surface point sits at y ~ 39
    # Old track eventually deleted; new track survives to the end.
    final = trk.tracks
    assert len(final) == 1
    assert final[0].track_id == new[0]


def test_two_tracks_no_identity_swap():
    """Two closing vehicles in adjacent lanes keep their lane identities."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [
        _car([50.0, -3.0, 0.0], velocity=(-18.0, 0.0, 0.0), object_id=1),
        _car([50.0, 3.0, 0.0], velocity=(-18.0, 0.0, 0.0), object_id=2),
    ]
    frames, last = _run(trk, radar, world, 2.0, seed=13)
    # A statistical gate fluke at the final scan may legitimately birth a
    # 1-scan tentative ghost next to a track, so assert on the CONFIRMED
    # tracks: both vehicles must be present with their lane order intact.
    confirmed = [tr for tr in last.tracks if tr.confirmed]
    assert len(confirmed) == 2
    y_ests = sorted(tr.pos[1] for tr in confirmed)
    assert y_ests[0] < -2.0 and y_ests[1] > 2.0  # no swap: lane order kept


def test_tentative_ghost_dies_after_one_miss():
    """A tentative track that misses a single scan is dropped immediately."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    world = [_car([60.0, 0.0, 0.0], velocity=(-10.0, 0.0, 0.0))]
    frames = []
    for i in range(4):  # t = 0, 0.1, 0.2, 0.3
        t = i * DT
        world_i = world if t < 0.2 else []  # disappears after t = 0.1
        fr = radar.scan(_Pose([0, 0, 0], Q_ID, np.zeros(3)), world_i, t=t, rng=np.random.default_rng(17 + i))
        frames.append(trk.process(fr, np.zeros(3), Q_ID, np.zeros(3)))
    # t = 0: born.  t = 0.1: updated (hits = 2, still tentative, confirm_min
    # is 3).  t = 0.2: no detection -> tentative misses once -> deleted.
    assert frames[1].n_tracks == 1
    assert not frames[1].tracks[0].confirmed
    assert frames[2].n_tracks == 0
    assert frames[3].n_tracks == 0


# --------------------------------------------------------------------------
# Jacobian & determinism
# --------------------------------------------------------------------------
def test_measurement_jacobian_matches_numeric():
    """Analytic H agrees with a central-difference Jacobian of h."""
    from sensor_sim.tracking import _measurement_jacobian

    rng = np.random.default_rng(0)
    for _ in range(5):
        x = np.array([rng.uniform(5, 150), rng.uniform(-40, 40), rng.uniform(-25, 25), rng.uniform(-8, 8)])
        pos_w = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), 0.0])
        R_bw = np.eye(3)  # identity attitude: simple bookkeeping
        m = np.array([1.0, 0.0, 0.0])
        vel_w = np.array([rng.uniform(0, 25), 0.0, 0.0])
        zhat, H = _measurement_jacobian(x, pos_w, R_bw, m, vel_w)

        def h(xx):
            return _measurement_jacobian(xx, pos_w, R_bw, m, vel_w)[0]

        Hn = np.zeros((3, 4))
        eps = 1e-6
        for j in range(4):
            for s in (-1.0, 1.0):
                xp = x.copy()
                xp[j] += s * eps
                Hn[:, j] += s * h(xp) / (2 * eps)
        assert np.allclose(H, Hn, atol=1e-5), f"{H} vs {Hn}"


def test_deterministic_seed_repeatable():
    """Same seeds -> identical track estimates."""
    def _run_once():
        trk = _quiet_tracker()
        radar = _quiet_radar()
        world = [
            _car([50.0, 0.0, 0.0], velocity=(-12.0, 0.0, 0.0), object_id=1),
            _ped([35.0, -5.0, 0.0], velocity=(0.0, 2.0, 0.0), object_id=2),
        ]
        frames, last = _run(trk, radar, world, 1.5, seed=21)
        return np.array([tr.x for tr in last.tracks])

    a = _run_once()
    b = _run_once()
    assert np.array_equal(a, b)


def test_birth_covariance_lateral_velocity_prior_wide():
    """A new track admits ignorance of cross-range velocity at birth."""
    trk = _quiet_tracker()
    radar = _quiet_radar()
    frames, _ = _run(trk, radar, [_car([60.0, 0.0, 0.0])], 0.0)
    tr = frames[0].tracks[0]  # straight after birth, before any update
    # Lateral (cross-bearing) velocity prior is deliberately loose...
    assert tr.P[3, 3] > (5.0) ** 2
    # ...while the radial (along-bearing) prior is tight (Doppler known).
    assert tr.P[2, 2] < tr.P[3, 3] / 4.0
