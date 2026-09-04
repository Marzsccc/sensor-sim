"""Tests for the automotive radar simulator (v0.17.0)."""

import numpy as np
import pytest

from sensor_sim.lidar import Box, GroundPlane, Sphere
from sensor_sim.radar import RadarConfig, RadarFrame, RadarSensor


class _Pose:
    """Minimal stand-in for a trajectory point to keep tests independent."""

    def __init__(self, pos, att, vel=None):
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)
        self.vel = None if vel is None else np.asarray(vel, dtype=float)


Q_ID = np.array([1.0, 0.0, 0.0, 0.0])  # identity (no rotation)


def _quiet_cfg(**kw):
    """Config with measurement noise off so geometry checks are exact."""
    kw.setdefault("range_noise_sigma_m", 0.0)
    kw.setdefault("angle_noise_sigma_deg", 0.0)
    kw.setdefault("range_rate_noise_sigma_mps", 0.0)
    return RadarConfig(**kw)


def _car(center, velocity=(0.0, 0.0, 0.0), object_id=1, reflectivity=1.0):
    """A car-sized box (4.4 m long) with full radar reflectivity.

    Centered at z = 0 (sitting on the ground plane) so rays from a
    z = 0 host are exactly horizontal -- range/Doppler expectations in
    the geometry tests stay exact.
    """
    return Box(
        object_id=object_id,
        center=np.asarray(center, dtype=float),
        half_extents=np.array([2.2, 1.0, 0.7]),
        velocity=np.asarray(velocity, dtype=float),
        reflectivity=reflectivity,
    )


# --------------------------------------------------------------------------
# Geometry & gating
# --------------------------------------------------------------------------
def test_no_doppler_stationary():
    """Ego and target both stationary -> range-rate ~ 0."""
    rs = RadarSensor(_quiet_cfg())
    world = [_car([60.0, 0.0, 0.0])]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    assert abs(fr.range_rates[0]) < 1e-9
    assert abs(fr.range_rate_true[0]) < 1e-9


def test_pose_without_vel_defaults_to_stationary_host():
    """A pose object lacking ``vel`` must not break the Doppler term."""

    class _NoVel:
        def __init__(self, pos, att):
            self.pos = np.asarray(pos, dtype=float)
            self.att = np.asarray(att, dtype=float)

    rs = RadarSensor(_quiet_cfg())
    world = [_car([60.0, 0.0, 0.0], velocity=(0.0, 0.0, 0.0))]
    fr = rs.scan(_NoVel([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    assert abs(fr.range_rates[0]) < 1e-9
    assert np.isfinite(fr.range_rates[0])


def test_closing_positive_convention():
    """Target approaching the (stationary) ego -> range-rate positive."""
    # Target 60 m ahead driving toward the ego at 10 m/s (-x direction).
    rs = RadarSensor(_quiet_cfg())
    world = [_car([60.0, 0.0, 0.0], velocity=(-10.0, 0.0, 0.0))]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    assert fr.range_rate_true[0] == pytest.approx(10.0, abs=1e-9)
    assert fr.range_rates[0] == pytest.approx(10.0, abs=1e-9)


def test_receding_negative_convention():
    """Target driving away -> range-rate negative."""
    rs = RadarSensor(_quiet_cfg())
    world = [_car([60.0, 0.0, 0.0], velocity=(8.0, 0.0, 0.0))]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert fr.range_rate_true[0] == pytest.approx(-8.0, abs=1e-9)


def test_doppler_is_relative_speed():
    """Same-direction convoy: closing rate = v_ego - v_lead."""
    # Ego at 25 m/s chasing a lead at 15 m/s -> closing +10 m/s.
    rs = RadarSensor(_quiet_cfg())
    world = [_car([80.0, 0.0, 0.0], velocity=(15.0, 0.0, 0.0))]
    fr = rs.scan(
        _Pose([10.0, 0, 0], Q_ID, vel=(25.0, 0.0, 0.0)),
        world,
        t=0.0,
        rng=np.random.default_rng(0),
    )
    assert len(fr.ranges) == 1
    assert fr.range_rate_true[0] == pytest.approx(10.0, abs=1e-9)


def test_range_to_near_surface_of_extended_target():
    """A 4.4 m-long box centred at 60 m reads ~57.8 m (near surface)."""
    rs = RadarSensor(_quiet_cfg())
    world = [_car([60.0, 0.0, 0.0])]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    assert fr.range_true[0] == pytest.approx(57.8, abs=1e-9)
    assert fr.ranges[0] == pytest.approx(57.8, abs=1e-9)


def test_azimuth_geometry_lateral_offset():
    """Target 20 m ahead / 20 m left -> azimuth +45 deg (left is +y)."""
    rs = RadarSensor(_quiet_cfg())
    world = [_car([20.0, 20.0, 0.0])]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    assert fr.azimuths[0] == pytest.approx(np.deg2rad(45.0), abs=1e-9)


def test_mount_offset_forward_reduces_range():
    """Radar mounted 1 m ahead of the body origin sees targets ~1 m closer."""
    rs = RadarSensor(_quiet_cfg(), mount_t_body=np.array([1.0, 0.0, 0.0]))
    world = [_car([60.0, 0.0, 0.0])]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    assert fr.range_true[0] == pytest.approx(56.8, abs=1e-9)


def test_azimuth_fov_gate_excludes_side_target():
    """Target at 80 deg azimuth is outside the +-60 deg FOV -> no detection."""
    rs = RadarSensor(_quiet_cfg())
    world = [_car([10.0, 60.0, 0.0])]  # atan2(60, 10) ~ 80.5 deg
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 0


def test_range_gate_excludes_close_and_far():
    """Objects inside range_min or beyond range_max are dropped."""
    cfg = _quiet_cfg(range_min=5.0, range_max=100.0)
    rs = RadarSensor(cfg)
    world = [
        _car([3.0, 0.0, 0.0], object_id=1),  # too close
        _car([50.0, 0.0, 0.0], object_id=2),  # valid
        _car([400.0, 0.0, 0.0], object_id=3),
    ]  # too far
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert list(fr.object_ids) == [2]


def test_ground_plane_never_detected():
    """The ground plane is clutter, not a target: skipped entirely."""
    rs = RadarSensor(_quiet_cfg())
    world = [GroundPlane(0), _car([50.0, 0.0, 0.0])]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert list(fr.object_ids) == [1]


def test_dynamic_target_tracking_over_time():
    """Closing target: measured range drops at ~ the closing speed."""
    rs = RadarSensor(_quiet_cfg())
    lead = _car([80.0, 0.0, 0.0], velocity=(15.0, 0.0, 0.0))
    ranges = []
    rates = []
    for i in range(11):
        t = i * 0.1
        fr = rs.scan(
            _Pose([10.0 + 25.0 * t, 0, 0], Q_ID, vel=(25.0, 0.0, 0.0)),
            [lead],
            t=t,
            rng=np.random.default_rng(0),
        )
        assert len(fr.ranges) == 1
        ranges.append(fr.ranges[0])
        rates.append(fr.range_rates[0])
    # True closing speed 10 m/s over 1 s -> range drops ~10 m.
    assert (ranges[0] - ranges[-1]) == pytest.approx(10.0, abs=1e-6)
    assert all(abs(v - 10.0) < 1e-9 for v in rates)


# --------------------------------------------------------------------------
# Noise statistics (Monte-Carlo)
# --------------------------------------------------------------------------
def test_range_noise_matches_sigma():
    """Range noise std dev matches the configured sigma (loose MC band)."""
    cfg = RadarConfig(
        range_noise_sigma_m=0.2,
        angle_noise_sigma_deg=0.0,
        range_rate_noise_sigma_mps=0.0,
        detect_threshold_linear=1e-9,
    )  # always detect
    rs = RadarSensor(cfg)
    world = [_car([100.0, 0.0, 0.0])]
    rng = np.random.default_rng(7)
    errs = []
    for _ in range(3000):
        fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=rng)
        assert len(fr.ranges) == 1
        errs.append(fr.ranges[0] - fr.range_true[0])
    errs = np.asarray(errs)
    assert abs(errs.mean()) < 0.03
    assert 0.15 < errs.std() < 0.30


def test_range_rate_noise_matches_sigma():
    """Doppler noise std dev matches the configured sigma (loose MC band)."""
    cfg = RadarConfig(
        range_rate_noise_sigma_mps=0.3,
        range_noise_sigma_m=0.0,
        angle_noise_sigma_deg=0.0,
        detect_threshold_linear=1e-9,
    )  # always detect
    rs = RadarSensor(cfg)
    world = [_car([60.0, 0.0, 0.0], velocity=(5.0, 0.0, 0.0))]
    rng = np.random.default_rng(11)
    errs = []
    for _ in range(3000):
        fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=rng)
        assert len(fr.ranges) == 1
        errs.append(fr.range_rates[0] - fr.range_rate_true[0])
    errs = np.asarray(errs)
    assert 0.22 < errs.std() < 0.42


def test_detection_probability_curve_values():
    """P_det at the reference point ~ 1, decaying as R^-4 away from it."""
    cfg = RadarConfig()
    # 40 dB at 50 m with threshold 100 -> SNR 1e4 -> P_d ~ 0.9901.
    assert cfg.detection_probability(50.0, 10.0) == pytest.approx(
        1e4 / (1e4 + 100.0), abs=1e-9
    )
    # Double range -> SNR /16 -> P_d ~ 0.862; 4x range -> ~ 0.28.
    p100 = cfg.detection_probability(100.0, 10.0)
    p200 = cfg.detection_probability(200.0, 10.0)
    assert p100 == pytest.approx(625.0 / 725.0, abs=1e-6)
    assert p200 == pytest.approx(39.0625 / 139.0625, abs=1e-6)
    # Small RCS at long range -> dim.
    assert cfg.detection_probability(200.0, 1.0) < p200
    # Monotone decreasing in range.
    assert p100 > p200


def test_far_dim_targets_drop_out_more_often():
    """Detection rate at 50 m >> detection rate at 200 m (MC band)."""
    # 200 m = 4x the 50 m reference -> SNR /256 -> P_d ~ 0.28.
    cfg = RadarConfig(range_max=300.0)  # keep both inside the gate
    rs = RadarSensor(cfg)
    near_w = [_car([50.0, 0.0, 0.0], object_id=1)]
    far_w = [_car([200.0, 0.0, 0.0], object_id=1)]
    rng = np.random.default_rng(3)
    n_near = sum(
        1
        for _ in range(400)
        if len(rs.scan(_Pose([0, 0, 0], Q_ID), near_w, rng=rng).ranges) == 1
    )
    rng = np.random.default_rng(3)
    n_far = sum(
        1
        for _ in range(400)
        if len(rs.scan(_Pose([0, 0, 0], Q_ID), far_w, rng=rng).ranges) == 1
    )
    # Expected ~0.99 vs ~0.28; assert wide bands + strict ordering.
    assert n_near > 370
    assert 60 < n_far < 180
    assert n_near > n_far


def test_small_rcs_object_detected_less_than_car():
    """A 1 m^2 reflector (reflectivity 0.1) drops out more than a 10 m^2 car."""
    cfg = RadarConfig()
    rs = RadarSensor(cfg)
    car = _car([120.0, 0.0, 0.0], object_id=1, reflectivity=1.0)
    bike = Sphere(
        object_id=2, center=np.array([120.0, 0.0, 0.0]), radius=0.4, reflectivity=0.1
    )
    rng = np.random.default_rng(5)
    n_car = n_bike = 0
    for _ in range(2000):
        fr = rs.scan(_Pose([0, 0, 0], Q_ID), [car, bike], rng=rng)
        ids = set(fr.object_ids.tolist())
        n_car += 1 in ids
        n_bike += 2 in ids
    assert n_car > n_bike


# --------------------------------------------------------------------------
# Frame contract
# --------------------------------------------------------------------------
def test_empty_frame_when_nothing_detected():
    """An empty scan returns a well-formed zero-length frame."""
    rs = RadarSensor(_quiet_cfg())
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), [], rng=np.random.default_rng(0))
    assert isinstance(fr, RadarFrame)
    assert fr.ranges.shape == (0,)
    assert fr.points.shape == (0, 3)
    assert fr.object_ids.dtype.kind == "i"


def test_points_are_sensor_frame_from_noisy_polar():
    """Reconstructed points satisfy x = r cos(el) cos(az) in the body frame."""
    cfg = RadarConfig(
        range_noise_sigma_m=0.05,
        angle_noise_sigma_deg=0.3,
        range_rate_noise_sigma_mps=0.1,
    )
    rs = RadarSensor(cfg)
    world = [_car([60.0, 0.0, 0.0])]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, t=0.0, rng=np.random.default_rng(0))
    assert len(fr.ranges) == 1
    x = fr.ranges[0] * np.cos(fr.elevations[0]) * np.cos(fr.azimuths[0])
    y = fr.ranges[0] * np.cos(fr.elevations[0]) * np.sin(fr.azimuths[0])
    z = fr.ranges[0] * np.sin(fr.elevations[0])
    assert fr.points[0] == pytest.approx([x, y, z], abs=1e-9)


def test_snr_db_reported_for_each_detection():
    """snr_db is the radar-equation SNR of the *surface* range."""
    cfg = _quiet_cfg()
    rs = RadarSensor(cfg)
    world = [_car([60.0, 0.0, 0.0], object_id=1)]
    fr = rs.scan(_Pose([0, 0, 0], Q_ID), world, rng=np.random.default_rng(0))
    # RCS = 10 m^2 (reflectivity 1.0); surface range 57.8 m -> exact dB.
    snr_lin = 1e4 * (cfg.range_ref_m / fr.range_true[0]) ** 4
    assert fr.snr_db[0] == pytest.approx(10.0 * np.log10(snr_lin), abs=1e-6)
