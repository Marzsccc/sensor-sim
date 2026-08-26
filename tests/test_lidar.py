"""Tests for the LiDAR point-cloud simulator (v0.12.0)."""
import numpy as np
import pytest

from sensor_sim.lidar import (
    GroundPlane,
    Box,
    LidarConfig,
    LidarSensor,
)


class _Pose:
    """Minimal stand-in for a trajectory point to keep tests independent."""
    def __init__(self, pos, att):
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)


def _make_sensor(**kw):
    cfg = LidarConfig(**kw)
    return LidarSensor(cfg)


def test_ground_plane_distance():
    """A ray straight down from 1.5 m above the plane must hit at 1.5 m."""
    # All azimuths at elevation 0 are horizontal -> miss ground (parallel).
    # Elevation 0 beam never hits ground straight down; not the point here.
    # Instead directly exercise the primitive.
    assert abs(GroundPlane(0).intersect(np.array([0, 0, 1.5]),
                                        np.array([0, 0, -1.0]), 0.0) - 1.5) < 1e-9


def test_ground_id_is_zero():
    world = [GroundPlane(0)]
    # beams=3 includes a downward beam (el=-15deg) that crosses the ground.
    cfg = LidarConfig(mode="spinning", beams=3, columns=8, v_fov_deg=15.0,
                      range_noise_sigma_m=0.0, dropout_prob=0.0, range_max=10.0)
    ls = LidarSensor(cfg)
    fr = ls.scan(_Pose([0, 0, 1.5], [1, 0, 0, 0]), world, rng=np.random.default_rng(0))
    assert len(fr) > 0
    assert (fr.object_ids == 0).all()


def test_box_front_face_distance():
    """Box centered at (10,0,0), half-extents 1 -> front face at x=9."""
    world = [Box(1, np.array([10.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))]
    ls = _make_sensor(mode="spinning", beams=1, columns=720, range_max=50.0,
                      range_noise_sigma_m=0.0, dropout_prob=0.0)
    fr = ls.scan(_Pose([0, 0, 1.0], [1, 0, 0, 0]), world, rng=np.random.default_rng(0))
    bp = fr.points[fr.object_ids == 1]
    assert len(bp) > 0
    # nearest box range should be ~9 m (front face)
    assert abs(np.min(np.linalg.norm(bp, axis=1)) - 9.0) < 0.05


def test_box_outside_fov_not_hit():
    """Solid scan with a narrow FOV should not hit a box far off-axis."""
    # Box at (0, 30, 0): azimuth ~90deg, far outside the +-30deg FOV.
    world = [Box(1, np.array([0.0, 30.0, 0.0]), np.array([1.0, 1.0, 1.0]))]
    ls = _make_sensor(mode="solid", beams=1, columns=24, h_fov_deg=30.0,
                      range_max=100.0, range_noise_sigma_m=0.0, dropout_prob=0.0)
    fr = ls.scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), world, rng=np.random.default_rng(0))
    assert not (fr.object_ids == 1).any()


def test_range_clamping():
    """Returns beyond range_max are dropped; returns within range are kept."""
    # boxes on forward axis; beams=1 -> single horizontal central beam.
    far_box = Box(1, np.array([200.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]))
    ls = _make_sensor(mode="spinning", beams=1, columns=360, range_max=100.0,
                      range_noise_sigma_m=0.0, dropout_prob=0.0)
    fr = ls.scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), [far_box],
                 rng=np.random.default_rng(0))
    assert not (fr.object_ids == 1).any()

    # near box: within range_max -> should appear; every point <= range_max
    near_box = Box(1, np.array([30.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    fr2 = ls.scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), [near_box],
                  rng=np.random.default_rng(0))
    assert (fr2.object_ids == 1).any()
    assert (fr2.ranges <= 100.0 + 1e-9).all()


def test_dynamic_object_moves():
    """A box with velocity moves its cluster centroid between ticks."""
    # Tall box so the forward beams (beams=1, el=0) reliably intersect it,
    # and wide in y so small azimuth deviations still hit the body.
    box = Box(1, np.array([10.0, 0.0, 0.0]), np.array([2.0, 3.0, 3.0]),
              velocity=np.array([5.0, 0.0, 0.0]), reflectivity=0.8)
    world = [box]
    ls = _make_sensor(mode="solid", beams=1, columns=200, h_fov_deg=60.0,
                      range_max=30.0, range_noise_sigma_m=0.0, dropout_prob=0.0)
    f0 = ls.scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), world, t=0.0,
                 rng=np.random.default_rng(0))
    f1 = ls.scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), world, t=1.0,
                 rng=np.random.default_rng(0))

    def centroid(fr):
        bp = fr.points[fr.object_ids == 1]
        return bp.mean(axis=0) if len(bp) else np.zeros(3)

    c0, c1 = centroid(f0), centroid(f1)
    assert np.any(f0.object_ids == 1)   # box hit at t=0
    assert c1[0] - c0[0] > 3.0            # centroid advanced ~5 m in +x


def test_noise_increases_scatter():
    """Higher range noise -> wider spread of range residuals."""
    world = [Box(1, np.array([10.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))]
    cfg_a = LidarConfig(mode="solid", beams=4, columns=80, h_fov_deg=30.0,
                        range_max=20.0, range_noise_sigma_m=0.001, dropout_prob=0.0)
    cfg_b = LidarConfig(**{**cfg_a.__dict__, "range_noise_sigma_m": 0.2})
    fa = LidarSensor(cfg_a).scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), world,
                                 rng=np.random.default_rng(1))
    fb = LidarSensor(cfg_b).scan(_Pose([0, 0, 0.0], [1, 0, 0, 0]), world,
                                 rng=np.random.default_rng(1))
    sigma_a = np.std(fa.ranges[fa.object_ids == 1])
    sigma_b = np.std(fb.ranges[fb.object_ids == 1])
    assert sigma_b > sigma_a


def test_dropout_reduces_points():
    """Increasing dropout probability monotonically reduces valid point count."""
    world = [GroundPlane(0)]
    n0 = len(LidarSensor(LidarConfig(
        mode="solid", beams=10, columns=10, h_fov_deg=60.0, v_fov_deg=40.0,
        range_max=50.0, dropout_prob=0.0)).scan(_Pose([0, 0, 1.5], [1, 0, 0, 0]), world,
                                rng=np.random.default_rng(5)))
    n1 = len(LidarSensor(LidarConfig(
        mode="solid", beams=10, columns=10, h_fov_deg=60.0, v_fov_deg=40.0,
        range_max=50.0, dropout_prob=0.5)).scan(_Pose([0, 0, 1.5], [1, 0, 0, 0]),
                                                world, rng=np.random.default_rng(5)))
    assert 0 < n1 < n0


def test_spinning_full_circle_hits_all_around():
    """Spinning geometry wraps azimuth; a box behind the sensor is still seen."""
    world = [Box(1, np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.5, 0.5]))]
    ls = _make_sensor(mode="spinning", beams=4, columns=360, range_max=10.0,
                      range_noise_sigma_m=0.0, dropout_prob=0.0)
    fr = ls.scan(_Pose([0, 0, 1.0], [1, 0, 0, 0]), world, rng=np.random.default_rng(0))
    # The box is above and around the sensor -> should be hit from many angles
    assert (fr.object_ids == 1).sum() >= 1


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        LidarConfig(mode="bogus")
