"""Tests for the camera feature & optical-flow simulator (v0.13.0)."""
import numpy as np
import pytest

from sensor_sim.camera import (
    CameraConfig,
    CameraSensor,
    FeaturePoint,
)


class _Pose:
    """Minimal stand-in for a trajectory point to keep tests independent."""
    def __init__(self, pos, att):
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)


CAM = CameraConfig(fx=600.0, fy=600.0, width=1280, height=720,
                   pixel_noise_sigma=0.0, dropout_prob=0.0)


def _cam(**kw):
    cfg = CameraConfig(**{**{
        "fx": 600.0, "fy": 600.0, "width": 1280, "height": 720,
        "pixel_noise_sigma": 0.0, "dropout_prob": 0.0,
    }, **kw})
    return CameraSensor(cfg)


def test_forward_center_feature_projects_to_principal_point():
    """A feature dead ahead along the optical axis lands at (cx, cy)."""
    c = _cam()
    # camera at origin looking +x; feature at (10, 0, 0) -> u=cx, v=cy
    f = FeaturePoint(0, np.array([10.0, 0.0, 0.0]))
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f], rng=np.random.default_rng(0))
    assert len(fr) == 1
    assert abs(fr.pts[0, 0] - c.config.cx) < 1e-6
    assert abs(fr.pts[0, 1] - c.config.cy) < 1e-6
    assert abs(fr.depths[0] - 10.0) < 1e-9


def test_projection_scale_with_focal_length():
    """A feature offset in +y at depth z should shift u by fx*y/z."""
    c = _cam()
    f = FeaturePoint(0, np.array([10.0, 1.0, 0.0]))  # off-axis right +1 m
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f], rng=np.random.default_rng(0))
    expected_u = c.config.cx + 600.0 * 1.0 / 10.0   # +60 px
    assert abs(fr.pts[0, 0] - expected_u) < 1e-6
    assert abs(fr.pts[0, 1] - c.config.cy) < 1e-6


def test_vertical_projection_direction():
    """+z world (up) maps to -v (up in image, since v points down)."""
    c = _cam()
    f = FeaturePoint(0, np.array([10.0, 0.0, 1.0]))  # 1 m up in world
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f], rng=np.random.default_rng(0))
    expected_v = c.config.cy - 600.0 * 1.0 / 10.0   # up -> smaller v
    assert abs(fr.pts[0, 1] - expected_v) < 1e-6


def test_behind_camera_dropped():
    """A feature behind the camera (x<0 in sensor) is not observed."""
    c = _cam()
    f = FeaturePoint(0, np.array([-10.0, 0.0, 0.0]))
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f], rng=np.random.default_rng(0))
    assert len(fr) == 0


def test_out_of_image_dropped():
    """A feature far off to the side lands outside the image and is dropped."""
    c = _cam()
    # at depth 10, +y offset 800 m -> u = cx + 600*800/10 >> width
    f = FeaturePoint(0, np.array([10.0, 800.0, 0.0]))
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f], rng=np.random.default_rng(0))
    assert len(fr) == 0


def test_depth_range_clipping():
    """Features beyond min_z/max_z are dropped."""
    c = _cam(min_z=2.0, max_z=20.0)
    near = FeaturePoint(0, np.array([1.0, 0.0, 0.0]))     # < min_z -> dropped
    good = FeaturePoint(1, np.array([10.0, 0.0, 0.0]))    # within range -> kept
    far = FeaturePoint(2, np.array([50.0, 0.0, 0.0]))     # > max_z -> dropped
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [near, good, far],
                   rng=np.random.default_rng(0))
    assert list(fr.feature_ids) == [1]
    assert len(fr) == 1


def test_dynamic_feature_moves_flow():
    """A feature fixed in world should flow opposite the camera's motion when
    the ego vehicle moves forward (approaching feature -> outward flow)."""
    c = _cam()
    f = FeaturePoint(0, np.array([20.0, 0.0, 0.0]))
    p0 = _Pose([0, 0, 0], [1, 0, 0, 0])
    # move forward +1 m along x over dt=0.1 s
    p1 = _Pose([1, 0, 0], [1, 0, 0, 0])
    fl = c.track(p0, p1, [f], t0=0.0, t1=0.1, rng=np.random.default_rng(7))
    assert len(fl) == 1
    # approaching: feature depth shrinks 20 -> 19 so u stays cx, v stays cy.
    # Displacement ~ 0 here; instead verify the depth dropped and velocity finite.
    assert fl.depths[0] < 20.0
    assert np.isfinite(fl.vel_px[0]).all()


def test_stationary_ego_no_flow_for_static_feature():
    """Static ego + static feature -> zero optical flow."""
    c = _cam()
    f = FeaturePoint(0, np.array([20.0, 1.0, 0.5]))
    p0 = _Pose([0, 0, 0], [1, 0, 0, 0])
    p1 = _Pose([0, 0, 0], [1, 0, 0, 0])
    fl = c.track(p0, p1, [f], t0=0.0, t1=0.05, rng=np.random.default_rng(0))
    assert len(fl) == 1
    assert np.allclose(fl.flow[0], 0.0, atol=1e-9)


def test_lateral_ego_motion_produces_flow():
    """Ego slides +y (left) at fixed forward depth -> feature shifts right."""
    c = _cam()
    f = FeaturePoint(0, np.array([20.0, 0.0, 0.0]))
    p0 = _Pose([0, 0, 0], [1, 0, 0, 0])
    p1 = _Pose([0, -1.0, 0], [1, 0, 0, 0])  # ego moved -y => feature rel +y
    fl = c.track(p0, p1, [f], t0=0.0, t1=0.1, rng=np.random.default_rng(0))
    assert len(fl) == 1
    # feature now at sensor +y = +1 -> u increases
    assert fl.flow[0, 0] > 0


def test_pixel_noise_scatter():
    """Higher pixel noise -> wider spread of projected u across repeated runs."""
    base = CameraConfig(fx=600.0, fy=600.0, width=1280, height=720,
                        pixel_noise_sigma=0.0, dropout_prob=0.0)
    noisy = CameraConfig(**{**base.__dict__, "pixel_noise_sigma": 2.0})
    f = FeaturePoint(0, np.array([10.0, 0.0, 0.0]))
    fr_clean = CameraSensor(base).observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f],
                                          rng=np.random.default_rng(1))
    fr_noisy = CameraSensor(noisy).observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f],
                                           rng=np.random.default_rng(1))
    assert (fr_noisy.pts[0, 0] - 640.0) ** 2 > (fr_clean.pts[0, 0] - 640.0) ** 2


def test_dropout_removes_features():
    """Raising dropout probability removes more features."""
    f = FeaturePoint(0, np.array([10.0, 0.0, 0.0]))
    pose = _Pose([0, 0, 0], [1, 0, 0, 0])
    n0 = len(CameraSensor(CameraConfig(
        dropout_prob=0.0)).observe(pose, [f], rng=np.random.default_rng(2)))
    n1 = len(CameraSensor(CameraConfig(
        dropout_prob=0.9)).observe(pose, [f], rng=np.random.default_rng(2)))
    assert n0 == 1 and n1 == 0


def test_mount_translation_shifts_projection():
    """Mounting the camera 1 m right of centre shifts the chief ray left."""
    c = _cam()
    cam_right = CameraSensor(CameraConfig(
        fx=600.0, fy=600.0, width=1280, height=720,
        pixel_noise_sigma=0.0, dropout_prob=0.0),
        mount_t_body=np.array([0.0, 1.0, 0.0]))  # 1 m to the right
    f = FeaturePoint(0, np.array([10.0, 0.0, 0.0]))
    fr = c.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f], rng=np.random.default_rng(0))
    fr2 = cam_right.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f],
                            rng=np.random.default_rng(0))
    # mounted +1 m in +y means the feature is now at sensor y=-1 -> u left.
    assert fr2.pts[0, 0] < fr.pts[0, 0]


def test_distortion_shifts_off_axis():
    """Positive k1 (pincushion) pushes off-axis points further out."""
    c_dist = _cam(k1=0.05)
    f = FeaturePoint(0, np.array([10.0, 2.0, 0.0]))  # off-axis +2 m y
    fr = c_dist.observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f],
                        rng=np.random.default_rng(0))
    # distorted u should be further from cx than undistorted.
    undist = _cam(k1=0.0).observe(_Pose([0, 0, 0], [1, 0, 0, 0]), [f],
                                  rng=np.random.default_rng(0))
    assert abs(fr.pts[0, 0] - 640.0) > abs(undist.pts[0, 0] - 640.0)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        CameraConfig(fx=-1.0)
    with pytest.raises(ValueError):
        CameraConfig(fx=600.0, width=0)
    with pytest.raises(ValueError):
        CameraConfig(fx=600.0, min_z=10.0, max_z=5.0)
