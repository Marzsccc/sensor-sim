"""Tests for ADAS marker projection & display-time latency compensation (v0.9.0)."""

import numpy as np
import pytest

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.adas import (
    AdasMarker,
    MarkerProjector,
    AdasPipeline,
    lead_vehicle_marker,
    hazard_marker,
    lane_line_marker,
)


def _straight_traj(speed=20.0, dur=10.0, dt=0.01):
    ts = np.arange(0.0, dur, dt)
    wps = [
        Waypoint(
            t=t,
            pos=np.array([speed * t, 0.0, 0.0]),
            vel=np.array([speed, 0.0, 0.0]),
            att=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
        )
        for t in ts
    ]
    return Trajectory(wps, dt=dt)


def _P0():
    P = np.eye(9)
    P[0:3, 0:3] *= 0.05
    P[3:6, 3:6] *= 0.1
    P[6:9, 6:9] *= 1e-4
    return P


def test_lead_vehicle_marker_has_points():
    m = lead_vehicle_marker(np.array([25.0, 2.0, 0.0]))
    assert m.kind == "lead_vehicle"
    assert m.points_world is not None and len(m.points_world) >= 8
    assert np.allclose(m.ref_world, [25.0, 2.0, 0.0])


def test_hazard_marker_carries_wedge_params():
    m = hazard_marker(np.array([15.0, 0.0, 0.0]), half_angle_deg=25.0, max_range=40.0)
    assert m.kind == "hazard"
    assert m.half_angle_deg == 25.0
    assert m.max_range == 40.0


def test_lane_line_marker():
    m = lane_line_marker(np.array([0.0, -1.75, 0.0]), np.array([80.0, -1.75, 0.0]), "lane_right")
    assert m.kind == "lane_right"
    assert m.points_world.shape == (2, 3)


def test_to_body_identity_pose():
    # ego at origin with identity attitude -> world == body coords
    p = np.array([10.0, 2.0, -1.0])
    body = MarkerProjector.to_body(p, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(body, p)


def test_to_body_translation():
    # ego at x=5 -> world point 15 appears at body x=10
    body = MarkerProjector.to_body(
        np.array([15.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
    )
    assert np.allclose(body, [10.0, 0.0, 0.0])


def test_screen_xy_forward():
    # point straight ahead, slightly right and down
    _, valid = MarkerProjector.screen_xy(np.array([20.0, 2.0, -1.0]), f_px_x=1000.0, f_px_y=1000.0)
    assert valid
    xy, valid = MarkerProjector.screen_xy(np.array([20.0, 2.0, -1.0]), 1000.0, 1000.0)
    assert valid
    assert xy[0] > 0 and xy[1] < 0  # right positive, down negative (z down)

    # behind HUD -> invalid
    xy, valid = MarkerProjector.screen_xy(np.array([-5.0, 2.0, 0.0]), 1000.0, 1000.0)
    assert not valid


def test_pipeline_improvement():
    """Compensated renderer must reduce marker reprojection error vs naive."""
    traj = _straight_traj()
    lead = lead_vehicle_marker(np.array([25.0, 2.0, 0.0]))
    pipe = AdasPipeline(traj=traj, markers=[lead], sensor_delay_s=0.05,
                        display_rate=60.0, use_wheel=True)
    res = pipe.run(_P0(), sigma_scale=1.0, f_px=1500.0)
    m = res["markers"]["lead_vehicle"]
    assert m["naive_radial_mean"] > m["comp_radial_mean"]
    # naive lag should be on the order of delay*speed (~1.3 m)
    assert m["naive_radial_mean"] > 1.0
    assert m["comp_radial_mean"] < 0.5
    # pixel improvement, if measurable
    if m.get("naive_px_mean") is not None and m.get("comp_px_mean") is not None:
        assert m["naive_px_mean"] > m["comp_px_mean"]


def test_pipeline_fade_alpha_in_range():
    traj = _straight_traj()
    lead = lead_vehicle_marker(np.array([25.0, 2.0, 0.0]))
    pipe = AdasPipeline(traj=traj, markers=[lead], sensor_delay_s=0.05,
                        display_rate=60.0, use_wheel=True)
    res = pipe.run(_P0(), sigma_scale=1.0, f_px=1500.0)
    assert 0.0 <= res["fade_alpha_mean"] <= 1.0


def test_pipeline_multiple_markers_consistency():
    traj = _straight_traj()
    lead = lead_vehicle_marker(np.array([25.0, 2.0, 0.0]))
    haz = hazard_marker(np.array([15.0, 0.0, 0.0]))
    ll = lane_line_marker(np.array([0.0, -1.75, 0.0]), np.array([80.0, -1.75, 0.0]), "lane_left")
    pipe = AdasPipeline(traj=traj, markers=[lead, haz, ll], sensor_delay_s=0.05,
                        display_rate=60.0, use_wheel=True)
    res = pipe.run(_P0(), sigma_scale=1.0, f_px=1500.0)
    assert set(res["markers"].keys()) == {"lead_vehicle", "hazard", "lane_left"}
    for m in res["markers"].values():
        assert m["count"] > 0
        assert m["comp_radial_mean"] <= m["naive_radial_mean"] + 1e-9
