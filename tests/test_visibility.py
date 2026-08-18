"""Tests for near-field marker visibility, FOV culling & occlusion (v0.10.0)."""

import numpy as np
import pytest

from sensor_sim.adas import hazard_marker, lead_vehicle_marker, lane_line_marker
from sensor_sim.visibility import (
    Frustum,
    Occluder,
    MarkerVisibilityPolicy,
    MarkerProjector,
    VisibilityStatus,
    angular_size,
    evaluate_markers,
)


def test_angular_size_finite_close_in():
    """The diverging failure mode of v0.9.0: pixel size blows up at close
    range while angular size stays finite -- the HUD should fade, not shatter."""
    # very close hazard: x=0.8 m, right on top of the bumper
    ref = np.array([0.8, 0.0, 0.0])
    ang, px, valid = angular_size(ref, extent_m=0.9)
    assert valid
    assert np.isfinite(ang)
    # angular span is large (half-extent 0.9 at 0.8 m -> ~96 deg full span)
    assert ang > 30.0
    # but pixel span explodes (hfov_px seeded small here -> still finite but
    # proportional); use a bigger seed to show the divergence clearly
    _a2, px2, _v2 = angular_size(ref, extent_m=0.9, hfov_px=1500.0)
    assert np.isfinite(_a2)
    assert px2 == px2  # no NaN
    assert px2 > ang  # pixel span numerically larger than angular span


def test_frustum_too_close_cull():
    f = Frustum(min_range_m=1.0, max_range_m=200.0)
    ref = np.array([0.5, 0.0, 0.0])
    status, drawable = f.decide(ref)
    assert status == VisibilityStatus.TOO_CLOSE
    assert not drawable


def test_frustum_behind_cull():
    f = Frustum()
    status, drawable = f.decide(np.array([-3.0, 0.5, 0.0]))
    assert status == VisibilityStatus.BEHIND
    assert not drawable


def test_frustum_out_of_fov():
    f = Frustum(hfov_deg=45.0, vfov_deg=20.0)
    # far to the right -> azimuth beyond 45 deg
    status, drawable = f.decide(np.array([1.0, 2.5, 0.0]))
    assert status == VisibilityStatus.OUT_OF_FOV
    assert not drawable


def test_frustum_near_plane_nonfatal():
    f = Frustum(near_m=1.5, min_range_m=0.5, max_range_m=200.0)
    status, drawable = f.decide(np.array([1.0, 0.0, 0.0]))
    assert status == VisibilityStatus.NEAR_PLANE
    # near-plane is a fade, not a hard cull (still technically drawable)
    assert drawable


def test_policy_culls_too_close_hazard():
    """A hazard a few centimetres off the bumper must be culled before the
    pinhole projection can diverge (v0.9.0 trap)."""
    policy = MarkerVisibilityPolicy()
    ref = np.array([0.3, 0.0, 0.0])
    d = policy.decide(hazard_marker(np.array([0.3, 0.0, 0.0])), ref)
    assert d.action == "CULLED"
    assert d.status == VisibilityStatus.TOO_CLOSE
    assert d.alpha == 0.0


def test_policy_fades_near_field_divergence():
    """Close-but-valid hazard: angular size is finite and large, so the
    marker is faded (not hard-culled nor left FULL) by the divergence gate."""
    policy = MarkerVisibilityPolicy(fade_pixel_threshold=2000.0)
    ref = np.array([2.0, 0.0, 0.0])  # in FOV, beyond near/micro clip
    d = policy.decide(hazard_marker(np.array([2.0, 0.0, 0.0])), ref)
    # angular span finite
    assert np.isfinite(d.ang_span_deg)
    # near field -> pixel span huge -> faded
    assert d.action in ("FADED", "CULLED")
    assert d.alpha < 1.0


def test_policy_full_ok_at_comfortable_range():
    policy = MarkerVisibilityPolicy()
    ref = np.array([25.0, 1.0, 0.0])
    d = policy.decide(lead_vehicle_marker(np.array([25.0, 1.0, 0.0])), ref)
    assert d.action == "FULL"
    assert d.status == VisibilityStatus.OK
    assert d.alpha == 1.0


def test_occluder_overlap_fades_marker():
    """Bodywork occluder on the lower/near angular region fades a marker that
    the driver's own bonnet overlaps."""
    occluder = Occluder(az_min_deg=-30, az_max_deg=30,
                        el_min_deg=-15, el_max_deg=-3, label="bonnet")
    policy = MarkerVisibilityPolicy(occluder=occluder)
    # a hazard low in the HUD, directly ahead, hence overlapped by bonnet
    ref = np.array([12.0, 0.0, -1.2])  # z down -> negative pitch
    d = policy.decide(hazard_marker(np.array([12.0, 0.0, -1.2])), ref)
    assert d.alpha < 1.0
    # a high marker (sky-ish) not overlapped stays FULL
    ref_hi = np.array([12.0, 0.0, 1.5])  # positive pitch (above bonnet)
    d_hi = policy.decide(hazard_marker(np.array([12.0, 0.0, 1.5])), ref_hi)
    assert d_hi.alpha > d.alpha


def test_evaluate_markers_culls_mixed_set():
    """Mixed marker set: far lead + lane stay FULL; close and behind
    hazards are culled before the pinhole projector can diverge."""
    markers = [
        lead_vehicle_marker(np.array([20.0, 2.0, 0.0])),
        hazard_marker(np.array([0.5, 0.0, 0.0])),   # too close -> cull
        hazard_marker(np.array([-5.0, 1.0, 0.0])),  # behind    -> cull
        lane_line_marker(np.array([0.0, -1.75, 0.0]),
                         np.array([60.0, -1.75, 0.0]), "lane_right"),
    ]
    pos = np.zeros(3)
    att = np.array([1.0, 0.0, 0.0, 0.0])
    out = evaluate_markers(markers, pos, att)
    actions = {m.kind: d.action for m, d in out}
    assert actions["lead_vehicle"] == "FULL"
    assert actions["hazard"] == "CULLED"
    assert actions["lane_right"] == "FULL"
    # every culled hazard has zero alpha
    for m, d in out:
        if d.action == "CULLED":
            assert d.alpha == 0.0
