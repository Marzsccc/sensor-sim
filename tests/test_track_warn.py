"""Tests for v0.20.0 fused-track -> ADAS-marker bridging and arbitration.

Covers:

- ``TrackToMarker.closing_rate``: kinematic closing from the track velocity
  (positive closing toward the ego origin; crossing -> ~0; sign convention).
- ``TrackToMarker.map``: confirmation gating, range cap, marker fields.
- ``FusionThreatPipeline.decide``: closing rates flow through, arbitration
  picks the truly threatening target, tentative tracks cannot claim the
  headline warning, receding targets are de-prioritized.
"""

from __future__ import annotations

import numpy as np
import pytest

from sensor_sim.adas import AdasMarker, MarkerProjector
from sensor_sim.hazard import WarningLevel
from sensor_sim.track_warn import (
    FusionThreatPipeline,
    TrackToMarker,
    TrackToMarkerConfig,
)
from sensor_sim.tracking import Track


def _track(x: float, y: float, vx: float, vy: float, *, confirmed: bool = True,
           track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        x=np.array([x, y, vx, vy], dtype=float),
        P=np.eye(4),
        born_t=0.0,
        last_t=0.1,
        age=4,
        missed=0,
        hits=4,
        confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# closing_rate
# ---------------------------------------------------------------------------

class TestClosingRate:
    def test_direct_approach_positive_closing(self):
        m = TrackToMarker()
        tr = _track(x=40.0, y=0.0, vx=-15.0, vy=0.0)  # moving straight at ego
        assert m.closing_rate(tr) == pytest.approx(15.0)

    def test_receding_negative_closing(self):
        m = TrackToMarker()
        tr = _track(x=40.0, y=0.0, vx=12.0, vy=0.0)  # moving away
        assert m.closing_rate(tr) == pytest.approx(-12.0)

    def test_crossing_near_zero(self):
        # Pure lateral motion: the v0.18/0.19 blind-lateral lesson.
        m = TrackToMarker()
        tr = _track(x=35.0, y=0.0, vx=0.0, vy=2.2)
        assert m.closing_rate(tr) == pytest.approx(0.0, abs=1e-6)

    def test_off_bearing_projects(self):
        # Target at 45 deg with velocity along +x: physical closing is
        # -(v . r_hat); vx only partially closes the gap.
        m = TrackToMarker()
        tr = _track(x=10.0, y=10.0, vx=-5.0, vy=0.0)
        expected = 5.0 / np.sqrt(2.0)
        assert m.closing_rate(tr) == pytest.approx(expected, rel=1e-5)

    def test_host_motion_subtracted(self):
        m = TrackToMarker()
        # Ego moves +x at 10 m/s, target static in the world frame:
        # relative velocity is -10 m/s along +x -> closing = +10 m/s.
        tr = _track(x=50.0, y=0.0, vx=0.0, vy=0.0)
        assert m.closing_rate(tr, vel_w=np.array([10.0, 0.0, 0.0])) == pytest.approx(10.0)

    def test_small_closing_snapped_to_zero(self):
        cfg = TrackToMarkerConfig(min_closing_rate_mps=0.05)
        m = TrackToMarker(cfg)
        tr = _track(x=100.0, y=0.5, vx=-1e-3, vy=0.0)  # ~1e-5 closing
        assert m.closing_rate(tr) == 0.0

    def test_at_origin_zero(self):
        m = TrackToMarker()
        tr = _track(x=0.0, y=0.0, vx=1.0, vy=0.0)
        assert m.closing_rate(tr) == 0.0


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------

class TestMap:
    def test_maps_confirmed_track(self):
        m = TrackToMarker()
        tr = _track(x=30.0, y=4.0, vx=-8.0, vy=1.0)
        marker = m.map(tr)
        assert marker is not None
        assert marker.kind == "fused_track"
        assert marker.label == "track#1"
        assert marker.ref_world[:2] == pytest.approx([30.0, 4.0])
        assert marker.ref_world[2] == 0.0
        # Hazard-layer sign: negative = closing (matches TTCModel).
        expected_closing = -(
            8.0 * 30.0 / np.hypot(30.0, 4.0) - 1.0 * 4.0 / np.hypot(30.0, 4.0)
        )
        assert marker.closing_rate_mps == pytest.approx(expected_closing)

    def test_tentative_filtered(self):
        m = TrackToMarker()
        tr = _track(x=30.0, y=0.0, vx=-8.0, vy=0.0, confirmed=False)
        assert m.map(tr) is None

    def test_tentative_allowed_when_configured(self):
        m = TrackToMarker(TrackToMarkerConfig(require_confirmed=False))
        tr = _track(x=30.0, y=0.0, vx=-8.0, vy=0.0, confirmed=False)
        assert m.map(tr) is not None

    def test_range_cap(self):
        m = TrackToMarker(TrackToMarkerConfig(max_marker_range_m=100.0))
        far = _track(x=250.0, y=0.0, vx=-1.0, vy=0.0)
        near = _track(x=50.0, y=0.0, vx=-1.0, vy=0.0, track_id=2)
        assert m.map(far) is None
        assert m.map(near) is not None

    def test_no_range_cap(self):
        m = TrackToMarker(TrackToMarkerConfig(max_marker_range_m=None))
        far = _track(x=1000.0, y=0.0, vx=-1.0, vy=0.0)
        assert m.map(far) is not None


# ---------------------------------------------------------------------------
# FusionThreatPipeline.decide
# ---------------------------------------------------------------------------

def _ego_pose():
    pos = np.zeros(3)
    att = np.array([1.0, 0.0, 0.0, 0.0])  # identity: world == body
    return pos, att


class TestPipeline:
    def test_closing_rate_of_winner(self):
        pipe = FusionThreatPipeline()
        pos, att = _ego_pose()
        # Approaching target at 40 m, closing at 15 m/s.
        tr_app = _track(x=40.0, y=0.0, vx=-15.0, vy=0.0, track_id=1)
        res = pipe.decide([tr_app], pos, att)
        assert res["level"] != WarningLevel.OFF
        assert res["winner"] is not None
        assert res["winner"].kind == "fused_track"
        # Hazard-layer sign (negative = closing) flows to the arbitrator.
        assert res["closing"][id(res["winner"])] == pytest.approx(-15.0)

    def test_tentative_does_not_claim(self):
        pipe = FusionThreatPipeline()
        pos, att = _ego_pose()
        # Tentative target very close and fast -- must NOT claim the warning.
        tr_tent = _track(x=10.0, y=0.0, vx=-20.0, vy=0.0, confirmed=False)
        res = pipe.decide([tr_tent], pos, att)
        assert res["level"] == WarningLevel.OFF
        assert res["winner"] is None

    def test_far_target_off(self):
        pipe = FusionThreatPipeline()
        pos, att = _ego_pose()
        tr_far = _track(x=500.0, y=0.0, vx=-30.0, vy=0.0)
        res = pipe.decide([tr_far], pos, att)
        assert res["level"] == WarningLevel.OFF

    def test_receding_never_wins_over_approaching(self):
        pipe = FusionThreatPipeline()
        pos, att = _ego_pose()
        tr_app = _track(x=40.0, y=0.0, vx=-15.0, vy=0.0, track_id=1)
        tr_rec = _track(x=20.0, y=0.0, vx=10.0, vy=0.0, track_id=2)
        res = pipe.decide([tr_app, tr_rec], pos, att)
        assert res["winner"] is not None
        assert res["winner"].label == "track#1"

    def test_merges_extra_markers(self):
        pipe = FusionThreatPipeline()
        pos, att = _ego_pose()
        tr = _track(x=40.0, y=0.0, vx=-15.0, vy=0.0, track_id=1)
        lane = AdasMarker(kind="lane_center", ref_world=np.array([10.0, 0.0, 0.0]))
        res = pipe.decide([tr], pos, att, extra_markers=[lane])
        assert any(m.kind == "lane_center" for m in res["markers"])
        assert id(lane) in res["closing"]

    def test_empty_tracks_off(self):
        pipe = FusionThreatPipeline()
        pos, att = _ego_pose()
        res = pipe.decide([], pos, att)
        assert res["level"] == WarningLevel.OFF
        assert len(res["markers"]) == 0

    def test_e2e_with_radar_tracker(self):
        """Full chain: RadarTracker -> marker -> arbitration."""
        from sensor_sim.radar import RadarConfig, RadarSensor
        from sensor_sim.tracking import RadarTracker, TrackConfig
        from sensor_sim.lidar import Box, Sphere

        rs = RadarSensor(
            RadarConfig(range_noise_sigma_m=0.05, angle_noise_sigma_deg=0.1),
            mount_t_body=np.array([0.5, 0.0, 0.0]),
        )
        trk = RadarTracker(
            TrackConfig(confirm_min=2, coast_max=4),
            mount_t_body=np.array([0.5, 0.0, 0.0]),
        )
        WORLD = [
            Box(center=np.array([40.0, 0.0, 0.0]), half_extents=np.array([2.0, 1.0, 1.0]),
                velocity=np.array([-15.0, 0.0, 0.0]), object_id=1),
            Sphere(center=np.array([35.0, -6.0, 0.0]), radius=0.4,
                   velocity=np.array([0.0, 2.2, 0.0]), object_id=2),
        ]
        pos = np.zeros(3)
        att = np.array([1.0, 0.0, 0.0, 0.0])
        pipe = FusionThreatPipeline()
        rng = np.random.default_rng(9)

        point = type("Pose", (), {"pos": pos, "att": att})()
        levels = []
        for i in range(30):
            t = i * 0.1
            fr = rs.scan(point, WORLD, t=t, rng=rng)
            tf = trk.process(fr, pos, att, np.zeros(3))
            res = pipe.decide(list(tf.tracks), pos, att)
            levels.append(res["level"])
        # After convergence the approaching lead vehicle must arbitrate
        # above OFF and above the crossing pedestrian.
        assert max(lv.value for lv in levels) >= WarningLevel.WARN.value
        # The final arbitration should at least not be OFF with a closing
        # lead present.
        assert levels[-1] >= WarningLevel.CAUTION