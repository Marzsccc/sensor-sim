"""Tests for hazard assessment & warning arbitration (v0.11.0)."""

import numpy as np
import pytest

from sensor_sim.adas import (
    MarkerProjector,
    lead_vehicle_marker,
    hazard_marker,
    lane_line_marker,
)
from sensor_sim.visibility import VisibilityDecision
from sensor_sim.hazard import (
    WarningLevel,
    TTCModel,
    HazardAssessment,
    WarningArbitrator,
    ThreatPipeline,
    level_from_score,
)


def _decision(action="FULL"):
    return VisibilityDecision(action=action, status="OK", alpha=1.0,
                              reason="ok", ang_span_deg=5.0, pixel_span_px=50.0)


# ---------------------------------------------------------------------------
# WarningLevel
# ---------------------------------------------------------------------------

def test_warning_level_ordering():
    assert WarningLevel.OFF < WarningLevel.CAUTION < WarningLevel.WARN
    assert WarningLevel.WARN < WarningLevel.CRITICAL < WarningLevel.EMERGENCY
    assert WarningLevel.EMERGENCY.label == "EMERGENCY"


# ---------------------------------------------------------------------------
# TTC
# ---------------------------------------------------------------------------

def test_ttc_const_speed_basic():
    m = TTCModel()
    # 50 m gap closing at 20 m/s -> 2.5 s
    assert m.ttc_const_speed(50.0, -20.0) == pytest.approx(2.5, rel=1e-6)
    # not closing -> None
    assert m.ttc_const_speed(50.0, 0.0) is None
    assert m.ttc_const_speed(50.0, 5.0) is None
    # out of range -> None
    assert m.ttc_const_speed(500.0, -20.0) is None
    # non-positive range -> None
    assert m.ttc_const_speed(-5.0, -20.0) is None


def test_ttc_const_decel_harder_than_speed():
    m = TTCModel()
    # sustaining a deceleration closes the gap faster; check decel TTC < speed TTC
    speed_ttc = m.ttc_const_speed(40.0, -15.0)
    decel_ttc = m.ttc_const_decel(40.0, -15.0, a_rel=-5.0)
    assert decel_ttc is not None
    assert decel_ttc < speed_ttc
    assert decel_ttc > 0.0


def test_ttc_const_decel_degenerate_or_gated_returns_none():
    m = TTCModel()
    # out-of-gate range -> None
    assert m.ttc_const_decel(500.0, -15.0) is None
    # non-positive / behind range -> None
    assert m.ttc_const_decel(0.0, -15.0) is None
    assert m.ttc_const_decel(-5.0, -15.0) is None
    # a_rel >= 0 falls back to constant speed
    assert m.ttc_const_decel(10.0, -0.1, a_rel=0.0) == m.ttc_const_speed(10.0, -0.1)


# ---------------------------------------------------------------------------
# level_from_score
# ---------------------------------------------------------------------------

def test_level_from_score_ladder():
    assert level_from_score(0.00) == WarningLevel.OFF
    assert level_from_score(0.20) == WarningLevel.CAUTION
    assert level_from_score(0.40) == WarningLevel.WARN
    assert level_from_score(0.70) == WarningLevel.CRITICAL
    assert level_from_score(0.95) == WarningLevel.EMERGENCY


# ---------------------------------------------------------------------------
# HazardAssessment
# ---------------------------------------------------------------------------

def test_assess_ttc_and_alignment_drive_score():
    ha = HazardAssessment()
    mark = lead_vehicle_marker(np.array([30.0, 2.0, 0.0]))
    # closing fast, dead ahead -> high score
    a_ahead = ha.assess(mark, rng=30.0, closing_rate=-15.0, azimuth_rad=0.0)
    # same marker, far off-axis -> alignment kills the score
    a_off = ha.assess(mark, rng=30.0, closing_rate=-15.0, azimuth_rad=np.deg2rad(30.0))
    assert a_ahead["score"] > a_off["score"]
    assert a_ahead["alignment"] == pytest.approx(1.0)
    assert a_off["alignment"] == pytest.approx(0.0)


def test_assess_invisible_never_wins():
    ha = HazardAssessment()
    mark = hazard_marker(np.array([15.0, 0.0, 0.0]))
    vis = ha.assess(mark, rng=15.0, closing_rate=-25.0, azimuth_rad=0.0, visible=True)
    invis = ha.assess(mark, rng=15.0, closing_rate=-25.0, azimuth_rad=0.0, visible=False)
    assert vis["score"] > 0.0
    assert invis["score"] == 0.0
    assert invis["level"] == WarningLevel.OFF


def test_far_ttc_is_no_threat():
    ha = HazardAssessment()
    mark = lane_line_marker(np.array([0.0, -3.5, 0.0]), np.array([150.0, -3.5, 0.0]), "lane_left")
    # 150 m away with tiny closing rate -> essentially no threat
    a = ha.assess(mark, rng=150.0, closing_rate=-1.0, azimuth_rad=0.0)
    assert a["score"] < 0.2


# ---------------------------------------------------------------------------
# WarningArbitrator
# ---------------------------------------------------------------------------

def test_arbitrator_picks_most_threatening():
    proj = MarkerProjector()
    arb = WarningArbitrator()
    ego_pos = np.zeros(3)
    ego_att = np.array([1.0, 0.0, 0.0, 0.0])
    hazard = hazard_marker(np.array([12.0, 0.0, 0.0]))
    lead = lead_vehicle_marker(np.array([40.0, 0.0, 0.0]))
    lane = lane_line_marker(np.array([0.0, -3.5, 0.0]), np.array([100.0, -3.5, 0.0]), "lane_left")
    markers = [hazard, lead, lane]
    cr = {id(hazard): -22.0, id(lead): -10.0, id(lane): 0.0}
    vis = {id(hazard): _decision("FULL"), id(lead): _decision("FULL"), id(lane): _decision("FULL")}
    res = arb.arbitrate(markers, proj, ego_pos, ego_att, cr, vis)
    assert res["winner"] is hazard
    assert res["level"].value >= WarningLevel.WARN.value


def test_arbitrator_culled_marker_cannot_win():
    proj = MarkerProjector()
    arb = WarningArbitrator()
    ego_pos = np.zeros(3)
    ego_att = np.array([1.0, 0.0, 0.0, 0.0])
    hazard = hazard_marker(np.array([12.0, 0.0, 0.0]))
    lead = lead_vehicle_marker(np.array([40.0, 0.0, 0.0]))
    markers = [hazard, lead]
    cr = {id(hazard): -22.0, id(lead): -10.0}
    # hazard is CULLED -> cannot win, lead wins instead
    vis = {id(hazard): _decision("CULLED"), id(lead): _decision("FULL")}
    res = arb.arbitrate(markers, proj, ego_pos, ego_att, cr, vis)
    assert res["winner"] is lead


# ---------------------------------------------------------------------------
# ThreatPipeline (latency-aware decision change)
# ---------------------------------------------------------------------------

def _straight_poses(ego_true_x, speed, delay_s):
    """Return (true_display, fused_naive, predicted) poses as (pos, att).
    The hazard sits AHEAD of the ego; the naive fused pose lags behind the
    truth, so latency makes the hazard *appear further away* (less urgent).
    """
    att = np.array([1.0, 0.0, 0.0, 0.0])
    true_pos = np.array([ego_true_x, 0.0, 0.0])
    # naive fused pose lags by speed*delay (further behind the hazard)
    naive_pos = np.array([ego_true_x - speed * delay_s, 0.0, 0.0])
    # compensated pose ~ truth (small residual)
    comp_pos = np.array([ego_true_x - 0.05, 0.0, 0.0])
    return (true_pos, att), (naive_pos, att), (comp_pos, att)


def test_threat_pipeline_latency_can_flip_warning():
    pipeline = ThreatPipeline()
    # hazard 5 m AHEAD of the true ego position
    hazard = hazard_marker(np.array([30.0, 0.0, 0.0]))
    markers = [hazard]
    cr = {id(hazard): -22.0}   # closing fast
    vis = {id(hazard): _decision("FULL")}
    display, naive, comp = _straight_poses(25.0, speed=20.0, delay_s=0.25)
    res = pipeline.decide(markers, display, naive, comp, cr, vis)
    # Naive pose sits 5 m behind truth -> hazard appears 10 m away (less urgent).
    # Compensated pose ~truth -> hazard ~5 m away (more urgent).
    assert res["compensated"]["level"].value >= res["naive"]["level"].value
    assert res["level_delta"] >= 0
