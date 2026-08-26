"""Hazard assessment & warning arbitration demo (v0.11.0).

Builds a realistic AR-HUD scene -- a fast-closing FCW hazard, a distant ACC
lead vehicle, and lane markers -- and shows, per marker, the threat score and
the arbitrated headline warning.  Also demonstrates the *latency* effect: it
renders the same scene with the naive fused pose (which lags) vs the
display-time compensated pose, and reports whether uncompensated latency
flips the warning level (the safety-critical consequence of the v0.5 / v0.9
delay-compensation work).

Run:
    python3 examples/hazard_demo.py
"""

import numpy as np

from sensor_sim.adas import (
    MarkerProjector,
    lead_vehicle_marker,
    hazard_marker,
    lane_line_marker,
)
from sensor_sim.visibility import MarkerVisibilityPolicy, evaluate_markers
from sensor_sim.hazard import (
    HazardAssessment,
    WarningArbitrator,
    ThreatPipeline,
)

ATT = np.array([1.0, 0.0, 0.0, 0.0])


def main():
    print("=" * 68)
    print("sensor-sim v0.11.0 — Hazard assessment & warning arbitration")
    print("=" * 68)

    # --- Scene: ego at origin, looking +x ---
    # FCW/AEB hazard: a pedestrian/stopped object 18 m ahead, closing hard.
    hazard = hazard_marker(np.array([18.0, 0.0, 0.0]), half_angle_deg=25.0,
                           max_range=40.0, label="FCW")
    # ACC lead vehicle: 45 m ahead, moderate closing.
    lead = lead_vehicle_marker(np.array([45.0, 0.0, 0.0]))
    # Left / right lane boundaries (advisory only).
    lane_l = lane_line_marker(np.array([0.0, -3.5, 0.0]), np.array([120.0, -3.5, 0.0]), "lane_left")
    lane_r = lane_line_marker(np.array([0.0,  3.5, 0.0]), np.array([120.0,  3.5, 0.0]), "lane_right")

    markers = [hazard, lead, lane_l, lane_r]

    # Closing rates (m/s): negative = gap shrinking. Keyed by id() so the
    # arbitrator can look them up per marker.
    closing = {
        id(hazard): -18.0,
        id(lead):   -6.0,
        id(lane_l):  0.0,
        id(lane_r):  0.0,
    }

    # --- Visibility gate (v0.10) ---
    # Ego at origin (pos 0), hazard 18 m ahead, we want it visible.
    ego_pos = np.zeros(3)
    policy = MarkerVisibilityPolicy()
    vis = {id(m): d for m, d in evaluate_markers(markers, ego_pos, ATT, policy)}

    print("\nPer-marker visibility decision (v0.10):")
    for m in markers:
        d = vis[id(m)]
        print(f"  {m.kind:<12} action={d.action:<6} alpha={d.alpha:.2f}  {d.reason}")

    # --- Assessment & arbitration (truth pose) ---
    print("\nThreat assessment at TRUE ego pose:")
    ha = HazardAssessment()
    proj = MarkerProjector()
    for m in markers:
        a = ha.assess(m, proj.project(m, ego_pos, ATT)["range"],
                      closing[id(m)], proj.project(m, ego_pos, ATT)["azimuth"],
                      visible=vis[id(m)].action != "CULLED")
        print(f"  {m.kind:<12} score={a['score']:.3f} "
              f"ttc={a['ttc'] if a['ttc'] is None else round(a['ttc'],2)} "
              f"level={a['level'].label:<9}")

    arb = WarningArbitrator()
    res = arb.arbitrate(markers, proj, ego_pos, ATT, closing, vis)
    print(f"\nHeadline warning: "
          f"{res['winner'].kind if res['winner'] else None} -> "
          f"{res['level'].label}")

    # --- Latency flip demonstration (ThreatPipeline) ---
    # Ego coasting at 25 m/s; camera+pipeline delay ~0.2 s. By the time the
    # HUD displays, the TRUE ego has advanced 5 m to x=5.0. The *naive* fused
    # pose still reports the old x=0 (lags), so the hazard at x=12 looks 5 m
    # farther away -> read as CRITICAL instead of EMERGENCY. The compensated
    # pose (~truth) sees the real 7 m gap -> EMERGENCY.
    print("\nLatency comparison (25 m/s, 0.2 s delay; hazard closing slowly):")
    pipeline = ThreatPipeline()
    V, D = 25.0, 0.2
    true_pos = np.array([V * D, 0.0, 0.0])            # x=5.0 at display time
    naive_pos = np.array([0.0, 0.0, 0.0])             # lags behind truth
    comp_pos = np.array([V * D - 0.03, 0.0, 0.0])     # ~truth + small residual
    # use a slow-closing hazard (closing -2 m/s) at 12 m ahead, so the
    # latency-induced 5 m gap error sits right at the CRITICAL/EMERGENCY seam
    slow = hazard_marker(np.array([12.0, 0.0, 0.0]), label="FCW(slow)")
    slow_closing = {id(slow): -2.0}
    slow_vis = {id(slow): policy.decide(slow, MarkerProjector().to_body(slow.ref_world, true_pos, ATT))}
    time_to_impact = pipeline.decide(
        [slow],
        (true_pos, ATT),
        (naive_pos, ATT),
        (comp_pos, ATT),
        slow_closing,
        slow_vis,
    )
    print(f"  naive       : {time_to_impact['naive']['level'].label}")
    print(f"  compensated : {time_to_impact['compensated']['level'].label}")
    print(f"  level_delta : {time_to_impact['level_delta']:+d} "
          f"(level_changed={time_to_impact['level_changed']})")

    print("\nProduction lessons from this module:")
    for i, lesson in enumerate(__import__("sensor_sim.hazard", fromlist=["__PRODUCTION_LESSONS__"]).__PRODUCTION_LESSONS__, 1):
        print(f"  {i}. {lesson}")


if __name__ == "__main__":
    main()
