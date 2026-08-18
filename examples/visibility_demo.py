"""Example: near-field marker visibility, FOV culling & occlusion (v0.10.0).

Demonstrates why the v0.9.0 pinhole projector had to be guarded: a hazard
that closes on the bumper produces a *divergent* pixel span, but a finite,
physiologically meaningful angular span. This example runs the visibility
policy on a scene as the ego passes a roadside hazard and shows how the HUD
culls / fades each marker instead of rendering garbage pixels.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sensor_sim.adas import (
    MarkerProjector,
    lead_vehicle_marker,
    hazard_marker,
    lane_line_marker,
)
from sensor_sim.visibility import (
    Frustum,
    Occluder,
    MarkerVisibilityPolicy,
    evaluate_markers,
)


def main():
    # ego at origin, heading +x (identity attitude)
    pos = np.zeros(3)
    att = np.array([1.0, 0.0, 0.0, 0.0])

    markers = [
        lead_vehicle_marker(np.array([25.0, 3.0, 0.0])),     # ACC target, ahead right
        hazard_marker(np.array([1.2, 0.0, -0.35])),           # FCW hazard, very close ahead
        lane_line_marker(np.array([0.0, -1.75, 0.0]),
                         np.array([80.0, -1.75, 0.0]), "lane_right"),
        lane_line_marker(np.array([0.0, 1.75, 0.0]),
                         np.array([80.0, 1.75, 0.0]), "lane_left"),
    ]

    # HUD frustum + bodywork occluder (bonnet occupies the lower near field)
    frustum = Frustum(hfov_deg=45.0, vfov_deg=20.0,
                      near_m=0.5, min_range_m=1.0, max_range_m=200.0)
    bonnet = Occluder(az_min_deg=-35, az_max_deg=35,
                      el_min_deg=-18, el_max_deg=-4, label="bonnet")
    policy = MarkerVisibilityPolicy(frustum=frustum, occluder=bonnet,
                                    half_extent_m=1.8, max_ang_deg=45.0)

    print(f"frustum : hfov={frustum.hfov_deg:.0f}° vfov={frustum.vfov_deg:.0f}° "
          f"near={frustum.near_m:.1f}m min={frustum.min_range_m:.1f}m max={frustum.max_range_m:.0f}m")
    print(f"occluder: {bonnet.label} [{bonnet.az_min_deg}°..{bonnet.az_max_deg}°] "
          f"x [{bonnet.el_min_deg}°..{bonnet.el_max_deg}°]\n")
    print(f"{'marker':<12}{'range':>7}{'angle':>8}{'pixels':>9} {'action':<7}{'alpha':>6}  reason")

    proj = MarkerProjector()
    for m, d in evaluate_markers(markers, pos, att, policy):
        ref = proj.to_body(m.ref_world, pos, att)
        rng = np.hypot(ref[0], ref[1])
        px = d.pixel_span_px if np.isfinite(d.pixel_span_px) else float("inf")
        px_s = f"{px:9.1f}" if np.isfinite(px) else "      inf"
        print(f"{m.kind:<12}{rng:7.1f}{d.ang_span_deg:8.1f}{px_s:>9} "
              f"{d.action:<7}{d.alpha:6.2f}  {d.reason}")

    print("\nKey takeaway: the close FCW hazard is faded/culled by the angular-"
          "oversize and bonnet-occlusion gates instead of producing divergent "
          "(inf px) pinhole output as in the unguarded v0.9.0 path.")
    # the ACC target & lanes are far and stay FULL


if __name__ == "__main__":
    main()
