"""
Near-field marker visibility, FOV culling & occlusion (v0.10.0).

v0.9.0 surfaced a production trap in `adas.py`: the pinhole projection

    u = cx + f_px_x * y / x ,   v = cy + f_px_y * z / x

diverges as a marker's forward range *x* -> 0. A hazard or lead vehicle
that comes very close blows up to enormous, meaningless pixel coordinates
(the HUD would flicker, the box would fly off-screen, and the uncertainty
fade no longer protects the renderer because the *geometry itself* becomes
ill-conditioned). This module isolates the near-field problem so the render
loop can make an explicit decision per marker:

  1. `marker_fov_visibility(ref_body, ...)` -- classify a body-frame
     reference point against the HUD field-of-view frustum, a near clip
     and a max range. Returns flags (`in_fov`, `in_range`, `front`) plus a
     `status` enum: OK / NEAR_PLANE / TOO_CLOSE / OUT_OF_FOV / TOO_FAR /
     BEHIND.

  2. `angular_size(...)` -- express the marker's on-screen size in *angle*
     (deg) instead of raw pixels. Angle is the stable quantity close-in:
     as x shrinks the pixel size explodes but the angular size converges to
     a finite value, which is what the human eye actually resolves. This is
     the same reason optical designers quote angular (not pixel) resolution.

  3. `occlusion_fraction(marker, occluder, ...)` -- how much of a marker's
     projected extent is blocked by a near occluder (vehicle bodywork /
     A-pillar / bonnet silhouette). Uses a conservative overlap test on the
     marker's angular span against the occluder's angular span in the HUD.

  4. `MarkerVisibilityPolicy.decide(...)` -- combines all of the above into
     a per-marker *render decision*: FULL / FADED / CULLED, plus the fade
     alpha and a machine-readable reason. The HUD render loop calls this
     once per marker per display tick instead of naively projecting.

References: Azuma (1997) registration error taxonomy ("occlusion" depth
error); for AR-HUD the classic failure mode is drawing a marker that is
behind bodywork or so close the projection is numerically meaningless --
this is the AR analogue of a 3D engine's near-clip plane.

The key production lesson encoded here: **cull by a near clip and by FOV
in the *body* (look) frame before projecting to pixels**, and drive the
fade close-in by *angular* size rather than pixel size, because pixel size
diverges exactly where the driver needs the marker most.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .adas import AdasMarker, MarkerProjector


# ---------------------------------------------------------------------------
# FOV / near-field classification
# ---------------------------------------------------------------------------

class VisibilityStatus:
    """Enum-like status of a marker against the HUD frustum."""
    OK = "ok"              # fully valid
    NEAR_PLANE = "near"    # inside FOV but closer than the near clip
    TOO_CLOSE = "too_close"  # closer than the hard min range (divergent)
    OUT_OF_FOV = "out_of_fov"
    TOO_FAR = "too_far"
    BEHIND = "behind"


@dataclass
class Frustum:
    """HUD field-of-view frustum in the body (x-forward) frame.

    Parameters
    ----------
    hfov_deg : float
        Horizontal half-angle of the HUD (deg). Total width = 2*hfov.
    vfov_deg : float
        Vertical half-angle (deg).
    near_m : float
        Near clip distance along +x (m). Markers closer than this in
        *range* touch the near plane; below `min_range_m` they are treated
        as unusable.
    min_range_m : float
        Hard minimum forward range (m). Below this the pinhole projection
        diverges and the marker is culled.
    max_range_m : float
        Max range (m); beyond it the marker is too faint to draw.
    """
    hfov_deg: float = 45.0
    vfov_deg: float = 20.0
    near_m: float = 0.5
    min_range_m: float = 1.0
    max_range_m: float = 200.0

    def decide(self, ref_body: np.ndarray) -> Tuple[str, bool]:
        """Classify a body-frame point against the frustum.

        Returns (status, drawable):
          drawable is True for statuses we can still render (OK and
          NEAR_PLANE -- near-plane is a *fade*, not a hard cull).
        """
        x, y, z = float(ref_body[0]), float(ref_body[1]), float(ref_body[2])
        if x <= 0.0:
            return VisibilityStatus.BEHIND, False
        rng = float(np.hypot(x, y))
        if rng < self.min_range_m:
            return VisibilityStatus.TOO_CLOSE, False
        # horizontal / vertical angular offsets from the look axis (+x)
        az = float(np.arctan2(y, x))
        el = float(np.arctan2(z, x))
        h_ok = abs(az) <= np.radians(self.hfov_deg)
        v_ok = abs(el) <= np.radians(self.vfov_deg)
        if not (h_ok and v_ok):
            return VisibilityStatus.OUT_OF_FOV, False
        if rng > self.max_range_m:
            return VisibilityStatus.TOO_FAR, False
        if rng < self.near_m:
            # inside FOV & range but touching the near clip -> fade, not cull
            return VisibilityStatus.NEAR_PLANE, True
        return VisibilityStatus.OK, True


# ---------------------------------------------------------------------------
# Angular size -- the divergence-proof on-screen size metric
# ---------------------------------------------------------------------------

def angular_size(
    ref_body: np.ndarray,
    extent_m: float = 1.8,
    hfov_px: float = 1000.0,
) -> Tuple[float, float, bool]:
    """Angular on-screen size of a marker of half-width `extent_m`.

    For a frustum-independent notion of "how big does this marker look", we
    fit the marker's *physical* half-extent to the HUD angular scale.

    The HUD shows an object of half-size `extent_m` at range `r` subtending

        half_angle = atan(extent_m / r)

    We report the full angular span = 2 * half_angle (deg).

    `hfov_px` only seeds the *pixel* analogue so the two can be compared
    (the pixel span comes out of the same small-angle model used by the
    pinhole projector, but the angular quantity does NOT diverge as r->0).

    Returns (ang_span_deg, pixel_span_px, valid).
      - ang_span_deg is finite for all r > 0.
      - pixel_span_px = 2 * hfov_px * tan(half_angle); this DOES diverge as
        r -> 0, which is exactly the trap the HUD must avoid.
    """
    x, y = float(ref_body[0]), float(ref_body[1])
    if x <= 1e-6:
        return float("inf"), float("inf"), False
    r = float(np.hypot(x, y))
    half_angle = np.arctan2(extent_m, r)
    ang_span = 2.0 * float(np.degrees(half_angle))
    pixel_span = 2.0 * hfov_px * float(np.tan(half_angle))
    return ang_span, pixel_span, True


def _ang_span_of(marker_ref: np.ndarray, half_extent: float) -> float:
    """Angular half-... full span of a marker at the given body point."""
    ang, _px, valid = angular_size(marker_ref, half_extent)
    return ang if valid else float("inf")


# ---------------------------------------------------------------------------
# Occlusion by a near occluder (bonnet / bodywork silhouette)
# ---------------------------------------------------------------------------

@dataclass
class Occluder:
    """A near-body occluder in the HUD look frame (angular occupancy).

    The occluder is described by the angular span it occupies on the HUD,
    e.g. a bonnet wedge from `az_min_deg`..`az_max_deg` and
    `el_min_deg`..`el_max_deg` (the driver's own vehicle body / A-pillars
    sitting in front of the HUD). A marker whose projected angular footprint
    overlaps this region is partially blocked.

    This is deliberately a 2D angular test in the look frame -- the cheap,
    conservative approximation used for HUD fade/cull against bodywork.
    """
    az_min_deg: float
    az_max_deg: float
    el_min_deg: float
    el_max_deg: float
    label: str = "bodywork"

    def contains(self, ref_body: np.ndarray) -> bool:
        x, y, z = (float(v) for v in ref_body)
        if x <= 1e-6:
            return False
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arctan2(z, x))
        return (self.az_min_deg <= az <= self.az_max_deg
                and self.el_min_deg <= el <= self.el_max_deg)

    def overlap_frac(self, ref_body: np.ndarray, half_extent: float,
                     hfov_deg: float) -> float:
        """Fraction (0-1) of the marker's angular span covered by occluder.

        Conservative 1-D overlap in azimuth * elevation, clamped to [0,1].
        """
        az = np.degrees(np.arctan2(float(ref_body[1]), float(ref_body[0])))
        el = np.degrees(np.arctan2(float(ref_body[2]), float(ref_body[0])))
        span = 0.5 * min(_ang_span_of(ref_body, half_extent), 2.0 * hfov_deg)
        if span <= 0.0:
            return 0.0
        # az overlap
        az_lo = az - span
        az_hi = az + span
        az_overlap = max(0.0, min(az_hi, self.az_max_deg) - max(az_lo, self.az_min_deg))
        el_lo = el - span
        el_hi = el + span
        el_overlap = max(0.0, min(el_hi, self.el_max_deg) - max(el_lo, self.el_min_deg))
        if az_overlap <= 0.0 or el_overlap <= 0.0:
            return 0.0
        # area ratio of the intersecting box vs the marker's own box
        inter = az_overlap * el_overlap
        own = (2 * span) * (2 * span)
        return float(np.clip(inter / own, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Policy: decide how to render each marker
# ---------------------------------------------------------------------------

@dataclass
class VisibilityDecision:
    action: str            # FULL / FADED / CULLED
    status: str
    alpha: float           # 0..1 render alpha
    reason: str
    ang_span_deg: float    # angular on-screen size (finite!)
    pixel_span_px: float   # would-be pixel size (may diverge / inf)


@dataclass
class MarkerVisibilityPolicy:
    """Decide, per marker per tick, whether / how to draw it.

    Combines frustum culling (near clip / FOV / range) with angular-size
    monitoring and bodywork occlusion, returning a render decision and a
    fade alpha. This keeps the near-field divergence trap out of the naive
    pinhole path in `adas.py`.

    Parameters
    ----------
    frustum : Frustum
        HUD frustum (default full forward HUD).
    occluder : Occluder, optional
        Bodywork/A-pillar angular occupancy; if provided, overlap fades the
        marker.
    half_extent_m : float
        Representative physical half-size of the marker used to size its
        angular footprint (used only for occlusion and angular monitoring;
        the true geometry stays in `marker.points_world`).
    fade_pixel_threshold : float
        When angular monitoring flags that the pixel span would exceed this
        (a proxy for "so close the projection is unstable"), treat the
        marker as FADED rather than FULL even if still in FOV.
    """
    frustum: Frustum = field(default_factory=Frustum)
    occluder: Optional[Occluder] = None
    half_extent_m: float = 1.8
    max_ang_deg: float = 45.0
    fade_pixel_threshold: float = 2500.0

    def decide(self, marker: AdasMarker, ref_body: np.ndarray) -> VisibilityDecision:
        # angular / pixel size (divergence-proof angular part)
        _extent = self.half_extent_m
        if marker.points_world is not None:
            # use marker's own size to size the footprint if available
            pts = marker.points_world
            diam = float(np.max(np.linalg.norm(pts - np.mean(pts, axis=0), axis=1))) or self.half_extent_m
            _extent = max(0.1 * diam, self.half_extent_m)
        ang, px, valid = angular_size(ref_body, _extent,
                                      hfov_px=float(np.tan(np.radians(self.frustum.hfov_deg))))
        status, drawable = self.frustum.decide(ref_body)

        # hard cull conditions (decode drawable/status)
        if status in (VisibilityStatus.BEHIND,
                      VisibilityStatus.TOO_CLOSE,
                      VisibilityStatus.OUT_OF_FOV,
                      VisibilityStatus.TOO_FAR):
            return VisibilityDecision("CULLED", status, 0.0, status, ang, px)

        alpha = 1.0
        reason = "ok"
        # near plane -> fade progressively as it closes on the near clip
        if status == VisibilityStatus.NEAR_PLANE:
            rng = float(np.hypot(ref_body[0], ref_body[1]))
            near = self.frustum.near_m
            minr = self.frustum.min_range_m
            span = max(near - minr, 1e-6)
            alpha = float(np.clip((rng - minr) / span, 0.0, 1.0))
            reason = "near-plane fade"

        # --- near-field angular oversize (the divergence-proof gate) ---
        # As the marker closes in, its angular footprint grows: a hazard that
        # fills a large fraction of the HUD is useless as a crisp registration
        # cue (and its *pixel* footprint diverges in the pinhole model that
        # v0.9.0 exposed). Fade it out as its full angular span exceeds
        # `max_ang_deg`. The angular span stays finite for all r>0, so this is
        # numerically safe -- unlike the raw pixel count.
        if np.isfinite(ang) and ang > self.max_ang_deg:
            a = float(np.clip(self.max_ang_deg / max(ang, 1e-9), 0.0, 1.0))
            alpha = min(alpha, a)
            reason = "near-field angular oversize -> fade"

        # secondary, belt-and-braces: raw pixel divergence guard (catches the
        # tail of the near-field region where not even the angle gate fired).
        if np.isfinite(px) and px > self.fade_pixel_threshold:
            alpha = min(alpha, float(np.clip(self.fade_pixel_threshold / max(px, 1e-9), 0.0, 1.0)))
            reason = "near-field pixel divergence -> fade"

        # bodywork occlusion
        if self.occluder is not None and alpha > 0.0:
            occ = self.occluder.overlap_frac(ref_body, _extent, self.frustum.hfov_deg)
            if occ > 0.0:
                alpha *= float(np.clip(1.0 - occ, 0.0, 1.0))
                reason += f" + occluded {occ:.0%}"

        action = "FULL" if alpha >= 0.99 else ("FADED" if alpha > 0.02 else "CULLED")
        if action == "CULLED" and status == VisibilityStatus.OK:
            reason = "faded to zero"
        return VisibilityDecision(action, status, float(np.clip(alpha, 0.0, 1.0)),
                                  reason, ang, px)


# ---------------------------------------------------------------------------
# Convenience: classify a whole marker set for one display tick
# ---------------------------------------------------------------------------

def evaluate_markers(
    markers: List[AdasMarker],
    pos: np.ndarray,
    att: np.ndarray,
    policy: Optional[MarkerVisibilityPolicy] = None,
) -> List[Tuple[AdasMarker, VisibilityDecision]]:
    """Project every marker to body frame and run the policy.

    Returns [(marker, decision), ...], with decisions' alphas ready to feed
    the render loop. Any marker BEHIND / TOO_CLOSE / OUT_OF_FOV is CULLED
    before it ever reaches the pinhole projector -- this is the principal
    near-field guard.
    """
    if policy is None:
        policy = MarkerVisibilityPolicy()
    out = []
    proj = MarkerProjector()
    for m in markers:
        ref_body = proj.to_body(m.ref_world, pos, att)
        out.append((m, policy.decide(m, ref_body)))
    return out
