"""Fused-track to ADAS-marker bridging and threat arbitration (v0.20.0).

The v0.18/0.19 tracker stack answers "where is that target and how fast is
it moving" -- but the AR-HUD consumes *markers* (``AdasMarker``) and the
v0.11 warning layer consumes markers with a *closing rate* fed in by the
caller.  This module closes that loop:

- ``TrackToMarker`` maps a fused ``Track`` (world-frame ``[px, py, vx, vy]``)
  into an ``AdasMarker`` whose ``ref_world`` is the track position plus a
  *kinematically derived closing rate*: the relative velocity projected on
  the line of sight, with the sign convention the hazard layer expects
  (positive closing).  No more hand-planted ``closing_rates`` dict.
- ``FusionThreatPipeline`` is a single end-to-end decision entry point that
  takes the fused track list, the ego pose, and an optional per-track
  visibility decision, and returns the same headline-arbitration dict as
  ``WarningArbitrator`` -- so the fused target list plugs straight into the
  HUD warning slot.

The design keeps the two existing layers untouched (shorten the blast
radius): this module *adapts* ``Track -> AdasMarker`` and *wraps* the
``WarningArbitrator``.  The radar-blind cross-range lesson from v0.18/0.19
is preserved end-to-end: a crossing pedestrian's threat score only reaches
full strength *after* the camera-aided fusion has resolved its lateral
velocity, which is exactly the behaviour a production HUD wants (do not
yell EMERGENCY at a pedestrian the sensors cannot yet prove is closing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..adas import AdasMarker, MarkerProjector
from ..hazard import HazardAssessment, WarningArbitrator, WarningLevel
from ..tracking import Track
from ..utils import quat_to_rotmat
from ..visibility import VisibilityDecision


@dataclass
class TrackToMarkerConfig:
    """Mapping policy from a fused ``Track`` to an ``AdasMarker``.

    Parameters
    ----------
    kind : str
        Marker kind stamped on the mapped marker.  ``FusionThreatPipeline``
        keys its raw-arbitration map on ``id(marker)``, so the kind is purely
        informational here (default ``"fused_track"`` -- distinct from the
        hand-authored ``"hazard"`` / ``"lead_vehicle"`` markers).
    min_closing_rate_mps : float
        Smallest |closing rate| that is reported as-is; anything closer to
        zero than this is snapped to zero.  Keeps a numerically tiny
        projection of a near-perpendicular velocity from looking like a weak
        but real closing target.
    max_marker_range_m : float
        Tracks farther than this are dropped when mapping (the HUD does not
        care about a target beyond its marker range).  None = keep all.
    require_confirmed : bool
        Only map confirmed tracks (default True).  Tentative tracks are
        typically a single weak detection and should not claim a marker.
    """

    kind: str = "fused_track"
    min_closing_rate_mps: float = 0.05
    max_marker_range_m: Optional[float] = 150.0
    require_confirmed: bool = True


class TrackToMarker:
    """Bridge one fused ``Track`` into an ``AdasMarker`` + closing rate."""

    def __init__(self, config: Optional[TrackToMarkerConfig] = None):
        self.config = config or TrackToMarkerConfig()

    def closing_rate(
        self, track: Track, vel_w: np.ndarray | None = None
    ) -> float:
        """Kinematic closing rate (m/s, positive = closing) of a track.

        Uses the track's absolute velocity in the world frame; if the host
        is moving (vel_w given) the *relative* velocity is projected on the
        line of sight from the host to the track position::

            r_hat = (p_track - p_host) / |p_track - p_host|
            closing = - (v_track - v_host) . r_hat

        A target moving directly toward the host gives closing > 0; a target
        receding gives closing < 0; a pure crossing target gives ~0 (the
        v0.18/0.19 blind-lateral-motion lesson survives here).
        """
        cfg = self.config
        p = track.pos
        v = track.vel
        if vel_w is not None:
            v = v - np.asarray(vel_w, dtype=float)[:2]
        n = float(np.hypot(p[0], p[1]))
        if n < 1e-9:
            return 0.0
        # NOTE: Track.pos is a world offset; the *ego* is at the tracking
        # origin (pos_w passed to the tracker).  For a world-anchored track
        # the line of sight is simply the position vector.
        los = p / n
        closing = -float(v @ los)
        if abs(closing) < cfg.min_closing_rate_mps:
            return 0.0
        # Kinematic closing (positive = the gap shrinks).  The hazard layer
        # (v0.11 TTCModel) uses the opposite sign: negative closing = the
        # range is shrinking.  We keep the *kinematic* convention here and
        # flip at the arbitration boundary (see `map`).
        return float(closing)

    def map(self, track: Track, vel_w: np.ndarray | None = None) -> Optional[AdasMarker]:
        """Return an ``AdasMarker`` for a track, or None when filtered out."""
        cfg = self.config
        if cfg.require_confirmed and not track.confirmed:
            return None
        p = track.pos
        if cfg.max_marker_range_m is not None:
            if float(np.hypot(p[0], p[1])) > cfg.max_marker_range_m:
                return None
        world = np.array([p[0], p[1], 0.0], dtype=float)
        marker = AdasMarker(
            kind=cfg.kind,
            ref_world=world,
            label=f"track#{track.track_id}",
        )
        # Store the sign the hazard layer expects (negative = closing) so
        # `arbitrate` can consume `marker.closing_rate_mps` directly.
        object.__setattr__(marker, "closing_rate_mps", -self.closing_rate(track, vel_w))
        return marker


class FusionThreatPipeline:
    """End-to-end fused-track -> HUD warning decision.

    Wraps the v0.11 ``WarningArbitrator`` with the v0.20 track bridge: feed
    it the fused track list and the ego pose, and it builds the marker set,
    derives the closing rates kinematically, runs the arbitration and
    returns the headline warning.

    The pipeline deliberately *keeps the raw arbitration map* keyed by
    ``id(marker)`` so callers can overlay extra hand-authored markers
    (lane lines, static hazards) without losing the fused targets.
    """

    def __init__(
        self,
        mapper: Optional[TrackToMarker] = None,
        arbitrator: Optional[WarningArbitrator] = None,
        projector: Optional[MarkerProjector] = None,
    ):
        self.mapper = mapper or TrackToMarker()
        self.arbitrator = arbitrator or WarningArbitrator()
        self.projector = projector or MarkerProjector()

    def decide(
        self,
        tracks: List[Track],
        pos: np.ndarray,
        att: np.ndarray,
        vel_w: np.ndarray | None = None,
        visibility: Optional[Dict[int, object]] = None,
        extra_markers: Optional[List[AdasMarker]] = None,
    ) -> Dict[str, object]:
        """Arbitrate the fused target list.

        Parameters
        ----------
        tracks : fused ``Track`` list (e.g. ``FusedTracker.tracks``).
        pos, att : ego pose (world) for projection.
        vel_w : optional ego velocity (world); drives kinematic closing.
        visibility : optional {track_id -> VisibilityDecision} gating; keys
            are *track ids* (not marker ids).
        extra_markers : optional extra ``AdasMarker`` list to merge in.

        Returns
        -------
        The v0.11 arbitration dict (winner / level / best / scores) plus
        ``"markers"`` and ``"closing"`` for inspection.
        """
        markers: List[AdasMarker] = []
        closing: Dict[int, float] = {}
        vis_map: Dict[int, object] = {}

        for tr in tracks:
            m = self.mapper.map(tr, vel_w)
            if m is None:
                continue
            markers.append(m)
            closing[id(m)] = m.closing_rate_mps  # hazard-signed
            if visibility is not None and tr.track_id in visibility:
                vis_map[id(m)] = visibility[tr.track_id]
            else:
                # No visibility policy supplied: the track sits in FOV and
                # is not occluded, so it is *visible* by default.  Only an
                # explicit CULLED / faded decision disables it.
                vis_map[id(m)] = VisibilityDecision(
                    action="FULL", status="ok", alpha=1.0,
                    reason="no visibility policy", ang_span_deg=0.0,
                    pixel_span_px=0.0,
                )
        if extra_markers:
            markers.extend(extra_markers)
            for m in extra_markers:
                closing.setdefault(id(m), 0.0)

        res = self.arbitrator.arbitrate(
            markers, self.projector, pos, att, closing, vis_map
        )
        res["markers"] = markers
        res["closing"] = closing
        return res


# ---------------------------------------------------------------------------
# Production lessons (v0.20)
# ---------------------------------------------------------------------------

__PRODUCTION_LESSONS__ = [
    "Derive closing rate kinematically from the fused track (relative "
    "velocity projected on the line of sight), never from a hand-set "
    "constant.  A crossing pedestrian's closing rate converges only as the "
    "v0.19 fusion resolves lateral velocity -- that *delay* is the correct "
    "HUD behaviour, not a bug to paper over.",
    "Gate track->marker mapping on track confirmation.  A tentative "
    "(single-detection) track has a huge lateral-velocity prior and would "
    "otherwise flash a phantom warning marker at the HUD.",
    "Keep the marker range cap on the *unfiltered track*: a target beyond "
    "its marker range is not a warning candidate regardless of its closing "
    "rate.",
    "closing rate sign: positive = the line-of-sight distance is shrinking. "
    "Feed the hazard layer exactly this sign; a receding target (closing "
    "< 0) must never arbitrate above an approaching one.",
]