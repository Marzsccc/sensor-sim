"""
Hazard assessment & warning arbitration (v0.11.0).

The v0.9.0 `MarkerProjector` / `AdasPipeline` already places ADAS markers
(ACC lead-vehicle bbox, FCW/AEB hazard wedge, lane lines) on the HUD at the
right place and time.  v0.10.0's `MarkerVisibilityPolicy` decides whether a
marker should be FULL / FADED / CULLED based on FOV, near-field angular
oversize and bodywork occlusion.  What neither of them does is *decide what
to tell the driver*: how urgent is this hazard, and if several hazards are
present, which one claims the warning?

This module adds the decision layer.  It is deliberately downstream of the
perception/render chain so it reuses the *compensated* (display-time) marker
geometry that v0.5/v0.9 already produced:

  1. `TimeToCollision` -- solid TTC on the marker's projected body-frame
     range + closing rate.  Two models are provided: constant-relative-
     speed (the ISO 15623 / FCW backbone) and constant-deceleration
     (the richer AEB braking model).  Both consume display-time range so
     latency is already accounted for by the caller.

  2. `HazardAssessment` -- turns a marker + ego kinematics into a normalized
     threat score in [0,1].  The score fuses (a) TTC vs a model horizon,
     (b) a marker-type urgency weight (hazard > lead_vehicle > lane), and
     (c) a cross-track alignment term so only markers actually in the ego
     lane / bearing contribute.

  3. `WarningArbitrator` -- across many assessed markers, selects the single
     most threatening one and maps its score to a discrete warning level
     (OFF / CAUTION / WARN / CRITICAL / EMERGENCY).  This mirrors an AR-HUD
     that has one "headline" warning slot while less urgent markers stay
     drawn but quiet.

  4. `ThreatPipeline` -- end-to-end loop combining the existing
     `AdasPipeline` marker projection with this decision layer, and exposing
     both a nominal and a *reacted* path so latency-compensation error can
     be measured down to the warning level (i.e. does latency flip a WARN
     into a CRITICAL?).

Warning-level semantics follow the familiar automotive ladder:

  OFF       -- nothing above the latch threshold
  CAUTION   -- advisory (e.g. longish-TTC merge traffic); draw, don't alarm
  WARN      -- FCW-level; imminent braking advisory
  CRITICAL  -- AEB-deploy region; near-limit TTC
  EMERGENCY -- "can't rely on driver reaction alone" region

Production notes (as with v0.8/v0.10) are collected at the bottom of the
module -- the traps an integrator hits when these rules go into a real HUD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .adas import AdasMarker, MarkerProjector
from .visibility import VisibilityDecision


# ---------------------------------------------------------------------------
# Warning level
# ---------------------------------------------------------------------------

class WarningLevel(IntEnum):
    """Discrete HUD warning ladder, ordered by urgency."""
    OFF = 0
    CAUTION = 1
    WARN = 2
    CRITICAL = 3
    EMERGENCY = 4

    @property
    def label(self) -> str:
        return {
            WarningLevel.OFF: "OFF",
            WarningLevel.CAUTION: "CAUTION",
            WarningLevel.WARN: "WARN",
            WarningLevel.CRITICAL: "CRITICAL",
            WarningLevel.EMERGENCY: "EMERGENCY",
        }[self]


# ---------------------------------------------------------------------------
# Time-to-collision
# ---------------------------------------------------------------------------

@dataclass
class TTCModel:
    """Compute TTC from body-frame range and closing rate.

    Two kinematic anchors:

      * constant relative speed  (relative_accel = 0):
          ttc = -r / r_dot            (r_dot < 0)
        This is the ISO 15623 forward-collision-warning backbone -- it needs
        no assumption about the lead's braking, only that the *relative*
        speed is constant over the short horizon.

      * constant relative deceleration (a_rel < 0):
          ttc = ( -v_rel - sqrt(v_rel^2 - 2 a_rel r) ) / a_rel
        The AEB-style model: the gap closes under a sustained relative
        deceleration (e.g. lead braking harder than ego).  Falls back to the
        constant-speed value when the discriminant would be imaginary.

    Range is expected to already be the *display-time* (latency-compensated)
    range from the caller; this class does not re-derive it.
    """
    min_decel: float = -9.0       # most aggressive sustained rel. decel (m/s^2)
    gating_range_m: float = 200.0 # ignore threats beyond this range (m)

    def ttc_const_speed(self, rng: float, closing_rate: float) -> Optional[float]:
        """TTC under constant relative speed. None when not closing or out of range."""
        rng = float(rng)
        if rng > self.gating_range_m or rng <= 0.0:
            return None
        if closing_rate >= 0.0:   # not closing
            return None
        return -rng / closing_rate   # closing_rate < 0 => positive

    def ttc_const_decel(self, rng: float, closing_rate: float,
                        a_rel: Optional[float] = None) -> Optional[float]:
        """TTC under constant relative deceleration (`a_rel`, default min_decel).

        Returns None when the model is degenerate (not closing, out of range)
        or the discriminant is imaginary (the quadratic has no real positive
        root within the assumed deceleration, i.e. the closest approach is
        not a collision in this model).
        """
        rng = float(rng)
        if rng > self.gating_range_m or rng <= 0.0:
            return None
        v = float(closing_rate)
        a = float(a_rel) if a_rel is not None else self.min_decel
        if a >= 0.0:
            # no assumed deceleration -> fall back to constant speed
            return self.ttc_const_speed(rng, v)
        disc = v * v - 2.0 * a * rng
        if disc < 0.0:
            # cannot collide within this deceleration assumption
            return None
        sqrt_disc = np.sqrt(disc)
        # choose the smaller positive root
        t1 = (-v - sqrt_disc) / a
        t2 = (-v + sqrt_disc) / a
        cand = [t for t in (t1, t2) if t is not None and np.isfinite(t) and t >= 0.0]
        if not cand:
            return None
        return min(cand)


# ---------------------------------------------------------------------------
# Per-marker threat score
# ---------------------------------------------------------------------------

@dataclass
class HazardAssessment:
    """Normalize one marker+kinematics into a threat score / level.

    score in [0,1]: a blend of
      - `ttc_term` : how near TTC is relative to a model horizon
        (nearer TTC -> 1)  -- drives the urgency.
      - `type_weight` : fixed urgency by marker kind
        (hazard/lead_vehicle > lane), 0 for pure-lane advisory markers.
      - `alignment` : cross-track / bearing term, 1 when dead-ahead in the
        ego path, decaying with |azimuth|.  Keeps off-axis hazards from
        claiming the driver's headline warning.

    Properly, `closing_rate` should come from a tracker (range-rate from the
    perception pipeline); for the simulator it can be derived from ego vs
    object velocity if the caller passes them.
    """
    ttc_horizon_s: float = 4.0   # TTC at/under which score ramps to ~1
    long_ttc_s: float = 12.0     # TTC beyond which ttc_term ~ 0 (no threat)
    weight_by_kind: Dict[str, float] = field(default_factory=lambda: {
        "hazard": 1.00,
        "lead_vehicle": 0.90,
        "lane_left": 0.15,
        "lane_right": 0.15,
        "lane_center": 0.35,
    })
    ttc_model: TTCModel = field(default_factory=TTCModel)
    # Only invoke the *constant-deceleration* TTC (the aggressive AEB model)
    # once the closing speed is genuinely high.  A slow-closing hazard is
    # more honestly read by the conservative constant-speed FCW floor; a
    # panicking -9 m/s^2 relative decel on a 2 m/s closure would over-claim.
    decel_fallback_min_speed: float = 4.0

    def ttc_term(self, ttc: Optional[float]) -> float:
        """Ramp TTC -> threat: 1 at ttc<=horizon, 0 at ttc>=long_ttc."""
        if ttc is None:
            return 0.0
        if ttc <= self.ttc_horizon_s:
            return 1.0
        if ttc >= self.long_ttc_s:
            return 0.0
        return (self.long_ttc_s - ttc) / (self.long_ttc_s - self.ttc_horizon_s)

    def alignment_term(self, azimuth_rad: float, half_width_deg: float = 12.0) -> float:
        """1 when dead-ahead, decaying with |azimuth| to 0 at half_width_deg."""
        half = np.deg2rad(half_width_deg)
        azimuth_rad = float(azimuth_rad)
        if abs(azimuth_rad) >= half:
            return 0.0
        return 1.0 - abs(azimuth_rad) / half

    def assess(
        self,
        marker: AdasMarker,
        rng: float,
        closing_rate: float,
        azimuth_rad: float = 0.0,
        visible: bool = True,
    ) -> Dict[str, object]:
        """Produce a threat assessment for one marker.

        Parameters
        ----------
        marker : the ADAS marker being assessed.
        rng : display-time horizontal range (m) to its reference point.
        closing_rate : closing rate (m/s), negative when the gap shrinks.
        azimuth_rad : bearing of the marker in the body frame (0 = ahead).
        visible : whether v0.10 visibility gave it a drawable status; a
            CULLED / fully-faded marker cannot claim the warning.

        Returns
        -------
        dict with keys score, ttc, ttc_term, type_weight, alignment, level.
        """
        ttc = self.ttc_model.ttc_const_speed(rng, closing_rate)
        # Use the richer constant-deceleration TTC for the threat score only
        # for genuinely fast closures, and even then only when it is more
        # urgent than the conservative constant-speed estimate; otherwise
        # keep the FCW floor.
        if ttc is None or ttc > self.ttc_horizon_s:
            if abs(closing_rate) >= self.decel_fallback_min_speed:
                ttc_decel = self.ttc_model.ttc_const_decel(rng, closing_rate)
                if ttc_decel is not None and ttc is not None and ttc_decel < ttc:
                    ttc = ttc_decel
        tw = self.weight_by_kind.get(marker.kind, 0.5)
        alg = self.alignment_term(azimuth_rad)
        base = 0.0 if not visible else (tw * alg)
        tterm = self.ttc_term(ttc)
        # Score = geometric blend: alignment/type gate the *urgency*, TTC
        # gates the *magnitude*.  A perfectly-aligned, high-priority marker
        # with a near TTC scores 1; an off-axis lane edge with a long TTC ~ 0.
        score = base * (0.35 + 0.65 * tterm)
        score = float(np.clip(score, 0.0, 1.0))
        level = level_from_score(score)
        return {
            "score": score,
            "ttc": ttc,
            "ttc_term": tterm,
            "type_weight": tw,
            "alignment": alg,
            "level": level,
        }


def level_from_score(score: float, thresholds: Optional[Tuple[float, float, float, float]] = None) -> WarningLevel:
    """Map a [0,1] threat score to a WarningLevel.

    Default ladder: EMERGENCY >= 0.85, CRITICAL >= 0.55, WARN >= 0.35,
    CAUTION >= 0.12, else OFF.  `thresholds` = (caution, warn, critical,
    emergency) to override.
    """
    t_c, t_w, t_crit, t_emerg = thresholds or (0.12, 0.35, 0.55, 0.85)
    if score >= t_emerg:
        return WarningLevel.EMERGENCY
    if score >= t_crit:
        return WarningLevel.CRITICAL
    if score >= t_w:
        return WarningLevel.WARN
    if score >= t_c:
        return WarningLevel.CAUTION
    return WarningLevel.OFF


# ---------------------------------------------------------------------------
# Arbitration
# ---------------------------------------------------------------------------

@dataclass
class WarningArbitrator:
    """Select the single most threatening marker across a scene.

    Mirrors an AR-HUD that has one headline warning slot.  `assess` is run
    per marker; the winner is the highest `score`, with an explicit
    tie-break toward the most urgent WarningLevel then earliest TTC.
    Markers that are CULLED / invisible cannot win.
    """
    assessment: HazardAssessment = field(default_factory=HazardAssessment)

    def arbitrate(
        self,
        markers: List[AdasMarker],
        project: MarkerProjector,
        pos: np.ndarray,
        att: np.ndarray,
        closing_rates: Dict[str, float],
        visibility: Dict[str, VisibilityDecision],
    ) -> Dict[str, object]:
        """Assess every marker and return the headline arbitration.

        Parameters
        ----------
        markers : markers in the scene.
        project : the v0.9 MarkerProjector (share one instance).
        pos, att : display-time ego pose used for projection.
        closing_rates : {marker label/kind index -> closing rate m/s}.  Used
            as a per-marker key map; keyed by `id(marker)` by default.
        visibility : {key -> v0.10 VisibilityDecision} so CULLED markers are
            excluded from claiming the warning.

        Returns
        -------
        dict with keys
          winner     : the most threatening AdasMarker, or None.
          level      : WarningLevel of the winner (OFF when none).
          best       : the winner's assessment dict, or None.
          scores     : list of (marker, assessment) for all markers.
        """
        out = []
        for m in markers:
            key = id(m)
            vdec = visibility.get(key)
            # A marker is eligible to claim the warning only if the v0.10
            # decision did not cull it (CULLED = fully faded / culled)
            # and drawable in FOV/range.
            visible = bool(vdec is not None and vdec.action != "CULLED")
            proj = project.project(m, pos, att)
            rng = proj["range"]
            az = proj["azimuth"]
            cr = closing_rates.get(key, closing_rates.get(m.kind, 0.0))
            a = self.assessment.assess(m, rng, cr, az, visible=visible)
            out.append((m, a))
        if not out:
            return {"winner": None, "level": WarningLevel.OFF, "best": None, "scores": []}
        winner_m, winner_a = max(
            out,
            key=lambda ma: (
                ma[1]["score"],
                ma[1]["level"].value,
                -(ma[1]["ttc"] if ma[1]["ttc"] is not None else np.inf),
            ),
        )
        return {
            "winner": winner_m,
            "level": winner_a["level"],
            "best": winner_a,
            "scores": out,
        }


# ---------------------------------------------------------------------------
# End-to-end threat pipeline
# ---------------------------------------------------------------------------

@dataclass
class ThreatPipeline:
    """End-to-end hazard assessment under latency.

    Wraps the v0.9 `AdasPipeline` marker-render path and adds the decision
    layer above, so the *warning level itself* can be compared between the
    naive (latency-lagged) and compensated (display-time predicted) paths.
    This answers the question v0.9 answered for marker pixels but now for
    the safety decision: does uncompensated latency flip a WARN into a
    CRITICAL or update it one frame too late?
    """
    assessment: HazardAssessment = field(default_factory=HazardAssessment)
    arbitrator: WarningArbitrator = field(default_factory=WarningArbitrator)

    def decide(
        self,
        markers: List[AdasMarker],
        display_pose,       # (pos, att) at display time (truth)
        fused_pose,         # (pos, att) lagged by pipeline latency (naive)
        predicted_pose,     # (pos, att) compensated to display time
        closing_rates: Dict[str, float],
        visibility: Dict[str, VisibilityDecision],
        project: Optional[MarkerProjector] = None,
    ) -> Dict[str, object]:
        """Compute naive vs compensated warning levels for the same scene.

        Returns a dict summarizing both paths and whether the *decision*
        (warning level) changed due to latency.
        """
        project = project or MarkerProjector()
        out = {}
        for name, pose, tag in (
            ("true", display_pose, "truth"),
            ("naive", fused_pose, "naive"),
            ("compensated", predicted_pose, "compensated"),
        ):
            ar = self.arbitrator.arbitrate(
                markers, project, pose[0], pose[1], closing_rates, visibility
            )
            out[tag] = ar
        naive_lvl = out["naive"]["level"]
        comp_lvl = out["compensated"]["level"]
        return {
            **out,
            "level_changed": naive_lvl != comp_lvl,
            "level_delta": comp_lvl.value - naive_lvl.value,
        }


# ---------------------------------------------------------------------------
# Production lessons (for AR-HUD integration)
# ---------------------------------------------------------------------------

__PRODUCTION_LESSONS__ = [
    "Compute TTC on display-time (compensated) range, never the raw fused "
    "range -- uncompensated latency reports the ego behind truth, so the "
    "hazard looks farther than it is and the HUD UNDER-claims urgency "
    "(a would-be EMERGENCY is read as CRITICAL).",
    "Gate warnings on the v0.10 visibility decision: a CULLED/fully-faded "
    "marker (behind, too close, out of FOV or bodywork-occluded) must not "
    "claim the headline warning slot.",
    "Cross-track / bearing alignment matters: an FCW wedge a few degrees "
    "off-axis in an adjacent lane is an advisory, not an alert.  Decay the "
    "threat by azimuth before it can arbitrate.",
    "Choose the constant-deceleration TTC for AEB-style decisions; the "
    "constant-speed model is the FCW conservative floor.",
    "Arbitration is single-slot; less-urgent markers stay drawn but quiet. "
    "Do not let a distant lane-edge advisory override a real collision threat.",
]
