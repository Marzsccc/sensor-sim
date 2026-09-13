"""Multi-object tracking (MOT) evaluation metrics (v0.21.0).

The v0.17-v0.20 radar stack produces *tracks*; this module answers the next
obvious question -- **how good is the track set, scan by scan?**  It fills the
"GT-to-track metrics" slot left open by ``docs/v0.18.0-radar-tracking-*`` and is
deliberately self-contained (no filter / no radar imports) so it can grade any
tracker, including a vendor black box.

Two metric families, both fed per scan:

* **Set-distance metrics** -- ``ospa`` (Schuhmacher, Vo & Vo 2008) and
  ``gospa`` (Rahmani et al. 2021, the 2-alpha variant).  These compare the
  *point sets* ``{estimate}`` vs ``{truth}`` without any identity, so they are
  the right headline number when tracks come and go: a single scalar that
  penalises missed targets, false tracks **and** localisation error, with a
  cutoff ``c`` that bounds the influence of far-away garbage.
* **Identity metrics** -- ``MotAccumulator`` computes the classic CLEAR-MOT
  scores (MOTA / MOTP / FP / FN / ID-switches) plus mostly-tracked / mostly-lost
  and track-purity bookkeeping when per-object ids are supplied.

Everything uses the world frame and 2-D horizontal positions (m), matching
``Track.pos`` and ``RadarFrame.truth_world_points``.  ``truth_world_points`` is
the convenience that turns a radar ``RadarFrame`` (with its new v0.21.0 truth
bearing channels) into the exact ground-truth reflection points the tracker is
trying to estimate.

References
----------
Schuhmacher, Vo & Vo, "A Consistent Metric for Performance Evaluation of
Multi-Object Filters", IEEE TSP 56(8), 2008.
Rahmani, Dehghan & Gholami, "A Generalised Labelled Multi-Bernoulli Filter"
+ the generalised OSPA (GOSPA) metric, 2021.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Assignment",
    "MotAccumulator",
    "MotFrameResult",
    "assign",
    "gospa",
    "ospa",
    "truth_world_points",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_xy(a) -> np.ndarray:
    """Coerce an ``(N, 1|2|3)`` or ``(N,)`` array (or nested lists) to ``(N, 2)``."""
    arr = np.asarray(a, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2))
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise ValueError("points must have at least 2 columns (x, y)")
    return arr[:, :2]


def _pdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(M, N)`` Euclidean distance matrix between two ``(., 2)`` point sets."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    d = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.einsum("mnk,mnk->mn", d, d))


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-Python O(n^3) Hungarian fallback (square matrices only).

    Used when SciPy is unavailable; pads rectangular input to square with a
    large cost so the returned matching is still minimum-cost.
    """
    a = np.asarray(cost, dtype=float)
    n, m = a.shape
    if n == 0 or m == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    k = max(n, m)
    big = (float(np.max(a)) + 1.0) * (k + 1) if a.size else 1.0
    mat = np.full((k, k), big, dtype=float)
    mat[:n, :m] = a

    u = np.zeros(k + 1)
    v = np.zeros(k + 1)
    p = np.zeros(k + 1, dtype=int)
    way = np.zeros(k + 1, dtype=int)
    inf = np.inf
    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = np.full(k + 1, inf)
        used = np.zeros(k + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, k + 1):
                if not used[j]:
                    cur = mat[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, k + 1):
        i = p[j]
        if i and i <= n and j <= m:
            rows.append(i - 1)
            cols.append(j - 1)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _solve_square(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-cost assignment for a **square** matrix (SciPy if present)."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except ImportError:
        return _hungarian(cost)


# ---------------------------------------------------------------------------
# set-distance metrics
# ---------------------------------------------------------------------------

def ospa(
    est_xy,
    gt_xy,
    cutoff: float,
    order: float = 1.0,
    *,
    return_components: bool = False,
):
    """OSPA distance between two 2-D point sets.

    ``d = [ 1/n_max * ( min_assign sum d_c(x, y)^p + c^p * (n_max - m) ) ]^(1/p)``
    where ``d_c = min(||x-y||, c)``, ``m = |est|``, ``n = |gt|`` and
    ``n_max = max(m, n)``.  Invariant to point ordering; ``0`` iff the sets
    coincide.

    Parameters
    ----------
    est_xy, gt_xy : array_like
        ``(M, 2)`` / ``(N, 2)`` estimated and true points (m).  Empty is fine.
    cutoff : float
        OSPA cutoff ``c > 0`` (m) -- the per-point penalty for a miss/false.
    order : float
        Order ``p >= 1``.  ``p = 1`` is the standard choice.
    return_components : bool
        Also return ``(localisation, cardinality)`` as a dict.

    Returns
    -------
    float | dict
    """
    c = float(cutoff)
    if c <= 0:
        raise ValueError("cutoff must be > 0")
    p = float(order)
    if p < 1:
        raise ValueError("order p must be >= 1")
    e = _as_xy(est_xy)
    g = _as_xy(gt_xy)
    m, n = e.shape[0], g.shape[0]
    nmax = max(m, n)
    if nmax == 0:
        val = 0.0
        if return_components:
            return {"d": val, "localisation": 0.0, "cardinality": 0.0}
        return val

    # Square n_max x n_max cost: real-real pairs truncated at c; a real
    # matched to a dummy (i.e. cardinality penalty) costs c^p.  The
    # bottom-right dummy/dummy block is always empty because
    # n_max = max(m, n).
    cost = np.full((nmax, nmax), c, dtype=float)
    if m and n:
        cost[:m, :n] = np.minimum(_pdist(e, g), c)

    # Full matching on the square matrix.
    row, col = _solve_square(cost)

    # Recompute the objective from the returned matching (robust to padding).
    obj = 0.0
    loc = 0.0
    for i, j in zip(row, col):
        if i < m and j < n:
            d = min(float(np.hypot(*(e[i] - g[j]))), c)
            obj += d ** p
            loc += d ** p
        else:
            obj += c ** p
    d_ospa = (obj / nmax) ** (1.0 / p)
    if return_components:
        card = obj - loc
        return {
            "d": float(d_ospa),
            "localisation": float((loc / nmax) ** (1.0 / p)),
            "cardinality": float((card / nmax) ** (1.0 / p)),
        }
    return float(d_ospa)


def gospa(
    est_xy,
    gt_xy,
    cutoff: float,
    order: float = 1.0,
    alpha: float = 2.0,
) -> Dict[str, float]:
    """Generalised OSPA (2-alpha variant) with a full decomposition.

    ``d^p = min_gamma [ sum_{matched} ||x-y||^p
                        + (c^p/alpha)(|X| - |gamma|)
                        + (c^p/alpha)(|Y| - |gamma|) ]``

    Returns a dict ``{"total", "localisation", "missed", "false"}`` where the
    three components are already in the *same* p-th-root units as ``total``
    (i.e. ``total^p == localisation^p + missed^p + false^p``).

    Parameters
    ----------
    cutoff : float
        Cutoff ``c > 0`` (m).
    order : float
        Order ``p >= 1``.
    alpha : float
        Trade-off ``alpha >= 1`` (``alpha = 2`` is the standard, cardinality
        penalty ``c^p / alpha`` per unmatched object).
    """
    c = float(cutoff)
    p = float(order)
    a = float(alpha)
    if c <= 0 or p < 1 or a <= 0:
        raise ValueError("need cutoff > 0, order >= 1, alpha > 0")
    e = _as_xy(est_xy)
    g = _as_xy(gt_xy)
    m, n = e.shape[0], g.shape[0]
    if m == 0 and n == 0:
        return {"total": 0.0, "localisation": 0.0, "missed": 0.0, "false": 0.0}

    # (m + n) x (n + m) square: real rows x real cols = d^p - 2c^p/alpha,
    # anything involving a dummy = 0 (i.e. "leave it unmatched").
    size = m + n
    cost = np.zeros((size, size), dtype=float)
    pen = 2.0 * c ** p / a  # benefit of matching one pair (2 * c^p / alpha)
    if m and n:
        cost[:m, :n] = _pdist(e, g) ** p - pen
    row, col = _solve_square(cost)

    loc = 0.0
    n_matched = 0
    for i, j in zip(row, col):
        if i < m and j < n:
            d = float(np.hypot(*(e[i] - g[j])))
            if d ** p < pen:  # only beneficial matches count
                loc += d ** p
                n_matched += 1
    missed = (c ** p / a) * (n - n_matched)
    false = (c ** p / a) * (m - n_matched)
    total = loc + missed + false
    inv = 1.0 / p
    return {
        "total": float(total ** inv),
        "localisation": float(loc ** inv),
        "missed": float(missed ** inv),
        "false": float(false ** inv),
    }


# ---------------------------------------------------------------------------
# identity-based association
# ---------------------------------------------------------------------------

@dataclass
class Assignment:
    """One scan's estimate<->truth association."""

    matches: List[Tuple[int, int]] = field(default_factory=list)
    unmatched_est: List[int] = field(default_factory=list)
    unmatched_gt: List[int] = field(default_factory=list)
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_matches(self) -> int:
        return len(self.matches)


def assign(
    est_xy, gt_xy, *, gate: Optional[float] = None, use_hungarian: bool = True
) -> Assignment:
    """Associate estimate points to truth points (globally, min total distance).

    Parameters
    ----------
    gate : float, optional
        Pairing distance ceiling (m).  Pairs farther apart than ``gate`` are
        forbidden; a leftover estimate/truth on either side is reported as
        unmatched.
    use_hungarian : bool
        ``True`` = optimal (global) assignment; ``False`` = greedy
        nearest-neighbour (cheaper, order-dependent).
    """
    e = _as_xy(est_xy)
    g = _as_xy(gt_xy)
    m, n = e.shape[0], g.shape[0]
    out = Assignment()
    if m == 0 or n == 0:
        out.unmatched_est = list(range(m))
        out.unmatched_gt = list(range(n))
        return out

    D = _pdist(e, g)
    if use_hungarian:
        size = m + n
        # Cost of leaving an estimate / a truth object unmatched.  A match is
        # always (weakly) preferred over two unmatched objects, so any pair
        # inside the gate is taken over discarding either side.
        pen = float(gate) if gate is not None else float(D.max()) + 1.0
        forbidden = pen * (size + 1) + float(D.max()) + 1.0
        cost = np.full((size, size), 0.0)
        cost[:m, n:] = pen          # estimate -> dummy truth
        cost[m:, :n] = pen          # dummy estimate -> truth
        block = D if gate is None else np.where(D <= gate, D, forbidden)
        cost[:m, :n] = block
        row, col = _solve_square(cost)
        pairs = [(i, j) for i, j in zip(row, col) if i < m and j < n]
        if gate is not None:
            pairs = [(i, j) for i, j in pairs if D[i, j] <= gate]
    else:
        pairs = []
        used_e, used_g = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        for i, j in order:
            if i in used_e or j in used_g:
                continue
            if gate is not None and D[i, j] > gate:
                continue
            pairs.append((int(i), int(j)))
            used_e.add(int(i))
            used_g.add(int(j))

    pairs.sort()
    out.matches = [(int(i), int(j)) for i, j in pairs]
    matched_e = {i for i, _ in out.matches}
    matched_g = {j for _, j in out.matches}
    out.unmatched_est = [i for i in range(m) if i not in matched_e]
    out.unmatched_gt = [j for j in range(n) if j not in matched_g]
    out.distances = np.asarray([D[i, j] for i, j in out.matches], dtype=float)
    return out


# ---------------------------------------------------------------------------
# streaming MOTA / MOTP / ID-switch accumulator
# ---------------------------------------------------------------------------

@dataclass
class MotFrameResult:
    """Per-scan MOT outcome (returned by :meth:`MotAccumulator.add`)."""

    t: float
    n_est: int
    n_gt: int
    matches: int
    false_positives: int
    false_negatives: int
    id_switches: int
    localization_error_sum: float
    ospa: float
    gospa: Dict[str, float]
    distances: np.ndarray = field(default_factory=lambda: np.zeros(0))


class MotAccumulator:
    """Streaming CLEAR-MOT + OSPA/GOSPA evaluator for a tracker run.

    Feed one scan at a time::

        acc = MotAccumulator(gate_m=3.0, cutoff_m=6.0)
        for frame in scan_stream:
            acc.add(frame.tracks_xy, frame.truth_xy, ...)
        print(acc.summary())

    Parameters
    ----------
    gate_m : float
        Association gate (m) used for the identity metrics (FP/FN/IDSW).
    cutoff_m : float, optional
        Set-metric cutoff (m).  Defaults to ``2 * gate_m`` -- the classic
        "a track beyond the gate is as bad as a missing one" choice.
    order, alpha : float
        OSPA order ``p`` and GOSPA ``alpha``.
    use_hungarian : bool
        Global assignment for the identity metrics (``False`` = greedy).
    """

    def __init__(
        self,
        gate_m: float = 3.0,
        cutoff_m: Optional[float] = None,
        order: float = 1.0,
        alpha: float = 2.0,
        use_hungarian: bool = True,
    ):
        self.gate_m = float(gate_m)
        self.cutoff_m = float(cutoff_m) if cutoff_m is not None else 2.0 * float(gate_m)
        self.order = float(order)
        self.alpha = float(alpha)
        self.use_hungarian = bool(use_hungarian)

        # totals
        self.n_frames = 0
        self.n_gt = 0
        self.n_fp = 0
        self.n_fn = 0
        self.n_matches = 0
        self.n_idsw = 0
        self.loc_sum = 0.0
        self._ospa_sum = 0.0
        self._gospa_sum = 0.0
        self._gospa_loc = 0.0
        self._gospa_missed = 0.0
        self._gospa_false = 0.0

        # per-id bookkeeping
        self._gt_present: Dict[object, int] = {}
        self._gt_tracked: Dict[object, int] = {}
        self._last_gt_of_track: Dict[object, object] = {}

    # -- ingest -----------------------------------------------------------
    def add(
        self,
        est_xy,
        gt_xy,
        *,
        est_ids: Optional[Sequence] = None,
        gt_ids: Optional[Sequence] = None,
        t: float = 0.0,
    ) -> MotFrameResult:
        """Ingest one scan's estimate set and truth set."""
        e = _as_xy(est_xy)
        g = _as_xy(gt_xy)
        res = assign(e, g, gate=self.gate_m, use_hungarian=self.use_hungarian)

        idsw = 0
        est_id_list = None if est_ids is None else list(est_ids)
        gt_id_list = None if gt_ids is None else list(gt_ids)
        if est_id_list is not None and gt_id_list is not None:
            if len(est_id_list) != e.shape[0] or len(gt_id_list) != g.shape[0]:
                raise ValueError("est_ids/gt_ids length mismatch")
            for i, j in res.matches:
                tid = est_id_list[i]
                gid = gt_id_list[j]
                prev = self._last_gt_of_track.get(tid)
                if prev is not None and prev != gid:
                    idsw += 1
                self._last_gt_of_track[tid] = gid
                self._gt_tracked[gid] = self._gt_tracked.get(gid, 0) + 1
            for gid in gt_id_list:
                self._gt_present[gid] = self._gt_present.get(gid, 0) + 1

        o = ospa(e, g, self.cutoff_m, self.order)
        gs = gospa(e, g, self.cutoff_m, self.order, self.alpha)

        self.n_frames += 1
        self.n_gt += g.shape[0]
        self.n_fp += len(res.unmatched_est)
        self.n_fn += len(res.unmatched_gt)
        self.n_matches += res.n_matches
        self.n_idsw += idsw
        self.loc_sum += float(res.distances.sum())
        self._ospa_sum += o
        self._gospa_sum += gs["total"]
        self._gospa_loc += gs["localisation"]
        self._gospa_missed += gs["missed"]
        self._gospa_false += gs["false"]

        return MotFrameResult(
            t=float(t),
            n_est=e.shape[0],
            n_gt=g.shape[0],
            matches=res.n_matches,
            false_positives=len(res.unmatched_est),
            false_negatives=len(res.unmatched_gt),
            id_switches=idsw,
            localization_error_sum=float(res.distances.sum()),
            ospa=float(o),
            gospa=gs,
            distances=res.distances,
        )

    # -- reporting --------------------------------------------------------
    def summary(self) -> Dict[str, float]:
        """Aggregate scores over everything ingested so far."""
        errors = self.n_fp + self.n_fn + self.n_idsw
        if self.n_gt:
            mota = 1.0 - errors / self.n_gt
        elif errors == 0:
            mota = 1.0  # nothing happened -- vacuously perfect
        else:
            mota = float("nan")  # errors with no truth -> undefined ratio
        motp = self.loc_sum / self.n_matches if self.n_matches else 0.0
        f = max(self.n_frames, 1)
        out: Dict[str, float] = {
            "frames": self.n_frames,
            "gt": self.n_gt,
            "matches": self.n_matches,
            "false_positives": self.n_fp,
            "false_negatives": self.n_fn,
            "id_switches": self.n_idsw,
            "mota": float(mota),
            "motp": float(motp),
            "mean_ospa": self._ospa_sum / f,
            "mean_gospa": self._gospa_sum / f,
            "mean_gospa_localisation": self._gospa_loc / f,
            "mean_gospa_missed": self._gospa_missed / f,
            "mean_gospa_false": self._gospa_false / f,
        }
        # mostly-tracked / mostly-lost (needs gt ids across scans)
        if self._gt_present:
            mt = pt = ml = 0
            for gid, life in self._gt_present.items():
                frac = self._gt_tracked.get(gid, 0) / life
                if frac >= 0.8:
                    mt += 1
                elif frac <= 0.2:
                    ml += 1
                else:
                    pt += 1
            out.update(
                {
                    "mostly_tracked": mt,
                    "partially_tracked": pt,
                    "mostly_lost": ml,
                    "mt_ratio": mt / len(self._gt_present),
                    "ml_ratio": ml / len(self._gt_present),
                }
            )
        return out

    @property
    def mota(self) -> float:
        return self.summary()["mota"]

    @property
    def motp(self) -> float:
        return self.summary()["motp"]


# ---------------------------------------------------------------------------
# radar-truth convenience
# ---------------------------------------------------------------------------

def truth_world_points(frame, host_pos, att_wxyz, mount_t_body=None) -> np.ndarray:
    """``(N, 2)`` horizontal ground-truth reflection points from a ``RadarFrame``.

    Wrapper over ``RadarFrame.truth_world_points`` that (a) works from the
    **host** position (not the sensor origin) so the rigid mount offset is
    accounted for, and (b) drops the vertical axis, so the result plugs
    straight into :func:`ospa` / :class:`MotAccumulator`.

    Parameters
    ----------
    host_pos : array_like
        Host position in the world frame (the same ``pos`` fed to
        ``RadarSensor.scan``).
    att_wxyz : array_like
        Host attitude quaternion ``(w, x, y, z)``.
    mount_t_body : array_like, optional
        Radar mount translation in the body frame (must match the
        ``RadarSensor`` that produced the frame).  ``None`` = zero offset.
    """
    from .utils import quat_to_rotmat

    R_bw = quat_to_rotmat(np.asarray(att_wxyz, dtype=float))
    R_wb = R_bw.T
    origin = np.asarray(host_pos, dtype=float)
    if mount_t_body is not None:
        origin = origin + R_wb @ np.asarray(mount_t_body, dtype=float)
    pts = frame.truth_world_points(origin, att_wxyz)
    if pts.size == 0:
        return np.zeros((0, 2))
    return np.asarray(pts, dtype=float)[:, :2]
