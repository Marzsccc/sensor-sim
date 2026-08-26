"""
Unified ground-truth evaluation for estimators (v0.14.0).

One place to answer "is my fusion output any good today":

  - collect per-tick errors (+ optional covariance blocks) from ANY
    estimator run (ESKF, dead reckoning, a vendor black box ...)
  - RMSE / mean / max / percentile statistics per quantity
  - NEES consistency of the reported covariance against the true error
    (chi-square gate, same math as ``outage.consistency`` but streaming)
  - pass/fail regression gates ("position RMSE < 2 m") so a CI job or
    an overnight sweep can say YES/NO instead of dumping plots

The module never runs a filter itself -- it consumes samples. See
``examples/evaluation_demo.py`` for an end-to-end ESKF vs dead-reckoning
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Streaming NEES accumulator
# ---------------------------------------------------------------------------

class NeesAccumulator:
    """Streaming NEES (normalized estimation error squared) collector.

    NEES for one sample is ``eps = e^T P^-1 e ~ chi2(dim)``.  The sample
    mean over N independent ticks should lie inside
    ``[chi2(a/2, N*dim), chi2(1-a/2, N*dim)] / N`` for a consistent
    estimator (covariance matches actual error spread).
    """

    def __init__(self, dim: int, alpha: float = 0.05):
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2 or 3")
        self.dim = int(dim)
        self.alpha = float(alpha)
        self._nees: List[float] = []

    def add(self, error: np.ndarray, covariance: np.ndarray) -> float:
        """Add one tick; returns that tick's NEES value."""
        e = np.asarray(error, dtype=float).ravel()[0:self.dim]
        P = np.asarray(covariance, dtype=float)[0:self.dim, 0:self.dim]
        P = 0.5 * (P + P.T)
        eps = float(e @ np.linalg.solve(P, e))
        self._nees.append(eps)
        return eps

    @property
    def n(self) -> int:
        return len(self._nees)

    def result(self) -> Dict[str, float]:
        """chi2 consistency verdict for the accumulated samples."""
        if not self._nees:
            return {"n": 0}
        arr = np.asarray(self._nees)
        try:
            from scipy import stats  # lazy, optional dep
            lo = stats.chi2.ppf(self.alpha / 2.0, self.n * self.dim) / self.n
            hi = stats.chi2.ppf(1 - self.alpha / 2.0, self.n * self.dim) / self.n
        except ImportError:
            # Normal approximation fallback (fine for N > ~100)
            lo, hi = self.dim * (1 - 2 * np.sqrt(2.0 / self.n)), \
                     self.dim * (1 + 2 * np.sqrt(2.0 / self.n))
        mean = float(arr.mean())
        return {
            "n": self.n,
            "nees_mean": mean,
            "nees_median": float(np.median(arr)),
            "nees_max": float(arr.max()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "consistent": bool(lo <= mean <= hi),
        }


# ---------------------------------------------------------------------------
# Scalar metric stream (RMSE etc.)
# ---------------------------------------------------------------------------

class MetricStream:
    """Accumulates one scalar error series and computes its statistics."""

    def __init__(self, name: str):
        self.name = name
        self._values: List[float] = []

    def add(self, v: float) -> None:
        self._values.append(float(v))

    @property
    def n(self) -> int:
        return len(self._values)

    def result(self) -> Dict[str, float]:
        if not self._values:
            return {"n": 0}
        a = np.asarray(self._values)
        return {
            "n": self.n,
            "rmse": float(np.sqrt(np.mean(a ** 2))),
            "mean": float(a.mean()),
            "max": float(np.abs(a).max()),
            "p95_abs": float(np.percentile(np.abs(a), 95)),
        }


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Outcome of one pass/fail regression gate."""
    name: str
    passed: bool
    detail: str


@dataclass
class EvalReport:
    """Full evaluation report for one estimator run."""
    name: str
    duration_s: float = 0.0
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    nees: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gates: List[GateResult] = field(default_factory=list)

    def metric(self, key: str, stat: str = "rmse") -> Optional[float]:
        m = self.metrics.get(key)
        return None if not m else m.get(stat)

    def all_gates_passed(self) -> bool:
        return bool(self.gates) and all(g.passed for g in self.gates)

    # -- rendering ---------------------------------------------------------

    def summary(self) -> str:
        lines = [f"=== Evaluation report: {self.name} "
                 f"({self.duration_s:.1f} s of data) ==="]
        for key, st in self.metrics.items():
            if not st.get("n"):
                continue
            lines.append(
                f"{key:>12s}: RMSE {st['rmse']:.4f} | mean {st['mean']:+.4f} "
                f"| p95|e| {st['p95_abs']:.4f} | max {st['max']:.4f} "
                f"(N={st['n']})"
            )
        for key, st in self.nees.items():
            if not st.get("n"):
                continue
            verdict = "consistent" if st["consistent"] else "**INCONSISTENT**"
            lines.append(
                f"{key:>12s}: NEES {st['nees_mean']:.3f} "
                f"(CI {st['ci_low']:.3f}-{st['ci_high']:.3f}) -> {verdict}"
            )
        if self.gates:
            lines.append("-- gates --")
            for g in self.gates:
                mark = "PASS" if g.passed else "FAIL"
                lines.append(f"[{mark}] {g.name}: {g.detail}")
        overall = "ALL GATES PASSED" if self.all_gates_passed() else (
            "GATES FAILED" if self.gates else "(no gates configured)")
        lines.append(f"=> {overall}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "nees": self.nees,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
        }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class TrajectoryEvaluator:
    """Collects per-tick estimator output and produces an :class:`EvalReport`.

    Typical use inside an estimation loop::

        ev = TrajectoryEvaluator("eskf-tactical-automotive")
        for pt in traj.points:
            eskf.predict(a_m, w_m, dt)
            ...
            ev.add_pose(pt.t, eskf.state.p, pt.pos,
                        P=eskf.state.P[0:3, 0:3])   # optional, enables NEES
            ev.add_attitude_yaw(pt.t, yaw_est, yaw_true)
        report = ev.finalize()
        report.add_gate("pos_rmse", "<", 2.0)       # regression gates
        print(report.summary())
    """

    def __init__(self, name: str):
        self.name = name
        self._t0: Optional[float] = None
        self._t_end: Optional[float] = None
        self._pos = MetricStream("pos_err_m")
        self._vel = MetricStream("vel_err_ms")
        self._yaw = MetricStream("yaw_err_deg")
        self._nees_pos = NeesAccumulator(dim=2)   # horizontal position
        self._nees_pos3 = NeesAccumulator(dim=3)  # full position

    # -- collection --------------------------------------------------------

    def _tick(self, t: float) -> None:
        if self._t0 is None:
            self._t0 = t
        self._t_end = t

    def add_pose(self, t: float, est_p: np.ndarray, true_p: np.ndarray,
                 P: Optional[np.ndarray] = None,
                 dims: Tuple[int, ...] = (0, 1)) -> np.ndarray:
        """Add a position error sample.  ``P`` (2x2/3x3 block) enables NEES;
        ``dims`` selects which axes enter the scalar error (default xy)."""
        self._tick(t)
        e = np.asarray(true_p, dtype=float) - np.asarray(est_p, dtype=float)
        err_sel = float(np.linalg.norm(e[list(dims)]))
        self._pos.add(err_sel)
        if P is not None:
            Pv = np.asarray(P, dtype=float)
            k = len(dims)
            blk_idx = list(dims)
            blk = Pv[np.ix_(blk_idx, blk_idx)]
            self._nees_pos.add(e[list(dims)], blk)
            if k == 3 or min(dims) == 0 and max(dims) == 2:
                self._nees_pos3.add(e[[0, 1, 2]], Pv[[0, 1, 2]][:, [0, 1, 2]])
        return e

    def add_velocity(self, t: float, est_v: np.ndarray, true_v: np.ndarray,
                     dims: Tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
        self._tick(t)
        e = np.asarray(true_v, dtype=float) - np.asarray(est_v, dtype=float)
        self._vel.add(float(np.linalg.norm(e[list(dims)])))
        return e

    def add_yaw(self, t: float, est_yaw: float, true_yaw: float) -> float:
        """Yaw error in degrees, wrapped to [-180, 180]."""
        self._tick(t)
        d = (true_yaw - est_yaw + np.pi) % (2 * np.pi) - np.pi
        deg = float(np.degrees(d))
        self._yaw.add(deg)
        return deg

    # -- finalization ------------------------------------------------------

    def finalize(self) -> EvalReport:
        rep = EvalReport(name=self.name)
        if self._t0 is not None and self._t_end is not None:
            rep.duration_s = self._t_end - self._t0
        rep.metrics["pos_err_m"] = self._pos.result()
        rep.metrics["vel_err_ms"] = self._vel.result()
        rep.metrics["yaw_err_deg"] = self._yaw.result()
        rep.nees["nees_hpos"] = self._nees_pos.result()
        rep.nees["nees_pos3d"] = self._nees_pos3.result()
        return rep


def add_gate(report: EvalReport, metric: str, op: str,
             threshold: float) -> GateResult:
    """Attach a pass/fail gate on a reported statistic.

    ``op`` is one of ``"<"``, ``"<="``, ``">"``, ``">="`` applied to
    ``report.metric(metric, stat)`` where stat defaults to ``rmse``
    (use ``"metric@stat"`` syntax to pick another statistic, e.g.
    ``"pos_err_m@max"``).
    """
    stat = "rmse"
    key = metric
    ops = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
    }
    if metric.startswith("nees:"):
        # NEES statistics live in report.nees, not report.metrics
        parts = metric[5:].split("@", 1)
        nees_key = parts[0]
        stat = parts[1] if len(parts) > 1 else "nees_mean"
        st = report.nees.get(nees_key, {})
        val = st.get(stat) if isinstance(st, dict) else None
        src = f"nees={nees_key}@{stat}"
    else:
        if "@" in metric:
            key, stat = metric.split("@", 1)
        val = report.metric(key, stat)
        src = f"metric={key}@{stat}"
    if val is None or op not in ops:
        g = GateResult(metric, False, f"missing value ({src})")
    else:
        ok = bool(ops[op](val, threshold))
        g = GateResult(metric, ok, f"{val:.4f} {op} {threshold}")
    report.gates.append(g)
    return g


# ---------------------------------------------------------------------------
# Side-by-side comparison helper
# ---------------------------------------------------------------------------

def compare_reports(*reports: EvalReport) -> str:
    """Render a compact side-by-side RMSE table for several reports."""
    keys = ["pos_err_m", "vel_err_ms", "yaw_err_deg"]
    head = f"{'metric':>12s}" + "".join(f" | {r.name:>22s}" for r in reports)
    lines = [head, "-" * len(head)]
    for k in keys:
        row = f"{k:>12s}"
        for r in reports:
            v = r.metric(k, "rmse")
            row += f" | {'--' if v is None else f'{v:22.4f}'}"
        lines.append(row)
    lines.append("-" * len(head))
    row = f"{'gates':>12s}"
    for r in reports:
        ok = r.all_gates_passed()
        row += f" | {'FAIL' if r.gates and not ok else ('PASS' if r.gates else '--'):>22s}"
    lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monte-Carlo aggregation (v0.15.1)
# ---------------------------------------------------------------------------

class MonteCarloGate:
    """Aggregate pass/fail over MANY seeds -- gates on the distribution,
    not on one lucky/unlucky run.

    A single-seed gate silently overfits to that seed's random draw of
    sensor biases/noise (measured on parking-garage: 4/10 seeds passed a
    gate calibrated on seed 42 only).  Monte-Carlo gates fix this by
    requiring e.g. ``p90 of pos_err_m < 100 m``.

    Example::

        mc = MonteCarloGate("garage", seeds=range(42, 52))
        mc.add("pos_err_m", "<", 100.0, quantile=0.9)
        mc.add("nees:nees_hpos@nees_mean", "<", 50.0, quantile=0.9)
        for seed in mc.seeds:
            rep = run_estimator(seed)          # your loop
            mc.add_report(rep)
        verdict = mc.verdict()                 # aggregated pass/fail
        print(mc.summary())
    """

    def __init__(self, name: str, seeds):
        self.name = name
        self.seeds: List[int] = list(seeds)
        self._reports: Dict[int, EvalReport] = {}
        # spec_key -> (metric_spec, op, threshold, quantile)
        self._specs: Dict[str, Tuple[str, str, float, float]] = {}

    def add(self, metric: str, op: str, threshold: float,
            quantile: float = 0.9) -> None:
        """Declare one aggregate gate.

        ``quantile`` selects which point of the across-seed distribution
        must satisfy the threshold (0.5 = median, 0.95 = worst 5% may
        fail).  NEES metrics use the ``'nees:key@stat'`` syntax.
        """
        self._specs[metric] = (metric, op, threshold, float(quantile))

    def add_report(self, report: EvalReport) -> None:
        if report.name in self._reports:
            raise ValueError(f"duplicate report for '{report.name}'; "
                             "use unique names per run")
        self._reports[report.name] = report

    @property
    def complete(self) -> bool:
        return len(self._reports) >= len(self.seeds)

    def _value_for(self, report: EvalReport, metric_spec: str) -> Optional[float]:
        if metric_spec.startswith("nees:"):
            parts = metric_spec[5:].split("@", 1)
            key = parts[0]
            stat = parts[1] if len(parts) > 1 else "nees_mean"
            st = report.nees.get(key, {})
            return st.get(stat) if isinstance(st, dict) else None
        stat = "rmse"
        key = metric_spec
        if "@" in metric_spec:
            key, stat = metric_spec.split("@", 1)
        return report.metric(key, stat)

    def verdict(self) -> Dict[str, object]:
        """Evaluate all declared gates against the collected reports.

        Returns dict with per-gate quantile values and overall pass.
        Missing reports / metrics fail closed.
        """
        ops = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }
        gates_out = []
        all_ok = True
        for key, (spec, op, thr, q) in self._specs.items():
            vals = [self._value_for(r, spec) for r in self._reports.values()]
            vals = [v for v in vals if v is not None]
            n_expected = len(self.seeds)
            if len(vals) < n_expected or op not in ops:
                g = {"gate": key, "passed": False,
                     "detail": f"incomplete data ({len(vals)}/{n_expected})"
                               + (f" or bad op '{op}'" if op not in ops else "")}
                all_ok = False
            else:
                qv = float(np.quantile(vals, q))
                ok = bool(ops[op](qv, thr))
                all_ok &= ok
                g = {"gate": key, "passed": ok,
                     "detail": f"p{int(q*100)}={qv:.4g} {op} {thr:g} "
                               f"(n={len(vals)})"}
            gates_out.append(g)
        return {"name": self.name, "passed": bool(all_ok),
                "n_runs": len(self._reports), "gates": gates_out}

    def summary(self) -> str:
        v = self.verdict()
        lines = [f"=== Monte-Carlo gate: {self.name} "
                 f"({v['n_runs']} runs) ==="]
        for g in v["gates"]:
            mark = "PASS" if g["passed"] else "FAIL"
            lines.append(f"[{mark}] {g['gate']}: {g['detail']}")
        lines.append("=> " + ("PASS ✅" if v["passed"] else "FAIL ❌"))
        return "\n".join(lines)
