"""Tests for sensor_sim.evaluation (v0.14.0)."""

import numpy as np
import pytest

from sensor_sim.evaluation import (
    NeesAccumulator,
    MetricStream,
    TrajectoryEvaluator,
    EvalReport,
    add_gate,
    compare_reports,
)


# ---------------------------------------------------------------------------
# NeesAccumulator
# ---------------------------------------------------------------------------

class TestNeesAccumulator:

    def test_consistent_estimator(self):
        """Errors drawn from the reported covariance must be consistent."""
        rng = np.random.default_rng(0)
        acc = NeesAccumulator(dim=2)
        P_true = np.array([[1.0, 0.2], [0.2, 0.8]])
        for _ in range(2000):
            e = rng.multivariate_normal(np.zeros(2), P_true)
            acc.add(e, P_true)
        r = acc.result()
        assert r["n"] == 2000
        assert r["consistent"], f"NEES {r['nees_mean']:.3f} outside " \
                                f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]"
        # For chi2(2): mean NEES should be near 2
        assert 1.5 < r["nees_mean"] < 2.5

    def test_overconfident_detected(self):
        """Reported covariance too small -> NEES blows up -> inconsistent."""
        rng = np.random.default_rng(1)
        acc = NeesAccumulator(dim=2)
        for _ in range(1000):
            e = rng.multivariate_normal(
                np.zeros(2), np.diag([4.0, 4.0]))
            acc.add(e, np.eye(2))          # claims 1 m^2, truth is 4 m^2
        r = acc.result()
        assert not r["consistent"]
        assert r["nees_mean"] > r["ci_high"]

    def test_bad_dim_raises(self):
        with pytest.raises(ValueError):
            NeesAccumulator(dim=5)


class TestMetricStream:

    def test_stats(self):
        ms = MetricStream("x")
        for v in [3.0, -4.0, 0.0]:
            ms.add(v)
        r = ms.result()
        assert r["rmse"] == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
        assert r["max"] == pytest.approx(4.0)
        assert r["n"] == 3

    def test_empty(self):
        assert MetricStream("e").result() == {"n": 0}


# ---------------------------------------------------------------------------
# TrajectoryEvaluator end-to-end
# ---------------------------------------------------------------------------

def _fake_run(ev: TrajectoryEvaluator, seed: int = 0, n: int = 500):
    """Synthesize a plausible estimator run: random-walk truth + noisy est."""
    rng = np.random.default_rng(seed)
    p_true = np.zeros(3)
    p_est = np.array([0.5, -0.5, 0.0])
    yaw_true = 0.0
    yaw_est = np.radians(2.0)
    for i in range(n):
        t = i * 0.01
        p_true += rng.normal(0, 0.01, 3)
        p_est = p_est + (p_true - p_est) * 0.05 + rng.normal(0, 0.02, 3)
        yaw_true += rng.normal(0, 0.002)
        yaw_est += (yaw_true - yaw_est) * 0.1 + rng.normal(0, 0.003)

        ev.add_pose(t, p_est, p_true, P=np.diag([0.04, 0.04, 0.0016]),
                    dims=(0, 1))
        ev.add_velocity(t, p_est * 0.1, p_true * 0.1)
        ev.add_yaw(t, yaw_est, yaw_true)


class TestTrajectoryEvaluator:

    def test_full_pipeline(self):
        ev = TrajectoryEvaluator("unit-test")
        _fake_run(ev)
        rep = ev.finalize()

        assert rep.duration_s == pytest.approx(4.99, abs=0.01)
        assert rep.metric("pos_err_m") is not None
        assert rep.metrics["pos_err_m"]["n"] == 500
        # estimator tracks the walk -> sub-decimeter RMSE expected here
        assert rep.metric("pos_err_m") < 1.0

        nees = rep.nees["nees_hpos"]
        assert nees["n"] == 500

    def test_yaw_wrapping(self):
        ev = TrajectoryEvaluator("wrap")
        ev.add_yaw(0.0, np.radians(179), np.radians(-179))
        d = ev._yaw._values[-1]     # smallest wrap is +2 deg
        assert 0 < d <= 2.0

    def test_gates(self):
        ev = TrajectoryEvaluator("gates")
        _fake_run(ev)
        rep = ev.finalize()
        g1 = add_gate(rep, "pos_err_m", "<", 10.0)   # loose -> pass
        g2 = add_gate(rep, "pos_err_m", "<", 0.0000001)  # impossible -> fail
        g3 = add_gate(rep, "pos_err_m@max", "<=", 100.0)
        assert g1.passed and not g2.passed and g3.passed
        assert rep.all_gates_passed() is False
        assert "GATES FAILED" in rep.summary()
        assert "[FAIL]" in rep.summary()

    def test_summary_and_dict(self):
        ev = TrajectoryEvaluator("render")
        _fake_run(ev, n=50)
        rep = ev.finalize()
        s = rep.summary()
        assert "pos_err_m" in s and "NEES" in s.upper()
        d = rep.to_dict()
        assert d["name"] == "render"
        assert "nees_hpos" in d["nees"]

    def test_compare_reports(self):
        a, b = TrajectoryEvaluator("a"), TrajectoryEvaluator("b")
        _fake_run(a, seed=1)
        _fake_run(b, seed=2)
        ra, rb = a.finalize(), b.finalize()
        table = compare_reports(ra, rb)
        assert "a" in table and "b" in table
        assert "pos_err_m" in table


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

class TestGateOnMissingMetric:

    def test_missing_metric_gate_fails_closed(self):
        rep = EvalReport(name="empty")
        g = add_gate(rep, "does_not_exist", "<", 1.0)
        assert not g.passed          # fail-closed, never silently pass
