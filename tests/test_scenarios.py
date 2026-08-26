"""Tests for sensor_sim.scenarios (v0.15.0)."""

import numpy as np
import pytest

from sensor_sim.imu import SensorGrade
from sensor_sim.gnss import GNSSGrade
from sensor_sim.scenarios import (
    Scenario,
    eskf_config_for,
    default_library,
    run_scenario,
    run_scenario_mc,
    batch_summary,
    scenario_highway_straight,
    scenario_parking_garage,
)


class TestConfigPresets:

    def test_config_matches_grades(self):
        cfg = eskf_config_for(SensorGrade.TACTICAL, GNSSGrade.RTK)
        assert cfg.gnss_pos_std == pytest.approx(0.15)

    def test_consumer_looser_than_rtk(self):
        c_cons = eskf_config_for(SensorGrade.CONSUMER, GNSSGrade.CONSUMER)
        c_rtk = eskf_config_for(SensorGrade.NAVIGATION, GNSSGrade.RTK)
        assert c_cons.gnss_pos_std > c_rtk.gnss_pos_std
        assert c_cons.acc_noise_density > c_rtk.acc_noise_density


class TestLibrary:

    def test_library_complete(self):
        lib = default_library()
        names = [s.name for s in lib]
        assert len(lib) == 5
        assert "highway-straight" in names
        assert "tunnel-outage" in names
        assert "parking-garage" in names
        assert "rtk-baseline" in names

    def test_tunnel_has_outage(self):
        sc = [s for s in default_library() if s.name == "tunnel-outage"][0]
        assert len(sc.outage_periods) == 1
        a, b = sc.outage_periods[0]
        assert b > a and a > 0

    def test_scenario_duration(self):
        sc = scenario_highway_straight()
        assert sc.duration == pytest.approx(40.0)


class TestRunner:

    def test_highway_passes(self):
        rep = run_scenario(scenario_highway_straight())
        assert rep.metric("pos_err_m") is not None
        # open-sky straight + automotive GNSS must be well inside gate
        assert rep.all_gates_passed(), rep.summary()

    def test_rtk_tighter_than_automotive(self):
        r_auto = run_scenario(
            [s for s in default_library() if s.name == "urban-s-curve"][0])
        r_rtk = run_scenario(
            [s for s in default_library() if s.name == "rtk-baseline"][0])
        assert r_rtk.metric("pos_err_m") < r_auto.metric("pos_err_m")

    def test_seed_reproducibility(self):
        s1 = run_scenario(scenario_highway_straight(), seed=7)
        s2 = run_scenario(scenario_highway_straight(), seed=7)
        assert (s1.metric("pos_err_m")
                == pytest.approx(s2.metric("pos_err_m")))
        s3 = run_scenario(scenario_highway_straight(), seed=8)
        assert (s1.metric("pos_err_m")
                != pytest.approx(s3.metric("pos_err_m"), rel=1e-6))

    def test_outage_worse_than_open_sky(self):
        straight = scenario_highway_straight()
        tunnel = scenario_highway_straight()
        tunnel.name = "tmp-tunnel"
        tunnel.outage_periods = [(14.0, 26.0)]
        r_open = run_scenario(straight)
        r_blk = run_scenario(tunnel)
        assert (r_blk.metric("pos_err_m", "max")
                >= r_open.metric("pos_err_m", "max"))


class TestMonteCarloGates:

    def test_mc_passes_on_distribution(self):
        """The garage corner must pass on the across-seed distribution."""
        sc = scenario_parking_garage()
        mc = run_scenario_mc(sc, seeds=range(42, 48))   # 6 seeds, faster
        v = mc.verdict()
        assert v["passed"], mc.summary()
        assert v["n_runs"] == 6
        for g in v["gates"]:
            assert g["passed"], g

    def test_mc_fails_closed_on_incomplete(self):
        from sensor_sim.evaluation import MonteCarloGate
        mc = MonteCarloGate("partial", seeds=range(42, 45))
        mc.add("pos_err_m", "<", 100.0, quantile=0.9)
        # no reports added at all -> fail closed, never silent pass
        v = mc.verdict()
        assert not v["passed"]


class TestBatchSummary:

    def test_matrix_rendering(self):
        reports = [run_scenario(s) for s in
                   [scenario_highway_straight(), scenario_parking_garage()]]
        text = batch_summary(reports)
        assert "highway-straight" in text
        assert "parking-garage" in text
        assert "/2 scenarios" in text

    def test_full_library_runs(self):
        """The whole point: every scenario in the library must execute."""
        reports = []
        for sc in default_library():
            try:
                reports.append(run_scenario(sc))
            except Exception as e:  # pragma: no cover
                pytest.fail(f"scenario {sc.name} crashed: {e}")
        assert len(reports) == 5
        text = batch_summary(
            reports,
            display_rows=[(r, None) for r in reports])
        # single-run rows may FAIL on stochastic corners; the matrix must
        # still render and count them honestly
        assert "scenarios passed" in text
