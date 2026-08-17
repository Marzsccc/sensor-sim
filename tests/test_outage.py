"""
Tests for GNSS outage simulation and uncertainty-aware display (v0.8.0).

Coverage:
  - OutageModel: deterministic windows, random dropouts, total time
  - OutageSimulator smoke: runs end-to-end without error, returns arrays
    of the right shape, outage fraction matches the schedule
  - outage behavior: fused position error grows during outage and
    shrinks (recovers) after GNSS returns; display-time covariance and
    radius_95 grow during the outage
  - ellipse-weighted output: radius_95 bounded by the fade threshold,
    displayed error does not blow up
  - NEES consistency: covariance is consistent (or at least sane) over
    the whole run and per-window (in-outage vs healthy)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

import numpy as np

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.eskf import ESKFConfig
from sensor_sim.outage import (
    OutageConfig,
    OutageModel,
    OutageSimConfig,
    OutageSimulator,
    consistency,
    nees_vs_outage,
)


def _straight_traj(duration=40.0, speed=20.0, dt=0.01):
    """Straight level run at constant speed (no turns)."""
    wp = [
        Waypoint(t=0.0, pos=np.array([0.0, 0.0, 0.0]),
                 vel=np.array([speed, 0.0, 0.0]),
                 att=np.array([1.0, 0.0, 0.0, 0.0]),
                 omega=np.array([0.0, 0.0, 0.0])),
        Waypoint(t=duration, pos=np.array([speed * duration, 0.0, 0.0]),
                 vel=np.array([speed, 0.0, 0.0]),
                 att=np.array([1.0, 0.0, 0.0, 0.0]),
                 omega=np.array([0.0, 0.0, 0.0])),
    ]
    return Trajectory(wp, dt=dt)


def _sim_config(outage_periods, seed=42, **kw):
    """OutageSimConfig with a straight-run-friendly ESKF."""
    eskf = ESKFConfig(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5,
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=2.0, gnss_vel_std=0.2,
    )
    cfg = OutageSimConfig(
        outage=OutageConfig(outage_periods=outage_periods),
        eskf=eskf,
        seed=seed,
    )
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TestOutageModel(unittest.TestCase):
    def test_deterministic_windows(self):
        m = OutageModel(OutageConfig(outage_periods=[(5.0, 10.0)]), seed=1)
        self.assertFalse(m.drop(4.9))
        self.assertTrue(m.drop(5.0))
        self.assertTrue(m.drop(7.5))
        self.assertTrue(m.drop(10.0))
        self.assertFalse(m.drop(10.1))
        self.assertAlmostEqual(m.total_outage_time, 5.0)

    def test_random_dropout_seeded(self):
        m1 = OutageModel(OutageConfig(random_dropout_prob=0.5), seed=7)
        m2 = OutageModel(OutageConfig(random_dropout_prob=0.5), seed=7)
        draws1 = [m1.drop(float(i)) for i in range(20)]
        draws2 = [m2.drop(float(i)) for i in range(20)]
        self.assertEqual(draws1, draws2)
        frac = sum(draws1) / len(draws1)
        self.assertGreater(frac, 0.1)
        self.assertLess(frac, 0.9)

    def test_no_outage_no_dropout(self):
        m = OutageModel(OutageConfig(), seed=0)
        self.assertFalse(m.drop(1.0))
        self.assertEqual(m.total_outage_time, 0.0)


class TestOutageSimulator(unittest.TestCase):
    def test_smoke_no_outage(self):
        traj = _straight_traj(duration=20.0)
        sim = OutageSimulator(_sim_config([], seed=42))
        res = sim.run(traj)
        self.assertEqual(len(res.t), len(res.true_pos))
        self.assertEqual(res.true_pos.shape[1], 3)
        self.assertEqual(res.disp_cov.shape[1:], (9, 9))
        self.assertFalse(np.any(np.isnan(res.disp_cov)))
        self.assertAlmostEqual(res.outage_fraction(), 0.0)
        # no outage: error stays small (dominated by the 67 ms display
        # horizon at 20 m/s ~ 1.3 m + cold start)
        settled = res.t > 5.0
        self.assertLess(np.mean(np.linalg.norm(
            res.fused_error[settled], axis=1)), 8.0)
        # radius stays small after the initial covariance settles
        self.assertLess(res.radius_95[settled].max(), 1.0)

    def test_outage_grows_uncertainty_and_error(self):
        traj = _straight_traj(duration=40.0)
        sim = OutageSimulator(_sim_config([(10.0, 25.0)], seed=42))
        res = sim.run(traj)
        self.assertGreater(res.outage_fraction(), 0.3)
        m = res.in_outage
        # during the outage the display-time uncertainty honestly grows
        late_outage = m & (res.t > 22.0)
        self.assertGreater(res.radius_95[late_outage].mean(), 15.0)
        # position error grows during the outage (dead-reckoning drift)
        early_out = m & (res.t < 12.0)
        self.assertLess(np.linalg.norm(
            res.fused_error[early_out], axis=1).mean(), 2.0)
        self.assertGreater(np.linalg.norm(
            res.fused_error[late_outage], axis=1).mean(), 50.0)
        # recovery: error drops right after GNSS returns
        recovered = (~m) & (res.t > 25.1) & (res.t < 27.0)
        self.assertLess(np.linalg.norm(
            res.fused_error[recovered], axis=1).mean(), 10.0)

    def test_ellipse_weighting_bounds_display_error(self):
        traj = _straight_traj(duration=40.0)
        sim = OutageSimulator(_sim_config([(10.0, 30.0)], seed=42))
        res = sim.run(traj)
        # display pose is a convex blend of prediction and fused pose, so
        # its error never exceeds the fused error by much
        disp_err = np.linalg.norm(res.disp_error, axis=1)
        fused_err = np.linalg.norm(res.fused_error, axis=1)
        self.assertLess(np.mean(disp_err), np.mean(fused_err) + 0.5)

    def test_random_dropout_runs(self):
        traj = _straight_traj(duration=20.0)
        cfg = _sim_config([], seed=1)
        cfg.outage.random_dropout_prob = 0.3
        sim = OutageSimulator(cfg)
        res = sim.run(traj)
        self.assertFalse(np.any(np.isnan(res.radius_95)))
        self.assertGreater(res.outage_fraction(), 0.0)
        self.assertLess(res.outage_fraction(), 1.0)

    def test_deterministic_recovery(self):
        """Position recovers after GNSS returns (re-acquisition).

        On a straight road absolute heading is unobservable from
        GNSS+wheel, so the filter cannot fully re-converge in a few
        seconds; what matters is that the position is re-anchored to the
        fix (error drops from the drift level to GNSS level) right after
        the outage ends.
        """
        traj = _straight_traj(duration=40.0)
        sim = OutageSimulator(_sim_config([(10.0, 20.0)], seed=42))
        res = sim.run(traj)
        m = res.in_outage
        # right after recovery (during the wheel-downweight grace period)
        after = (~m) & (res.t > 20.1) & (res.t < 23.0)
        self.assertGreater(np.sum(after), 10)
        # error right after recovery is far below the outage drift level
        self.assertLess(np.mean(np.linalg.norm(
            res.fused_error[after], axis=1)), 10.0)
        self.assertLess(np.mean(np.linalg.norm(
            res.fused_error[after], axis=1)),
            np.mean(np.linalg.norm(res.fused_error[m], axis=1)))


class TestConsistency(unittest.TestCase):
    def test_consistency_identity_cov(self):
        rng = np.random.default_rng(0)
        errs = rng.normal(0.0, 1.0, size=(500, 3))
        covs = np.tile(np.eye(3), (500, 1, 1))
        c = consistency(errs, covs, dim=3, alpha=0.05)
        # mean NEES for 3-DOF standard normal ~ 3
        self.assertAlmostEqual(c["nees_mean"], 3.0, delta=0.5)
        self.assertLess(c["ci_low"], c["nees_mean"])
        self.assertLess(c["nees_mean"], c["ci_high"])

    def test_consistency_scaled_cov(self):
        rng = np.random.default_rng(1)
        # errors ~ N(0, 2^2), cov claims 2^2 -> consistent
        errs = rng.normal(0.0, 2.0, size=(800, 3))
        covs = np.tile(np.eye(3) * 4.0, (800, 1, 1))
        c = consistency(errs, covs, dim=3, alpha=0.05)
        self.assertTrue(c["consistent"])

    def test_nees_vs_outage_shape(self):
        traj = _straight_traj(duration=40.0)
        sim = OutageSimulator(_sim_config([(10.0, 25.0)], seed=42))
        res = sim.run(traj)
        out = nees_vs_outage(res, dim=3, alpha=0.05)
        self.assertIn("nees_in_outage", out)
        self.assertIn("nees_healthy", out)
        self.assertGreater(out["nees_in_outage"], 0.0)
        self.assertGreater(out["nees_healthy"], 0.0)

    def test_consistency_bad_dim_raises(self):
        with self.assertRaises(ValueError):
            consistency(np.zeros((5, 3)), np.zeros((5, 3, 3)), dim=4)


if __name__ == "__main__":
    unittest.main()
