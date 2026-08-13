"""
Tests for the ESKF GNSS+IMU loose-coupling filter.

Coverage:
  - state initialization and covariance layout
  - pure IMU prediction keeps the filter stable (no NaN, bounded growth)
  - GNSS update reduces position error on a straight line
  - full trajectory fusion beats pure dead reckoning
  - bias estimation converges (accel bias estimable on level straight run
    thanks to velocity updates; gyro bias estimable during turns)
  - outlier rejection gate
"""

import unittest

import numpy as np

from sensor_sim.eskf import ESKF, ESKFConfig, _quat_exp, _quat_log
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade
from sensor_sim.utils import quat_multiply, quat_to_rotmat, quat_to_euler


def _straight_traj(dt=0.01, T=10.0, v=10.0):
    """Straight level flight at constant speed."""
    wp = [
        Waypoint(t=0.0, pos=np.array([0, 0, 0]), vel=np.array([v, 0, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
        Waypoint(t=T, pos=np.array([v * T, 0, 0]), vel=np.array([v, 0, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
    ]
    return Trajectory(wp, dt=dt)


class TestESKFBasics(unittest.TestCase):
    def test_init_state(self):
        cfg = ESKFConfig()
        eskf = ESKF(cfg)
        eskf.set_initial_state(0.0, [1, 2, 3], [0, 0, 0], [1, 0, 0, 0])
        np.testing.assert_allclose(eskf.position, [1, 2, 3])
        self.assertEqual(eskf.state.P.shape, (15, 15))
        self.assertEqual(eskf.state.n_updates, 0)

    def test_quat_exp_log_roundtrip(self):
        for v in [np.array([0.1, 0.2, -0.3]), np.array([1e-9, 0, 0]),
                  np.zeros(3), np.array([0.5, -0.5, 0.5])]:
            q = _quat_exp(v)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0)
            v2 = _quat_log(q)
            np.testing.assert_allclose(v2, v, atol=1e-9)

    def test_imu_only_predict_stable(self):
        """Pure IMU integration (no update) must not blow up in 5 s."""
        traj = _straight_traj(T=5.0)
        imu = IMUSensor(SensorGrade.CONSUMER, dt=0.01, seed=7)
        eskf = ESKF(ESKFConfig())
        eskf.set_initial_state(0.0, traj.positions[0], traj.velocities[0], traj.attitudes[0])
        for pt in traj.points:
            a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
            eskf.predict(a_m, w_m, 0.01)
        self.assertTrue(np.all(np.isfinite(eskf.state.P)))
        self.assertTrue(np.all(np.isfinite(eskf.position)))
        # error must remain bounded (pure DR, 5 s, consumer IMU)
        e = np.linalg.norm(eskf.position - traj.positions[-1])
        self.assertLess(e, 50.0)


class TestESKFUpdate(unittest.TestCase):
    def test_gnss_update_reduces_error(self):
        """Straight run: GNSS updates keep the solution near truth."""
        traj = _straight_traj(T=10.0, v=10.0)
        imu = IMUSensor(SensorGrade.TACTICAL, dt=0.01, seed=3)
        gnss = GNSSSensor(GNSSGrade.RTK, dt=0.01, seed=4)
        eskf = ESKF(ESKFConfig(gnss_pos_std=0.3, gnss_vel_std=0.05))
        p0 = traj.points[0]
        eskf.set_initial_state(p0.t, p0.pos + np.array([5, 5, 2]), p0.vel, p0.att)

        gnss_next = 0.2
        max_err = 0.0
        max_err_settled = 0.0
        for pt in traj.points:
            a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
            eskf.predict(a_m, w_m, 0.01)
            pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
            if valid:
                eskf.update_gnss(pos_m, vel_m)
            e = np.linalg.norm(eskf.position - pt.pos)
            max_err = max(max_err, e)
            if pt.t > 1.0:      # skip initial transient
                max_err_settled = max(max_err_settled, e)
        self.assertLess(max_err_settled, 2.0)   # converged <2 m
        self.assertGreater(eskf.state.n_updates, 0)

    def test_full_fusion_beats_dead_reckoning(self):
        """ESKF with updates must beat pure integration on a turning run."""
        wp = [
            Waypoint(t=0.0, pos=np.array([0, 0, 0]), vel=np.array([10, 0, 0]),
                     att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
            Waypoint(t=10.0, pos=np.array([100, 0, 0]), vel=np.array([10, 0, 0]),
                     att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
            Waypoint(t=20.0, pos=np.array([100, 100, 0]), vel=np.array([0, 10, 0]),
                     att=np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]),
                     omega=np.array([0, 0, 0])),
        ]
        traj = Trajectory(wp, dt=0.01)
        imu = IMUSensor(SensorGrade.CONSUMER, dt=0.01, seed=11)
        gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=0.01, seed=12)
        eskf = ESKF(ESKFConfig())
        p0 = traj.points[0]
        eskf.set_initial_state(p0.t, p0.pos, p0.vel, p0.att)

        # dead reckoning
        dr = traj.positions[0].copy()
        drv = traj.velocities[0].copy()
        drq = traj.attitudes[0].copy()

        eskf_err = 0.0
        dr_err = 0.0
        for pt in traj.points:
            a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
            eskf.predict(a_m, w_m, 0.01)
            pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
            if valid:
                eskf.update_gnss(pos_m, vel_m)
            eskf_err = max(eskf_err, np.linalg.norm(eskf.position - pt.pos))

            # DR
            drq = quat_multiply(drq, _quat_exp(w_m * 0.01)); drq /= np.linalg.norm(drq)
            R = quat_to_rotmat(drq)
            drv += (R @ (a_m) + np.array([0, 0, -9.80665])) * 0.01
            dr += drv * 0.01
            dr_err = max(dr_err, np.linalg.norm(dr - pt.pos))

        self.assertLess(eskf_err, dr_err * 0.5)

    def test_bias_estimation(self):
        """On a straight level run with velocity updates, accel bias estimate
        should move toward truth (identifiability requires motion/velocity)."""
        traj = _straight_traj(T=20.0, v=10.0)
        imu = IMUSensor(SensorGrade.TACTICAL, dt=0.01, seed=5)
        gnss = GNSSSensor(GNSSGrade.RTK, dt=0.01, seed=6)
        eskf = ESKF(ESKFConfig(gnss_pos_std=0.2, gnss_vel_std=0.05,
                               acc_bias_rw=1e-5, gyr_bias_rw=1e-6))
        p0 = traj.points[0]
        eskf.set_initial_state(p0.t, p0.pos, p0.vel, p0.att)

        true_b = imu.get_errors()["bias_acc"]
        for pt in traj.points:
            a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
            eskf.predict(a_m, w_m, 0.01)
            pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
            if valid:
                eskf.update_gnss(pos_m, vel_m)
        est_b = eskf.estimated_biases()[0]
        # bias estimate should be much closer to truth than 0
        err0 = np.linalg.norm(true_b)
        err_est = np.linalg.norm(est_b - true_b)
        self.assertLess(err_est, err0)

    def test_outlier_gate(self):
        """A 100 m GNSS jump must be rejected (innovation gate)."""
        eskf = ESKF(ESKFConfig())
        eskf.set_initial_state(0.0, [0, 0, 0], [0, 0, 0], [1, 0, 0, 0])
        n0 = eskf.state.n_updates
        eskf.update_gnss(np.array([0.1, 0.1, 0.1]))
        self.assertEqual(eskf.state.n_updates, n0 + 1)
        eskf.update_gnss(np.array([1000, 0, 0]))   # 1000 m jump
        self.assertEqual(eskf.state.n_updates, n0 + 1)   # rejected
        self.assertLess(np.linalg.norm(eskf.position), 1.0)


if __name__ == "__main__":
    unittest.main()
