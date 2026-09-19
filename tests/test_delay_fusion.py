"""Unit tests for out-of-sequence / delayed measurement fusion (v0.22.0)."""

import numpy as np
import pytest

from sensor_sim.delay_fusion import (
    CVModel,
    DelayedFusionFilter,
    DelayedMeasurement,
    nees_position,
    position_covariance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_meas(model, t_meas, t_recv, p, sigma, sensor="gps"):
    H, R = model.observe_position(sigma)
    return DelayedMeasurement(t_meas=t_meas, t_recv=t_recv, z=np.asarray(p, float),
                              H=H, R=R, sensor=sensor)


def _fresh_filter(model, *, mode="oosm", history_s=5.0, t0=0.0, x0=None, P0=None):
    n = model.n_dim
    if x0 is None:
        x0 = np.zeros(model.state_dim)
    if P0 is None:
        P0 = np.eye(model.state_dim) * 100.0
    return DelayedFusionFilter(model, x0, P0, t0=t0, mode=mode, history_s=history_s)


def _run_in_order(model, meas_list, t_final):
    f = _fresh_filter(model)
    for m in meas_list:
        f.add_measurement(m)
    f.predict_to(t_final)
    return f


# ---------------------------------------------------------------------------
# CVModel
# ---------------------------------------------------------------------------
class TestCVModel:
    def test_state_dim(self):
        assert CVModel(n_dim=2).state_dim == 4
        assert CVModel(n_dim=3).state_dim == 6

    def test_F_matches_constant_velocity(self):
        F = CVModel(n_dim=2).F(0.5)
        np.testing.assert_allclose(F, [[1, 0, 0.5, 0],
                                       [0, 1, 0, 0.5],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])

    def test_zero_dt_is_identity_and_zero_noise(self):
        m = CVModel(n_dim=2)
        np.testing.assert_allclose(m.F(0.0), np.eye(4))
        np.testing.assert_allclose(m.Q(0.0), np.zeros((4, 4)))

    def test_Q_symmetric_psd_and_grows_with_dt(self):
        m = CVModel(n_dim=3, accel_psd=2.0)
        Q1, Q2 = m.Q(0.1), m.Q(0.2)
        np.testing.assert_allclose(Q1, Q1.T)
        np.testing.assert_allclose(Q2, Q2.T)
        assert np.all(np.linalg.eigvalsh(Q1) >= -1e-12)
        assert np.trace(Q2) > np.trace(Q1)

    def test_Q_matches_closed_form_blocks(self):
        q, dt = 1.7, 0.3
        m = CVModel(n_dim=1, accel_psd=q)
        Q = m.Q(dt)
        assert Q[0, 0] == pytest.approx(q * dt ** 3 / 3)
        assert Q[0, 1] == pytest.approx(q * dt ** 2 / 2)
        assert Q[1, 1] == pytest.approx(q * dt)

    def test_observe_position(self):
        H, R = CVModel(n_dim=2).observe_position(sigma=0.5)
        assert H.shape == (2, 4)
        np.testing.assert_allclose(R, np.eye(2) * 0.25)
        assert H[0, 0] == 1.0 and H[1, 1] == 1.0 and H[0, 2] == 0.0


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
class TestBasics:
    def test_in_order_update_shrinks_covariance(self):
        m = CVModel(n_dim=2, accel_psd=1.0)
        f = _fresh_filter(m, P0=np.eye(4) * 25.0)
        trace0 = np.trace(f.P)
        f.add_measurement(_pos_meas(m, 0.1, 0.1, [1.0, 0.0], 0.3))
        assert np.trace(f.P) < trace0

    def test_tracks_position_close_to_measurements(self):
        m = CVModel(n_dim=2, accel_psd=0.5)
        meas, t_final = [], 0.0
        for k in range(1, 41):
            t = 0.1 * k
            p = np.array([2.0 * t, -1.0 * t])
            meas.append(_pos_meas(m, t, t, p, 0.2))
            t_final = t
        f = _run_in_order(m, meas, t_final)
        np.testing.assert_allclose(f.position, [2.0 * t_final, -1.0 * t_final], atol=0.15)

    def test_in_order_keeps_moving_forward(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m)
        f.add_measurement(_pos_meas(m, 0.1, 0.1, [1.0, 1.0], 0.5))
        f.predict_to(0.5)
        assert f.t == pytest.approx(0.5)

    def test_predict_to_same_time_is_noop(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m, x0=np.array([1.0, 2.0, 3.0, 4.0]))
        x_before = f.x.copy()
        f.predict_to(0.0)
        np.testing.assert_allclose(f.x, x_before)


# ---------------------------------------------------------------------------
# Exactness: reprocessing == in-order
# ---------------------------------------------------------------------------
class TestExactness:
    def test_out_of_order_matches_in_order(self):
        rng = np.random.default_rng(7)
        m = CVModel(n_dim=2, accel_psd=1.5)
        t_grid = 0.1 * np.arange(1, 9)  # 0.1 .. 0.8
        # Variable per-sample delay so arrival order != validity order.
        delays = rng.uniform(0.05, 0.6, size=t_grid.size)
        meas = []
        for t, d in zip(t_grid, delays):
            p = np.array([20.0 * np.sin(0.4 * t), 20.0 * np.cos(0.4 * t)])
            p = p + rng.normal(0.0, 0.3, size=2)
            meas.append(_pos_meas(m, t, t + d, p, 0.3))

        t_final = t_grid[-1]
        ref = _run_in_order(m, meas, t_final)

        # Feed the same measurements in arrival order (t_recv non-decreasing).
        shuffled = sorted(meas, key=lambda mm: mm.t_recv)
        assert all(shuffled[i].t_recv <= shuffled[i + 1].t_recv for i in range(len(shuffled) - 1))
        f = _fresh_filter(m)
        for mm in shuffled:
            f.add_measurement(mm)
        f.predict_to(t_final)

        np.testing.assert_allclose(f.x, ref.x, atol=1e-9)
        np.testing.assert_allclose(f.P, ref.P, atol=1e-9)
        assert f.diag.n_out_of_order > 0

    def test_reverse_order_matches_in_order(self):
        m = CVModel(n_dim=2, accel_psd=1.0)
        delay = 0.5
        meas = []
        for k in range(1, 7):
            t = 0.1 * k
            p = np.array([3.0 * t, 3.0 * t ** 2])
            meas.append(_pos_meas(m, t, t + delay, p, 0.25))
        t_final = 0.6
        ref = _run_in_order(m, meas, t_final)
        f = _fresh_filter(m)
        for mm in reversed(meas):  # arrival order = reverse of validity order
            f.add_measurement(mm)
        f.predict_to(t_final)
        np.testing.assert_allclose(f.x, ref.x, atol=1e-9)
        np.testing.assert_allclose(f.P, ref.P, atol=1e-9)

    def test_duplicate_times_stay_stable(self):
        m = CVModel(n_dim=2)
        meas = [
            _pos_meas(m, 0.2, 0.4, [2.0, 0.0], 0.3, sensor="a"),
            _pos_meas(m, 0.2, 0.4, [2.0, 0.0], 0.3, sensor="b"),
        ]
        f = _fresh_filter(m)
        for mm in meas:
            f.add_measurement(mm)
        assert np.all(np.isfinite(f.x)) and np.all(np.isfinite(f.P))


# ---------------------------------------------------------------------------
# Diagnostics / history management
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_counters(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m)
        f.add_measurement(_pos_meas(m, 0.3, 0.5, [1.0, 0.0], 0.3))  # in order
        f.add_measurement(_pos_meas(m, 0.1, 0.6, [0.0, 0.0], 0.3))  # out of order
        assert f.diag.n_received == 2
        assert f.diag.n_accepted == 2
        assert f.diag.n_out_of_order == 1
        assert f.diag.max_rewind_s == pytest.approx(0.2, abs=1e-9)
        assert f.diag.n_reprocessed >= 2

    def test_receive_time_mode_marks_everything_accepted(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m, mode="receive_time")
        f.add_measurement(_pos_meas(m, 0.1, 0.3, [1.0, 1.0], 0.3))
        f.add_measurement(_pos_meas(m, 0.2, 0.4, [2.0, 2.0], 0.3))  # would be OOO in oosm mode
        assert f.diag.n_out_of_order == 0
        assert f.t == pytest.approx(0.4)

    def test_pruning_keeps_latest_state_usable(self):
        m = CVModel(n_dim=2, accel_psd=1.0)
        f = _fresh_filter(m, history_s=0.5)
        for k in range(1, 21):
            t = 0.1 * k
            f.add_measurement(_pos_meas(m, t, t, [t, t], 0.2))
        assert np.all(np.isfinite(f.x))
        assert np.all(np.isfinite(f.P))
        assert f.diag.n_pruned > 0

    def test_old_out_of_order_beyond_history_raises(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m, history_s=0.2)
        f.add_measurement(_pos_meas(m, 1.0, 1.0, [1.0, 1.0], 0.3))
        with pytest.raises(RuntimeError):
            f.add_measurement(_pos_meas(m, 0.1, 1.1, [0.0, 0.0], 0.3))

    def test_history_survives_repeated_out_of_order_updates(self):
        """Regression: reprocessing must not amputate older snapshots.

        The first implementation rebuilt ``self._snapshots`` from the anchor
        onward, so every late sample silently discarded the prefix.  After a
        long run of rewinds the history collapsed to a couple of snapshots
        and the next rewind raised ``RuntimeError``.

        Semantics note: the filter clock is the *newest fused measurement
        time*, so a stream that is merely delayed but still monotone in
        validity time never rewinds (see
        ``test_uniformly_delayed_monotone_stream_needs_no_rewind``).  To
        exercise repeated reprocessing we therefore interleave a fast,
        in-order sensor (which drives the clock) with a steady-lag sensor
        that is always 100 ms stale, i.e. out of sequence on every sample.
        """
        m = CVModel(n_dim=2, accel_psd=1.0)
        f = _fresh_filter(m, history_s=2.0)
        n_pairs = 300
        for k in range(11, 11 + n_pairs):
            t = 0.01 * k
            # Fast, in-order sample: moves the clock to ``t``.
            f.add_measurement(_pos_meas(m, t, t, [t, -t], 0.2))
            # Slow sample valid 100 ms ago: always out of sequence.
            t_old = t - 0.1
            f.add_measurement(_pos_meas(m, t_old, t, [t_old, -t_old], 0.2))
        assert f.diag.n_out_of_order == n_pairs
        assert f.diag.n_reprocessed >= n_pairs
        assert np.all(np.isfinite(f.x)) and np.all(np.isfinite(f.P))
        # The retained history still spans roughly ``history_s``.
        span = f._snapshots[-1].t - f._snapshots[0].t
        assert span > 0.9 * f.history_s
        # ...and a rewind near the edge of the window must still be possible.
        t_now = f.t
        f.add_measurement(_pos_meas(m, t_now - 0.5, t_now + 0.01, [1.0, -1.0], 0.2))
        assert np.all(np.isfinite(f.x))

    def test_uniformly_delayed_monotone_stream_needs_no_rewind(self):
        """Documented clock semantics: ``f.t`` = newest *fused* measurement time.

        A stream that is delayed but monotone in validity time is
        *in-sequence*: the state honestly carries the sensor's timestamp and
        the remaining latency is removed at the display instant by
        ``predict_to_time``.  (The naive ``receive_time`` mode is the one that
        silently stamps the state at arrival time.)
        """
        m = CVModel(n_dim=2, accel_psd=1.0)
        f = _fresh_filter(m, history_s=2.0)
        lag = 0.2
        for k in range(1, 201):
            t = 0.02 * k
            f.add_measurement(_pos_meas(m, t, t + lag, [t, -t], 0.2))
        assert f.diag.n_out_of_order == 0
        assert f.diag.n_reprocessed == 0
        assert f.t == pytest.approx(0.02 * 200)
        # Arrival clock is ``lag`` ahead of the state stamp -> that is exactly
        # the horizon ``predict_to_time`` is designed to bridge.
        x_fut, _ = f.predict_to_time(f.t + lag)
        assert np.all(np.isfinite(x_fut))
        assert np.all(np.isfinite(f.x)) and np.all(np.isfinite(f.P))

    def test_clock_stays_at_newest_validity_time_after_rewind(self):
        """A late sample updates the *past* without dragging the clock back."""
        m = CVModel(n_dim=2)
        f = _fresh_filter(m)
        f.add_measurement(_pos_meas(m, 0.5, 0.6, [5.0, 0.0], 0.3))
        f.add_measurement(_pos_meas(m, 0.2, 0.7, [2.0, 0.0], 0.3))
        assert f.diag.n_out_of_order == 1
        assert f.t == pytest.approx(0.5)
        # The stale sample still improved the estimate at t=0.5.
        assert np.all(np.isfinite(f.x)) and np.all(np.isfinite(f.P))

    def test_interleaved_multirate_stream_is_stable(self):
        """Three sensors at different rates/delays delivered in recv order."""
        rng = np.random.default_rng(5)
        m = CVModel(n_dim=2, accel_psd=4.0)
        f = _fresh_filter(m, history_s=2.0)
        meas = []
        for period, delay, sigma in ((0.01, 0.005, 1.2),
                                     (0.05, 0.030, 0.6),
                                     (0.10, 0.060, 0.25)):
            t = period
            while t <= 4.0:
                jitter = rng.normal(0.0, delay * 0.15)
                meas.append(_pos_meas(m, t, t + delay + jitter,
                                      [10.0 * np.cos(t), 10.0 * np.sin(t)],
                                      sigma))
                t += period
        meas.sort(key=lambda mm: mm.t_recv)
        for mm in meas:
            f.add_measurement(mm)  # must not raise
        assert np.all(np.isfinite(f.x)) and np.all(np.isfinite(f.P))
        assert f.diag.n_out_of_order > 0
        assert f.diag.n_reprocessed > f.diag.n_out_of_order


# ---------------------------------------------------------------------------
# Forward prediction (display-time)
# ---------------------------------------------------------------------------
class TestPredictToTime:
    def test_matches_predict_to_then_read(self):
        m = CVModel(n_dim=2, accel_psd=1.0)
        f = _fresh_filter(m, x0=np.array([1.0, 0.0, 2.0, 0.0]))
        f.add_measurement(_pos_meas(m, 0.1, 0.1, [1.2, 0.0], 0.2))
        x_fut, P_fut = f.predict_to_time(0.5)
        t_keep, x_keep, P_keep = f.estimate()
        f.predict_to(0.5)
        np.testing.assert_allclose(f.x, x_fut, atol=1e-12)
        np.testing.assert_allclose(f.P, P_fut, atol=1e-12)
        # predict_to_time must not mutate
        assert f.t == pytest.approx(0.5)
        assert t_keep == pytest.approx(0.1)
        assert x_keep.shape == x_keep.shape
        assert P_keep.shape == P_keep.shape

    def test_forward_prediction_moves_along_velocity(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m, x0=np.array([0.0, 0.0, 10.0, -4.0]), P0=np.eye(4) * 0.01)
        x_fut, _ = f.predict_to_time(0.5)
        np.testing.assert_allclose(x_fut[:2], [5.0, -2.0], atol=1e-9)

    def test_backwards_prediction_raises(self):
        m = CVModel(n_dim=2)
        f = _fresh_filter(m)
        f.predict_to(1.0)
        with pytest.raises(ValueError):
            f.predict_to_time(0.5)
        with pytest.raises(ValueError):
            f.predict_to(0.5)


# ---------------------------------------------------------------------------
# Naive receive-time fusion is biased on a manoeuvring target
# ---------------------------------------------------------------------------
class TestCompensationBenefit:
    @staticmethod
    def _run(mode):
        m = CVModel(n_dim=2, accel_psd=4.0)
        n = m.n_dim
        omega, radius, delay, dt = 0.35, 25.0, 0.12, 0.05
        x0 = np.array([radius, 0.0, 0.0, omega * radius], float)
        f = DelayedFusionFilter(m, x0, np.eye(4) * 1.0, t0=0.0, mode=mode, history_s=5.0)
        rng = np.random.default_rng(11)

        def truth(t):
            return np.array([radius * np.cos(omega * t), radius * np.sin(omega * t)])

        errs = []
        for k in range(1, 121):
            t = dt * k
            p = truth(t) + rng.normal(0.0, 0.4, size=n)
            f.add_measurement(_pos_meas(m, t, t + delay, p, 0.4))
            errs.append(np.linalg.norm(f.position - truth(f.t)))
        return float(np.sqrt(np.mean(np.square(errs))))

    def test_oosm_beats_receive_time(self):
        rmse_oosm = self._run("oosm")
        rmse_naive = self._run("receive_time")
        assert rmse_oosm < rmse_naive
        # The naive filter must eat at least part of the 0.12 s lag
        # (speed ~8.75 m/s => ~1.0 m of pure lag).
        assert rmse_naive - rmse_oosm > 0.2


# ---------------------------------------------------------------------------
# Errors & helpers
# ---------------------------------------------------------------------------
class TestErrors:
    def test_receive_before_measurement_raises(self):
        m = CVModel(n_dim=2)
        with pytest.raises(ValueError):
            _pos_meas(m, 0.5, 0.4, [0.0, 0.0], 0.3)

    def test_bad_mode_raises(self):
        m = CVModel(n_dim=2)
        with pytest.raises(ValueError):
            DelayedFusionFilter(m, np.zeros(4), np.eye(4), mode="magic")

    def test_bad_initial_shape_raises(self):
        m = CVModel(n_dim=2)
        with pytest.raises(ValueError):
            DelayedFusionFilter(m, np.zeros(3), np.eye(4))

    def test_bad_n_dim_raises(self):
        with pytest.raises(ValueError):
            CVModel(n_dim=0)


class TestHelpers:
    def test_position_covariance_block(self):
        M = np.arange(16, dtype=float).reshape(4, 4)
        np.testing.assert_allclose(position_covariance(M, 2), M[:2, :2])

    def test_nees_zero_when_exact(self):
        x = np.array([1.0, 2.0, 0.0, 0.0])
        P = np.eye(4)
        assert nees_position(x, P, np.array([1.0, 2.0]), 2) == pytest.approx(0.0)

    def test_nees_scales_with_error(self):
        P = np.eye(4)
        small = nees_position(np.array([0.1, 0.0, 0, 0]), P, np.zeros(2), 2)
        big = nees_position(np.array([1.0, 0.0, 0, 0]), P, np.zeros(2), 2)
        assert big > small
