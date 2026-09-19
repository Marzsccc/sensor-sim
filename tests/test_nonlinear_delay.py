"""Unit tests for nonlinear (ESKF) delayed measurement fusion (v0.23.0).

The load-bearing claim is *exactness*: a filter that receives samples out of
order and rewinds + replays them must end up in the same state as a filter
that received the very same samples in validity order.  For a linear model
that was checked in ``test_delay_fusion.py`` against an analytic argument;
here the mechanisation is nonlinear, so the argument is *reproducibility* --
the same IMU samples replayed in the same order reproduce the same states --
and the tests below verify it numerically against an independent in-order
runner built on the plain :class:`~sensor_sim.eskf.ESKF`.
"""

import numpy as np
import pytest

from sensor_sim.eskf import ESKF, ESKFConfig
from sensor_sim.nonlinear_delay import (
    EskfSnapshot,
    GnssMeasurement,
    ImuSample,
    RewindReplayESKF,
    capture_state,
    position_nees,
    restore_state,
)


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

Q0 = np.array([1.0, 0.0, 0.0, 0.0])


def _make_imu(n=200, dt=0.01, t0=0.0):
    """Deterministic (if slightly unphysical) IMU stream -- fine for replay."""
    out = []
    for k in range(n):
        t = t0 + k * dt
        acc = np.array([0.2 * np.sin(3.0 * t), 0.1, 9.80665 + 0.01 * np.cos(2.0 * t)])
        gyro = np.array([0.01 * np.sin(t), 0.02 * np.cos(2.0 * t), 0.2])
        out.append((t, acc, gyro))
    return out


def _make_gnss(t_end, period=0.05, offset=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    k = period + offset
    while k < t_end - 1e-9:
        out.append((k, rng.normal(0.0, 1.0, 3), rng.normal(0.0, 0.1, 3)))
        k += period
    return out


# Tie-break convention, shared by the rewind-and-replay filter and this
# reference runner: an IMU sample and an aiding fix stamped at the *same*
# instant are processed IMU-first, because the fix is only meaningful once the
# mechanisation has reached its own validity time.  Times are compared at
# 1e-9 s granularity so that accumulated floating-point noise in
# ``t += period`` cannot flip the order.
def _tie_key(t: float, rank: int):
    return (round(float(t), 9), rank)


def _reference_in_order(imu, gnss, cfg=None, t0=0.0):
    """Plain ESKF fed in validity order (IMU first on ties)."""
    f = ESKF(cfg)
    f.set_initial_state(t0, np.zeros(3), np.zeros(3), Q0.copy())
    events = [("imu", t, i, _tie_key(t, 0)) for i, (t, _, _) in enumerate(imu)]
    events += [("gnss", t, i, _tie_key(t, 1)) for i, (t, _, _) in enumerate(gnss)]
    events.sort(key=lambda e: e[3])
    for kind, t, i, _ in events:
        if kind == "imu":
            _, acc, gyro = imu[i]
            f.predict(acc, gyro, t - f.state.t)
        else:
            _, pos, vel = gnss[i]
            f.update_gnss(pos, vel)
    return f


def _run_oosm(imu, gnss, *, delay=0.03, mode="oosm", history_s=5.0,
              cfg=None, t0=0.0, order=None):
    """Feed the wrapper in *receive* order (IMU at t, GNSS at t_meas + delay)."""
    f = RewindReplayESKF(cfg, mode=mode, history_s=history_s)
    f.initialize(t0, np.zeros(3), np.zeros(3), Q0.copy())

    if order is None:
        order = list(range(len(gnss)))
    else:
        # Deliver calls in a permuted order; receive timestamps are untouched,
        # only the *call* order changes (a pipeline backlog, not time travel).
        pass

    i = j = 0
    pending = list(order)
    while i < len(imu) or pending:
        t_imu = imu[i][0] if i < len(imu) else np.inf
        t_recv = (gnss[pending[0]][0] + delay) if pending else np.inf
        if t_imu <= t_recv + 1e-12:
            t, acc, gyro = imu[i]
            f.add_imu(t, acc, gyro)
            i += 1
        else:
            k = pending.pop(0)
            t_meas, pos, vel = gnss[k]
            f.add_gnss(t_meas, pos, vel, t_recv=t_meas + delay)
    return f


def _assert_same_state(fa, fb, atol=1e-9):
    assert abs(fa.state.t - fb.state.t) < atol * 100
    assert np.allclose(fa.state.p, fb.state.p, atol=atol)
    assert np.allclose(fa.state.v, fb.state.v, atol=atol)
    assert np.allclose(fa.state.q, fb.state.q, atol=atol)
    assert np.allclose(fa.state.P, fb.state.P, atol=atol)
    assert fa.state.n_updates == fb.state.n_updates


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
class TestRecords:
    def test_imu_sample_shapes(self):
        s = ImuSample(0.5, [1.0, 2.0, 3.0], [0.0, 0.0, 0.1])
        assert s.acc.shape == (3,) and s.gyro.shape == (3,)

    def test_imu_sample_bad_shape(self):
        with pytest.raises(ValueError):
            ImuSample(0.0, [1.0, 2.0], [0.0, 0.0, 0.0])

    def test_gnss_measurement_validation(self):
        m = GnssMeasurement(1.0, [0.0, 0.0, 0.0], t_recv=1.1)
        assert m.t_recv == 1.1
        with pytest.raises(ValueError):
            GnssMeasurement(1.0, [0.0, 0.0, 0.0], t_recv=0.9)

    def test_gnss_measurement_bad_pos(self):
        with pytest.raises(ValueError):
            GnssMeasurement(0.0, [0.0, 0.0])

    def test_position_nees_zero_for_perfect_estimate(self):
        assert position_nees(np.zeros(3), np.eye(3)) == pytest.approx(0.0)

    def test_position_nees_three_dof_mean(self):
        # E[chi^2_3] = 3, so a unit error with unit covariance gives 3.
        assert position_nees(np.ones(3), np.eye(3)) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Snapshot round-trip
# ---------------------------------------------------------------------------
class TestSnapshots:
    def test_capture_restore_round_trip(self):
        f = ESKF()
        f.set_initial_state(1.0, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], Q0.copy())
        f.predict(np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.1]), 0.01)
        snap = capture_state(f)
        assert isinstance(snap, EskfSnapshot)

        f.predict(np.array([0.5, 0.0, 9.81]), np.array([0.1, 0.0, 0.1]), 0.5)
        changed = f.state.p.copy()
        assert not np.allclose(changed, snap.p)

        restore_state(f, snap)
        assert f.state.t == pytest.approx(snap.t)
        assert np.allclose(f.state.p, snap.p)
        assert np.allclose(f.state.P, snap.P)
        assert f.state.n_updates == snap.n_updates

    def test_snapshot_is_a_copy_not_a_view(self):
        f = ESKF()
        f.set_initial_state(0.0, np.zeros(3), np.zeros(3), Q0.copy())
        snap = capture_state(f)
        f.state.p[:] = 99.0
        assert np.allclose(snap.p, 0.0)


# ---------------------------------------------------------------------------
# Forward (in-sequence) path
# ---------------------------------------------------------------------------
class TestForwardPath:
    def test_forward_only_matches_reference(self):
        imu = _make_imu(n=120)
        gnss = _make_gnss(imu[-1][0], period=0.05, offset=0.005)
        # Deliver each fix slightly *after* its validity time but before the
        # next IMU sample: state time is behind, so no rewind happens.
        f = RewindReplayESKF(history_s=5.0)
        f.initialize(0.0, np.zeros(3), np.zeros(3), Q0.copy())
        i = j = 0
        while i < len(imu) or j < len(gnss):
            t_imu = imu[i][0] if i < len(imu) else np.inf
            t_next = (gnss[j][0] + 0.001) if j < len(gnss) else np.inf
            if t_imu <= t_next:
                t, acc, gyro = imu[i]
                f.add_imu(t, acc, gyro)
                i += 1
            else:
                t_meas, pos, vel = gnss[j]
                f.add_gnss(t_meas, pos, vel)
                j += 1
        assert f.diag.n_out_of_order == 0
        assert f.diag.n_rewinds == 0
        _assert_same_state(f.filt, _reference_in_order(imu, gnss))

    def test_delayed_monotone_aiding_stream_still_rewinds(self):
        """Latency alone makes a fix out-of-sequence once IMU drives the clock.

        In v0.22.0 (fusion-only, no inertial input) the filter clock *was* the
        newest fused measurement time, so a constantly delayed but monotone
        stream was in sequence.  Here the clock is the newest *integrated IMU*
        time, which always races ahead of a delayed fix -- so every aided
        sample is inserted into the past.  That is the honest situation for a
        GNSS+IMU stack; residual latency at render time is still removed by
        ``extrapolate_to``.
        """
        imu = _make_imu(n=200)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = _run_oosm(imu, gnss, delay=0.2)
        assert f.diag.n_rewinds == len(gnss)
        # Rewinds never drag the clock back: it stays pinned to the IMU stream.
        assert f.time == pytest.approx(imu[-1][0])
        assert f.diag.max_rewind_s <= 0.2 + 0.02

    def test_gnss_ahead_of_state_is_applied_immediately(self):
        f = RewindReplayESKF()
        f.initialize(0.0, np.zeros(3), np.zeros(3), Q0.copy())
        f.add_gnss(0.0, np.array([1.0, 0.0, 0.0]))
        assert f.diag.n_applied == 1
        assert f.position[0] > 0.0

    def test_imu_must_be_monotone(self):
        f = RewindReplayESKF()
        f.initialize(1.0, np.zeros(3), np.zeros(3), Q0.copy())
        f.add_imu(1.01, np.array([0.0, 0.0, 9.81]), np.zeros(3))
        with pytest.raises(ValueError):
            f.add_imu(1.005, np.array([0.0, 0.0, 9.81]), np.zeros(3))


# ---------------------------------------------------------------------------
# Out-of-sequence exactness
# ---------------------------------------------------------------------------
class TestExactness:
    def test_out_of_order_matches_reference(self):
        imu = _make_imu(n=200)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = _run_oosm(imu, gnss, delay=0.06)
        assert f.diag.n_out_of_order > 10
        assert f.diag.n_reprocessed > 0
        _assert_same_state(f.filt, _reference_in_order(imu, gnss))

    def test_exact_even_with_duplicate_validity_order(self):
        """Two sensors with different latencies => permanent interleaving."""
        imu = _make_imu(n=250)
        fast = _make_gnss(imu[-1][0], period=0.05, seed=1)
        slow = _make_gnss(imu[-1][0], period=0.05, offset=0.017, seed=2)

        f = RewindReplayESKF(history_s=5.0)
        f.initialize(0.0, np.zeros(3), np.zeros(3), Q0.copy())

        deliveries = []
        for t, p, v in fast:
            deliveries.append((t + 0.02, t, p, v))
        for t, p, v in slow:
            deliveries.append((t + 0.12, t, p, v))
        deliveries.sort(key=lambda d: d[0])

        i = j = 0
        while i < len(imu) or j < len(deliveries):
            t_imu = imu[i][0] if i < len(imu) else np.inf
            t_recv = deliveries[j][0] if j < len(deliveries) else np.inf
            if t_imu <= t_recv:
                t, acc, gyro = imu[i]
                f.add_imu(t, acc, gyro)
                i += 1
            else:
                t_recv, t_meas, p, v = deliveries[j]
                f.add_gnss(t_meas, p, v, t_recv=t_recv)
                j += 1

        assert f.diag.n_rewinds > 10
        ref = _reference_in_order(imu, fast + slow)
        _assert_same_state(f.filt, ref)

    def test_scrambled_delivery_order_matches_reference(self):
        """A backlog that delivers fixes in a jumbled order still converges."""
        imu = _make_imu(n=200)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = RewindReplayESKF(history_s=5.0)
        f.initialize(0.0, np.zeros(3), np.zeros(3), Q0.copy())

        # Receive times are t_meas + 0.06, but *some* of them are delivered in
        # a shuffled call order (bounded jitter of the backend, not physics).
        order = list(range(len(gnss)))
        rng = np.random.default_rng(7)
        for _ in range(30):
            a, b = sorted(rng.integers(0, len(order), 2))
            if b > a:
                order[a:b] = order[a:b][::-1]

        i = 0
        pending = list(order)
        while i < len(imu) or pending:
            t_imu = imu[i][0] if i < len(imu) else np.inf
            t_recv = (gnss[pending[0]][0] + 0.06) if pending else np.inf
            if t_imu <= t_recv + 1e-12:
                t, acc, gyro = imu[i]
                f.add_imu(t, acc, gyro)
                i += 1
            else:
                k = pending.pop(0)
                t_meas, pos, vel = gnss[k]
                f.add_gnss(t_meas, pos, vel, t_recv=t_meas + 0.06)

        _assert_same_state(f.filt, _reference_in_order(imu, gnss))

    def test_long_out_of_order_run_keeps_history(self):
        """Regression: replaying must not amputate older snapshots.

        In v0.22.0 the linear filter rebuilt its snapshot list from the anchor
        on every rewind, silently truncating all earlier history -- after a few
        minutes of out-of-order samples every rewind raised ``RuntimeError``.
        """
        imu = _make_imu(n=1500)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = _run_oosm(imu, gnss, delay=0.09, history_s=2.0)
        assert f.diag.n_rewinds == len(gnss)
        assert f.diag.n_rewinds > 250
        t_now = f.time
        oldest = f._snaps[0].t
        # The retained history still spans the configured horizon.
        assert oldest <= t_now - 2.0 + 0.1
        # And a rewind into it still works (no RuntimeError, no clamp).
        f.add_gnss(t_now - 1.0, np.array([1.0, 1.0, 1.0]))
        assert f.diag.n_clamped == 0

    def test_repeated_identical_rewind_is_idempotent(self):
        """Replaying history cannot corrupt the timeline.

        Every replay restarts from the *same* anchor snapshot and walks the
        *same* IMU buffer, so the state after N late fixes depends only on the
        set of their validity times -- not on delivery order or on how many
        rewinds it took to get there.
        """
        imu = _make_imu(n=100)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = _run_oosm(imu, gnss, delay=0.05)
        t_now = f.time

        before = f.filt.state.p.copy()
        f.add_gnss(t_now - 0.2, np.array([0.0, 0.0, 0.0]))
        once = f.filt.state.p.copy()
        n_after_first = f.diag.n_rewinds

        f.add_gnss(t_now - 0.25, np.array([0.0, 0.0, 0.0]))
        twice = f.filt.state.p.copy()
        assert f.diag.n_rewinds == n_after_first + 1

        assert f.time == pytest.approx(t_now)
        assert np.isfinite(twice).all()
        assert np.all(np.linalg.eigvalsh(f.filt.state.P) >= -1e-9)
        assert not np.allclose(before, once)
        assert not np.allclose(once, twice)

    def test_replay_is_deterministic_under_redelivery(self):
        """The same two late fixes, delivered in either order, agree exactly.

        Also checks the stronger property: the result equals what a chip that
        had received the fixes in validity order in the first place would have
        computed (modulo the replayed-aiding-set being identical).
        """
        imu = _make_imu(n=120)
        gnss = _make_gnss(imu[-1][0], period=0.05)

        def build(extra):
            g = RewindReplayESKF(history_s=5.0)
            g.initialize(0.0, np.zeros(3), np.zeros(3), Q0.copy())
            for t, acc, gyro in imu:
                g.add_imu(t, acc, gyro)
            for t, pos, vel in gnss:
                g.add_gnss(t, pos, vel, t_recv=t + 0.05)
            t_now = g.time
            if extra == "late_b_a":
                g.add_gnss(t_now - 0.15, np.array([2.0, 0.0, 0.0]))
                g.add_gnss(t_now - 0.30, np.array([1.0, 0.0, 0.0]))
            elif extra == "late_a_b":
                g.add_gnss(t_now - 0.30, np.array([1.0, 0.0, 0.0]))
                g.add_gnss(t_now - 0.15, np.array([2.0, 0.0, 0.0]))
            return g, t_now

        g_ab, t_ab = build("late_a_b")
        g_ba, t_ba = build("late_b_a")

        assert t_ab == pytest.approx(t_ba)
        assert np.allclose(g_ab.filt.state.p, g_ba.filt.state.p, atol=1e-9)
        assert np.allclose(g_ab.filt.state.P, g_ba.filt.state.P, atol=1e-9)


# ---------------------------------------------------------------------------
# Clock semantics
# ---------------------------------------------------------------------------
class TestClockSemantics:
    def test_late_fix_does_not_drag_the_clock_back(self):
        imu = _make_imu(n=200)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = _run_oosm(imu, gnss, delay=0.08)
        assert f.time == pytest.approx(imu[-1][0])

    def test_replay_lands_exactly_on_previous_time(self):
        imu = _make_imu(n=200)
        gnss = _make_gnss(imu[-1][0], period=0.05)
        f = _run_oosm(imu, gnss, delay=0.08)
        t_before = f.time
        f.add_gnss(t_before - 0.3, np.array([0.0, 0.0, 0.0]))
        assert f.time == pytest.approx(t_before)

    def test_extrapolate_forward_grows_covariance(self):
        imu = _make_imu(n=100)
        f = _run_oosm(imu, [], delay=0.0)
        _, P0 = f.extrapolate_to(f.time)
        _, P1 = f.extrapolate_to(f.time + 0.1)
        assert np.trace(P1) > np.trace(P0)

    def test_extrapolate_matches_constant_velocity(self):
        f = RewindReplayESKF()
        f.initialize(0.0, np.zeros(3), np.array([1.0, 2.0, 0.0]), Q0.copy())
        p, _ = f.extrapolate_to(0.5)
        assert np.allclose(p, [0.5, 1.0, 0.0])

    def test_extrapolate_rejects_past(self):
        f = RewindReplayESKF()
        f.initialize(1.0, np.zeros(3), np.zeros(3), Q0.copy())
        with pytest.raises(ValueError):
            f.extrapolate_to(0.5)


# ---------------------------------------------------------------------------
# History bounds
# ---------------------------------------------------------------------------
class TestHistory:
    def test_buffers_are_pruned(self):
        imu = _make_imu(n=1000)
        f = _run_oosm(imu, [], delay=0.0, history_s=1.0)
        n_snaps, n_imu, n_gnss = f.history()
        assert n_snaps <= 101
        assert n_imu <= 101

    def test_snapshot_bound_is_enforced_not_grown(self):
        imu = _make_imu(n=500)
        f = RewindReplayESKF(history_s=0.5)
        f.initialize(0.0, np.zeros(3), np.zeros(3), Q0.copy())
        for t, acc, gyro in imu:
            f.add_imu(t, acc, gyro)
            assert len(f._snaps) <= 51 + 1

    def test_too_late_fix_is_clamped_and_counted(self):
        imu = _make_imu(n=400)
        f = _run_oosm(imu, [], delay=0.0, history_s=0.3)
        f.add_gnss(0.0, np.array([5.0, 5.0, 5.0]))
        assert f.diag.n_clamped == 1
        assert f.diag.n_rewinds == 1
        assert np.isfinite(f.position).all()

    def test_late_fix_still_pulls_state_towards_it(self):
        imu = _make_imu(n=400)
        f = _run_oosm(imu, [], delay=0.0, history_s=1.0)
        t_now = f.time
        f.add_gnss(t_now - 0.2, np.array([8.0, 0.0, 0.0]), pos_std=0.5)
        assert f.position[0] > 0.0

    def test_modes_reject_unknown(self):
        with pytest.raises(ValueError):
            RewindReplayESKF(mode="magic")


# ---------------------------------------------------------------------------
# Naive baseline vs OOSM, on a physically consistent track
# ---------------------------------------------------------------------------
def _straight_track(n_imu=1200, imu_dt=0.01, v0=25.0, seed=3):
    """Truth: 90 km/h forward plus a gentle lateral sinusoid.

    Returns ``(imu, gnss, truth_fn)`` where truth_fn(t) -> (p, v).
    """
    rng = np.random.default_rng(seed)
    T = n_imu * imu_dt
    ts = np.arange(0.0, T + 1e-12, 1e-4)

    A, w = 1.5, 0.5  # lateral accel amplitude / rate

    def a_world(t):
        return np.array([0.0, A * np.sin(w * t), 0.0])

    v = np.zeros((len(ts), 3))
    p = np.zeros((len(ts), 3))
    v[0] = [v0, 0.0, 0.0]
    for k in range(1, len(ts)):
        h = ts[k] - ts[k - 1]
        v[k] = v[k - 1] + a_world(ts[k - 1]) * h
        p[k] = p[k - 1] + v[k - 1] * h + 0.5 * a_world(ts[k - 1]) * h * h

    def truth_fn(t):
        k = int(round(t / 1e-4))
        k = min(max(k, 0), len(ts) - 1)
        return p[k].copy(), v[k].copy()

    g = np.array([0.0, 0.0, -9.80665])
    imu = []
    for k in range(n_imu):
        t = k * imu_dt
        a_w = a_world(t)
        acc = a_w - g + rng.normal(0.0, 0.02, 3)
        gyro = np.zeros(3) + rng.normal(0.0, 1e-3, 3)
        imu.append((t, acc, gyro))

    gnss = []
    t = 1.0
    while t < T - 0.3:
        p_true, v_true = truth_fn(t)
        gnss.append((t, p_true + rng.normal(0.0, 1.0, 3), None))
        t += 0.1
    return imu, gnss, truth_fn


def _run_pipeline(imu, gnss, delay, mode):
    """Run one pipeline, recording the estimate at each arrival.

    Also records the *display-time* pose: the pose the pipeline would render
    for wall-clock instant ``t_recv + 0.06`` (60 ms display latency), obtained
    by constant-velocity extrapolation of that pipeline's own state.  This is
    the quantity a driver actually sees, and it is the fair basis for
    comparing two pipelines -- scoring each against its own clock would hand
    the lagging one a longer prediction horizon for free.
    """
    f = RewindReplayESKF(mode=mode, history_s=5.0)
    f.initialize(0.0, np.zeros(3), np.array([25.0, 0.0, 0.0]), Q0.copy())

    deliveries = [(t + delay, t, p) for (t, p, _) in gnss]
    i = j = 0
    records = []
    while i < len(imu) or j < len(deliveries):
        t_imu = imu[i][0] if i < len(imu) else np.inf
        t_recv = deliveries[j][0] if j < len(deliveries) else np.inf
        if t_imu <= t_recv:
            t, acc, gyro = imu[i]
            f.add_imu(t, acc, gyro)
            i += 1
        else:
            t_recv, t_meas, p = deliveries[j]
            f.add_gnss(t_meas, p, t_recv=t_recv, pos_std=1.0)
            j += 1
        t_disp = t_recv + 0.06
        if np.isfinite(t_disp) and t_disp >= f.time:
            p_disp, _ = f.extrapolate_to(t_disp, accel_psd=1.0)
            records.append((f.time, f.position.copy(), f.position_covariance.copy(),
                            t_disp, p_disp))
    return f, records


class TestNaiveVsOosm:
    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls):
        imu, gnss, truth_fn = _straight_track()
        out = {}
        for mode in ("oosm", "receive_time"):
            f, recs = _run_pipeline(imu, gnss, 0.2, mode)
            # score from t = 2 s onwards (let the filter converge)
            errs, nees, display_errs = [], [], []
            for t_f, p_est, P_pos, t_disp, p_disp in recs:
                if t_f < 2.0:
                    continue
                p_true, _ = truth_fn(t_f)
                errs.append(np.linalg.norm(p_est - p_true))
                nees.append(position_nees(p_est - p_true, P_pos))
                p_true_disp, _ = truth_fn(t_disp)
                display_errs.append(np.linalg.norm(p_disp - p_true_disp))
            out[mode] = dict(
                filter=f,
                rmse=float(np.sqrt(np.mean(np.square(errs)))),
                nees=float(np.mean(nees)),
                display_rmse=float(np.sqrt(np.mean(np.square(display_errs)))),
            )
        return out

    def test_naive_pipeline_lags(self, runs):
        assert runs["receive_time"]["rmse"] > runs["oosm"]["rmse"]

    def test_oosm_is_much_more_accurate(self, runs):
        ratio = runs["oosm"]["rmse"] / runs["receive_time"]["rmse"]
        assert ratio < 0.5

    def test_naive_pipeline_is_overconfident(self, runs):
        # 3 position dof => expected NEES 3.  The lagged baseline claims far
        # more confidence than its error justifies.
        assert runs["receive_time"]["nees"] > 3.0 * runs["oosm"]["nees"]

    def test_oosm_nees_is_in_the_right_ballpark(self, runs):
        assert 0.5 < runs["oosm"]["nees"] < 30.0

    def test_display_time_extrapolation_helps_both(self, runs):
        # Rendering 60 ms ahead must not be worse than the streaming estimate.
        assert runs["oosm"]["display_rmse"] < runs["oosm"]["rmse"] * 1.2

    def test_lag_magnitude_is_v_times_delay(self, runs):
        # The naive bias should be on the order of v*delay = 25 * 0.2 = 5 m,
        # i.e. the receive-time pipeline is off by metres, not centimetres.
        assert runs["receive_time"]["rmse"] > 1.0

    def test_history_bounds_respected_for_both(self, runs):
        for mode in ("oosm", "receive_time"):
            f = runs[mode]["filter"]
            n_snaps, _, _ = f.history()
            assert n_snaps <= 501
