"""Tests for online clock-offset estimation (v0.24.0).

The theory being pinned down here has three levels, and the tests are grouped
to match it:

* standstill                       -> ``dh/db = 0``: unobservable (and harmless),
* constant speed (+ odometry)      -> degenerate with the initial position:
                                      unobservable, prior-dependent,
* varying speed (+ odometry)       -> observable, prior-independent, with
                                      ``std -> sigma / sqrt(sum (v - vbar)^2)``.

A fourth, structural level is asserted in ``TestClockIsDrivenByReference``:
the filter's clock must be advanced by the *reference*-clock channel. If the
skewed position stream drives it, the estimator acquires a flat valley of
equally-good solutions and a sign-dependent bias -- which is what this module
did wrong once.
"""

import numpy as np
import pytest

from sensor_sim.delay_fusion import CVModel, DelayedFusionFilter, DelayedMeasurement
from sensor_sim.time_offset import (
    ClockOffset,
    OffsetAugmentedFilter,
    bias_fisher_information,
    crlb_bias_std,
)

# ---------------------------------------------------------------------------
# scenario helpers
# ---------------------------------------------------------------------------

V_MEAN, V_AMP, V_PERIOD = 15.0, 10.0, 20.0


def speed_profile(t, vary=True):
    t = np.asarray(t, dtype=float)
    if not vary:
        return np.full_like(t, V_MEAN)
    return V_MEAN + V_AMP * np.sin(2.0 * np.pi * t / V_PERIOD)


def _position_table(t_end, vary=True, dt=1e-3):
    ts = np.arange(0.0, t_end + dt, dt)
    v = speed_profile(ts, vary)
    p = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(ts))])
    return ts, p


def position_at(t, table):
    return np.interp(t, table[0], table[1])


def simulate(b_true=0.12, rate_ppm=0.0, vary=True, use_odo=True, b0=0.0, b_std0=0.5,
             b_rw_std=0.0, t_start=10.0, t_end=70.0, dt_pos=0.1, dt_vel=0.05,
             sigma=1.0, sigma_v=0.05, seed=0, P_pos=1e4, P_vel=1e-2,
             accel_psd=1e-3, estimate_rate=False, rate_std0=2e-4):
    """Run one filtered scenario; returns the filter (with a scorer attached).

    ``use_odo=True`` drives the filter clock with the reference-clock velocity
    channel (the realistic vehicle setup).  ``use_odo=False`` has no reference
    clock at all, so the clock is advanced by the skewed stamps instead -- the
    degenerate configuration used to test the *absence* of information.
    """
    rng = np.random.default_rng(seed)
    clk = ClockOffset(bias_s=b_true, rate_ppm=rate_ppm, t_ref=0.0)
    table = _position_table(t_end, vary)

    f = OffsetAugmentedFilter(
        CVModel(n_dim=1, accel_psd=accel_psd),
        x0=np.array([position_at(t_start, table), float(speed_profile(t_start, vary))]),
        P0=np.diag([P_pos, P_vel]),
        obs_sigma=sigma,
        estimate_rate=estimate_rate,
        bias0=b0,
        bias_std0=b_std0,
        rate0=0.0,
        rate_std0=rate_std0,
        bias_rw_std=b_rw_std,
        rate_rw_std=0.0,
        record_history=True,
    )
    f.t = float(t_start)
    f.history.clear()
    f._record()

    t_pos = np.arange(t_start, t_end, dt_pos)
    t_vel = np.arange(t_start, t_end, dt_vel) if use_odo else np.zeros(0)
    errors: list = []
    iv = ip = 0
    while ip < t_pos.size:
        t_vel_next = t_vel[iv] if iv < t_vel.size else np.inf
        t_rep = float(clk.to_reported(t_pos[ip]))
        if t_vel_next <= t_rep:
            f.update_velocity(float(t_vel[iv]),
                              np.array([speed_profile(t_vel[iv], vary)
                                        + rng.normal(0.0, sigma_v)]),
                              sigma_v)
            iv += 1
        else:
            if not use_odo:
                f.predict_to(t_rep)          # skewed-stream clock (degenerate mode)
            f.update(t_rep, np.array([position_at(t_pos[ip], table)
                                      + rng.normal(0.0, sigma)]))
            ip += 1
            errors.append(f.position[0] - position_at(f.t, table))
    f.errors = np.asarray(errors)
    f.table = table
    return f


def run_plain(f, t_rep, z, sigma=None, drive_clock=True):
    """Feed a pure position stream, advancing the clock with it."""
    for t, zi in zip(t_rep, z):
        if drive_clock and float(t) > f.t + 1e-12:
            f.predict_to(float(t))
        f.update(float(t), np.array([zi]), sigma=sigma)
    return f


def make_filter(v0=0.0, estimate_rate=False, sigma=1.0, bias0=0.0, bias_std0=0.5,
                bias_rw_std=0.0, P_pos=1e-6, P_vel=1e-6, accel_psd=1e-4):
    model = CVModel(n_dim=1, accel_psd=accel_psd)
    return OffsetAugmentedFilter(
        model,
        x0=np.array([0.0, v0]),
        P0=np.diag([P_pos, P_vel]),
        obs_sigma=sigma,
        estimate_rate=estimate_rate,
        bias0=bias0,
        bias_std0=bias_std0,
        rate_std0=2e-4,
        bias_rw_std=bias_rw_std,
        rate_rw_std=0.0,
    )


def straight_line(v, t0, t1, dt, b=0.0, sigma=1.0, seed=0, rate_ppm=0.0):
    rng = np.random.default_rng(seed)
    clk = ClockOffset(bias_s=b, rate_ppm=rate_ppm, t_ref=0.0)
    t_true = np.arange(t0, t1, dt)
    p_true = v * t_true
    z = p_true + rng.normal(0.0, sigma, t_true.size)
    return t_true, clk.to_reported(t_true), z


# ---------------------------------------------------------------------------
# ClockOffset
# ---------------------------------------------------------------------------

class TestClockOffset:
    def test_round_trip_scalar(self):
        clk = ClockOffset(bias_s=0.12, rate_ppm=40.0, t_ref=0.0)
        for t in (0.0, 1.0, 37.5, 120.0):
            assert clk.to_true(clk.to_reported(t)) == pytest.approx(t, abs=1e-12)

    def test_round_trip_array(self):
        clk = ClockOffset(bias_s=-0.03, rate_ppm=-75.0, t_ref=10.0)
        t = np.linspace(0.0, 200.0, 17)
        assert np.allclose(clk.to_true(clk.to_reported(t)), t, atol=1e-9)

    def test_positive_bias_means_clock_reads_ahead(self):
        clk = ClockOffset(bias_s=0.1)
        assert clk.to_reported(5.0) > 5.0
        assert clk.residual_s(5.0) == pytest.approx(0.1)

    def test_negative_bias_means_clock_reads_behind(self):
        assert ClockOffset(bias_s=-0.1).to_reported(5.0) < 5.0

    def test_rate_property(self):
        assert ClockOffset(rate_ppm=20.0).rate == pytest.approx(2e-5)

    def test_zero_offset_is_identity(self):
        clk = ClockOffset()
        assert clk.to_reported(3.3) == pytest.approx(3.3)
        assert clk.to_true(3.3) == pytest.approx(3.3)

    @pytest.mark.parametrize("field", ["bias_s", "rate_ppm", "t_ref"])
    def test_non_finite_rejected(self, field):
        with pytest.raises(ValueError):
            ClockOffset(**{field: np.inf})

    def test_repr_mentions_units(self):
        assert "ms" in repr(ClockOffset(bias_s=0.05)) and "ppm" in repr(ClockOffset())


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------

class TestObservabilityHelpers:
    def test_fisher_information_uses_centred_speeds(self):
        assert bias_fisher_information([10.0, 20.0], 1.0) == pytest.approx(50.0)

    def test_constant_speed_carries_no_information(self):
        assert bias_fisher_information([20.0] * 500, 1.0) == pytest.approx(0.0, abs=1e-18)

    def test_constant_speed_crlb_is_infinite(self):
        assert crlb_bias_std([20.0] * 500, 1.0) == np.inf

    def test_standstill_crlb_is_infinite(self):
        assert crlb_bias_std([0.0] * 1000, 1.0) == np.inf

    def test_crlb_matches_closed_form(self):
        v = speed_profile(np.arange(0.0, 60.0, 0.1))
        expected = 1.0 / np.sqrt(np.sum((v - v.mean()) ** 2))
        assert crlb_bias_std(v, 1.0) == pytest.approx(expected)

    def test_sigma_scales_information(self):
        a = bias_fisher_information([10.0, 20.0], 1.0)
        b = bias_fisher_information([10.0, 20.0], 2.0)
        assert a / b == pytest.approx(4.0)

    def test_empty_speeds_return_zero_information(self):
        assert bias_fisher_information([], 1.0) == 0.0

    def test_bad_sigma_rejected(self):
        with pytest.raises(ValueError):
            bias_fisher_information([1.0], 0.0)
        with pytest.raises(ValueError):
            bias_fisher_information([1.0], np.nan)

    def test_non_finite_speed_rejected(self):
        with pytest.raises(ValueError):
            bias_fisher_information([1.0, np.nan], 1.0)

    def test_speeds_must_be_1d(self):
        with pytest.raises(ValueError):
            bias_fisher_information(np.ones((2, 2)), 1.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_dimension_without_rate(self):
        f = make_filter(estimate_rate=False)
        assert f.state_dim == 3
        assert f.rate == 0.0 and f.rate_std == 0.0

    def test_dimension_with_rate(self):
        assert make_filter(estimate_rate=True).state_dim == 4

    def test_bad_x0_shape(self):
        with pytest.raises(ValueError):
            OffsetAugmentedFilter(CVModel(n_dim=1), np.zeros(3), np.eye(2))

    def test_bad_P0_shape(self):
        with pytest.raises(ValueError):
            OffsetAugmentedFilter(CVModel(n_dim=1), np.zeros(2), np.eye(3))

    def test_bad_obs_sigma(self):
        with pytest.raises(ValueError):
            OffsetAugmentedFilter(CVModel(n_dim=1), np.zeros(2), np.eye(2), obs_sigma=0.0)

    def test_negative_std_rejected(self):
        with pytest.raises(ValueError):
            OffsetAugmentedFilter(CVModel(n_dim=1), np.zeros(2), np.eye(2), bias_std0=-1.0)

    def test_non_model_rejected(self):
        with pytest.raises(TypeError):
            OffsetAugmentedFilter(object(), np.zeros(2), np.eye(2))

    def test_p0_populated(self):
        assert make_filter(bias_std0=0.3).bias_std == pytest.approx(0.3)

    def test_two_dimensional_state(self):
        f = OffsetAugmentedFilter(CVModel(n_dim=2), np.zeros(4), np.eye(4))
        f.update(0.0, np.array([1.0, 2.0]))
        assert f.position.shape == (2,)
        f.update_velocity(1.0, np.array([0.0, 5.0]), 0.1)
        assert f.velocity[1] == pytest.approx(5.0, rel=0.05)

    def test_labels(self):
        f = make_filter(bias0=0.05, estimate_rate=True)
        assert f.bias == pytest.approx(0.05)
        assert f.state_dim == 4


# ---------------------------------------------------------------------------
# 0. The clock is driven by the reference channel
# ---------------------------------------------------------------------------

class TestClockIsDrivenByReference:
    def test_update_does_not_advance_the_clock(self):
        f = make_filter(v0=20.0)
        f.update(10.0, np.array([200.0]))
        assert f.t == 0.0

    def test_predict_to_advances_the_clock(self):
        f = make_filter(v0=20.0)
        f.update_velocity(10.0, np.array([20.0]), 0.01)
        assert f.t == pytest.approx(10.0)

    def test_offset_estimate_is_symmetric_in_the_sign_of_b(self):
        pos = [simulate(b_true=b, b0=0.0, seed=s).bias for b in (0.1, 0.2) for s in (0, 1)]
        neg = [simulate(b_true=-b, b0=0.0, seed=s).bias for b in (0.1, 0.2) for s in (0, 1)]
        for est, truth in zip(pos, [0.1, 0.1, 0.2, 0.2]):
            assert est == pytest.approx(truth, abs=0.03)
        for est, truth in zip(neg, [-0.1, -0.1, -0.2, -0.2]):
            assert est == pytest.approx(truth, abs=0.03)

    def test_estimator_is_accurate_across_the_offset_range(self):
        for b in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3):
            est = simulate(b_true=b, b0=0.0, seed=15).bias
            assert est == pytest.approx(b, abs=0.03), f"b={b}"


# ---------------------------------------------------------------------------
# 1. Standstill: unobservable -- and harmless
# ---------------------------------------------------------------------------

class TestStandstill:
    def test_offset_information_does_not_accumulate(self):
        f = make_filter(v0=0.0, bias_std0=0.5)
        _, t_rep, z = straight_line(0.0, 0.0, 30.0, 0.1, b=0.25, seed=1)
        run_plain(f, t_rep, z)
        assert f.bias_std == pytest.approx(0.5, rel=2e-3)

    def test_no_informative_updates_at_rest(self):
        f = make_filter(v0=0.0)
        _, t_rep, z = straight_line(0.0, 0.0, 10.0, 0.1, b=0.5, seed=2)
        run_plain(f, t_rep, z)
        assert f.diag.n_informative == 0
        assert f.diag.n_updates == 100

    def test_estimate_keeps_the_prior(self):
        f = make_filter(v0=0.0, bias0=0.2, bias_std0=0.5)
        _, t_rep, z = straight_line(0.0, 0.0, 20.0, 0.1, b=0.4, seed=4)
        run_plain(f, t_rep, z)
        assert f.bias == pytest.approx(0.2, abs=0.05)

    def test_huge_offset_is_harmless_at_rest(self):
        # 1 s of clock offset, but the body never moves -> no induced error.
        f = make_filter(v0=0.0, bias_std0=2.0)
        _, t_rep, z = straight_line(0.0, 0.0, 20.0, 0.1, b=1.0, seed=3)
        run_plain(f, t_rep, z)
        assert abs(f.position[0]) < 0.5      # truth is 0; stay inside the noise band


# ---------------------------------------------------------------------------
# 2. Constant speed: degenerate with the initial position
# ---------------------------------------------------------------------------

class TestConstantSpeedDegeneracy:
    def test_posterior_depends_on_the_position_prior(self):
        """A constant-speed drive cannot calibrate the offset (with loose p0)."""
        f = simulate(vary=False, b0=0.0, P_pos=1e4, seed=5)
        assert abs(f.bias - 0.12) > 0.03
        assert f.bias_std > 0.1                 # P_bb barely moved
        assert f.diag.n_updates > 100

    def test_two_priors_give_two_answers_on_identical_data(self):
        a = simulate(vary=False, b0=0.0, P_pos=1e4, seed=6)
        b = simulate(vary=False, b0=0.2, P_pos=1e4, seed=6)
        assert abs(b.bias - a.bias) > 0.1

    def test_offset_and_start_position_are_interchangeable(self):
        """Deterministic degeneracy: shift the clock by D and the start by v*D."""
        v, b, shift = 18.0, 0.12, 0.05
        t_rep = np.arange(20.0, 40.0, 0.1)
        clk = ClockOffset(bias_s=b)
        tau = clk.to_true(t_rep)
        z_original = v * tau                       # x0 = 0
        clk_alt = ClockOffset(bias_s=b + shift)
        tau_alt = clk_alt.to_true(t_rep)
        z_alternative = v * shift + v * tau_alt    # x0 = v * shift
        assert np.allclose(z_original, z_alternative, atol=1e-9)

    def test_tight_position_prior_removes_the_degeneracy(self):
        f = simulate(vary=False, b0=0.0, P_pos=1e-4, seed=8)
        assert f.bias == pytest.approx(0.12, abs=0.02)

    def test_no_reference_clock_leaves_the_posterior_at_the_prior(self):
        a = simulate(vary=True, use_odo=False, b0=0.0, P_pos=1e4, P_vel=1e4, seed=9)
        b = simulate(vary=True, use_odo=False, b0=0.2, P_pos=1e4, P_vel=1e4, seed=9)
        assert abs(b.bias - a.bias) > 0.1


# ---------------------------------------------------------------------------
# 3. Varying speed + odometry: observable
# ---------------------------------------------------------------------------

class TestVaryingSpeedObservability:
    def test_offset_converges_to_truth(self):
        f = simulate(vary=True, b0=0.0, P_pos=1e4, seed=10)
        assert f.bias == pytest.approx(0.12, abs=0.02)
        assert f.bias_std < 0.01

    def test_posterior_is_prior_independent(self):
        a = simulate(vary=True, b0=0.0, P_pos=1e4, seed=11)
        b = simulate(vary=True, b0=0.2, P_pos=1e4, seed=11)
        assert a.bias == pytest.approx(b.bias, abs=0.01)

    def test_posterior_std_matches_the_crlb(self):
        f = simulate(vary=True, b0=0.0, P_pos=1e4, seed=12)
        v = speed_profile(np.arange(10.0, 70.0, 0.1))
        assert f.bias_std == pytest.approx(crlb_bias_std(v, 1.0), rel=0.5)

    def test_empirical_scatter_matches_the_crlb(self):
        ests = [simulate(vary=True, b0=0.0, P_pos=1e4, seed=200 + k).bias
                for k in range(12)]
        emp = float(np.std(ests, ddof=1))
        v = speed_profile(np.arange(10.0, 70.0, 0.1))
        assert emp == pytest.approx(crlb_bias_std(v, 1.0), rel=0.6)

    def test_bias_std_monotone_decreasing_after_manoeuvres(self):
        f = simulate(vary=True, b0=0.0, P_pos=1e4, seed=14)
        hist = np.asarray([h[2] for h in f.history[200:]])
        assert np.all(np.diff(hist) < 1e-9)

    def test_negative_offset_also_converges(self):
        f = simulate(b_true=-0.08, vary=True, b0=0.0, P_pos=1e4, seed=15)
        assert f.bias == pytest.approx(-0.08, abs=0.02)

    def test_position_error_stays_near_the_measurement_noise(self):
        f = simulate(vary=True, b0=0.0, P_pos=1e4, seed=16)
        assert float(np.sqrt(np.mean(f.errors ** 2))) < 1.5

    def test_position_error_is_not_biased_by_the_offset(self):
        f = simulate(vary=True, b0=0.0, P_pos=1e4, seed=17)
        assert abs(float(np.mean(f.errors))) < 0.2


class TestNaiveBaseline:
    def test_zero_offset_baseline_trails_by_speed_times_bias(self):
        v, b = 20.0, 0.15
        _, t_rep, z = straight_line(v, 0.0, 20.0, 0.1, b=b, sigma=0.05, seed=8)
        naive = make_filter(v0=v, bias0=0.0, bias_std0=0.0, sigma=0.05, P_pos=1e-4)
        run_plain(naive, t_rep, z)
        err = naive.position[0] - v * naive.t
        assert err == pytest.approx(-v * b, rel=0.05)     # behind the truth
        assert err < 0.0

    def test_online_beats_the_zero_offset_baseline(self):
        v, b = 20.0, 0.12
        _, t_rep, z = straight_line(v, 0.0, 40.0, 0.1, b=b, seed=18)
        online = make_filter(v0=v, bias_std0=0.5, P_pos=1e-4)
        naive = make_filter(v0=v, bias0=0.0, bias_std0=0.0, P_pos=1e-4)
        for t, zi in zip(t_rep, z):
            for flt in (online, naive):
                flt.predict_to(float(t))     # clock driven by the stream at hand
                flt.update(float(t), np.array([zi]))
        e_online = abs(online.position[0] - v * online.t)
        e_naive = abs(naive.position[0] - v * naive.t)
        assert e_naive > 0.5 * v * b
        assert e_online < e_naive


# ---------------------------------------------------------------------------
# Oracle equivalence / rate estimation
# ---------------------------------------------------------------------------

class TestOracleEquivalence:
    def test_known_offset_matches_plain_filter_on_corrected_stamps(self):
        v, b, n, sigma = 12.0, 0.1, 200, 0.5
        _, t_rep, z = straight_line(v, 0.0, n * 0.1, 0.1, b=b, sigma=sigma, seed=30)

        oracle = make_filter(v0=v, sigma=sigma, bias0=b, bias_std0=0.0, P_pos=1e-6)
        model = CVModel(n_dim=1, accel_psd=1e-4)
        H, R = model.observe_position(sigma)
        plain = DelayedFusionFilter(model, x0=np.array([0.0, v]),
                                    P0=np.diag([1e-6, 1e-6]), mode="oosm")

        for t, zi in zip(t_rep, z):
            oracle.predict_to(float(t) - b)          # drive with the corrected clock
            oracle.update(float(t), np.array([zi]))
            plain.add_measurement(DelayedMeasurement(
                t_meas=float(t) - b, t_recv=float(t),
                z=np.array([zi]), H=H, R=R))

        assert oracle.position[0] == pytest.approx(plain.x[0], abs=1e-9)
        assert oracle.velocity[0] == pytest.approx(plain.x[1], abs=1e-9)

    def test_wide_prior_converges_to_the_same_place_as_a_tight_one(self):
        a = simulate(vary=True, b0=0.0, b_std0=0.5, P_pos=1e4, seed=31)
        b = simulate(vary=True, b0=0.0, b_std0=1.0, P_pos=1e4, seed=31)
        assert a.bias == pytest.approx(b.bias, abs=1e-2)


class TestRateEstimation:
    def test_rate_converges_over_a_long_baseline(self):
        f = simulate(b_true=0.05, rate_ppm=100.0, vary=True, estimate_rate=True,
                     b0=0.0, P_pos=1e4, t_end=200.0, seed=40)
        assert f.rate > 0.0
        assert f.rate == pytest.approx(100e-6, abs=8e-5)    # weak: see doc

    def test_rate_unobservable_at_rest(self):
        f = make_filter(v0=0.0, estimate_rate=True, bias_std0=1e-3)
        _, t_rep, z = straight_line(0.0, 0.0, 60.0, 0.1, b=0.2, seed=41, rate_ppm=50.0)
        run_plain(f, t_rep, z)
        assert f.rate_std == pytest.approx(2e-4, rel=1e-3)
        assert abs(f.rate) < 1e-5          # ~0 ppm: no meaningful rate information

    def test_rate_std_shrinks_when_moving(self):
        f = simulate(b_true=0.05, rate_ppm=60.0, vary=True, estimate_rate=True,
                     b0=0.0, P_pos=1e4, t_end=100.0, seed=42)
        assert f.rate_std < 2e-4


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

class TestMechanics:
    def test_backwards_predict_raises(self):
        f = make_filter()
        f.predict_to(1.0)
        with pytest.raises(ValueError):
            f.predict_to(0.5)

    def test_non_finite_predict_raises(self):
        with pytest.raises(ValueError):
            make_filter().predict_to(np.inf)

    def test_non_finite_stamp_raises(self):
        with pytest.raises(ValueError):
            make_filter().update(np.nan, np.array([0.0]))

    def test_non_finite_z_raises(self):
        with pytest.raises(ValueError):
            make_filter().update(0.0, np.array([np.nan]))

    def test_wrong_z_shape_raises(self):
        with pytest.raises(ValueError):
            make_filter().update(0.0, np.array([0.0, 0.0]))

    def test_bad_sigma_raises(self):
        with pytest.raises(ValueError):
            make_filter().update(0.0, np.array([0.0]), sigma=-1.0)

    def test_velocity_channel_validation(self):
        f = make_filter()
        with pytest.raises(ValueError):
            f.update_velocity(np.nan, np.array([0.0]), 0.1)
        with pytest.raises(ValueError):
            f.update_velocity(0.0, np.array([0.0, 1.0]), 0.1)
        with pytest.raises(ValueError):
            f.update_velocity(0.0, np.array([np.nan]), 0.1)
        with pytest.raises(ValueError):
            f.update_velocity(0.0, np.array([0.0]), 0.0)

    def test_diagnostics_counters(self):
        f = make_filter(v0=10.0, bias0=0.1)
        f.predict_to(1.0)
        f.update(1.0, np.array([10.0]))          # true time 0.9 -> behind
        f.update(2.0, np.array([20.0]))          # true time 1.9 -> ahead
        f.update_velocity(3.0, np.array([10.0]), 0.1)
        f.update_velocity(2.0, np.array([10.0]), 0.1)   # behind
        d = f.diag.as_dict()
        assert d["n_updates"] == 2 and d["n_velocity_updates"] == 2
        assert d["n_behind"] == 2 and d["n_ahead"] == 1
        assert d["n_informative"] >= 1

    def test_history_covers_prior_plus_updates(self):
        f = make_filter(v0=5.0)
        _, t_rep, z = straight_line(5.0, 0.0, 5.0, 0.1, seed=50)
        run_plain(f, t_rep, z)
        assert len(f.history) == f.diag.n_updates + f.diag.n_velocity_updates + 1

    def test_history_can_be_disabled(self):
        f = make_filter(v0=5.0)
        f.record_history = False
        f.history.clear()
        f.update(0.0, np.array([0.0]))
        assert f.history == []

    def test_covariance_stays_symmetric_and_pd(self):
        f = simulate(vary=True, estimate_rate=True, P_pos=1e4, seed=51)
        assert np.allclose(f.P, f.P.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(f.P) > -1e-15)

    def test_predict_position_to_matches_cv(self):
        f = make_filter(v0=8.0)
        f.update_velocity(0.0, np.array([8.0]), 1e-3)
        p, cov = f.predict_position_to(0.5)
        assert p[0] == pytest.approx(f.position[0] + f.velocity[0] * 0.5)
        assert cov.shape == (1, 1) and cov[0, 0] > 0

    def test_predict_position_to_backwards_raises(self):
        f = make_filter(v0=1.0)
        f.predict_to(1.0)
        with pytest.raises(ValueError):
            f.predict_position_to(0.5)

    def test_true_time_and_reported_time_are_inverse(self):
        f = make_filter(bias0=0.07, estimate_rate=True)
        f.x[f._i_d] = 50e-6
        for t in (0.0, 5.0, 40.0):
            assert f.true_time(f.reported_time(t)) == pytest.approx(t, abs=1e-9)

    def test_offset_at_reports_the_staleness(self):
        f = make_filter(bias0=0.12)
        assert f.offset_at() == pytest.approx(0.12, abs=1e-12)
        assert f.offset_at(3.0) == pytest.approx(0.12, abs=1e-12)

    def test_repr_has_units(self):
        s = repr(make_filter(v0=20.0))
        assert "ms" in s and "ppm" in s

    def test_speed_property(self):
        assert make_filter(v0=13.0).speed == pytest.approx(13.0)

    def test_per_update_sigma_override(self):
        f = make_filter(v0=20.0, sigma=1.0, bias0=0.12, bias_std0=1e-9, P_pos=1e-6)
        _, t_rep, z = straight_line(20.0, 0.0, 30.0, 0.1, b=0.12, sigma=0.01, seed=52)
        run_plain(f, t_rep, z, sigma=0.01)
        assert f.bias_std < 1e-3
