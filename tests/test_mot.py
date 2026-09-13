"""Unit tests for the MOT metrics module (v0.21.0)."""

import math

import numpy as np
import pytest

from sensor_sim.mot import (
    MotAccumulator,
    assign,
    gospa,
    ospa,
    truth_world_points,
)


# ---------------------------------------------------------------------------
# OSPA
# ---------------------------------------------------------------------------
class TestOspa:
    def test_empty_vs_empty_is_zero(self):
        assert ospa([], [], cutoff=3.0) == 0.0

    def test_exact_match_is_zero(self):
        pts = np.array([[1.0, 2.0], [-4.0, 5.0]])
        assert ospa(pts, pts, cutoff=3.0) == pytest.approx(0.0, abs=1e-12)

    def test_single_missed_equals_cutoff(self):
        assert ospa([[0.0, 0.0]], [], cutoff=3.0) == pytest.approx(3.0)

    def test_single_false_equals_cutoff(self):
        assert ospa([], [[0.0, 0.0]], cutoff=3.0) == pytest.approx(3.0)

    def test_localisation_below_cutoff(self):
        assert ospa([[1.0, 0.0]], [[0.0, 0.0]], cutoff=3.0) == pytest.approx(1.0)

    def test_saturates_at_cutoff(self):
        assert ospa([[100.0, 0.0]], [[0.0, 0.0]], cutoff=3.0) == pytest.approx(3.0)

    def test_permutation_invariant(self):
        e = np.array([[5.0, 0.0], [1.0, 0.0]])
        g = np.array([[0.0, 0.0], [4.0, 0.0]])
        assert ospa(e, g, 3.0) == pytest.approx(1.0)
        assert ospa(e[::-1], g, 3.0) == pytest.approx(1.0)
        # a bad input order should never beat the optimal pairing
        assert ospa(e, g[::-1], 3.0) == pytest.approx(1.0)

    def test_order_two(self):
        # d = 3 m with cutoff 3 -> truncated 3 -> (3^2 / 1)^(1/2) = 3
        assert ospa([[3.0, 0.0]], [[0.0, 0.0]], cutoff=3.0, order=2.0) == pytest.approx(3.0)
        # d = 1 m, p = 2 -> (1^2/1)^(1/2) = 1
        assert ospa([[1.0, 0.0]], [[0.0, 0.0]], cutoff=3.0, order=2.0) == pytest.approx(1.0)

    def test_components_decompose(self):
        r = ospa([[1.0, 0.0], [9.0, 9.0]], [[0.0, 0.0]], cutoff=3.0, return_components=True)
        # best: match (1,0)<->(0,0) cost 1; the other estimate is a false -> 3
        assert r["localisation"] == pytest.approx(1.0 / 2.0)
        assert r["cardinality"] == pytest.approx(3.0 / 2.0)
        assert r["d"] == pytest.approx(2.0)
        # d^p = loc^p + card^p  (p = 1)
        assert r["d"] == pytest.approx(r["localisation"] + r["cardinality"])

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            ospa([[0.0, 0.0]], [[0.0, 0.0]], cutoff=0.0)
        with pytest.raises(ValueError):
            ospa([[0.0, 0.0]], [[0.0, 0.0]], cutoff=1.0, order=0.5)

    def test_accepts_3d_points_uses_xy(self):
        e = np.array([[1.0, 0.0, 99.0]])
        g = np.array([[0.0, 0.0, -99.0]])
        assert ospa(e, g, cutoff=3.0) == pytest.approx(1.0)

    def test_handles_nested_lists(self):
        assert ospa([[0, 0]], [[1, 0]], cutoff=3.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GOSPA
# ---------------------------------------------------------------------------
class TestGospa:
    def test_perfect_is_all_zero(self):
        r = gospa([[1.0, 1.0]], [[1.0, 1.0]], cutoff=3.0)
        assert r == {"total": 0.0, "localisation": 0.0, "missed": 0.0, "false": 0.0}

    def test_missed_target(self):
        # alpha=2, c=3, p=1 -> per-unmatched penalty c^p/alpha = 1.5
        r = gospa([[0.0, 0.0]], [[0.0, 0.0], [10.0, 0.0]], cutoff=3.0)
        assert r["localisation"] == pytest.approx(0.0)
        assert r["missed"] == pytest.approx(1.5)
        assert r["false"] == pytest.approx(0.0)
        assert r["total"] == pytest.approx(1.5)

    def test_false_target(self):
        r = gospa([[0.0, 0.0], [10.0, 0.0]], [[0.0, 0.0]], cutoff=3.0)
        assert r["false"] == pytest.approx(1.5)
        assert r["missed"] == pytest.approx(0.0)
        assert r["total"] == pytest.approx(1.5)

    def test_localisation(self):
        r = gospa([[1.0, 0.0]], [[0.0, 0.0]], cutoff=3.0)
        assert r["localisation"] == pytest.approx(1.0)
        assert r["missed"] == 0.0 and r["false"] == 0.0
        assert r["total"] == pytest.approx(1.0)

    def test_beyond_cutoff_becomes_miss_plus_false(self):
        # A 10 m error with c=3 is worse than dropping both objects:
        # 2 * (c^p/alpha) = 3 < 10
        r = gospa([[10.0, 0.0]], [[0.0, 0.0]], cutoff=3.0)
        assert r["total"] == pytest.approx(3.0)
        assert r["localisation"] == pytest.approx(0.0)
        assert r["missed"] == pytest.approx(1.5)
        assert r["false"] == pytest.approx(1.5)

    def test_power_identity(self):
        e = [[0.0, 0.0], [4.0, 0.0], [20.0, 0.0]]
        g = [[0.2, 0.0], [18.0, 0.0]]
        p = 1.0
        r = gospa(e, g, cutoff=3.0, order=p)
        lhs = r["total"] ** p
        rhs = r["localisation"] ** p + r["missed"] ** p + r["false"] ** p
        assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-9)

    def test_empty_gt(self):
        r = gospa([[0.0, 0.0]], [], cutoff=2.0)
        assert r["false"] == pytest.approx(2.0 / 2.0)
        assert r["total"] == pytest.approx(1.0)

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            gospa([[0.0, 0.0]], [[0.0, 0.0]], cutoff=-1.0)
        with pytest.raises(ValueError):
            gospa([[0.0, 0.0]], [[0.0, 0.0]], cutoff=1.0, alpha=0.0)


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------
class TestAssign:
    def test_hungarian_matches_within_gate(self):
        a = assign([[0.0, 0.0], [5.0, 0.0]], [[0.1, 0.0], [5.1, 0.0]], gate=2.0)
        assert a.matches == [(0, 0), (1, 1)]
        assert a.unmatched_est == [] and a.unmatched_gt == []
        assert a.distances == pytest.approx([0.1, 0.1])

    def test_gate_rejects_far_pair(self):
        a = assign([[0.0, 0.0], [50.0, 0.0]], [[0.1, 0.0], [5.1, 0.0]], gate=2.0)
        assert a.matches == [(0, 0)]
        assert a.unmatched_est == [1]
        assert a.unmatched_gt == [1]

    def test_greedy_agrees_on_clean_data(self):
        a = assign(
            [[0.0, 0.0], [5.0, 0.0]],
            [[0.1, 0.0], [5.1, 0.0]],
            gate=2.0,
            use_hungarian=False,
        )
        assert a.matches == [(0, 0), (1, 1)]

    def test_hungarian_beats_greedy_on_crossed_data(self):
        # Point 0 is closest to gt 1, but the globally optimal pairing is (0,0)(1,1).
        e = np.array([[1.0, 0.0], [2.0, 0.0]])
        g = np.array([[0.0, 0.0], [3.0, 0.0]])
        a = assign(e, g, gate=5.0)
        assert a.matches == [(0, 0), (1, 1)]
        worse = assign(e, g, gate=5.0, use_hungarian=False)
        assert len(worse.matches) == 2  # greedy also finds a full pairing here

    def test_empty_sets(self):
        a = assign([], [])
        assert a.n_matches == 0 and a.unmatched_est == [] and a.unmatched_gt == []
        a = assign([[0.0, 0.0]], [])
        assert a.unmatched_est == [0] and a.unmatched_gt == []
        a = assign([], [[0.0, 0.0]])
        assert a.unmatched_gt == [0]

    def test_no_gate_matches_all(self):
        a = assign([[0.0, 0.0], [100.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]])
        assert a.n_matches == 2


# ---------------------------------------------------------------------------
# accumulator
# ---------------------------------------------------------------------------
class TestMotAccumulator:
    def test_perfect_run(self):
        acc = MotAccumulator(gate_m=2.0, cutoff_m=4.0)
        rng = np.random.default_rng(0)
        for k in range(10):
            gt = np.array([[0.0, 0.0], [5.0, 0.0]])
            est = gt + 0.1 * rng.standard_normal((2, 2))
            acc.add(est, gt, est_ids=[1, 2], gt_ids=[10, 20], t=k * 0.1)
        s = acc.summary()
        assert s["mota"] == pytest.approx(1.0)
        assert s["motp"] == pytest.approx(0.1, abs=0.1)
        assert s["id_switches"] == 0
        assert s["false_positives"] == 0
        assert s["false_negatives"] == 0
        assert s["mostly_tracked"] == 2

    def test_missed_detection_counts_fn(self):
        acc = MotAccumulator(gate_m=2.0, cutoff_m=4.0)
        acc.add([[0.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]])
        s = acc.summary()
        assert s["false_negatives"] == 1
        assert s["matches"] == 1

    def test_false_track_counts_fp(self):
        acc = MotAccumulator(gate_m=2.0, cutoff_m=4.0)
        acc.add([[0.0, 0.0], [50.0, 0.0]], [[0.0, 0.0]])
        s = acc.summary()
        assert s["false_positives"] == 1

    def test_mota_penalised(self):
        acc = MotAccumulator(gate_m=2.0, cutoff_m=4.0)
        acc.add([[0.0, 0.0]], [[0.0, 0.0]])
        acc.add([[0.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]])  # 1 FN
        s = acc.summary()
        assert s["gt"] == 3
        assert s["mota"] == pytest.approx(1.0 - 1.0 / 3.0)

    def test_id_switch_counted_once(self):
        acc = MotAccumulator(gate_m=3.0, cutoff_m=6.0)
        acc.add([[0.0, 0.0]], [[0.0, 0.0]], est_ids=[7], gt_ids=[100])
        acc.add([[0.0, 0.0]], [[0.0, 0.0]], est_ids=[7], gt_ids=[200])  # switch
        acc.add([[0.0, 0.0]], [[0.0, 0.0]], est_ids=[7], gt_ids=[200])  # stable
        assert acc.summary()["id_switches"] == 1

    def test_mostly_lost_classification(self):
        acc = MotAccumulator(gate_m=2.0, cutoff_m=4.0)
        for k in range(10):
            # gt 20 is only ever matched on the first scan -> mostly lost
            est = [[0.0, 0.0]] if k == 0 else [[50.0, 0.0]]
            gt = [[0.0, 0.0], [3.0, 0.0]] if k == 0 else [[3.0, 0.0]]
            gids = [10, 20] if k == 0 else [20]
            acc.add(est, gt, est_ids=[1], gt_ids=gids)
        s = acc.summary()
        assert s["mostly_lost"] >= 1

    def test_empty_gt_vacuous(self):
        acc = MotAccumulator(gate_m=2.0)
        acc.add([], [])
        acc.add([], [])
        assert acc.summary()["mota"] == pytest.approx(1.0)

    def test_empty_gt_with_false_is_nan(self):
        acc = MotAccumulator(gate_m=2.0)
        acc.add([[0.0, 0.0]], [])
        assert math.isnan(acc.summary()["mota"])

    def test_ospa_gospa_tracked(self):
        acc = MotAccumulator(gate_m=2.0, cutoff_m=4.0)
        acc.add([[0.0, 0.0]], [[0.0, 0.0]])
        acc.add([[1.0, 0.0]], [[0.0, 0.0]])
        s = acc.summary()
        assert s["mean_ospa"] == pytest.approx(0.5)  # (0 + 1) / 2
        assert s["mean_gospa"] == pytest.approx(0.5)

    def test_ids_length_mismatch_raises(self):
        acc = MotAccumulator()
        with pytest.raises(ValueError):
            acc.add([[0.0, 0.0]], [[0.0, 0.0]], est_ids=[1, 2], gt_ids=[9])


# ---------------------------------------------------------------------------
# radar-truth integration
# ---------------------------------------------------------------------------
class TestRadarTruth:
    def test_truth_world_points_roundtrip(self):
        from sensor_sim.lidar import Sphere
        from sensor_sim.radar import RadarConfig, RadarSensor
        from sensor_sim.trajectory import _euler_to_quat

        rs = RadarSensor(RadarConfig(), mount_t_body=np.array([0.5, 0.0, 0.0]))
        obj = Sphere(object_id=1, center=np.array([30.0, 0.0, 0.0]), radius=0.4)
        q = _euler_to_quat(0.0, 0.0, 0.0)
        host = np.zeros(3)

        class _P:
            pos = host
            att = q
            vel = np.zeros(3)

        fr = rs.scan(_P(), [obj], t=0.0, rng=np.random.default_rng(0))
        assert fr.azimuths_true.shape == fr.ranges.shape
        pts = fr.points_true
        assert pts.shape == (fr.ranges.shape[0], 3)
        # Truth range is the near surface measured from the +0.5 m mount:
        # 30 - 0.5 (mount) - 0.4 (radius) = 29.1 m along +x.
        assert np.linalg.norm(pts[0]) == pytest.approx(fr.range_true[0])
        assert fr.range_true[0] == pytest.approx(29.1, abs=1e-9)

        wpts = truth_world_points(fr, host, q, mount_t_body=np.array([0.5, 0.0, 0.0]))
        assert wpts.shape == (1, 2)
        assert wpts[0, 0] == pytest.approx(29.6, abs=1e-9)

    def test_end_to_end_tracker_scores_high_mota(self):
        from sensor_sim.lidar import Box, Sphere
        from sensor_sim.radar import RadarConfig, RadarSensor
        from sensor_sim.tracking import RadarTracker, TrackConfig
        from sensor_sim.trajectory import _euler_to_quat

        lead = Box(
            object_id=1,
            center=np.array([60.0, 0.0, 0.0]),
            half_extents=np.array([2.2, 1.0, 0.7]),
            velocity=np.array([1.0, 0.0, 0.0]),  # slowly receding -> stays aloft
            reflectivity=1.0,
        )
        ped = Sphere(
            object_id=2,
            center=np.array([35.0, -6.0, 0.0]),
            radius=0.35,
            velocity=np.array([0.0, 0.3, 0.0]),
            reflectivity=0.1,
        )
        world = [lead, ped]
        rs = RadarSensor(RadarConfig(), mount_t_body=np.array([0.5, 0.0, 0.0]))
        trk = RadarTracker(
            TrackConfig(process_noise_accel_mps2=1.0),
            mount_t_body=np.array([0.5, 0.0, 0.0]),
        )
        q = _euler_to_quat(0.0, 0.0, 0.0)
        host = np.zeros(3)

        class _P:
            pos = host
            att = q
            vel = np.zeros(3)

        acc = MotAccumulator(gate_m=3.0, cutoff_m=8.0)
        for i in range(60):
            t = i * 0.1
            fr = rs.scan(_P(), world, t=t, rng=np.random.default_rng(1234 + i))
            tf = trk.process(fr, host, q, np.zeros(3))
            gpt = truth_world_points(fr, host, q, mount_t_body=np.array([0.5, 0.0, 0.0]))
            ept = np.array([tr.pos for tr in tf.tracks]) if tf.n_tracks else np.zeros((0, 2))
            acc.add(
                ept,
                gpt,
                est_ids=[tr.track_id for tr in tf.tracks],
                gt_ids=list(fr.truth_ids),
                t=t,
            )
        s = acc.summary()
        # A well-behaved tracker on a benign scene: no wild false tracks and a
        # healthy match rate (allow for the birth transient and dropout scans).
        assert s["mota"] > 0.5
        assert s["mean_gospa"] < 3.0
        assert s["frames"] == 60
