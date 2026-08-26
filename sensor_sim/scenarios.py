"""
Scenario orchestration & batch regression (v0.15.0).

A :class:`Scenario` is a declarative, reproducible experiment:

    trajectory shape + sensor grades + GNSS outages + regression gates

:class:`run_scenario` pushes one scenario through the standard ESKF
pipeline (IMU predict -> GNSS update, skipped during outages) and scores
it with the v0.14 evaluator.  :func:`batch_summary` renders the whole
library as one pass/fail matrix -- the overnight-regression view.

Adding a new estimator later means swapping the inside of
``run_scenario`` while every scenario keeps its gates fixed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .imu import IMUSensor, SensorGrade
from .gnss import GNSSSensor, GNSSGrade
from .trajectory import Trajectory, Waypoint
from .eskf import ESKF, ESKFConfig
from .outage import OutageConfig, OutageModel
from .utils import quat_to_euler
from .evaluation import (
    EvalReport,
    TrajectoryEvaluator,
    add_gate,
)


# ---------------------------------------------------------------------------
# Grade-matched filter configuration (heuristic presets)
# ---------------------------------------------------------------------------

_ESKF_PRESETS: Dict[SensorGrade, Dict[str, float]] = {
    SensorGrade.NAVIGATION: dict(
        acc_noise_density=3e-3, gyr_noise_density=2e-4,
        acc_bias_rw=1e-5, gyr_bias_rw=1e-6),
    SensorGrade.TACTICAL: dict(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5),
    SensorGrade.CONSUMER: dict(
        acc_noise_density=5e-2, gyr_noise_density=3e-3,
        acc_bias_rw=1e-3, gyr_bias_rw=1e-4),
}

_GNSS_STD: Dict[GNSSGrade, Tuple[float, float]] = {
    # (pos_std_m, vel_std_mps)
    GNSSGrade.CONSUMER: (5.0, 0.50),
    GNSSGrade.AUTOMOTIVE: (2.2, 0.10),
    GNSSGrade.RTK: (0.15, 0.05),
    GNSSGrade.SURVEY: (0.03, 0.02),
}


def eskf_config_for(imu_grade: SensorGrade, gnss_grade: GNSSGrade) -> ESKFConfig:
    """Filter tuning matched to the sensor grades under test."""
    pos_std, vel_std = _GNSS_STD[gnss_grade]
    return ESKFConfig(
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=pos_std, gnss_vel_std=vel_std,
        **_ESKF_PRESETS[imu_grade],
    )


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """Declarative, reproducible experiment definition."""
    name: str
    description: str
    waypoints: List[Waypoint]
    imu_grade: SensorGrade = SensorGrade.TACTICAL
    gnss_grade: GNSSGrade = GNSSGrade.AUTOMOTIVE
    dt_imu: float = 0.01
    dt_gnss: float = 0.2
    outage_periods: List[Tuple[float, float]] = field(default_factory=list)
    init_pos_err: Tuple[float, float, float] = (3.0, -2.0, 1.0)
    # gate spec -> (op, threshold); key syntax mirrors add_gate()
    gates: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    seed: int = 42

    @property
    def duration(self) -> float:
        return self.waypoints[-1].t


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _wp(t: float, x: float, y: float, vx: float, vy: float,
        yaw: float = 0.0, omega_z: float = 0.0) -> Waypoint:
    return Waypoint(t=t, pos=np.array([x, y, 0.0]),
                    vel=np.array([vx, vy, 0.0]),
                    att=_yaw_quat(yaw),
                    omega=np.array([0, 0, omega_z]))


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------

def scenario_highway_straight() -> Scenario:
    """40 s @ 25 m/s straight motorway cruise. Baseline sanity."""
    return Scenario(
        name="highway-straight",
        description="40 s @ 25 m/s straight, tac IMU + auto GNSS",
        waypoints=[
            _wp(0.0, 0, 0, 25, 0),
            _wp(20.0, 500, 0, 25, 0),
            _wp(40.0, 1000, 0, 25, 0),
        ],
        gates={
            "pos_err_m": ("<", 2.5),
            "vel_err_ms": ("<", 0.5),
            "yaw_err_deg": ("<", 1.5),
            "nees:nees_hpos@nees_mean": ("<", 150.0),
        },
    )


def scenario_urban_s_curve() -> Scenario:
    """60 s urban route with two 90-deg turns @ 10 m/s (the classic demo)."""
    q90 = np.pi / 2
    return Scenario(
        name="urban-s-curve",
        description="60 s, two 90-deg turns @ 10 m/s",
        waypoints=[
            _wp(0.0, 0, 0, 10, 0),
            _wp(15.0, 150, 0, 10, 0),
            _wp(20.0, 150, 50, 0, 10, yaw=q90, omega_z=np.pi / 10),
            _wp(40.0, 150, 250, 0, 10, yaw=q90),
            _wp(50.0, 50, 250, -10, 0, yaw=q90, omega_z=-np.pi / 10),
            _wp(60.0, -50, 250, -10, 0),
        ],
        gates={
            "pos_err_m": ("<", 3.0),
            "vel_err_ms": ("<", 0.5),
            "yaw_err_deg": ("<", 2.0),
            "nees:nees_hpos@nees_mean": ("<", 150.0),
        },
    )


def scenario_tunnel_outage() -> Scenario:
    """Highway cruise with a 12 s tunnel (full GNSS blackout) mid-route."""
    sc = scenario_highway_straight()
    sc.name = "tunnel-outage"
    sc.description = "straight @ 25 m/s, 12 s GNSS blackout (t=14-26)"
    sc.outage_periods = [(14.0, 26.0)]
    # outage inflates errors: relax position gate vs open-sky baseline
    sc.gates = {
        "pos_err_m": ("<", 12.0),
        "vel_err_ms": ("<", 1.2),
        "yaw_err_deg": ("<", 2.0),
        "nees:nees_hpos@nees_mean": ("<", 400.0),
    }
    return sc


def scenario_parking_garage() -> Scenario:
    """Slow maneuvering with consumer sensors; GNSS dies for the last
    20 s inside a garage.  The 'cheap hardware, worst case' corner."""
    return Scenario(
        name="parking-garage",
        description="consumer IMU/GNSS, slow maneuver, 20 s outage tail",
        waypoints=[
            _wp(0.0, 0, 0, 5, 0),
            _wp(10.0, 50, 0, 5, 0),
            _wp(16.0, 50, 30, 0, 5, yaw=np.pi / 2, omega_z=np.pi / 12),
            _wp(24.0, 50, 70, 0, 5, yaw=np.pi / 2),
            _wp(32.0, 20, 70, -5, 0, yaw=np.pi / 2, omega_z=-np.pi / 12),
            _wp(44.0, -40, 70, -5, 0),
        ],
        imu_grade=SensorGrade.CONSUMER,
        gnss_grade=GNSSGrade.CONSUMER,
        outage_periods=[(24.0, 44.0)],
        # Known-weak corner: consumer IMU drift dominates once GNSS drops.
        # Gates below encode "no worse than cheap-hardware physics", NOT
        # HUD requirements -- measured baseline (seed 42):
        #   open sky pos 16.2 m / vel 2.7 m/s; with 20 s outage
        #   pos 22.4 m / vel 4.2 m/s.  A future estimator must beat THIS.
        gates={
            "pos_err_m": ("<", 30.0),
            "vel_err_ms": ("<", 6.0),
            "yaw_err_deg": ("<", 8.0),
            "nees:nees_hpos@nees_mean": ("<", 800.0),
        },
    )


def scenario_rtk_baseline() -> Scenario:
    """Urban shape with RTK + tactical IMU.  Tight-gate golden reference."""
    sc = scenario_urban_s_curve()
    sc.name = "rtk-baseline"
    sc.description = "S-curve w/ RTK cm-level GNSS + tactical IMU"
    sc.gnss_grade = GNSSGrade.RTK
    sc.gates = {
        "pos_err_m": ("<", 0.6),
        "vel_err_ms": ("<", 0.25),
        "yaw_err_deg": ("<", 1.5),
        "nees:nees_hpos@nees_mean": ("<", 150.0),
    }
    return sc


def default_library() -> List[Scenario]:
    """The standard regression suite, ordered cheap -> expensive corners."""
    return [
        scenario_highway_straight(),
        scenario_urban_s_curve(),
        scenario_tunnel_outage(),
        scenario_parking_garage(),
        scenario_rtk_baseline(),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario, seed: Optional[int] = None,
                 verbose: bool = False) -> EvalReport:
    """Run one scenario through the standard ESKF pipeline and score it.

    ``seed`` overrides ``scenario.seed`` for sweeps over random seeds.
    """
    if seed is not None:
        scenario.seed = int(seed)

    traj = Trajectory(scenario.waypoints, dt=scenario.dt_imu)
    imu = IMUSensor(scenario.imu_grade, dt=scenario.dt_imu, seed=scenario.seed)
    gnss = GNSSSensor(scenario.gnss_grade, dt=scenario.dt_imu,
                      seed=scenario.seed + 1)
    outage = OutageModel(OutageConfig(outage_periods=scenario.outage_periods))

    eskf = ESKF(eskf_config_for(scenario.imu_grade, scenario.gnss_grade))
    p0 = traj.points[0]
    eskf.set_initial_state(p0.t,
                           p0.pos + np.asarray(scenario.init_pos_err),
                           p0.vel, p0.att)

    ev = TrajectoryEvaluator(scenario.name)
    gnss_next = scenario.dt_gnss

    for pt in traj.points:
        t = pt.t
        a_m, w_m = imu.measure(traj.body_accel(pt), pt.omega)
        eskf.predict(a_m, w_m, scenario.dt_imu)

        if not outage.drop(t) and t >= gnss_next:
            pos_m, vel_m, valid = gnss.measure(pt.pos, pt.vel)
            if valid:
                eskf.update_gnss(pos_m, vel_m)
                gnss_next += scenario.dt_gnss

        st = eskf.state
        ev.add_pose(t, st.p, pt.pos, P=st.P[np.ix_([0, 1], [0, 1])],
                    dims=(0, 1))
        ev.add_velocity(t, st.v, pt.vel)
        ev.add_yaw(t, float(quat_to_euler(st.q)[2]),
                   float(quat_to_euler(pt.att)[2]))

    rep = ev.finalize()
    for key, (op, thr) in scenario.gates.items():
        add_gate(rep, key, op, thr)

    if verbose:
        print(rep.summary())
    return rep


# ---------------------------------------------------------------------------
# Batch rendering
# ---------------------------------------------------------------------------

def batch_summary(reports: Sequence[EvalReport]) -> str:
    """One-line-per-scenario pass/fail matrix."""
    head = (f"{'scenario':>18s} {'pos_rmse':>9s} {'vel_rmse':>9s} "
            f"{'yaw_rmse':>9s} {'NEES':>8s}  gates")
    lines = [head, "-" * len(head)]
    n_fail = 0
    for r in reports:
        ok = r.all_gates_passed()
        n_fail += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        nees = r.nees.get("nees_hpos", {})
        nees_txt = f"{nees.get('nees_mean', float('nan')):8.1f}"
        lines.append(
            f"{r.name:>18s} "
            f"{r.metric('pos_err_m') or float('nan'):9.3f} "
            f"{r.metric('vel_err_ms') or float('nan'):9.4f} "
            f"{r.metric('yaw_err_deg') or float('nan'):9.3f} "
            f"{nees_txt}  [{mark}]")
    lines.append("-" * len(head))
    total = len(reports)
    lines.append(f"=> {total - n_fail}/{total} scenarios passed"
                 + ("  ✅" if n_fail == 0 else f"  ❌ ({n_fail} failed)"))
    return "\n".join(lines)
