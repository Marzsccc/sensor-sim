"""
GNSS outage demo for the AR-HUD display pipeline (v0.8.0).

Shows what happens when GNSS is lost (tunnel / underground garage /
urban canyon) and how the display-time uncertainty ellipse should grow
so the HUD can fade/clamp AR markers:

  1. A 60 s trajectory: straight run, 90-deg right turn, straight.
  2. A 15 s GNSS outage (t = 20-35 s, during the second straight).
  3. ESKF (IMU 200 Hz + wheel 100 Hz + GNSS 10 Hz, fixes dropped in the
     outage) + display-time prediction (67 ms horizon) + covariance.
  4. Report:
       - outage fraction, mean fused/display error
       - radius_95 inside vs outside the outage (the HUD fade signal)
       - NEES consistency (covariance vs true error)
       - recovery: position error right after GNSS returns

Usage:  .venv/bin/python examples/outage_demo.py [--seed 42] [--plot]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.eskf import ESKFConfig
from sensor_sim.outage import (
    OutageConfig,
    OutageSimConfig,
    OutageSimulator,
    consistency,
    nees_vs_outage,
)


def build_trajectory(dt: float) -> Trajectory:
    """Straight + 90-deg right turn + straight (60 s)."""
    wp = [
        Waypoint(t=0.0, pos=np.array([0, 0, 0]), vel=np.array([20, 0, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
        Waypoint(t=20.0, pos=np.array([400, 0, 0]), vel=np.array([20, 0, 0]),
                 att=np.array([1, 0, 0, 0]), omega=np.array([0, 0, 0])),
        Waypoint(t=25.0, pos=np.array([400, 100, 0]), vel=np.array([0, 20, 0]),
                 att=np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]),
                 omega=np.array([0, 0, np.pi / 10])),
        Waypoint(t=60.0, pos=np.array([400, 800, 0]), vel=np.array([0, 20, 0]),
                 att=np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]),
                 omega=np.array([0, 0, 0])),
    ]
    return Trajectory(wp, dt=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true",
                    help="save an error/ellipse plot (needs matplotlib)")
    args = ap.parse_args()

    traj = build_trajectory(0.01)
    outage = OutageConfig(outage_periods=[(20.0, 35.0)])   # 15 s tunnel
    eskf = ESKFConfig(
        acc_noise_density=1e-2, gyr_noise_density=1e-3,
        acc_bias_rw=1e-4, gyr_bias_rw=1e-5,
        init_pos_std=5.0, init_att_std_deg=5.0,
        gnss_pos_std=2.0, gnss_vel_std=0.2,
    )
    cfg = OutageSimConfig(outage=outage, eskf=eskf, seed=args.seed)
    sim = OutageSimulator(cfg)
    res = sim.run(traj)

    print("=== GNSS outage / AR-HUD display (v0.8.0) ===")
    print(f"Trajectory      : 60 s, 20 m/s, one 90-deg right turn")
    print(f"GNSS outage     : 20-35 s (15 s, tunnel)")
    print(f"Display horizon : {cfg.horizon * 1000:.0f} ms")
    print(f"Outage fraction : {res.outage_fraction() * 100:.1f}%")
    print(f"Mean fused err  : {res.mean_fused_error():.2f} m")
    print(f"Mean disp err   : {res.mean_disp_error():.2f} m")

    m = res.in_outage
    print(f"\n--- Display-time uncertainty (HUD fade signal) ---")
    print(f"  radius_95 inside  outage : {res.radius_95[m].mean():.2f} m "
          f"(max {res.radius_95[m].max():.2f})")
    print(f"  radius_95 outside outage : {res.radius_95[~m].mean():.2f} m")

    print(f"\n--- NEES consistency (covariance vs true error) ---")
    for label, sel in (("in-outage", m), ("healthy", ~m)):
        c = consistency(res.disp_error[sel], res.disp_cov[sel], dim=3)
        print(f"  {label:10s}: NEES={c['nees_mean']:.2f} "
              f"CI=[{c['ci_low']:.2f},{c['ci_high']:.2f}] "
              f"consistent={c['consistent']}")

    print(f"\n--- Recovery (GNSS returns at 35 s) ---")
    recovered = (~m) & (res.t > 35.1) & (res.t < 38.0)
    print(f"  mean fused err right after recovery: "
          f"{np.mean(np.linalg.norm(res.fused_error[recovered], axis=1)):.2f} m")
    print(f"  mean fused err at end of outage    : "
          f"{np.mean(np.linalg.norm(res.fused_error[m & (res.t > 33)], axis=1)):.2f} m")
    print(f"  (straight-road heading is unobservable from GNSS+wheel, so a")
    print(f"   full re-convergence needs a turn or map matching)")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            ax[0].plot(res.t, np.linalg.norm(res.fused_error, axis=1),
                       label="fused error")
            ax[0].plot(res.t, np.linalg.norm(res.disp_error, axis=1),
                       label="display error", ls="--")
            ax[0].fill_between(res.t, 0, np.linalg.norm(res.fused_error, axis=1),
                               where=m, alpha=0.3, color="r",
                               label="GNSS outage")
            ax[0].set_ylabel("pos error (m)")
            ax[0].legend()
            ax[1].plot(res.t, res.radius_95, color="tab:orange")
            ax[1].fill_between(res.t, 0, res.radius_95, where=m,
                               alpha=0.3, color="r")
            ax[1].axhline(cfg.fade_radius_95, color="k", ls=":",
                          label=f"fade threshold {cfg.fade_radius_95} m")
            ax[1].set_ylabel("radius_95 (m)")
            ax[1].set_xlabel("t (s)")
            ax[1].legend()
            out = os.path.join(os.path.dirname(__file__),
                               "outage_demo_output.png")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            print(f"\nPlot saved: {out}")
        except ImportError:
            print("(matplotlib not installed; skip plot)")


if __name__ == "__main__":
    main()
