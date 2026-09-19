"""Delayed / out-of-sequence measurement fusion demo (v0.22.0).

Shows why fusing a sensor at its *receive* time is wrong, and what
measurement-time fusion with out-of-order reprocessing buys you.

Scene
-----
A target drives a circle (radius 25 m, 0.35 rad/s => ~8.75 m/s).  Three
sensors observe its position, each with a realistic pipeline delay:

    sensor   rate    delay
    ------   ----    ---------------------
    odom     100 Hz   5 ms   (fast, coarse)
    radar     20 Hz  30 ms
    camera    10 Hz  60 ms   (slow, precise)

All three are delivered on their *receive* order, which means the camera
samples routinely land *after* newer radar/odom samples (out of sequence).

Two pipelines are compared
--------------------------
1. ``receive_time`` -- fuse each sample when it arrives (the naive baseline;
   injects the whole pipeline delay as a lag into the state).
2. ``oosm``         -- insert every sample at its measurement time, rewinding
   and replaying history when it arrives late.

Metrics
-------
* Streaming position RMSE: at every arrival, compare the filter's *current*
  position against the truth at the filter's *current* timestamp.  This is
  exactly the quantity an AR-HUD consumer feels as "the marker trails the
  world".
* NEES (position, 2 dof) for consistency.
* Display-time error: at *every* arrival both pipelines must render the pose
  for the same wall-clock instant ``t_recv + 60 ms`` (what the driver actually
  sees).  Comparing each pipeline against its *own* clock would be an unfair
  test -- the naive one would get a shorter horizon for free.  Mean over the
  run is reported.

Saves ``examples/delay_fusion_demo.png``.

Run:
    python3 examples/delay_fusion_demo.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sensor_sim.delay_fusion import (
    CVModel,
    DelayedFusionFilter,
    DelayedMeasurement,
    nees_position,
)

DT_SENSORS = {
    # name: (period, delay, sigma)
    "odom": (0.01, 0.005, 1.20),
    "radar": (0.05, 0.030, 0.60),
    "camera": (0.10, 0.060, 0.25),
}

OMEGA = 0.35          # rad/s
RADIUS = 25.0         # m
T_END = 8.0           # s
DISPLAY_HORIZON = 0.06  # s of forward prediction (HUD display lag)
ACCEL_PSD = 4.0       # m^2/s^3


def truth(t: float) -> np.ndarray:
    return np.array([RADIUS * np.cos(OMEGA * t), RADIUS * np.sin(OMEGA * t)])


def truth_vel(t: float) -> np.ndarray:
    return np.array([-RADIUS * OMEGA * np.sin(OMEGA * t), RADIUS * OMEGA * np.cos(OMEGA * t)])


def build_stream(seed: int = 3):
    """Return measurements sorted by receive time (out-of-order by design)."""
    rng = np.random.default_rng(seed)
    ms = []
    for name, (period, delay, sigma) in DT_SENSORS.items():
        t = period
        while t <= T_END:
            jitter = rng.normal(0.0, delay * 0.15)
            H = np.zeros((2, 4))
            H[:2, :2] = np.eye(2)
            z = truth(t) + rng.normal(0.0, sigma, size=2)
            ms.append(DelayedMeasurement(
                t_meas=t,
                t_recv=t + delay + jitter,
                z=z, H=H, R=np.eye(2) * sigma * sigma, sensor=name))
            t += period
    ms.sort(key=lambda m: m.t_recv)
    return ms


def run(mode: str, ms):
    model = CVModel(n_dim=2, accel_psd=ACCEL_PSD)
    x0 = np.array([RADIUS, 0.0, 0.0, OMEGA * RADIUS], float)
    f = DelayedFusionFilter(model, x0, np.eye(4) * 1.0, t0=0.0,
                            mode=mode, history_s=2.0)
    t_hist, err_hist, nees_hist, lag_hist, disp_hist = [], [], [], [], []
    for m in ms:
        f.add_measurement(m)
        t_cur = f.t
        err = float(np.linalg.norm(f.position - truth(t_cur)))
        t_hist.append(t_cur)
        # Effective reporting lag relative to the arrival clock.
        lag_hist.append(m.t_recv - t_cur)
        err_hist.append(err)
        nees_hist.append(nees_position(f.x, f.P, truth(t_cur), 2))

        # Display-time render: the *same* wall-clock instant for both modes.
        t_disp = m.t_recv + DISPLAY_HORIZON
        x_pred, _ = f.predict_to_time(t_disp)
        disp_hist.append(float(np.linalg.norm(x_pred[:2] - truth(t_disp))))

    return {
        "t": np.array(t_hist),
        "err": np.array(err_hist),
        "nees": np.array(nees_hist),
        "lag": np.array(lag_hist),
        "disp": np.array(disp_hist),
        "rmse": float(np.sqrt(np.mean(np.square(err_hist)))),
        "disp_rmse": float(np.sqrt(np.mean(np.square(disp_hist)))),
        "diag": f.diag.as_dict(),
        "final_t": f.t,
    }


def main() -> None:
    ms = build_stream()
    res = {mode: run(mode, ms) for mode in ("receive_time", "oosm")}

    print("=" * 74)
    print("Delayed-measurement fusion demo (v0.22.0)")
    print(f"  sensors: {', '.join(f'{k}@{1/v[0]:.0f}Hz+{v[1]*1e3:.0f}ms' for k, v in DT_SENSORS.items())}")
    print(f"  measurements: {len(ms)}   target: circle r={RADIUS} m, {OMEGA} rad/s")
    print("=" * 74)
    hdr = f"{'metric':<34}{'receive_time':>16}{'oosm':>16}"
    print(hdr)
    print("-" * len(hdr))
    print(f"{'streaming position RMSE [m]':<34}"
          f"{res['receive_time']['rmse']:>16.3f}{res['oosm']['rmse']:>16.3f}")
    print(f"{'mean effective lag [ms]':<34}"
          f"{res['receive_time']['lag'].mean()*1e3:>16.1f}{res['oosm']['lag'].mean()*1e3:>16.1f}")
    print(f"{'display-time (+60ms) RMSE [m]':<34}"
          f"{res['receive_time']['disp_rmse']:>16.3f}{res['oosm']['disp_rmse']:>16.3f}")
    print(f"{'mean position NEES (2 dof)':<34}"
          f"{res['receive_time']['nees'].mean():>16.3f}{res['oosm']['nees'].mean():>16.3f}")
    print(f"{'out-of-order samples':<34}"
          f"{res['receive_time']['diag']['n_out_of_order']:>16d}{res['oosm']['diag']['n_out_of_order']:>16d}")
    print(f"{'reprocessed updates':<34}"
          f"{res['receive_time']['diag']['n_reprocessed']:>16d}{res['oosm']['diag']['n_reprocessed']:>16d}")
    print(f"{'max rewind [ms]':<34}"
          f"{res['receive_time']['diag']['max_rewind_s']*1e3:>16.1f}{res['oosm']['diag']['max_rewind_s']*1e3:>16.1f}")
    print("=" * 74)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fig.suptitle("Delayed / out-of-sequence measurement fusion (sensor-sim v0.22.0)",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    tt = np.linspace(0, T_END, 400)
    ax.plot(RADIUS * np.cos(OMEGA * tt), RADIUS * np.sin(OMEGA * tt),
            "k--", lw=1.2, label="truth")
    for mode, style in (("receive_time", "C3"), ("oosm", "C0")):
        r = res[mode]
        idx = np.linspace(0, len(r["t"]) - 1, 90).astype(int)
        pts = np.array([truth(r["t"][i]) for i in idx])
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=style, alpha=0.6,
                   label=mode)
    ax.set_aspect("equal")
    ax.set_title("Sampled state along the track (colour = filter)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for mode, style in (("receive_time", "C3"), ("oosm", "C0")):
        r = res[mode]
        ax.plot(r["t"], r["err"], style, lw=1.3,
                label=f"{mode} streaming (RMSE {r['rmse']:.2f} m)")
        ax.plot(r["t"], r["disp"], style, lw=1.0, ls="--", alpha=0.7,
                label=f"... + display +60 ms (RMSE {r['disp_rmse']:.2f} m)")
    ax.set_title("Position error (streaming and at display time)")
    ax.set_xlabel("filter time [s]")
    ax.set_ylabel("|p_est - p_true| [m]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for mode, style in (("receive_time", "C3"), ("oosm", "C0")):
        r = res[mode]
        ax.plot(r["t"], r["lag"] * 1e3, style, lw=1.0, alpha=0.85,
                label=f"{mode} (mean {r['lag'].mean()*1e3:.0f} ms)")
    ax.set_title("Effective lag: arrival clock - reported time")
    ax.set_xlabel("filter time [s]")
    ax.set_ylabel("lag [ms]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for mode, style in (("receive_time", "C3"), ("oosm", "C0")):
        r = res[mode]
        ax.plot(r["t"], r["nees"], style, lw=1.0, alpha=0.8, label=mode)
    ax.axhline(2.0, color="k", ls=":", lw=1, label="E[NEES] = 2 dof")
    ax.set_title("Position NEES (filter consistency)")
    ax.set_xlabel("filter time [s]")
    ax.set_ylabel("NEES")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "examples/delay_fusion_demo.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
