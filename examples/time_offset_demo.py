"""Online sensor-clock-offset estimation on an urban drive (v0.24.0).

Shows the failure mode that v0.22.0/v0.23.0 could not fix: the reference
stream's *timestamps themselves* are wrong.  The GNSS receiver runs its own
oscillator, 120 ms fast, so every fix describes a scene that is 120 ms older
than its stamp claims and the pose built from it trails reality by
``speed x 0.12 s`` -- 2.6 m at 22 m/s, i.e. two lane widths of "marker trails
the world" that no rewind/replay can undo.  What does undo it is estimating the
offset online, from the mismatch between the fix stream and the wheel-odometry
speed that shares the host clock.

Scene
-----
90 s urban drive: standstill 0-15 s, acceleration, a slow section, a fast
section, a stop at the end; wheel-odometry speed at 20 Hz on the reference
clock (it is what drives the filter's clock); GNSS position fixes at 10 Hz with
1 m noise and a **+120 ms** clock offset.

Three pipelines
---------------
1. ``naive``  -- trust the stamps (offset pinned to 0): trails by ``v * b``.
2. ``online`` -- estimate the offset jointly with the state (this module).
3. ``oracle`` -- know the offset exactly: the noise-floor reference.

The standstill is the interesting part: the offset is *unobservable* while
``v == 0`` -- and simultaneously harmless, since the induced error is ``v * b``.
The estimate therefore only starts moving once the vehicle pulls away, and the
demo scales the achievable accuracy against the closed-form bound
``sigma / sqrt(sum((v - mean(v))^2))``.

The plotted run is the *median* run of a 9-seed ensemble (not the best one);
the ensemble scatter is printed and compared with that bound.

Saves ``examples/time_offset_demo.png``.

Run:
    PYTHONPATH=. uv run --with numpy --with matplotlib python examples/time_offset_demo.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sensor_sim.delay_fusion import CVModel
from sensor_sim.time_offset import ClockOffset, OffsetAugmentedFilter, crlb_bias_std

B_TRUE = 0.120          # s, GNSS clock fast by 120 ms
DT_POS = 0.1            # 10 Hz fixes
DT_VEL = 0.05           # 20 Hz wheel odometry (drives the filter clock)
SIGMA_POS = 1.0         # m   (automotive GNSS)
SIGMA_VEL = 0.05        # m/s (wheel encoder)
T_END = 90.0
TARGETS = [(0.0, 0.0), (15.0, 20.0), (45.0, 8.0), (55.0, 22.0), (75.0, 0.0)]
TAU = 3.0               # s, first-order speed lag
N_SEEDS = 9


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def make_truth(dt=1e-3):
    """First-order-tracked speed targets -> smooth speed + integrated position."""
    ts = np.arange(0.0, T_END + dt, dt)
    speeds = np.zeros_like(ts)
    v = 0.0
    k = 0
    for i in range(1, ts.size):
        t = ts[i]
        while k + 1 < len(TARGETS) and t >= TARGETS[k + 1][0]:
            k += 1
        v += (TARGETS[k][1] - v) * (1.0 - np.exp(-dt / TAU))
        speeds[i] = v
    pos = np.concatenate([[0.0], np.cumsum((speeds[1:] + speeds[:-1]) * 0.5 * dt)])
    return ts, speeds, pos


TS, SPEED, POS = make_truth()


def speed_at(t):
    return float(np.interp(t, TS, SPEED))


def pos_at(t):
    return float(np.interp(t, TS, POS))


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def build(role: str) -> OffsetAugmentedFilter:
    model = CVModel(n_dim=1, accel_psd=5e-2)
    if role == "naive":
        bias0, bias_std0 = 0.0, 0.0
    elif role == "oracle":
        bias0, bias_std0 = B_TRUE, 0.0
    else:
        bias0, bias_std0 = 0.0, 0.5
    return OffsetAugmentedFilter(
        model,
        x0=np.array([pos_at(0.0), speed_at(0.0)]),
        P0=np.diag([1e2, 1e-2]),
        obs_sigma=SIGMA_POS,
        estimate_rate=False,
        bias0=bias0,
        bias_std0=bias_std0,
        bias_rw_std=0.0,
        rate_rw_std=0.0,
        record_history=True,
    )


def run(role: str, seed: int):
    rng = np.random.default_rng(seed)
    clk = ClockOffset(bias_s=B_TRUE, t_ref=0.0)
    f = build(role)

    t_pos = np.arange(0.0, T_END, DT_POS)
    t_vel = np.arange(0.0, T_END, DT_VEL)
    rec = {k: [] for k in ("t", "err", "speed", "bias", "bias_std")}

    iv = ip = 0
    while ip < t_pos.size:
        t_rep = float(clk.to_reported(t_pos[ip]))
        if iv < t_vel.size and t_vel[iv] <= t_rep:
            f.update_velocity(float(t_vel[iv]),
                              np.array([speed_at(t_vel[iv])
                                        + rng.normal(0.0, SIGMA_VEL)]),
                              SIGMA_VEL)
            iv += 1
            continue
        f.update(t_rep, np.array([pos_at(t_pos[ip]) + rng.normal(0.0, SIGMA_POS)]))
        ip += 1
        rec["t"].append(f.t)
        rec["speed"].append(speed_at(f.t))
        rec["err"].append(f.position[0] - pos_at(f.t))
        rec["bias"].append(f.bias)
        rec["bias_std"].append(f.bias_std)

    rec = {k: np.asarray(v) for k, v in rec.items()}
    moving = rec["speed"] > 1.0
    rec["rmse_moving"] = float(np.sqrt(np.mean(rec["err"][moving] ** 2)))
    rec["rmse_all"] = float(np.sqrt(np.mean(rec["err"] ** 2)))
    # the vehicle is stopped for the last ~10 s, so the estimate is frozen there:
    # score the last value it could actually still learn from (t = 78 s)
    i_freeze = int(np.argmin(np.abs(rec["t"] - 78.0)))
    rec["bias_final"] = float(rec["bias"][i_freeze])
    rec["bias_std_final"] = float(rec["bias_std"][i_freeze])
    return f, rec


def main() -> None:
    # ensemble: pick the median run to plot, so the figure is not a lucky draw
    ensemble = [run("online", s)[1]["bias_final"] for s in range(N_SEEDS)]
    median = float(np.median(ensemble))
    seed = int(np.argmin([abs(b - median) for b in ensemble]))

    results = {}
    for role in ("naive", "online", "oracle"):
        f, rec = run(role, seed)
        results[role] = (f, rec)
        print(f"[{role:6s}] offset estimate {rec['bias_final'] * 1e3:+8.2f} ms "
              f"(truth {B_TRUE * 1e3:+.1f} ms, +-{rec['bias_std_final'] * 1e3:.2f}) "
              f"| pos RMSE moving {rec['rmse_moving']:6.3f} m | all {rec['rmse_all']:6.3f} m")

    f_on, rec_on = results["online"]
    _, rec_na = results["naive"]
    _, rec_or = results["oracle"]

    v_max = float(np.max(SPEED))
    print(f"\nplotted seed = {seed} (median of {N_SEEDS} seeds)")
    print(f"ensemble offset error: {np.mean(ensemble) * 1e3:.2f} ms "
          f"(bias vs truth {np.mean(ensemble) * 1e3 - B_TRUE * 1e3:+.2f} ms, "
          f"scatter {np.std(ensemble, ddof=1) * 1e3:.2f} ms)")
    print(f"speed x offset = {v_max:.1f} m/s x {B_TRUE * 1e3:.0f} ms "
          f"= {v_max * B_TRUE:.2f} m  (the naive pipeline's peak bias)")
    print(f"naive / online RMSE ratio = {rec_na['rmse_moving'] / rec_on['rmse_moving']:.2f}x")
    print(f"online / oracle RMSE ratio = {rec_on['rmse_moving'] / rec_or['rmse_moving']:.2f}x")

    # analytic bound from the reference speed sequence
    speeds = np.array([speed_at(t) for t in np.arange(0.0, T_END, DT_POS)])
    crlb = crlb_bias_std(speeds, SIGMA_POS)
    print(f"\nCRLB on the offset from this drive: {crlb * 1e3:.2f} ms "
          f"(estimator std {rec_on['bias_std_final'] * 1e3:.2f} ms)")
    print(f"diag: {f_on.diag.as_dict()}")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    ax = axes[0, 0]
    ax.axvspan(0.0, 15.0, color="0.85", zorder=0,
               label="standstill: offset unobservable\n(and harmless)")
    ax.plot(TS, SPEED, color="black", lw=1.0)
    for role, colour in (("naive", "crimson"), ("online", "seagreen"), ("oracle", "steelblue")):
        rec = results[role][1]
        ax.plot(rec["t"], rec["speed"], colour, lw=0.8, alpha=0.4)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Urban speed profile (targets lagged by 3 s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    t = rec_on["t"]
    band = 2.0 * rec_on["bias_std"]
    ax.axvspan(0.0, 15.0, color="0.85", zorder=0)
    ax.fill_between(t, (rec_on["bias"] - band) * 1e3, (rec_on["bias"] + band) * 1e3,
                    color="seagreen", alpha=0.25, label="online: $\\hat b \\pm 2\\sigma$")
    ax.plot(t, rec_on["bias"] * 1e3, color="seagreen", lw=1.4, label="online: $\\hat b$")
    ax.axhline(B_TRUE * 1e3, color="black", ls="--", lw=1.0, label="truth = 120 ms")
    ax.axhline(0.0, color="crimson", ls=":", lw=1.2, label="naive: $b$ pinned to 0")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("clock offset [ms]")
    ax.set_title("Offset is learned only while the vehicle moves")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for role, colour, lbl in (("naive", "crimson", "naive (trust stamps)"),
                              ("online", "seagreen", "online estimate"),
                              ("oracle", "steelblue", "oracle (knows $b$)")):
        rec = results[role][1]
        ax.plot(rec["t"], rec["err"], colour, lw=1.0,
                label=f"{lbl}: {rec['rmse_moving']:.2f} m")
    ax.axhline(v_max * B_TRUE, color="black", ls=":", lw=1.0,
               label=f"$\\pm v b$ = {v_max * B_TRUE:.1f} m")
    ax.axhline(-v_max * B_TRUE, color="black", ls=":", lw=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Position error at each pipeline's own claimed time")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, rec_on["bias_std"] * 1e3, color="seagreen", lw=1.4,
            label="online: predicted $\\sigma_b$")
    ax.axhline(crlb * 1e3, color="black", ls="--", lw=1.0,
               label=f"CRLB = {crlb * 1e3:.1f} ms")
    ax.axhline(0.5 * 1e3, color="grey", ls=":", lw=1.0, label="prior $\\sigma_b$ = 500 ms")
    ax.set_yscale("log")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("offset uncertainty [ms]")
    ax.set_title("Uncertainty collapses towards the information bound")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Online clock-offset estimation (v0.24.0): the fix stream's "
                 "timestamps are the problem", fontsize=12)
    fig.tight_layout()
    out = "examples/time_offset_demo.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
