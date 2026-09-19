"""Rewind-and-replay fusion for a GNSS+IMU ESKF demo (v0.23.0).

Shows the GNSS-aided flavour of the AR-HUD "marker trails the world" bug: a
GNSS fix that is fused when it *arrives* instead of when it was *valid* bakes
the receiver latency into the state estimate as a bias, and the bias is
``speed x delay`` -- metres, not centimetres.

Scene
-----
A vehicle drives at 25 m/s (90 km/h) with a gentle lateral sinusoid
(1.5 m/s^2 amplitude, 0.5 rad/s).  A 100 Hz IMU feeds the mechanisation; GNSS
position fixes arrive at 10 Hz with 200 ms of receiver/pipeline latency.

Two pipelines are compared
--------------------------
1. ``receive_time`` -- fuse each fix when it arrives (naive).
2. ``oosm``         -- insert each fix at its validity time (rewind + replay).

Metrics
-------
* Streaming position RMSE (and NEES) against truth at the filter's current
  time -- what a marker anchored to the state would show.
* Display-time RMSE at ``t_recv + 60 ms``, identical for both pipelines.  Each
  pipeline renders with its own constant-velocity extrapolation, so the naive
  one gets no free shorter horizon.
* The naive pipeline's error should sit near ``v * delay`` = 5 m at 25 m/s;
  the OOSM one should stay near the GNSS noise floor (1 m).

Saves ``examples/nonlinear_delay_demo.png``.

Run:
    PYTHONPATH=. uv run --with numpy --with matplotlib python examples/nonlinear_delay_demo.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sensor_sim.eskf import ESKFConfig
from sensor_sim.nonlinear_delay import RewindReplayESKF, position_nees

SEED = 11
IMU_DT = 0.01
N_IMU = 1500
GNSS_DT = 0.1
DELAY = 0.2
DISPLAY_AHEAD = 0.06
V0 = 25.0
A_LAT, W_LAT = 1.5, 0.5
Q0 = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
class Truth:
    """Analytic-ish straight track with a lateral sinusoid (integrated finely)."""

    def __init__(self, t_end: float, step: float = 1e-4) -> None:
        self.step = step
        self.ts = np.arange(0.0, t_end + step, step)
        n = len(self.ts)
        self.v = np.zeros((n, 3))
        self.p = np.zeros((n, 3))
        self.v[0] = [V0, 0.0, 0.0]
        for k in range(1, n):
            h = self.ts[k] - self.ts[k - 1]
            a = self.accel(self.ts[k - 1])
            self.v[k] = self.v[k - 1] + a * h
            self.p[k] = self.p[k - 1] + self.v[k - 1] * h + 0.5 * a * h * h

    @staticmethod
    def accel(t: float) -> np.ndarray:
        return np.array([0.0, A_LAT * np.sin(W_LAT * t), 0.0])

    def at(self, t: float):
        k = int(round(t / self.step))
        k = min(max(k, 0), len(self.ts) - 1)
        return self.p[k].copy(), self.v[k].copy()


def make_imu_gnss(truth: Truth, rng):
    g = np.array([0.0, 0.0, -9.80665])
    imu = []
    for k in range(N_IMU):
        t = k * IMU_DT
        acc = truth.accel(t) - g + rng.normal(0.0, 0.02, 3)
        gyro = rng.normal(0.0, 1e-3, 3)
        imu.append((t, acc, gyro))

    gnss = []
    t = 1.0
    while t < N_IMU * IMU_DT - 0.3:
        p_true, _ = truth.at(t)
        gnss.append((t, p_true + rng.normal(0.0, 1.0, 3), t + DELAY))
        t += GNSS_DT
    return imu, gnss


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(imu, gnss, truth, mode, cfg):
    """Feed samples in receive order; score at both stream and display time."""
    f = RewindReplayESKF(cfg, mode=mode, history_s=10.0)
    f.initialize(0.0, np.zeros(3), np.array([V0, 0.0, 0.0]), Q0.copy())

    i = j = 0
    rec = dict(t=[], est=[], est_true=[], disp=[], disp_true=[], nees=[])

    while i < len(imu) or j < len(gnss):
        t_imu = imu[i][0] if i < len(imu) else np.inf
        t_recv = gnss[j][2] if j < len(gnss) else np.inf
        if t_imu <= t_recv:
            t, acc, gyro = imu[i]
            f.add_imu(t, acc, gyro)
            i += 1
        else:
            t_meas, pos, t_r = gnss[j]
            f.add_gnss(t_meas, pos, t_recv=t_r, pos_std=1.0)
            j += 1

        t_disp = t_recv + DISPLAY_AHEAD
        if t_recv < np.inf and t_disp >= f.time and f.time >= 2.0:
            p_disp, _ = f.extrapolate_to(t_disp, accel_psd=1.0)
            p_true_now, _ = truth.at(f.time)
            p_true_disp, _ = truth.at(t_disp)
            rec["t"].append(f.time)
            rec["est"].append(f.position.copy())
            rec["est_true"].append(p_true_now)
            rec["disp"].append(p_disp)
            rec["disp_true"].append(p_true_disp)
            rec["nees"].append(position_nees(f.position - p_true_now, f.position_covariance))

    rec = {k: np.asarray(v) for k, v in rec.items()}
    rec["rmse"] = float(np.sqrt(np.mean(np.sum((rec["est"] - rec["est_true"]) ** 2, axis=1))))
    rec["disp_rmse"] = float(np.sqrt(np.mean(np.sum((rec["disp"] - rec["disp_true"]) ** 2, axis=1))))
    rec["mean_nees"] = float(np.mean(rec["nees"]))
    rec["p95_err"] = float(np.percentile(np.linalg.norm(rec["est"] - rec["est_true"], axis=1), 95))
    return f, rec


# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED)
    truth = Truth(N_IMU * IMU_DT)
    imu, gnss = make_imu_gnss(truth, rng)
    cfg = ESKFConfig()

    results = {}
    for mode in ("oosm", "receive_time"):
        f, rec = run_pipeline(imu, gnss, truth, mode, cfg)
        results[mode] = (f, rec)
        print(f"[{mode:12s}] stream RMSE {rec['rmse']:6.3f} m | "
              f"display RMSE {rec['disp_rmse']:6.3f} m | "
              f"p95 {rec['p95_err']:6.3f} m | NEES {rec['mean_nees']:6.2f}")
        print(f"              diag: {f.diag.as_dict()}")

    f_n, rec_n = results["receive_time"]
    f_o, rec_o = results["oosm"]

    speed = float(np.linalg.norm(truth.v[int(2.0 / truth.step)]))
    print(f"\nspeed x delay = {speed:.1f} m/s x {DELAY*1000:.0f} ms "
          f"= {speed*DELAY:.2f} m")
    print(f"naive / oosm stream RMSE ratio = {rec_n['rmse']/rec_o['rmse']:.2f}x")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(truth.p[:, 0], truth.p[:, 1], "k--", lw=1.0, label="truth")
    ax.plot(rec_n["est"][:, 0], rec_n["est"][:, 1], color="crimson", lw=1.2,
            label=f"receive-time (RMSE {rec_n['rmse']:.2f} m)")
    ax.plot(rec_o["est"][:, 0], rec_o["est"][:, 1], color="seagreen", lw=1.2,
            label=f"rewind-replay (RMSE {rec_o['rmse']:.2f} m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory: the naive pipeline trails the truth")
    ax.legend(loc="best", fontsize=8)
    ax.axis("equal")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rec_n["t"], np.linalg.norm(rec_n["est"] - rec_n["est_true"], axis=1),
            color="crimson", lw=1.0, label="receive-time")
    ax.plot(rec_o["t"], np.linalg.norm(rec_o["est"] - rec_o["est_true"], axis=1),
            color="seagreen", lw=1.0, label="rewind-replay")
    ax.axhline(speed * DELAY, color="k", ls=":", lw=1.0,
               label=f"v x delay = {speed*DELAY:.1f} m")
    ax.set_xlabel("filter time [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Streaming position error (each vs truth at its own clock)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(rec_n["t"], rec_n["nees"], color="crimson", lw=1.0, label="receive-time")
    ax.plot(rec_o["t"], rec_o["nees"], color="seagreen", lw=1.0, label="rewind-replay")
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="expected (3 dof)")
    ax.set_yscale("log")
    ax.set_xlabel("filter time [s]")
    ax.set_ylabel("position NEES")
    ax.set_title("Consistency: the lagged state is over-confident")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(rec_n["t"], np.linalg.norm(rec_n["disp"] - rec_n["disp_true"], axis=1),
            color="crimson", lw=1.0, label=f"receive-time ({rec_n['disp_rmse']:.2f} m)")
    ax.plot(rec_o["t"], np.linalg.norm(rec_o["disp"] - rec_o["disp_true"], axis=1),
            color="seagreen", lw=1.0, label=f"rewind-replay ({rec_o['disp_rmse']:.2f} m)")
    ax.set_xlabel("filter time [s]")
    ax.set_ylabel("error at t_recv + 60 ms [m]")
    ax.set_title("Display-time error: same render instant for both")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("v0.23.0  Nonlinear delayed-measurement fusion (ESKF rewind + replay)",
                 fontsize=13)
    fig.tight_layout()
    out = "examples/nonlinear_delay_demo.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
