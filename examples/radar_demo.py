"""Automotive radar detection simulation demo (v0.17.0).

A forward-looking 77 GHz radar on an ego vehicle driving straight at 25 m/s.
Two targets:

1. a lead vehicle 90 m ahead driving 15 m/s -- a classic ACC / FCW closing
   target: range shrinks at the 10 m/s closing speed, Doppler reads +10 m/s;
2. a pedestrian crossing the road from the left (constant lateral velocity,
   fixed down-range position) -- the radar's classic blind spot: while the
   pedestrian is clearly visible in range/azimuth, the *radial* Doppler stays
   near zero until the target is almost abeam, because the motion is mostly
   cross-range.  FCW systems therefore fuse radar with camera, not radar
   alone.

Every scan prints a detection table; the figure (saved to
``examples/radar_demo.png``) overlays measured vs truth range / range-rate /
azimuth so dropouts (far / dim targets) and the Doppler blind spot are
visible.

Run:
    python3 examples/radar_demo.py
"""

import matplotlib
import numpy as np

from sensor_sim.lidar import Box, Sphere
from sensor_sim.radar import RadarConfig, RadarSensor
from sensor_sim.trajectory import _euler_to_quat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

V_EGO = 25.0  # m/s
DT = 0.1  # s
T_END = 3.5  # s


def main():
    # -- world -------------------------------------------------------------
    lead = Box(
        object_id=1,
        center=np.array([90.0, 0.0, 0.5]),
        half_extents=np.array([2.2, 1.0, 0.7]),
        velocity=np.array([15.0, 0.0, 0.0]),  # same direction, slower
        reflectivity=1.0,
    )  # -> 10 m^2 car
    ped = Sphere(
        object_id=2,
        center=np.array([45.0, -7.0, 0.9]),  # crossing from the left
        radius=0.35,
        velocity=np.array([0.0, 2.5, 0.0]),  # 2.5 m/s lateral walk
        reflectivity=0.1,
    )  # -> ~1 m^2 person
    world = [lead, ped]

    # Radar on the front bumper: 1 m ahead of the body origin, 0.6 m up.
    rs = RadarSensor(RadarConfig(), mount_t_body=np.array([1.0, 0.0, 0.6]))
    q_id = _euler_to_quat(0.0, 0.0, 0.0)

    print(
        f"{'t':>4} {'obj':>4} {'r_m':>7} {'r_tru':>7} {'vr_m':>7} "
        f"{'vr_tru':>7} {'az_deg':>7} {'snr_dB':>7}"
    )

    rows = []
    n = round(T_END / DT)
    for i in range(n + 1):
        t = i * DT
        pos = np.array([V_EGO * t, 0.0, 0.0])
        vel = np.array([V_EGO, 0.0, 0.0])
        fr = rs.scan(
            _P(pos, q_id, vel), world, t=t, rng=np.random.default_rng(20260904 + i)
        )
        for k in range(len(fr.ranges)):
            oid = fr.object_ids[k]
            name = "lead" if oid == 1 else "ped"
            az_deg = np.rad2deg(fr.azimuths[k])
            print(
                f"{t:5.1f} {name:>4} {fr.ranges[k]:7.2f} "
                f"{fr.range_true[k]:7.2f} {fr.range_rates[k]:7.2f} "
                f"{fr.range_rate_true[k]:7.2f} {az_deg:7.2f} "
                f"{fr.snr_db[k]:7.1f}"
            )
            rows.append(
                (
                    t,
                    oid,
                    fr.ranges[k],
                    fr.range_true[k],
                    fr.range_rates[k],
                    fr.range_rate_true[k],
                    az_deg,
                )
            )
    rows = np.asarray(rows)

    # Truth curves for the overlay (computed analytically, no noise).
    ts = np.arange(0.0, T_END + DT, DT)
    lead_r = 90.0 + 15.0 * ts - V_EGO * ts - 2.2 - 1.0  # surface range
    ped_x = 45.0 - V_EGO * ts - 1.0  # down-range gap (radar at +1 m)
    ped_y = -7.0 + 2.5 * ts  # lateral position
    ped_r = np.hypot(ped_x, ped_y)
    ped_az = np.rad2deg(np.arctan2(ped_y, ped_x))
    # Radial Doppler of the pedestrian: ego closing on its down-range
    # position (25 m/s term) plus the tiny lateral-walk contribution
    # (-2.5 * sin(az)): the crossing motion is nearly Doppler-invisible.
    ped_vr = 25.0 * (ped_x / ped_r) - 2.5 * (ped_y / ped_r)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True)
    lw = 1.0

    def _split(oid):
        m = rows[:, 1] == oid
        return rows[m]

    # 1) range ------------------------------------------------------------
    ax = axes[0]
    ax.plot(ts, lead_r, color="#e11d48", lw=lw, ls="--", label="lead truth")
    ax.plot(ts, ped_r, color="#0ea5e9", lw=lw, ls="--", label="ped truth")
    for oid, c in ((1, "#e11d48"), (2, "#0ea5e9")):
        d = _split(oid)
        ax.plot(d[:, 0], d[:, 2], ".", color=c, ms=5)
        ax.plot(d[:, 0], d[:, 3], "+", color=c, ms=6, alpha=0.55)
    ax.set_ylabel("range (m)")
    ax.set_title(
        "Automotive radar: closing lead vehicle + crossing pedestrian "
        "(v0.17.0)\nmarkers = measured, + = truth, dashed = analytic"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 2) range-rate -------------------------------------------------------
    ax = axes[1]
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.plot(
        ts,
        np.full_like(ts, 10.0),
        color="#e11d48",
        lw=lw,
        ls="--",
        label="lead closing +10 m/s",
    )
    ax.plot(
        ts,
        ped_vr,
        color="#0ea5e9",
        lw=lw,
        ls="--",
        label="ped radial ~ ego closing (crossing ~invisible)",
    )
    for oid, c in ((1, "#e11d48"), (2, "#0ea5e9")):
        d = _split(oid)
        ax.plot(d[:, 0], d[:, 4], ".", color=c, ms=5)
    ax.set_ylabel("range-rate (m/s, + closing)")
    ax.set_ylim(-2, 30)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # 3) azimuth ----------------------------------------------------------
    ax = axes[2]
    ax.plot(
        ts,
        np.rad2deg(np.arctan2(0.0, 90.0 + 15.0 * ts - V_EGO * ts)),
        color="#e11d48",
        lw=lw,
        ls="--",
        label="lead (on boresight)",
    )
    ax.plot(ts, ped_az, color="#0ea5e9", lw=lw, ls="--", label="ped sweep")
    for oid, c in ((1, "#e11d48"), (2, "#0ea5e9")):
        d = _split(oid)
        ax.plot(d[:, 0], d[:, 6], ".", color=c, ms=5)
    ax.axhline(-60.0, color="0.5", lw=0.8, ls=":")
    ax.axhline(60.0, color="0.5", lw=0.8, ls=":")
    ax.text(0.05, -57, "FOV edge \u00b160\u00b0", fontsize=7, color="0.4")
    ax.set_ylabel("azimuth (deg)")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("examples/radar_demo.png", dpi=130)
    n_det = len(rows)
    print(
        f"\nSaved radar demo -> examples/radar_demo.png "
        f"({n_det} detections over {n + 1} scans)"
    )

    # -- the take-away ------------------------------------------------------
    ped_last = rows[rows[:, 1] == 2]
    if len(ped_last):
        az_span = ped_last[:, 6].max() - ped_last[:, 6].min()
        vr_span = np.abs(ped_last[:, 4].max() - ped_last[:, 4].min())
        print(
            f"Crossing-target note: over its visible window the pedestrian "
            f"sweeps {az_span:.1f} deg of azimuth while its measured "
            f"range-rate changes by only {vr_span:.2f} m/s -- the 2.5 m/s "
            f"lateral walk is nearly Doppler-invisible (radar measures the "
            f"~25 m/s ego closing). Camera fusion is what sees the cross."
        )


class _P:
    """Minimal pose stand-in (keeps this demo independent of the trajectory)."""

    def __init__(self, pos, att, vel):
        self.pos = np.asarray(pos, dtype=float)
        self.att = np.asarray(att, dtype=float)
        self.vel = np.asarray(vel, dtype=float)


if __name__ == "__main__":
    main()
