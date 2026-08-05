"""
Trajectory generator: produces smooth 6-DoF ground-truth trajectories
from waypoints with vehicle dynamics constraints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from .utils import quat_to_rotmat


@dataclass
class Waypoint:
    """A 6-DoF waypoint in space."""
    t: float                          # time (s)
    pos: np.ndarray                   # [x,y,z] in world frame (m)
    vel: np.ndarray                   # [vx,vy,vz] in world frame (m/s)
    att: np.ndarray                   # quaternion [w,x,y,z] world<-body
    omega: np.ndarray                 # angular velocity in body frame (rad/s)


@dataclass
class TrajPoint:
    """Ground-truth state at a single timestep."""
    t: float
    pos: np.ndarray       # [3] position world frame (m)
    vel: np.ndarray       # [3] velocity world frame (m/s)
    acc: np.ndarray       # [3] acceleration world frame (m/s^2)
    att: np.ndarray       # [4] quaternion wxyz world<-body
    omega: np.ndarray     # [3] angular velocity body frame (rad/s)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if abs(dot) > 0.9995:
        # Linear interpolation for near-parallel quaternions
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(abs(dot))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    if dot < 0:
        q1 = -q1
        s1 = -s1
    return s0 * q0 + s1 * q1


def _quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Quaternion [w,x,y,z] → (roll, pitch, yaw) in radians (ZYX convention)."""
    w, x, y, z = q
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(roll, pitch, yaw) in radians → quaternion [w,x,y,z] (ZYX convention)."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])



class Trajectory:
    """
    Generate a smooth 6-DoF trajectory from waypoints.

    Between waypoints, position uses cubic spline and attitude uses SLERP.
    Acceleration and angular velocity are computed via finite differences.

    Parameters
    ----------
    waypoints : list of Waypoint
        Key frames defining the trajectory.
    dt : float
        Output timestep (s). Must evenly divide all waypoint intervals.
    """

    def __init__(self, waypoints: List[Waypoint], dt: float = 0.01,
                 const_speed: Optional[float] = None):
        """
        Generate a smooth 6-DoF trajectory from waypoints.

        Between waypoints, position uses cubic spline and attitude uses SLERP.
        Acceleration and angular velocity are computed via finite differences.

        Parameters
        ----------
        waypoints : list of Waypoint
            Key frames defining the trajectory.
        dt : float
            Output timestep (s). Must evenly divide all waypoint intervals.
        const_speed : float or None
            If set, renormalize velocity magnitude to this constant speed
            (m/s) after Hermite interpolation, keeping direction. Fixes the
            classic Hermite problem where velocity magnitude dips between
            waypoints with different headings (e.g. turns).
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        self.waypoints = sorted(waypoints, key=lambda w: w.t)
        self.dt = dt
        self._const_speed = const_speed
        self._points: Optional[List[TrajPoint]] = None
        self._generate()

    def _generate(self):
        t_start = self.waypoints[0].t
        t_end = self.waypoints[-1].t
        n = int(round((t_end - t_start) / self.dt)) + 1
        ts = np.linspace(t_start, t_end, n)

        # Fit cubic splines per segment for position
        wps = self.waypoints
        seg_idx = np.searchsorted([w.t for w in wps[1:]], ts, side='right')
        seg_idx = np.clip(seg_idx, 0, len(wps) - 2)

        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        att = np.zeros((n, 4))
        omega = np.zeros((n, 3))
        acc = np.zeros((n, 3))

        for i in range(len(wps) - 1):
            mask = seg_idx == i
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            t0, t1 = wps[i].t, wps[i+1].t
            dt_seg = t1 - t0

            # Cubic Hermite spline for position
            p0, v0 = wps[i].pos.copy(), wps[i].vel.copy()
            p1, v1 = wps[i+1].pos.copy(), wps[i+1].vel.copy()

            for j in idx:
                tau = (ts[j] - t0) / dt_seg
                tau = np.clip(tau, 0.0, 1.0)
                h00 = 2*tau**3 - 3*tau**2 + 1
                h10 = tau**3 - 2*tau**2 + tau
                h01 = -2*tau**3 + 3*tau**2
                h11 = tau**3 - tau**2
                pos[j] = h00*p0 + h10*dt_seg*v0 + h01*p1 + h11*dt_seg*v1
                # Velocity (derivative of Hermite)
                dh00 = (6*tau**2 - 6*tau) / dt_seg
                dh10 = (3*tau**2 - 4*tau + 1)
                dh01 = (-6*tau**2 + 6*tau) / dt_seg
                dh11 = (3*tau**2 - 2*tau)
                vel[j] = dh00*p0 + dh10*v0 + dh01*p1 + dh11*v1

            # SLERP for attitude
            q0, q1 = wps[i].att.copy(), wps[i+1].att.copy()
            for j in idx:
                tau = (ts[j] - t0) / dt_seg
                tau = np.clip(tau, 0.0, 1.0)
                att[j] = _slerp(q0, q1, tau)

        # Angular velocity via finite difference of attitude
        for i in range(1, n-1):
            q_prev = att[i-1]
            q_next = att[i+1]
            delta_t = ts[i+1] - ts[i-1]
            # q_dot ≈ (q_next - q_prev) / (2*dt), then convert to omega
            q_curr = att[i]
            # Compute angular velocity from quaternion derivative
            # omega_body = 2 * conj(q) * dq/dt
            q_dot = (q_next - q_prev) / max(delta_t, 1e-9)
            w = q_curr[0]
            x, y, z = q_curr[1], q_curr[2], q_curr[3]
            Q_conj = np.array([
                [w, x, y, z],
                [-x, w, z, -y],
                [-y, -z, w, x],
                [-z, y, -x, w]
            ])
            omega[i] = 2.0 * (Q_conj @ q_dot)[1:4]  # discard scalar part

        # Acceleration via finite difference of velocity
        for i in range(1, n-1):
            acc[i] = (vel[i+1] - vel[i-1]) / max(ts[i+1] - ts[i-1], 1e-9)

        # Optional: renormalize speed magnitude to a constant (vehicle-like)
        if self._const_speed is not None:
            for i in range(n):
                v_norm = np.linalg.norm(vel[i])
                if v_norm > 1e-9:
                    vel[i] = vel[i] / v_norm * self._const_speed
                    # Nonholonomic constraint: yaw must follow velocity heading.
                    # Rebuild attitude from velocity direction, keeping the
                    # original roll/pitch (extracted from the SLERP quat).
                    yaw = np.arctan2(vel[i][1], vel[i][0])
                    q_orig = att[i] / np.linalg.norm(att[i])
                    roll, pitch, _ = _quat_to_euler(q_orig)
                    att[i] = _euler_to_quat(roll, pitch, yaw)
            # Recompute acceleration from the renormalized velocity
            acc[:] = 0.0
            for i in range(1, n-1):
                acc[i] = (vel[i+1] - vel[i-1]) / max(ts[i+1] - ts[i-1], 1e-9)

        # Endpoints: copy neighbors
        omega[0], omega[-1] = omega[1], omega[-2]
        acc[0], acc[-1] = acc[1], acc[-2]

        self._points = [
            TrajPoint(t=ts[i], pos=pos[i], vel=vel[i], acc=acc[i],
                      att=att[i], omega=omega[i])
            for i in range(n)
        ]

    @property
    def points(self) -> List[TrajPoint]:
        return self._points

    @property
    def ts(self) -> np.ndarray:
        return np.array([p.t for p in self._points])

    @property
    def positions(self) -> np.ndarray:
        return np.array([p.pos for p in self._points])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([p.vel for p in self._points])

    @property
    def attitudes(self) -> np.ndarray:
        return np.array([p.att for p in self._points])

    def body_accel(self, point: TrajPoint) -> np.ndarray:
        """Acceleration in body frame (sensed by accelerometer)."""
        R_bw = quat_to_rotmat(point.att).T  # body <- world
        gravity_world = np.array([0, 0, 9.81])  # gravity in world frame
        return R_bw @ (point.acc + gravity_world)

    def body_accels(self) -> np.ndarray:
        """All body-frame accelerations."""
        return np.array([self.body_accel(p) for p in self._points])

    def body_omegas(self) -> np.ndarray:
        """All body-frame angular velocities."""
        return np.array([p.omega for p in self._points])

    def __len__(self):
        return len(self._points)

    def __repr__(self):
        t0 = self.waypoints[0].t
        t1 = self.waypoints[-1].t
        return f"Trajectory({t0:.1f}s → {t1:.1f}s, {len(self)} pts @ {self.dt}s)"
