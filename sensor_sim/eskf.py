"""
Error-State Kalman Filter (ESKF) for GNSS + IMU loose coupling.

Implements the classic error-state formulation (Solà 2017, "Quaternion
kinematics for the error-state Kalman filter"; Groves 2013 Ch.14) for a
strapdown inertial navigation system aided by position/velocity updates:

  Nominal state  x = [p, v, q]         (15 states total with biases)
  Error state    dx = [dp, dv, dtheta, db_a, db_g]

Mechanization (nominal-state integration):
    p <- p + v*dt + 0.5*(R(a_m - b_a) + g)*dt^2
    v <- v + (R(a_m - b_a) + g)*dt
    q <- q (x) q{ (w_m - b_g) * dt }

Error-state propagation:
    dtheta <- R^T * dv * dt            (velocity perturbation -> attitude)
    dv     <- ( -R[a_m - b_a]x * dtheta - R db_a ) * dt
    dp     <- dv * dt
    db     <- db   (random walk)

Update: GNSS position + velocity measurements (loose coupling), 6-dim
observation.  Joseph-form covariance update for numerical robustness.

The filter is entirely self-contained (only numpy) so it can be used both
as a reference implementation and as a drop-in estimator for the synthetic
measurements produced by the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utils import quat_multiply, quat_rotate, skew, quat_to_rotmat


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion [w,x,y,z] from rotation vector (axis * angle)."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def _quat_exp(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3) quaternion (wxyz)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec / angle
    return _quat_from_axis_angle(axis, angle)


def _quat_log(q: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) quaternion -> so(3) rotation vector."""
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    v_norm = np.linalg.norm(xyz)
    if v_norm < 1e-12:
        return np.zeros(3)
    return 2.0 * np.arctan2(v_norm, w) * xyz / v_norm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ESKFConfig:
    """ESKF tuning parameters.

    Process noise densities are per sqrt(Hz) and scaled by the time step
    inside the filter, i.e.  Q ~ diag(...) * dt  in the continuous-discrete
    formulation.
    """

    # Accelerometer noise density (m/s^2 / sqrt(Hz))
    acc_noise_density: float = 1.0e-2
    # Gyroscope noise density (rad/s / sqrt(Hz))
    gyr_noise_density: float = 1.0e-3
    # Accel bias random-walk density (m/s^3 / sqrt(Hz))
    acc_bias_rw: float = 1.0e-4
    # Gyro bias random-walk density (rad/s^2 / sqrt(Hz))
    gyr_bias_rw: float = 1.0e-5

    # Initial error covariance (diagonal)
    init_pos_std: float = 10.0        # m
    init_vel_std: float = 5.0         # m/s
    init_att_std_deg: float = 10.0    # deg
    init_accbias_std: float = 1.0e-1  # m/s^2
    init_gyrbias_std: float = 1.0e-2  # rad/s

    # GNSS measurement noise (1 sigma)
    gnss_pos_std: float = 1.5         # m
    gnss_vel_std: float = 0.2         # m/s


@dataclass
class ESKFState:
    """Full ESKF state (nominal + error covariance)."""

    t: float = 0.0
    # Nominal state
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Error covariance (15x15)
    P: np.ndarray = field(default_factory=lambda: np.eye(15) * 1.0)

    # bookkeeping
    n_updates: int = 0

    def error_state_vector(self) -> np.ndarray:
        """Return the current error-state vector dx (15,)."""
        return np.zeros(15)


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------

class ESKF:
    """Error-state Kalman filter for GNSS+IMU loose coupling."""

    # Index layout of the 15-dim error state
    IDX_P = slice(0, 3)
    IDX_V = slice(3, 6)
    IDX_THETA = slice(6, 9)
    IDX_BA = slice(9, 12)
    IDX_BG = slice(12, 15)

    def __init__(self, cfg: Optional[ESKFConfig] = None):
        self.cfg = cfg or ESKFConfig()
        self.state = ESKFState()
        self._init_covariance()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_covariance(self):
        c = self.cfg
        std = np.array([
            c.init_pos_std, c.init_pos_std, c.init_pos_std,
            c.init_vel_std, c.init_vel_std, c.init_vel_std,
            np.deg2rad(c.init_att_std_deg), np.deg2rad(c.init_att_std_deg),
            np.deg2rad(c.init_att_std_deg),
            c.init_accbias_std, c.init_accbias_std, c.init_accbias_std,
            c.init_gyrbias_std, c.init_gyrbias_std, c.init_gyrbias_std,
        ], dtype=float)
        self.state.P = np.diag(std ** 2)

    def set_initial_state(self, t: float, p: np.ndarray, v: np.ndarray,
                          q: np.ndarray, b_a: Optional[np.ndarray] = None,
                          b_g: Optional[np.ndarray] = None):
        """Seed the filter with an initial (possibly approximate) pose."""
        st = self.state
        st.t = t
        st.p = np.asarray(p, dtype=float).copy()
        st.v = np.asarray(v, dtype=float).copy()
        st.q = np.asarray(q, dtype=float).copy()
        st.q /= np.linalg.norm(st.q)
        st.b_a = np.zeros(3) if b_a is None else np.asarray(b_a, dtype=float).copy()
        st.b_g = np.zeros(3) if b_g is None else np.asarray(b_g, dtype=float).copy()

    # ------------------------------------------------------------------
    # Prediction (IMU mechanization)
    # ------------------------------------------------------------------

    def predict(self, acc: np.ndarray, gyro: np.ndarray, dt: float):
        """Propagate nominal state with one IMU sample.

        acc, gyro: specific force and angular rate in the body frame
                   (i.e. what the IMU reports, before bias removal).
        """
        if dt <= 0:
            return
        st = self.state
        c = self.cfg

        a = np.asarray(acc, dtype=float) - st.b_a
        w = np.asarray(gyro, dtype=float) - st.b_g

        R = quat_to_rotmat(st.q)          # world <- body
        g = np.array([0.0, 0.0, -9.80665])

        # --- nominal state integration -------------------------------
        a_w = R @ a + g
        st.p = st.p + st.v * dt + 0.5 * a_w * dt * dt
        st.v = st.v + a_w * dt
        st.q = quat_multiply(st.q, _quat_exp(w * dt))
        st.q /= np.linalg.norm(st.q)

        # --- error-state covariance propagation ----------------------
        # Continuous-time error dynamics (Solà 2017 eq. 178-180):
        #   d/dt dp      = dv
        #   d/dt dv      = -R[a_m - b_a]x * dtheta - R * db_a
        #   d/dt dtheta  = -[w_m - b_g]x * dtheta - db_g
        #   d/dt db_a    = 0,  d/dt db_g = 0
        # F_d = I + F_c * dt (first-order discretization)
        F = np.eye(15)
        F[self.IDX_P, self.IDX_V] = np.eye(3) * dt
        F[self.IDX_V, self.IDX_THETA] = (-R @ skew(a)) * dt
        F[self.IDX_V, self.IDX_BA] = -R * dt
        F[self.IDX_THETA, self.IDX_THETA] = np.eye(3) - skew(w) * dt
        F[self.IDX_THETA, self.IDX_BG] = -np.eye(3) * dt
        # biases: identity (random walk)

        # Process noise (continuous-discrete): G = [0,0,0, I,0; 0,0,0,0,I]^T
        # -> Q_d = diag(0,0,0, sigma_a^2 dt, sigma_g^2 dt,
        #               sigma_ba^2 dt, sigma_bg^2 dt)
        n_a = c.acc_noise_density
        n_g = c.gyr_noise_density
        n_ba = c.acc_bias_rw
        n_bg = c.gyr_bias_rw
        qd = np.zeros(15)
        qd[3:6] = (n_a * n_a) * dt
        qd[6:9] = (n_g * n_g) * dt
        qd[9:12] = (n_ba * n_ba) * dt
        qd[12:15] = (n_bg * n_bg) * dt
        Qd = np.diag(qd)

        st.P = F @ st.P @ F.T + Qd
        st.t += dt

    # ------------------------------------------------------------------
    # Update (GNSS position + velocity)
    # ------------------------------------------------------------------

    def update_gnss(self, pos: np.ndarray, vel: Optional[np.ndarray] = None,
                    pos_std: Optional[float] = None,
                    vel_std: Optional[float] = None):
        """Loose-coupling update with GNSS position (and optionally velocity).

        Measurement model: z = [p; v] = H x_nom + noise, H = [I3 0 0 0 0;
        0 I3 0 0 0].  The error-state innovation is  dz = z - h(x_nom).
        """
        st = self.state
        c = self.cfg

        p_std = c.gnss_pos_std if pos_std is None else pos_std
        v_std = c.gnss_vel_std if vel_std is None else vel_std

        # Build observation
        if vel is None:
            z = np.concatenate([np.asarray(pos, dtype=float), st.v.copy()])
            H = np.zeros((6, 15))
            H[0:3, self.IDX_P] = np.eye(3)
            # velocity not observed -> innovation for velocity rows is 0,
            # handled by giving them infinite noise (large R).
            R = np.diag([p_std ** 2] * 3 + [1e12] * 3)
        else:
            z = np.concatenate([np.asarray(pos, dtype=float),
                                np.asarray(vel, dtype=float)])
            H = np.zeros((6, 15))
            H[0:3, self.IDX_P] = np.eye(3)
            H[3:6, self.IDX_V] = np.eye(3)
            R = np.diag([p_std ** 2] * 3 + [v_std ** 2] * 3)

        # Innovation
        h = np.concatenate([st.p, st.v])
        dz = z - h
        # Gate: reject outliers beyond 6 sigma (GNSS dropouts / jumps)
        S = H @ st.P @ H.T + R
        innov_cov = np.sqrt(np.diag(S))
        if np.any(np.abs(dz) > 6.0 * np.clip(innov_cov, 1e-9, None)):
            return

        # Kalman gain
        K = st.P @ H.T @ np.linalg.inv(S)
        dx = K @ dz

        # Inject error state into nominal state
        self._inject(dx)

        # Joseph-form covariance update (robust to numerical issues)
        I = np.eye(15)
        IKH = I - K @ H
        st.P = IKH @ st.P @ IKH.T + K @ R @ K.T
        st.P = 0.5 * (st.P + st.P.T)      # symmetrize
        st.n_updates += 1

    def _inject(self, dx: np.ndarray):
        """Add the error state to the nominal state (ESKF reset step)."""
        st = self.state
        st.p += dx[self.IDX_P]
        st.v += dx[self.IDX_V]
        dtheta = dx[self.IDX_THETA]
        st.q = quat_multiply(st.q, _quat_exp(dtheta))
        st.q /= np.linalg.norm(st.q)
        st.b_a += dx[self.IDX_BA]
        st.b_g += dx[self.IDX_BG]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        return self.state.p

    @property
    def velocity(self) -> np.ndarray:
        return self.state.v

    @property
    def quaternion(self) -> np.ndarray:
        return self.state.q

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_rotmat(self.state.q)

    def attitude_euler(self) -> np.ndarray:
        """Return [roll, pitch, yaw] in radians."""
        from .utils import quat_to_euler
        return quat_to_euler(self.state.q)

    def estimated_biases(self):
        return self.state.b_a.copy(), self.state.b_g.copy()
