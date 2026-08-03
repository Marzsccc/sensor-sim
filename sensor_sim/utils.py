"""
Coordinate transformation utilities for sensor simulation.

All quaternions use [w, x, y, z] convention (scalar-first).
All rotation matrices map world → body.
"""

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] to rotation matrix R_wb (world → body)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),     1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion [w,x,y,z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Euler angles (rad) to quaternion [w,x,y,z].
    Convention: ZYX intrinsic (yaw-pitch-roll).
    """
    cr, sr = np.cos(roll*0.5), np.sin(roll*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cy, sy = np.cos(yaw*0.5), np.sin(yaw*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] to Euler angles [roll, pitch, yaw] (rad)."""
    w, x, y, z = q / np.linalg.norm(q)
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2*(w*y - z*x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.array([0, v[0], v[1], v[2]])
    q_inv = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, qv), q_inv)
    return result[1:4]


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0],
    ])
