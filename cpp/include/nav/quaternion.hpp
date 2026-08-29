/**
 * Quaternion / rotation utilities — port of sensor_sim/utils.py.
 *
 * Conventions: quaternion [w, x, y, z] scalar-first; quatToRotmat maps
 * body -> world (same matrix the Python code calls R_wb / "world <- body").
 */

#pragma once

#include "nav/types.hpp"

namespace nav {

/// Normalize in place; falls back to identity for degenerate input.
inline Eigen::Vector4d quatNormalize(const Eigen::Vector4d& q) {
  double n = q.norm();
  if (n < 1e-12) return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  return q / n;
}

/// Quaternion [w,x,y,z] -> rotation matrix R_bw (body -> world).
inline Mat3 quatToRotmat(const Eigen::Vector4d& q) {
  Eigen::Vector4d u = quatNormalize(q);
  double w = u(0), x = u(1), y = u(2), z = u(3);
  Mat3 R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
       2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return R;
}

/// Rotation matrix -> quaternion [w,x,y,z] (Shepperd's method, matches
/// the branch structure of rotmat_to_quat() in utils.py).
inline Eigen::Vector4d rotmatToQuat(const Mat3& R) {
  Eigen::Vector4d q;
  double tr = R.trace();
  if (tr > 0) {
    double s = 0.5 / std::sqrt(tr + 1.0);
    q << 0.25 / s, (R(2, 1) - R(1, 2)) * s, (R(0, 2) - R(2, 0)) * s,
        (R(1, 0) - R(0, 1)) * s;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    double s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
    q << (R(2, 1) - R(1, 2)) / s, 0.25 * s, (R(0, 1) + R(1, 0)) / s,
        (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    double s = 2.0 * std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
    q << (R(0, 2) - R(2, 0)) / s, (R(0, 1) + R(1, 0)) / s, 0.25 * s,
        (R(1, 2) + R(2, 1)) / s;
  } else {
    double s = 2.0 * std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
    q << (R(1, 0) - R(0, 1)) / s, (R(0, 2) + R(2, 0)) / s,
        (R(1, 2) + R(2, 1)) / s, 0.25 * s;
  }
  return q;
}

/// Hamilton product q1 * q2 (both [w,x,y,z]).
inline Eigen::Vector4d quatMultiply(const Eigen::Vector4d& q1,
                                    const Eigen::Vector4d& q2) {
  double w1 = q1(0), x1 = q1(1), y1 = q1(2), z1 = q1(3);
  double w2 = q2(0), x2 = q2(1), y2 = q2(2), z2 = q2(3);
  Eigen::Vector4d q;
  q << w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
       w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
       w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
  return q;
}

/// Quaternion exponential map so(3) -> SO(3): q_exp(rotvec) with
/// rotvec = axis * angle.  Small-angle guard matches _quat_exp().
inline Eigen::Vector4d quatExp(const Vec3& rotvec) {
  double angle = rotvec.norm();
  if (angle < 1e-12) return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  Vec3 axis = rotvec / angle;
  double half = 0.5 * angle;
  Eigen::Vector4d q;
  q << std::cos(half), axis.x() * std::sin(half), axis.y() * std::sin(half),
      axis.z() * std::sin(half);
  return q;
}

/// Quaternion log map SO(3) -> so(3) rotation vector.
inline Vec3 quatLog(const Eigen::Vector4d& q_in) {
  Eigen::Vector4d q = quatNormalize(q_in);
  Vec3 xyz = q.tail<3>();
  double vnorm = xyz.norm();
  if (vnorm < 1e-12) return Vec3::Zero();
  return 2.0 * std::atan2(vnorm, q(0)) * xyz / vnorm;
}

/// Rotate a vector by a quaternion (v' = R(q) v).
inline Vec3 quatRotate(const Eigen::Vector4d& q, const Vec3& v) {
  return quatToRotmat(q) * v;
}

/// Euler -> quaternion (ZYX intrinsic, rad) — euler_to_quat().
inline Eigen::Vector4d eulerToQuat(double roll, double pitch, double yaw) {
  double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  Eigen::Vector4d q;
  q << cr * cp * cy + sr * sp * sy,
       sr * cp * cy - cr * sp * sy,
       cr * sp * cy + sr * cp * sy,
       cr * cp * sy - sr * sp * cy;
  return q;
}

/// Quaternion -> Euler [roll, pitch, yaw] (ZYX, rad) — quat_to_euler().
inline Vec3 quatToEuler(const Eigen::Vector4d& q_in) {
  Eigen::Vector4d u = quatNormalize(q_in);
  double w = u(0), x = u(1), y = u(2), z = u(3);
  double roll = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  double sinp = 2 * (w * y - z * x);
  double pitch = std::abs(sinp) >= 1.0 ? std::copysign(M_PI / 2, sinp)
                                       : std::asin(sinp);
  double yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return {roll, pitch, yaw};
}

/// Spherical linear interpolation between quaternions — _slerp().
inline Eigen::Vector4d quatSlerp(const Eigen::Vector4d& q0_in,
                                 const Eigen::Vector4d& q1_in, double t) {
  Eigen::Vector4d q0 = quatNormalize(q0_in);
  Eigen::Vector4d q1 = quatNormalize(q1_in);
  double dot = q0.dot(q1);
  dot = std::max(-1.0, std::min(1.0, dot));
  if (std::abs(dot) > 0.9995) {
    Eigen::Vector4d r = q0 + t * (q1 - q0);
    return quatNormalize(r);
  }
  double theta_0 = std::acos(dot);
  double theta = theta_0 * t;
  double sin_theta = std::sin(theta);
  double sin_theta_0 = std::sin(theta_0);
  double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  double s1 = sin_theta / sin_theta_0;
  if (dot < 0) {
    q1 = -q1;
    s1 = -s1;
  }
  return s0 * q0 + s1 * q1;
}

}  // namespace nav
