/**
 * navsim — C++ port of the sensor-sim Python navigation core.
 *
 * Semantic parity target: sensor_sim/{utils,imu,gnss,eskf,trajectory}.py
 * (v0.15.1). Random streams are NOT bit-identical to NumPy; the port is
 * validated statistically (see tests/test_nav.cpp and cpp/README.md).
 *
 * Conventions (identical to the Python reference):
 *   - Quaternions: [w, x, y, z], scalar-first, normalized.
 *   - quatToRotmat(q) maps body -> world (so R @ a_body = a_world).
 *   - World frame: x/y horizontal, z up; gravity g = [0, 0, -9.80665].
 *   - ESKF error state: dx = [dp, dv, dtheta, db_a, db_g] (15 dims).
 */

#pragma once

#include <Eigen/Dense>

namespace nav {

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;
using MatX = Eigen::MatrixXd;

/// Standard gravity used consistently everywhere (Python demo mixed
/// 9.81/9.80665; the C++ port uses 9.80665 in both mechanization and
/// trajectory body-frame specific force).
inline constexpr double kGravity = 9.80665;

/// Skew-symmetric (cross-product) matrix from a 3-vector.
inline Mat3 skew(const Vec3& v) {
  Mat3 m;
  m <<    0.0, -v.z(),  v.y(),
        v.z(),    0.0, -v.x(),
       -v.y(),  v.x(),    0.0;
  return m;
}

}  // namespace nav
