/**
 * Trajectory generator — port of sensor_sim/trajectory.py.
 * Cubic Hermite position spline + quaternion SLERP attitude, angular
 * rate and acceleration via central finite differences, optional
 * constant-speed renormalization with nonholonomic yaw alignment.
 */

#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include "nav/quaternion.hpp"
#include "nav/types.hpp"

namespace nav {

struct Waypoint {
  double t = 0.0;
  Vec3 pos = Vec3::Zero();
  Vec3 vel = Vec3::Zero();
  Eigen::Vector4d att = Eigen::Vector4d(1, 0, 0, 0);  // world <- body
  Vec3 omega = Vec3::Zero();                          // body frame, rad/s
};

struct TrajPoint {
  double t = 0.0;
  Vec3 pos = Vec3::Zero();
  Vec3 vel = Vec3::Zero();
  Vec3 acc = Vec3::Zero();
  Eigen::Vector4d att = Eigen::Vector4d(1, 0, 0, 0);
  Vec3 omega = Vec3::Zero();
};

class Trajectory {
 public:
  Trajectory(const std::vector<Waypoint>& waypoints, double dt = 0.01,
             std::optional<double> const_speed = std::nullopt)
      : dt_(dt), const_speed_(const_speed) {
    if (waypoints.size() < 2) {
      throw std::invalid_argument("Trajectory needs at least 2 waypoints");
    }
    wps_ = waypoints;
    std::sort(wps_.begin(), wps_.end(),
              [](const Waypoint& a, const Waypoint& b) { return a.t < b.t; });
    generate();
  }

  /// Acceleration in body frame (specific force sensed by accelerometer).
  /// Python: R_bw @ (acc + [0,0,+9.81]) with R_bw = R_wb^T.
  Vec3 bodyAccel(const TrajPoint& pt) const {
    Mat3 R_bw = quatToRotmat(pt.att).transpose();
    Vec3 gravity_world(0.0, 0.0, 9.81);
    return R_bw * (pt.acc + gravity_world);
  }

  const std::vector<TrajPoint>& points() const { return points_; }
  const std::vector<double>& ts() const { return ts_; }

  /// Ground-truth state nearest to host time t (searchsorted + clip).
  TrajPoint at(double t) const {
    auto it = std::lower_bound(ts_.begin(), ts_.end(), t);
    size_t i;
    if (it == ts_.begin()) {
      i = 0;
    } else if (it == ts_.end()) {
      i = ts_.size() - 1;
    } else {
      i = static_cast<size_t>(it - ts_.begin());
    }
    return points_[i];
  }

 private:
  void generate() {
    double t_start = wps_.front().t;
    double t_end = wps_.back().t;
    size_t n = static_cast<size_t>(std::lround((t_end - t_start) / dt_)) + 1;
    ts_.resize(n);
    for (size_t i = 0; i < n; ++i) ts_[i] = t_start + i * dt_;

    std::vector<TrajPoint> pts(n);
    for (size_t i = 0; i < n; ++i) pts[i].t = ts_[i];

    for (size_t seg = 0; seg + 1 < wps_.size(); ++seg) {
      const Waypoint& w0 = wps_[seg];
      const Waypoint& w1 = wps_[seg + 1];
      double t0 = w0.t, t1 = w1.t;
      double dt_seg = t1 - t0;

      for (size_t i = 0; i < n; ++i) {
        if (ts_[i] < t0 || ts_[i] > t1 + 1e-12) continue;
        // Skip the shared endpoint of the next segment (t1 belongs to seg+1
        // unless this is the final segment).
        if (ts_[i] == t1 && seg + 2 < wps_.size()) continue;

        double tau = std::clamp((ts_[i] - t0) / dt_seg, 0.0, 1.0);
        double tau2 = tau * tau, tau3 = tau2 * tau;
        // Cubic Hermite basis
        double h00 = 2 * tau3 - 3 * tau2 + 1;
        double h10 = tau3 - 2 * tau2 + tau;
        double h01 = -2 * tau3 + 3 * tau2;
        double h11 = tau3 - tau2;
        pts[i].pos = h00 * w0.pos + h10 * dt_seg * w0.vel + h01 * w1.pos +
                     h11 * dt_seg * w1.vel;
        double dh00 = (6 * tau2 - 6 * tau) / dt_seg;
        double dh10 = 3 * tau2 - 4 * tau + 1;
        double dh01 = (-6 * tau2 + 6 * tau) / dt_seg;
        double dh11 = 3 * tau2 - 2 * tau;
        pts[i].vel = dh00 * w0.pos + dh10 * w0.vel + dh01 * w1.pos +
                     dh11 * w1.vel;
        pts[i].att = quatSlerp(w0.att, w1.att, tau);
      }
    }

    // Angular velocity: omega_body = 2 * conj(q) (x) dq/dt (central diff).
    for (size_t i = 1; i + 1 < n; ++i) {
      Eigen::Vector4d q = pts[i].att;
      Eigen::Vector4d q_dot = (pts[i + 1].att - pts[i - 1].att) /
                              std::max(ts_[i + 1] - ts_[i - 1], 1e-9);
      // Quaternion inverse-product matrix (same as trajectory.py Q_conj).
      double w = q(0), x = q(1), y = q(2), z = q(3);
      Eigen::Matrix4d Q_conj;
      // clang-format off
      Q_conj <<  w,  x,  y,  z,
                -x,  w,  z, -y,
                -y, -z,  w,  x,
                -z,  y, -x,  w;
      // clang-format on
      Eigen::Vector4d prod = Q_conj * q_dot;
      pts[i].omega = 2.0 * prod.tail<3>();
    }

    // Acceleration: central finite difference of velocity.
    for (size_t i = 1; i + 1 < n; ++i) {
      pts[i].acc = (pts[i + 1].vel - pts[i - 1].vel) /
                   std::max(ts_[i + 1] - ts_[i - 1], 1e-9);
    }

    // Optional constant-speed renormalization (vehicle-like).
    if (const_speed_) {
      for (size_t i = 0; i < n; ++i) {
        double vnorm = pts[i].vel.norm();
        if (vnorm > 1e-9) {
          pts[i].vel = pts[i].vel / vnorm * *const_speed_;
          // Nonholonomic: yaw follows velocity heading, keep roll/pitch.
          double yaw = std::atan2(pts[i].vel.y(), pts[i].vel.x());
          Vec3 rpy = quatToEuler(pts[i].att);
          pts[i].att = eulerToQuat(rpy.x(), rpy.y(), yaw);
        }
      }
      for (size_t i = 0; i < n; ++i) pts[i].acc.setZero();
      for (size_t i = 1; i + 1 < n; ++i) {
        pts[i].acc = (pts[i + 1].vel - pts[i - 1].vel) /
                     std::max(ts_[i + 1] - ts_[i - 1], 1e-9);
      }
    }

    // Endpoints: copy neighbors.
    pts.front().omega = pts[1].omega;
    pts.back().omega = pts[n - 2].omega;
    pts.front().acc = pts[1].acc;
    pts.back().acc = pts[n - 2].acc;

    points_ = std::move(pts);
  }

  double dt_;
  std::optional<double> const_speed_;
  std::vector<Waypoint> wps_;
  std::vector<TrajPoint> points_;
  std::vector<double> ts_;
};

}  // namespace nav
