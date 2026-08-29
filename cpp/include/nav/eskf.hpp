/**
 * Error-State Kalman Filter for GNSS+IMU loose coupling — port of
 * sensor_sim/eskf.py (Solà 2017 / Groves 2013 Ch.14 formulation).
 *
 *   Nominal state:  x = [p, v, q, b_a, b_g]
 *   Error state:   dx = [dp, dv, dtheta, db_a, db_g]   (15 dims)
 *
 * Parity notes (vs eskf.py):
 *   - predict():  F is first-order discretized (I + Fc*dt); Qd has ZERO
 *     position rows (qd[0:3] == 0), velocity rows carry the accel noise
 *     density.  Do not "fix" this; it mirrors the Python reference.
 *   - update_gnss(): position-only case fills H velocity rows with zeros
 *     and R velocity entries with 1e12; the 6-sigma gate checks all 6
 *     innovation entries (velocity ones are 0 in position-only mode and
 *     can never trigger).
 */

#pragma once

#include <cmath>

#include "nav/quaternion.hpp"
#include "nav/types.hpp"

namespace nav {

/// 15-dim error-state index layout (block starts).
struct EskfIdx {
  static constexpr int kP = 0;       // dp
  static constexpr int kV = 3;       // dv
  static constexpr int kTheta = 6;   // dtheta
  static constexpr int kBa = 9;      // db_a
  static constexpr int kBg = 12;     // db_g
  static constexpr int kDim = 15;
};

struct EskfConfig {
  // Process noise densities
  double acc_noise_density = 1.0e-2;   // m/s^2/sqrt(Hz)
  double gyr_noise_density = 1.0e-3;   // rad/s/sqrt(Hz)
  double acc_bias_rw = 1.0e-4;         // m/s^3/sqrt(Hz)
  double gyr_bias_rw = 1.0e-5;         // rad/s^2/sqrt(Hz)

  // Initial error covariance (1-sigma, diagonal)
  double init_pos_std = 10.0;          // m
  double init_vel_std = 5.0;           // m/s
  double init_att_std_deg = 10.0;      // deg
  double init_accbias_std = 1.0e-1;    // m/s^2
  double init_gyrbias_std = 1.0e-2;    // rad/s

  // GNSS measurement noise (1 sigma)
  double gnss_pos_std = 1.5;           // m
  double gnss_vel_std = 0.2;           // m/s
};

class Eskf {
 public:
  explicit Eskf(const EskfConfig& cfg = {}) : cfg_(cfg) { initCovariance(); }

  /// Seed the filter with an initial (possibly approximate) pose.
  void setInitialState(double t, const Vec3& p, const Vec3& v,
                       const Eigen::Vector4d& q, const Vec3& b_a = Vec3::Zero(),
                       const Vec3& b_g = Vec3::Zero()) {
    t_ = t;
    p_ = p;
    v_ = v;
    q_ = quatNormalize(q);
    b_a_ = b_a;
    b_g_ = b_g;
  }

  /// Propagate nominal state + error covariance with one IMU sample.
  /// acc/gyro are the raw (biased) sensor outputs in the body frame.
  void predict(const Vec3& acc, const Vec3& gyro, double dt) {
    if (dt <= 0) return;

    Vec3 a = acc - b_a_;
    Vec3 w = gyro - b_g_;
    Mat3 R = quatToRotmat(q_);  // body -> world
    Vec3 g(0.0, 0.0, -kGravity);

    // --- nominal state integration (same equations as eskf.py) --------
    Vec3 a_w = R * a + g;
    p_ += v_ * dt + 0.5 * a_w * dt * dt;
    v_ += a_w * dt;
    q_ = quatNormalize(quatMultiply(q_, quatExp(w * dt)));

    // --- error-state covariance propagation ----------------------------
    // F_d = I + F_c * dt (Solà 2017 eq. 178-180, first-order).
    MatX F = MatX::Identity(EskfIdx::kDim, EskfIdx::kDim);
    F.block<3, 3>(EskfIdx::kP, EskfIdx::kV) = Mat3::Identity() * dt;
    F.block<3, 3>(EskfIdx::kV, EskfIdx::kTheta) = (-R * skew(a)) * dt;
    F.block<3, 3>(EskfIdx::kV, EskfIdx::kBa) = -R * dt;
    F.block<3, 3>(EskfIdx::kTheta, EskfIdx::kTheta) =
        Mat3::Identity() - skew(w) * dt;
    F.block<3, 3>(EskfIdx::kTheta, EskfIdx::kBg) = -Mat3::Identity() * dt;

    // Q_d: position rows zero (see header parity note).
    Vec15 qd = Vec15::Zero();
    qd.segment<3>(EskfIdx::kV) =
        Vec3::Constant(cfg_.acc_noise_density * cfg_.acc_noise_density) * dt;
    qd.segment<3>(EskfIdx::kTheta) =
        Vec3::Constant(cfg_.gyr_noise_density * cfg_.gyr_noise_density) * dt;
    qd.segment<3>(EskfIdx::kBa) =
        Vec3::Constant(cfg_.acc_bias_rw * cfg_.acc_bias_rw) * dt;
    qd.segment<3>(EskfIdx::kBg) =
        Vec3::Constant(cfg_.gyr_bias_rw * cfg_.gyr_bias_rw) * dt;

    P_ = F * P_ * F.transpose() + MatX(qd.asDiagonal());
    t_ += dt;
  }

  /// Loose-coupling update with GNSS position (+ optionally velocity).
  struct GnssUpdate {
    Vec3 pos;
    std::optional<Vec3> vel;
    std::optional<double> pos_std;
    std::optional<double> vel_std;
  };

  /// @return true if the update was accepted (passed the 6-sigma gate).
  bool updateGnss(const GnssUpdate& z) {
    double p_std = z.pos_std.value_or(cfg_.gnss_pos_std);
    double v_std = z.vel_std.value_or(cfg_.gnss_vel_std);

    MatX H = MatX::Zero(6, EskfIdx::kDim);
    H.block<3, 3>(0, EskfIdx::kP) = Mat3::Identity();
    MatX R = MatX::Zero(6, 6);
    for (int i = 0; i < 3; ++i) R(i, i) = p_std * p_std;

    Eigen::Matrix<double, 6, 1> h, dz;
    h.head<3>() = p_;
    if (z.vel) {
      H.block<3, 3>(3, EskfIdx::kV) = Mat3::Identity();
      for (int i = 0; i < 3; ++i) R(3 + i, 3 + i) = v_std * v_std;
      h.tail<3>() = v_;
      dz.head<3>() = z.pos - p_;
      dz.tail<3>() = *z.vel - v_;
    } else {
      // Position-only: velocity rows observed with "infinite" noise.
      // H velocity block stays zero (matches eskf.py), R = 1e12.
      for (int i = 0; i < 3; ++i) R(3 + i, 3 + i) = 1e12;
      h.tail<3>() = v_;
      dz.head<3>() = z.pos - p_;
      dz.tail<3>().setZero();
    }

    // 6-sigma outlier gate over ALL innovation entries.  Multiplier lets
    // callers widen the gate during re-acquisition without touching P.
    MatX S = H * P_ * H.transpose() + R;
    Eigen::Matrix<double, 6, 1> innov_cov = S.diagonal().cwiseSqrt();
    for (int i = 0; i < 6; ++i) {
      if (std::abs(dz(i)) >
          6.0 * gate_multiplier *
              std::max(innov_cov(i), 1e-9)) {
        return false;
      }
    }

    MatX K = P_ * H.transpose() * S.inverse();
    Vec15 dx = K * dz;
    inject(dx);

    // Joseph-form covariance update + symmetrization.
    MatX I = MatX::Identity(EskfIdx::kDim, EskfIdx::kDim);
    MatX IKH = I - K * H;
    P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
    ++n_updates_;
    return true;
  }

  // ---- accessors -------------------------------------------------------
  const Vec3& position() const { return p_; }
  const Vec3& velocity() const { return v_; }
  const Eigen::Vector4d& quaternion() const { return q_; }
  Mat3 rotationMatrix() const { return quatToRotmat(q_); }
  Vec3 attitudeEuler() const { return quatToEuler(q_); }
  std::pair<Vec3, Vec3> estimatedBiases() const { return {b_a_, b_g_}; }
  const MatX& covariance() const { return P_; }
  int numUpdates() const { return n_updates_; }
  double time() const { return t_; }

  /// Outlier gate width multiplier (>1 widens, e.g. during re-acquisition).
  double gate_multiplier = 1.0;

 private:
  using Vec15 = Eigen::Matrix<double, EskfIdx::kDim, 1>;

  /// Add the error state to the nominal state (ESKF reset step).
  void inject(const Vec15& dx) {
    p_ += dx.segment<3>(EskfIdx::kP);
    v_ += dx.segment<3>(EskfIdx::kV);
    q_ = quatNormalize(quatMultiply(q_, quatExp(dx.segment<3>(EskfIdx::kTheta))));
    b_a_ += dx.segment<3>(EskfIdx::kBa);
    b_g_ += dx.segment<3>(EskfIdx::kBg);
  }

  void initCovariance() {
    Vec15 std15;
    std15 << Vec3::Constant(cfg_.init_pos_std),
        Vec3::Constant(cfg_.init_vel_std),
        Vec3::Constant(cfg_.init_att_std_deg * M_PI / 180.0),
        Vec3::Constant(cfg_.init_accbias_std),
        Vec3::Constant(cfg_.init_gyrbias_std);
    P_ = std15.cwiseProduct(std15).asDiagonal();
  }

  EskfConfig cfg_;
  double t_ = 0.0;
  Vec3 p_ = Vec3::Zero();
  Vec3 v_ = Vec3::Zero();
  Eigen::Vector4d q_ = Eigen::Vector4d(1, 0, 0, 0);
  Vec3 b_a_ = Vec3::Zero();
  Vec3 b_g_ = Vec3::Zero();
  MatX P_ = MatX::Identity(EskfIdx::kDim, EskfIdx::kDim);
  int n_updates_ = 0;
};

}  // namespace nav
