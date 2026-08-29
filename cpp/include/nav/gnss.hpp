/**
 * GNSS error model — port of sensor_sim/gnss.py.
 * White noise + Gauss-Markov multipath + dropout state machine + rate
 * control.  Measure order matters: multipath evolves FIRST, then dropout
 * check, then output-rate check, then noise draw (matches gnss.py).
 */

#pragma once

#include <optional>

#include "nav/random.hpp"
#include "nav/types.hpp"

namespace nav {

using Vec2 = Eigen::Vector2d;

enum class GnssGrade { kConsumer, kAutomotive, kRtk, kSurvey };

struct GnssSpec {
  double pos_noise_h_m = 0.0;
  double pos_noise_v_m = 0.0;
  double vel_noise_ms = 0.0;
  double multipath_tau_s = 0.0;
  double multipath_sigma_h_m = 0.0;
  double multipath_sigma_v_m = 0.0;
  double dropout_prob = 0.0;           // per-measurement probability
  double mean_loss_duration_s = 0.0;   // mean outage duration (s)
  double output_rate_hz = 0.0;
};

inline GnssSpec gnssPreset(GnssGrade grade) {
  switch (grade) {
    case GnssGrade::kConsumer:
      return {3.0, 6.0, 0.1, 30.0, 5.0, 10.0, 0.01, 2.0, 1.0};
    case GnssGrade::kAutomotive:
      return {1.0, 2.0, 0.05, 20.0, 2.0, 4.0, 0.005, 1.0, 5.0};
    case GnssGrade::kRtk:
      return {0.02, 0.04, 0.01, 10.0, 0.05, 0.10, 0.001, 0.5, 10.0};
    case GnssGrade::kSurvey:
      return {0.005, 0.010, 0.005, 5.0, 0.02, 0.04, 0.0005, 0.3, 20.0};
  }
  return {};
}

struct GnssStats {
  int num_dropouts = 0;
  double total_dropout_time_s = 0.0;
};

class GnssSensor {
 public:
  GnssSensor(GnssGrade grade, double dt, uint64_t seed)
      : GnssSensor(gnssPreset(grade), dt, seed) {}

  GnssSensor(GnssSpec spec, double dt, uint64_t seed)
      : spec_(spec), dt_(dt), rng_(seed) {
    double tau = spec_.multipath_tau_s;
    mp_alpha_ = tau > 0 ? std::exp(-dt_ / tau) : 0.0;
    mp_drive_h_ =
        tau > 0 ? spec_.multipath_sigma_h_m * std::sqrt(1 - mp_alpha_ * mp_alpha_) : 0.0;
    mp_drive_v_ =
        tau > 0 ? spec_.multipath_sigma_v_m * std::sqrt(1 - mp_alpha_ * mp_alpha_) : 0.0;
    output_interval_steps_ =
        std::max(1, static_cast<int>(std::lround(1.0 / spec_.output_rate_hz / dt_)));
  }

  struct Measurement {
    std::optional<Vec3> pos;   ///< nullopt during dropout / non-output step
    std::optional<Vec3> vel;
    bool valid = false;
  };

  Measurement measure(const Vec3& pos_true, const std::optional<Vec3>& vel_true) {
    evolveMultipath();

    if (in_dropout_) {
      dropout_remaining_ -= dt_;
      stats_.total_dropout_time_s += dt_;
      if (dropout_remaining_ <= 0) in_dropout_ = false;
      return {};
    }
    if (rng_.uniform01() < spec_.dropout_prob) {
      in_dropout_ = true;
      dropout_remaining_ = rng_.exponential(spec_.mean_loss_duration_s);
      ++stats_.num_dropouts;
      return {};
    }
    if (++steps_since_output_ < output_interval_steps_) return {};
    steps_since_output_ = 0;

    Vec3 noise_h{rng_.standard() * spec_.pos_noise_h_m,
                 rng_.standard() * spec_.pos_noise_h_m, 0.0};
    double noise_v = rng_.standard() * spec_.pos_noise_v_m;

    Vec3 pos = pos_true;
    pos.x() += noise_h.x() + mp_h_.x();   // horizontal (E, N) in x, y
    pos.y() += noise_h.y() + mp_h_.y();
    pos.z() += noise_v + mp_v_;

    std::optional<Vec3> vel;
    if (vel_true) {
      vel = *vel_true + rng_.vec3() * spec_.vel_noise_ms;
    }

    last_position_ = pos;
    last_velocity_ = vel;
    return {pos, vel, true};
  }

  /// Convenience overload: measure with velocity (common case).
  Measurement measure(const Vec3& pos_true, const Vec3& vel_true) {
    return measure(pos_true, std::optional<Vec3>(vel_true));
  }

  const GnssStats& stats() const { return stats_; }
  const std::optional<Vec3>& lastPosition() const { return last_position_; }
  const std::optional<Vec3>& lastVelocity() const { return last_velocity_; }

  void reset() {
    mp_h_.setZero();
    mp_v_ = 0.0;
    in_dropout_ = false;
    dropout_remaining_ = 0.0;
    steps_since_output_ = 0;
    stats_ = {};
    last_position_.reset();
    last_velocity_.reset();
  }

 private:
  void evolveMultipath() {
    Vec2 drive_h{rng_.standard() * mp_drive_h_, rng_.standard() * mp_drive_h_};
    double drive_v = rng_.standard() * mp_drive_v_;
    mp_h_ = mp_alpha_ * mp_h_ + drive_h;
    mp_v_ = mp_alpha_ * mp_v_ + drive_v;
  }

  GnssSpec spec_;
  double dt_;
  Gaussian rng_;

  Vec2 mp_h_ = Vec2::Zero();
  double mp_v_ = 0.0;
  double mp_alpha_ = 0.0;
  double mp_drive_h_ = 0.0;
  double mp_drive_v_ = 0.0;

  bool in_dropout_ = false;
  double dropout_remaining_ = 0.0;
  int output_interval_steps_ = 1;
  int steps_since_output_ = 0;

  GnssStats stats_;
  std::optional<Vec3> last_position_;
  std::optional<Vec3> last_velocity_;
};

}  // namespace nav
