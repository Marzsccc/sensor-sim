/**
 * IMU error model — port of sensor_sim/imu.py (Allan-variance / IEEE 952
 * style per-axis error model: bias, scale factor + cross coupling,
 * misalignment, white noise, bias random walk).
 */

#pragma once

#include <map>
#include <string>

#include "nav/random.hpp"
#include "nav/types.hpp"

namespace nav {

enum class SensorGrade { kConsumer, kTactical, kNavigation };

struct ImuSpec {
  // Accelerometer
  double acc_bias_mg = 0.0;                 // bias repeatability (milli-g)
  double acc_noise_ug_hz = 0.0;             // velocity random walk (ug/rtHz)
  double acc_bias_instability_ug = 0.0;     // in-run instability (ug)
  double acc_scale_ppm = 0.0;               // scale factor error (ppm)
  // Gyroscope
  double gyr_bias_deg_h = 0.0;              // bias repeatability (deg/hr)
  double gyr_noise_deg_h_hz = 0.0;          // angular random walk (deg/rtHr)
  double gyr_bias_instability_deg_h = 0.0;  // in-run instability (deg/hr)
  double gyr_scale_ppm = 0.0;               // scale factor error (ppm)
  // Common
  double misalignment_arcmin = 0.0;         // axis misalignment (arcmin)

  // ---- unit conversions (identical factors to imu.py) -----------------
  double accBiasMs2() const { return acc_bias_mg * 9.81e-3; }
  double accNoiseMs2Hz() const { return acc_noise_ug_hz * 9.81e-6; }
  double accBiasInstabilityMs2() const {
    return acc_bias_instability_ug * 9.81e-6;
  }
  double gyrBiasRadS() const {
    return gyr_bias_deg_h * M_PI / 180.0 / 3600.0;
  }
  /// deg/sqrt(h) -> rad/sqrt(s): divide by 60 (sqrt(3600)), NOT 3600.
  double gyrNoiseRadSHz() const {
    return gyr_noise_deg_h_hz * M_PI / 180.0 / 60.0;
  }
  double gyrBiasInstabilityRadS() const {
    return gyr_bias_instability_deg_h * M_PI / 180.0 / 3600.0;
  }
  double misalignmentRad() const {
    return misalignment_arcmin * M_PI / 180.0 / 60.0;
  }
};

/// Preset grades (Groves 2013 / datasheet values, same as imu.py).
inline ImuSpec imuPreset(SensorGrade grade) {
  switch (grade) {
    case SensorGrade::kConsumer:
      return {/*acc_bias_mg*/ 50.0,  /*acc_noise*/ 200.0,
              /*acc_inst*/ 100.0,    /*acc_scale_ppm*/ 10000,
              /*gyr_bias*/ 10.0,     /*gyr_noise*/ 0.5,
              /*gyr_inst*/ 10.0,     /*gyr_scale_ppm*/ 10000,
              /*misalign*/ 30.0};
    case SensorGrade::kTactical:
      return {1.0, 50.0, 10.0, 1000, 1.0, 0.05, 0.5, 1000, 5.0};
    case SensorGrade::kNavigation:
      return {0.025, 5.0, 1.0, 100, 0.001, 0.001, 0.001, 10, 0.5};
  }
  return {};
}

/// Static error components, for inspection/plotting (mirrors get_errors()).
struct ImuErrors {
  Vec3 bias_acc;
  Vec3 bias_gyr;
  Mat3 scale_acc;
  Mat3 scale_gyr;
  Mat3 misalign_acc;
  Mat3 misalign_gyr;
  double noise_sigma_acc = 0.0;
  double noise_sigma_gyr = 0.0;
};

class ImuSensor {
 public:
  /// @param spec grade preset or a custom ImuSpec
  /// @param dt    output period (s), typically 0.005-0.01
  ImuSensor(SensorGrade grade, double dt, uint64_t seed)
      : ImuSensor(imuPreset(grade), dt, seed) {}

  ImuSensor(ImuSpec spec, double dt, uint64_t seed)
      : spec_(spec), dt_(dt), rng_(seed) {
    bias_acc_ = rng_.vec3() * spec.accBiasMs2();
    bias_gyr_ = rng_.vec3() * spec.gyrBiasRadS();
    scale_acc_ = drawScaleMatrix(spec.acc_scale_ppm);
    scale_gyr_ = drawScaleMatrix(spec.gyr_scale_ppm);
    mis_acc_ = drawMisalignment();
    mis_gyr_ = drawMisalignment();
    rw_sigma_acc_ = spec.accBiasInstabilityMs2();
    rw_sigma_gyr_ = spec.gyrBiasInstabilityRadS();
    noise_sigma_acc_ = spec.accNoiseMs2Hz() * std::sqrt(1.0 / dt_);
    noise_sigma_gyr_ = spec.gyrNoiseRadSHz() * std::sqrt(1.0 / dt_);
  }

  /// Generate one noisy measurement.  acc_true = specific force in body
  /// frame (m/s^2), omega_true = body angular rate (rad/s).
  void measure(const Vec3& acc_true, const Vec3& omega_true, Vec3& acc_meas,
               Vec3& gyr_meas) {
    evolveBias();
    Vec3 bias_acc_total = bias_acc_ + rw_acc_;
    Vec3 bias_gyr_total = bias_gyr_ + rw_gyr_;
    Vec3 noise_acc = rng_.vec3() * noise_sigma_acc_;
    Vec3 noise_gyr = rng_.vec3() * noise_sigma_gyr_;
    acc_meas = (Mat3::Identity() + mis_acc_) * (scale_acc_ * acc_true) +
               bias_acc_total + noise_acc;
    gyr_meas = (Mat3::Identity() + mis_gyr_) * (scale_gyr_ * omega_true) +
               bias_gyr_total + noise_gyr;
  }

  const ImuErrors& errors() const {
    errs_.bias_acc = bias_acc_;
    errs_.bias_gyr = bias_gyr_;
    errs_.scale_acc = scale_acc_;
    errs_.scale_gyr = scale_gyr_;
    errs_.misalign_acc = mis_acc_;
    errs_.misalign_gyr = mis_gyr_;
    errs_.noise_sigma_acc = noise_sigma_acc_;
    errs_.noise_sigma_gyr = noise_sigma_gyr_;
    return errs_;
  }

  /// Ground-truth bias draws (for validating ESKF bias observability).
  const Vec3& trueAccBias() const { return bias_acc_; }
  const Vec3& trueGyrBias() const { return bias_gyr_; }

  void resetBiasWalk() {
    rw_acc_.setZero();
    rw_gyr_.setZero();
  }

 private:
  Mat3 drawScaleMatrix(double scale_ppm) {
    double frac = scale_ppm * 1e-6;
    Vec3 diag = Vec3::Ones() + rng_.vec3() * frac / std::sqrt(3.0);
    Mat3 off;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) off(r, c) = rng_.standard() * frac * 0.1;
    off.diagonal().setZero();
    return Mat3(diag.asDiagonal()) + off;
  }

  Mat3 drawMisalignment() {
    return skew(rng_.vec3() * spec_.misalignmentRad() / std::sqrt(3.0));
  }

  void evolveBias() {
    rw_acc_ += rng_.vec3() * rw_sigma_acc_ * std::sqrt(dt_);
    rw_gyr_ += rng_.vec3() * rw_sigma_gyr_ * std::sqrt(dt_);
  }

  ImuSpec spec_;
  double dt_;
  Gaussian rng_;

  Vec3 bias_acc_ = Vec3::Zero();
  Vec3 bias_gyr_ = Vec3::Zero();
  Mat3 scale_acc_ = Mat3::Identity();
  Mat3 scale_gyr_ = Mat3::Identity();
  Mat3 mis_acc_ = Mat3::Zero();
  Mat3 mis_gyr_ = Mat3::Zero();
  Vec3 rw_acc_ = Vec3::Zero();
  Vec3 rw_gyr_ = Vec3::Zero();
  double rw_sigma_acc_ = 0.0;
  double rw_sigma_gyr_ = 0.0;
  double noise_sigma_acc_ = 0.0;
  double noise_sigma_gyr_ = 0.0;

  mutable ImuErrors errs_;
};

}  // namespace nav
