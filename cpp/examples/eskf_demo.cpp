/**
 * End-to-end GNSS+IMU loose-coupling fusion demo — C++ port of
 * examples/eskf_demo.py (same 60 s trajectory, tactical IMU @ 100 Hz,
 * automotive GNSS @ 5 Hz, 3 m initial position error).
 *
 * Build & run from cpp/:  ./build/eskf_demo --seed 42
 * Reference (Python, seed 42):  ESKF pos RMS ~2.3 m, DR mean ~12 m,
 * yaw err mean ~0.2 deg.  The C++ port should land in the same band
 * (streams differ; see cpp/README.md).
 */

#include <chrono>
#include <cmath>
#include <cstdio>

#include "nav/eskf.hpp"
#include "nav/gnss.hpp"
#include "nav/imu.hpp"
#include "nav/trajectory.hpp"

using namespace nav;

static Trajectory buildTrajectory(double dt) {
  using V4 = Eigen::Vector4d;
  auto yawQuat = [](double yaw) { return V4(std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2)); };

  std::vector<Waypoint> wp = {
      {0.0, {0, 0, 0}, {10, 0, 0}, yawQuat(0), {0, 0, 0}},
      {15.0, {150, 0, 0}, {10, 0, 0}, yawQuat(0), {0, 0, 0}},
      // Right turn: 90 deg over 5 s at 10 m/s
      {20.0, {150, 50, 0}, {0, 10, 0}, yawQuat(M_PI / 2), {0, 0, M_PI / 10}},
      {40.0, {150, 250, 0}, {0, 10, 0}, yawQuat(M_PI / 2), {0, 0, 0}},
      // Left turn back
      {50.0, {50, 250, 0}, {-10, 0, 0}, yawQuat(M_PI), {0, 0, -M_PI / 10}},
      {60.0, {-50, 250, 0}, {-10, 0, 0}, yawQuat(2 * M_PI), {0, 0, 0}},
  };
  return Trajectory(wp, dt);
}

int main(int argc, char** argv) {
  uint64_t seed = 42;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--seed") seed = std::stoul(argv[i + 1]);
  }
  const double dt_imu = 0.01;  // 100 Hz
  const double T = 60.0;

  Trajectory traj = buildTrajectory(dt_imu);
  ImuSensor imu(SensorGrade::kTactical, dt_imu, seed);
  GnssSensor gnss(GnssGrade::kAutomotive, dt_imu, seed + 1);

  // Match the Python demo tuning: automotive-grade GNSS -> 2.2 m, 0.1 m/s.
  EskfConfig cfg;
  cfg.acc_noise_density = 1e-2;
  cfg.gyr_noise_density = 1e-3;
  cfg.acc_bias_rw = 1e-4;
  cfg.gyr_bias_rw = 1e-5;
  cfg.init_pos_std = 5.0;
  cfg.init_att_std_deg = 5.0;
  cfg.gnss_pos_std = 2.2;
  cfg.gnss_vel_std = 0.1;
  Eskf eskf(cfg);

  const auto& pts = traj.points();
  const TrajPoint& p0 = pts.front();
  eskf.setInitialState(p0.t, p0.pos + Vec3(3, -2, 1.0), p0.vel, p0.att);

  // Pure dead-reckoning copy (no updates).
  Vec3 dr_p = p0.pos, dr_v = p0.vel;
  Eigen::Vector4d dr_q = p0.att;

  std::vector<double> t_rec, ep_rec, ev_rec, ey_rec, dr_rec;
  Vec3 ba_est_last = Vec3::Zero(), ba_true_last = Vec3::Zero();
  Vec3 bg_est = Vec3::Zero(), bg_true = Vec3::Zero();

  auto t_start = std::chrono::steady_clock::now();
  for (const auto& pt : pts) {
    Vec3 a_true = traj.bodyAccel(pt);
    Vec3 a_m, w_m;
    imu.measure(a_true, pt.omega, a_m, w_m);

    eskf.predict(a_m, w_m, dt_imu);

    // Dead reckoning: same mechanization, no GNSS.
    dr_q = quatNormalize(quatMultiply(dr_q, quatExp(w_m * dt_imu)));
    Vec3 dr_a_w = quatToRotmat(dr_q) * a_m + Vec3(0, 0, -kGravity);
    dr_v += dr_a_w * dt_imu;
    dr_p += dr_v * dt_imu;

    auto m = gnss.measure(pt.pos, pt.vel);
    if (m.valid) {
      eskf.updateGnss({*m.pos, m.vel, std::nullopt, std::nullopt});
    }

    // Record errors at 2 Hz.
    if (std::abs(pt.t - std::round(pt.t / 0.5) * 0.5) < dt_imu * 0.5) {
      t_rec.push_back(pt.t);
      ep_rec.push_back((eskf.position() - pt.pos).norm());
      ev_rec.push_back((eskf.velocity() - pt.vel).norm());
      double dyaw = quatToEuler(eskf.quaternion()).z() -
                    quatToEuler(pt.att).z();
      // Wrap to [-pi, pi] (the Python demo metric lacks this and spikes
      // to ~360 deg when either yaw crosses the +/-pi boundary).
      dyaw = std::atan2(std::sin(dyaw), std::cos(dyaw));
      ey_rec.push_back(std::abs(dyaw));
      dr_rec.push_back((dr_p - pt.pos).norm());
      ba_est_last = eskf.estimatedBiases().first;
      bg_est = eskf.estimatedBiases().second;
      ba_true_last = imu.trueAccBias();
      bg_true = imu.trueGyrBias();
    }
  }
  double t_el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

  auto mean = [](const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size(); };
  auto rms = [&mean](const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x * x; return std::sqrt(s / v.size()); };
  auto mx = [](const std::vector<double>& v) {
    return *std::max_element(v.begin(), v.end()); };

  std::printf("=== ESKF GNSS+IMU Loose Coupling (C++ port, seed=%llu, %.1fs sim) ===\n",
              static_cast<unsigned long long>(seed), t_el);
  std::printf("\nIMU-only dead reckoning position error:  mean %6.2f m  max %6.2f m\n",
              mean(dr_rec), mx(dr_rec));
  std::printf("ESKF position error:                    mean %6.3f m  max %6.3f m   (RMS %.3f m)\n",
              mean(ep_rec), mx(ep_rec), rms(ep_rec));
  std::printf("ESKF velocity error:                    mean %6.3f m/s max %6.3f m/s\n",
              mean(ev_rec), mx(ev_rec));
  std::printf("ESKF yaw error:                         mean %6.3f deg  max %6.3f deg\n",
              mean(ey_rec) * 180.0 / M_PI, mx(ey_rec) * 180.0 / M_PI);
  std::printf("\nAccel bias estimate (last 10 s):  est [%.4f %.4f %.4f]  true [%.4f %.4f %.4f]\n",
              ba_est_last.x(), ba_est_last.y(), ba_est_last.z(),
              ba_true_last.x(), ba_true_last.y(), ba_true_last.z());
  std::printf("Gyro bias estimate:               est [%.5f %.5f %.5f]  true [%.5f %.5f %.5f]\n",
              bg_est.x(), bg_est.y(), bg_est.z(), bg_true.x(), bg_true.y(), bg_true.z());
  std::printf("GNSS dropouts: %d (%.1f s)\n", gnss.stats().num_dropouts,
              gnss.stats().total_dropout_time_s);
  return 0;
}
