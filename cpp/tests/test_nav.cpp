/// Unit tests for the C++ nav port.  Build & run:  ./build/test_nav
/// Statistical parity bands vs the Python reference are documented in
/// cpp/README.md.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "nav/eskf.hpp"
#include "nav/gnss.hpp"
#include "nav/imu.hpp"
#include "nav/quaternion.hpp"
#include "nav/trajectory.hpp"

using namespace nav;

static int failures = 0;
#define CHECK_NEAR(a, b, tol, msg)                                   \
  do {                                                               \
    if (std::abs((a) - (b)) > (tol)) {                               \
      std::printf("FAIL %s: %f vs %f (tol %f)\n", msg, double(a),    \
                  double(b), double(tol));                           \
      ++failures;                                                    \
    }                                                                \
  } while (0)

static void testQuaternionRoundtrip() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-M_PI, M_PI);
  for (int i = 0; i < 200; ++i) {
    Eigen::Vector4d q = eulerToQuat(U(rng), U(rng) * 0.499, U(rng));
    Mat3 R = quatToRotmat(q);
    Eigen::Vector4d q2 = rotmatToQuat(R);
    // Sign-agnostic comparison
    if (q2.dot(q) < 0) q2 = -q2;
    CHECK_NEAR((q - q2).norm(), 0.0, 1e-9, "quat roundtrip");

    Vec3 rpy = quatToEuler(q);
    Eigen::Vector4d q3 = eulerToQuat(rpy.x(), rpy.y(), rpy.z());
    if (q3.dot(q) < 0) q3 = -q3;
    CHECK_NEAR((q - q3).norm(), 0.0, 1e-9, "euler roundtrip");
  }
}

static void testQuatBasics() {
  // 90 deg yaw
  Eigen::Vector4d q = eulerToQuat(0, 0, M_PI / 2);
  Vec3 v = quatRotate(q, {1, 0, 0});
  CHECK_NEAR(v.x(), 0.0, 1e-12, "rot x");
  CHECK_NEAR(v.y(), 1.0, 1e-12, "rot y");
  // exp/log roundtrip
  Vec3 rv(0.1, -0.2, 0.3);
  CHECK_NEAR((quatLog(quatExp(rv)) - rv).norm(), 0.0, 1e-12, "exp/log");
  // slerp endpoints + midpoint of 90deg yaw
  Eigen::Vector4d qa = eulerToQuat(0, 0, 0), qb = eulerToQuat(0, 0, M_PI / 2);
  CHECK_NEAR(quatToEuler(quatSlerp(qa, qb, 0.5)).z(), M_PI / 4, 1e-12,
             "slerp mid");
}

static void testTrajectoryGeometry() {
  // Straight line: Hermite should reproduce pos exactly, acc ~ 0.
  std::vector<Waypoint> wp = {{0.0, {0, 0, 0}, {10, 0, 0}},
                              {10.0, {100, 0, 0}, {10, 0, 0}}};
  Trajectory traj(wp, 0.01);
  const auto& pts = traj.points();
  assert(pts.size() == 1001);
  double max_pos_err = 0, max_acc = 0;
  for (const auto& p : pts) {
    max_pos_err = std::max(max_pos_err, std::abs(p.pos.x() - 10 * p.t));
    max_acc = std::max(max_acc, p.acc.norm());
    // yaw should be ~0 throughout
    max_acc = std::max(max_acc, std::abs(quatToEuler(p.att).z()));
  }
  CHECK_NEAR(max_pos_err, 0.0, 1e-9, "line pos");
  CHECK_NEAR(max_acc, 0.0, 1e-6, "line acc/yaw");

  // 90-deg arc turn: constant speed 10 m/s, omega = pi/10 rad/s.
  // NOTE: a 2-waypoint Hermite spline is NOT a true circle (it cuts the
  // corner): the Python reference gives body specific force ~2.84 m/s^2
  // at mid-turn (vs ideal v^2/R = 3.14).  The C++ port reproduces 2.8372
  // exactly — assert against that value.
  double R = 10.0 / (M_PI / 10);  // 31.83 m
  std::vector<Waypoint> wp2 = {
      {0.0, {0, 0, 0}, {10, 0, 0}, eulerToQuat(0, 0, 0)},
      {5.0, {R, R, 0}, {0, 10, 0}, eulerToQuat(0, 0, M_PI / 2)}};
  Trajectory traj2(wp2, 0.01, 10.0);  // const_speed
  for (const auto& p : traj2.points()) {
    CHECK_NEAR(p.vel.norm(), 10.0, 1e-6, "const speed");
  }
  TrajPoint mid = traj2.at(2.5);
  Vec3 a_b = traj2.bodyAccel(mid);
  double a_h = std::sqrt(a_b.x() * a_b.x() + a_b.y() * a_b.y());
  CHECK_NEAR(a_h, 2.8372, 0.02, "mid-turn specific force (py parity)");
}

static void testImuStatistics() {
  // Static IMU: measured accel mean should be ~ -g in body (level) + bias,
  // sample std of noise should match sigma_acc = density * sqrt(1/dt).
  ImuSpec spec = imuPreset(SensorGrade::kTactical);
  double dt = 0.01;
  ImuSensor imu(spec, dt, 123);
  Vec3 acc_mean = Vec3::Zero(), gyr_mean = Vec3::Zero();
  Vec3 acc_sq = Vec3::Zero();
  const int N = 200000;
  for (int i = 0; i < N; ++i) {
    Vec3 am, gm;
    imu.measure(Vec3(0, 0, kGravity), Vec3::Zero(), am, gm);
    acc_mean += am;
    acc_sq += am.cwiseProduct(am);
    gyr_mean += gm;
  }
  acc_mean /= N;
  gyr_mean /= N;
  // Expected mean = full error chain applied to specific force
  // (I + mis) * (scale * f) + bias — NOT raw bias alone: misalignment and
  // scale coupling leak ~g*0.15mrad into the horizontal axes (Python
  // reference shows the same ~0.02-0.03 m/s^2 offsets).  Residual slack
  // covers the time-averaged bias random walk (sigma_rw*sqrt(N*dt)/sqrt3
  // ~ 2.5e-3, so ~0.008 = 3 sigma; Python seed 123 shows -0.0056 in x).
  const ImuErrors& e = imu.errors();
  Vec3 acc_pred =
      (Mat3::Identity() + e.misalign_acc) * (e.scale_acc * Vec3(0, 0, kGravity)) +
      e.bias_acc;
  CHECK_NEAR(acc_mean.x(), acc_pred.x(), 0.008, "acc mean x (error chain)");
  CHECK_NEAR(acc_mean.y(), acc_pred.y(), 0.008, "acc mean y (error chain)");
  CHECK_NEAR(acc_mean.z(), acc_pred.z(), 0.02, "acc mean z");
  // Sample std (var = E[x^2] - E[x]^2)
  Vec3 acc_std = (acc_sq / N - acc_mean.cwiseProduct(acc_mean)).cwiseSqrt();
  double expected = spec.accNoiseMs2Hz() * std::sqrt(1.0 / dt);
  CHECK_NEAR(acc_std.z(), expected, 0.1 * expected, "acc noise std");
  // Gyro mean ~ bias (plus small random-walk drift)
  CHECK_NEAR(gyr_mean.norm(), imu.trueGyrBias().norm(), 2e-4, "gyr mean");
}

static void testGnssStatistics() {
  GnssSpec spec = gnssPreset(GnssGrade::kAutomotive);
  double dt = 0.01;
  GnssSensor gnss(spec, dt, 321);
  Vec3 true_pos(1000, 2000, 50), true_vel(10, 0, 0);
  Eigen::Vector2d err_sq = Eigen::Vector2d::Zero();
  int n_valid = 0, n_steps = 6000;  // 60 s
  for (int i = 0; i < n_steps; ++i) {
    auto m = gnss.measure(true_pos, true_vel);
    if (m.valid) {
      ++n_valid;
      Eigen::Vector2d e((*m.pos).x() - true_pos.x(), (*m.pos).y() - true_pos.y());
      err_sq += e.cwiseProduct(e);
    }
  }
  // Rate: nominal 5 Hz x 60 s = 300, minus dropout time.  Python
  // reference (seed 321): 186 valid, 24 dropouts.  Assert the band.
  CHECK_NEAR(double(n_valid), 185.0, 45.0, "gnss rate (incl. dropouts)");
  // Horizontal RMS should be ~ sqrt(pos_noise^2 + multipath^2) ~ 2.2 m
  double h_rms = std::sqrt(err_sq.sum() / n_valid);
  CHECK_NEAR(h_rms, std::sqrt(spec.pos_noise_h_m * spec.pos_noise_h_m +
                              spec.multipath_sigma_h_m * spec.multipath_sigma_h_m),
             0.35, "gnss h rms");
}

static void testEskfObservability() {
  // Full pipeline, tactical IMU + automotive GNSS: ESKF should converge
  // to ~2 m horizontal RMS and beat dead reckoning by ~5x.
  double dt = 0.01;
  uint64_t seed = 42;

  std::vector<Waypoint> wp = {
      {0.0, {0, 0, 0}, {10, 0, 0}, eulerToQuat(0, 0, 0)},
      {15.0, {150, 0, 0}, {10, 0, 0}, eulerToQuat(0, 0, 0)},
      {20.0, {150, 50, 0}, {0, 10, 0}, eulerToQuat(0, 0, M_PI / 2)},
      {40.0, {150, 250, 0}, {0, 10, 0}, eulerToQuat(0, 0, M_PI / 2)},
      {60.0, {50, 350, 0}, {-10, 0, 0}, eulerToQuat(0, 0, M_PI)}};
  Trajectory traj(wp, dt);
  ImuSensor imu(SensorGrade::kTactical, dt, seed);
  GnssSensor gnss(GnssGrade::kAutomotive, dt, seed + 1);

  EskfConfig cfg;
  cfg.gnss_pos_std = 2.2;
  cfg.gnss_vel_std = 0.1;
  cfg.init_pos_std = 5.0;
  cfg.init_att_std_deg = 5.0;
  Eskf eskf(cfg);
  const auto& pts = traj.points();
  eskf.setInitialState(pts.front().t, pts.front().pos + Vec3(3, -2, 1),
                       pts.front().vel, pts.front().att);

  Vec3 dr_p = pts.front().pos, dr_v = pts.front().vel;
  Eigen::Vector4d dr_q = pts.front().att;
  double eskf_sq = 0, dr_sq = 0;
  int n = 0;
  for (const auto& pt : pts) {
    Vec3 am, wm;
    imu.measure(traj.bodyAccel(pt), pt.omega, am, wm);
    eskf.predict(am, wm, dt);
    dr_q = quatNormalize(quatMultiply(dr_q, quatExp(wm * dt)));
    dr_v += (quatToRotmat(dr_q) * am + Vec3(0, 0, -kGravity)) * dt;
    dr_p += dr_v * dt;
    auto m = gnss.measure(pt.pos, pt.vel);
    if (m.valid) eskf.updateGnss({*m.pos, m.vel, std::nullopt, std::nullopt});
    if (pt.t > 10.0) {  // settled region only
      double e = (eskf.position() - pt.pos).norm();
      eskf_sq += e * e;
      dr_sq += (dr_p - pt.pos).squaredNorm();
      ++n;
    }
  }
  double eskf_rms = std::sqrt(eskf_sq / n);
  double dr_rms = std::sqrt(dr_sq / n);
  std::printf("  [info] eskf_rms=%.2f m  dr_rms=%.1f m  ratio=%.1f\n", eskf_rms,
              dr_rms, dr_rms / eskf_rms);
  if (!(eskf_rms < 3.5)) { std::printf("FAIL eskf rms %.2f\n", eskf_rms); ++failures; }
  if (!(dr_rms > 3.0 * eskf_rms)) {
    std::printf("FAIL dr/eskf ratio %.1f\n", dr_rms / eskf_rms);
    ++failures;
  }

  // Bias observability: accel bias estimate within 0.02 m/s^2 of truth.
  auto [ba_est, bg_est] = eskf.estimatedBiases();
  Vec3 ba_err = ba_est - imu.trueAccBias();
  if (!(ba_err.norm() < 0.02)) {
    std::printf("FAIL acc bias err %.4f\n", ba_err.norm());
    ++failures;
  }
}

static void testEskfOutlierGate() {
  // A 100 m position jump must be rejected by the 6-sigma gate.
  EskfConfig cfg;
  cfg.gnss_pos_std = 2.0;
  Eskf eskf(cfg);
  eskf.setInitialState(0.0, Vec3::Zero(), Vec3::Zero(),
                       Eigen::Vector4d(1, 0, 0, 0));
  // P is huge (init 10 m), so inflate the gate scenario: predict a bit with
  // tiny dt to shrink relative innovation, then feed an absurd jump.
  for (int i = 0; i < 100; ++i) eskf.predict(Vec3(0, 0, kGravity), Vec3::Zero(), 0.01);
  bool accepted = eskf.updateGnss({Vec3(100, 0, 0), Vec3::Zero(), 2.0, 0.2});
  if (accepted) { std::printf("FAIL outlier gate accepted 100 m jump\n"); ++failures; }
  CHECK_NEAR(eskf.position().x(), 0.0, 1e-6, "gate kept position");
}

int main() {
  testQuaternionRoundtrip();
  testQuatBasics();
  testTrajectoryGeometry();
  testImuStatistics();
  testGnssStatistics();
  testEskfObservability();
  testEskfOutlierGate();
  if (failures == 0) {
    std::printf("All nav tests passed ✅\n");
    return 0;
  }
  std::printf("%d test failure(s) ❌\n", failures);
  return 1;
}
