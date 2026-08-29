/**
 * Random number generation (mt19937) — port of the RandomState usage in
 * sensor_sim.  NumPy and std::mt19937 use different stream layouts, so
 * samples are not bit-identical; parity is statistical, not per-draw.
 */

#pragma once

#include <random>

#include "nav/types.hpp"

namespace nav {

/// Box-Muller polar (Marsaglia) Gaussian, cached second draw — same
/// algorithm family as NumPy's legacy RandomState normal().
class Gaussian {
 public:
  explicit Gaussian(uint64_t seed) : rng_(seed) {}

  double standard() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    double u, v, s;
    do {
      u = uniform_01_(rng_) * 2.0 - 1.0;
      v = uniform_01_(rng_) * 2.0 - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    double factor = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * factor;
    has_spare_ = true;
    return u * factor;
  }

  /// Vector of independent N(0, 1) draws.
  Vec3 vec3() { return {standard(), standard(), standard()}; }

  /// Uniform [0, 1).
  double uniform01() { return uniform_01_(rng_); }

  /// Exponential with mean `mean` (matches np.random.exponential(mean)).
  double exponential(double mean) {
    return -mean * std::log(1.0 - uniform_01_(rng_));
  }

 private:
  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_01_{0.0, 1.0};
  bool has_spare_ = false;
  double spare_ = 0.0;
};

}  // namespace nav
