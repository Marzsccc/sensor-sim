# Sensor Simulator

Generate realistic multi-sensor measurements for SLAM, state estimation, and sensor fusion algorithm development.

## Features

- **6-DoF Trajectory Generator**: Cubic Hermite splines + SLERP for smooth, physically plausible motion
- **IMU Sensor Model** (3 grades):
  - Consumer (BMI160 / ICM-20948)
  - Tactical (ADIS16490)
  - Navigation (Honeywell HG9900)
  - Error sources: bias, scale factor, misalignment, white noise, bias instability (random walk)
- **GNSS Sensor Model** (4 grades):
  - Consumer (phone L1)
  - Automotive (u-blox F9)
  - RTK (cm-level)
  - Survey (mm-level)
  - Error sources: white noise, multipath (Gauss-Markov), signal dropouts, configurable output rate
- **Latency & Time-Sync Model** (v0.4.0):
  - Per-sensor fixed + jittered transport/computation delay, packet loss, stale timestamps
  - Sensor clock drift (ppm rate error + random-walk wander)
  - Clock offset/skew estimation (linear regression, IEEE 1588-style) and effective-delay computation
  - Directly relevant to AR-HUD display-time delay compensation
- **Display-Time Pose Prediction** (v0.5.0):
  - `PosePredictor`: propagate a fused 6-DoF state to a future display time using
    IMU (attitude + gravity-compensated acceleration) and/or wheel odometry
    (planar nonholonomic model) -- the AR-HUD reprojection step
  - `DisplayPipeline`: end-to-end AR-HUD render-loop simulator comparing a naive
    renderer (latest fused pose) vs a compensated renderer (predict to display
    time); measures position/heading error at the display instant
  - Demo: 50 ms camera + 60 Hz display (~67 ms horizon) at 20 m/s -- naive
    renderer lags by ~1.3 m, compensated drops to ~0.1 m (-92%)
- **Display-Time Uncertainty Propagation** (v0.7.0):
  - `PredictorUncertainty`: propagate the ESKF 9×9 [p, v, θ] covariance
    through the display-time prediction -- initial-state uncertainty via
    the same error-state Jacobian F, plus IMU process noise (velocity/
    angular random walks, bias RWs) and a tunable constant-acceleration
    model-error term
  - `horizontal_ellipse()`: renders the horizontal position uncertainty
    as 1σ/95% ellipse parameters (axes, rotation, area, radius) for HUD
    marker fading/clamping
  - Demo: ESKF steady state (0.21 m) → 80 ms horizon → 95% radius
    0.52 m, ellipse area 0.14 m²
- **GNSS Outage & Uncertainty-Aware Display** (v0.8.0):
  - `OutageModel`: deterministic outage windows or random dropouts
  - `OutageSimulator`: end-to-end ESKF (IMU 200 Hz + wheel 100 Hz +
    GNSS 10 Hz) run under GNSS loss (tunnel / garage / urban canyon);
    per display tick computes mean display pose + full 15×15 covariance
    + horizontal 95% ellipse, then blends toward wheel-only dead
    reckoning as uncertainty grows (ellipse-weighted fade/clamp)
  - `consistency()`: NEES analysis validating the propagated covariance
    is consistent with true error, both in-outage and after recovery
  - Production traps surfaced: the 6σ gate silently drops *every* fix
    after a long outage (fixed with a re-acquisition reset), and
    absolute heading is unobservable on straight roads (drift model
    inflates the ellipse during outages)
- **ADAS Marker Projection & Display-Time Latency Compensation** (v0.9.0):
  - `MarkerProjector`: project world-anchored ADAS markers (ACC lead-vehicle
    box, FCW/AEB hazard, lane lines) into the HUD body frame (x-forward /
    right / down) and pinhole-project to screen pixels
  - `AdasPipeline`: end-to-end render loop that renders each marker with the
    naive fused ego pose vs the display-time predicted pose; reports the
    *marker reprojection error* (radial m + screen px) -- the quantity the
    driver actually sees
  - Uncertainty fade: propagates the 9×9 pose covariance through the
    display horizon and maps the 95% horizontal ellipse to a marker alpha
    in [0,1]; steady state stays fully opaque, cold-start / outage / big
    delay fade the marker down
  - Demo result at 20 m/s + 50 ms camera delay + 60 Hz display: naive
    marker lags ~1.33 m (>150 px off); compensated drops to ~0.10 m
    (−92%) and < ~50 px on lane markers
- **Near-Field Marker Visibility, FOV Culling & Occlusion** (v0.10.0):
  - `Frustum`: HUD field-of-view frustum (hfov/vfov, near clip, min/max
    range); classifies a marker as OK / NEAR_PLANE / TOO_CLOSE /
    OUT_OF_FOV / TOO_FAR / BEHIND
  - `angular_size`: marker on-screen size quoted in *angle* (deg) -- the
    finite, divergence-proof quantity. The v0.9.0 pinhole projector's pixel
    span explodes as forward range → 0, but angular size converges, which is
    what the eye actually resolves
  - `Occluder` + `overlap_frac`: bodywork / A-pillar / bonnet angular
    occupancy that fades markers the driver's own vehicle overlaps
  - `MarkerVisibilityPolicy.decide()`: per-marker-per-tick render decision
    FULL / FADED / CULLED with a fade alpha and machine-readable reason;
    culls BEHIND / TOO_CLOSE / OUT_OF_FOV markers *before* they reach the
    pinhole projector
  - Production trap fixed: near-field hazard pixel divergence (a close
    FCW/AEB hazard formerly shattered the render); the near-field region is
    now gated by angular oversize and near-plane fade instead of raw pixels
- **Hazard Assessment & Warning Arbitration** (v0.11.0):
  - `TTCModel`: time-to-collision from display-time range + closing rate
    (constant-speed FCW floor + constant-deceleration AEB model)
  - `HazardAssessment`: normalizes each marker into a threat score in [0,1]
    fused from TTC, marker-kind urgency and cross-track alignment
  - `WarningArbitrator` + `WarningLevel`: selects the single most
    threatening marker as the HUD headline warning (OFF/CAUTION/WARN/
    CRITICAL/EMERGENCY), gated by the v0.10 visibility decision
  - `ThreatPipeline`: shows uncompensated latency *under-warns* — a
    slow-closing hazard read CRITICAL under the naive pose but EMERGENCY
    under display-time compensation (the safety decision, not just pixels)
  - Demo: 25 m/s + 0.2 s delay → naive CRITICAL vs compensated EMERGENCY
- **LiDAR Point-Cloud Simulation** (v0.12.0):
  - `LidarSensor`: raycast a configurable LiDAR against an analytic world
    (ground plane, AABBs, spheres) from the host pose with a rigid body
    mount offset; nearest-hit analytic intersection, clamped to
    [range_min, range_max]
  - Two scan geometries: `spinning` (360° VLP-16 style) and `solid`
    (rectangular automotive forward FOV)
  - Gaussian range noise, per-ray dropout probability, reflectivity
    intensity, and dynamic objects moving at constant velocity (a
    cross-beam target leaves a smeared moving cluster)
  - Throws per-object point counts / range / intensity; demo
    `examples/lidar_demo.py`; 10 new unit tests (→ 110 total)
- **Camera Feature & Optical-Flow Simulation** (v0.13.0):
  - `CameraSensor`: forward-facing pinhole camera with a rigid body mount
    offset, projecting analytic 3D world features (any world point, with or
    without constant velocity) into pixels using the same world/body frame
    convention as the rest of the stack
  - Configurable intrinsics (fx/fy/cx/cy, resolution), optional radial
    distortion (k1), depth-range clipping (min_z/max_z), and image-bounds
    culling
  - `track()` computes per-feature optical flow between two host poses:
    previous/current pixel position, pixel displacement, current depth and
    pixel velocity (px/s) -- the measurement a VO / feature-tracker consumes
  - Realistic measurement effects: Gaussian pixel noise and per-feature
    detection dropout (contrast / motion blur / far range)
  - Dynamic features move at constant velocity, so a crossing pedestrian
    produces a distinct motion field from the static background expansion
  - Demo `examples/camera_demo.py` (saves flow-field plot); 14 new unit
    tests (→ 124 total)
- **Unified Ground-Truth Evaluation & Regression Gates** (v0.14.0):
  - `TrajectoryEvaluator`: stream per-tick pose/velocity/yaw errors from
    ANY estimator (ESKF, dead reckoning, a vendor black box) into one
    report; position NEES consistency comes free when you pass the
    filter covariance block
  - `EvalReport`: RMSE / mean / p95 / max per quantity + chi-square NEES
    verdict (`nees_hpos`, `nees_pos3d`) + pass/fail gates via
    `add_gate(report, "pos_err_m", "<", 2.0)`
  - Gate syntax supports any statistic (`"pos_err_m@max"`) and NEES
    fields (`"nees:nees_hpos@nees_mean"`); missing metrics fail closed
  - `compare_reports()`: side-by-side RMSE table for estimator sweeps
  - Demo `examples/evaluation_demo.py`: ESKF vs pure dead-reckoning on
    the 60 s S-curve -- fused RMSE 1.98 m vs DR 14.8 m, and the evaluator
    flags the automotive-GNSS ESKF covariance as over-confident
    (NEES 34 vs CI [1.95, 2.05]), matching the v0.8.0 finding; 11 new
    unit tests (→ 135 total)
- **Scenario Orchestration & Batch Regression** (v0.15.0):
  - `Scenario`: declarative experiment = trajectory + sensor grades +
    GNSS outages + init error + regression gates, fully reproducible by
    seed
  - Scenario library: highway-straight (baseline), urban-s-curve,
    tunnel-outage (12 s blackout), parking-garage (consumer HW + 20 s
    outage -- the known-weak corner), rtk-baseline (cm-level golden ref)
  - `run_scenario()` pushes any scenario through the standard ESKF
    pipeline and scores it with the v0.14 evaluator; `batch_summary()`
    renders the pass/fail matrix for overnight sweeps
  - `eskf_config_for()`: filter presets matched to sensor grades under
    test
  - Measured baselines: RTK 0.26 m vs automotive 2.0 m pos RMSE;
    consumer+outage corner quantified at 22 m / 4.2 m/s (physics of
    cheap hardware, gates encode "no worse than this")
  - Demo `examples/scenarios_demo.py` (5/5 PASS); 11 new unit tests
    (→ 146 total)
- **C++/Eigen Port (v0.16.0)**:
  - Header-only C++17 library `cpp/include/nav/`: quaternion utils,
    Allan/IEEE-952 IMU error model, Gauss-Markov multipath + dropout GNSS,
    15-dim ESKF (Joseph form, 6-sigma gate), Hermite+slerp trajectory
  - Same 60 s GNSS+IMU loose-coupling demo scenario as `eskf_demo.py`;
    statistical parity: ESKF pos RMS 1.8-3.9 m across seeds vs Python
    2.3 m (streams intentionally not bit-identical; see cpp/README.md)
  - 7 unit tests incl. static-IMU error-chain mean, GNSS rate+RMS band,
    ESKF bias observability, outlier-gate rejection (`cpp/build/test_nav`)
- **Monte-Carlo Regression Gates** (v0.15.1):
  - Audit found single-seed gates overfit: parking-garage passed on
    seed 42 but failed 6/10 other seeds -- consumer-IMU drift during a
    20 s outage scales with the LUCK of the drawn acc bias
    (physics ~98 m 1-sigma; measured range 3.9-192 m across seeds)
  - `MonteCarloGate` + `run_scenario_mc()`: gate the ACROSS-SEED
    distribution (e.g. p90 < threshold) instead of one run; missing
    data fails closed
  - Sensor-model audit: static IMU/GNSS error stats verified against
    spec sheets (acc noise 2.1 mg vs 2.0 theoretical, ARW 0.0846 vs
    0.0833 deg/sqrt(s), automotive GNSS RMS 3.0 m vs sqrt(1^2+2^2)*sqrt2)
  - MC gates also exposed that the automotive-GNSS NEES inconsistency
    is seed-stable (~34 on every seed) -> correlated multipath treated
    as white by the filter, documented as known issue for v0.16
  - scenarios_demo now: stable scenarios single-run, stochastic corners
    MC-gated (`MC-PASS` in matrix); 13 tests in test_scenarios.py
- **Automotive Radar Detection** (v0.17.0):
  - `RadarSensor`: forward-looking 77 GHz radar on a rigid body mount;
    analytic one-detection-per-object model over the same world primitives
    as LiDAR/camera, gated by azimuth/elevation FOV (+-60 / +-10 deg) and
    a [range_min, range_max] gate
  - Spherical measurements: range to the *near surface* (same raycast as
    LiDAR -- an extended lead vehicle reads closer than its centre),
    azimuth/elevation to the object centre, and Doppler range-rate with
    the ACC convention (positive = closing) from host + target velocity
  - Radar equation: SNR = snr_ref * (rcs/rcs_ref) * (R_ref/R)^4; per-object
    RCS (explicit `rcs_sqm` or mapped from reflectivity); detection is a
    Bernoulli draw P_d = SNR/(SNR+thresh), so far / dim targets drop out
  - Per-axis Gaussian noise (range / angle / range-rate) + `RadarFrame`
    truth channels (range_true / range_rate_true) for validation
  - Demo `examples/radar_demo.py`: closing lead (Doppler +10 m/s) vs a
    crossing pedestrian (azimuth sweeps ~28 deg while range-rate moves
    <4 m/s -- the radar blind spot that camera fusion exists for);
    20 new unit tests (-> 168 total)
- **Radar Detection Tracking** (v0.18.0):
  - `RadarTracker`: constant-velocity EKF over `RadarFrame` detections --
    the layer between "sparse scans" and an ACC / FCW / AEB / RCTA target
    list.  State is the horizontal-plane position + *absolute* velocity of
    the radar reflection point in the **world frame** (CV predict is exact
    for any constant-velocity target; the host pose only enters the
    measurement equation), measured through a spherical EKF model
    `h(x) = [range, azimuth, range-rate]` with an analytic Jacobian
  - Recovering the v0.17 blind spot: single-frame Doppler cannot see
    cross-range motion, but temporal fusion of the azimuth history can -- a
    crossing pedestrian (true |range-rate| < 0.4 m/s) gets its lateral
    velocity estimated to ~2 m/s within ~0.5 s
  - Nearest-neighbour association with a Mahalanobis gate
    (chi-square 3-dof 99%); **two-class track management**: confirmed tracks
    claim detections first, tentative tracks take leftovers and die on their
    first miss -- stopping the classic "newborn-with-wide-prior steals the
    detection" death spiral of pure greedy NN; births from unassociated
    detections use a Doppler-derived radial velocity prior with a
    deliberately large cross-range prior (a single scan cannot know lateral
    speed); confirmed tracks coast up to `coast_max` scans then are deleted;
    re-acquired targets get a fresh track id
  - Demo `examples/radar_tracking_demo.py`: closing lead (Doppler +20 m/s)
    vs a crossing pedestrian (Doppler ~0) on a stationary host -- the
    crossing target's lateral velocity converges to truth while the raw
    range-rate stays blind; 17 new unit tests (-> 185 total)
- **Radar + Camera Fusion Tracking** (v0.19.0):
  - `FusedTracker`: a single constant-velocity EKF that fuses radar
    detections (`[range, azimuth, range-rate]`) and camera azimuths
    (pixel columns converted through the camera intrinsics) onto shared
    4-state tracks -- the production ADAS perception-stack pattern: the
    radar provides stable range/Doppler, the camera provides a per-scan
    azimuthal measurement that directly observes cross-range motion, the
    axis the radar alone is blind to in a single frame
  - Two sequential EKF updates per tick (radar then camera), each with its
    own Mahalanobis gate; camera alone cannot birth (no range) but refines
    lateral kinematics -- validated at 2 s integration: pedestrian lateral
    velocity error 0.03 m/s vs 0.14 m/s radar-only (~5x closer)
  - Graceful degradation without mode switches: camera dropout (night /
    glare) degrades to radar-only, radar dropout keeps the track alive via
    camera -- the covariance honestly reflects whichever measurement
    arrived
  - Production trap caught & fixed during validation: reusing the pre-radar
    azimuth Jacobian for the post-radar camera update over-corrected the
    state (lateral velocity blew up to ~6 m/s); the camera update now
    recomputes its prediction from the current state -- standard
    multi-sensor EKF discipline, documented in the module docstring
  - Demo `examples/fused_demo.py`: fused vy converges within ~0.3 s and
    stays within ±0.2 m/s while radar-only oscillates for the first second;
    when the pedestrian crosses abeam (t ~ 2.3 s) radar-only loses the
    lateral velocity entirely while the fused track keeps it -- the
    camera matters exactly where the radar goes blind; 10 new unit tests
    (-> 195 total)
- **Multi-Object Tracking Metrics** (v0.21.0):
  - `mot.py` answers "how good is the track set?" -- self-contained (no
    radar/tracker import) so it can grade any tracker, including a vendor
    black box
  - Set-distance metrics: `ospa()` (Schuhmacher-Vo-Vo, cutoff-bounded,
    ordering-invariant) and `gospa()` (2-alpha, decomposed into
    localisation / missed / false with `total^p = loc^p + missed^p + false^p`)
  - Identity metrics: `assign()` (global Hungarian or greedy, gated) and
    `MotAccumulator` streaming `mota / motp / FP / FN / id_switches /
    mean_ospa / mean_gospa` plus mostly-tracked / mostly-lost
  - `RadarFrame` gains a full ground-truth channel (`truth_ids`,
    `truth_points`) recorded *before* the detection draw, so a tracker's
    missed targets (false negatives) become observable; `truth_world_points()`
    feeds it straight into the metrics
  - Demo `examples/mot_demo.py`: on one scene the naive "raw detections as
    tracks" front-end scores MOTA 0.91 / OSPA 1.02 m (29 FN) while the EKF
    tracker scores MOTA 0.99 / OSPA 0.25 m -- the lesson being that a single
    MOTA number rewards raw detections, so localisation-aware metrics ship
    alongside; 38 new unit tests (-> 252 total)
- **Delayed / Out-of-Sequence Measurement Fusion** (v0.22.0):
  - `delay_fusion.py` fixes the *second* half of the latency problem.  The
    v0.4-v0.9 stack assumes the fused state is honestly stamped at its
    *measurement* time; a naive pipeline that fuses on *arrival* stamps the
    state with the receive clock while describing a scene that is already
    one sensor latency old -- a bias that no amount of forward prediction
    can undo
  - `DelayedFusionFilter` keeps a bounded state/measurement history: an
    out-of-order sample is inserted into the past (rewind to the nearest
    stored state, apply, replay every later measurement forward).  For a
    linear time-invariant model this reproduces the in-order estimate
    **exactly** (asserted at `atol=1e-9`, including a fully reversed stream)
  - `mode="receive_time"` is kept as the naive baseline so a pipeline can be
    A/B measured instead of argued about
  - Clock semantics pinned by tests: `t` is the newest *fused validity* time,
    so a delayed-but-monotone stream never rewinds (its residual lag is the
    display-time bridge), a late sample restores rather than drags the clock,
    and only genuine multi-rate interleaving triggers reprocessing
  - Demo `examples/delay_fusion_demo.py` (odom 100 Hz/5 ms, radar 20 Hz/30 ms,
    camera 10 Hz/60 ms, 1040 samples): streaming RMSE 0.440 -> 0.273 m,
    display-time RMSE 0.480 -> 0.339 m, and position NEES 6.79 -> 1.88
    against the 2-dof expectation -- the naive filter is *over-confident by
    3.4x* about a biased state, which is exactly the failure that fades a
    HUD marker in when it should fade out; 32 new unit tests (-> 284 total)
- **Nonlinear (ESKF) Delayed-Measurement Fusion** (v0.23.0):
  - `nonlinear_delay.py` takes the v0.22.0 idea to the inertial estimator.
    Linear rewind-and-replay worked because updates compose as matrix
    products; an ESKF's mechanisation does not.  The fix is to retain the
    *inputs* as well as the states: snapshots alone cannot rewind a nonlinear
    filter, the raw IMU samples can
  - `RewindReplayESKF` keeps bounded snapshot / IMU / GNSS buffers; a late fix
    anchors at the newest snapshot `<= t_meas`, then replays the identical IMU
    samples interleaved with the pending fixes in validity order.  Because the
    replay is deterministic and restarts from the same anchor every time, the
    result is **exact to `atol=1e-9`** versus a plain `ESKF` fed in order --
    verified against a scrambled delivery order and a two-sensor interleaving
    that forces 100+ rewinds.  Only the covariance linearisation stays
    approximate, as it always was
  - Clock semantics corrected for a nonlinear estimator: with an IMU driving
    the clock, even a *delayed-but-monotone* aiding stream is out of sequence
    (the clock races ahead of the fix), so rewinds are the normal case rather
    than the exception.  A rewind still never drags the clock back -- `t` is
    restored to exactly its pre-rewind value
  - `extrapolate_to()` now rejects a non-finite display instant instead of
    returning a NaN pose via `0 * inf` in the velocity term
  - Demo `examples/nonlinear_delay_demo.py` (25 m/s, 100 Hz IMU, 10 Hz GNSS,
    200 ms latency, 1500 IMU samples): streaming RMSE 4.904 -> 0.719 m,
    display-time RMSE 4.916 -> 0.783 m, position NEES 121.40 -> 1.95 against
    the 3-dof expectation.  The naive error lands on the predicted delay bias
    `v x delay = 25 x 0.2 = 5.01 m`, and forward extrapolation does *not*
    rescue it (4.904 -> 4.916 m) -- latency baked in as bias cannot be
    predicted away; 35 new unit tests (-> 319 total)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from sensor_sim.trajectory import Trajectory, Waypoint
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.gnss import GNSSSensor, GNSSGrade

# Build an S-curve trajectory
waypoints = [
    Waypoint(t=0, pos=[0,0,0], vel=[25,0,0], att=[1,0,0,0], omega=[0,0,0]),
    Waypoint(t=5, pos=[..., ...], ...),
    ...
]
traj = Trajectory(waypoints, dt=0.01)

# Create sensors
imu = IMUSensor(SensorGrade.CONSUMER, dt=0.01, seed=42)
gnss = GNSSSensor(GNSSGrade.AUTOMOTIVE, dt=0.01, seed=123)

# Simulate
for pt in traj.points:
    acc_meas, gyr_meas = imu.measure(traj.body_accel(pt), pt.omega)
    pos_meas, vel_meas, valid = gnss.measure(pt.pos, pt.vel)
    # Feed into your EKF / ESKF / SLAM pipeline
```

## Run Example

```bash
python3 examples/s_curve_car.py
```

Generates an S-curve trajectory with consumer IMU + automotive GNSS, and plots the results.

## Delay Compensation

Three demos cover the latency problem end to end:

| Demo | Question it answers |
|---|---|
| `examples/latency_demo.py` | how much latency is in the pipeline? |
| `examples/delay_fusion_demo.py` (v0.22) | what does fusing on *arrival* cost a linear filter? |
| `examples/nonlinear_delay_demo.py` (v0.23) | same question for a GNSS+IMU ESKF |

```bash
PYTHONPATH=. uv run --with numpy --with matplotlib python examples/nonlinear_delay_demo.py
```

On a 25 m/s track with 200 ms GNSS latency the naive receive-time pipeline
sits ~5 m off (exactly `v x delay`) with a position NEES of 121 (3 dof
expected), while rewind-and-replay holds 0.72 m and NEES 1.95 -- and forward
extrapolation does not rescue the naive one.  See
`docs/v0.23.0-nonlinear-delay-validation.md`.

## Allan Variance Analysis (IMU Noise Characterization)

`sensor_sim/allan.py` implements the classic **overlapping Allan variance**
(IEEE Std 952-1997 style) for characterizing IMU noise from a static run:

```python
import numpy as np
from sensor_sim.imu import IMUSensor, SensorGrade
from sensor_sim.allan import overlapping_allan_deviation, extract_noise_parameters

imu = IMUSensor(SensorGrade.TACTICAL, dt=0.01, seed=7)
# record ~1 h of static gyro data, then:
taus, adev = overlapping_allan_deviation(gyro_x, fs=100.0)  # tau vs Allan dev
params = extract_noise_parameters(taus, adev)
print(params.summary(unit="rad/s"))
```

The log-log Allan curve has characteristic slopes that identify each noise term:

| Slope | Noise source | Extracted coefficient |
|-------|-------------|----------------------|
| -1    | Quantization | `Q = adev·τ/√3` |
| -1/2  | White noise (ARW/VRW) | `N = adev·√τ` |
| 0     | Bias instability | `B = min(adev)/0.664` |
| +1/2  | Rate random walk (RRW) | `K = adev·√(3/τ)` |

Units follow the input data (rad/s for a gyro, m/s² for an accelerometer).
Convert to datasheet units e.g. ARW: `N·180/π·√3600` → deg/√h.

Run the demo (extracts ARW/BI/RRW from a tactical-grade gyro and compares
against the model settings):

```bash
python3 examples/allan_example.py
```

## Run Tests

```bash
PYTHONPATH=. uv run --with pytest --with numpy python -m pytest tests/ -q
```

319 tests, ~50 s.

## Sensor Grades

| Grade | Acc Bias | Acc Noise | Gyr Bias | Gyr Noise | GNSS H-pos σ |
|-------|---------|-----------|---------|-----------|-------------|
| Consumer | 50 mg | 200 µg/√Hz | 10°/h | 0.5°/√h | 3.0 m |
| Tactical | 1 mg | 50 µg/√Hz | 1°/h | 0.05°/√h | — |
| Automotive | — | — | — | — | 1.0 m |
| RTK | — | — | — | — | 0.02 m |
| Navigation | 25 µg | 5 µg/√Hz | 0.001°/h | 0.001°/√h | — |
| Survey | — | — | — | — | 0.005 m |

## Project Structure

```
sensor-sim/
├── sensor_sim/
│   ├── __init__.py      # Package entry
│   ├── trajectory.py    # 6-DoF trajectory generation
│   ├── imu.py           # IMU sensor model
│   ├── allan.py         # Overlapping Allan variance + noise extraction
│   ├── gnss.py          # GNSS sensor model
│   ├── wheel.py         # Wheel odometry model
│   ├── latency.py       # Latency / clock sync / effective delay
│   ├── predict.py       # Display-time pose prediction & AR-HUD pipeline
│   ├── visibility.py    # FOV culling, angular size, occlusion (v0.10)
│   ├── hazard.py        # TTC, threat score, warning arbitration (v0.11)
│   ├── lidar.py         # LiDAR point-cloud simulation (v0.12)
│   ├── tracking.py      # Radar CV-EKF tracker (v0.18)
│   ├── fused.py         # Radar+camera fusion tracker (v0.19)
│   ├── track_warn/      # Track->marker bridging & fusion warning (v0.20)
│   ├── mot.py           # Multi-object tracking metrics: OSPA/GOSPA/MOTA (v0.21)
│   ├── delay_fusion.py  # Delayed / out-of-sequence measurement fusion (v0.22)
│   ├── nonlinear_delay.py  # ESKF rewind + replay for delayed aiding (v0.23)
│   ├── camera.py        # Camera feature & optical-flow simulation (v0.13)
│   └── utils.py         # Quaternion/rotation utilities
├── examples/
│   ├── s_curve_car.py   # Car S-curve demo
│   ├── hazard_demo.py   # Hazard assessment & warning arbitration demo
│   ├── lidar_demo.py    # LiDAR point-cloud scan demo
│   ├── camera_demo.py   # Camera feature & optical-flow demo
│   ├── allan_example.py # Allan variance demo (gyro noise characterization)
│   ├── latency_demo.py  # Latency & time-sync demo
│   ├── mot_demo.py      # MOT metrics (OSPA/GOSPA/MOTA) demo (v0.21)
│   ├── delay_fusion_demo.py  # Delayed / out-of-sequence fusion demo (v0.22)
│   ├── nonlinear_delay_demo.py  # ESKF rewind/replay fusion demo (v0.23)
│   └── predict_demo.py  # AR-HUD display-time prediction demo
├── tests/
│   ├── test_basic.py    # Unit tests
│   ├── test_allan.py    # Allan variance tests
│   ├── test_latency.py  # Latency / time-sync tests
│   ├── test_predict.py  # Display-time prediction tests
│   └── test_camera.py   # Camera feature & optical-flow tests
└── requirements.txt
```

## License

MIT

## v0.2.0 新增

- **轮速计模型** (`wheel.py`)：Ackermann（速度+横摆角速度）或差速（左右轮速）输出，建模轮半径误差、编码器量化、相关性滑动（Gauss-Markov）、白噪声；三档（消费/车载/精密）
- **非完整约束恒速轨迹**：`Trajectory(..., const_speed=25.0)` 保持速度模恒定，且姿态 yaw 自动跟随速度方向（解决 Hermite 样条转弯掉速 + 车体侧滑不一致问题）
- **多传感器示例**：`examples/multi_sensor_car.py` 一键生成 IMU+GNSS+轮速计同步数据集（npz 格式）
- 14 个单元测试（10 → 14）

- **Allan 方差分析工具** (`allan.py`)：经典 overlapping Allan variance，输入静态采样序列+采样率，输出 log-log 曲线数据 (τ, σ(τ))，按曲线特征斜率自动提取量化噪声、白噪声（ARW/VRW）、零偏不稳定性、随机游走（RRW）；示例 `examples/allan_example.py` 对战术级陀螺 1h 静态数据提取噪声系数并与模型设定对比（ARW 误差 <1%）；单元测试 10 → 24
- **IMU 陀螺白噪声单位修正**：`gyr_noise_deg_h_hz` 实际为 deg/√h（ARW 系数），rad/s√s 换算由 ÷3600 修正为 ÷60

## v0.5.0 新增

- **显示时刻位姿预测** (`predict.py`)：AR-HUD 延时补偿闭环核心
  - `PosePredictor`：从最新融合状态（位置/速度/姿态）向显示时刻传播——IMU 积分（姿态一阶精确积分 + 重力补偿加速度）+ 轮速计平面非完整模型，短时域（20-80ms）预测比全 15 维 ESKF 更稳健
  - `DisplayPipeline`：端到端 AR-HUD 渲染循环仿真——传感器延时（camera 50ms / GNSS 200ms / wheel 5ms）→ 融合状态滞后 → naive vs 补偿渲染对比，输出显示时刻位置/航向误差
  - 关键语义：AR 内容锚定在主传感器（camera）采集时刻，预测 horizon = camera 延时 + 1 帧显示延时（~67ms）
  - 效果：20 m/s 直线 1.33m → 0.10m（-92.5%）；15 m/s 转弯 0.98m → 0.08m（-91.4%）
  - 单元测试 24 → 32

## v0.7.0 新增

- **显示时刻不确定度传播** (`predict.py` 新增 `PredictorUncertainty`)：AR-HUD 渲染不确定度椭圆
  - 显示时刻协方差 = 初始状态不确定度（ESKF 9×9 [p,v,θ] 协方差块，沿预测雅可比 F 传播）+ 过程噪声（IMU 速度/角速度随机游走、零偏随机游走）+ 常加速度假设的模型误差项（默认 0.5 m/s²，随 horizon 增长，高速转向场景主导）
  - `horizontal_ellipse()`：水平位置协方差 → 1σ/95% 椭圆参数（长短轴、旋转角、面积、95% 等效半径），直接供 HUD 淡出/限幅 AR 标记
  - 闭环：ESKF 稳态协方差 0.21m → 80ms 预测后 95% 半径 0.52m（初始不确定度主导；纯 IMU 噪声 80ms 仅 ~0.1mm）
  - 示例 `examples/uncertainty_demo.py`；单元测试 46 → 58

## v0.8.0 新增

- **GNSS 中断仿真与不确定度感知显示** (`outage.py`)：隧道/地下车库/城市峡谷场景
  - `OutageModel`：确定性中断窗口或随机失锁，支持丢星率/恢复暖机时间
  - `OutageSimulator`：端到端 ESKF（IMU 200Hz + 轮速 100Hz + GNSS 10Hz）在中断下的全流程——每显示 tick 输出显示时刻均值位姿 + 15×15 全协方差 + 水平 95% 椭圆；`use_ellipse_output=True` 时按椭圆权重向纯轮速推算位姿混合（HUD 标记淡出/限幅的简单模型）
  - **生产陷阱 1：6σ 门限阻断恢复**——长时间中断后真实误差已远超过度自信的协方差，默认 `recovery="reset"`（首次修复后重捕获重置：位置/速度用 fix 重播种 + 协方差膨胀到初始水平），否则滤波器永远无法重新收敛
  - **生产陷阱 2：直线道路绝对航向不可观**——轮速只观测速度模与横摆角速度，中断期间航向误差无法修正；仿真器在中断窗口用漂移模型（航向 std 增长 + 横向积分）膨胀协方差，让椭圆诚实反映增长的不确定度
  - `consistency()`：NEES 一致性分析——验证传播协方差与真实误差一致，覆盖正常运行与中断恢复窗口
  - `Trajectory.at()` 最近点采样 + ESKF `gate_multiplier` 重捕获支持；示例 `examples/outage_demo.py`；单元测试 +12（→ 全量 19 通过）

## v0.9.0 新增

- **ADAS 标记投影与显示时刻延时补偿** (`adas.py`)：把世界系 ADAS 标记投影到 AR-HUD 本体坐标系
  - `MarkerProjector`：`to_body()` 世界→本体（x 前 / y 右 / z 下）+ `screen_xy()` 针孔投影到屏幕像素（深度取前向 +x，勿用竖直轴）
  - 标记工具：`lead_vehicle_marker()`（ACC 目标 8 角包围盒）、`hazard_marker()`（FCW/AEB 警戒楔角/最大距离）、`lane_line_marker()`（左/右/中车道线）
  - `AdasPipeline`：端到端渲染循环——naive（lag 融合位姿）vs 补偿（显示时刻预测位姿）投影每个标记，报告**标记重投影误差**（径向米 + 屏幕像素），即驾驶员真实感知的偏差
  - 不确定度淡出：传播 9×9 [p,v,θ] 协方差 → 95% 水平椭圆 → 标记 alpha∈[0,1]；稳态保持不透明，冷启动/中断/大延时自动淡出；`sigma_scale` 调淡出强度
  - **生产陷阱 1：近距危险目标像素发散**——range→0 时透视投影数值不稳定，HUD 应限幅屏幕位移并依赖 range/告警语义，用径向米误差而非像素评估保真度
  - **生产陷阱 2：前向深度约定**——本体系 z-down 时深度必须取前向 +x，首个版本误用竖直轴导致全 NaN 像素
  - **生产陷阱 3：淡出 vs 信任**——纯靠位姿协方差淡出会在大中断时隐藏真实危险，FCW/AEB 需要最小不透明度下限，确保警戒楔始终可见
  - 效果：20 m/s 直线 + 50ms camera 延时 + 60Hz 显示——naive 标记滞后 ~1.33m（车道线 >190px 偏移）→ 补偿 ~0.10m（<52px），-92%
  - 单元测试 +7（→ 全量 79 通过）；验证文档 `docs/v0.9.0-adas-validation.md`

## v0.12.0 新增

- **LiDAR 点云仿真**（`lidar.py`）：raycasting + 噪声 + 动态物体，填补传感层最后一块拼图
  - 复用轨迹位姿 + 刚体安装：`LidarSensor` 把 `mount_t_body` 安装到车身，按 `_quat_to_rotmat(att)` 世界→本体、其转置 本体→世界 组成传感器位姿（与 `MarkerProjector` 同体基调约）
  - 解析光线求交（无网格）：相机射线对 地面 `GroundPlane` / 轴对齐盒 `Box` / 球 `Sphere` 求最近交点，再按 `[range_min, range_max]` 截断；确定、可手验、快速
  - 两种扫描几何：`spinning`（VLP-16 式全 360° 水平扫描，固定俯仰束）与 `solid`（汽车前置式矩形 az×el 视场角）
  - 真实噪声与丢点：高斯测距噪声（分档可配）、每点丢弃概率（低反射/远距）、`reflectivity` 强度字段供下游按亮度过滤、动态物体匀速运动（跨束目标在点云中留下移动拖影簇）
  - 坐标约定与 trajectory 一致：世界 z-up、车体 +x 前/+y 左/+z 上；输出点在内体坐标（+x 前）
  - 效果：16 束 spinning 场景——地面 10923 点、障碍物/盒/柱/动态车按 object_id 区分并带 min/mean intensity；示例 `examples/lidar_demo.py`；单元测试 +10（→ 全量 110 通过）

## v0.13.0 新增

- **相机特征与光流仿真**（`camera.py`）：把视觉传感层接入传感器栈，直接对标 monocular VO / VIO / AR-HUD
  - 前向针孔相机 + 刚体安装：`CameraSensor` 复用 `_quat_to_rotmat(att)` 与 `mount_t_body` 组成传感器位姿（与 `MarkerProjector`/`LidarSensor` 同体基调约、世界 z-up、车体 +x 前/+y 左/+z 上），世界特征经相机系投影到像素
  - 可配内参 fx/fy/cx/cy、分辨率、径向畸变 k1、深度范围 [min_z, max_z]、图像边界裁剪
  - `track()` 双帧跨帧关联：按稳定 feature_id 匹配同一世界点在两帧的像素位置 → 像素位移（光流）+ 当前深度 + 像素速度 px/s，即 VO/光流跟踪器消费的测量
  - 真实测量效应：高斯像素噪声 + 每特征漏检概率（低对比/运动模糊/远距）
  - 动态特征匀速运动：横穿行人的运动场与静止背景的膨胀场（FOE 外扩）可明显区分
  - 效果：前向直行 10 m/s——近处特征流速快（depth 9.5m→8.7px/帧）远处慢（29.5m→1.7px/帧），左/右分别左/右流（FOE 外扩），行人特征左侧强流；示例 `examples/camera_demo.py`（保存光流场图）；单元测试 +14（→ 全量 124 通过）

## v0.21.0 新增

- **多目标跟踪评测指标（OSPA / GOSPA / CLEAR-MOT）**（`mot.py`）：回答 v0.17–v0.20 一直没回答的问题——「这套航迹到底好不好？」。自包含（不依赖 radar/tracker），可评任意跟踪器（含厂商黑盒）。
  - **雷达全量真值通道**：`RadarFrame` 新增 `truth_ids` / `truth_points`（**FOV+距离门内所有物体**，无论是否被检出）与 `azimuths_true`/`elevations_true`（逐检出的真方位）。真值行在 Bernoulli 检出抽样**之前**记录，因此漏检目标对评测可见——此前跟踪器的 FN 根本无法观测。新增 `truth_world_points(host_pos, att, mount_t_body)` 直接产出世界系 `(M,2)` 真值集。
  - **集合距离指标**：`ospa()`（Schuhmacher–Vo–Vo，截断 c 抑制远端垃圾，对排序不变，集合相同时为 0）、`gospa()`（2-α 变体，返回 `{total, localisation, missed, false}`，满足 `total^p = loc^p + missed^p + false^p`）。
  - **身份指标**：`assign()`（默认全局 Hungarian，可选贪心；gate 禁止远配对）+ `MotAccumulator` 流式累计 `mota / motp / FP / FN / id_switches / mean_ospa / mean_gospa`（含 GOSPA 分解）与 `mostly_tracked / partially_tracked / mostly_lost`。
  - **约定（已测试）**：`MOTA = 1 - (FP+FN+IDSW)/N_gt`，`N_gt=0` 且无错误时为 1.0、否则为 `nan`（不白送分）；MOTP 为匹配对平均欧氏距离（米）；GOSPA 每个未匹配对象罚 `c^p/α`。
  - **踩坑**：Hungarian 版若把「忽略配对」代价设为 0，会退化成「全部不匹配」（测试抓到，改为 gate 内配对恒优于丢弃、gate 外恒被丢弃）。
  - 效果：demo `examples/mot_demo.py` 同场景双前端对比——naive（原始检出当航迹）MOTA 0.910 / OSPA 1.018 / 29 个 FN；EKF 跟踪器 MOTA 0.994 / OSPA 0.252 / 1 个 FN（出生瞬态 FP）。即「单看 MOTA 会奖励噪声检出」，故必须同时报告定位类指标。图 `examples/mot_demo.png`。
  - 单元测试 +38（→ 全量 **252 通过**）；验证文档 `docs/v0.21.0-mot-metrics-validation.md`。

## v0.22.0 新增

- **延迟 / 乱序测量融合（OOSM）**（`delay_fusion.py`）：补上延迟问题的**另一半**。v0.4–v0.9 的整条链都默认「融合状态是按**测量时刻**打戳的」——而朴素管线按**到达时刻**融合，等于用接收时钟给一个「已经老了一个传感器延时」的场景打戳，这是**偏差**而非延迟，再多的前向预测也补不回来（AR-HUD「标记拖尾」的第二种、更隐蔽的形态）。
  - **`DelayedMeasurement`**：同时携带 `t_meas`（有效时刻）与 `t_recv`（到达时刻），拒绝 `t_recv < t_meas`。
  - **`CVModel`**：时不变匀速模型（任意位置维数 `n`，状态 `[p,v]` 共 `2n`），`F(dt)`/`Q(dt)`（连续白加速度、逐轴 PSD `q`）+ `observe_position(sigma)`；`F_fn`/`Q_fn` 可换动力学而不动滤波器。
  - **`DelayedFusionFilter`**：保留有界状态/测量历史；乱序样本**插回过去**——回卷到 `t_meas` 之前最近的快照，施加该测量，再把之后所有测量**重放**一遍。对**线性时不变**系统，结果与顺序滤波**逐位相等**（`atol=1e-9`，含完全逆序流）。
  - **时钟语义（有测试钉死）**：`t` = **最新已融合的有效时刻**，非墙钟。① 仅「延迟但单调」的流**永不回卷**（残余延迟由显示时刻 `predict_to_time` 桥接）；② 回卷后时钟**恢复**、不被拖回；③ 只有多速率真正交错才触发重放。
  - **踩坑 ①（回归测试）**：初版 `_reprocess()` 从锚点起重建快照列表，**悄悄截断了更早的历史**——连续乱序若干分钟后历史塌缩成两三个快照，下一次回卷直接 `RuntimeError`（典型「跑几分钟才炸」的现场故障）。改为「保留严格更早的前缀、只重建后缀」。
  - **踩坑 ②**：「均匀延迟」的流在测量时刻语义下**根本不回卷**——差点写出一个「证明了重放正确但一次回卷都没跑」的假测试。改为交错「快速在序传感器 + 永久滞后 100 ms 传感器」才真正逼出重放。
  - 效果：demo `examples/delay_fusion_demo.py`（odom 100 Hz/5 ms、radar 20 Hz/30 ms、camera 10 Hz/60 ms，共 1040 样本，两条管线在**同一墙钟显示时刻**评分）——流式 RMSE **0.440 → 0.273 m**、显示时刻 RMSE **0.480 → 0.339 m**、位置 NEES **6.79 → 1.88**（2 自由度期望=2，即朴素滤波**过自信 3.4×**：它对自己的偏差状态「很确定」，正是 HUD 该淡出标记却淡入的那种故障）。图 `examples/delay_fusion_demo.png`。
  - 单元测试 +32（→ 全量 **284 通过**）；验证文档 `docs/v0.22.0-delayed-fusion-validation.md`。

## v0.23.0 新增

- **非线性（ESKF）延迟测量融合：回卷 + 重放**（`nonlinear_delay.py`）：把 v0.22.0 的思路推到惯性估计器上。线性回卷-重放之所以成立，是因为更新是矩阵乘法、可结合；ESKF 的力学编排不是。**关键洞察：只留状态历史救不了非线性滤波器，必须把输入也留下** —— 同一批 IMU 样本按同一顺序重放，必然复现同一串状态（这是确定性，不是近似）。
  - `RewindReplayESKF` 维护三个有界缓冲（快照 / IMU / GNSS）。迟到定位的处理：锚定到 `t <= t_meas` 的最近快照 → 按**有效时刻**把「重放的 IMU + 待融合定位」归并执行 → `t` 精确恢复到回卷前的值。
  - **精确性**：与「一路按有效时刻喂进去」的朴素 `ESKF` 逐位相等（`atol=1e-9`），并用**打乱的到达顺序**与**双传感器交错**（逼出 100+ 次回卷）两条路径验证。只有协方差线性化是近似的 —— 它本来也是。
  - **时钟语义修正（值得记）**：v0.22.0 说「延迟但单调的流永不回卷」，那是因为当时滤波器时钟**就是**最新测量时刻。一旦有 IMU 在推进时钟，时钟永远跑在延迟定位之前 → **每个延迟定位都是乱序**，回卷成为常态而非例外。这不是回归，是正确行为（当初按 v0.22.0 断言写的测试就是被这条推翻后重写的）。教训：**时钟语义属于整个估计器，不属于融合技巧**。回卷依然绝不把时钟拖回。
  - `extrapolate_to()` 拒绝非有限的显示时刻。这不是形式主义：调用方算 `t_recv + 0.06` 而 `t_recv = inf`（「还有数据吗？」）时，速度项里的 `0 * inf` 会**静默产出 NaN 位姿**并一路流进渲染器。
  - 效果：demo `examples/nonlinear_delay_demo.py`（25 m/s、100 Hz IMU、10 Hz GNSS 延迟 200 ms，1500 个 IMU 样本，两条管线在**同一墙钟显示时刻**评分）——流式 RMSE **4.904 → 0.719 m**、显示时刻 RMSE **4.916 → 0.783 m**、位置 NEES **121.40 → 1.95**（3 自由度期望=3）。朴素误差正好落在预测的延迟偏差 `v × delay = 25 × 0.2 = 5.01 m` 上；且**前向外推救不回来**（4.904 → 4.916 m）—— 这是两次发布的共同主结论：**以偏差形式烙进去的延迟，预测不掉**。图 `examples/nonlinear_delay_demo.png`。
  - 单元测试 +35（→ 全量 **319 通过**）；验证文档 `docs/v0.23.0-nonlinear-delay-validation.md`。

## Roadmap

- [x] ~~LiDAR 点云仿真（raycasting + 噪声 + 动态物体）~~ ✅ v0.12.0
- [x] ~~相机图像仿真（光流/特征投影）~~ ✅ v0.13.0
- [x] ~~C++/Eigen 移植~~ ✅ v0.16.0（cpp/ 目录：ESKF/IMU/GNSS/轨迹 + 测试，统计等价验证）
- [x] ~~车载雷达目标检测仿真~~ ✅ v0.17.0（`radar.py`：测距/方位/多普勒 + RCS 雷达方程检测）
- [x] ~~雷达+相机融合跟踪~~ ✅ v0.19.0（`fused.py`：单 EKF 共轨雷达/相机，跨界行人横向速度 5× 收敛）
- [x] ~~融合跟踪→ADAS 标记→告警仲裁~~ ✅ v0.20.0（`track_warn.py`：`TrackToMarker` + `FusionThreatPipeline`，closing rate 从跟踪动力学推导，符号与 v0.11 对齐）
- [x] ~~多目标跟踪真值指标（OSPA / GOSPA / MOTA）~~ ✅ v0.21.0（`mot.py`：`ospa`/`gospa`/`assign`/`MotAccumulator` + `RadarFrame` 全量真值通道）
- [x] ~~延迟/乱序测量融合（measurement-time fusion + out-of-order reprocessing）~~ ✅ v0.22.0（`delay_fusion.py`：`DelayedMeasurement`/`CVModel`/`DelayedFusionFilter` + 诊断计数 + NEES 辅助；线性时不变下与顺序滤波逐位相等）
- [x] ~~非线性（ESKF）前端的回卷-重放~~ ✅ v0.23.0（`nonlinear_delay.py`：`RewindReplayESKF` 保留快照 + IMU + GNSS 三个缓冲，迟到定位锚定到 `<= t_meas` 的最近快照后**重放同一批 IMU 样本**再按有效时刻插入；与顺序 ESKF 逐位相等 `atol=1e-9`）
