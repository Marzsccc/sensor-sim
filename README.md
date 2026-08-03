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

## Run Tests

```bash
python3 tests/test_basic.py
```

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
│   ├── gnss.py          # GNSS sensor model
│   └── utils.py         # Quaternion/rotation utilities
├── examples/
│   └── s_curve_car.py   # Car S-curve demo
├── tests/
│   └── test_basic.py    # Unit tests
└── requirements.txt
```

## License

MIT
