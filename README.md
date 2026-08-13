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
│   ├── allan.py         # Overlapping Allan variance + noise extraction
│   ├── gnss.py          # GNSS sensor model
│   ├── wheel.py         # Wheel odometry model
│   ├── latency.py       # Latency / clock sync / effective delay
│   ├── predict.py       # Display-time pose prediction & AR-HUD pipeline
│   └── utils.py         # Quaternion/rotation utilities
├── examples/
│   ├── s_curve_car.py   # Car S-curve demo
│   ├── allan_example.py # Allan variance demo (gyro noise characterization)
│   ├── latency_demo.py  # Latency & time-sync demo
│   └── predict_demo.py  # AR-HUD display-time prediction demo
├── tests/
│   ├── test_basic.py    # Unit tests
│   ├── test_allan.py    # Allan variance tests
│   ├── test_latency.py  # Latency / time-sync tests
│   └── test_predict.py  # Display-time prediction tests
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

## Roadmap

- [ ] LiDAR 点云仿真（raycasting + 噪声 + 动态物体）
- [ ] 相机图像仿真（光流/特征投影）
- [ ] C++/Eigen 移植
- [ ] 真太阳时支持（八字引擎 v0.2）
