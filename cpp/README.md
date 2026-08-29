# navsim — C++/Eigen port of the sensor-sim navigation core

Roadmap item "C++/Eigen 移植" (v0.16.0). Header-only C++17 library + demo +
tests, mirroring the Python reference modules 1:1:

| C++ header | Python source | 内容 |
|---|---|---|
| `include/nav/types.hpp` | `utils.py` (skew) + 全局约定 | Vec3/Mat3, kGravity=9.80665 |
| `include/nav/quaternion.hpp` | `utils.py` | wxyz 四元数全套：rotmat↔quat、exp/log、slerp、euler |
| `include/nav/random.hpp` | `np.random.RandomState` 用法 | mt19937 + Box-Muller（极式），exponential |
| `include/nav/imu.hpp` | `imu.py` | Allan/IEEE952 误差模型：bias/scale+交叉耦合/失准角/噪声/随机游走，三档等级 |
| `include/nav/gnss.hpp` | `gnss.py` | 白噪声+一阶 Gauss-Markov 多径+骤降状态机+输出率控制，四档等级 |
| `include/nav/eskf.hpp` | `eskf.py` | 15 维误差状态 ESKF（Solà 2017），Joseph 形式更新，6σ 野值门限 |
| `include/nav/trajectory.hpp` | `trajectory.py` | Hermite 样条 + SLERP + 差分 ω/a + 恒速重归一化 |
| `examples/eskf_demo.cpp` | `examples/eskf_demo.py` | 60 s GNSS+IMU 松组合 demo（同一场景） |
| `tests/test_nav.cpp` | `tests/test_eskf.py` 等 | 单元+统计校验 |

## 构建

```bash
cd cpp
cmake -B build -S .          # Eigen3 自动探测（含 /opt/homebrew 兜底）
cmake --build build -j
./build/test_nav             # 单元测试
./build/eskf_demo --seed 42  # 端到端 demo
```

依赖：CMake ≥ 3.16，C++17，Eigen3（`brew install eigen`）。

## 与 Python 版的对照口径

随机数流**有意不逐位对齐**（mt19937 ≠ NumPy MT19937 流布局），移植正确性
按**统计等价**验证。同参数（seed 42，60 s，tactical IMU + automotive GNSS）：

| 指标 | Python 参考 | C++ 移植（seed 42/7/99） |
|---|---|---|
| ESKF 位置 RMS | 2.28 m | 1.76 / 3.92 / 2.63 m |
| 纯惯导推算位置误差 (mean) | 12.0 m | 7.0 / 7.8 / 6.6 m |
| ESKF 速度误差 (mean) | 0.045 m/s | 0.038 / 0.080 / 0.047 m/s |
| ESKF yaw 误差 (mean) | 0.21° | 0.25° / 0.90° / 0.21° |
| GNSS 骤降 | 少量 | 17–21 次 / ~19 s |

差异主要来自随机流与骤降序列不同（C++ 侧 60 s 内 ~19 s 无 GNSS，
纯惯导段更短，DR 误差反而更低）。测试 `test_nav` 断言的统计带与
Python 参考实测值（turn specific force 2.8372 m/s²、GNSS 有效样本
186/60s、静态 IMU 均值含失准角-比例因子耦合链）逐一核对过。

### 已知口径修正（相对 Python 参考的改进）

- demo 的 yaw 误差指标加了 ±π 角度回绕（Python demo 在 yaw 跨 ±π 时
  会飙到 ~360°，属于度量 bug，滤波本身无此问题）。
- 重力统一用 9.80665（Python 侧机械编排用 9.80665、`body_accel` 用
  9.81，C++ 端一致）。

### 移植时保持的"反直觉"语义（勿"修复"）

- `Eskf::predict` 的 Qd 位置行为 0：速度不确定度经姿态/偏置耦合项进入。
- 位置-only 更新时 H 的速度块置零、R 速度项 1e12（而非删行）。
- GNSS `measure` 顺序：多径先演化 → 骤降判定 → 输出率判定 → 噪声采样。

## 后续

- [ ] CI：把 `./build/test_nav` 挂进 qa-harness 回归
- [ ] 高层接口：`FusionPipeline`（对齐 Python 端 `latency/predict` 组合链）
- [ ] 性能基准：Eigen 向量化 vs NumPy 耗时对比
