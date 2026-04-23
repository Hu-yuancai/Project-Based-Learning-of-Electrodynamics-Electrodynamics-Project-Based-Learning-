# 项目总览

## 项目名称

基于深度学习的超构表面逆向设计系统（λ = 1550 nm）

## 核心创新

### Tandem Network 架构

解决传统逆向网络的"非唯一映射"问题，通过在物理响应空间（相位）而非几何空间定义损失函数：

```
目标相位 → [逆向网络] → 预测(w,p) → [冻结的正向网络] → 预测相位
                                                              ↓
                                              HuberLoss(预测相位, 目标相位)
```

---

## 文件清单

### 核心代码模块

| 文件 | 功能 | 关键类/函数 |
|:---|:---|:---|
| `main.py` | 主程序，4 步骤流程编排 | `MetasurfaceDesignPipeline` |
| `data_generator.py` | Rytov + FP 物理模型，数据集生成 | `RigorousMetasurfaceSimulator` |
| `forward_model.py` | 正向网络：$(w,p) \to (\sin\phi, \cos\phi)$ | `ForwardPredictor`, `train_forward_model` |
| `inverse_design.py` | Tandem 逆向网络训练 | `TandemTrainer`, `InverseDesigner` |
| `metasurface_design.py` | 阵列设计与可视化 | `design_anomalous_refraction_array` |

### 脚本与测试

| 文件 | 用途 | 运行时间 |
|:---|:---|:---|
| `quick_demo.py` | 快速演示（需更新至新 API） | 5–10 分钟 |
| `verify.py` | 环境检查 | <1 分钟 |
| `smoke_test.py` | 端到端冒烟测试 | ~2 分钟 |

### 文档

| 文件 | 内容 |
|:---|:---|
| `README.md` | 物理原理、网络架构、实验结果、参考文献 |
| `USAGE.md` | 运行方式、参数配置、结果解读、故障排除 |
| `INDEX.md` | 本文件，项目总览 |

---

## 项目架构

```
数据生成（data_generator.py）
  Rytov 二阶近似有效折射率 + Fabry-Perot 透射系数
  拉丁超立方采样，5000 样本
  输出：X=[w_nm, p_nm]，Y=[T, phi_deg]
        ↓
正向网络（forward_model.py）
  输入：StandardScaler 归一化的 (w, p)
  输出：(sin φ, cos φ)，单位圆归一化
  损失：sin/cos MSE + 角度惩罚 + 单位模约束
        ↓
Tandem 逆向网络（inverse_design.py）
  输入：归一化目标相位 φ/180
  输出：归一化几何参数，Sigmoid 约束
  训练：正向网络冻结，Huber Loss 在相位空间
        ↓
阵列设计（metasurface_design.py）
  广义斯涅尔定律计算理想相位梯度
  逆向网络为每个单元设计 (w, p)
  正向网络验证，计算等效折射角
```

---

## 物理参数

| 参数 | 符号 | 取值 |
|:---|:---|:---|
| 工作波长 | λ | 1550 nm |
| 硅折射率 | $n_{Si}$ | 3.4777 |
| SiO₂ 折射率 | $n_{sub}$ | 1.444 |
| 纳米柱宽度 | $w$ | 80–500 nm |
| 单元周期 | $p$ | 400–1073 nm |
| 纳米柱高度 | $H$ | 900 nm（固定） |

---

## 实验结果（v2.0）

| 模块 | 指标 | 数值 |
|:---|:---|:---|
| 数据集 | 透射率均值 | 0.65 |
| 数据集 | 相位范围 | [-180°, 180°] |
| 正向网络 | 相位 MAE | 1.04° |
| 正向网络 | 最大误差 | 9.32° |
| Tandem 网络 | 平均误差 | 1.72° |
| Tandem 网络 | 最大误差 | 5.80° |
| 阵列设计 | 平均相位误差 | 1.44° |
| 阵列设计 | 折射角偏差 | 0.0°（目标 30°） |

---

## 关键设计决策

### sin/cos 双通道相位编码

直接回归相位角度在 ±180° 处产生 360° 虚假误差。改用 $(\sin\phi, \cos\phi)$ 双通道，输出层归一化至单位圆，彻底解决周期性跳变。

### Tandem 架构解决非唯一性

同一目标相位可对应多个几何结构。直接训练"相位→结构"时，网络输出多解的平均值，该平均结构通常无法实现目标相位。Tandem 架构将 Loss 定义在相位响应空间，网络自动找到任意一个物理可行解。

### Huber Loss 提升鲁棒性

相位空间中存在少量离群点（相位跳变边界附近）。Huber Loss 对小误差（<10°）用二次惩罚，对大误差用线性惩罚，比纯 MSE 更鲁棒。

---

## 如何扩展

- 改变目标折射角：修改 `main.py` 中 `config['target_angle']`
- 改变阵列规模：修改 `config['n_elements']`
- 多折射角批量设计：循环调用 `design_anomalous_refraction_array()`
- 更高精度：增加 `n_samples`、`forward_epochs`、`tandem_epochs`

---

## 参考文献

[1] YU N, CAPASSO F. Flat optics with designer metasurfaces[J/OL]. Nature Materials, 2014, 13(2): 139-150. DOI:10.1038/nmat3839.

[2] DONG Y, AN S, JIANG H, 等. Advanced deep learning approaches in metasurface modeling and design: A review[J/OL]. Progress in Quantum Electronics, 2025, 99: 100554. DOI:10.1016/j.pquantelec.2025.100554.

[3] PEURIFOY J, SHEN Y, JING L, 等. Nanophotonic particle simulation and inverse design using artificial neural networks[J/OL]. Science Advances, 2018, 4(6): eaar4206. DOI:10.1126/sciadv.aar4206.

[4] MA W, CHENG F, LIU Y. Deep-Learning-Enabled On-Demand Design of Chiral Metamaterials[J/OL]. ACS Nano, 2018, 12(6): 6326-6334. DOI:10.1021/acsnano.8b03569.

[5] JIANG J, SELL D, HOYER S, 等. Free-Form Diffractive Metagrating Design Based on Generative Adversarial Networks[J/OL]. ACS Nano, 2019, 13(8): 8872-8878. DOI:10.1021/acsnano.9b02371.
