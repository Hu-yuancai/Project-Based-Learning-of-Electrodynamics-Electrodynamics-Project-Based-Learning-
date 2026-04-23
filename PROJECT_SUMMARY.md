# 项目完成清单（v2.0）

**最后更新：** 2026 年 4 月  
**系统版本：** v2.0（Rytov + FP 物理建模版）  
**状态：** ✅ 完全可用

---

## 项目概述

基于等效介质理论（Rytov 二阶近似）与 Fabry-Perot 共振模型构建物理自洽数据集，采用 Tandem 串联神经网络架构，在 1550 nm 通信波段实现超构表面单元从目标相位到几何参数的高效逆向设计。

---

## 文件清单

### 核心代码（5 个模块）

| 文件 | 状态 | 功能 |
|:---|:---|:---|
| `main.py` | ✅ | 主程序，4 步骤流程编排 |
| `data_generator.py` | ✅ | Rytov + FP 物理模型，`RigorousMetasurfaceSimulator` |
| `forward_model.py` | ✅ | 正向网络，sin/cos 双通道输出，单位圆归一化 |
| `inverse_design.py` | ✅ | Tandem 逆向网络，Huber Loss，噪声增强 |
| `metasurface_design.py` | ✅ | 阵列设计，广义斯涅尔定律，可视化 |

### 脚本

| 文件 | 状态 | 说明 |
|:---|:---|:---|
| `smoke_test.py` | ✅ | 端到端冒烟测试（已通过） |
| `verify.py` | ✅ | 环境检查 |
| `quick_demo.py` | ⚠️ | 使用旧 API，需更新 |

### 文档

| 文件 | 状态 | 内容 |
|:---|:---|:---|
| `README.md` | ✅ | 物理原理、架构、结果、参考文献 |
| `USAGE.md` | ✅ | 运行方式、参数配置、故障排除 |
| `INDEX.md` | ✅ | 项目总览、文件清单 |
| `PROJECT_SUMMARY.md` | ✅ | 本文件 |

---

## 物理模型（v2.0）

| 组件 | 实现 |
|:---|:---|
| 有效折射率 | Rytov 二阶近似（TE 偏振） |
| 透射系数 | Fabry-Perot 复透射系数 |
| 模式耦合效率 | 高斯模型，下限 0.88，上限 0.99 |
| 散射损耗 | 周长/面积比模型，上限 0.05 |
| 填充因子增益 | $\sqrt{1 + (1-f) \times 0.55}$，补偿间隙直接透射 |
| 参数范围 | $w \in [80, 500]$ nm，$p \in [400, 1073]$ nm，$H = 900$ nm |

---

## 网络架构（v2.0）

### 正向网络（ForwardPredictor）

- 输入：2 维，StandardScaler 归一化的 $(w, p)$
- 隐藏层：[256, 512, 512, 256, 128]，LeakyReLU(0.1) + BN + Dropout(0.05)
- 输出：$(\sin\phi, \cos\phi)$，输出层归一化至单位圆
- 损失：sin/cos MSE + 角度直接惩罚 + 单位模约束
- 训练：Adam(lr=2e-3)，早停耐心 100 轮

### 逆向网络（InverseDesigner）

- 输入：1 维，归一化目标相位 $\phi/180$
- 隐藏层：[256, 512, 512, 256, 128]，Kaiming 初始化
- 输出：2 维，Sigmoid 约束至 [0,1]，反归一化到物理范围
- 损失：Huber Loss（阈值 10°）
- 训练：Adam(lr=1e-3)，噪声增强 ±1.5°，早停耐心 150 轮

---

## 实验结果（v2.0）

### 数据集

| 指标 | 数值 |
|:---|:---|
| 有效样本数 | ~4552 / 5000（91%） |
| 透射率均值 | 0.65 |
| 透射率范围 | [0.39, 0.87] |
| T > 0.95 比例 | 0.0% |
| 相位范围 | [-180°, 180°] |

### 正向网络

| 指标 | 数值 |
|:---|:---|
| 相位 MAE | 1.04° |
| 相位 RMSE | 1.47° |
| 最大误差 | 9.32° |

### Tandem 逆向网络

| 指标 | 数值 |
|:---|:---|
| 平均误差 | 1.72° |
| 最大误差 | 5.80° |
| 中位数误差 | 1.47° |

### 反常折射阵列（目标 30°，21 单元）

| 指标 | 数值 |
|:---|:---|
| 实现折射角 | 30.0°（偏差 0.0°） |
| 平均相位误差 | 1.44° |
| 最大相位误差 | 3.56° |

---

## 输出目录结构

```
results/
├── dataset_visualization.png
├── forward_training.png
├── forward_validation.png
├── tandem_training.png
├── tandem_validation.png
├── metasurface_design_results.png   ← 最重要
├── metasurface_parameters.png
├── dataset.npz
├── forward_model_weights.pth
├── inverse_model_weights.pth
└── design_result.npz
```

---

## 运行命令

```bash
# 环境检查
python verify.py

# 冒烟测试（~2 分钟）
"C:\Users\23229\AppData\Local\Programs\Python\Python312\python.exe" smoke_test.py

# 完整训练（CPU 约 20–40 分钟）
"C:\Users\23229\AppData\Local\Programs\Python\Python312\python.exe" main.py
```

---

## 参考文献

[1] YU N, CAPASSO F. Flat optics with designer metasurfaces[J/OL]. Nature Materials, 2014, 13(2): 139-150. DOI:10.1038/nmat3839.

[2] KHORASANINEJAD M, CHEN W T, DEVLIN R C, 等. Metalenses at visible wavelengths[J/OL]. Science, 2016, 352(6290): 1190-1194. DOI:10.1126/science.aaf6644.

[3] DONG Y, AN S, JIANG H, 等. Advanced deep learning approaches in metasurface modeling and design: A review[J/OL]. Progress in Quantum Electronics, 2025, 99: 100554. DOI:10.1016/j.pquantelec.2025.100554.

[4] MALKIEL I, MREJEN M, NAGLER A, 等. Plasmonic nanostructure design and characterization via Deep Learning[J/OL]. Light: Science & Applications, 2018, 7(1): 60. DOI:10.1038/s41377-018-0060-7.

[5] PEURIFOY J, SHEN Y, JING L, 等. Nanophotonic particle simulation and inverse design using artificial neural networks[J/OL]. Science Advances, 2018, 4(6): eaar4206. DOI:10.1126/sciadv.aar4206.

[6] DEVLIN R, KHORASANINEJAD R, CHEN W T, 等. High efficiency dielectric metasurfaces at visible wavelengths[J/OL]. 2016. DOI:10.48550/arXiv.1603.02735.

[7] MA W, CHENG F, LIU Y. Deep-Learning-Enabled On-Demand Design of Chiral Metamaterials[J/OL]. ACS Nano, 2018, 12(6): 6326-6334. DOI:10.1021/acsnano.8b03569.

[8] JIANG J, SELL D, HOYER S, 等. Free-Form Diffractive Metagrating Design Based on Generative Adversarial Networks[J/OL]. ACS Nano, 2019, 13(8): 8872-8878. DOI:10.1021/acsnano.9b02371.
