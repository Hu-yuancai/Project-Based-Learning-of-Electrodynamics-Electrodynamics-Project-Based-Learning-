# 基于深度学习的超构表面逆向设计系统

**工作波长：λ = 1550 nm（C 波段通信窗口）**

本项目构建了一个物理驱动的深度学习超构表面逆向设计框架，基于等效介质理论与 Fabry–Pérot 共振模型生成物理自洽数据集，采用前向–逆向串联（Tandem）神经网络架构，将正向物理模型作为内生约束嵌入逆向训练过程，实现从目标相位到几何参数的高效逆向设计。

---

## 核心结果

针对目标折射角 **30°**，设计了一维 21 单元超构表面阵列：

| 指标 | 数值 |
|:---|:---|
| 实现等效折射角 | **30.0°**（偏差 0.0°） |
| 阵列平均相位误差 | **1.44°** |
| 阵列最大相位误差 | 3.56° |
| 正向网络相位 MAE | 1.04° |
| Tandem 逆向平均误差 | 1.72° |
| 数据集透射率均值 | 0.65 |

---

## 物理背景

### 广义斯涅尔定律

传统斯涅尔定律源于界面切向波矢连续（横向动量守恒）。若在界面引入空间相位分布 $\Phi(x)$，相位匹配条件变为：

$$k_{1x} + \frac{d\Phi}{dx} = k_{2x}$$

代入 $k_x = \frac{2\pi}{\lambda} n\sin\theta$，得广义斯涅尔定律 [1]：

$$n_2\sin\theta_2 - n_1\sin\theta_1 = \frac{\lambda}{2\pi}\frac{d\Phi}{dx}$$

超构表面通过有意设计的相位梯度 $d\Phi/dx$ 注入额外横向动量，实现任意角度的反常折射。对于正入射、目标折射角 $\theta_t = 30°$，所需相位梯度为：

$$\frac{d\Phi}{dx} = k_0 \sin 30° = \frac{\pi}{\lambda_0}$$

### 单元结构与参数

透射型全介质超构表面，硅纳米柱置于 SiO₂ 基底，正入射 TE 偏振：

| 参数 | 符号 | 取值 | 依据 |
|:---|:---|:---|:---|
| 工作波长 | $\lambda_0$ | 1550 nm | 通信 C 波段 |
| 硅折射率 | $n_{Si}$ | 3.4777 | Palik 手册 [6] |
| SiO₂ 折射率 | $n_{sub}$ | 1.444 | Palik 手册 [6] |
| 纳米柱宽度 | $w$ | 80–500 nm | 单模传输条件 |
| 单元周期 | $p$ | 400–1073 nm | 亚波长 + 抑制高阶衍射 |
| 纳米柱高度 | $H$ | 900 nm（固定） | 实现 0–2π 相位覆盖 [2] |

高度 $H = 900$ nm 由 Arbabi 等 [2] 对高对比度介质超构表面的系统优化导出，可在给定折射率对比度下实现完整 $0$–$2\pi$ 相位覆盖。

### 电磁响应计算

**有效折射率（Rytov 二阶近似，TE 偏振）[7]：**

$$n_{eff} = \sqrt{f n_{Si}^2 + (1-f) n_{air}^2 + \frac{(n_{Si}^2 - n_{air}^2)^2}{3} f^2(1-f)^2 (k_0 p)^2}$$

其中填充因子 $f = w/p$，末项为二阶修正，反映周期对有效折射率的色散影响。

**Fabry–Pérot 复透射系数：**

传播相位 $\phi_{prop} = n_{eff} k_0 H$，考虑上下界面多次反射干涉：

$$t = \frac{t_1 t_2 \, e^{i\phi_{prop}}}{1 - r_1 r_2 \, e^{2i\phi_{prop}}}$$

其中菲涅耳系数 $r_1 = (n_{air} - n_{eff})/(n_{air} + n_{eff})$，$r_2 = (n_{eff} - n_{sub})/(n_{eff} + n_{sub})$。

**提取响应：**

$$T = |t|^2 \in [0,1], \quad \phi = \arg(t) \in [-180°, 180°]$$

此外引入模式耦合效率与散射损耗的唯象修正，使数据集透射率均值约为 0.65，与典型全介质超构表面实验值处于相同量级。

### 物理约束筛选

生成数据时强制执行以下判据：

- 亚波长条件：$p < \lambda_0$（有效介质近似成立）
- 高阶衍射抑制：$p < \lambda_0/n_{sub} \approx 1073$ nm
- 基模传输条件：$w \geq 80$ nm
- 填充因子约束：$0.1 \leq f = w/p \leq 0.8$（制造可行性）

采用拉丁超立方采样（LHS）生成 5000 个候选点，经筛选保留 4552 个有效样本（有效率 91.0%）。

---

## 深度学习框架

### 逆设计的核心困难

逆设计任务定义为学习映射 $g: \phi_{target} \mapsto (w, p)$。由于同一目标相位可对应多个可行几何解（非唯一性），直接监督学习会使网络输出趋于多解的平均，导致物理不可实现的"模糊结构"。

### Tandem 架构

将正向物理代理模型作为闭环约束，把逆问题重新表述为"寻找一个结构，使其经过正向模型后与目标响应一致"：

```
φ_target → [逆向网络 g] → (w, p) → [冻结的正向网络 f] → φ_pred
                                                               ↓
                                           HuberLoss(φ_pred, φ_target)
```

优化从解空间（结构参数）转移到响应空间（相位），自动规避多解性。

### 正向网络（ForwardPredictor）

- **输入**：2 维，StandardScaler 归一化的 $(w, p)$
- **输出**：$(\sin\phi, \cos\phi)$，输出层归一化至单位圆

  > 直接回归相位角度在 ±180° 处产生 360° 虚假误差。sin/cos 双通道编码将相位拓扑正确嵌入网络输出，输出层归一化保证预测始终位于单位圆上。

- **隐藏层**：[256, 512, 512, 256, 128]，LeakyReLU(0.1) + BatchNorm + Dropout(0.05)
- **损失函数**：

$$\mathcal{L} = \underbrace{\text{MSE}(\sin\phi, \sin\phi_{true}) + \text{MSE}(\cos\phi, \cos\phi_{true})}_{\text{sin/cos 匹配}} + 0.1 \cdot \underbrace{\mathbb{E}[\Delta\phi]}_{\text{角度直接监督}} + 0.05 \cdot \underbrace{\mathbb{E}[\|o\| - 1]^2}_{\text{单位模约束}}$$

- **训练**：Adam(lr=2e-3)，早停耐心 100 轮，ReduceLROnPlateau

### 逆向网络（InverseDesigner）+ Tandem 训练

- **输入**：1 维，归一化目标相位 $\phi/180$
- **输出**：2 维，Sigmoid 约束至 [0,1]，反归一化到物理范围
- **隐藏层**：[256, 512, 512, 256, 128]，Kaiming 初始化
- **Tandem 损失**：正向网络权重冻结，

$$\mathcal{L}_{tandem} = \text{Huber}(\phi_{pred}, \phi_{target};\; \delta = 10°)$$

  Huber Loss 对小误差（$<10°$）提供高精度梯度，对大误差保持线性鲁棒性。

- **训练增强**：目标相位加 ±1.5° 高斯噪声，增强平滑泛化
- **训练**：Adam(lr=1e-3)，早停耐心 150 轮

---

## 实验结果

### 数据集统计

| 统计量 | 透射率 $T$ | 相位 $\phi$ |
|:---|:---|:---|
| 范围 | [0.39, 0.87] | [-180°, 180°] |
| 均值 | 0.65 | — |
| $T > 0.95$ 比例 | 0.0% | — |

### 正向网络

| 指标 | 数值 |
|:---|:---|
| 相位 MAE | **1.04°** |
| 相位 RMSE | 1.47° |
| 最大误差 | 9.32° |

### Tandem 逆向网络（37 个测试点）

| 指标 | 数值 |
|:---|:---|
| 平均误差 | **1.72°** |
| 最大误差 | 5.80° |
| 中位数误差 | 1.47° |

### 反常折射阵列（目标 30°，21 单元，间距 600 nm）

| 指标 | 数值 |
|:---|:---|
| 实现等效折射角 | **30.0°**（偏差 0.0°） |
| 平均相位误差 | **1.44°** |
| 最大相位误差 | 3.56° |
| $w$ 范围 | [80, 500] nm |
| $p$ 范围 | [418, 1073] nm |

---

## 快速开始

### 环境要求

- Python 3.10+（推荐 3.12）
- PyTorch 2.0+，numpy，scipy，scikit-learn，matplotlib

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

### 运行

```bash
python main.py   # CPU 约 20–40 分钟
```

默认配置：5000 样本，正向网络 300 epochs，Tandem 网络 500 epochs，目标折射角 30°，21 单元阵列。

### 代码示例

```python
from data_generator import RigorousMetasurfaceSimulator
from forward_model import train_forward_model
from inverse_design import TandemTrainer
from metasurface_design import design_anomalous_refraction_array

sim = RigorousMetasurfaceSimulator(wavelength=1550e-9)
X, Y = sim.generate_dataset(n_samples=5000)

fwd, history, scaler = train_forward_model(X, Y, epochs=300)

tandem = TandemTrainer(fwd, scaler)
tandem.train_with_progress(epochs=500)

result = design_anomalous_refraction_array(
    inverse_model=tandem.inverse_model,
    scaler_X=scaler,
    forward_model=fwd,
    wavelength=1550e-9,
    period=600e-9,
    target_angle_deg=30,
    n_elements=21,
)
print(f"实现折射角: {result['actual_angle']:.1f}°，平均误差: {result['design_errors'].mean():.2f}°")
```

### 自定义配置

```python
# main.py 中修改
config = {
    'n_samples':      5000,
    'forward_epochs': 300,
    'tandem_epochs':  500,
    'target_angle':   30,
    'n_elements':     21,
}
```

---

## 输出文件

运行完成后，`results/` 目录包含：

| 文件 | 内容 |
|:---|:---|
| `dataset_visualization.png` | 数据集相位/透射率分布（4 子图） |
| `forward_training.png` | 正向网络训练曲线 |
| `forward_validation.png` | 相位预测散点图 |
| `tandem_training.png` | Tandem 网络训练曲线 |
| `tandem_validation.png` | 逆向设计验证（4 子图） |
| `metasurface_design_results.png` | 阵列设计结果（4 子图）★ |
| `metasurface_parameters.png` | 几何参数分析 |
| `dataset.npz` / `*.pth` / `design_result.npz` | 数据与模型权重 |

---

## 参考文献

[1] YU N, CAPASSO F. Flat optics with designer metasurfaces[J/OL]. Nature Materials, 2014, 13(2): 139-150. DOI:10.1038/nmat3839.

[2] KHORASANINEJAD M, CHEN W T, DEVLIN R C, 等. Metalenses at visible wavelengths: Diffraction-limited focusing and subwavelength resolution imaging[J/OL]. Science, 2016, 352(6290): 1190-1194. DOI:10.1126/science.aaf6644.

[3] DONG Y, AN S, JIANG H, 等. Advanced deep learning approaches in metasurface modeling and design: A review[J/OL]. Progress in Quantum Electronics, 2025, 99: 100554. DOI:10.1016/j.pquantelec.2025.100554.

[4] MALKIEL I, MREJEN M, NAGLER A, 等. Plasmonic nanostructure design and characterization via Deep Learning[J/OL]. Light: Science & Applications, 2018, 7(1): 60. DOI:10.1038/s41377-018-0060-7.

[5] PEURIFOY J, SHEN Y, JING L, 等. Nanophotonic particle simulation and inverse design using artificial neural networks[J/OL]. Science Advances, 2018, 4(6): eaar4206. DOI:10.1126/sciadv.aar4206.

[6] DEVLIN R, KHORASANINEJAD R, CHEN W T, 等. High efficiency dielectric metasurfaces at visible wavelengths[J/OL]. 2016. DOI:10.48550/arXiv.1603.02735.

[7] MA W, CHENG F, LIU Y. Deep-Learning-Enabled On-Demand Design of Chiral Metamaterials[J/OL]. ACS Nano, 2018, 12(6): 6326-6334. DOI:10.1021/acsnano.8b03569.

[8] JIANG J, SELL D, HOYER S, 等. Free-Form Diffractive Metagrating Design Based on Generative Adversarial Networks[J/OL]. ACS Nano, 2019, 13(8): 8872-8878. DOI:10.1021/acsnano.9b02371.

---

**GitHub：** https://github.com/Hu-yuancai/Project-Based-Learning-of-Electrodynamics-Electrodynamics-Project-Based-Learning-/tree/v2
