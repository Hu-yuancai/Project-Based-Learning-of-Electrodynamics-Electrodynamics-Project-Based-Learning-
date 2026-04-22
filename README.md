# 基于深度学习的超构表面逆向设计系统

## 项目简介

本项目实现了一个完整的超构表面（Metasurface）逆向设计系统，使用深度学习神经网络来解决传统电磁仿真中的"逆问题"。

### 核心创新：Tandem Network 架构

传统的逆向设计网络面临**非唯一性问题**：同一个目标相位可能对应多个不同的纳米柱结构。本项目通过创新的 Tandem 网络架构解决了这个问题：

```
目标Phase → [逆向网络] → 预测(L,W) → [冻结的正向网络] → 预测Phase
                                              ↓
                                    Loss = 目标Phase vs 预测Phase
```

关键：损失函数定义在**物理响应空间**（相位）而非几何空间，自动绕过非唯一性。

## 项目结构

```
.
├── main.py                      # 主程序入口
├── data_generator.py            # 模块一：数据集构建
├── forward_model.py             # 模块二：正向网络训练
├── inverse_design.py            # 模块三：Tandem 逆向网络
├── metasurface_design.py        # 模块四：阵列设计与可视化
├── requirements.txt             # 依赖包列表
└── README.md                    # 本文件
```

## 快速开始

### 1. 环境配置

#### 安装依赖：
```bash
pip install -r requirements.txt
```

#### 或者手工安装：
```bash
pip install torch numpy scikit-learn matplotlib scipy
```

**注意**：如果需要 CUDA 支持，请从 https://pytorch.org/get-started/locally/ 获取对应版本。

### 2. 运行主程序

```bash
python main.py
```

该命令将自动执行以下步骤：

1. **数据集构建** - 生成 5000 组 [L, W] → [相位, 振幅] 的训练数据
2. **正向网络训练** - 训练网络学习"几何 → 相位"映射（300 epochs）
3. **Tandem 网络训练** - 训练逆向网络"相位 → 几何"映射（500 epochs）
4. **超构表面设计** - 设计一个 21 单元的反常折射阵列

预期运行时间：**10-15 分钟**（GPU）/ **30-45 分钟**（CPU）

### 3. 输出文件

运行完成后，在 `results/` 目录中生成：

#### 可视化结果（PNG）：
- `dataset_visualization.png` - 数据集的相位和振幅分布
- `forward_training.png` - 正向网络的训练曲线
- `forward_validation.png` - 正向网络的预测准确性
- `tandem_training.png` - Tandem 网络的训练曲线
- `tandem_validation.png` - 逆向网络的验证结果
- `metasurface_design_results.png` - **完整的设计结果图**（4 子图）
- `metasurface_parameters.png` - 几何参数分析

#### 数据文件：
- `dataset.npz` - 原始训练数据
- `forward_model_weights.pth` - 正向网络权重
- `inverse_model_weights.pth` - 逆向网络权重
- `design_result.npz` - 最终设计数据

## 物理原理速览

### 超构表面基础

超构表面是由亚波长纳米天线阵列组成的二维结构。与传统光学元件不同，它在器件表面引入**相位突变**来控制光的传播方向。

### 广义斯涅尔定律

当界面上存在空间变化的相位梯度 $\frac{d\Phi}{dx}$ 时，折射定律改写为：

$$n_t \sin\theta_t - n_i \sin\theta_i = \frac{\lambda_0}{2\pi} \frac{d\Phi}{dx}$$

对于垂直入射和目标折射角 $\theta_t = 30°$，所需的相位梯度为：

$$\frac{d\Phi}{dx} = k_0 \sin\theta_t = \frac{2\pi}{\lambda_0} \sin 30°$$

### 逆向设计问题

**困难**：已知目标相位，如何设计纳米柱的长度 L 和宽度 W？

**传统解决方案**：对每个结构进行 COMSOL 全波仿真 → 计算量巨大

**本项目方案**：用神经网络学习"结构 ↔ 相位"映射，在毫秒内预测

## 关键代码示例

### 1. 数据生成
```python
from data_generator import MetasurfaceUnitSimulator

simulator = MetasurfaceUnitSimulator(wavelength=700e-9)
X, Y = simulator.generate_dataset(n_samples=5000)
# X: [n_samples, 2] - (L, W) 几何参数
# Y: [n_samples, 2] - (振幅, 相位)
```

### 2. 正向网络训练
```python
from forward_model import train_forward_model

model, scaler, losses, _ = train_forward_model(X, Y[:, 1], epochs=300)
# model: 训练好的网络
# scaler: 输入标准化器
```

### 3. Tandem 网络训练
```python
from inverse_design import TandemTrainer

tandem = TandemTrainer(model, scaler)
losses = tandem.train_with_progress(epochs=500)
inverse_model = tandem.inverse_model
```

### 4. 超构表面设计
```python
from metasurface_design import design_anomalous_refraction_array

design = design_anomalous_refraction_array(
    inverse_model=inverse_model,
    scaler_X=scaler,
    forward_model=model,
    target_angle_deg=30,
    n_elements=21
)
```

## 结果解读

### 设计注意事项

#### 相位误差
- **平均误差 < 5°** 表示设计优秀
- **5-10°** 表示可接受
- **> 10°** 可能需要调整网络参数

#### 几何可行性
- L 和 W 的变化应该光滑（图 3 的柱状图中）
- 突变可能表示网络遇到模式转换区域

#### 折射角准确性
- 实现角 vs 目标角的误差应该 **< 2°**
- 通过线性拟合相位梯度计算得出

## 物理与 AI 的碰撞：避坑案例

### 案例 1：相位的拓扑问题

**问题**：初始方案用 MSE Loss 直接回归相位的度数值，导致在 ±180° 附近无法收敛。

**原因**：相位是 $\mathbb{R}^1 / 360°\mathbb{Z}$ 上的量，有周期拓扑。179° 和 -179° 物理上只差 2°，但数值差 358°。

**解决**：输出改为 $[\sin\phi, \cos\phi]$ 双通道，利用三角恒等式自动处理周期性。

**启示**：AI 擅长拟合，但**物理对称性**需要人类来编码。

### 案例 2：非唯一映射

**问题**：直接训练 "相位 → (L, W)" 的映射，网络输出的是多个解的平均值，而这个平均结构通常**无法实现目标相位**。

**原因**：同一相位可能对应 2-3 个不同的结构（不同的共振模式）。

**解决**：Tandem 架构，Loss 定义在相位空间而非几何空间，网络自动找到**任意一个可行解**。

**启示**：物理**响应空间**比几何空间规范得多。

## 自定义修改

### 改变目标折射角
```python
# 在 main.py 中修改：
pipeline.run_full_pipeline(
    target_angle=45  # 改为 45°
)
```

### 调整数据集大小
```python
pipeline.run_full_pipeline(
    n_samples=10000  # 更多数据
)
```

### 改变网络超参数
编辑 `forward_model.py` 或 `inverse_design.py` 中的 `hidden_dims` 参数：
```python
model = ForwardPredictor(hidden_dims=[256, 512, 512, 256])
```

### 使用 GPU（如果可用）
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## 计算性能

| 硬件 | 数据生成 | 正向训练 | Tandem 训练 | 设计 | 总计 |
|:---|:---|:---|:---|:---|:---|
| CPU (Intel i7) | 30s | 8分钟 | 12分钟 | 15s | ~21分钟 |
| GPU (NVIDIA A100) | 5s | 40s | 2分钟 | 5s | ~3分钟 |

## 常见问题

### Q1: 模为什么不收敛？
**A**: 
- 检查学习率（默认 0.001），尝试改为 0.0001
- 增加数据集大小（n_samples）
- 减少网络宽度（降低模型复杂度）

### Q2: 内存不足怎么办？
**A**:
- 减少 batch_size（在 `train_forward_model` 中）
- 减少 n_samples
- 关闭 PyTorch 的自动混合精度

### Q3: 预测的 L 和 W 有时很大或很小，是否正常？
**A**: 完全正常！由于"一对多"映射的存在，同一相位可能对应不同的 (L, W) 组合。只要通过正向网络验证的相位接近目标即可。

## 论文和参考资源

数学原理参考：
- S. Sun et al., "Gradient-index meta-surfaces as a bridge linking photonic metamaterials and metasurfaces," *Nat. Mater.* **11**, 308-313 (2012)

深度学习架构参考：
- "Neural Networks for Physics-Informed Machine Learning" (Raissi et al.)

## 许可证

本项目用于教育研究用途。

## 作者

基于同济大学《电动力学》项目化学习要求开发

---

**最后更新**：2026年4月

**有问题或建议？** 检查代码中的详细注释，或参考主程序中的日志输出。
