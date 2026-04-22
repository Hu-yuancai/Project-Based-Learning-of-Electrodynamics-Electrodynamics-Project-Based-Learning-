# 使用指南

## 目录结构

```
.
├── main.py                    ← 主程序（推荐）
├── quick_demo.py             ← 快速演示（快 3-5 倍）
├── verify.py                 ← 环境检查
├── data_generator.py         ← 数据生成模块
├── forward_model.py          ← 正向网络模块
├── inverse_design.py         ← 逆向网络（Tandem）模块
├── metasurface_design.py     ← 阵列设计和可视化
├── requirements.txt          ← 依赖包
├── README.md                 ← 完整说明文档
└── USAGE.md                  ← 本文件
```

## 三种运行方式

### 方式 1：完整流程（推荐用于最终项目）

```bash
python main.py
```

**特点：**
- 数据集：5000 个样本
- 正向网络：300 epochs
- Tandem 网络：500 epochs
- 结果质量：最高
- 运行时间：**10-15 分钟**（GPU）/ **30-45 分钟**（CPU）

**输出：** `results/` 目录下的完整结果

### 方式 2：快速演示（推荐用于测试和演示）

```bash
python quick_demo.py
```

**特点：**
- 数据集：1000 个样本
- 正向网络：50 epochs
- Tandem 网络：100 epochs
- 结果质量：中等
- 运行时间：**3-5 分钟**（GPU）/ **10-15 分钟**（CPU）

**输出：** `quick_demo_results/` 目录下的结果

**适用场景：**
- 快速验证代码是否工作
- 演示设计流程
- 调试参数

### 方式 3：环境检查

```bash
python verify.py
```

运行前先检查环境配置是否正确。

## 安装步骤（完整指南）

### 1. 安装 Python 包

#### 快速安装（推荐）：
```bash
pip install -r requirements.txt
```

#### 手动安装：
```bash
pip install torch numpy scikit-learn matplotlib scipy
```

#### 如果需要特定的 PyTorch CUDA 版本：
访问 https://pytorch.org/get-started/locally/ 并选择对应的安装命令。

示例（CUDA 11.8）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 验证环境（可选但推荐）

```bash
python verify.py
```

输出应该类似：
```
[1/6] 检查 Python 版本... ✓
[2/6] 检查依赖库... ✓
[3/6] 检查 PyTorch 配置... ✓
[4/6] 测试数据生成模块... ✓
[5/6] 测试正向网络模块... ✓
[6/6] 检查输出目录... ✓

✓ 所有检查通过！可以运行 main.py
```

### 3. 运行程序

选择上述三种方式之一。

## 修改参数

### 修改目标折射角

在 `main.py` 中修改 `config` 字典：

```python
config = {
    'target_angle': 45,  # 改为 45°（默认 30°）
    ...
}
```

### 修改阵列规模

```python
config = {
    'n_elements': 31,    # 改为 31 个单元（默认 21）
    ...
}
```

### 修改数据集大小

```python
config = {
    'n_samples': 10000,  # 更多数据，更好的结果
    ...
}
```

### 调整训练轮数

```python
config = {
    'forward_epochs': 500,  # 正向网络训练更久
    'tandem_epochs': 1000,  # Tandem 网络训练更久
    ...
}
```

## 输出文件解读

### 关键可视化结果

| 文件 | 说明 |
|:---|:---|
| `metasurface_design_results.png` | **最重要**：4 个子图展示完整设计 |
| `metasurface_parameters.png` | 几何参数的详细分析 |
| `forward_training.png` | 正向网络训练曲线 |
| `tandem_training.png` | Tandem 网络训练曲线 |
| `forward_validation.png` | 正向网络预测准确性 |

### 数据文件

| 文件 | 内容 |
|:---|:---|
| `dataset.npz` | 原始训练数据（X, Y） |
| `forward_model_weights.pth` | 正向网络权重 |
| `inverse_model_weights.pth` | 逆向网络权重 |
| `design_result.npz` | 最终设计数据 |

## 性能基准

### 硬件要求

| 硬件 | 最小配置 | 推荐配置 |
|:---|:---|:---|
| CPU | Intel i5 | Intel i7/i9 |
| 内存 | 4 GB | 8 GB+ |
| GPU | - | NVIDIA RTX 3060+ |

### 运行时间

| 组件 | CPU（分钟） | GPU（秒） |
|:---|:---|:---|
| 数据生成 | 0.5 | 5 |
| 正向网络 (300 epochs) | 8 | 40 |
| Tandem 网络 (500 epochs) | 12 | 120 |
| 设计和可视化 | 2 | 10 |
| **总计** | **22-23** | **170-180** |

**快速演示（半 size）：**
- CPU: 7-10 分钟
- GPU: 45-60 秒

## 常见问题

### Q: 运行 `python main.py` 后什么都没有输出？
**A**: 这可能是绘图后端的问题。尝试：
```python
# 在 main.py 开始处添加：
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
```

### Q: "ModuleNotFoundError: No module named 'torch'"
**A**: PyTorch 未安装。运行：
```bash
pip install torch
```

### Q: GPU 显存不足
**A**: 尝试以下方法：
1. 减少 batch_size
2. 减少 n_samples
3. 使用 CPU（自动降级）

### Q: 结果的相位误差很大（>20°）
**A**: 
- 增加 n_samples（更好的数据）
- 增加 epochs（更好的训练）
- 检查是否有 NaN 或 inf 出现

### Q: 程序在"Tandem 网络训练"卡住了
**A**: 
- 这是正常的，Tandem 训练比较耗时
- 可以按 Ctrl+C 中断，然后尝试 `quick_demo.py`
- 检查 GPU 是否充分利用（nvidia-smi）

## 调试技巧

### 1. 检查 GPU 使用情况

在 Windows PowerShell 或 Linux 终端运行：
```bash
# Linux/Mac
watch -n 1 nvidia-smi

# Windows PowerShell
nvidia-smi -l 1
```

### 2. 检查 PyTorch 是否使用 GPU

在 Python 中运行：
```python
import torch
print(torch.cuda.is_available())  # 应该是 True
print(torch.cuda.get_device_name(0))  # 显示 GPU 名称
```

### 3. 部分运行

如果想单独运行某个模块，在 Python 中：
```python
# 只生成数据
from data_generator import MetasurfaceUnitSimulator
sim = MetasurfaceUnitSimulator()
X, Y = sim.generate_dataset(n_samples=500)

# 只训练正向网络
from forward_model import train_forward_model
model, scaler, _, _ = train_forward_model(X, Y[:, 1], epochs=100)

# 只训练 Tandem
from inverse_design import TandemTrainer
tandem = TandemTrainer(model, scaler)
tandem.train_with_progress(epochs=200)

# 只做设计
from metasurface_design import design_anomalous_refraction_array
design = design_anomalous_refraction_array(...)
```

## 项目工作流

```
┌─────────────────────────────────────┐
│  1. 环境检查 (verify.py)              │
│     - 检查 Python 版本和库            │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  2. 快速演示 (quick_demo.py)         │  ← 可选，用于测试
│     - 3-5 分钟验证代码功能          │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  3. 完整流程 (main.py)               │  ← 最终项目
│     - 10-15 分钟生成完整结果         │
└─────────────────────────────────────┘
               ↓
        查看 results/ 中的图表
```

## 进阶用法

### 加载预训练的模型

```python
import torch
from forward_model import ForwardPredictor
from inverse_design import InverseDesigner

# 加载正向模型
forward_model = ForwardPredictor()
forward_model.load_state_dict(torch.load('results/forward_model_weights.pth'))

# 加载逆向模型
inverse_model = InverseDesigner()
inverse_model.load_state_dict(torch.load('results/inverse_model_weights.pth'))

# 现在可以用预训练的模型做设计
from metasurface_design import design_anomalous_refraction_array
design = design_anomalous_refraction_array(
    inverse_model, scaler_X, forward_model, target_angle_deg=45
)
```

### 批量设计多个折射角

```python
for angle in [20, 25, 30, 35, 40, 45]:
    design = design_anomalous_refraction_array(
        inverse_model, scaler_X, forward_model, target_angle_deg=angle
    )
    print(f"角度 {angle}°: 实现角 {design['actual_angle']:.1f}°")
```

## 得到帮助

1. 查看代码中的详细注释
2. 阅读 README.md
3. 检查输出中的日志信息
4. 运行 `verify.py` 诊断环境问题

---

**祝运行顺利！** 有任何问题欢迎检查代码注释或重新阅读本指南。
