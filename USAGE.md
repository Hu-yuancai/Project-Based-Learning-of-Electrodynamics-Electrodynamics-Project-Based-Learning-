# 📖 超构表面逆向设计系统使用指南

## 🎯 快速导航

| 我想要... | 运行命令 | 预期时间 | 输出位置 |
|:---:|:---:|:---:|:---:|
| **完整项目演示** | `python main.py` | 15-20分钟 | `results/` |
| **快速功能测试** | `python quick_demo.py` | 3-5分钟 | `quick_demo_results/` |
| **环境验证** | `python verify.py` | 10秒 | 控制台 |
| **自定义设计** | 编辑 `main.py` | 视参数而定 | `results/` |

---

## 📂 项目结构详解

```
metasurface-inverse-design/
├── 🏠 main.py                    # 🚀 主程序入口（完整流程）
├── ⚡ quick_demo.py             # 🧪 快速演示（参数缩放版）
├── ✅ verify.py                 # 🔍 环境和依赖验证
├── 🔬 data_generator.py         # 📊 RCWA物理建模数据集生成
├── 🧠 forward_model.py          # 📈 正向神经网络训练
├── 🎯 inverse_design.py         # 🎪 Tandem逆向网络设计
├── 🎨 metasurface_design.py     # 🏗️ 超构表面阵列设计与可视化
├── 📦 requirements.txt          # 📋 Python依赖包列表
├── 📚 README.md                 # 📖 项目完整说明
├── 📋 USAGE.md                  # 📋 本使用指南
├── 🔧 .gitignore               # 🚫 Git忽略文件
└── 📁 results/                  # 📊 输出结果目录（运行时生成）
```

---

## 🚀 运行模式详解

### 模式 1：完整项目运行（推荐）

```bash
python main.py
```

#### 执行流程
1. **🔬 数据生成**：RCWA仿真5000个超构单元样本
2. **📈 正向训练**：学习结构→响应的映射关系
3. **🎯 Tandem训练**：学习响应→结构的逆向设计
4. **🎨 阵列设计**：生成30°异常折射超构表面
5. **📊 可视化**：生成完整的分析图表

#### 参数配置
```python
# 在 main.py 中可修改的关键参数
config = {
    'n_samples': 5000,           # 数据集大小
    'wavelength': 1550e-9,       # 工作波长 (m)
    'forward_epochs': 300,       # 正向网络训练轮数
    'tandem_epochs': 500,        # Tandem网络训练轮数
    'target_angle': 30,          # 设计目标折射角 (度)
}
```

#### 输出文件
```
results/
├── 📈 training_curves.png          # 网络训练曲线
├── 🎯 design_validation.png        # 设计验证结果
├── 🏗️ metasurface_array.png        # 超构表面阵列布局
├── 📊 phase_distribution.png       # 相位分布分析
├── ⚡ efficiency_map.png           # 效率分布图
├── 📋 design_summary.txt           # 设计参数总结
└── 💾 trained_models/              # 保存的模型文件
    ├── forward_model.pth
    └── tandem_model.pth
```

### 模式 2：快速演示模式

```bash
python quick_demo.py
```

#### 特点
- **数据集**：1000个样本（vs 完整版的5000个）
- **训练轮数**：正向50轮，Tandem 100轮（vs 完整版的300+500轮）
- **质量**：足以验证功能，参数稍粗糙
- **时间**：3-5分钟（vs 完整版的15-20分钟）

#### 适用场景
- ✅ 快速验证系统是否正常工作
- ✅ 演示项目功能给他人看
- ✅ 开发过程中的功能测试
- ✅ 有限计算资源的环境

### 模式 3：环境验证模式

```bash
python verify.py
```

#### 检查内容
- ✅ Python版本兼容性
- ✅ 所有依赖包安装状态
- ✅ PyTorch配置和CUDA可用性
- ✅ 各模块导入测试
- ✅ 数据生成功能验证

---

## ⚙️ 高级配置选项

### 自定义物理参数

编辑 `data_generator.py`：

```python
# 修改参数空间范围
param_bounds = {
    'w': [30e-9, 600e-9],    # 纳米柱宽度范围 (m)
    'h': [100e-9, 1000e-9],  # 纳米柱高度范围 (m)
    'p': [300e-9, 1200e-9]   # 周期间距范围 (m)
}

# 修改工作波长
wavelength = 1310e-9  # 改为1310nm通信波段
```

### 调整网络架构

编辑 `forward_model.py` 或 `inverse_design.py`：

```python
# 更深的网络架构
hidden_dims = [512, 1024, 2048, 1024, 512]

# 更宽的网络
hidden_dims = [1024, 1024, 1024, 1024]
```

### 优化训练参数

编辑 `main.py`：

```python
training_config = {
    'batch_size': 128,           # 更大的批量（需要更多显存）
    'learning_rate': 0.0005,     # 更小的学习率（更稳定）
    'patience': 100,             # 更长的早停耐心
    'val_split': 0.3,            # 更大的验证集比例
}
```

---

## 🎨 输出结果解释

### 训练曲线分析

**正向网络训练曲线**：
- **损失下降**：表明网络在学习结构-响应的映射
- **收敛速度**：通常在50-100轮内快速下降，然后缓慢优化
- **过拟合检查**：训练损失和验证损失应该同步下降

**Tandem网络训练曲线**：
- **损失含义**：响应空间的一致性误差
- **收敛特点**：可能有震荡，这是正常的（多解性表现）
- **最终性能**：损失<0.01表示设计精度<5°

### 设计验证图表

**相位分布图**：
- **颜色映射**：从蓝(-180°)到红(+180°)
- **梯度要求**：30°折射需要~π的相位范围
- **均匀性**：理想情况下每个相位值出现次数相似

**效率分布图**：
- **透射率T**：理想>0.8（高透射）
- **反射率R**：理想<0.2（低反射）
- **能量守恒**：T + R ≤ 1（物理约束）

### 阵列布局图

**可视化元素**：
- 🔵 **纳米柱位置**：黑色圆点表示Si纳米柱
- 📏 **尺寸编码**：圆点大小表示纳米柱宽度
- 🎯 **相位目标**：颜色表示目标相位值
- 📐 **坐标系**：x轴为传播方向，y轴为垂直方向

---

## 🔧 故障排除

### 常见问题及解决方案

#### 问题1：内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：
```python
# 减小批量大小
batch_size = 32  # 默认64，改为32或16

# 减小数据集大小
n_samples = 2000  # 默认5000，改为2000
```

#### 问题2：训练不收敛
**检查**：
- 学习率是否过大（>0.01）
- 数据是否正确归一化
- 网络是否过深（梯度消失）

**解决方案**：
```python
learning_rate = 0.001  # 减小学习率
# 或
hidden_dims = [128, 256, 128]  # 简化网络
```

#### 问题3：中文显示乱码
**解决方案**：
- 确保安装了中文字体（SimHei）
- 或在代码中添加：
```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
```

#### 问题4：CUDA不可用
**自动处理**：代码会自动检测并使用CPU
**性能影响**：训练时间约增加3-5倍

---

## 📊 性能基准

### 计算资源需求

| 模式 | CPU时间 | GPU时间 | 内存需求 | 磁盘空间 |
|:---:|:---:|:---:|:---:|:---:|
| 快速演示 | 10-15分钟 | 3-5分钟 | 2GB | 50MB |
| 完整运行 | 45-60分钟 | 15-20分钟 | 4GB | 200MB |
| 大数据集 | 2-3小时 | 30-45分钟 | 8GB | 500MB |

### 硬件推荐配置

**最低配置**（快速演示）：
- CPU: 双核2.5GHz
- 内存: 4GB
- 磁盘: 10GB可用空间

**推荐配置**（完整运行）：
- CPU: 四核3.0GHz以上
- 内存: 8GB以上
- GPU: NVIDIA GTX 1060或以上（可选）
- 磁盘: 50GB可用空间

---

## 🎯 自定义设计任务

### 修改设计目标

编辑 `metasurface_design.py`：

```python
# 改变折射角度
target_angle = 45  # 改为45°折射

# 改变阵列尺寸
array_size = (20, 20)  # 改为20x20阵列

# 改变工作波长
wavelength = 980e-9  # 改为980nm
```

### 添加新的物理约束

在 `forward_model.py` 中修改损失函数：

```python
def physics_constrained_loss(...):
    # 添加新的物理约束
    fabrication_penalty = torch.relu(width - max_width)  # 制造约束
    total_loss = phase_loss + efficiency_loss + 0.1 * fabrication_penalty
```

---

## 📈 扩展开发

### 添加新的数据集类型

1. 创建新的数据生成器类
2. 实现RCWA/FDTD接口
3. 集成到主流程

### 实现多目标优化

1. 修改损失函数为多目标
2. 调整网络输出维度
3. 更新可视化函数

### 支持多波长设计

1. 扩展数据集为多波长
2. 修改网络架构
3. 实现波长插值

---

## 📞 获取帮助

### 调试信息收集

运行以下命令收集系统信息：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python verify.py
```

### 常见问题解答

**Q: 为什么训练这么慢？**
A: 使用CPU而不是GPU。安装CUDA版本的PyTorch可加速3-5倍。

**Q: 结果看起来不对？**
A: 检查数据归一化和参数范围。运行 `python verify.py` 检查环境。

**Q: 如何保存训练好的模型？**
A: 模型自动保存在 `results/trained_models/` 目录中。

---

## 📋 版本兼容性

| Python版本 | PyTorch版本 | 支持状态 |
|:---:|:---:|:---:|
| 3.8+ | 2.0+ | ✅ 完全支持 |
| 3.7 | 1.13+ | ⚠️ 基本支持 |
| 3.6 | 1.10+ | ❌ 不推荐 |

---

**🎉 祝你设计出优秀的超构表面！有问题随时在GitHub Issues中提问。**

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
