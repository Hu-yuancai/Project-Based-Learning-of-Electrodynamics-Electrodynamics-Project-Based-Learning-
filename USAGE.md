# 超构表面逆向设计系统使用指南

## 快速导航

| 我想要... | 命令 | 预期时间 | 输出位置 |
|:---|:---|:---|:---|
| 完整项目运行 | `python main.py` | 20–40 分钟（CPU） | `results/` |
| 快速功能测试 | `python quick_demo.py` | 5–10 分钟 | `quick_demo_results/` |
| 环境验证 | `python verify.py` | <1 分钟 | 控制台 |

---

## 项目结构

```
项目化学习/
├── main.py                  # 主程序入口（完整流程）
├── data_generator.py        # Rytov + FP 物理模型，数据集生成
├── forward_model.py         # 正向网络：(w,p) → (sin φ, cos φ)
├── inverse_design.py        # Tandem 逆向网络：φ_target → (w,p)
├── metasurface_design.py    # 阵列设计与可视化
├── quick_demo.py            # 快速演示脚本
├── verify.py                # 环境检查脚本
├── smoke_test.py            # 端到端冒烟测试
├── requirements.txt         # 依赖包列表
└── results/                 # 输出目录（运行时生成）
```

---

## 运行方式

### 完整流程

```bash
python main.py
```

执行步骤：
1. 数据集构建（Rytov + FP 物理模型，5000 样本）
2. 正向网络训练（300 epochs，早停）
3. Tandem 逆向网络训练（500 epochs，早停）
4. 反常折射阵列设计（目标 30°，21 单元）

### 参数配置

在 `main.py` 中修改 `config` 字典：

```python
config = {
    'output_dir':     'results',
    'n_samples':      5000,    # 数据集大小
    'forward_epochs': 300,     # 正向网络训练轮数
    'tandem_epochs':  500,     # Tandem 网络训练轮数
    'target_angle':   30,      # 目标折射角（度）
    'n_elements':     21,      # 阵列单元数
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
| `dataset.npz` | 原始训练数据 |
| `forward_model_weights.pth` | 正向网络权重 |
| `inverse_model_weights.pth` | 逆向网络权重 |
| `design_result.npz` | 最终设计数据 |

---

## 结果解读

### metasurface_design_results.png（最重要）

- 左上：相位分布对比——绿色为理想相位，橙色为网络实现相位，二者应高度吻合
- 右上：设计误差柱状图——所有单元应在 10° 以下，均值 <5° 为优秀
- 左下：各单元几何参数——w 和 p 的变化应平滑
- 右下：阵列结构示意图——红色虚线为折射光线，角度应与目标一致

### 精度参考

| 指标 | 优秀 | 可接受 | 需调整 |
|:---|:---|:---|:---|
| 正向网络相位 MAE | <2° | <5° | >5° |
| Tandem 平均误差 | <3° | <8° | >8° |
| 阵列平均误差 | <5° | <10° | >10° |
| 折射角偏差 | <1° | <2° | >2° |

---

## 代码示例

### 单独调用各模块

```python
from data_generator import RigorousMetasurfaceSimulator
from forward_model import train_forward_model, validate_forward_model
from inverse_design import TandemTrainer
from metasurface_design import design_anomalous_refraction_array

# 数据生成
sim = RigorousMetasurfaceSimulator(wavelength=1550e-9)
X, Y = sim.generate_dataset(n_samples=5000)

# 正向网络
fwd, history, scaler = train_forward_model(X, Y, epochs=300)

# Tandem 逆向训练
tandem = TandemTrainer(fwd, scaler)
tandem.train_with_progress(epochs=500)

# 阵列设计
result = design_anomalous_refraction_array(
    inverse_model=tandem.inverse_model,
    scaler_X=scaler,
    forward_model=fwd,
    wavelength=1550e-9,
    period=600e-9,
    target_angle_deg=30,
    n_elements=21,
)
print(f"实现折射角: {result['actual_angle']:.1f}°")
```

### 加载已训练的模型

```python
import torch
from forward_model import ForwardPredictor
from inverse_design import InverseDesigner

fwd = ForwardPredictor()
checkpoint = torch.load('results/forward_model_weights.pth')
fwd.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler_X']

inv = InverseDesigner()
inv.load_state_dict(torch.load('results/inverse_model_weights.pth'))
```

### 批量设计多个折射角

```python
for angle in [20, 25, 30, 35, 40]:
    res = design_anomalous_refraction_array(
        tandem.inverse_model, scaler, fwd,
        target_angle_deg=angle, n_elements=21
    )
    print(f"{angle}° → 实现 {res['actual_angle']:.1f}°，误差 {res['design_errors'].mean():.2f}°")
```

---

## 故障排除

### 训练不收敛（Loss 不下降）

- 检查数据集相位范围是否覆盖 [-180°, 180°]
- 尝试减小学习率（在 `forward_model.py` 中 `lr=1e-3`）
- 增加数据集大小（`n_samples=8000`）

### 相位误差偏大（>10°）

- 增加训练轮数：`forward_epochs=500`，`tandem_epochs=800`
- 增加数据集：`n_samples=8000`

### 内存不足

- 减小 `batch_size`（在 `train_forward_model` 中默认 256，改为 128）
- 减小 `n_samples`

### 中文字体警告

代码已自动检测 Windows 中文字体（SimHei/微软雅黑），若找不到则回退到 DejaVu Sans，不影响运行。

---

## 性能参考

| 硬件 | 数据生成 | 正向训练 | Tandem 训练 | 总计 |
|:---|:---|:---|:---|:---|
| CPU（i7） | ~30 s | ~8 min | ~12 min | ~21 min |
| GPU（RTX 3060） | ~5 s | ~40 s | ~2 min | ~3 min |

---


