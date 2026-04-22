"""
模块二：正向预测网络训练

输入：几何参数 (L, W)
输出：相位响应

关键物理约束：
- 相位是周期的（2π周期），输出需处理为 sin 和 cos 分量以保持连续性
- 振幅必须满足能量守恒（≤ 1）

这是 AI 避坑案例1：
直接用 MSE(预测相位, 真实相位) 会导致 ±180° 跳变问题
解决方案：用 sin/cos 双通道输出利用三角函数的圆拓扑性质
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===== 配置中文字体 =====
def setup_chinese_font():
    """自动检测并配置 matplotlib 中文字体"""
    try:
        # 优先使用 SimHei（黑体）
        if Path('C:/Windows/Fonts/simhei.ttf').exists():
            matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
            plt.rcParams['font.sans-serif'] = ['SimHei']
        # 备选方案
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    except Exception as e:
        print(f"警告：字体配置失败 {e}，将使用默认字体")

setup_chinese_font()


class ForwardPredictor(nn.Module):
    """
    正向预测网络：几何参数 -> 电磁响应
    
    网络架构：
    输入层 (2) → 隐层 (128→256→256→128) → 输出层 (2)
    
    输出含义：[sin(φ), cos(φ)]
    这种表示方式保留了相位的周期性，避免了±180°的不连续性
    """
    
    def __init__(self, hidden_dims=None):
        """
        初始化网络
        
        参数:
        - hidden_dims: 隐层维度列表，默认 [128, 256, 256, 128]
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]
        
        layers = []
        dims = [2] + hidden_dims  # 输入维度为 2 (L, W)
        
        # 构建隐层
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # 输出层：预测 [sin(φ), cos(φ)]
        layers.append(nn.Linear(dims[-1], 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 归一化后的 [L, W]，shape: [batch_size, 2]
        
        返回:
        - (phase_sin, phase_cos): 相位的三角函数表示
        """
        out = self.network(x)
        
        # 约束输出到 [-1, 1] 范围（sin/cos 的有效范围）
        phase_sin = torch.tanh(out[:, 0])
        phase_cos = torch.tanh(out[:, 1])
        
        return phase_sin, phase_cos
    
    def predict_phase_deg(self, x):
        """
        将网络输出转换为角度
        
        参数:
        - x: 输入数据
        
        返回:
        - 相位（度数），范围 [-180, 180]
        """
        sin_val, cos_val = self.forward(x)
        phase_rad = torch.atan2(sin_val, cos_val)
        return torch.rad2deg(phase_rad)


def train_forward_model(X, Y_phase, epochs=300, batch_size=128, verbose=True):
    """
    训练正向预测网络
    
    参数:
    - X: 几何参数 [n_samples, 2]
    - Y_phase: 相位（度）[n_samples]
    - epochs: 训练轮数
    - batch_size: 批量大小
    - verbose: 是否打印训练日志
    
    返回:
    - model: 训练好的模型
    - scaler_X: 用于标准化输入的缩放器
    - losses: 训练损失曲线
    """
    
    if verbose:
        print("\n" + "="*60)
        print("模块二：正向预测网络训练")
        print("="*60)
    
    # ========== 数据预处理 ==========
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 将相位转换为 sin/cos 分量
    phase_rad = np.deg2rad(Y_phase)
    Y_sin = np.sin(phase_rad)
    Y_cos = np.cos(phase_rad)
    
    # 验证四象限正确性
    Y_sin = Y_sin.reshape(-1, 1)
    Y_cos = Y_cos.reshape(-1, 1)
    
    # 划分数据集 (80% 训练, 20% 验证)
    X_train, X_val, y_sin_train, y_sin_val, y_cos_train, y_cos_val = train_test_split(
        X_scaled, Y_sin, Y_cos, test_size=0.2, random_state=42
    )
    
    # 转换为 PyTorch 张量
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_sin_train_t = torch.tensor(y_sin_train, dtype=torch.float32)
    y_cos_train_t = torch.tensor(y_cos_train, dtype=torch.float32)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_sin_val_t = torch.tensor(y_sin_val, dtype=torch.float32)
    y_cos_val_t = torch.tensor(y_cos_val, dtype=torch.float32)
    
    if verbose:
        print(f"\n数据划分: {len(X_train)} 训练 + {len(X_val)} 验证")
    
    # ========== 模型初始化 ==========
    model = ForwardPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25, verbose=False
    )
    
    # ========== 自定义损失函数 ==========
    def phase_loss(pred_sin, pred_cos, true_sin, true_cos, weight_norm=0.1):
        """
        相位损失函数，包含三个分量：
        1. sin 分量的 MSE
        2. cos 分量的 MSE
        3. 归一化约束：sin^2 + cos^2 = 1
        
        这个设计体现了在网络架构中嵌入物理对称性的思想
        """
        loss_sin = nn.MSELoss()(pred_sin, true_sin)
        loss_cos = nn.MSELoss()(pred_cos, true_cos)
        
        # 圆约束：sin^2 + cos^2 应该接近1（消除神经网络的数值自由度）
        norm_constraint = torch.mean((pred_sin**2 + pred_cos**2 - 1)**2)
        
        return loss_sin + loss_cos + weight_norm * norm_constraint
    
    # ========== 训练循环 ==========
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    if verbose:
        print(f"开始训练... (epochs={epochs}, batch_size={batch_size})")
    
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # 创建 DataLoader 以实现小批量训练
        indices = np.random.permutation(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_sin_batch = y_sin_train_t[batch_idx]
            y_cos_batch = y_cos_train_t[batch_idx]
            
            optimizer.zero_grad()
            
            pred_sin, pred_cos = model(X_batch)
            loss = phase_loss(pred_sin, pred_cos, y_sin_batch, y_cos_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        train_losses.append(epoch_loss / n_batches)
        
        # ===== 验证阶段 =====
        model.eval()
        with torch.no_grad():
            pred_sin_val, pred_cos_val = model(X_val_t)
            val_loss = phase_loss(pred_sin_val, pred_cos_val, y_sin_val_t, y_cos_val_t)
        
        val_losses.append(val_loss.item())
        scheduler.step(val_loss)
        
        # ===== 早停机制 =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  → 早停触发 (epoch {epoch})")
                model.load_state_dict(best_state)
                break
        
        # ===== 日志输出 =====
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"  [Epoch {epoch:3d}] 训练 Loss: {train_losses[-1]:.6f}, "
                  f"验证 Loss: {val_losses[-1]:.6f}")
    
    if verbose:
        print("✓ 正向网络训练完成")
    
    return model, scaler_X, train_losses, val_losses


def visualize_training(train_losses, val_losses):
    """
    可视化训练曲线
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(len(train_losses))
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='训练 Loss')
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='验证 Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('正向网络训练曲线', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('forward_training.png', dpi=150, bbox_inches='tight')
    print("✓ 训练曲线已保存至 forward_training.png")
    plt.show()


def validate_forward_model(model, scaler_X, X, Y_phase):
    """
    验证和评估正向网络的性能
    """
    X_scaled = scaler_X.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        pred_phase_deg = model.predict_phase_deg(X_t)
        pred_phase_deg = pred_phase_deg.numpy().flatten()
    
    # 计算误差（考虑周期性）
    error = np.abs(Y_phase - pred_phase_deg)
    error = np.minimum(error, 360 - error)
    
    print(f"\n正向网络性能评估:")
    print(f"  平均相位误差: {error.mean():.2f}°")
    print(f"  最大相位误差: {error.max():.2f}°")
    print(f"  误差标准差: {error.std():.2f}°")
    print(f"  误差中位数: {np.median(error):.2f}°")
    
    # 绘制预测值 vs 真实值
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 散点图
    ax1 = axes[0]
    scatter = ax1.scatter(Y_phase, pred_phase_deg, c=error, cmap='YlOrRd', 
                         s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.plot([-180, 180], [-180, 180], 'g--', linewidth=2, label='完美预测')
    ax1.set_xlabel('真实相位 (度)', fontsize=12)
    ax1.set_ylabel('预测相位 (度)', fontsize=12)
    ax1.set_title('预测相位 vs 真实相位', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    plt.colorbar(scatter, ax=ax1, label='误差 (度)')
    
    # 误差分布
    ax2 = axes[1]
    ax2.hist(error, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(error.mean(), color='r', linestyle='--', linewidth=2, label=f'均值={error.mean():.2f}°')
    ax2.set_xlabel('相位误差 (度)', fontsize=12)
    ax2.set_ylabel('样本数', fontsize=12)
    ax2.set_title('相位误差分布', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('forward_validation.png', dpi=150, bbox_inches='tight')
    print("✓ 验证图表已保存至 forward_validation.png")
    plt.show()
    
    return error


def main_forward_training(X=None, Y=None):
    """
    正向网络训练的独立测试函数
    """
    from data_generator import MetasurfaceUnitSimulator
    
    if X is None or Y is None:
        print("正向网络需要数据集，正在生成...")
        simulator = MetasurfaceUnitSimulator(wavelength=700e-9)
        X, Y = simulator.generate_dataset(n_samples=5000)
    
    Y_phase = Y[:, 1]  # 提取相位
    
    # 训练正向网络
    model, scaler_X, train_losses, val_losses = train_forward_model(
        X, Y_phase, epochs=300, batch_size=128, verbose=True
    )
    
    # 可视化训练过程
    visualize_training(train_losses, val_losses)
    
    # 验证模型性能
    validate_forward_model(model, scaler_X, X, Y_phase)
    
    return model, scaler_X


if __name__ == "__main__":
    model, scaler_X = main_forward_training()
    torch.save(model.state_dict(), 'forward_model_weights.pth')
    print("\n✓ 模型权重已保存至 forward_model_weights.pth")
