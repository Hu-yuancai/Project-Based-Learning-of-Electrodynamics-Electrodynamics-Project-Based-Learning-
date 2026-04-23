"""
模块三：逆向设计网络（Tandem Network）

这是整个项目的核心创新点。传统逆向网络面临"非唯一性问题"：
同一个目标相位对应多种 (L, W) 组合

解决方案：Tandem Network 架构
目标Phase → [逆向网络] → 预测(L,W) → [冻结的正向网络] → 预测Phase
                                              ↓
                                    Loss = MSE(预测Phase, 目标Phase)

关键洞察：
Loss 不是定义在"结构空间"（L, W 的准确性），
而是定义在"响应空间"（相位的准确性）。
这样就绕过了非唯一性问题。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

# ===== 配置中文字体 =====
try:
    if Path('C:/Windows/Fonts/simhei.ttf').exists():
        matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


class InverseDesigner(nn.Module):
    """
    逆向设计网络：目标相位 -> 几何参数
    
    网络架构：
    输入层 (1) → 隐层 (256→512→512→256) → 输出层 (2)
    
    输入：目标相位（度）
    输出：(L, W) 几何参数（经过 Sigmoid 归一化到 [0, 1]）
    """
    
    def __init__(self, hidden_dims=None, L_range=(60, 240), W_range=(60, 240)):
        """
        初始化网络
        
        参数:
        - hidden_dims: 隐层维度列表，默认 [256, 512, 512, 256]
        - L_range: 长度的物理范围 (nm)
        - W_range: 宽度的物理范围 (nm)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256]
        
        layers = []
        dims = [1] + hidden_dims  # 输入维度为 1 (目标相位)
        
        # 构建隐层
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.1))
        
        # 输出层：(L, W)，用 Sigmoid 约束到 [0, 1]
        layers.append(nn.Linear(dims[-1], 2))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self.L_min, self.L_max = L_range
        self.W_min, self.W_max = W_range
    
    def forward(self, phase_target):
        """
        前向传播
        
        参数:
        - phase_target: 目标相位，形状 [batch_size, 1]
        
        返回:
        - 几何参数 (L, W)，范围 [0, 1]（需要反归一化）
        """
        return self.network(phase_target)
    
    def denormalize_geometry(self, norm):
        """
        反归一化：从 [0, 1] 映射回物理范围
        
        参数:
        - norm: 归一化的参数，形状 [batch_size, 2]
        
        返回:
        - (L, W): 物理尺寸 (nm)
        """
        if isinstance(norm, np.ndarray):
            norm = torch.tensor(norm, dtype=torch.float32)
        
        if norm.dim() == 1:
            norm = norm.unsqueeze(0)
        
        L = norm[:, 0] * (self.L_max - self.L_min) + self.L_min
        W = norm[:, 1] * (self.W_max - self.W_min) + self.W_min
        return L, W


class TandemTrainer:
    """
    Tandem 网络训练器

    核心思想：
    1. 正向网络冻结（weights 不更新）
    2. 逆向网络的目标不是预测准确的 (L, W)，而是预测能产生目标相位的任意 (L, W)
    3. Loss 定义在相位空间，而非几何空间
    """

    def __init__(self, forward_model, scaler_X, L_range=(60, 240), W_range=(60, 240)):
        """
        初始化 Tandem 训练器

        参数:
        - forward_model: 训练好的正向神经网络
        - scaler_X: 用于标准化输入的 StandardScaler 对象
        - L_range: 长度的物理范围 (nm)
        - W_range: 宽度的物理范围 (nm)
        """
        self.forward_model = forward_model
        self.scaler_X = scaler_X
        self.L_min, self.L_max = L_range
        self.W_min, self.W_max = W_range

        # ===== 冻结正向网络 =====
        # 这是关键步骤：我们只想更新逆向网络的权重
        for param in self.forward_model.parameters():
            param.requires_grad = False

        # 设置为评估模式（BatchNorm 等不会更新）
        self.forward_model.eval()

        # 初始化逆向网络
        self.inverse_model = InverseDesigner(L_range=L_range, W_range=W_range)

        print(f"\n正向网络参数已冻结（requires_grad=False）")
        print(f"可训练参数仅为逆向网络的权重")

    def tandem_loss(self, target_phase, predicted_phase, weight_smoothness=0.01):
        """
        Tandem 损失函数

        参数:
        - target_phase: 目标相位（度），形状 [batch_size, 1]
        - predicted_phase: 预测的 (L, W) 经过正向网络后的相位（度）
        - weight_smoothness: 平滑性约束的权重

        返回:
        - 损失值（标量）

        物理含义：
        我们想要预测的结构一旦通过正向网络，其输出相位应该等于目标相位。
        这个目标在结构空间可能有多个解，但在相位空间是唯一的。
        """
        # 计算相位差（考虑周期性的最小夹角）
        diff = torch.abs(target_phase - predicted_phase)
        diff = torch.min(diff, torch.tensor(360.0) - diff)
        
        phase_loss = torch.mean(diff ** 2)
        
        return phase_loss

    def train_with_progress(self, epochs=500, lr=0.001, batch_sizes=None,
                          val_split=0.2, verbose=True):
        """
        带进度条的 Tandem 网络训练

        参数:
        - epochs: 训练轮数
        - lr: 学习率
        - batch_sizes: 不同阶段的批量大小列表，如 [32, 64, 128]
        - val_split: 用于验证的目标相位比例
        - verbose: 是否打印详细日志

        返回:
        - losses: 训练损失曲线
        """
        
        if verbose:
            print("\n" + "="*60)
            print("模块三：Tandem 逆向设计网络训练")
            print("="*60)
        
        optimizer = torch.optim.Adam(self.inverse_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        losses = []
        phase_targets_train = None
        
        # 生成一个固定的验证集用于监控
        phase_targets_val = torch.rand(256, 1) * 360 - 180
        
        if verbose:
            print(f"开始训练 Tandem 网络...")
            print(f"  学习率: {lr}")
            print(f"  总 Epoch: {epochs}")
        
        best_val_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in range(epochs):
            # 生成训练集（每个 epoch 重新采样，增加多样性）
            phase_targets_train = torch.rand(2048, 1) * 360 - 180
            
            # 分小批次训练
            batch_size = 256
            epoch_loss = 0.0
            n_batches = 0
            
            indices = torch.randperm(len(phase_targets_train))
            for i in range(0, len(phase_targets_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                phase_batch = phase_targets_train[batch_idx]
                
                self.inverse_model.train()
                self.forward_model.eval()
                
                optimizer.zero_grad()
                
                # ===== 逆向预测 =====
                pred_geo_norm = self.inverse_model(phase_batch)
                L_pred, W_pred = self.denormalize_geometry(pred_geo_norm)
                
                # ===== 正向验证 =====
                # 将预测的几何参数标准化，输入正向网络（保持梯度链）
                # 注意：正向网络输入是 (w, h, p)，但我们只预测 (L, W)，p 固定为 650 nm
                p_fixed = torch.full_like(L_pred, 650.0)  # 固定周期 650 nm
                geo_pred = torch.stack([L_pred, W_pred, p_fixed], dim=1)  # [batch, 3]，保持梯度
                
                # 使用 scaler 参数进行标准化，但保持 PyTorch 梯度链
                scaler_mean = torch.tensor(self.scaler_X.mean_, dtype=torch.float32, device=geo_pred.device)
                scaler_scale = torch.tensor(self.scaler_X.scale_, dtype=torch.float32, device=geo_pred.device)
                geo_scaled = (geo_pred - scaler_mean) / scaler_scale
                
                # 正向网络的参数已经冻结，所以不会被更新，但梯度可以流向逆向网络
                pred_phi_scaled, _, _ = self.forward_model(geo_scaled)
                pred_phase_deg = pred_phi_scaled * 180.0
                
                # ===== 计算损失 =====
                loss = self.tandem_loss(phase_batch, pred_phase_deg)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_epoch_loss = epoch_loss / n_batches
            losses.append(avg_epoch_loss)
            scheduler.step()
            
            # ===== 验证评估 =====
            self.inverse_model.eval()
            with torch.no_grad():
                pred_geo_norm_val = self.inverse_model(phase_targets_val)
                L_val, W_val = self.denormalize_geometry(pred_geo_norm_val)
                
                p_val_fixed = torch.full_like(L_val, 650.0)
                geo_val_input = torch.stack([L_val, W_val, p_val_fixed], dim=1).numpy()
                geo_val_scaled = torch.tensor(
                    self.scaler_X.transform(geo_val_input),
                    dtype=torch.float32
                )
                
                pred_phi_val_scaled, _, _ = self.forward_model(geo_val_scaled)
                pred_phase_val_deg = pred_phi_val_scaled * 180.0
                
                val_loss = self.tandem_loss(phase_targets_val, pred_phase_val_deg).item()
            
            # ===== 早停机制 =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.inverse_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  → 早停触发 (epoch {epoch})")
                    self.inverse_model.load_state_dict(best_state)
                    break
            
            # ===== 日志输出 =====
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"  [Epoch {epoch:3d}] 训练 Loss: {avg_epoch_loss:.4f}, "
                      f"验证 Loss: {val_loss:.4f}")
        
        if verbose:
            print("✓ Tandem 网络训练完成")
        
        return losses

    def denormalize_geometry(self, norm):
        """
        反归一化：从 [0, 1] 映射回物理范围
        
        参数:
        - norm: 归一化的参数，形状 [batch_size, 2]
        
        返回:
        - (L, W): 物理尺寸 (nm)
        """
        if isinstance(norm, np.ndarray):
            norm = torch.tensor(norm, dtype=torch.float32)
        
        if norm.dim() == 1:
            norm = norm.unsqueeze(0)
        
        L = norm[:, 0] * (self.L_max - self.L_min) + self.L_min
        W = norm[:, 1] * (self.W_max - self.W_min) + self.W_min
        return L, W

    def validate_inverse_design(self, test_phases=None, visualize=True):
        """
        验证逆向设计网络的性能
        
        参数:
        - test_phases: 要测试的相位列表（度）
        - visualize: 是否绘制可视化图表
        
        返回:
        - 验证结果统计
        """
        if test_phases is None:
            test_phases = np.linspace(-180, 180, 37)
        
        results = []
        
        self.inverse_model.eval()
        self.forward_model.eval()
        
        with torch.no_grad():
            for target_phase in test_phases:
                phase_tensor = torch.tensor([[target_phase]], dtype=torch.float32)
                
                # 逆向预测
                pred_geo_norm = self.inverse_model(phase_tensor)
                L_pred, W_pred = self.denormalize_geometry(pred_geo_norm)
                
                # 正向验证
                geo_input = np.array([[L_pred.item(), W_pred.item(), 650.0]])
                geo_scaled = torch.tensor(
                    self.scaler_X.transform(geo_input),
                    dtype=torch.float32
                )
                
                pred_phi_scaled, _, _ = self.forward_model(geo_scaled)
                actual_phase_deg = (pred_phi_scaled * 180.0).item()
                
                error = abs(target_phase - actual_phase_deg)
                error = min(error, 360 - error)
                
                results.append({
                    'target_phase': target_phase,
                    'L': L_pred.item(),
                    'W': W_pred.item(),
                    'actual_phase': actual_phase_deg,
                    'error': error
                })
        
        # 统计信息
        errors = np.array([r['error'] for r in results])
        print(f"\n逆向网络验证结果:")
        print(f"  测试相位数: {len(results)}")
        print(f"  平均误差: {errors.mean():.3f}°")
        print(f"  最大误差: {errors.max():.3f}°")
        print(f"  误差标准差: {errors.std():.3f}°")
        print(f"  中位数误差: {np.median(errors):.3f}°")
        
        if visualize:
            self._visualize_validation(results)
        
        return results
    
    def _visualize_validation(self, results):
        """
        绘制验证结果的可视化图表
        """
        targets = np.array([r['target_phase'] for r in results])
        actuals = np.array([r['actual_phase'] for r in results])
        L_vals = np.array([r['L'] for r in results])
        W_vals = np.array([r['W'] for r in results])
        errors = np.array([r['error'] for r in results])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 目标 vs 实际相位
        ax1 = axes[0, 0]
        ax1.plot(targets, targets, 'g--', linewidth=2, label='完美预测')
        ax1.scatter(targets, actuals, c=errors, cmap='coolwarm', s=100, 
                   edgecolors='black', linewidth=1, alpha=0.8)
        ax1.set_xlabel('目标相位 (度)', fontsize=11)
        ax1.set_ylabel('实际相位 (度)', fontsize=11)
        ax1.set_title('Tandem 逆向设计验证', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 误差分布
        ax2 = axes[0, 1]
        ax2.scatter(targets, errors, c='steelblue', s=100, edgecolors='black', 
                   linewidth=1, alpha=0.8)
        ax2.axhline(y=5, color='r', linestyle='--', linewidth=2, label='5° 阈值')
        ax2.set_xlabel('目标相位 (度)', fontsize=11)
        ax2.set_ylabel('相位误差 (度)', fontsize=11)
        ax2.set_title('设计误差随相位的变化', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3: L 和 W 的设计值
        ax3 = axes[1, 0]
        width = 8
        x_pos = np.arange(len(targets))
        ax3.bar(x_pos - width/2, L_vals, width, label='L (nm)', alpha=0.8, edgecolor='black')
        ax3.bar(x_pos + width/2, W_vals, width, label='W (nm)', alpha=0.8, edgecolor='black')
        ax3.set_xlabel('目标相位索引', fontsize=11)
        ax3.set_ylabel('几何尺寸 (nm)', fontsize=11)
        ax3.set_title('设计的纳米柱尺寸', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 子图4: 误差直方图
        ax4 = axes[1, 1]
        ax4.hist(errors, bins=20, color='salmon', edgecolor='black', alpha=0.7)
        ax4.axvline(errors.mean(), color='r', linestyle='--', linewidth=2, 
                   label=f'均值={errors.mean():.2f}°')
        ax4.axvline(np.median(errors), color='g', linestyle='--', linewidth=2,
                   label=f'中位数={np.median(errors):.2f}°')
        ax4.set_xlabel('相位误差 (度)', fontsize=11)
        ax4.set_ylabel('出现次数', fontsize=11)
        ax4.set_title('误差分布统计', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('tandem_validation.png', dpi=150, bbox_inches='tight')
        print("✓ Tandem 验证图表已保存至 tandem_validation.png")
        plt.show()







def main_tandem_training(forward_model=None, scaler_X=None, X=None, Y=None):
    """
    Tandem 逆向网络训练的独立测试函数
    """
    from data_generator import MetasurfaceUnitSimulator
    from forward_model import train_forward_model
    
    # 如果没有提供前向模型，则训练一个
    if forward_model is None or scaler_X is None:
        print("Tandem 网络需要预训练的正向网络...")
        
        if X is None or Y is None:
            simulator = MetasurfaceUnitSimulator(wavelength=700e-9)
            X, Y = simulator.generate_dataset(n_samples=5000)
        
        Y_phase = Y[:, 1]
        forward_model, history, scaler_X = train_forward_model(X, Y_phase, epochs=300, verbose=False)
    
    # 创建和训练 Tandem 网络
    tandem = TandemTrainer(forward_model, scaler_X)
    losses = tandem.train_with_progress(epochs=500, lr=0.001, verbose=True)
    
    # 可视化训练曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Tandem 网络训练曲线', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('tandem_training.png', dpi=150, bbox_inches='tight')
    print("✓ 训练曲线已保存至 tandem_training.png")
    plt.show()
    
    # 验证性能
    validation_results = tandem.validate_inverse_design(visualize=True)
    
    return tandem


if __name__ == "__main__":
    tandem = main_tandem_training()
    torch.save(tandem.inverse_model.state_dict(), 'inverse_model_weights.pth')
    print("\n✓ 逆向模型权重已保存至 inverse_model_weights.pth")