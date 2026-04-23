"""
模块二：正向预测网络训练

基于RCWA物理建模的前向电磁响应预测网络

输入：结构参数 (w, h, p)
输出：电磁响应 (φ, T, R)

关键物理约束：
- 相位φ ∈ [-180°, 180°]，具有周期性
- 透射率 T ∈ [0, 1]，反射率 R ∈ [0, 1]
- 能量守恒：T + R ≤ 1

网络架构：
- 输入层 (3) → 隐层 (256→512→512→256) → 输出层 (3)
- 输出：[φ_deg, T, R]
- 使用物理约束的激活函数
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

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


class PhysicsConstrainedActivation(nn.Module):
    """
    物理约束激活函数

    确保输出满足物理规律：
    - 相位：无约束（网络学习）
    - 透射率：sigmoid ∈ [0, 1]
    - 反射率：sigmoid ∈ [0, 1]
    """

    def forward(self, x):
        """
        参数:
        - x: 网络输出 [batch, 3]

        返回:
        - constrained_output: 物理约束后的输出 [batch, 3]
        """
        phi = x[:, 0]  # 相位，无约束
        T = torch.sigmoid(x[:, 1])  # 透射率 ∈ [0, 1]
        R = torch.sigmoid(x[:, 2])  # 反射率 ∈ [0, 1]

        return torch.stack([phi, T, R], dim=1)


class ForwardPredictor(nn.Module):
    """
    正向预测网络：结构参数 -> 电磁响应

    网络架构：
    输入层 (3) → 隐层 (256→512→512→256) → 输出层 (3)

    输入：归一化的 (w, h, p)
    输出：(φ_deg, T, R)
    """

    def __init__(self, hidden_dims=None):
        """
        初始化网络

        参数:
        - hidden_dims: 隐层维度列表，默认 [256, 512, 512, 256]
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256]

        layers = []
        dims = [3] + hidden_dims  # 输入维度为 3 (w, h, p)

        # 构建隐层
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        # 输出层：预测 [φ, T, R]
        layers.append(nn.Linear(dims[-1], 3))

        self.network = nn.Sequential(*layers)
        self.physics_activation = PhysicsConstrainedActivation()

    def forward(self, x):
        """
        前向传播

        参数:
        - x: 归一化后的 [w, h, p]，shape: [batch_size, 3]

        返回:
        - (phi_deg, T, R): 电磁响应
        """
        out = self.network(x)

        # 应用物理约束
        out = self.physics_activation(out)
        phi_deg, T, R = out[:, 0], out[:, 1], out[:, 2]

        return phi_deg, T, R

    def predict_responses(self, x):
        """
        预测完整电磁响应

        参数:
        - x: 归一化后的结构参数 [batch, 3]

        返回:
        - responses: [phi_deg, T, R] [batch, 3]
        """
        phi_deg, T, R = self.forward(x)
        return torch.stack([phi_deg, T, R], dim=1)


def physics_constrained_loss(pred_phi, pred_T, pred_R, target_phi, target_T, target_R,
                           lambda_energy=1.0, lambda_phase=1.0):
    """
    物理约束损失函数

    参数:
    - pred_phi, pred_T, pred_R: 预测值
    - target_phi, target_T, target_R: 目标值
    - lambda_energy: 能量守恒约束权重
    - lambda_phase: 相位损失权重

    返回:
    - 总损失
    """
    # 相位损失 (考虑周期性)
    phase_diff = torch.abs(pred_phi - target_phi)
    phase_diff = torch.min(phase_diff, 360.0 - phase_diff)
    phase_loss = torch.mean(phase_diff ** 2)

    # 效率损失
    T_loss = torch.mean((pred_T - target_T) ** 2)
    R_loss = torch.mean((pred_R - target_R) ** 2)
    efficiency_loss = T_loss + R_loss

    # 能量守恒约束 (T + R ≤ 1)
    energy_violation = torch.relu(pred_T + pred_R - 1.0)
    energy_loss = lambda_energy * torch.mean(energy_violation ** 2)

    # 总损失
    total_loss = lambda_phase * phase_loss + efficiency_loss + energy_loss

    return total_loss, {
        'phase_loss': phase_loss.item(),
        'efficiency_loss': efficiency_loss.item(),
        'energy_loss': energy_loss.item(),
        'total_loss': total_loss.item()
    }


def train_forward_model(X_data, Y_data, epochs=300, batch_size=64, lr=0.001,
                       val_split=0.2, verbose=True, save_path=None):
    """
    训练正向预测网络

    参数:
    - X_data: 输入参数 (n_samples, 2) 或 (n_samples, 3) 归一化
    - Y_data: 输出响应 (n_samples, 2) [Amplitude, Phase] 或 (n_samples, 3) [phi_deg, T, R] 或 (n_samples,) [Phase]
    - epochs: 训练轮数
    - batch_size: 批量大小
    - lr: 学习率
    - val_split: 验证集比例
    - verbose: 是否打印进度
    - save_path: 模型保存路径

    返回:
    - model: 训练好的模型
    - history: 训练历史
    - scaler_X: 输入标准化器
    """
    if verbose:
        print("🚀 开始训练正向预测网络...")
        print(f"   数据集大小: {len(X_data)} 样本")
        print(f"   输入维度: {X_data.shape[1]}")
        if len(Y_data.shape) == 1:
            print(f"   输出维度: 1 (仅相位)")
        else:
            print(f"   输出维度: {Y_data.shape[1]}")

    # 数据预处理
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_data)

    # 处理不同格式的 Y_data
    if len(Y_data.shape) == 1:
        # 仅相位数据
        phi_targets = Y_data
        T_targets = np.zeros_like(Y_data)
        R_targets = np.zeros_like(Y_data)
    elif Y_data.shape[1] == 2:
        # [Amplitude, Phase] 格式
        phi_targets = Y_data[:, 1]
        T_targets = Y_data[:, 0]
        R_targets = np.zeros_like(T_targets)
    else:
        # [phi_deg, T, R] 格式
        phi_targets = Y_data[:, 0]
        T_targets = Y_data[:, 1]
        R_targets = Y_data[:, 2]

    # 归一化相位到 [-1, 1] (用于训练稳定性)
    phi_scaled = phi_targets / 180.0

    # 合并处理后的目标
    Y_scaled = np.column_stack([phi_scaled, T_targets, R_targets])

    # 训练/验证分割
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_scaled, Y_scaled, test_size=val_split, random_state=42
    )

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = ForwardPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # 训练历史
    history = {
        'train_loss': [], 'val_loss': [],
        'train_phase_loss': [], 'val_phase_loss': [],
        'train_efficiency_loss': [], 'val_efficiency_loss': [],
        'train_energy_loss': [], 'val_energy_loss': []
    }

    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        epoch_train_details = {'phase_loss': 0, 'efficiency_loss': 0, 'energy_loss': 0}

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()

            # 前向传播
            pred_phi_scaled, pred_T, pred_R = model(batch_X)

            # 反归一化预测相位
            pred_phi = pred_phi_scaled * 180.0

            # 目标值
            target_phi = batch_Y[:, 0] * 180.0
            target_T = batch_Y[:, 1]
            target_R = batch_Y[:, 2]

            # 计算损失
            loss, loss_details = physics_constrained_loss(
                pred_phi, pred_T, pred_R,
                target_phi, target_T, target_R
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            for key in epoch_train_details:
                epoch_train_details[key] += loss_details[key]

        # 平均训练损失
        n_batches = len(train_loader)
        epoch_train_loss /= n_batches
        for key in epoch_train_details:
            epoch_train_details[key] /= n_batches

        # 验证阶段
        model.eval()
        with torch.no_grad():
            pred_phi_scaled, pred_T, pred_R = model(X_val_tensor)
            pred_phi = pred_phi_scaled * 180.0

            target_phi = Y_val_tensor[:, 0] * 180.0
            target_T = Y_val_tensor[:, 1]
            target_R = Y_val_tensor[:, 2]

            val_loss, val_details = physics_constrained_loss(
                pred_phi, pred_T, pred_R,
                target_phi, target_T, target_R
            )

        # 记录历史
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss.item())

        for key in ['phase_loss', 'efficiency_loss', 'energy_loss']:
            history[f'train_{key}'].append(epoch_train_details[key])
            history[f'val_{key}'].append(val_details[key])

        # 学习率调度
        scheduler.step(val_loss)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler_X': scaler_X,
                    'epoch': epoch,
                    'val_loss': val_loss.item()
                }, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"🎯 早停于第 {epoch+1} 轮，验证损失未改善")
            break

        # 打印进度
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Phase Loss: {val_details['phase_loss']:.4f} | "
                  f"Energy Loss: {val_details['energy_loss']:.4f}")

    if verbose:
        print(f"✅ 正向网络训练完成！最佳验证损失: {best_val_loss:.4f}")
    return model, history, scaler_X


def validate_forward_model(model, scaler_X, X_test, Y_test, plot_results=True):
    """
    验证正向模型性能

    参数:
    - model: 训练好的模型
    - scaler_X: 输入标准化器
    - X_test: 测试输入
    - Y_test: 测试目标，一维数组 [Phase] 或二维数组 [Amplitude, Phase] 或 [phi_deg, T, R]
    - plot_results: 是否绘制结果

    返回:
    - metrics: 性能指标字典
    """
    model.eval()

    # 数据预处理
    X_scaled = scaler_X.transform(X_test)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_phi_scaled, pred_T, pred_R = model(X_tensor)
        pred_phi = pred_phi_scaled.numpy() * 180.0
        pred_T = pred_T.numpy()
        pred_R = pred_R.numpy()

    # 处理不同格式的 Y_test
    if len(Y_test.shape) == 1:
        # 仅相位数据
        true_phi = Y_test
        true_T = np.zeros_like(Y_test)
        true_R = np.zeros_like(Y_test)
    elif Y_test.shape[1] == 2:
        # [Amplitude, Phase] 格式
        true_phi = Y_test[:, 1]
        true_T = Y_test[:, 0]
        true_R = np.zeros_like(true_T)
    else:
        # [phi_deg, T, R] 格式
        true_phi = Y_test[:, 0]
        true_T = Y_test[:, 1]
        true_R = Y_test[:, 2]

    # 计算指标
    phi_error = np.abs(pred_phi - true_phi)
    phi_error = np.minimum(phi_error, 360 - phi_error)  # 考虑周期性

    metrics = {
        'mae_phi': np.mean(phi_error),
        'rmse_phi': np.sqrt(np.mean(phi_error ** 2)),
        'mae_T': np.mean(np.abs(pred_T - true_T)),
        'mae_R': np.mean(np.abs(pred_R - true_R)),
        'max_phi_error': np.max(phi_error),
        'energy_conservation_violations': np.sum(pred_T + pred_R > 1.0),
        'r2_phi': 1 - np.var(phi_error) / np.var(true_phi),
        'r2_T': 1 - np.var(pred_T - true_T) / np.var(true_T),
        'r2_R': 1 - np.var(pred_R - true_R) / np.var(true_R)
    }

    if plot_results:
        visualize_training(model, scaler_X, X_test[:500], Y_test[:500],
                          title="正向网络验证结果")

    return metrics


def visualize_training(model=None, scaler_X=None, X_sample=None, Y_sample=None,
                      history=None, title="正向网络训练结果"):
    """
    可视化训练结果

    参数:
    - model: 训练好的模型
    - scaler_X: 输入标准化器
    - X_sample: 样本输入
    - Y_sample: 样本目标
    - history: 训练历史
    - title: 图表标题
    """
    if history is not None:
        # 绘制训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{title} - 训练曲线', fontsize=14, fontweight='bold')

        epochs = range(1, len(history['train_loss']) + 1)

        # 总损失
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('训练轮数')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('总损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')

        # 相位损失
        axes[0, 1].plot(epochs, history['train_phase_loss'], 'b-', label='训练相位损失')
        axes[0, 1].plot(epochs, history['val_phase_loss'], 'r-', label='验证相位损失')
        axes[0, 1].set_xlabel('训练轮数')
        axes[0, 1].set_ylabel('相位损失')
        axes[0, 1].set_title('相位损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 效率损失
        axes[1, 0].plot(epochs, history['train_efficiency_loss'], 'b-', label='训练效率损失')
        axes[1, 0].plot(epochs, history['val_efficiency_loss'], 'r-', label='验证效率损失')
        axes[1, 0].set_xlabel('训练轮数')
        axes[1, 0].set_ylabel('效率损失')
        axes[1, 0].set_title('效率损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 能量损失
        axes[1, 1].plot(epochs, history['train_energy_loss'], 'b-', label='训练能量损失')
        axes[1, 1].plot(epochs, history['val_energy_loss'], 'r-', label='验证能量损失')
        axes[1, 1].set_xlabel('训练轮数')
        axes[1, 1].set_ylabel('能量损失')
        axes[1, 1].set_title('能量守恒约束')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    if model is not None and X_sample is not None and Y_sample is not None:
        # 预测 vs 真实值对比
        model.eval()
        X_scaled = scaler_X.transform(X_sample)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred_phi_scaled, pred_T, pred_R = model(X_tensor)
            pred_phi = pred_phi_scaled.numpy() * 180.0
            pred_T = pred_T.numpy()
            pred_R = pred_R.numpy()

        # 处理不同格式的 Y_sample
        if len(Y_sample.shape) == 1:
            # 仅相位数据
            true_phi = Y_sample
            true_T = np.zeros_like(Y_sample)
            true_R = np.zeros_like(Y_sample)
        elif Y_sample.shape[1] == 2:
            # [Amplitude, Phase] 格式
            true_phi = Y_sample[:, 1]
            true_T = Y_sample[:, 0]
            true_R = np.zeros_like(true_T)
        else:
            # [phi_deg, T, R] 格式
            true_phi = Y_sample[:, 0]
            true_T = Y_sample[:, 1]
            true_R = Y_sample[:, 2]

        # 预测误差分析
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{title} - 预测精度分析', fontsize=14, fontweight='bold')

        # 相位预测
        phi_error = np.abs(pred_phi - true_phi)
        phi_error = np.minimum(phi_error, 360 - phi_error)

        axes[0].scatter(true_phi, pred_phi, alpha=0.6, s=20, c=phi_error, cmap='Reds')
        axes[0].plot([-180, 180], [-180, 180], 'k--', linewidth=2)
        axes[0].set_xlabel('真实相位 (度)')
        axes[0].set_ylabel('预测相位 (度)')
        axes[0].set_title(f'相位预测 (MAE: {np.mean(phi_error):.2f}°)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(axes[0].collections[0], ax=axes[0], label='误差 (度)')

        # 透射率预测
        T_error = np.abs(pred_T - true_T)
        axes[1].scatter(true_T, pred_T, alpha=0.6, s=20, c=T_error, cmap='Blues')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2)
        axes[1].set_xlabel('真实透射率')
        axes[1].set_ylabel('预测透射率')
        axes[1].set_title(f'透射率预测 (MAE: {np.mean(T_error):.4f})')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(axes[1].collections[0], ax=axes[1], label='误差')

        # 反射率预测
        R_error = np.abs(pred_R - true_R)
        axes[2].scatter(true_R, pred_R, alpha=0.6, s=20, c=R_error, cmap='Greens')
        axes[2].plot([0, 1], [0, 1], 'k--', linewidth=2)
        axes[2].set_xlabel('真实反射率')
        axes[2].set_ylabel('预测反射率')
        axes[2].set_title(f'反射率预测 (MAE: {np.mean(R_error):.4f})')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(axes[2].collections[0], ax=axes[2], label='误差')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试正向网络模块...")

    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 模拟结构参数 (归一化)
    X_data = np.random.rand(n_samples, 3)

    # 模拟电磁响应
    phi_true = np.random.uniform(-180, 180, n_samples)
    T_true = np.random.uniform(0.1, 0.9, n_samples)
    R_true = np.random.uniform(0.05, 0.8, n_samples)

    # 确保能量守恒
    mask = T_true + R_true > 1.0
    R_true[mask] = 1.0 - T_true[mask]

    Y_data = np.column_stack([phi_true, T_true, R_true])

    print(f"生成测试数据集: {n_samples} 样本")
    print(f"输入形状: {X_data.shape}, 输出形状: {Y_data.shape}")

    # 训练模型
    model, history, scaler_X = train_forward_model(
        X_data, Y_data, epochs=50, batch_size=32, verbose=True
    )

    # 验证模型
    metrics = validate_forward_model(model, scaler_X, X_data[:200], Y_data[:200])

    print("\n📊 验证结果:")
    print(f"相位MAE: {metrics['mae_phi']:.2f}°")
    print(f"透射率MAE: {metrics['mae_T']:.4f}")
    print(f"反射率MAE: {metrics['mae_R']:.4f}")
    print(f"能量守恒违规: {metrics['energy_conservation_violations']}")

    print("✅ 正向网络测试完成！")

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
    model, history, scaler_X = train_forward_model(
        X, Y_phase, epochs=300, batch_size=128, verbose=True
    )
    
    # 可视化训练过程
    visualize_training(history=history)
    
    # 验证模型性能
    validate_forward_model(model, scaler_X, X, Y_phase)
    
    return model, scaler_X


if __name__ == "__main__":
    model, scaler_X = main_forward_training()
    torch.save(model.state_dict(), 'forward_model_weights.pth')
    print("\n✓ 模型权重已保存至 forward_model_weights.pth")