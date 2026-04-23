"""
模块二：正向预测网络

输入：归一化后的 (w, p)（柱高 H 固定，不作为输入）
输出：相位的 (sin φ, cos φ) 编码（归一化到单位圆）

sin/cos 双通道 + 输出层归一化，彻底解决 ±180° 跳变和向量退化问题。
Loss = sin/cos MSE + 角度直接惩罚 + 单位模约束
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 字体配置（与 data_generator.py 保持一致）
matplotlib.rcParams['font.family'] = 'sans-serif'
try:
    if Path('C:/Windows/Fonts/simhei.ttf').exists():
        matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/simhei.ttf')
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    elif Path('C:/Windows/Fonts/msyh.ttc').exists():
        matplotlib.font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class ForwardPredictor(nn.Module):
    """
    正向预测网络 v2.0：(w, p) -> (sin φ, cos φ)

    输入维度：2（w_nm, p_nm，经 StandardScaler 归一化）
    输出：sin φ、cos φ，在输出层归一化到单位圆（不再依赖损失函数约束）
    """

    def __init__(self, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256, 128]

        dims = [2] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.05),
            ]
        layers.append(nn.Linear(dims[-1], 2))   # [sin_phi_raw, cos_phi_raw]
        self.network = nn.Sequential(*layers)

        # Kaiming 初始化（适配 LeakyReLU）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        参数：x  shape [batch, 2]，已标准化的 (w, p)
        返回：sin_phi, cos_phi  各 shape [batch]，已归一化到单位圆
        """
        out = self.network(x)
        norm = torch.sqrt(out[:, 0]**2 + out[:, 1]**2 + 1e-8)
        sin_phi = out[:, 0] / norm
        cos_phi = out[:, 1] / norm
        return sin_phi, cos_phi


def phase_loss_fn(pred_sin, pred_cos, true_phi_deg):
    """
    正向网络损失函数 v2.0

    三重约束：
    1. sin/cos MSE（主要梯度信号）
    2. 角度直接惩罚（解决 sin/cos 对小角度误差不敏感的问题）
    3. 单位模约束（防止向量退化，输出层已归一化时权重可小）
    """
    true_rad = torch.deg2rad(true_phi_deg)
    true_sin = torch.sin(true_rad)
    true_cos = torch.cos(true_rad)

    mse = nn.MSELoss()
    loss_sc = mse(pred_sin, true_sin) + mse(pred_cos, true_cos)

    # 角度直接损失（周期性处理）
    pred_rad = torch.atan2(pred_sin, pred_cos)
    diff = torch.abs(pred_rad - true_rad)
    diff = torch.min(diff, 2 * torch.pi - diff)
    loss_phase = torch.mean(diff) * (180.0 / torch.pi)   # 转为度，量级约 0~180

    # 单位模约束（输出层已归一化，此项权重很小）
    norm = torch.sqrt(pred_sin**2 + pred_cos**2)
    loss_norm = torch.mean((norm - 1.0)**2)

    total = loss_sc + 0.1 * loss_phase + 0.05 * loss_norm
    return total, loss_sc.item(), loss_phase.item()


def train_forward_model(X_data, Y_data, epochs=500, batch_size=256, lr=2e-3,
                        val_split=0.2, verbose=True, save_path=None):
    """
    训练正向预测网络

    参数：
      X_data : (N, 2)  [w_nm, p_nm]
      Y_data : (N, 2)  [T, phi_deg]  或  (N,) [phi_deg]
    返回：
      model, history, scaler_X
    """
    if Y_data.ndim == 1:
        phi_deg = Y_data
    else:
        phi_deg = Y_data[:, 1]

    if verbose:
        print("开始训练正向预测网络...")
        print(f"  样本数: {len(X_data)},  输入维度: {X_data.shape[1]}")

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_data)

    X_tr, X_val, phi_tr, phi_val = train_test_split(
        X_scaled, phi_deg, test_size=val_split, random_state=42
    )

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    X_tr_t, phi_tr_t = to_t(X_tr), to_t(phi_tr)
    X_val_t, phi_val_t = to_t(X_val), to_t(phi_val)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, phi_tr_t),
        batch_size=batch_size, shuffle=True
    )

    model = ForwardPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-5
    )

    history = {'train_loss': [], 'val_loss': [], 'val_mae_phi': []}
    best_val = float('inf')
    patience_cnt = 0
    PATIENCE = 100

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for bx, bphi in loader:
            optimizer.zero_grad()
            s, c = model(bx)
            loss, _, _ = phase_loss_fn(s, c, bphi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)

        model.eval()
        with torch.no_grad():
            s_v, c_v = model(X_val_t)
            val_loss, _, _ = phase_loss_fn(s_v, c_v, phi_val_t)
            pred_phi = torch.rad2deg(torch.atan2(s_v, c_v))
            diff = torch.abs(pred_phi - phi_val_t)
            mae = torch.mean(torch.min(diff, 360.0 - diff)).item()

        history['train_loss'].append(ep_loss)
        history['val_loss'].append(val_loss.item())
        history['val_mae_phi'].append(mae)

        scheduler.step(val_loss)

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save({'model_state_dict': model.state_dict(),
                            'scaler_X': scaler_X}, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    print(f"  早停于第 {epoch+1} 轮")
                break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train={ep_loss:.4f} | val={val_loss:.4f} | "
                  f"MAE_phi={mae:.2f}°")

    model.load_state_dict(best_state)
    if verbose:
        print(f"训练完成，最佳验证损失: {best_val:.4f}")
    return model, history, scaler_X


def validate_forward_model(model, scaler_X, X_test, Y_test, plot_results=True):
    """
    验证正向模型性能

    参数：
      X_test : (N, 2)  [w_nm, p_nm]
      Y_test : (N, 2)  [T, phi_deg]  或  (N,) [phi_deg]
    返回：metrics 字典
    """
    if Y_test.ndim == 1:
        phi_true = Y_test
    else:
        phi_true = Y_test[:, 1]

    model.eval()
    X_scaled = scaler_X.transform(X_test)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        s, c = model(X_t)
        pred_phi = torch.rad2deg(torch.atan2(s, c)).numpy()

    diff = np.abs(pred_phi - phi_true)
    diff = np.minimum(diff, 360.0 - diff)

    metrics = {
        'mae_phi':  np.mean(diff),
        'rmse_phi': np.sqrt(np.mean(diff**2)),
        'max_phi':  np.max(diff),
    }

    print(f"\n正向网络验证结果:")
    print(f"  相位 MAE:  {metrics['mae_phi']:.2f}°")
    print(f"  相位 RMSE: {metrics['rmse_phi']:.2f}°")
    print(f"  相位最大误差: {metrics['max_phi']:.2f}°")

    if plot_results:
        _, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(phi_true, pred_phi, alpha=0.4, s=10, c=diff, cmap='Reds')
        ax.plot([-180, 180], [-180, 180], 'k--', lw=1.5)
        ax.set_xlabel('真实相位 (度)')
        ax.set_ylabel('预测相位 (度)')
        ax.set_title(f'相位预测 (MAE={metrics["mae_phi"]:.2f}°)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('forward_validation.png', dpi=150, bbox_inches='tight')
        plt.close()

    return metrics


def visualize_training(history=None, title='正向网络训练结果'):
    """绘制训练曲线（供 main.py 调用）"""
    if history is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13)
    ep = range(1, len(history['train_loss']) + 1)

    axes[0].plot(ep, history['train_loss'], label='训练')
    axes[0].plot(ep, history['val_loss'],   label='验证')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('总损失')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, history['val_mae_phi'], color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (度)')
    axes[1].set_title('验证集相位 MAE')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from data_generator import RigorousMetasurfaceSimulator
    sim = RigorousMetasurfaceSimulator()
    X, Y = sim.generate_dataset(n_samples=2000)
    model, history, scaler_X = train_forward_model(X, Y, epochs=200, verbose=True)
    validate_forward_model(model, scaler_X, X[:500], Y[:500])
