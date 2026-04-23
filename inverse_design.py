"""
模块三：逆向设计网络（Tandem Network）

核心思想：
  目标相位 → 逆向网络 → (w, p) → [冻结的正向网络] → 预测相位
                                                         ↑
                                               Loss 定义在相位空间

这样绕过了"一个相位对应多个结构"的非唯一性问题。
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 字体配置（与其他模块保持一致）
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


class InverseDesigner(nn.Module):
    """
    逆向设计网络 v2.0：φ_target (归一化) -> (w_norm, p_norm)

    输入：1 维，目标相位归一化到 [-1, 1]（即 phi_deg / 180）
    输出：2 维，Sigmoid 约束到 [0,1]，需反归一化到物理范围
    """

    def __init__(self, hidden_dims=None, w_range=(80.0, 500.0), p_range=(400.0, 1073.0)):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256, 128]

        self.w_min, self.w_max = w_range
        self.p_min, self.p_max = p_range

        dims = [1] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.05),
            ]
        layers += [nn.Linear(dims[-1], 2), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: [batch, 1]，归一化相位 phi_deg / 180"""
        return self.network(x)

    def denormalize(self, norm):
        """[0,1] -> 物理范围 (nm)，返回 (w_nm, p_nm) 各 shape [batch]"""
        w = norm[:, 0] * (self.w_max - self.w_min) + self.w_min
        p = norm[:, 1] * (self.p_max - self.p_min) + self.p_min
        return w, p


class TandemTrainer:
    """
    Tandem 网络训练器

    正向网络权重冻结，只训练逆向网络。
    Loss 定义在相位响应空间（Huber loss），而非几何参数空间。
    """

    def __init__(self, forward_model, scaler_X,
                 w_range=(80.0, 500.0), p_range=(400.0, 1073.0)):
        self.forward_model = forward_model
        self.scaler_X = scaler_X

        for param in self.forward_model.parameters():
            param.requires_grad = False
        self.forward_model.eval()

        self.inverse_model = InverseDesigner(w_range=w_range, p_range=p_range)

        self._scaler_mean  = torch.tensor(scaler_X.mean_,  dtype=torch.float32)
        self._scaler_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32)

        print("正向网络已冻结，仅训练逆向网络")

    def _scale(self, w_nm, p_nm):
        """将 (w_nm, p_nm) 标准化，保持 PyTorch 梯度链"""
        geo = torch.stack([w_nm, p_nm], dim=1)
        return (geo - self._scaler_mean) / self._scaler_scale

    @staticmethod
    def _phi_to_input(phi_deg):
        """角度 -> 归一化输入 [batch, 1]，范围 [-1, 1]"""
        return (phi_deg / 180.0)

    @staticmethod
    def _huber_phase_loss(pred_sin, pred_cos, target_sin, target_cos,
                          delta_deg=10.0):
        """
        Huber loss 在相位空间

        小误差（<delta）用二次，大误差用线性，对离群点更鲁棒。
        """
        pred_rad   = torch.atan2(pred_sin,   pred_cos)
        target_rad = torch.atan2(target_sin.squeeze(), target_cos.squeeze())
        diff = torch.abs(pred_rad - target_rad)
        diff = torch.min(diff, 2 * torch.pi - diff)   # 周期性

        delta = torch.deg2rad(torch.tensor(delta_deg))
        loss = torch.where(
            diff < delta,
            0.5 * diff**2 / delta,
            diff - 0.5 * delta,
        )
        return loss.mean()

    def train_with_progress(self, epochs=800, lr=1e-3, batch_size=512,
                            noise_std=1.5, verbose=True):
        """
        训练 Tandem 网络

        每 epoch 随机采样目标相位（覆盖全相位空间），
        并对输入加轻微噪声增强泛化。

        参数：
          noise_std : 训练时对目标相位加的高斯噪声标准差（度）
        返回：losses 列表
        """
        optimizer = torch.optim.Adam(self.inverse_model.parameters(),
                                     lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=40, min_lr=1e-5
        )

        # 固定验证集（无噪声）
        val_phi = torch.rand(512, 1) * 360 - 180
        val_sin = torch.sin(torch.deg2rad(val_phi))
        val_cos = torch.cos(torch.deg2rad(val_phi))
        val_inp = self._phi_to_input(val_phi)

        losses = []
        best_val = float('inf')
        best_state = None
        patience_cnt = 0
        PATIENCE = 150

        if verbose:
            print(f"\nTandem 网络训练开始 (epochs={epochs}, lr={lr}, noise={noise_std}°)")

        for epoch in range(epochs):
            train_phi = torch.rand(2048, 1) * 360 - 180
            # 加噪声增强泛化
            train_phi_noisy = train_phi + noise_std * torch.randn_like(train_phi)
            train_phi_noisy = torch.clamp(train_phi_noisy, -180, 180)

            train_sin = torch.sin(torch.deg2rad(train_phi))
            train_cos = torch.cos(torch.deg2rad(train_phi))
            train_inp = self._phi_to_input(train_phi_noisy)

            self.inverse_model.train()
            ep_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_phi), batch_size):
                inp_b  = train_inp[i:i + batch_size]
                sin_b  = train_sin[i:i + batch_size]
                cos_b  = train_cos[i:i + batch_size]

                optimizer.zero_grad()

                geo_norm = self.inverse_model(inp_b)
                w_pred, p_pred = self.inverse_model.denormalize(geo_norm)

                geo_scaled = self._scale(w_pred, p_pred)
                pred_sin, pred_cos = self.forward_model(geo_scaled)

                loss = self._huber_phase_loss(pred_sin, pred_cos, sin_b, cos_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(), 1.0)
                optimizer.step()

                ep_loss += loss.item()
                n_batches += 1

            ep_loss /= n_batches
            losses.append(ep_loss)

            # 验证
            self.inverse_model.eval()
            with torch.no_grad():
                geo_norm_v = self.inverse_model(val_inp)
                w_v, p_v   = self.inverse_model.denormalize(geo_norm_v)
                geo_sc_v   = self._scale(w_v, p_v)
                ps_v, pc_v = self.forward_model(geo_sc_v)
                v_loss = self._huber_phase_loss(ps_v, pc_v, val_sin, val_cos).item()

            scheduler.step(v_loss)

            if v_loss < best_val:
                best_val = v_loss
                patience_cnt = 0
                best_state = {k: v.clone() for k, v in self.inverse_model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    if verbose:
                        print(f"  早停于第 {epoch+1} 轮")
                    break

            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"  [Epoch {epoch:3d}] train={ep_loss:.4f}  val={v_loss:.4f}")

        if best_state is not None:
            self.inverse_model.load_state_dict(best_state)
        if verbose:
            print(f"训练完成，最佳验证损失: {best_val:.4f}")
        return losses

    def validate_inverse_design(self, test_phases=None, visualize=True):
        """
        验证逆向设计性能

        参数：test_phases  角度列表（度），默认 -180~180 均匀 37 点
        返回：results 列表
        """
        if test_phases is None:
            test_phases = np.linspace(-180, 180, 37)

        self.inverse_model.eval()
        self.forward_model.eval()
        results = []

        with torch.no_grad():
            for phi_t in test_phases:
                phi_tensor = torch.tensor([[phi_t]], dtype=torch.float32)
                inp = self._phi_to_input(phi_tensor)

                geo_norm = self.inverse_model(inp)
                w_nm, p_nm = self.inverse_model.denormalize(geo_norm)

                geo_scaled = self._scale(w_nm, p_nm)
                ps, pc = self.forward_model(geo_scaled)
                actual_phi = float(torch.rad2deg(torch.atan2(ps, pc)).item())

                err = abs(phi_t - actual_phi)
                err = min(err, 360 - err)
                results.append({
                    'target': phi_t,
                    'w_nm':   w_nm.item(),
                    'p_nm':   p_nm.item(),
                    'actual': actual_phi,
                    'error':  err,
                })

        errors = np.array([r['error'] for r in results])
        print(f"\n逆向网络验证:")
        print(f"  测试点数: {len(results)}")
        print(f"  平均误差: {errors.mean():.2f}°")
        print(f"  最大误差: {errors.max():.2f}°")
        print(f"  中位数误差: {np.median(errors):.2f}°")

        if visualize:
            self._plot_validation(results)
        return results

    def _plot_validation(self, results):
        targets = np.array([r['target'] for r in results])
        actuals = np.array([r['actual'] for r in results])
        errors  = np.array([r['error']  for r in results])
        w_vals  = np.array([r['w_nm']   for r in results])
        p_vals  = np.array([r['p_nm']   for r in results])

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle('Tandem 逆向设计验证', fontsize=13)

        axes[0, 0].plot(targets, targets, 'g--', lw=1.5, label='理想')
        axes[0, 0].scatter(targets, actuals, c=errors, cmap='coolwarm', s=80,
                           edgecolors='k', lw=0.8)
        axes[0, 0].set_xlabel('目标相位 (度)')
        axes[0, 0].set_ylabel('实现相位 (度)')
        axes[0, 0].set_title('目标 vs 实现相位')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(targets, errors, c='steelblue', s=60, edgecolors='k', lw=0.8)
        axes[0, 1].axhline(5, color='r', ls='--', lw=1.5, label='5° 阈值')
        axes[0, 1].set_xlabel('目标相位 (度)')
        axes[0, 1].set_ylabel('误差 (度)')
        axes[0, 1].set_title('设计误差')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].scatter(targets, w_vals, label='w (nm)', s=50, alpha=0.8)
        axes[1, 0].scatter(targets, p_vals, label='p (nm)', s=50, alpha=0.8, marker='s')
        axes[1, 0].set_xlabel('目标相位 (度)')
        axes[1, 0].set_ylabel('尺寸 (nm)')
        axes[1, 0].set_title('设计几何参数')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(errors, bins=20, color='salmon', edgecolor='k', alpha=0.8)
        axes[1, 1].axvline(errors.mean(), color='r', ls='--', lw=1.5,
                           label=f'均值={errors.mean():.2f}°')
        axes[1, 1].set_xlabel('误差 (度)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_title('误差分布')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('tandem_validation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Tandem 验证图已保存: tandem_validation.png")


if __name__ == "__main__":
    from data_generator import RigorousMetasurfaceSimulator
    from forward_model import train_forward_model

    sim = RigorousMetasurfaceSimulator()
    X, Y = sim.generate_dataset(n_samples=3000)
    fwd, _, scaler = train_forward_model(X, Y, epochs=200, verbose=True)

    tandem = TandemTrainer(fwd, scaler)
    tandem.train_with_progress(epochs=300, verbose=True)
    tandem.validate_inverse_design()
