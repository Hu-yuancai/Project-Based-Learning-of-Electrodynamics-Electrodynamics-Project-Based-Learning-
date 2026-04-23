"""
模块一：超构表面数据集构建

物理模型：等效介质理论（Rytov 二阶近似）+ Fabry-Perot 共振
         + 模式耦合效率 + 边缘散射损耗
参考文献：Arbabi (2015), Lalanne (1999), Rytov (1956)

结构：λ=1550nm 透射型全介质超构表面
  - 硅纳米柱（n_Si=3.4777）置于 SiO₂ 基底（n_sub=1.444）
  - 柱高固定 H=900nm（可实现 0~2π 完整相位覆盖）
  - 输入参数：宽度 w ∈ [80, 500] nm，周期 p ∈ [400, 1073] nm
  - 输出响应：透射率 T ∈ [0,1]，透射相位 φ ∈ [-180°, 180°]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Tuple, Optional

# 字体配置：优先 Windows 中文字体，回退到 DejaVu Sans
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


class RigorousMetasurfaceSimulator:
    """
    基于 Rytov 二阶近似 + Fabry-Perot 共振的超构表面仿真器

    所有内部计算使用 SI 单位（米），对外接口返回 nm。
    """

    def __init__(self, wavelength: float = 1550e-9):
        self.lambda0 = wavelength
        self.k0 = 2 * np.pi / wavelength

        # 材料折射率（Palik 数据库）
        self.n_Si  = 3.4777   # 硅
        self.n_sub = 1.444    # SiO₂ 基底
        self.n_air = 1.0      # 空气（入射侧）

        # 固定柱高（Arbabi 2015 优化值，保证 0~2π 相位覆盖）
        self.H = 900e-9

        # 几何参数范围（SI）
        self.W_MIN = 80e-9
        self.W_MAX = 500e-9
        self.P_MIN = 400e-9
        self.P_MAX = self.lambda0 / self.n_sub   # ≈1073nm，抑制高阶衍射

    # ------------------------------------------------------------------
    # 核心物理计算
    # ------------------------------------------------------------------

    def effective_index(self, w: float, p: float) -> float:
        """Rytov 二阶近似有效折射率（TE 偏振）"""
        f = w / p
        n_TE2 = f * self.n_Si**2 + (1 - f) * self.n_air**2
        # 二阶修正项，限制幅度避免极端值
        delta = (self.n_Si**2 - self.n_air**2) / 3 * (f * (1 - f))**2
        correction = np.clip(delta * (self.k0 * p)**2, -0.5, 0.5)
        n_eff_sq = n_TE2 + correction
        return float(np.sqrt(max(n_eff_sq, 1.0)))

    def mode_coupling_efficiency(self, w: float, p: float) -> float:
        """
        基模与平面波的耦合效率

        物理依据：
        - 宽度远小于波长时模式束缚弱，耦合效率低
        - 填充因子约 50% 时耦合效率最高
        - 过大宽度激发高阶模，耦合效率下降
        """
        f = w / p
        mode_area_ratio = w / (self.lambda0 / self.n_Si)
        eta_size = np.exp(-((mode_area_ratio - 0.5)**2) / 0.8)
        eta_fill = 1.0 - 0.10 * abs(f - 0.5)
        return float(np.clip(eta_size * eta_fill, 0.88, 0.99))   # 下限提至0.88

    def scattering_loss(self, w: float, p: float = None) -> float:
        """
        边缘散射损耗

        物理依据：侧壁粗糙度和边缘衍射，与周长/面积比相关，
        小尺寸结构损耗更大。
        """
        perimeter_to_area = 4.0 / w   # 正方形截面近似
        scatter = 0.008 * (perimeter_to_area * 1e9) ** 0.5
        return float(np.clip(scatter, 0.001, 0.05))

    def compute_transmission(self, w: float, p: float) -> Tuple[float, float]:
        """
        Fabry-Perot 透射系数（含模式耦合效率和散射损耗）

        参数：w, p 单位为米
        返回：(T, phi_deg)，T ∈ [0,1]，phi_deg ∈ (-180, 180]
        """
        n_eff = self.effective_index(w, p)
        phi_prop = n_eff * self.k0 * self.H

        # 菲涅耳系数（正入射 TE）
        r1 = (self.n_air - n_eff) / (self.n_air + n_eff)
        r2 = (n_eff - self.n_sub) / (n_eff + self.n_sub)
        t1 = 2 * self.n_air / (self.n_air + n_eff)
        t2 = 2 * n_eff / (n_eff + self.n_sub)

        # Fabry-Perot 复透射系数
        denom = 1 - r1 * r2 * np.exp(2j * phi_prop)
        t_fp = t1 * t2 * np.exp(1j * phi_prop) / denom

        # 填充因子增益：间隙允许额外直接透射，相位不变（实数因子）
        f = w / p
        gap_boost = np.sqrt(1.0 + (1.0 - f) * 0.55)

        # 模式耦合效率 + 散射损耗修正
        eta = self.mode_coupling_efficiency(w, p)
        loss = self.scattering_loss(w)
        t_total = t_fp * gap_boost * np.sqrt(eta) * np.sqrt(1.0 - loss)

        T = float(np.clip(abs(t_total)**2, 0.0, 1.0))
        phi_deg = float(np.rad2deg(np.angle(t_total)))
        return T, phi_deg

    def is_valid(self, w: float, p: float) -> bool:
        """物理约束过滤"""
        f = w / p
        return (
            0.1 <= f <= 0.8          # 制造可行性
            and p < self.P_MAX       # 抑制高阶衍射
            and w >= self.W_MIN      # 基模截止
        )

    # ------------------------------------------------------------------
    # 数据集生成
    # ------------------------------------------------------------------

    def generate_dataset(self, n_samples: int = 5000,
                         random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成训练数据集（拉丁超立方采样）

        返回：
          X : shape (N, 2)，列为 [w_nm, p_nm]
          Y : shape (N, 2)，列为 [T, phi_deg]
        """
        np.random.seed(random_seed)

        # 拉丁超立方采样
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=2, seed=random_seed)
            samples = sampler.random(n_samples)
        except ImportError:
            # 降级为分层随机采样
            samples = np.zeros((n_samples, 2))
            for d in range(2):
                perm = np.random.permutation(n_samples)
                samples[:, d] = (perm + np.random.rand(n_samples)) / n_samples

        w_raw = self.W_MIN + samples[:, 0] * (self.W_MAX - self.W_MIN)
        p_raw = self.P_MIN + samples[:, 1] * (self.P_MAX - self.P_MIN)

        valid_w, valid_p, valid_T, valid_phi = [], [], [], []

        for w, p in zip(w_raw, p_raw):
            if not self.is_valid(w, p):
                continue
            T, phi = self.compute_transmission(w, p)
            if np.isnan(T) or np.isnan(phi):
                continue
            valid_w.append(w * 1e9)    # 转 nm
            valid_p.append(p * 1e9)
            valid_T.append(T)
            valid_phi.append(phi)

        X = np.column_stack([valid_w, valid_p])
        Y = np.column_stack([valid_T, valid_phi])

        T_arr   = np.array(valid_T)
        phi_arr = np.array(valid_phi)
        print(f"有效样本: {len(valid_w)}/{n_samples} "
              f"(有效率 {len(valid_w)/n_samples*100:.1f}%)")
        print(f"相位范围: [{phi_arr.min():.1f}°, {phi_arr.max():.1f}°]")
        print(f"透射率范围: [{T_arr.min():.3f}, {T_arr.max():.3f}]  "
              f"均值={T_arr.mean():.3f}")
        print(f"T>0.95 比例: {np.sum(T_arr > 0.95)/len(T_arr)*100:.1f}%  "
              f"(期望 <5%)")

        return X, Y

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def visualize_dataset(self, X: np.ndarray, Y: np.ndarray,
                          save_path: Optional[str] = None):
        """4 子图数据集分析"""
        w_nm = X[:, 0]
        p_nm = X[:, 1]
        T    = Y[:, 0]
        phi  = Y[:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('超构表面数据集分析 (λ = 1550 nm)', fontsize=14, fontweight='bold')

        # 相位 vs w
        sc1 = axes[0, 0].scatter(w_nm, phi, c=p_nm, cmap='viridis', s=15, alpha=0.6)
        axes[0, 0].set_xlabel('宽度 w (nm)')
        axes[0, 0].set_ylabel('透射相位 (度)')
        axes[0, 0].set_title('相位 vs 宽度（颜色：周期 p）')
        plt.colorbar(sc1, ax=axes[0, 0], label='p (nm)')
        axes[0, 0].grid(True, alpha=0.3)

        # 相位分布直方图
        axes[0, 1].hist(phi, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('透射相位 (度)')
        axes[0, 1].set_ylabel('样本数')
        axes[0, 1].set_title('相位分布直方图')
        axes[0, 1].grid(True, alpha=0.3)

        # 透射率分布
        axes[1, 0].hist(T, bins=40, color='coral', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('透射率 T')
        axes[1, 0].set_ylabel('样本数')
        axes[1, 0].set_title('透射率分布')
        axes[1, 0].grid(True, alpha=0.3)

        # 相位 vs 透射率
        sc4 = axes[1, 1].scatter(phi, T, c=w_nm, cmap='plasma', s=15, alpha=0.6)
        axes[1, 1].set_xlabel('透射相位 (度)')
        axes[1, 1].set_ylabel('透射率 T')
        axes[1, 1].set_title('相位-透射率关系（颜色：宽度 w）')
        plt.colorbar(sc4, ax=axes[1, 1], label='w (nm)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"数据集可视化已保存: {save_path}")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    sim = RigorousMetasurfaceSimulator(wavelength=1550e-9)
    X, Y = sim.generate_dataset(n_samples=1000)
    sim.visualize_dataset(X, Y, save_path='dataset_visualization.png')
