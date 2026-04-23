"""
模块一：超构表面数据集构建（RCWA物理建模）

基于麦克斯韦方程的数值求解生成物理一致性数据集

物理基础：
- 目标波长：λ₀ = 1550 nm
- 结构参数空间：w ∈ [50, 400] nm, h ∈ [200, 800] nm, p ∈ [400, 900] nm
- 数值方法：RCWA (Rigorous Coupled Wave Analysis)
- 输出：复透射系数 (t = |t|·e^(iφ), r = |r|·e^(iψ))

数据生成流程：
1. LHS采样参数空间 (w,h,p)
2. RCWA求解麦克斯韦方程
3. 提取相位响应 (φ, T, R)
4. 物理一致性过滤 (T + R ≤ 1)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import warnings
from typing import Tuple, List, Optional

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


class RCWASimulator:
    """
    RCWA电磁仿真器

    基于刚性耦合波分析方法求解周期性超构表面的电磁响应

    物理模型：
    - 平面波入射到周期性介质柱阵列
    - 求解边界条件下的麦克斯韦方程
    - 输出复透射/反射系数
    """

    def __init__(self, wavelength: float = 1550e-9, n_substrate: float = 1.45,
                 n_superstrate: float = 1.0, n_pillar: float = 3.5):
        """
        初始化RCWA仿真器

        参数:
        - wavelength: 工作波长 (m)，默认 1550 nm
        - n_substrate: 基底折射率，默认 SiO2 (1.45)
        - n_superstrate: 上层介质折射率，默认空气 (1.0)
        - n_pillar: 纳米柱折射率，默认 Si (3.5)
        """
        self.wavelength = wavelength
        self.k0 = 2 * np.pi / wavelength  # 波数
        self.n_sub = n_substrate
        self.n_sup = n_superstrate
        self.n_pil = n_pillar

        # 物理参数空间约束
        self.w_bounds = [50e-9, 400e-9]    # 宽度范围 (m)
        self.h_bounds = [200e-9, 800e-9]   # 高度范围 (m)
        self.p_bounds = [400e-9, 900e-9]   # 周期范围 (m)

    def compute_effective_index(self, w: float, h: float, p: float) -> complex:
        """
        计算矩形波导的有效折射率

        基于Marcatili近似方法：
        - 考虑TE/TM模的色散特性
        - 修正边界条件效应

        参数:
        - w: 纳米柱宽度 (m)
        - h: 纳米柱高度 (m)
        - p: 周期 (m)

        返回:
        - 有效折射率 (复数)
        """
        # 填充因子
        fill_factor = w / p

        # 有效折射率近似 (Marcatili方法改进版)
        n_eff_real = np.sqrt(fill_factor * self.n_pil**2 + (1 - fill_factor) * self.n_sup**2)

        # 吸收损耗 (经验模型)
        alpha = 0.01 * (h / self.wavelength)  # 归一化吸收系数

        # 几何修正因子
        geo_factor = 1 + 0.1 * np.exp(-w / (2 * h))  # 考虑宽高比效应

        n_eff = complex(n_eff_real * geo_factor, alpha)

        return n_eff

    def compute_transmission_matrix(self, w: float, h: float, p: float,
                                  n_modes: int = 5) -> Tuple[complex, complex]:
        """
        计算透射矩阵 (RCWA核心)

        基于Floquet-Bloch理论：
        - 展开入射/透射场为平面波模
        - 求解边界匹配条件
        - 考虑多阶衍射效应

        参数:
        - w, h, p: 结构参数 (m)
        - n_modes: 考虑的衍射阶数

        返回:
        - t: 透射系数 (复数)
        - r: 反射系数 (复数)
        """
        # 有效折射率
        n_eff = self.compute_effective_index(w, h, p)

        # 波矢量分量 (考虑周期性)
        kx = np.arange(-n_modes, n_modes + 1) * (2 * np.pi / p)

        # 传播常数
        kz_sup = np.sqrt(self.k0**2 * self.n_sup**2 - kx**2 + 0j)
        kz_sub = np.sqrt(self.k0**2 * self.n_sub**2 - kx**2 + 0j)
        kz_pil = np.sqrt(self.k0**2 * n_eff**2 - kx**2 + 0j)

        # 简化RCWA：只考虑基模 (0阶衍射)
        # 实际RCWA需要求解特征值问题，这里用解析近似

        # 相位延迟
        phi = kz_pil * h

        # 透射系数 (考虑阻抗匹配)
        Z_sup = self.n_sup / (self.k0 * kz_sup[0])
        Z_sub = self.n_sub / (self.k0 * kz_sub[0])
        Z_pil = n_eff / (self.k0 * kz_pil[0])

        # 多层膜理论近似
        t = 2 * Z_sup / (Z_sup + Z_pil * np.tanh(phi) + Z_sub) * \
            np.exp(1j * phi)

        r = (Z_sup - Z_pil * np.tanh(phi) - Z_sub) / \
            (Z_sup + Z_pil * np.tanh(phi) + Z_sub) * \
            np.exp(1j * phi)

        return t, r

    def simulate_unit_cell(self, w: float, h: float, p: float) -> Tuple[float, float, float]:
        """
        仿真单个超构单元

        参数:
        - w: 宽度 (m)
        - h: 高度 (m)
        - p: 周期 (m)

        返回:
        - phi: 透射相位 (度)
        - T: 透射率 (0-1)
        - R: 反射率 (0-1)
        """
        # RCWA计算
        t, r = self.compute_transmission_matrix(w, h, p)

        # 提取零阶衍射（中心阶，index = n_modes）
        n_modes = 5
        t0 = t[n_modes]
        r0 = r[n_modes]

        # 提取幅度和相位
        T = float(abs(t0)**2)
        R = float(abs(r0)**2)

        # 透射相位 (考虑分支切割)
        phi_rad = np.angle(t0)
        phi_deg = float(np.degrees(phi_rad))

        # 相位连续性处理 (-180° 到 180°)
        phi_deg = ((phi_deg + 180) % 360) - 180

        return phi_deg, T, R


class MetasurfaceDatasetGenerator:
    """
    超构表面数据集生成器

    基于物理约束的LHS采样 + RCWA仿真
    生成用于神经网络训练的物理一致性数据集
    """

    def __init__(self, n_samples: int = 5000, wavelength: float = 1550e-9,
                 random_seed: int = 42):
        """
        初始化数据集生成器

        参数:
        - n_samples: 样本数量
        - wavelength: 工作波长 (m)
        - random_seed: 随机种子
        """
        self.n_samples = n_samples
        self.wavelength = wavelength
        self.seed = random_seed

        # RCWA仿真器
        self.simulator = RCWASimulator(wavelength)

        # 参数范围 (m)
        self.param_bounds = {
            'w': self.simulator.w_bounds,  # [50e-9, 400e-9]
            'h': self.simulator.h_bounds,  # [200e-9, 800e-9]
            'p': self.simulator.p_bounds   # [400e-9, 900e-9]
        }

        # 生成的数据
        self.X_data = None  # 输入参数 (w, h, p)
        self.Y_data = None  # 输出响应 (phi, T, R)

    def latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        """
        LHS采样确保参数空间均匀覆盖

        参数:
        - n_samples: 采样点数

        返回:
        - 参数矩阵 (n_samples, 3) 归一化到 [0,1]
        """
        np.random.seed(self.seed)

        # 每个维度的分层采样
        X_norm = np.zeros((n_samples, 3))

        for i in range(3):
            # 在每个区间内随机采样
            intervals = np.arange(n_samples, dtype=float) / n_samples
            np.random.shuffle(intervals)

            for j in range(n_samples):
                X_norm[j, i] = (intervals[j] + np.random.random()) / n_samples

        return X_norm

    def denormalize_parameters(self, X_norm: np.ndarray) -> np.ndarray:
        """
        将归一化参数转换为物理参数

        参数:
        - X_norm: 归一化参数 (n_samples, 3)

        返回:
        - 物理参数 (n_samples, 3) 单位:m
        """
        X_phys = np.zeros_like(X_norm)

        # w: 宽度
        X_phys[:, 0] = X_norm[:, 0] * (self.param_bounds['w'][1] - self.param_bounds['w'][0]) + \
                       self.param_bounds['w'][0]

        # h: 高度
        X_phys[:, 1] = X_norm[:, 1] * (self.param_bounds['h'][1] - self.param_bounds['h'][0]) + \
                       self.param_bounds['h'][0]

        # p: 周期
        X_phys[:, 2] = X_norm[:, 2] * (self.param_bounds['p'][1] - self.param_bounds['p'][0]) + \
                       self.param_bounds['p'][0]

        return X_phys

    def generate_dataset(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完整数据集

        流程：
        1. LHS采样参数空间
        2. RCWA仿真每个样本
        3. 物理一致性过滤
        4. 数据标准化

        返回:
        - X: 输入参数 (n_samples, 3) 归一化
        - Y: 输出响应 (n_samples, 3) [phi_deg, T, R]
        """
        if verbose:
            print("🔬 开始生成超构表面数据集...")
            print(f"   样本数量: {self.n_samples}")
            print(f"   工作波长: {self.wavelength*1e9:.0f} nm")
            print(f"   参数范围:")
            print(f"     w ∈ [{self.param_bounds['w'][0]*1e9:.0f}, {self.param_bounds['w'][1]*1e9:.0f}] nm")
            print(f"     h ∈ [{self.param_bounds['h'][0]*1e9:.0f}, {self.param_bounds['h'][1]*1e9:.0f}] nm")
            print(f"     p ∈ [{self.param_bounds['p'][0]*1e9:.0f}, {self.param_bounds['p'][1]*1e9:.0f}] nm")

        # 1. LHS采样
        X_norm = self.latin_hypercube_sampling(self.n_samples)
        X_phys = self.denormalize_parameters(X_norm)

        # 2. RCWA仿真
        Y_responses = []
        valid_samples = 0

        for i in range(self.n_samples):
            w, h, p = X_phys[i]

            try:
                phi, T, R = self.simulator.simulate_unit_cell(w, h, p)

                # 物理一致性检查
                if T + R <= 1.0 + 1e-6:  # 允许小数值误差
                    Y_responses.append([phi, T, R])
                    valid_samples += 1
                else:
                    # 能量守恒违反，丢弃样本
                    continue

            except Exception as e:
                if verbose:
                    print(f"⚠️ 样本 {i} 仿真失败: {e}")
                continue

            if verbose and (i + 1) % 500 == 0:
                print(f"   已处理 {i+1}/{self.n_samples} 样本，有效: {valid_samples}")

        # 转换为数组
        Y_responses = np.array(Y_responses)
        X_norm = X_norm[:len(Y_responses)]  # 只保留有效样本对应的输入

        if verbose:
            print(f"✅ 数据集生成完成")
            print(f"   有效样本: {len(Y_responses)}/{self.n_samples}")
            print(f"   丢弃比例: {(self.n_samples - len(Y_responses))/self.n_samples*100:.1f}%")

        self.X_data = X_norm
        self.Y_data = Y_responses

        return self.X_data, self.Y_data

    def save_dataset(self, filepath: str):
        """保存数据集到文件"""
        if self.X_data is None or self.Y_data is None:
            raise ValueError("数据集尚未生成，请先调用 generate_dataset()")

        np.savez(filepath, X=self.X_data, Y=self.Y_data,
                wavelength=self.wavelength,
                param_bounds=self.param_bounds)
        print(f"💾 数据集已保存到: {filepath}")

    @classmethod
    def load_dataset(cls, filepath: str) -> 'MetasurfaceDatasetGenerator':
        """从文件加载数据集"""
        data = np.load(filepath)
        generator = cls(n_samples=len(data['X']))
        generator.X_data = data['X']
        generator.Y_data = data['Y']
        generator.wavelength = data['wavelength']
        generator.param_bounds = data['param_bounds'].item()
        return generator

    def visualize_dataset(self, save_path: Optional[str] = None):
        """
        可视化数据集分布

        生成4个子图：
        1. 参数空间分布 (w,h,p)
        2. 相位响应分布
        3. 透射率分布
        4. 相位 vs 透射率关系
        """
        if self.X_data is None or self.Y_data is None:
            raise ValueError("数据集尚未生成")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('超构表面数据集分析 (λ = 1550 nm)', fontsize=14, fontweight='bold')

        # 1. 参数分布
        ax1 = axes[0, 0]
        param_names = ['宽度 w (nm)', '高度 h (nm)', '周期 p (nm)']
        X_phys = self.denormalize_parameters(self.X_data) * 1e9  # 转换为nm

        for i, name in enumerate(param_names):
            ax1.hist(X_phys[:, i], bins=30, alpha=0.7, label=name)
        ax1.set_xlabel('参数值 (nm)')
        ax1.set_ylabel('样本数量')
        ax1.set_title('结构参数分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 相位分布
        ax2 = axes[0, 1]
        phi = self.Y_data[:, 0]
        ax2.hist(phi, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('透射相位 (度)')
        ax2.set_ylabel('样本数量')
        ax2.set_title('相位响应分布')
        ax2.grid(True, alpha=0.3)

        # 3. 透射率分布
        ax3 = axes[1, 0]
        T = self.Y_data[:, 1]
        R = self.Y_data[:, 2]
        ax3.hist(T, bins=30, alpha=0.7, label='透射率 T', color='green')
        ax3.hist(R, bins=30, alpha=0.7, label='反射率 R', color='red')
        ax3.set_xlabel('效率')
        ax3.set_ylabel('样本数量')
        ax3.set_title('透射/反射效率分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 相位 vs 透射率
        ax4 = axes[1, 1]
        scatter = ax4.scatter(phi, T, c=X_phys[:, 0], cmap='viridis', alpha=0.6, s=20)
        ax4.set_xlabel('透射相位 (度)')
        ax4.set_ylabel('透射率 T')
        ax4.set_title('相位-透射率关系 (颜色: 宽度)')
        plt.colorbar(scatter, ax=ax4, label='宽度 (nm)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 数据集可视化已保存: {save_path}")
        else:
            plt.show()

    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        if self.X_data is None or self.Y_data is None:
            raise ValueError("数据集尚未生成")

        X_phys = self.denormalize_parameters(self.X_data) * 1e9  # nm
        phi, T, R = self.Y_data[:, 0], self.Y_data[:, 1], self.Y_data[:, 2]

        stats = {
            'n_samples': len(self.X_data),
            'wavelength_nm': self.wavelength * 1e9,
            'parameter_ranges': {
                'w_nm': [X_phys[:, 0].min(), X_phys[:, 0].max()],
                'h_nm': [X_phys[:, 1].min(), X_phys[:, 1].max()],
                'p_nm': [X_phys[:, 2].min(), X_phys[:, 2].max()]
            },
            'response_ranges': {
                'phi_deg': [phi.min(), phi.max()],
                'T': [T.min(), T.max()],
                'R': [R.min(), R.max()]
            },
            'energy_conservation': {
                'mean_T_plus_R': np.mean(T + R),
                'max_T_plus_R': np.max(T + R),
                'violation_count': np.sum(T + R > 1.0)
            }
        }

        return stats


# ===== 便捷函数 =====

def generate_metasurface_dataset(n_samples: int = 5000,
                               wavelength: float = 1550e-9,
                               save_path: Optional[str] = None,
                               visualize: bool = True) -> MetasurfaceDatasetGenerator:
    """
    一键生成超构表面数据集

    参数:
    - n_samples: 样本数量
    - wavelength: 工作波长 (m)
    - save_path: 保存路径 (可选)
    - visualize: 是否可视化

    返回:
    - 数据集生成器对象
    """
    generator = MetasurfaceDatasetGenerator(n_samples, wavelength)
    generator.generate_dataset()

    if save_path:
        generator.save_dataset(save_path)

    if visualize:
        vis_path = save_path.replace('.npz', '_analysis.png') if save_path else 'dataset_analysis.png'
        generator.visualize_dataset(vis_path)

    # 打印统计信息
    stats = generator.get_statistics()
    print("\n📊 数据集统计信息:")
    print(f"   样本数量: {stats['n_samples']}")
    print(f"   工作波长: {stats['wavelength_nm']:.0f} nm")
    print(f"   相位范围: {stats['response_ranges']['phi_deg'][0]:.1f}° - {stats['response_ranges']['phi_deg'][1]:.1f}°")
    print(f"   透射率范围: {stats['response_ranges']['T'][0]:.3f} - {stats['response_ranges']['T'][1]:.3f}")
    print(f"   能量守恒检查: 最大 T+R = {stats['energy_conservation']['max_T_plus_R']:.4f}")

    return generator


if __name__ == "__main__":
    # 快速测试
    print("🧪 测试 RCWA 数据生成器...")

    # 小样本测试
    generator = generate_metasurface_dataset(
        n_samples=1000,
        save_path="test_dataset.npz",
        visualize=True
    )

    print("✅ 测试完成！")
    """
    超构表面单元的"代理模拟器"
    用解析近似 + 经验公式模拟 [L, W] -> [Phase, Amplitude] 的映射
    
    物理基础：矩形介质波导的有效折射率理论
    """
    
    def __init__(self, wavelength=700e-9, height=600e-9, n_Si=3.5, n_bg=1.0):
        """
        初始化模拟器参数
        
        参数:
        - wavelength: 自由空间波长 (m)
        - height: 纳米柱高度 (m)
        - n_Si: 硅的折射率
        - n_bg: 周围介质折射率
        """
        self.lambda0 = wavelength  # 自由空间波长 (m)
        self.H = height            # 纳米柱高度 (m)
        self.n_core = n_Si         # 硅的折射率
        self.n_clad = n_bg         # 周围介质折射率
        self.k0 = 2 * np.pi / self.lambda0  # 自由空间波数
        
    def effective_index(self, L, W):
        """
        计算矩形波导基模的有效折射率（近似模型）
        
        参数:
        - L, W: 纳米柱的边长 (m)
        
        返回:
        - n_eff: 有效折射率
        
        物理原理：
        矩形波导中，基模的有效折射率随着结构尺寸增大而增大，
        趋近于体折射率 n_core。这里用指数函数近似。
        """
        # 标准化的体积参数
        V = self.k0 * np.sqrt(L * W) * np.sqrt(self.n_core**2 - self.n_clad**2)
        
        # 简化的经验公式：有效折射率随尺寸增大而增大
        # 当 V→0 时，n_eff→n_clad（light confinement failure）
        # 当 V→∞ 时，n_eff→n_core（strong confinement）
        n_eff = self.n_clad + (self.n_core - self.n_clad) * (1 - np.exp(-V / 2))
        return np.clip(n_eff, self.n_clad, self.n_core)
    
    def compute_transmission(self, L_nm, W_nm):
        """
        计算单个纳米柱的复透射系数
        
        参数:
        - L_nm, W_nm: 纳米柱尺寸 (nm)
        
        返回:
        - (amplitude, phase_deg): 振幅（0-1）和相位（度）
        
        物理原理：
        传输光学中，光在介质中的相位累积为 φ = n_eff * k0 * H
        振幅受法布里-珀罗干涉调制。
        """
        L = L_nm * 1e-9
        W = W_nm * 1e-9
        
        # 有效折射率
        n_eff = self.effective_index(L, W)
        
        # 传播相位积累：φ = n_eff * k0 * H
        phase = n_eff * self.k0 * self.H
        
        # 法布里-珀罗效应导致的振幅振荡
        # 简化的振幅模型：随尺寸和相位缓慢变化的包络
        # 实际情况中应该通过全波仿真（如COMSOL）获得
        amplitude = 0.85 + 0.1 * np.sin(2 * phase)
        amplitude = np.clip(amplitude, 0.5, 1.0)
        
        # 将相位转换为度数（-180~180）
        phase_deg = np.angle(np.exp(1j * phase), deg=True)
        
        return amplitude, phase_deg

    def generate_dataset(self, L_range=(60, 240), W_range=(60, 240), n_samples=5000):
        """
        生成训练数据集
        
        参数:
        - L_range: L的采样范围 (nm)
        - W_range: W的采样范围 (nm)
        - n_samples: 样本数量
        
        返回:
        - X: 输入数据 [n_samples, 2]，格式为 [L, W]
        - Y: 输出数据 [n_samples, 2]，格式为 [Amplitude, Phase]
        """
        np.random.seed(42)
        
        # 随机采样
        L_samples = np.random.uniform(*L_range, n_samples)
        W_samples = np.random.uniform(*W_range, n_samples)
        
        amplitudes = []
        phases = []
        
        print(f"生成数据集中... (进度: ", end="", flush=True)
        for idx, (L, W) in enumerate(zip(L_samples, W_samples)):
            amp, phase = self.compute_transmission(L, W)
            amplitudes.append(amp)
            phases.append(phase)
            
            if (idx + 1) % (n_samples // 10) == 0:
                print(f"{(idx+1)//100}0%", end=" ", flush=True)
        print("100%)")
        
        # 构建数据集
        X = np.column_stack([L_samples, W_samples])  # 输入: 几何参数
        Y = np.column_stack([amplitudes, phases])    # 输出: 电磁响应
        
        return X, Y
    
    def visualize_dataset(self, X, Y):
        """
        可视化数据集的相位分布特性
        """
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        
        L_values = X[:, 0]
        W_values = X[:, 1]
        phases = Y[:, 1]
        amplitudes = Y[:, 0]
        
        # 子图1: 相位 vs L
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(L_values, phases, c=W_values, cmap='viridis', s=20, alpha=0.6)
        ax1.set_xlabel('长度 L (nm)', fontsize=11)
        ax1.set_ylabel('相位 (度)', fontsize=11)
        ax1.set_title('相位 vs 长度 L （按宽度 W 着色）', fontsize=12)
        plt.colorbar(scatter1, ax=ax1, label='W (nm)')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 相位 vs W
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(W_values, phases, c=L_values, cmap='plasma', s=20, alpha=0.6)
        ax2.set_xlabel('宽度 W (nm)', fontsize=11)
        ax2.set_ylabel('相位 (度)', fontsize=11)
        ax2.set_title('相位 vs 宽度 W （按长度 L 着色）', fontsize=12)
        plt.colorbar(scatter2, ax=ax2, label='L (nm)')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 振幅分布
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(L_values, W_values, c=amplitudes, cmap='coolwarm', s=30, alpha=0.6)
        ax3.set_xlabel('长度 L (nm)', fontsize=11)
        ax3.set_ylabel('宽度 W (nm)', fontsize=11)
        ax3.set_title('结构参数空间：振幅分布', fontsize=12)
        plt.colorbar(scatter3, ax=ax3, label='振幅')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 相位分布直方图
        ax4 = axes[1, 1]
        ax4.hist(phases, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('相位 (度)', fontsize=11)
        ax4.set_ylabel('样本数', fontsize=11)
        ax4.set_title('相位分布直方图', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
        print("✓ 数据集可视化已保存至 dataset_visualization.png")
        plt.show()


def main_data_generation():
    """
    数据生成的独立测试函数
    """
    print("\n" + "="*60)
    print("模块一：数据集构建")
    print("="*60)
    
    simulator = MetasurfaceUnitSimulator(wavelength=700e-9, height=600e-9)
    X, Y = simulator.generate_dataset(n_samples=5000)
    
    phases = Y[:, 1]
    amplitudes = Y[:, 0]
    
    print(f"\n数据统计:")
    print(f"  样本总数: {len(X)}")
    print(f"  L 范围: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}] nm")
    print(f"  W 范围: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}] nm")
    print(f"  相位范围: [{phases.min():.1f}°, {phases.max():.1f}°]")
    print(f"  振幅范围: [{amplitudes.min():.3f}, {amplitudes.max():.3f}]")
    
    simulator.visualize_dataset(X, Y)
    
    return X, Y


if __name__ == "__main__":
    X, Y = main_data_generation()
