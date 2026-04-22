"""
模块一：数据集构建

超构表面单元的"代理模拟器"
用解析近似 + 经验公式模拟 [L, W] -> [Phase, Amplitude] 的映射

物理基础：矩形介质波导的有效折射率理论
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

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


class MetasurfaceUnitSimulator:
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
