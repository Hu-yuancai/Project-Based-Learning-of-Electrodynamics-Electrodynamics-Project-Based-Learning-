"""
模块四：反常折射阵列设计与可视化

物理原理：
根据广义斯涅尔定律，对于正入射，折射角 θ_t 所需的相位梯度为：
    dΦ/dx = (2π/λ₀) sin(θ_t)

对于离散阵列（间距 P），第 n 个单元的理想相位为：
    Φ_n = Φ_0 + n · (2π P / λ₀) sin(θ_t)  (mod 360°)

本模块实现：
1. 根据目标折射角计算理想相位分布
2. 用逆向网络为每个相位生成纳米柱尺寸
3. 用正向网络验证实现的相位
4. 多种方式可视化最终的超构表面阵列
"""

import numpy as np
import torch
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
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.patches import Arc
import matplotlib.patches as mpatches


def design_anomalous_refraction_array(inverse_model, scaler_X, forward_model,
                                       wavelength=700e-9, period=350e-9,
                                       target_angle_deg=30, n_elements=21):
    """
    设计实现反常折射的超构表面阵列
    
    参数:
    - inverse_model: 训练好的逆向网络
    - scaler_X: 输入标准化器
    - forward_model: 训练好的正向网络
    - wavelength: 工作波长 (m)，默认 700 nm
    - period: 单元周期 (m)，默认 350 nm（半波长）
    - target_angle_deg: 目标折射角 (度)，默认 30°
    - n_elements: 阵列单元数量，默认 21
    
    返回:
    - designer_result: 包含设计结果的字典
    """
    
    print("\n" + "="*60)
    print("模块四：反常折射阵列设计")
    print("="*60)
    
    # ===== 第1步：计算理想相位分布 =====
    k0 = 2 * np.pi / wavelength
    phase_gradient = k0 * np.sin(np.deg2rad(target_angle_deg))
    
    positions = np.arange(n_elements) * period
    ideal_phases_rad = phase_gradient * positions
    ideal_phases_deg = np.rad2deg(ideal_phases_rad) % 360
    
    # 映射到 [-180, 180] 范围
    ideal_phases_deg = (ideal_phases_deg + 180) % 360 - 180
    
    print(f"\n[1/3] 计算理想相位分布")
    print(f"  工作波长: {wavelength*1e9:.1f} nm")
    print(f"  单元周期: {period*1e9:.1f} nm")
    print(f"  目标折射角: {target_angle_deg:.1f}°")
    print(f"  相位梯度: {phase_gradient:.4f} rad/m")
    print(f"  阵列单元数: {n_elements}")
    
    # ===== 第2步：逆向设计几何参数 =====
    designed_geometries = []
    predicted_phases = []
    design_errors = []
    
    print(f"\n[2/3] 逆向设计几何参数...")
    
    inverse_model.eval()
    forward_model.eval()
    
    with torch.no_grad():
        for idx, phase_target in enumerate(ideal_phases_deg):
            # 逆向网络预测
            phase_tensor = torch.tensor([[phase_target]], dtype=torch.float32)
            geo_norm = inverse_model(phase_tensor)
            L_pred, W_pred = inverse_model.denormalize_geometry(geo_norm)
            L_nm = L_pred.item()
            W_nm = W_pred.item()
            
            designed_geometries.append((L_nm, W_nm))
            
            # 正向网络验证
            geo_input = np.array([[L_nm, W_nm]])
            geo_scaled = torch.tensor(
                scaler_X.transform(geo_input),
                dtype=torch.float32
            )
            
            sin_val, cos_val = forward_model(geo_scaled)
            actual_phase_rad = torch.atan2(sin_val, cos_val).item()
            actual_phase_deg = np.rad2deg(actual_phase_rad)
            
            predicted_phases.append(actual_phase_deg)
            
            # 计算误差（考虑周期性）
            error = abs(phase_target - actual_phase_deg)
            error = min(error, 360 - error)
            design_errors.append(error)
            
            if (idx + 1) % max(1, n_elements // 5) == 0:
                print(f"  已处理 {idx+1}/{n_elements} 单元")
    
    print(f"✓ 逆向设计完成")
    
    # ===== 第3步：性能评估 =====
    design_errors = np.array(design_errors)
    print(f"\n[3/3] 设计性能评估")
    print(f"  平均相位误差: {design_errors.mean():.2f}°")
    print(f"  最大相位误差: {design_errors.max():.2f}°")
    print(f"  误差标准差: {design_errors.std():.2f}°")
    
    # 验证等效折射角
    actual_angle, gradient_measured = verify_refraction_angle(
        predicted_phases, period, wavelength
    )
    print(f"\n  理想折射角: {target_angle_deg:.1f}°")
    print(f"  实现折射角: {actual_angle:.1f}° (误差: {abs(actual_angle - target_angle_deg):.1f}°)")
    
    return {
        'positions': positions,
        'ideal_phases': ideal_phases_deg,
        'designed_geometries': designed_geometries,
        'predicted_phases': predicted_phases,
        'design_errors': design_errors,
        'target_angle': target_angle_deg,
        'actual_angle': actual_angle,
        'wavelength': wavelength,
        'period': period,
        'n_elements': n_elements
    }


def verify_refraction_angle(phase_distribution, period, wavelength):
    """
    根据设计的相位分布验证等效折射角
    
    原理：
    从相位梯度计算折射角: sin(θ) = (dΦ/dx) * λ₀ / (2π)
    
    参数:
    - phase_distribution: 相位数组（度）
    - period: 单元间距 (m)
    - wavelength: 工作波长 (m)
    
    返回:
    - (angle, gradient): 折射角 (度) 和相位梯度 (rad/m)
    """
    positions = np.arange(len(phase_distribution)) * period
    
    # 相位转换为弧度
    phase_rad = np.deg2rad(phase_distribution)
    
    # 展开相位（处理周期性跳变）
    unwrapped = np.unwrap(phase_rad)
    
    # 线性拟合求斜率（相位梯度）
    coeffs = np.polyfit(positions, unwrapped, 1)
    gradient = coeffs[0]
    
    # 计算等效折射角
    k0 = 2 * np.pi / wavelength
    sin_theta = gradient / k0
    
    # 处理超出范围的情况
    if abs(sin_theta) <= 1:
        angle = np.rad2deg(np.arcsin(sin_theta))
    else:
        angle = np.nan  # 传播波条件不满足
    
    return angle, gradient


def visualize_metasurface_comprehensive(design_result, save_prefix='metasurface'):
    """
    全面可视化超构表面阵列的设计结果
    
    包含 4 个子图：
    1. 相位分布对比
    2. 相位误差分析
    3. 纳米柱尺寸分布
    4. 超构表面阵列结构示意图
    """
    
    positions = design_result['positions'] * 1e9  # 转换为 nm
    ideal = design_result['ideal_phases']
    predicted = design_result['predicted_phases']
    errors = design_result['design_errors']
    geometries = design_result['designed_geometries']
    target_angle = design_result['target_angle']
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # ===== 子图1: 相位分布对比 =====
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(positions, ideal, 'o-', label='理想相位', markersize=10, 
            linewidth=2.5, color='forestgreen', markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax1.plot(positions, predicted, 's--', label='网络实现相位', markersize=8,
            linewidth=2, color='darkorange', markeredgecolor='darkred', markeredgewidth=1.5)
    ax1.fill_between(positions, ideal, predicted, alpha=0.2, color='gray')
    ax1.set_xlabel('位置 (nm)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('相位 (度)', fontsize=13, fontweight='bold')
    ax1.set_title(f'相位分布对比 (目标折射角 = {target_angle}°，实现角 = {design_result["actual_angle"]:.1f}°)',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_xlim([positions.min()-30, positions.max()+30])
    
    # ===== 子图2: 相位误差分析 =====
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.RdYlGn_r(errors / errors.max())
    bars = ax2.bar(positions, errors, width=np.diff(positions).mean()*0.8, 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=errors.mean(), color='r', linestyle='--', linewidth=2.5,
               label=f'平均误差={errors.mean():.2f}°')
    ax2.axhline(y=10, color='orange', linestyle=':', linewidth=2,
               label='10° 阈值')
    ax2.set_xlabel('位置 (nm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('相位误差 (度)', fontsize=12, fontweight='bold')
    ax2.set_title('设计误差分布', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([positions.min()-30, positions.max()+30])
    
    # ===== 子图3: 纳米柱尺寸分布 =====
    ax3 = fig.add_subplot(gs[1, 1])
    L_values = np.array([g[0] for g in geometries])
    W_values = np.array([g[1] for g in geometries])
    width = positions[1] - positions[0] if len(positions) > 1 else 30
    width = width * 0.35
    
    ax3.bar(positions - width/2, L_values, width, label='长度 L', 
           alpha=0.8, color='steelblue', edgecolor='darkblue', linewidth=1.5)
    ax3.bar(positions + width/2, W_values, width, label='宽度 W',
           alpha=0.8, color='coral', edgecolor='darkred', linewidth=1.5)
    ax3.set_xlabel('位置 (nm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('尺寸 (nm)', fontsize=12, fontweight='bold')
    ax3.set_title('各单元纳米柱几何尺寸', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim([positions.min()-30, positions.max()+30])
    
    # ===== 子图4: 超构表面阵列结构示意图（侧视图）=====
    ax4 = fig.add_subplot(gs[2, :])
    
    # 设置坐标系
    ax4.set_xlim([positions.min()-100, positions.max()+100])
    ax4.set_ylim([-300, 250])
    
    # 绘制基底
    substrate_y = -50
    ax4.plot([positions.min()-100, positions.max()+100], [substrate_y, substrate_y],
            color='gray', linewidth=4)
    ax4.fill_between([positions.min()-100, positions.max()+100], substrate_y, -300,
                     color='lightgray', alpha=0.5, edgecolor='none')
    ax4.text(positions.min()-80, -200, '玻璃基底\n(n=1.0)',
            fontsize=10, ha='left', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 绘制纳米柱（柱状图示）
    H_nm = 200  # 纳米柱高度 (nm)
    for i, (pos, (L, W)) in enumerate(zip(positions, geometries)):
        # 绘制纳米柱 (矩形)
        rect = Rectangle((pos - L/2, substrate_y), L, H_nm,
                        facecolor='royalblue', edgecolor='darkblue', linewidth=2, alpha=0.8)
        ax4.add_patch(rect)
        
        # 标注单元索引
        ax4.text(pos, substrate_y - 30, f'{i}', ha='center', va='top',
                fontsize=10, fontweight='bold')
        
        # 在顶部标注相位
        phase_text = f'{predicted[i]:.0f}°'
        color = 'darkred' if errors[i] > 10 else 'darkgreen'
        ax4.text(pos, substrate_y + H_nm + 20, phase_text, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # 添加折射光线示意
    arrow_start_x = positions[0]
    arrow_start_y = substrate_y + H_nm/2 + 100
    arrow_end_x = positions[-1] + 150
    arrow_end_y = arrow_start_y - 150 * np.tan(np.deg2rad(design_result['actual_angle']))
    
    arrow = FancyArrowPatch((arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y),
                           arrowstyle='->', mutation_scale=30, linewidth=3,
                           color='red', linestyle='--')
    ax4.add_patch(arrow)
    
    # 标注折射角
    angle_arc = Arc((arrow_start_x, arrow_start_y), 100, 100,
                   angle=0, theta1=270, theta2=270+design_result['actual_angle'],
                   color='red', linewidth=2, linestyle='--')
    ax4.add_patch(angle_arc)
    ax4.text(arrow_start_x + 60, arrow_start_y - 40,
            f'{design_result["actual_angle"]:.1f}°',
            fontsize=11, fontweight='bold', color='red')
    
    ax4.set_xlabel('位置 (nm)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('高度 (nm)', fontsize=12, fontweight='bold')
    ax4.set_title('超构表面阵列结构示意图（单位：纳米）', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('auto')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='royalblue', edgecolor='darkblue', label='硅纳米柱'),
        mpatches.Patch(facecolor='lightgray', edgecolor='gray', label='玻璃基底'),
        mpatches.Patch(facecolor='yellow', alpha=0.3, label='相位值')
    ]
    ax4.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.savefig(f'{save_prefix}_design_results.png', dpi=200, bbox_inches='tight')
    print(f"\n✓ 完整设计结果可视化已保存至 {save_prefix}_design_results.png")
    plt.show()
    
    return fig


def visualize_metasurface_parameters(design_result, save_prefix='metasurface'):
    """
    绘制参数分析图表
    """
    positions = design_result['positions'] * 1e9
    geometries = design_result['designed_geometries']
    errors = design_result['design_errors']
    ideal = design_result['ideal_phases']
    predicted = design_result['predicted_phases']
    
    L_values = np.array([g[0] for g in geometries])
    W_values = np.array([g[1] for g in geometries])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: L 和 W 的相位依赖性
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(L_values, predicted, c=errors, cmap='RdYlGn_r', s=150,
                     edgecolors='black', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('长度 L (nm)', fontsize=11)
    ax1.set_ylabel('实现相位 (度)', fontsize=11)
    ax1.set_title('L 与相位的关系', fontsize=12)
    plt.colorbar(sc1, ax=ax1, label='误差 (度)')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: W 和相位关系
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(W_values, predicted, c=errors, cmap='RdYlGn_r', s=150,
                     edgecolors='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('宽度 W (nm)', fontsize=11)
    ax2.set_ylabel('实现相位 (度)', fontsize=11)
    ax2.set_title('W 与相位的关系', fontsize=12)
    plt.colorbar(sc2, ax=ax2, label='误差 (度)')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: L vs W 二维分布
    ax3 = axes[1, 0]
    sc3 = ax3.scatter(L_values, W_values, c=predicted, cmap='hsv', s=150,
                     edgecolors='black', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('长度 L (nm)', fontsize=11)
    ax3.set_ylabel('宽度 W (nm)', fontsize=11)
    ax3.set_title('(L, W) 结构参数空间', fontsize=12)
    plt.colorbar(sc3, ax=ax3, label='相位 (度)')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 尺寸变化范围
    ax4 = axes[1, 1]
    ax4.boxplot([L_values, W_values], labels=['L', 'W'], patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    ax4.set_ylabel('尺寸 (nm)', fontsize=11)
    ax4.set_title('几何参数统计分布', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_parameters.png', dpi=150, bbox_inches='tight')
    print(f"✓ 参数分析图表已保存至 {save_prefix}_parameters.png")
    plt.show()
    
    return fig


def print_design_summary(design_result):
    """
    打印设计结果的文字总结
    """
    print("\n" + "="*60)
    print("超构表面设计总结")
    print("="*60)
    
    print(f"\n物理参数:")
    print(f"  工作波长: {design_result['wavelength']*1e9:.1f} nm")
    print(f"  单元周期: {design_result['period']*1e9:.1f} nm")
    print(f"  阵列规模: {design_result['n_elements']} 个单元")
    
    print(f"\n设计指标:")
    print(f"  目标折射角: {design_result['target_angle']:.1f}°")
    print(f"  实现折射角: {design_result['actual_angle']:.1f}°")
    print(f"  角度误差: {abs(design_result['actual_angle'] - design_result['target_angle']):.1f}°")
    
    print(f"\n相位设计:")
    print(f"  相位范围: [{design_result['ideal_phases'].min():.1f}°, {design_result['ideal_phases'].max():.1f}°]")
    print(f"  平均相位误差: {design_result['design_errors'].mean():.2f}°")
    print(f"  最大相位误差: {design_result['design_errors'].max():.2f}°")
    print(f"  标准差: {design_result['design_errors'].std():.2f}°")
    
    geometries = np.array(design_result['designed_geometries'])
    print(f"\n几何尺寸:")
    print(f"  L 范围: [{geometries[:, 0].min():.1f}, {geometries[:, 0].max():.1f}] nm")
    print(f"  W 范围: [{geometries[:, 1].min():.1f}, {geometries[:, 1].max():.1f}] nm")
    print(f"  L 平均值: {geometries[:, 0].mean():.1f} nm")
    print(f"  W 平均值: {geometries[:, 1].mean():.1f} nm")


def main_metasurface_design(inverse_model=None, scaler_X=None, forward_model=None):
    """
    超构表面设计的独立测试函数
    """
    from data_generator import MetasurfaceUnitSimulator
    from forward_model import train_forward_model
    from inverse_design import TandemTrainer
    
    # 如果没有提供模型，则训练一个
    if inverse_model is None or scaler_X is None or forward_model is None:
        print("MetaSurface 设计需要预训练的网络...")
        
        simulator = MetasurfaceUnitSimulator(wavelength=700e-9)
        X, Y = simulator.generate_dataset(n_samples=5000)
        Y_phase = Y[:, 1]
        
        forward_model, history, scaler_X = train_forward_model(X, Y_phase, epochs=300, verbose=False)
        
        tandem = TandemTrainer(forward_model, scaler_X)
        tandem.train_with_progress(epochs=500, verbose=False)
        
        inverse_model = tandem.inverse_model
    
    # 设计超构表面阵列
    design = design_anomalous_refraction_array(
        inverse_model=inverse_model,
        scaler_X=scaler_X,
        forward_model=forward_model,
        wavelength=700e-9,
        period=350e-9,
        target_angle_deg=30,
        n_elements=21
    )
    
    # 可视化和分析
    visualize_metasurface_comprehensive(design)
    visualize_metasurface_parameters(design)
    print_design_summary(design)
    
    return design


if __name__ == "__main__":
    design = main_metasurface_design()