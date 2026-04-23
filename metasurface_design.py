"""
模块四：反常折射阵列设计与可视化

广义斯涅尔定律（正入射）：
    dΦ/dx = (2π/λ₀) sin(θ_t)

第 n 个单元的理想相位：
    Φ_n = Φ_0 + n · (2π · p_avg / λ₀) · sin(θ_t)  (mod 2π，映射到 [-π, π])

工作波长：λ₀ = 1550 nm
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc
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


def design_anomalous_refraction_array(inverse_model, scaler_X, forward_model,
                                       wavelength=1550e-9, period=600e-9,
                                       target_angle_deg=30, n_elements=21):
    """
    设计实现反常折射的超构表面阵列

    参数：
      inverse_model    : 训练好的 InverseDesigner
      scaler_X         : StandardScaler（fit 在 [w_nm, p_nm] 上）
      forward_model    : 训练好的 ForwardPredictor
      wavelength       : 工作波长 (m)，默认 1550 nm
      period           : 单元间距 (m)，默认 600 nm（取参数范围中值）
      target_angle_deg : 目标折射角 (度)
      n_elements       : 阵列单元数

    返回：design_result 字典
    """
    print("\n" + "=" * 60)
    print("模块四：反常折射阵列设计")
    print("=" * 60)

    k0 = 2 * np.pi / wavelength
    phase_gradient = k0 * np.sin(np.deg2rad(target_angle_deg))

    positions = np.arange(n_elements) * period
    ideal_phases_rad = phase_gradient * positions
    ideal_phases_deg = np.rad2deg(ideal_phases_rad) % 360
    ideal_phases_deg = (ideal_phases_deg + 180) % 360 - 180   # 映射到 [-180, 180]

    print(f"\n[1/3] 理想相位分布")
    print(f"  工作波长: {wavelength*1e9:.0f} nm")
    print(f"  单元间距: {period*1e9:.0f} nm")
    print(f"  目标折射角: {target_angle_deg:.1f}°")
    print(f"  阵列单元数: {n_elements}")

    # ------------------------------------------------------------------
    # 逆向设计每个单元
    # ------------------------------------------------------------------
    print(f"\n[2/3] 逆向设计几何参数...")

    scaler_mean  = torch.tensor(scaler_X.mean_,  dtype=torch.float32)
    scaler_scale = torch.tensor(scaler_X.scale_, dtype=torch.float32)

    designed_geometries = []
    predicted_phases    = []
    design_errors       = []

    inverse_model.eval()
    forward_model.eval()

    with torch.no_grad():
        for phi_t in ideal_phases_deg:
            # 目标相位 -> 归一化输入 [batch, 1]
            phi_tensor = torch.tensor([[phi_t / 180.0]], dtype=torch.float32)

            # 逆向预测
            geo_norm = inverse_model(phi_tensor)
            w_nm, p_nm = inverse_model.denormalize(geo_norm)

            # 正向验证
            geo = torch.stack([w_nm, p_nm], dim=1)
            geo_scaled = (geo - scaler_mean) / scaler_scale
            pred_sin, pred_cos = forward_model(geo_scaled)
            actual_phi = float(torch.rad2deg(torch.atan2(pred_sin, pred_cos)).item())

            err = abs(phi_t - actual_phi)
            err = min(err, 360 - err)

            designed_geometries.append((w_nm.item(), p_nm.item()))
            predicted_phases.append(actual_phi)
            design_errors.append(err)

    design_errors = np.array(design_errors)
    print(f"  平均相位误差: {design_errors.mean():.2f}°")
    print(f"  最大相位误差: {design_errors.max():.2f}°")

    # ------------------------------------------------------------------
    # 验证等效折射角
    # ------------------------------------------------------------------
    actual_angle, _ = verify_refraction_angle(predicted_phases, period, wavelength)
    print(f"\n[3/3] 性能评估")
    print(f"  目标折射角: {target_angle_deg:.1f}°")
    print(f"  实现折射角: {actual_angle:.1f}°  "
          f"(误差: {abs(actual_angle - target_angle_deg):.1f}°)")

    return {
        'positions':           positions,
        'ideal_phases':        ideal_phases_deg,
        'designed_geometries': designed_geometries,
        'predicted_phases':    predicted_phases,
        'design_errors':       design_errors,
        'target_angle':        target_angle_deg,
        'actual_angle':        actual_angle,
        'wavelength':          wavelength,
        'period':              period,
        'n_elements':          n_elements,
    }


def verify_refraction_angle(phase_distribution, period, wavelength):
    """从相位分布线性拟合折射角"""
    positions = np.arange(len(phase_distribution)) * period
    phase_rad = np.unwrap(np.deg2rad(phase_distribution))
    coeffs    = np.polyfit(positions, phase_rad, 1)
    gradient  = coeffs[0]
    k0        = 2 * np.pi / wavelength
    sin_theta = gradient / k0
    angle = float(np.rad2deg(np.arcsin(np.clip(sin_theta, -1, 1))))
    return angle, gradient


def visualize_metasurface_comprehensive(design_result, save_prefix='metasurface'):
    """4 子图综合可视化"""
    pos   = design_result['positions'] * 1e9
    ideal = design_result['ideal_phases']
    pred  = design_result['predicted_phases']
    errs  = design_result['design_errors']
    geos  = design_result['designed_geometries']
    ta    = design_result['target_angle']
    aa    = design_result['actual_angle']

    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.3)

    # 子图1：相位分布对比
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(pos, ideal, 'o-', label='理想相位', color='forestgreen',
             markersize=8, lw=2)
    ax1.plot(pos, pred,  's--', label='网络实现相位', color='darkorange',
             markersize=7, lw=2)
    ax1.fill_between(pos, ideal, pred, alpha=0.15, color='gray')
    ax1.set_xlabel('位置 (nm)')
    ax1.set_ylabel('相位 (度)')
    ax1.set_title(f'相位分布对比  目标={ta}°  实现={aa:.1f}°')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：误差柱状图
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.RdYlGn_r(errs / max(errs.max(), 1e-6))
    ax2.bar(pos, errs, width=np.diff(pos).mean() * 0.7,
            color=colors, edgecolor='k', lw=1, alpha=0.85)
    ax2.axhline(errs.mean(), color='r', ls='--', lw=2,
                label=f'均值={errs.mean():.2f}°')
    ax2.axhline(10, color='orange', ls=':', lw=1.5, label='10° 阈值')
    ax2.set_xlabel('位置 (nm)')
    ax2.set_ylabel('相位误差 (度)')
    ax2.set_title('设计误差分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 子图3：几何参数
    ax3 = fig.add_subplot(gs[1, 1])
    w_vals = np.array([g[0] for g in geos])
    p_vals = np.array([g[1] for g in geos])
    bw = np.diff(pos).mean() * 0.35
    ax3.bar(pos - bw / 2, w_vals, bw, label='宽度 w', color='steelblue',
            edgecolor='darkblue', lw=1, alpha=0.85)
    ax3.bar(pos + bw / 2, p_vals, bw, label='周期 p', color='coral',
            edgecolor='darkred', lw=1, alpha=0.85)
    ax3.set_xlabel('位置 (nm)')
    ax3.set_ylabel('尺寸 (nm)')
    ax3.set_title('各单元几何参数')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 子图4：阵列结构示意图
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_xlim([pos.min() - 100, pos.max() + 100])
    ax4.set_ylim([-300, 280])

    sub_y = -50
    ax4.axhline(sub_y, color='gray', lw=3)
    ax4.fill_between([pos.min() - 100, pos.max() + 100], sub_y, -300,
                     color='lightgray', alpha=0.5)
    ax4.text(pos.min() - 80, -200, 'SiO₂ 基底\n(n=1.444)',
             fontsize=9, ha='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    H_draw = 200
    for i, (p_x, (w, _)) in enumerate(zip(pos, geos)):
        rect = Rectangle((p_x - w / 2, sub_y), w, H_draw,
                         facecolor='royalblue', edgecolor='darkblue', lw=1.5, alpha=0.85)
        ax4.add_patch(rect)
        ax4.text(p_x, sub_y - 28, str(i), ha='center', fontsize=9, fontweight='bold')
        color = 'darkred' if errs[i] > 10 else 'darkgreen'
        ax4.text(p_x, sub_y + H_draw + 15, f'{pred[i]:.0f}°',
                 ha='center', fontsize=8, color=color,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.3))

    # 折射光线示意
    ax_start = (pos[0], sub_y + H_draw / 2 + 80)
    ax_end   = (pos[-1] + 120,
                ax_start[1] - 120 * np.tan(np.deg2rad(aa)))
    arrow = FancyArrowPatch(ax_start, ax_end, arrowstyle='->', mutation_scale=25,
                            lw=2.5, color='red', linestyle='--')
    ax4.add_patch(arrow)
    arc = Arc(ax_start, 80, 80, angle=0,
              theta1=270, theta2=270 + aa, color='red', lw=1.5)
    ax4.add_patch(arc)
    ax4.text(ax_start[0] + 50, ax_start[1] - 30, f'{aa:.1f}°',
             fontsize=10, color='red', fontweight='bold')

    legend_els = [
        mpatches.Patch(facecolor='royalblue', edgecolor='darkblue', label='Si 纳米柱'),
        mpatches.Patch(facecolor='lightgray', edgecolor='gray',     label='SiO₂ 基底'),
    ]
    ax4.legend(handles=legend_els, loc='upper left', fontsize=10)
    ax4.set_xlabel('位置 (nm)')
    ax4.set_ylabel('高度 (nm)')
    ax4.set_title('超构表面阵列结构示意图（λ=1550 nm）')
    ax4.grid(True, alpha=0.3)

    save_path = f'{save_prefix}_design_results.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"设计结果已保存: {save_path}")
    return fig


def visualize_metasurface_parameters(design_result, save_prefix='metasurface'):
    """参数分析 4 子图"""
    geos  = design_result['designed_geometries']
    errs  = design_result['design_errors']
    pred  = design_result['predicted_phases']

    w_vals = np.array([g[0] for g in geos])
    p_vals = np.array([g[1] for g in geos])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('几何参数分析', fontsize=13)

    sc1 = axes[0, 0].scatter(w_vals, pred, c=errs, cmap='RdYlGn_r', s=100,
                              edgecolors='k', lw=1)
    axes[0, 0].set_xlabel('宽度 w (nm)')
    axes[0, 0].set_ylabel('实现相位 (度)')
    axes[0, 0].set_title('w 与相位关系')
    plt.colorbar(sc1, ax=axes[0, 0], label='误差 (度)')
    axes[0, 0].grid(True, alpha=0.3)

    sc2 = axes[0, 1].scatter(p_vals, pred, c=errs, cmap='RdYlGn_r', s=100,
                              edgecolors='k', lw=1)
    axes[0, 1].set_xlabel('周期 p (nm)')
    axes[0, 1].set_ylabel('实现相位 (度)')
    axes[0, 1].set_title('p 与相位关系')
    plt.colorbar(sc2, ax=axes[0, 1], label='误差 (度)')
    axes[0, 1].grid(True, alpha=0.3)

    sc3 = axes[1, 0].scatter(w_vals, p_vals, c=pred, cmap='hsv', s=100,
                              edgecolors='k', lw=1)
    axes[1, 0].set_xlabel('宽度 w (nm)')
    axes[1, 0].set_ylabel('周期 p (nm)')
    axes[1, 0].set_title('(w, p) 参数空间')
    plt.colorbar(sc3, ax=axes[1, 0], label='相位 (度)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].boxplot([w_vals, p_vals], labels=['w', 'p'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', lw=2))
    axes[1, 1].set_ylabel('尺寸 (nm)')
    axes[1, 1].set_title('几何参数统计分布')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = f'{save_prefix}_parameters.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"参数分析图已保存: {save_path}")


def print_design_summary(design_result):
    """打印设计总结"""
    print("\n" + "=" * 60)
    print("超构表面设计总结")
    print("=" * 60)
    print(f"  工作波长: {design_result['wavelength']*1e9:.0f} nm")
    print(f"  单元间距: {design_result['period']*1e9:.0f} nm")
    print(f"  阵列规模: {design_result['n_elements']} 个单元")
    print(f"  目标折射角: {design_result['target_angle']:.1f}°")
    print(f"  实现折射角: {design_result['actual_angle']:.1f}°  "
          f"(误差: {abs(design_result['actual_angle'] - design_result['target_angle']):.1f}°)")
    errs = design_result['design_errors']
    print(f"  平均相位误差: {errs.mean():.2f}°")
    print(f"  最大相位误差: {errs.max():.2f}°")
    geos = np.array(design_result['designed_geometries'])
    print(f"  w 范围: [{geos[:,0].min():.1f}, {geos[:,0].max():.1f}] nm")
    print(f"  p 范围: [{geos[:,1].min():.1f}, {geos[:,1].max():.1f}] nm")
