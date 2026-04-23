"""
快速演示脚本

用更小的参数快速运行完整流程，用于测试和演示
预期运行时间：3-5 分钟（GPU）/ 10-15 分钟（CPU）

运行：python quick_demo.py
"""

import os
import sys
import io

# 修复 Windows 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

from data_generator import MetasurfaceUnitSimulator
from forward_model import train_forward_model, validate_forward_model
from inverse_design import TandemTrainer
from metasurface_design import (
    design_anomalous_refraction_array,
    visualize_metasurface_comprehensive,
    print_design_summary
)


def quick_demo():
    """
    快速演示完整设计流程
    """
    
    print("\n" + "="*70)
    print(" "*15 + "超构表面设计系统 - 快速演示")
    print("="*70)
    
    # 创建输出目录
    os.makedirs('quick_demo_results', exist_ok=True)
    
    # ===== 步骤 1：生成数据 =====
    print("\n[1/4] 生成数据集 (1000 样本)...")
    simulator = MetasurfaceUnitSimulator(wavelength=700e-9)
    X, Y = simulator.generate_dataset(n_samples=1000)
    Y_phase = Y[:, 1]
    print(f"✓ 生成 {len(X)} 组数据")
    print(f"  相位范围: [{Y_phase.min():.1f}°, {Y_phase.max():.1f}°]")
    
    # ===== 步骤 2：训练正向网络 =====
    print("\n[2/4] 训练正向网络 (50 epochs, 快速模式)...")
    forward_model, history, scaler_X = train_forward_model(
        X, Y_phase, epochs=50, batch_size=128, verbose=False
    )
    
    # 快速验证
    metrics = validate_forward_model(forward_model, scaler_X, X, Y_phase, plot_results=False)
    print(f"✓ 正向网络完成，平均相位误差: {metrics['mae_phi']:.2f}°")
    
    # 保存训练曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], 'b-', label='训练 Loss')
    ax.plot(history['val_loss'], 'r-', label='验证 Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('正向网络训练曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('quick_demo_results/forward_training.png', dpi=100)
    plt.close()
    print("  → 训练曲线已保存")
    
    # ===== 步骤 3：训练 Tandem 网络 =====
    print("\n[3/4] 训练 Tandem 逆向网络 (100 epochs, 快速模式)...")
    tandem = TandemTrainer(forward_model, scaler_X)
    losses = tandem.train_with_progress(epochs=100, lr=0.001, verbose=False)
    
    # 保存训练曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Tandem Loss')
    ax.set_title('Tandem 网络训练曲线')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('quick_demo_results/tandem_training.png', dpi=100)
    plt.close()
    print("✓ Tandem 网络完成")
    print("  → 训练曲线已保存")
    
    # ===== 步骤 4：设计超构表面 =====
    print("\n[4/4] 设计反常折射超构表面...")
    design = design_anomalous_refraction_array(
        inverse_model=tandem.inverse_model,
        scaler_X=scaler_X,
        forward_model=forward_model,
        wavelength=700e-9,
        period=350e-9,
        target_angle_deg=30,
        n_elements=21
    )
    
    # 保存可视化
    print("\n绘制完整设计结果...")
    visualize_metasurface_comprehensive(design, save_prefix='quick_demo_results/metasurface')
    plt.close('all')
    print("  → 完整设计图已保存")
    
    # 打印总结
    print_design_summary(design)
    
    # ===== 完成 =====
    print("\n" + "="*70)
    print("✓ 快速演示完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  ├── quick_demo_results/forward_training.png")
    print("  ├── quick_demo_results/tandem_training.png")
    print("  └── quick_demo_results/metasurface_design_results.png")
    print("\n提示:")
    print("  • 这个演示使用了较小的参数以加快速度")
    print("  • 要运行完整版本，请执行: python main.py")
    print("  • 完整版本会得到更好的设计结果（更低的误差）")


if __name__ == "__main__":
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n⚠ 程序被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)