"""
主程序：基于深度学习的超构表面逆向设计系统

完整工作流程：
1. 数据集构建：用解析模型生成 [L, W] -> [Phase, Amplitude]
2. 正向网络训练：学习"结构 -> 相位"的映射
3. Tandem 网络训练：逆向设计，"目标相位 -> 结构"
4. 超构表面阵列设计：用训练好的网络设计实现反常折射的纳米柱阵列

版本：1.0
"""

import sys
import os
import io

# 修复 Windows 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
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

from data_generator import MetasurfaceDatasetGenerator
from forward_model import train_forward_model, visualize_training, validate_forward_model
from inverse_design import TandemTrainer
from metasurface_design import (
    design_anomalous_refraction_array,
    visualize_metasurface_comprehensive,
    visualize_metasurface_parameters,
    print_design_summary
)


class MetasurfaceDesignPipeline:
    """
    超构表面逆向设计系统的主管理类
    """
    
    def __init__(self, output_dir='results'):
        """
        初始化流程
        
        参数:
        - output_dir: 输出结果的目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.simulator = None
        self.X = None
        self.Y = None
        
        self.forward_model = None
        self.scaler_X = None
        
        self.tandem = None
        self.inverse_model = None
        
        self.design_result = None
        
        print("\n" + "="*70)
        print(" "*15 + "基于深度学习的超构表面逆向设计系统")
        print(" "*20 + "版本 1.0")
        print("="*70)
        print(f"\n输出目录: {os.path.abspath(output_dir)}")
    
    def step1_generate_dataset(self, n_samples=5000):
        """
        步骤1：生成数据集
        """
        print("\n" + "-"*70)
        print("[步骤 1/4] 数据集构建")
        print("-"*70)
        
        self.simulator = MetasurfaceDatasetGenerator(n_samples=n_samples, wavelength=1550e-9)
        self.X, self.Y = self.simulator.generate_dataset(verbose=True)

        phases = self.Y[:, 0]      # phi_deg
        amplitudes = self.Y[:, 1]  # T (透射率)

        print(f"\n✓ 成功生成 {len(self.X)} 组训练数据")
        print(f"  w 范围: [{self.X[:, 0].min():.3f}, {self.X[:, 0].max():.3f}] (归一化)")
        print(f"  h 范围: [{self.X[:, 1].min():.3f}, {self.X[:, 1].max():.3f}] (归一化)")
        print(f"  p 范围: [{self.X[:, 2].min():.3f}, {self.X[:, 2].max():.3f}] (归一化)")
        print(f"  相位范围: [{phases.min():.1f}°, {phases.max():.1f}°]")
        print(f"  透射率范围: [{amplitudes.min():.3f}, {amplitudes.max():.3f}]")
        
        # 保存数据集
        dataset_path = os.path.join(self.output_dir, 'dataset.npz')
        np.savez(dataset_path, X=self.X, Y=self.Y)
        print(f"  数据已保存到: {dataset_path}")
        
        # 可视化数据集
        dataset_vis_path = os.path.join(self.output_dir, 'dataset_visualization.png')
        self.simulator.visualize_dataset(save_path=dataset_vis_path)
    
    def step2_train_forward_model(self, epochs=300):
        """
        步骤2：训练正向网络
        """
        print("\n" + "-"*70)
        print("[步骤 2/4] 正向网络训练")
        print("-"*70)
        
        if self.X is None or self.Y is None:
            raise ValueError("请先运行 step1_generate_dataset()")
        
        Y_phase = self.Y[:, 0]  # phi_deg (index 0)

        self.forward_model, history, self.scaler_X = train_forward_model(
            self.X, Y_phase, epochs=epochs, batch_size=128, verbose=True
        )

        # 可视化训练曲线
        print("\n绘制训练曲线...")
        visualize_training(self.forward_model, self.scaler_X, history=history)
        self._save_figure('forward_training.png')
        
        # 验证模型性能
        print("\n验证正向网络性能...")
        error = validate_forward_model(self.forward_model, self.scaler_X, self.X, Y_phase)
        self._save_figure('forward_validation.png')
        
        # 保存模型
        model_path = os.path.join(self.output_dir, 'forward_model_weights.pth')
        torch.save(self.forward_model.state_dict(), model_path)
        print(f"  模型已保存到: {model_path}")
    
    def step3_train_inverse_network(self, epochs=500):
        """
        步骤3：训练 Tandem 逆向网络
        """
        print("\n" + "-"*70)
        print("[步骤 3/4] Tandem 逆向网络训练")
        print("-"*70)
        
        if self.forward_model is None or self.scaler_X is None:
            raise ValueError("请先运行 step2_train_forward_model()")
        
        self.tandem = TandemTrainer(self.forward_model, self.scaler_X)
        history = self.tandem.train_with_progress(
            epochs=epochs, lr=0.001, verbose=True
        )

        self.inverse_model = self.tandem.inverse_model

        # 可视化训练曲线
        print("\n绘制 Tandem 训练曲线...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history, 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Tandem 网络训练曲线', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        self._save_figure('tandem_training.png')
        plt.close(fig)

        # 验证逆向网络
        print("\n验证逆向网络性能...")
        import torch
        test_targets = torch.tensor([[0.0, 0.8, 0.15], [90.0, 0.7, 0.2]], dtype=torch.float32)
        self.tandem.validate_inverse_design(test_targets)
        self._save_figure('tandem_validation.png')
        
        # 保存模型
        model_path = os.path.join(self.output_dir, 'inverse_model_weights.pth')
        torch.save(self.inverse_model.state_dict(), model_path)
        print(f"  模型已保存到: {model_path}")
    
    def step4_design_metasurface(self, target_angle=30, n_elements=21):
        """
        步骤4：设计超构表面阵列
        
        参数:
        - target_angle: 目标折射角（度）
        - n_elements: 阵列单元数
        """
        print("\n" + "-"*70)
        print("[步骤 4/4] 超构表面阵列设计与可视化")
        print("-"*70)
        
        if self.inverse_model is None or self.forward_model is None:
            raise ValueError("请先运行前面的训练步骤")
        
        # 设计阵列
        self.design_result = design_anomalous_refraction_array(
            inverse_model=self.inverse_model,
            scaler_X=self.scaler_X,
            forward_model=self.forward_model,
            wavelength=700e-9,
            period=350e-9,
            target_angle_deg=target_angle,
            n_elements=n_elements
        )
        
        # 可视化结果
        print("\n绘制设计结果...")
        visualize_metasurface_comprehensive(self.design_result,
                                           save_prefix=os.path.join(self.output_dir, 'metasurface'))
        self._save_figure('metasurface_design_results.png')
        
        visualize_metasurface_parameters(self.design_result,
                                        save_prefix=os.path.join(self.output_dir, 'metasurface'))
        self._save_figure('metasurface_parameters.png')
        
        # 打印总结
        print_design_summary(self.design_result)
        
        # 保存设计结果
        result_path = os.path.join(self.output_dir, 'design_result.npz')
        np.savez(
            result_path,
            positions=self.design_result['positions'],
            ideal_phases=self.design_result['ideal_phases'],
            predicted_phases=self.design_result['predicted_phases'],
            geometries=np.array(self.design_result['designed_geometries']),
            errors=self.design_result['design_errors']
        )
        print(f"  设计结果已保存到: {result_path}")
    
    def _save_figure(self, filename):
        """将当前图表保存到输出目录"""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close('all')
    
    def run_full_pipeline(self, n_samples=5000, forward_epochs=300, tandem_epochs=500,
                         target_angle=30, n_elements=21):
        """
        执行完整的设计流程
        
        参数:
        - n_samples: 数据集大小
        - forward_epochs: 正向网络训练轮数
        - tandem_epochs: Tandem 网络训练轮数
        - target_angle: 目标折射角（度）
        - n_elements: 阵列单元数
        """
        try:
            self.step1_generate_dataset(n_samples=n_samples)
            self.step2_train_forward_model(epochs=forward_epochs)
            self.step3_train_inverse_network(epochs=tandem_epochs)
            self.step4_design_metasurface(target_angle=target_angle, n_elements=n_elements)
            
            self._print_completion_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠ 程序被用户中断")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n✗ 出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _print_completion_summary(self):
        """打印完成总结"""
        print("\n" + "="*70)
        print(" "*20 + "✓ 所有任务完成！")
        print("="*70)
        
        print(f"\n生成的输出文件:")
        print(f"  ├── dataset_visualization.png          (数据集可视化)")
        print(f"  ├── forward_training.png               (正向网络训练曲线)")
        print(f"  ├── forward_validation.png             (正向网络验证)")
        print(f"  ├── tandem_training.png                (Tandem 训练曲线)")
        print(f"  ├── tandem_validation.png              (Tandem 验证)")
        print(f"  ├── metasurface_design_results.png     (设计结果)")
        print(f"  ├── metasurface_parameters.png         (参数分析)")
        print(f"  ├── dataset.npz                        (原始数据)")
        print(f"  ├── forward_model_weights.pth          (正向模型)")
        print(f"  ├── inverse_model_weights.pth          (逆向模型)")
        print(f"  └── design_result.npz                  (设计结果)")
        
        print(f"\n主要成果:")
        if self.design_result is not None:
            print(f"  • 目标折射角: {self.design_result['target_angle']}°")
            print(f"  • 实现折射角: {self.design_result['actual_angle']:.1f}°")
            print(f"  • 平均相位误差: {self.design_result['design_errors'].mean():.2f}°")
            print(f"  • 最大相位误差: {self.design_result['design_errors'].max():.2f}°")
        
        print(f"\n更多信息:")
        print(f"  • 完整结果保存在: {os.path.abspath(self.output_dir)}")
        print(f"  • 可以查看生成的 PNG 文件了解详细设计过程")


def main():
    """主函数"""
    
    # ===== 配置参数 =====
    config = {
        'output_dir': 'results',           # 输出目录
        'n_samples': 5000,                  # 数据集大小
        'forward_epochs': 300,              # 正向网络训练轮数
        'tandem_epochs': 500,               # Tandem 网络训练轮数
        'target_angle': 30,                 # 目标折射角（度）
        'n_elements': 21                    # 阵列单元数
    }
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   基于深度学习的超构表面逆向设计系统                              ║
    ║   Metasurface Inverse Design via Deep Learning                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    配置信息:
      • 数据集大小: {config['n_samples']} 样本
      • 正向网络: {config['forward_epochs']} epochs
      • Tandem 网络: {config['tandem_epochs']} epochs
      • 目标折射角: {config['target_angle']}°
      • 阵列规模: {config['n_elements']} 个单元
    
    """)
    
    # 创建流程对象并运行
    pipeline = MetasurfaceDesignPipeline(output_dir=config['output_dir'])
    
    # 执行完整流程
    pipeline.run_full_pipeline(
        n_samples=config['n_samples'],
        forward_epochs=config['forward_epochs'],
        tandem_epochs=config['tandem_epochs'],
        target_angle=config['target_angle'],
        n_elements=config['n_elements']
    )


if __name__ == "__main__":
    main()
