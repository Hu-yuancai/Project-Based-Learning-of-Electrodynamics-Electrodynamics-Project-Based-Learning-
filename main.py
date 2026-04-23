"""
主程序：基于深度学习的超构表面逆向设计系统

流程：
  1. 数据集构建  (data_generator.py)
  2. 正向网络训练 (forward_model.py)
  3. Tandem 逆向网络训练 (inverse_design.py)
  4. 超构表面阵列设计与可视化 (metasurface_design.py)
"""

import sys
import io
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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

from data_generator import RigorousMetasurfaceSimulator
from forward_model import (train_forward_model, visualize_training,
                           validate_forward_model)
from inverse_design import TandemTrainer
from metasurface_design import (design_anomalous_refraction_array,
                                visualize_metasurface_comprehensive,
                                visualize_metasurface_parameters,
                                print_design_summary)


class MetasurfaceDesignPipeline:
    """超构表面逆向设计系统主流程"""

    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.X = self.Y = None
        self.forward_model = self.scaler_X = None
        self.tandem = self.inverse_model = None
        self.design_result = None

        print("\n" + "=" * 70)
        print(" " * 12 + "基于深度学习的超构表面逆向设计系统  (λ=1550 nm)")
        print("=" * 70)
        print(f"输出目录: {os.path.abspath(output_dir)}")

    # ------------------------------------------------------------------

    def step1_generate_dataset(self, n_samples=5000):
        print("\n" + "-" * 70)
        print("[步骤 1/4] 数据集构建")
        print("-" * 70)

        sim = RigorousMetasurfaceSimulator(wavelength=1550e-9)
        self.X, self.Y = sim.generate_dataset(n_samples=n_samples)

        np.savez(os.path.join(self.output_dir, 'dataset.npz'), X=self.X, Y=self.Y)

        vis_path = os.path.join(self.output_dir, 'dataset_visualization.png')
        sim.visualize_dataset(self.X, self.Y, save_path=vis_path)

        print(f"数据已保存: {os.path.join(self.output_dir, 'dataset.npz')}")

    def step2_train_forward_model(self, epochs=300):
        print("\n" + "-" * 70)
        print("[步骤 2/4] 正向网络训练")
        print("-" * 70)

        if self.X is None:
            raise RuntimeError("请先运行 step1_generate_dataset()")

        model_path = os.path.join(self.output_dir, 'forward_model_weights.pth')
        self.forward_model, history, self.scaler_X = train_forward_model(
            self.X, self.Y, epochs=epochs, batch_size=128,
            verbose=True, save_path=model_path
        )

        visualize_training(history=history, title='正向网络训练曲线')
        plt.savefig(os.path.join(self.output_dir, 'forward_training.png'),
                    dpi=150, bbox_inches='tight')
        plt.close('all')

        validate_forward_model(self.forward_model, self.scaler_X,
                               self.X[:500], self.Y[:500], plot_results=True)
        val_fig = os.path.join(self.output_dir, 'forward_validation.png')
        if Path('forward_validation.png').exists():
            import shutil
            shutil.move('forward_validation.png', val_fig)

        print(f"模型已保存: {model_path}")

    def step3_train_inverse_network(self, epochs=500):
        print("\n" + "-" * 70)
        print("[步骤 3/4] Tandem 逆向网络训练")
        print("-" * 70)

        if self.forward_model is None:
            raise RuntimeError("请先运行 step2_train_forward_model()")

        self.tandem = TandemTrainer(self.forward_model, self.scaler_X)
        losses = self.tandem.train_with_progress(epochs=epochs, lr=1e-3, verbose=True)
        self.inverse_model = self.tandem.inverse_model

        # 训练曲线
        fig, ax = plt.subplots(figsize=(9, 5))  # noqa: F841 — fig needed for savefig
        ax.plot(losses, 'b-', lw=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Tandem 网络训练曲线')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tandem_training.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 验证
        self.tandem.validate_inverse_design(visualize=True)
        val_fig = os.path.join(self.output_dir, 'tandem_validation.png')
        if Path('tandem_validation.png').exists():
            import shutil
            shutil.move('tandem_validation.png', val_fig)

        model_path = os.path.join(self.output_dir, 'inverse_model_weights.pth')
        torch.save(self.inverse_model.state_dict(), model_path)
        print(f"模型已保存: {model_path}")

    def step4_design_metasurface(self, target_angle=30, n_elements=21):
        print("\n" + "-" * 70)
        print("[步骤 4/4] 超构表面阵列设计")
        print("-" * 70)

        if self.inverse_model is None:
            raise RuntimeError("请先运行前面的训练步骤")

        self.design_result = design_anomalous_refraction_array(
            inverse_model=self.inverse_model,
            scaler_X=self.scaler_X,
            forward_model=self.forward_model,
            wavelength=1550e-9,
            period=600e-9,
            target_angle_deg=target_angle,
            n_elements=n_elements,
        )

        prefix = os.path.join(self.output_dir, 'metasurface')
        visualize_metasurface_comprehensive(self.design_result, save_prefix=prefix)
        visualize_metasurface_parameters(self.design_result, save_prefix=prefix)
        print_design_summary(self.design_result)

        np.savez(
            os.path.join(self.output_dir, 'design_result.npz'),
            positions=self.design_result['positions'],
            ideal_phases=self.design_result['ideal_phases'],
            predicted_phases=self.design_result['predicted_phases'],
            geometries=np.array(self.design_result['designed_geometries']),
            errors=self.design_result['design_errors'],
        )

    # ------------------------------------------------------------------

    def run_full_pipeline(self, n_samples=5000, forward_epochs=300,
                          tandem_epochs=500, target_angle=30, n_elements=21):
        try:
            self.step1_generate_dataset(n_samples=n_samples)
            self.step2_train_forward_model(epochs=forward_epochs)
            self.step3_train_inverse_network(epochs=tandem_epochs)
            self.step4_design_metasurface(target_angle=target_angle,
                                          n_elements=n_elements)
            self._print_summary()
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            sys.exit(1)
        except Exception as e:
            import traceback
            print(f"\n出错: {e}")
            traceback.print_exc()
            sys.exit(1)

    def _print_summary(self):
        print("\n" + "=" * 70)
        print(" " * 25 + "所有步骤完成")
        print("=" * 70)
        if self.design_result:
            dr = self.design_result
            print(f"  目标折射角: {dr['target_angle']}°")
            print(f"  实现折射角: {dr['actual_angle']:.1f}°")
            print(f"  平均相位误差: {dr['design_errors'].mean():.2f}°")
        print(f"  结果目录: {os.path.abspath(self.output_dir)}")


def main():
    config = {
        'output_dir':      'results',
        'n_samples':       5000,
        'forward_epochs':  300,
        'tandem_epochs':   500,
        'target_angle':    30,
        'n_elements':      21,
    }

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║   基于深度学习的超构表面逆向设计系统  (λ = 1550 nm)           ║
    ╚══════════════════════════════════════════════════════════════╝
      数据集: {config['n_samples']} 样本
      正向网络: {config['forward_epochs']} epochs
      Tandem 网络: {config['tandem_epochs']} epochs
      目标折射角: {config['target_angle']}°
      阵列规模: {config['n_elements']} 单元
    """)

    pipeline = MetasurfaceDesignPipeline(output_dir=config['output_dir'])
    pipeline.run_full_pipeline(
        n_samples=config['n_samples'],
        forward_epochs=config['forward_epochs'],
        tandem_epochs=config['tandem_epochs'],
        target_angle=config['target_angle'],
        n_elements=config['n_elements'],
    )


if __name__ == "__main__":
    main()
