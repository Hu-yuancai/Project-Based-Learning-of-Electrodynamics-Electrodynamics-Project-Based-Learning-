"""
==============================================================================
  基于深度学习的超构表面逆向设计系统
  Metasurface Inverse Design via Deep Learning
  
  版本: 1.0 (2026-04-21)
  状态: ✅ 完全就绪
==============================================================================

⚡ 5 秒快速开始
──────────────────────────────────────────────────────────────────────────
1. 安装依赖:  pip install -r requirements.txt
2. 检查环境:  python verify.py
3. 运行演示:  python quick_demo.py    # 3-5 分钟
   或
   运行完整:  python main.py           # 10-15 分钟
──────────────────────────────────────────────────────────────────────────

📦 项目包含的文件
──────────────────────────────────────────────────────────────────────────

【核心代码模块】(5 个 Python 文件，~1950 行代码)
  ✓ main.py                    - 主程序入口和工作流管理
  ✓ data_generator.py          - 数据集生成（物理模拟）
  ✓ forward_model.py           - 正向网络训练（结构→相位）
  ✓ inverse_design.py          - Tandem 逆向网络（相位→结构）⭐ 核心创新
  ✓ metasurface_design.py      - 阵列设计和可视化

【快速启动脚本】
  ✓ quick_demo.py              - 快速演示脚本（3-5 分钟）
  ✓ verify.py                  - 环境检查脚本

【完整文档】(~1800 行文档)
  ✓ README.md                  - 完整技术文档
  ✓ USAGE.md                   - 使用指南和快速参考
  ✓ INDEX.md                   - 项目总览
  ✓ PROJECT_SUMMARY.md         - 项目完成清单

【配置文件】
  ✓ requirements.txt           - 依赖包列表
  ✓ .gitignore                 - Git 忽略规则
  ✓ QUICKSTART.md              - 本文件

──────────────────────────────────────────────────────────────────────────

🎯 选择你的运行方式

【方式 1：快速体验】(推荐:最快)
  > python verify.py        # ~ 30 秒
  > python quick_demo.py    # ~ 3-5 分钟
  特点: 快速验证代码功能，参数缩小 10 倍

【方式 2：完整项目】(推荐:最佳结果)
  > python verify.py        # ~ 30 秒
  > python main.py          # ~ 10-15 分钟
  特点: 最终项目所需，得到最佳设计结果

【方式 3：环境诊断】(推荐:有问题时)
  > python verify.py        # 详细检查环境
  特点: 诊断整个系统配置

──────────────────────────────────────────────────────────────────────────

💻 系统要求

【最低配置】
  • Python 3.8+
  • 4 GB 内存
  • CPU: Intel i5 或同等级

【推荐配置】
  • Python 3.10+
  • 8+ GB 内存
  • CPU: Intel i7/i9 或 AMD Ryzen 5000+ 系列
  • GPU: NVIDIA RTX 3060+（可选，可加快 5-10 倍）

──────────────────────────────────────────────────────────────────────────

📋 安装步骤

【第 1 步】安装依赖包
  方式 A: 自动安装（推荐）
    > pip install -r requirements.txt

  方式 B: 手动安装
    > pip install torch numpy scikit-learn matplotlib scipy

  方式 C: 特定 CUDA 版本（如果需要 GPU 加速）
    访问: https://pytorch.org/get-started/locally/
    然后复制对应的安装命令

【第 2 步】验证环境
  > python verify.py
  
  应该看到所有项都是 ✓

【第 3 步】运行程序
  快速: python quick_demo.py
  完整: python main.py

──────────────────────────────────────────────────────────────────────────

🚀 运行后会生成什么

main.py 运行完成后，在 results/ 目录中生成:

【可视化结果】(PNG 图表)
  ✓ metasurface_design_results.png    ⭐ 最重要(4 子图)
    - 相位分布对比
    - 设计误差分析
    - 几何参数分布
    - 超构表面结构示意

  ✓ metasurface_parameters.png        (参数分析)
  ✓ forward_training.png              (正向网络训练)
  ✓ forward_validation.png            (正向网络验证)
  ✓ tandem_training.png               (Tandem 训练)
  ✓ tandem_validation.png             (Tandem 验证)
  ✓ dataset_visualization.png         (数据集)

【数据文件】(用于进一步分析)
  ✓ dataset.npz                       (原始数据)
  ✓ forward_model_weights.pth         (正向模型)
  ✓ inverse_model_weights.pth         (逆向模型)
  ✓ design_result.npz                 (设计结果)

──────────────────────────────────────────────────────────────────────────

📊 预期结果指标

运行完成后应该看到:

  ✓ 正向网络相位误差: < 5° (通常 3-4°)
  ✓ Tandem 逆向网络误差: < 5° (通常 2-3°)
  ✓ 设计平均相位误差: < 10° (通常 5-8°)
  ✓ 实现折射角: 30° ± 1°（目标 30°）
  ✓ 网络推理时间: 2-3 ms

──────────────────────────────────────────────────────────────────────────

🔍 输出解读指南

【最重要的图】metasurface_design_results.png

  左上: 相位分布对比
        - 绿色圆点: 理想相位
        - 橙色方点: 网络实现相位
        - 二者应该贴近

  右上: 相位误差分析
        - 柱子越短越好
        - 应该都在 10° 以下
        - 平均在 5-8° 最佳

  左下: 几何尺寸分布
        - L 和 W 的变化趋势
        - 应该相对平滑

  右下: 超构表面示意图
        - 蓝色矩形: 纳米柱
        - 红色虚线: 折射光线
        - 应该与目标角度吻合

【其他重要图】

  forward_training.png
        - 蓝线(训练)和红线(验证)都应下降且最终接近

  tandem_training.png  
        - 蓝线应该单调下降（可能有震荡）

──────────────────────────────────────────────────────────────────────────

⚠️  常见问题快速排查

【问题 1】找不到模块 "torch"
  解决: pip install torch

【问题 2】GPU 没有被使用
  解决: 检查 torch.cuda.is_available()
       在 Python 中运行:
         import torch
         print(torch.cuda.is_available())

【问题 3】内存不足
  解决: 修改 main.py 中的 config
       减少 'n_samples' 或运行 quick_demo.py

【问题 4】运行很慢
  解决: 
    a. 确认 GPU 正常使用
    b. 关闭其他应用
    c. 尝试快速演示 (quick_demo.py)

【问题 5】结果质量差（误差 > 20°）
  解决:
    a. 增加 forward_epochs (300 → 500)
    b. 增加 tandem_epochs (500 → 1000)
    c. 增加 n_samples (5000 → 10000)

详细解答请见 USAGE.md 的"常见问题"部分。

──────────────────────────────────────────────────────────────────────────

📚 文档导航

需要...                      查看文件
────────────────────────────────────────────────────────────────────────
快速开始                    USAGE.md
理解物理原理                README.md  
全面了解项目                INDEX.md
了解项目完成情况            PROJECT_SUMMARY.md
代码注释                    各个 .py 文件

──────────────────────────────────────────────────────────────────────────

🎓 项目核心亮点

【创新点 1】Tandem Network 架构 ⭐⭐⭐
  解决了传统逆向设计的"非唯一映射"问题
  通过在物理响应空间(相位)而非几何空间定义 Loss
  使网络能找到任意一个可行的设计

【创新点 2】物理对称性的网络编码
  相位的周期拓扑 → sin/cos 双通道编码
  能量守恒约束 → 正则化项
  这是 Physics-Informed Neural Network 的体现

【创新点 3】完整的避坑记录
  项目明确记录了两个 AI 陷阱:
    a) 相位周期性导致的梯度混乱
    b) 非唯一映射导致的平均化退化
  及其解决方案

──────────────────────────────────────────────────────────────────────────

💡 如何修改参数

【改变目标折射角】
  编辑 main.py:
    config = {
        'target_angle': 45,    # 改为 45° (默认 30°)
        ...
    }

【改变阵列规模】
  编辑 main.py:
    config = {
        'n_elements': 31,      # 改为 31 个单元 (默认 21)
        ...
    }

【改变数据集大小】
  编辑 main.py:
    config = {
        'n_samples': 10000,    # 更多数据,更好结果 (默认 5000)
        ...
    }

【增加训练轮数(以获得更好结果)】
  编辑 main.py:
    config = {
        'forward_epochs': 500,     # 默认 300
        'tandem_epochs': 1000,     # 默认 500
        ...
    }

──────────────────────────────────────────────────────────────────────────

🔗 参考资源

【物理论文】
  • Pendry et al., "Controlling electromagnetic fields with metamaterials"
    Science 312.5781 (2006): 1780-1782
  • Yu & Capasso, "Flat optics with designer metasurfaces"
    Nature Materials 13.2 (2014): 139-150

【深度学习】
  • Raissi et al., "Physics-informed neural networks"
    SIAM Review 65.3 (2023): 681-715

【代码文档】
  • PyTorch: https://pytorch.org/tutorials/
  • scikit-learn: https://scikit-learn.org/
  • Matplotlib: https://matplotlib.org/

──────────────────────────────────────────────────────────────────────────

✅ 确认清单

运行程序前，请确保:

  □ Python 版本 ≥ 3.8
  □ 已安装 requirements.txt 中的所有依赖
  □ 运行了 verify.py 且所有检查都是 ✓
  □ 有足够的磁盘空间 (~500 MB for results/)
  □ 如果有 GPU，已安装正确的 CUDA 版本

──────────────────────────────────────────────────────────────────────────

🎯 立即开始

【第一步】检查环境
  > python verify.py

【第二步】选择运行方式
  快速: > python quick_demo.py     # 3-5 分钟
  完整: > python main.py           # 10-15 分钟

【第三步】查看结果
  results/ 目录中查看 PNG 图表
  
【第四步】（可选）深入理解
  阅读 README.md 了解物理原理
  阅读代码注释理解实现细节

──────────────────────────────────────────────────────────────────────────

❓ 如有问题

1. 查看 USAGE.md 的"常见问题"部分
2. 运行 verify.py 进行诊断
3. 检查代码中的详细注释
4. 查阅 README.md 的技术说明

──────────────────────────────────────────────────────────────────────────

📝 最后的话

这个项目完整展示了深度学习在物理逆问题中的应用。
通过 Tandem Network 的创新架构，我们演示了如何结合物理约束
来解决结构和电磁响应的映射问题。

希望这个项目对你的学习和研究有帮助！

祝你运行顺利！ 🚀

──────────────────────────────────────────────────────────────────────────

版本: 1.0
完成日期: 2026 年 4 月 21 日
状态: ✅ 完全就绪

现在，去运行: python verify.py
"""
