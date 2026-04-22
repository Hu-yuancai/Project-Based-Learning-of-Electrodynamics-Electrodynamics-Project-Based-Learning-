"""
快速验证脚本

用于检查环境配置和代码基本功能是否正常

运行：python verify.py
"""

import sys
import os

print("\n" + "="*70)
print("环境验证和代码健康检查")
print("="*70)

# ===== 第1步：检查 Python 版本 =====
print("\n[1/6] 检查 Python 版本...", end=" ")
if sys.version_info >= (3, 8):
    print("✓")
    print(f"    Python {sys.version}")
else:
    print("✗")
    print(f"    错误：需要 Python 3.8+，当前版本 {sys.version}")
    sys.exit(1)

# ===== 第2步：检查必要库 =====
print("[2/6] 检查依赖库...", end=" ")
required_packages = {
    'numpy': 'NumPy',
    'torch': 'PyTorch',
    'sklearn': 'scikit-learn',
    'matplotlib': 'Matplotlib'
}

missing = []
for package, name in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing.append(name)

if not missing:
    print("✓")
    print("    所有依赖库已安装")
else:
    print("✗")
    print(f"    缺少以下库: {', '.join(missing)}")
    print("    运行: pip install -r requirements.txt")
    sys.exit(1)

# ===== 第3步：检查 PyTorch 和 GPU =====
print("[3/6] 检查 PyTorch 配置...", end=" ")
try:
    import torch
    print("✓")
    print(f"    PyTorch 版本: {torch.__version__}")
    print(f"    CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU 设备: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("✗")
    print(f"    错误: {e}")
    sys.exit(1)

# ===== 第4步：测试数据生成 =====
print("[4/6] 测试数据生成模块...", end=" ")
try:
    from data_generator import MetasurfaceUnitSimulator
    
    simulator = MetasurfaceUnitSimulator()
    X, Y = simulator.generate_dataset(n_samples=100)
    
    if X.shape == (100, 2) and Y.shape == (100, 2):
        print("✓")
        print(f"    数据形状: X={X.shape}, Y={Y.shape}")
    else:
        raise ValueError(f"数据形状错误: X={X.shape}, Y={Y.shape}")
except Exception as e:
    print("✗")
    print(f"    错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== 第5步：测试正向网络 =====
print("[5/6] 测试正向网络模块...", end=" ")
try:
    from forward_model import ForwardPredictor, train_forward_model
    import numpy as np
    
    # 创建小数据集
    X_test = np.random.uniform(60, 240, (100, 2))
    Y_test = np.random.uniform(-180, 180, 100)
    
    # 训练小网络（快速测试）
    model, scaler, _, _ = train_forward_model(
        X_test, Y_test, epochs=5, batch_size=32, verbose=False
    )
    
    # 测试预测
    X_scaled = scaler.transform(X_test[:10])
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    predictions = model.predict_phase_deg(X_t)
    
    if predictions.shape == (10,):
        print("✓")
        print(f"    网络预测输出: shape={predictions.shape}")
    else:
        raise ValueError(f"预测形状错误: {predictions.shape}")
except Exception as e:
    print("✗")
    print(f"    错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== 第6步：检查输出目录 =====
print("[6/6] 检查输出目录...", end=" ")
try:
    os.makedirs('results', exist_ok=True)
    
    # 测试写入
    test_file = 'results/.test_write'
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    
    print("✓")
    print("    输出目录可写入")
except Exception as e:
    print("✗")
    print(f"    错误: 无法写入输出目录 - {e}")
    sys.exit(1)

# ===== 完成 =====
print("\n" + "="*70)
print("✓ 所有检查通过！可以运行 main.py")
print("="*70)
print("\n快速测试命令:")
print("  python main.py          # 运行完整流程（~10-15 分钟）")
print("  python -c \"from data_generator import MetasurfaceUnitSimulator; ")
print("             MetasurfaceUnitSimulator().generate_dataset(1000)\"")
print("\n更多文档请参考 README.md")
