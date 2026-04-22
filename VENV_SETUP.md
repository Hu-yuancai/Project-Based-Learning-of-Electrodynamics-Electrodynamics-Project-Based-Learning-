# 虚拟环境使用指南

## ✅ 虚拟环境已创建并配置完成

虚拟环境位置：`./metasurface_env/`  
Python 版本：3.12.7  
所有依赖已安装并通过验证 ✓

---

## 🚀 三种启动方式

### 方式 1：使用 PowerShell
```powershell
.\activate.ps1
```

### 方式 2：使用 CMD
```cmd
activate.bat
```

### 方式 3：手动激活（PowerShell）
```powershell
.\metasurface_env\Scripts\Activate.ps1
```

### 方式 4：手动激活（CMD）
```cmd
metasurface_env\Scripts\activate.bat
```

---

## 📝 激活后的可用命令

激活虚拟环境后，可以运行以下命令：

### 快速演示（推荐先用这个）
```bash
python quick_demo.py
```
⏱️ 运行时间：3-5 分钟  
📊 生成基本的设计结果和可视化

### 完整流程（最终项目）
```bash
python main.py
```
⏱️ 运行时间：10-15 分钟  
📊 生成完整的设计结果和详细的可视化

### 环境检查
```bash
python verify.py
```
✓ 验证所有依赖是否正确安装  
✓ 测试各个模块的功能

### 进入 Python REPL
```bash
python
```

---

## 🔧 已安装的包列表

| 包名 | 版本 | 用途 |
|:---|:---|:---|
| numpy | 1.26.4 | 数值计算 |
| scipy | 1.13.0 | 科学计算 |
| scikit-learn | 1.4.2 | 机器学习 |
| torch | 2.2.1 | 深度学习框架 |
| torchvision | 0.17.1 | 计算机视觉工具 |
| matplotlib | 3.8.4 | 数据可视化 |
| Pillow | 10.2.0 | 图像处理 |
| tqdm | 4.66.2 | 进度条 |

所有版本都经过验证，兼容 Python 3.12，无版本冲突 ✓

---

## ⚠️ 重要提示

### 环境隔离
虚拟环境是完全隔离的。退出虚拟环境后系统 Python 不会受任何影响。

### 退出虚拟环境
```bash
deactivate
```

### 如果虚拟环境出现问题
删除 `metasurface_env/` 文件夹，然后重新运行：
```powershell
D:\ProgramData\anaconda3\python.exe -m venv metasurface_env
.\metasurface_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📊 验证结果

✅ Python 版本：3.12.7  
✅ NumPy：1.26.4（已验证）  
✅ PyTorch：2.2.1+cpu（已验证）  
✅ scikit-learn：1.4.2（已验证）  
✅ matplotlib：3.8.4（已验证）  
✅ 数据生成模块：正常  
✅ 正向网络模块：正常  
✅ 输出目录：可写入  

所有系统检查 **PASSED** ✓

---

## 🎯 立即开始

### 第 1 步：激活虚拟环境
```powershell
.\activate.ps1
```

### 第 2 步：运行快速演示（推荐）
```bash
python quick_demo.py
```

或者直接运行完整流程：
```bash
python main.py
```

### 第 3 步：查看结果
完成后，查看 `results/` 目录中的图表文件。

---

## 📚 更多信息

查看这些文件了解更多：
- `README.md` - 完整技术文档
- `USAGE.md` - 使用指南  
- `QUICKSTART.md` - 快速开始指南
- `INDEX.md` - 项目总览

---

**虚拟环境已完全配置，现在可以开始使用了！** 🎉

```bash
# 立即执行：
.\activate.ps1
python quick_demo.py
```
