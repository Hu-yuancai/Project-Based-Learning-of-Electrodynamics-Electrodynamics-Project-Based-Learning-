@echo off
REM 虚拟环境快速启动脚本

cd /d "%~dp0"

echo.
echo ============================================================
echo      超构表面逆向设计系统 - 虚拟环境启动
echo ============================================================
echo.

if exist metasurface_env\Scripts\activate.bat (
    call metasurface_env\Scripts\activate.bat
    echo [✓] 虚拟环境已激活
    echo.
    echo 可用命令:
    echo   python main.py         - 运行完整流程（10-15分钟）
    echo   python quick_demo.py   - 快速演示（3-5分钟）
    echo   python verify.py       - 环境检查
    echo   python                 - 进入 Python REPL
    echo.
    cmd /k
) else (
    echo.
    echo [错误] 未找到虚拟环境！
    echo 请先创建虚拟环境：
    echo   D:\ProgramData\anaconda3\python.exe -m venv metasurface_env
    echo   然后运行：pip install -r requirements.txt
    echo.
    pause
)
