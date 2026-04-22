# PowerShell 虚拟环境启动脚本

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "     超构表面逆向设计系统 - 虚拟环境启动" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$envPath = Join-Path $PSScriptRoot "metasurface_env\Scripts\Activate.ps1"

if (Test-Path $envPath) {
    & $envPath
    Write-Host "[✓] 虚拟环境已激活`n" -ForegroundColor Green
    
    Write-Host "可用命令:" -ForegroundColor Yellow
    Write-Host "  python main.py         - 运行完整流程（10-15分钟）"
    Write-Host "  python quick_demo.py   - 快速演示（3-5分钟）"
    Write-Host "  python verify.py       - 环境检查"
    Write-Host "  python                 - 进入 Python REPL`n"
    
    Write-Host "虚拟环境位置: $((Get-Command python).Source)" -ForegroundColor Gray
    Write-Host "Python 版本: $(python --version)`n" -ForegroundColor Gray
} else {
    Write-Host "[✗] 错误：未找到虚拟环境！`n" -ForegroundColor Red
    Write-Host "请先创建虚拟环境：" -ForegroundColor Yellow
    Write-Host "  D:\ProgramData\anaconda3\python.exe -m venv metasurface_env" -ForegroundColor Gray
    Write-Host "然后运行：" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt`n" -ForegroundColor Gray
}
