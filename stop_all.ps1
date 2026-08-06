# ============================================================
# 医疗问答助手 - 停止脚本
# 杀掉占用 8000/3000 端口的服务进程
# ============================================================

Write-Host "正在停止服务..." -ForegroundColor Yellow
$stopped = $false
foreach ($port in 8000, 3000) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host "  停止端口 $port 的进程 PID $($_.OwningProcess)" -ForegroundColor Green
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
            $stopped = $true
        }
}
if ($stopped) {
    Write-Host "服务已停止。" -ForegroundColor Green
} else {
    Write-Host "没有检测到运行中的服务（端口 8000/3000 空闲）。" -ForegroundColor DarkGray
}
