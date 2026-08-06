# ============================================================
# 医疗问答助手 - 一键启动脚本（健壮版）
# 用法：
#   1) 双击 start.bat（推荐，自动绕过执行策略）
#   2) 或 PowerShell 里执行：.\run_project.ps1
# 按 Ctrl+C 停止两个服务。日志写入 logs\*.log
# ============================================================

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

$python = Join-Path $projectDir ".venv\Scripts\python.exe"
$logsDir = Join-Path $projectDir "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

Write-Host ""
Write-Host "===== 医疗问答助手 一键启动 =====" -ForegroundColor Cyan

# ---- 1. 前置检查 ----
if (-not (Test-Path $python)) {
    Write-Host "[错误] 虚拟环境不存在: $python" -ForegroundColor Red
    Write-Host "请先执行: python -m venv .venv  并安装依赖" -ForegroundColor Yellow
    Read-Host "按回车退出"; exit 1
}

$kbChroma = Join-Path $projectDir "data\chroma\chroma.sqlite3"
if (-not (Test-Path $kbChroma)) {
    Write-Host "[提示] 未检测到本地知识库 (data\chroma\chroma.sqlite3)" -ForegroundColor Yellow
    Write-Host "       通用问答的知识检索将为空。导入命令见 scripts\import_jsonl_to_chroma.py" -ForegroundColor Yellow
}

# ---- 2. 释放被占用的端口 ----
function Clear-Port([int]$port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        Write-Host "[提示] 端口 $port 被 PID $($c.OwningProcess) 占用，正在停止..." -ForegroundColor Yellow
        Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
}
Clear-Port 8000
Clear-Port 3000

# ---- 3. 启动后端（uvicorn :8000）----
# 子进程继承本控制台（-NoNewWindow，不重定向），日志在一个窗口混流，便于观察。
Write-Host "[启动] 后端 API http://localhost:8000 ..." -ForegroundColor Green
$backend = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory $projectDir -NoNewWindow -PassThru

# ---- 4. 启动前端（:3000）----
Write-Host "[启动] 前端 http://localhost:3000 ..." -ForegroundColor Green
$frontend = Start-Process -FilePath $python `
    -ArgumentList "frontend_server.py" `
    -WorkingDirectory $projectDir -NoNewWindow -PassThru

# ---- 5. 健康检查 ----
function Wait-Healthy([string]$name, [string]$url, [int]$timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($r.StatusCode -eq 200) {
                Write-Host "[就绪] $name -> $url" -ForegroundColor Green
                return $true
            }
        } catch { }
        Start-Sleep -Milliseconds 700
    }
    Write-Host "[失败] $name 健康检查超时 ($timeoutSec 秒)，请查看 logs\" -ForegroundColor Red
    return $false
}

Start-Sleep -Seconds 3  # 给两个进程启动留缓冲
$backendOk = Wait-Healthy "后端" "http://127.0.0.1:8000/docs" 60
$frontendOk = Wait-Healthy "前端" "http://127.0.0.1:3000/" 60

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  后端:  http://127.0.0.1:8000   $($(if ($backendOk) {'[OK]'} else {'[失败]'}))" -ForegroundColor Cyan
Write-Host "  前端:  http://127.0.0.1:3000   $($(if ($frontendOk) {'[OK]'} else {'[失败]'}))" -ForegroundColor Cyan
Write-Host "  日志:  在本控制台窗口实时混流显示（uvicorn 与前端日志）" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

if ($frontendOk) {
    try { Start-Process "http://127.0.0.1:3000" } catch { }
}

# ---- 6. 保持运行；Ctrl+C 或服务异常退出时清理 ----
Write-Host "按 Ctrl+C 停止两个服务..." -ForegroundColor Yellow
try {
    while ($true) {
        Start-Sleep -Seconds 2
        if ($backend.HasExited -or $frontend.HasExited) {
            $gone = @()
            if ($backend.HasExited) { $gone += "后端" }
            if ($frontend.HasExited) { $gone += "前端" }
            Write-Host "[警告] 服务异常退出: $($gone -join '、')，请查看对应日志" -ForegroundColor Red
            break
        }
    }
} finally {
    Write-Host "[停止] 正在停止服务..." -ForegroundColor Yellow
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    if ($frontend -and -not $frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "[完成] 服务已停止" -ForegroundColor Green
}
