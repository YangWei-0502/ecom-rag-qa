# 电商 RAG（FastAPI :8002 + Streamlit :8512） · Windows 一键启动（默认本地 Ollama；DeepSeek 通过 MODEL_PROVIDER=deepseek 切换备选）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "python"
if (-not (Test-Path $py)) { $py = "python" }

function Test-Port([int]$p) { [bool](Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue) }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已从 .env.example 生成 .env（默认 ollama 模式）" -ForegroundColor Yellow
}

Write-Host '初始化数据库与向量索引（首次较慢，需本地 Ollama 运行）...' -ForegroundColor Cyan
& $py scripts\init_db.py
& $py scripts\reindex.py

if (Test-Port 8002) {
    Write-Host "后端端口 8002 已被占用，视为已在运行，跳过启动" -ForegroundColor Yellow
} else {
    Write-Host "启动 FastAPI 后端（http://localhost:8002）..." -ForegroundColor Cyan
    $env:PYTHONPATH = "."
    $backend = Start-Process -FilePath $py -ArgumentList -m uvicorn server.main:app --port 8002 -PassThru -WindowStyle Minimized
    Start-Sleep -Seconds 6
    if (-not (Test-Port 8002)) {
        Write-Host "后端启动失败（端口 8002 未监听），请查看日志" -ForegroundColor Red
    }
}

if (Test-Port 8512) {
    Write-Host "前端端口 8512 已被占用，视为已在运行，跳过启动（http://localhost:8512）" -ForegroundColor Yellow
    Read-Host "按回车结束"
    if ($backend) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    exit 0
}
Write-Host "启动 Streamlit 前端（http://localhost:8512）..." -ForegroundColor Cyan
try {
    & $py -m streamlit run app\Home.py --server.port 8512 --server.headless true
} finally {
    if ($backend) { Write-Host "关闭前端，同时停止后端 PID=$($backend.Id)..." -ForegroundColor Yellow; Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
}