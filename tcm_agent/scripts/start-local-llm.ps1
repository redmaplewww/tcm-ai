# 中医AI - 本地LLM继服务 + SSH反向隧道 一键启动
# 用法：右键"以管理员运行"或双击（需要保持窗口开启）
# 架构：red服务器(127.0.0.1:9000) --SSH隧道--> 本机(127.0.0.1:8001继服务) --HTTPS--> GLM

$ErrorActionPreference = "Continue"
$VENV_PY = "F:\zcode\中医AI\.venv\Scripts\python.exe"
$RELAY_PORT = 8001      # 本机继服务端口
$REMOTE_PORT = 9000     # red服务器上的隧道入口端口

Write-Host "=== 中医AI 本地LLM链路启动 ===" -ForegroundColor Cyan

# 1) 启动继服务（独立进程）
Write-Host "[1/2] 启动LLM继服务 (127.0.0.1:$RELAY_PORT)..."
$relay = Start-Process -FilePath $VENV_PY -ArgumentList "-m","tcm_agent.llm_relay","--port",$RELAY_PORT `
    -WorkingDirectory "F:\zcode\中医AI\tcm_agent" -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 4

# 健康检查
try {
    $h = Invoke-RestMethod "http://127.0.0.1:$RELAY_PORT/healthz" -TimeoutSec 5
    if ($h.has_key) { Write-Host "  继服务OK：上游=$($h.upstream) 模型=$($h.model)" -ForegroundColor Green }
    else { Write-Host "  警告：继服务未找到LLM密钥！" -ForegroundColor Yellow }
} catch { Write-Host "  警告：继服务健康检查失败：$_" -ForegroundColor Yellow }

# 2) SSH反向隧道（自动重连循环）
Write-Host "[2/2] 建立SSH反向隧道 red:127.0.0.1:$REMOTE_PORT -> 本机:$RELAY_PORT ..."
while ($true) {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "  [$ts] 建立隧道..."
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=3 `
        -o StrictHostKeyChecking=accept-new `
        -R "127.0.0.1:${REMOTE_PORT}:127.0.0.1:${RELAY_PORT}" red
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "  [$ts] 隧道断开，5秒后重连（Ctrl+C退出）" -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
