$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendHost = if ($env:DASHBOARD_BACKEND_HOST) { $env:DASHBOARD_BACKEND_HOST } else { "127.0.0.1" }
$BackendPort = if ($env:DASHBOARD_BACKEND_PORT) { [int]$env:DASHBOARD_BACKEND_PORT } else { 8010 }
$BackendUrl = "http://${BackendHost}:${BackendPort}"
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Không tìm thấy Python venv tại $Python"
}

$backendJob = $null
$reuseBackend = $false

try {
    $health = Invoke-RestMethod "$BackendUrl/api/health" -TimeoutSec 2
    if ($health.ok) {
        $reuseBackend = $true
        Write-Host "Backend already ready: $BackendUrl"
    }
}
catch {
    $reuseBackend = $false
}

if (-not $reuseBackend) {
    Write-Host "Starting backend: $BackendUrl"
    $backendJob = Start-Job -Name "sv16-dashboard-backend" -ScriptBlock {
        param($ProjectRoot, $HostName, $Port)
        Set-Location $ProjectRoot
        & ".\.venv\Scripts\python.exe" -m dashboard.backend.run --host $HostName --port $Port --strict-port --no-reload
    } -ArgumentList $Root.Path, $BackendHost, $BackendPort
}

try {
    $ready = $reuseBackend
    if (-not $ready) {
        for ($i = 0; $i -lt 90; $i++) {
            Start-Sleep -Seconds 1

            if ($backendJob.State -ne "Running") {
                Receive-Job $backendJob -Keep | Write-Host
                throw "Backend dừng trước khi sẵn sàng."
            }

            try {
                $health = Invoke-RestMethod "$BackendUrl/api/health" -TimeoutSec 2
                if ($health.ok) {
                    $ready = $true
                    break
                }
            }
            catch {
                # Backend đang khởi động và load model; thử lại.
            }
        }
    }

    if (-not $ready) {
        Receive-Job $backendJob -Keep | Write-Host
        throw "Backend chưa sẵn sàng tại $BackendUrl sau thời gian chờ."
    }

    Write-Host "Backend ready: $BackendUrl/api/health"
    $env:VITE_BACKEND_URL = $BackendUrl

    Set-Location (Join-Path $Root "dashboard\frontend")
    Write-Host "Starting frontend. Nếu 5173 đang bận, Vite sẽ tự chọn cổng kế tiếp."
    npm run dev -- --host 127.0.0.1 --port 5173
}
finally {
    if ($backendJob) {
        Stop-Job $backendJob -ErrorAction SilentlyContinue
        Remove-Job $backendJob -Force -ErrorAction SilentlyContinue
    }
}
