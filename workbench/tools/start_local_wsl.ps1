param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [int]$Port = 8765,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $projectDir '.venv\Scripts\python.exe'
$dataPath = (Resolve-Path -LiteralPath $DataDir).Path
if (!(Test-Path -LiteralPath (Join-Path $dataPath 'platform.db'))) {
    throw 'The selected data directory must contain platform.db.'
}
if (!(Test-Path -LiteralPath $pythonPath)) {
    throw 'The project control environment .venv is missing.'
}
$activePath = Join-Path $dataPath 'active-compute.json'
if (!(Test-Path -LiteralPath $activePath)) { throw 'Validate and activate the local compute target first.' }
$active = Get-Content -LiteralPath $activePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($active.profile.host -ne '127.0.0.1' -or $active.profile.port -ne 22226) {
    throw 'The selected workspace is not configured for the local WSL target.'
}
& wsl.exe -d Ubuntu-24.04 -u root -- systemctl start vision-workbench-ssh.service
if ($LASTEXITCODE -ne 0) { throw 'Unable to start the WSL training endpoint.' }
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "Port $Port is already in use. Open the existing workbench or choose a different port." }
$logDir = Join-Path $projectDir '.runtime\local-wsl'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$arguments = '"{0}" --port {1} --data-dir "{2}"' -f (Join-Path $projectDir 'run.py'), $Port, $dataPath
$service = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $projectDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'server.out.log') -RedirectStandardError (Join-Path $logDir 'server.err.log')
$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 500
    if ($service.HasExited) { throw "The workbench exited. Check $logDir." }
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        $runtime = Invoke-RestMethod "http://127.0.0.1:$Port/api/runtime" -TimeoutSec 2
        if ($health.status -eq 'ok' -and $runtime.data_dir -eq $dataPath) {
            $ready = $true
            break
        }
    } catch { }
}
if (!$ready) { throw "The workbench has not become ready. Check $logDir." }
Write-Output "Workbench: http://127.0.0.1:$Port/ (PID $($service.Id))"
if (!$NoBrowser) { Start-Process "http://127.0.0.1:$Port/" }
