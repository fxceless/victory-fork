# Pre-translate GTA 6 shaders into cache-rework-v7 (offline, no game launch).
#
# Usage:
#   cd C:\Users\Alpacher\Downloads\victory-fork-rework
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\Warm-v7.ps1
#   .\Warm-v7.ps1 -PortCbv    # optional second pass; PORT_CBV changes the cache key
#   .\Warm-v7.ps1 -Jobs 12
#
# Only shaders.rpf is passed (411,695,600 bytes). Do not point this at the
# whole game folder: levels.rpf alone is ~76 GB and is not a shader archive.
#
# Writes UTF-8 (no BOM) progress to logs\warm-v7-YYYYMMDD-HHMMSS.log.

param(
    [switch]$PortCbv,
    [int]$Jobs
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$exe = Join-Path $PSScriptRoot 'target-v7\release\victory.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "v7 release binary not found: $exe`nBuild it first: cargo build --locked --release -p victory --target-dir target-v7"
}

$shaders = 'C:\Users\Alpacher\Downloads\GTA\shaders.rpf'
if (-not (Test-Path -LiteralPath $shaders)) {
    throw "Shader archive not found: $shaders"
}

# Drop leftover Victory switches from an earlier session in this window,
# except path settings the user may have set on purpose.
$keep = @('VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_FIDELITYFX_DIR')
Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $keep } | ForEach-Object {
    Remove-Item "Env:$($_.Name)"
}

$env:VICTORY_GAME_DIR = 'C:\Users\Alpacher\Downloads\GTA'
$env:VICTORY_CACHE_DIR = Join-Path $env:LOCALAPPDATA 'VIctory\cache-rework-v7'
$env:VICTORY_LOG = 'info'
if ($PortCbv) { $env:VICTORY_PORT_CBV = '1' }

New-Item -ItemType Directory -Force -Path $env:VICTORY_CACHE_DIR | Out-Null
$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "warm-v7-$stamp.log"

$args = @('shaders', $shaders)
if ($Jobs -gt 0) { $args += @('--jobs', "$Jobs") }

$banner = @"
VIctory rework v7 shader warm
  exe:     $exe
  archive: $shaders
  cache:   $($env:VICTORY_CACHE_DIR)
  portcbv: $(if ($PortCbv) { 'VICTORY_PORT_CBV=1 (separate cache key)' } else { 'off (default; NVIDIA constant cache kept)' })
  jobs:    $(if ($Jobs -gt 0) { $Jobs } else { 'default (every logical core)' })
  log:     $log
  HEAD:    $((git -C $PSScriptRoot rev-parse --short HEAD))
Starting offline translation. This can take many minutes. You can stop and re-run; cached shaders are kept.
"@

$utf8 = New-Object System.Text.UTF8Encoding $false
$writer = New-Object System.IO.StreamWriter $log, $false, $utf8, 65536
$writer.AutoFlush = $false
$pending = 0
try {
    $writer.WriteLine($banner)
    $writer.WriteLine('')
    Write-Host $banner
    Write-Host ''

    $ErrorActionPreference = 'Continue'
    & $exe @args 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $line = $_.Exception.Message
        } else {
            $line = "$_"
        }
        $writer.WriteLine($line)
        $pending++
        if ($pending -ge 32) { $writer.Flush(); $pending = 0 }
        Write-Host $line
    }
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    $writer.WriteLine('')
    $writer.WriteLine("--- process exit $code ---")
} finally {
    $writer.Flush()
    $writer.Dispose()
}

Write-Host ''
Write-Host "warm log written to: $log"
exit $code