# Launch the rework/v7-5070 binary (Belobug a-publier + DexNom overflow fix).
#
# Usage:
#   cd C:\Users\Alpacher\Downloads\victory-fork-rework
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\Launch-Rework-v7.ps1              # safe high-ROI NVIDIA switches (PORT_CBV off)
#   .\Launch-Rework-v7.ps1 -Safe        # Belobug defaults only, no VICTORY_PORT_*
#   .\Launch-Rework-v7.ps1 -PortCbv     # A/B: also set VICTORY_PORT_CBV=1
#
# Uses target-v7\release\victory.exe and a fresh cache-rework-v7.
# Does not touch target-v3/v4/v5/v6 or cache-rework / cache-rework-v4 / cache-rework-v6.
#
# PORT_CBV is off by default: it works on NVIDIA but drops the constant-cache
# path. -PortCbv uses a different shader-cache key; warm the default cache
# first, then optionally .\Warm-v7.ps1 -PortCbv.
#
# Full console output is teed as UTF-8 (no BOM) to
# logs\run-v7-YYYYMMDD-HHMMSS.log. The writer is buffered (64 KiB,
# AutoFlush off, flush every 32 lines) so logging does not back-pressure
# Victory. Native stderr is converted to text so Windows PowerShell does
# not paint a red NativeCommandError banner.

param(
    [switch]$Safe,
    [switch]$PortCbv
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Drop leftovers from a v1-v6 or Belobug play.ps1 session in this window,
# except path settings.
$keep = @('VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_FIDELITYFX_DIR')
Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $keep } | ForEach-Object {
    Remove-Item "Env:$($_.Name)"
}

$env:VICTORY_GAME_DIR = 'C:\Users\Alpacher\Downloads\GTA'
$env:VICTORY_CACHE_DIR = Join-Path $env:LOCALAPPDATA 'VIctory\cache-rework-v7'
$env:VICTORY_LOG = 'info'
# Always pick the discrete NVIDIA adapter on this machine (not a PORT_* switch).
$env:VICTORY_D3D12_ADAPTER = 'NVIDIA'
# Device-loss fix: do not show a texture slot before its upload ran.
# Belobug scripts/play.ps1 sets this even in -Safe; it is not a PORT_* switch.
$env:VICTORY_SHOW_AFTER = '1'

$mode = 'default'
if ($Safe) {
    $mode = 'safe (Belobug defaults; no VICTORY_PORT_*)'
} else {
    $env:VICTORY_PORT_GUARD = '1'
    $env:VICTORY_PORT_JOURNAL = '1'
    $env:VICTORY_PORT_CONSTANTS = '1'
    $env:VICTORY_PORT_DESCRIPTORS = '1'
    $env:VICTORY_THIN = '1'
    # 5800X3D is 8c/16t; Belobug default is 8 (tuned on a 12900K).
    $env:VICTORY_THIN_THREADS = '6'
    $env:VICTORY_DSTORAGE_STAGING_MB = '256'
    # Already default-on in Belobug unless set to 0; set explicitly.
    $env:VICTORY_MAP_LEAN = '1'
    $env:VICTORY_STREAM_PRIORITY = '1'
    $env:VICTORY_OUTPUT = '1920x1080'
    $env:VICTORY_FRAME_INTERVAL_MS = '16.667'
}

if ($PortCbv) {
    $env:VICTORY_PORT_CBV = '1'
    $mode = "$mode + PORT_CBV=1"
}

$exe = Join-Path $PSScriptRoot 'target-v7\release\victory.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "v7 release binary not found: $exe`nBuild it first: cargo build --locked --release -p victory --target-dir target-v7"
}

$game = Join-Path $env:VICTORY_GAME_DIR 'game_scarlett_master.exe'
if (-not (Test-Path -LiteralPath $game)) {
    throw "Game executable not found: $game"
}

$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "run-v7-$stamp.log"
New-Item -ItemType Directory -Force -Path $env:VICTORY_CACHE_DIR | Out-Null

$banner = @"
VIctory rework v7 launch
  exe:      $exe
  game:     $game
  mode:     $mode
  cache:    $($env:VICTORY_CACHE_DIR)
  adapter:  VICTORY_D3D12_ADAPTER=NVIDIA
  show:     VICTORY_SHOW_AFTER=1
  ports:    $(if ($Safe) { 'none' } else { 'GUARD=1 JOURNAL=1 CONSTANTS=1 DESCRIPTORS=1' })
  cbv:      $(if ($PortCbv) { 'VICTORY_PORT_CBV=1 (A/B; loses NVIDIA constant cache; different shader key)' } else { 'off' })
  thin:     $(if ($Safe) { 'unset' } else { 'VICTORY_THIN=1  VICTORY_THIN_THREADS=6' })
  io:       $(if ($Safe) { 'Belobug defaults' } else { 'DSTORAGE_STAGING_MB=256  MAP_LEAN=1  STREAM_PRIORITY=1' })
  output:   $(if ($Safe) { 'title default' } else { 'VICTORY_OUTPUT=1920x1080  FRAME_INTERVAL_MS=16.667' })
  stats:    $log
  HEAD:     $((git -C $PSScriptRoot rev-parse --short HEAD))
Starting... (does not overwrite target-v3/v4/v5/v6 or earlier caches)
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
    & $exe run $game --execute 2>&1 | ForEach-Object {
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
Write-Host "stats log written to: $log"
exit $code