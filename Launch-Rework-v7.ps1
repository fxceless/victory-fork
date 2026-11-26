# Launch the rework/v7-5070 binary (Belobug a-publier + DexNom overflow fix)
# with Belobug's proven scripts\play.ps1 performance stack + on-screen HUD.
#
# Usage:
#   cd C:\Users\Alpacher\Downloads\victory-fork-rework
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\Launch-Rework-v7.ps1                 # proven play.ps1 stack + HUD (recommended)
#   .\Launch-Rework-v7.ps1 -SeriesS        # Xbox Series S profile (lower RAM, ~5 ms faster)
#   .\Launch-Rework-v7.ps1 -Ports          # opt-in: add experimental VICTORY_PORT_* patches
#   .\Launch-Rework-v7.ps1 -PortCbv        # opt-in: add VICTORY_PORT_CBV=1
#   .\Launch-Rework-v7.ps1 -Output 1920x1080
#   .\Launch-Rework-v7.ps1 -Safe           # Belobug play.ps1 -Safe
#
# Uses target-v7\release\victory.exe and cache-rework-v7.
# Does not touch target-v3/v4/v5/v6 or earlier caches.

param(
    [switch]$Safe,
    [switch]$Ports,
    [switch]$PortCbv,
    [switch]$SeriesS,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Drop leftover VICTORY_* switches from earlier sessions in this window,
# except path settings.
$keep = @('VICTORY_CONFIG', 'VICTORY_GAME_DIR', 'VICTORY_CACHE_DIR', 'VICTORY_DATA_DIR', 'VICTORY_FIDELITYFX_DIR')
Get-ChildItem Env:VICTORY_* | Where-Object { $_.Name -notin $keep } | ForEach-Object {
    Remove-Item "Env:$($_.Name)"
}

$env:VICTORY_GAME_DIR = 'C:\Users\Alpacher\Downloads\GTA'
$env:VICTORY_CACHE_DIR = Join-Path $env:LOCALAPPDATA 'VIctory\cache-rework-v7'
$env:VICTORY_LOG = 'info'
$env:VICTORY_D3D12_ADAPTER = 'NVIDIA'
$env:VICTORY_PANEL = '1'

# Belobug scripts\play.ps1 core performance & stability stack
$env:VICTORY_TEXTURE_NO_WAIT = '1'
$env:VICTORY_RELAY = '2'
$env:VICTORY_WIDE_BACK = '1'
$env:VICTORY_CARD_REHASH_RUNS = '1'
$env:VICTORY_JOIN_COPIES = '1'
$env:VICTORY_INDIRECT_WINDOW = '1'
$env:VICTORY_RELAY_COPY = '1'
$env:VICTORY_CARD_STEPS = '2'
$env:VICTORY_INDIRECT_BATCH = '2'
$env:VICTORY_MAP_THREADS = '1'
$env:VICTORY_TITLE_PRIORITIES = '0'
$env:VICTORY_CARD_AWAKE = '1'
$env:VICTORY_SHOW_AFTER = '1'
$env:VICTORY_AMBIENT = '0.8'
$env:VICTORY_D3D12_DRED = 'light'

# DirectStorage & memory tuning
$env:VICTORY_DSTORAGE_STAGING_MB = '256'
$env:VICTORY_MAP_LEAN = '1'
$env:VICTORY_STREAM_PRIORITY = '1'
$env:VICTORY_FRAME_INTERVAL_MS = '16.667'

$mode = 'play.ps1 stack + HUD'
if ($Safe) {
    $env:VICTORY_UNDER_WIDE = '0'
    $env:VICTORY_LAND_ECHO = '0'
    $mode = 'safe (play.ps1 -Safe)'
} else {
    $env:VICTORY_THIN = '1'
    # 5800X3D is 8c/16t; Belobug default is 8 (tuned on a 12900K).
    $env:VICTORY_THIN_THREADS = '6'
    $env:VICTORY_JOIN_PARTS = '8'
    $env:VICTORY_UPLOAD_AHEAD = '1'
    # NVIDIA RTX 5070 fixes from run-v7 log analysis:
    # - WAIT_RESIDENT=0 eliminates 162,000 synchronous residency stalls (~75.5 s in 4.5 min)
    # - CARD_BUFFERS_MIN_MB=1 offloads ~224 MB/frame of >=1 MiB buffer compares to GPU
    # - Safe renderer ports 1/6/7 (CONSTANTS/GUARD/JOURNAL); PORT_DESCRIPTORS stays opt-in via -Ports
    $env:VICTORY_WAIT_RESIDENT = '0'
    $env:VICTORY_CARD_BUFFERS_MIN_MB = '1'
    $env:VICTORY_PORT_GUARD = '1'
    $env:VICTORY_PORT_JOURNAL = '1'
    $env:VICTORY_PORT_CONSTANTS = '1'
}

if ($Ports) {
    $env:VICTORY_PORT_GUARD = '1'
    $env:VICTORY_PORT_JOURNAL = '1'
    $env:VICTORY_PORT_CONSTANTS = '1'
    $env:VICTORY_PORT_DESCRIPTORS = '1'
    $mode = "$mode + PORT_DESCRIPTORS"
}

if ($PortCbv) {
    $env:VICTORY_PORT_CBV = '1'
    $mode = "$mode + PORT_CBV=1"
}

if ($Output) {
    $env:VICTORY_OUTPUT = $Output
    $mode = "$mode + OUTPUT=$Output"
}

$extraArgs = @()
if ($SeriesS) {
    $extraArgs += @('--console', 'series-s')
    $mode = "$mode + SeriesS"
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
  console:  $(if ($SeriesS) { 'Xbox Series S' } else { 'Xbox Series X' })
  cache:    $($env:VICTORY_CACHE_DIR)
  adapter:  VICTORY_D3D12_ADAPTER=NVIDIA
  hud:      VICTORY_PANEL=1
  stack:    RELAY=2 JOIN_PARTS=$(if ($Safe) { 'off' } else { '8' }) UPLOAD_AHEAD=$(if ($Safe) { 'off' } else { '1' }) CARD_STEPS=2 TEXTURE_NO_WAIT=1 SHOW_AFTER=1
  thin:     $(if ($Safe) { 'off' } else { 'VICTORY_THIN=1  VICTORY_THIN_THREADS=6' })
  ports:    $(if ($Safe) { 'off' } elseif ($Ports) { 'GUARD=1 JOURNAL=1 CONSTANTS=1 DESCRIPTORS=1' } else { 'GUARD=1 JOURNAL=1 CONSTANTS=1 (DESCRIPTORS opt-in via -Ports)' })
  nvidia:   $(if ($Safe) { 'defaults' } else { 'WAIT_RESIDENT=0  CARD_BUFFERS_MIN_MB=1' })
  output:   $(if ($Output) { $Output } else { 'title default (normal HUD scale)' })
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
    & $exe run $game @extraArgs --execute 2>&1 | ForEach-Object {
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