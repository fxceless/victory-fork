# Launch the rework/v10-5070 binary (Belobug a-publier cd8dfc0 + v7..v10 fixes)
# on the Xbox Series X profile (default) with the on-screen HUD.
#
# Usage:
#   cd C:\Users\Alpacher\Downloads\victory-fork-rework
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\Launch-Rework-v10.ps1                      # Series X, 4K output, title's own 2/3 render scale
#   .\Launch-Rework-v10.ps1 -RenderScale 0.5     # Series X, 4K output (normal HUD), 1080p rendering
#   .\Launch-Rework-v10.ps1 -Output 2560x1440    # Series X, 1440p output
#   .\Launch-Rework-v10.ps1 -NoWaitResident      # A/B: residency waits off (as v7/v8; risk of page faults)
#   .\Launch-Rework-v10.ps1 -NoExactReuse        # A/B: exact-copy buffer reuse off entirely (as v7)
#   .\Launch-Rework-v10.ps1 -V8ExactReuse        # A/B: v8's unhardened reuse gate (VICTORY_EXACT_REUSE_SAFE=0)
#   .\Launch-Rework-v10.ps1 -CardBuffersMB 512   # A/B: v8's cap on card copies of title buffers (v9: 1024)
#   .\Launch-Rework-v10.ps1 -PortConstants       # A/B: renderer port 1 back on (v7-v9 default; face spikes, GPU 2x)
#   .\Launch-Rework-v10.ps1 -NoPlacedOrder       # A/B: v9's refusal of submissions with a placed copy (VICTORY_PLACED_ORDER off)
#   .\Launch-Rework-v10.ps1 -ThinThreads 8       # list-translating threads (default 6)
#   .\Launch-Rework-v10.ps1 -Ports               # opt-in: also VICTORY_PORT_DESCRIPTORS=1
#   .\Launch-Rework-v10.ps1 -Safe                # Belobug play.ps1 -Safe
#   .\Launch-Rework-v10.ps1 -SeriesS             # not recommended (looks worse; 3 compute shaders refused)
#
# Uses target-v10\release\victory.exe and cache-rework-v7 (v10 does not touch the
# shader translator, so the translation cache key is the same as v7..v9's).
# Does not touch target-v3..v9 or their caches. Native (title 2/3) scale by default.

param(
    [switch]$Safe,
    [switch]$Ports,
    [switch]$PortCbv,
    [switch]$SeriesS,
    [switch]$NoWaitResident,
    [switch]$NoExactReuse,
    [switch]$V8ExactReuse,
    [switch]$PortConstants,
    [switch]$NoPlacedOrder,
    [ValidateRange(64, 4096)][int]$CardBuffersMB = 1024,
    [ValidateRange(0.5, 1.0)][double]$RenderScale = 0,
    [ValidateSet('', '1280x720', '1664x936', '1920x1080', '2560x1440', '3840x2160')][string]$Output = '',
    [ValidateRange(1, 16)][int]$ThinThreads = 6
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
    $env:VICTORY_THIN_THREADS = "$ThinThreads"
    $env:VICTORY_JOIN_PARTS = '8'
    $env:VICTORY_UPLOAD_AHEAD = '1'
    # Measured in run-v7-20261126-213012 (d8c9965) on this PC: SUB 71.6 -> 32.7 ms.
    $env:VICTORY_CARD_BUFFERS_MIN_MB = '1'
    $env:VICTORY_PORT_GUARD = '1'
    $env:VICTORY_PORT_JOURNAL = '1'
    # v10: renderer port 1 off by default. In place, the title's 174 MB heap (its ring of
    # constants, what the skinning passes write) is read by the card from the PC's memory,
    # and CPU-side copies of views in it (5,000 a frame) snapshot bytes the card writes
    # there with nothing waited for: face spikes and GPU 202 ms vs 88 ms (user A/B, v9).
    if ($PortConstants) { $env:VICTORY_PORT_CONSTANTS = '1'; $mode = "$mode + PORT_CONSTANTS=1" }
    # v10: a copy at its place in a list no longer refuses the whole submission to the
    # card steps (v9: 1.0-1.8 a frame, each a 38-49 ms+ wait); VICTORY_PLACED_ORDER=1.
    if (-not $NoPlacedOrder) { $env:VICTORY_PLACED_ORDER = '1' } else { $mode = "$mode + PLACED_ORDER off" }
    # v9: v8 filled the 512 MB default cap (66 copies, 510.6 MB in run-v8-20261126-220247)
    # and 162 MB a frame of copies of 1 MiB or more were then compared on the CPU.
    $env:VICTORY_CARD_BUFFERS_MB = "$CardBuffersMB"
}

# Residency waits: v9 leaves Belobug's default on (new memory waited for until the
# card has it mapped; v8's shared waits keep it cheap). -NoWaitResident is v7/v8.
if ($NoWaitResident) {
    $env:VICTORY_WAIT_RESIDENT = '0'
    $mode = "$mode + WAIT_RESIDENT=0"
}

if ($NoExactReuse) {
    $env:VICTORY_EXACT_REUSE = '0'
    $mode = "$mode + EXACT_REUSE=0"
}
if ($V8ExactReuse) {
    $env:VICTORY_EXACT_REUSE_SAFE = '0'
    $mode = "$mode + EXACT_REUSE_SAFE=0"
}

if ($Ports) {
    $env:VICTORY_PORT_DESCRIPTORS = '1'
    $mode = "$mode + PORT_DESCRIPTORS"
}

if ($PortCbv) {
    $env:VICTORY_PORT_CBV = '1'
    $mode = "$mode + PORT_CBV=1 (separate shader cache key: cold translate)"
}

if ($Output) {
    $env:VICTORY_OUTPUT = $Output
    $mode = "$mode + OUTPUT=$Output"
}

if ($RenderScale -gt 0) {
    $env:VICTORY_RENDER_SCALE = $RenderScale.ToString([Globalization.CultureInfo]::InvariantCulture)
    $mode = "$mode + RENDER_SCALE=$($env:VICTORY_RENDER_SCALE)"
}

$extraArgs = @()
if ($SeriesS) {
    $extraArgs += @('--console', 'series-s')
    $mode = "$mode + SeriesS"
}

$exe = Join-Path $PSScriptRoot 'target-v10\release\victory.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw "v10 release binary not found: $exe`nBuild it first: cargo build --locked --release -p victory --target-dir target-v10"
}

$game = Join-Path $env:VICTORY_GAME_DIR 'game_scarlett_master.exe'
if (-not (Test-Path -LiteralPath $game)) {
    throw "Game executable not found: $game"
}

$logDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $logDir "run-v10-$stamp.log"
New-Item -ItemType Directory -Force -Path $env:VICTORY_CACHE_DIR | Out-Null

$banner = @"
VIctory rework v10 launch
  exe:      $exe
  game:     $game
  mode:     $mode
  console:  $(if ($SeriesS) { 'Xbox Series S' } else { 'Xbox Series X' })
  cache:    $($env:VICTORY_CACHE_DIR)
  adapter:  VICTORY_D3D12_ADAPTER=NVIDIA
  hud:      VICTORY_PANEL=1
  thin:     $(if ($Safe) { 'off' } else { "VICTORY_THIN=1  VICTORY_THIN_THREADS=$ThinThreads" })
  resident: $(if ($NoWaitResident) { 'VICTORY_WAIT_RESIDENT=0 (as v8)' } else { 'waits on (Belobug default)' })
  exact:    $(if ($NoExactReuse) { 'VICTORY_EXACT_REUSE=0 (as v7)' } elseif ($V8ExactReuse) { 'reused, v8 gate' } else { 'reused, v9 gate (64 submissions + 3 frames)' })
  port1:    $(if ($PortConstants) { 'VICTORY_PORT_CONSTANTS=1 (in place)' } else { 'off (v10 default)' })
  placed:   $(if ($NoPlacedOrder) { 'refused as v9' } else { 'VICTORY_PLACED_ORDER=1' })
  card:     $(if ($Safe) { 'Belobug defaults' } else { "VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_CARD_BUFFERS_MB=$CardBuffersMB" })
  output:   $(if ($Output) { $Output } else { 'title default (normal HUD scale)' })
  scale:    $(if ($RenderScale -gt 0) { $RenderScale } else { 'title default (2/3)' })
  stats:    $log
  HEAD:     $((git -C $PSScriptRoot rev-parse --short HEAD))
Starting... (does not overwrite target-v3..v9 or earlier caches)
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
