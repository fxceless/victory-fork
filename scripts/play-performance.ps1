# Targets for measurement, not claims of achieved FPS. See docs/performance-targets.md.
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$Exe,
    [ValidateSet('reference', 'high', 'mid', 'entry')][string]$Tier = 'reference',
    [ValidateSet('series-x', 'series-s')][string]$Console = 'series-x',
    [ValidateSet('1440', 'stock')][string]$Output = '1440',
    [ValidateSet('stock', 'quality', 'balanced', 'performance')][string]$Upscaling = 'quality',
    [ValidateSet('diag', 'release')][string]$Build = 'release',
    [ValidateRange(0, 1000)][int]$FrameCapFps = 0,
    [switch]$RayTracing,
    [switch]$SkipWarm,
    [switch]$ShowSettings,
    [string]$ReportDirectory,
    [string]$Runner,
    [hashtable]$More = @{}
)
$ErrorActionPreference = 'Stop'
if ($Console -eq 'series-s' -and $Output -eq '1440') { throw 'The existing 1440p output patch requires Series X; choose -Output stock for Series S.' }
$targets = @{reference=90; high=120; mid=60; entry=30}
$targetFps = $targets[$Tier]
if ($FrameCapFps -eq 0) { $FrameCapFps = [int][Math]::Ceiling($targetFps * 110.0 / 100.0) }
if ($FrameCapFps -lt $targetFps) { throw 'The frame cap cannot be below this tier target.' }
$interval = (1000.0 / $FrameCapFps).ToString('F9', [Globalization.CultureInfo]::InvariantCulture)
$workers = [Math]::Max(1, [Math]::Min(4, [Math]::Floor([Environment]::ProcessorCount / 3.0)))
$stageMs = [Math]::Max(1, [Math]::Floor((1000.0 / $targetFps) / 8.0))
if (-not $ReportDirectory) {
    $ReportDirectory = Join-Path $env:LOCALAPPDATA ('victory\benchmarks\' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
}
$ReportDirectory = [IO.Path]::GetFullPath($ReportDirectory)
$settings = @{
    VICTORY_FRAME_INTERVAL_MS = $interval
    VICTORY_TITLE_FPS = [string]$FrameCapFps
    VICTORY_MEMORY_PROFILE = $(if ($Tier -eq 'entry') { '16gb-8gb' } else { 'auto' })
    VICTORY_PERF_TARGET_FPS = [string]$targetFps
    VICTORY_FRAME_REPORT = (Join-Path $ReportDirectory 'frames.csv')
    VICTORY_PIPELINE_BOOK = '1'
    VICTORY_PIPELINE_WAIT_SHARED = '1'
    VICTORY_PIPELINE_WORKERS = [string]$workers
    VICTORY_DYNAMIC_WAIT = '1'
    VICTORY_TEXTURE_CATCHUP = '0'
    VICTORY_TEXTURE_STAGE_MS = [string]$stageMs
    VICTORY_TEXTURE_STAGE_MB = '32'
    VICTORY_POOL_GATE = '1'
    VICTORY_LEVELS_ON_DEMAND = '1'
    VICTORY_BVH_BUILDER = $(if ($RayTracing) { '1' } else { '0' })
    VICTORY_DLSS = '0'
    VICTORY_FRAMEGEN = '0'
    VICTORY_FSR_PATH = $(if ($Upscaling -eq 'stock') { '0' } else { '1' })
    VICTORY_DLSS_MODE = $(if ($Upscaling -eq 'stock') { 'quality' } else { $Upscaling })
    VICTORY_NATIVE_1440P = $(if ($Output -eq '1440') { '1' } else { '0' })
    VICTORY_RENDER_1080P = '0'
    VICTORY_WATER_LIGHT = '1'
    VICTORY_LEAVE_OUT = ''
    VICTORY_SKIP_PIPELINES = 'none'
}
foreach ($key in $More.Keys) { $settings[$key] = $More[$key] }
if ($ShowSettings) {
    return [pscustomobject]@{
        TargetFps=$targetFps; FrameCapFps=$FrameCapFps; Console=$Console; Output=$Output
        Warm=(-not $SkipWarm); WarmJobs=$workers; Settings=$settings
        Visuals='title rendering passes enabled; PC Ultra preset is not exposed'
    }
}
if ($PSCmdlet.ShouldProcess($Exe, "Run $Tier target ($targetFps+ FPS), $Console/$Output/$Upscaling; results unverified")) {
    Write-Host "Target $targetFps+ FPS; cap $FrameCapFps FPS. This is not an achieved-performance claim."
    & (Join-Path $PSScriptRoot 'play.ps1') -Exe $Exe -Build $Build -Console $Console -Runner $Runner -Warm:(-not $SkipWarm) -WarmJobs $workers -More $settings
}
