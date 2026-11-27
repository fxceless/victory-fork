# Small-memory settings for the local branch; see docs/local-low-memory.md.
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)][string]$Exe,
    [ValidateSet('diag', 'release')][string]$Build = 'release',
    [ValidateSet('stock', 'quality', 'balanced', 'performance')][string]$Upscaling = 'quality',
    [switch]$RayTracing,
    [switch]$ShowSettings,
    [string]$Runner,
    [hashtable]$More = @{}
)
$ErrorActionPreference = 'Stop'
$settings = @{
    VICTORY_MEMORY_PROFILE = '16gb-8gb'
    VICTORY_TEXTURE_BUDGET_MB = '2560'
    VICTORY_TEXTURE_ROOM_PERCENT = '85'
    VICTORY_TEXTURE_STAGE_MB = '32'
    VICTORY_TEXTURE_STAGE_MS = '4'
    VICTORY_TEXTURE_CATCHUP = '0'
    VICTORY_PIPELINE_BOOK = '1'
    VICTORY_LEVELS_ON_DEMAND = '1'
    VICTORY_POOL_GATE = '1'
    VICTORY_RIPE_BLOCKS = '4'
    VICTORY_BVH_BUILDER = $(if ($RayTracing) { '1' } else { '0' })
    VICTORY_DLSS = '0'
    VICTORY_FRAMEGEN = '0'
    VICTORY_NATIVE_1440P = '0'
    VICTORY_RENDER_1080P = '0'
    VICTORY_FSR_PATH = $(if ($Upscaling -eq 'stock') { '0' } else { '1' })
    # The existing runtime uses this mode selector for its native FSR path too.
    VICTORY_DLSS_MODE = $(if ($Upscaling -eq 'stock') { 'quality' } else { $Upscaling })
}
foreach ($key in $More.Keys) { $settings[$key] = $More[$key] }
if ($ShowSettings) {
    return [pscustomobject]@{
        Exe = $Exe; Build = $Build; Console = 'series-s'; Settings = $settings
    }
}
if ($PSCmdlet.ShouldProcess($Exe, "Launch Series S with small-memory settings and $Upscaling upscaling")) {
    & (Join-Path $PSScriptRoot 'play.ps1') -Exe $Exe -Build $Build -Console series-s -Runner $Runner -More $settings
}
