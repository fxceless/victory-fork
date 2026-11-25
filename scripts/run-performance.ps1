[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameExe,
    [ValidateSet('series-x', 'series-s')]
    [string]$Console = 'series-x',
    [ValidateRange(0, 240)]
    [int]$FrameRate = 0,
    [switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$binary = Join-Path $repo 'target\release\victory.exe'
$gamePath = (Resolve-Path -LiteralPath $GameExe).Path
$repoPrefix = [IO.Path]::GetFullPath($repo).TrimEnd('\') + '\'
if ($gamePath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The game must be outside the source repository.'
}
if (-not (Test-Path -LiteralPath $binary -PathType Leaf)) {
    throw 'Build first: cargo build --release -p victory'
}

# Keep output, caches, and game files outside the repository.
$runDir = Join-Path $env:LOCALAPPDATA ('victory\benchmarks\' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
if ([IO.Path]::GetFullPath($runDir).StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Benchmark output must be outside the source repository.'
}
$settings = [ordered]@{
    VICTORY_LEVELS_ON_DEMAND = '1'
    VICTORY_ASYNC_SUBMIT = '1'
    VICTORY_QUEUE2 = '1'
    VICTORY_INDIRECT_WINDOW = '1'
    VICTORY_TEXTURE_NO_WAIT = '1'
    VICTORY_TEXTURE_BACK = '1'
    VICTORY_ABSENT_ONCE = '1'
    VICTORY_ABSENT_MS = 'told'
    VICTORY_MERGE_GAP = '64'
    VICTORY_MERGE_RUNS = '4096'
    VICTORY_STAGING_POOL = '0'
    VICTORY_COPY_PLAIN = '1'
    VICTORY_JOIN_PARTS = '1'
    VICTORY_BACK_OWN_VIEW = '1'
    VICTORY_INDIRECT_GATHER = '1'
    VICTORY_CARD_REHASH_RUNS = '1'
    VICTORY_CARD_BUFFERS_MIN_KB = '64'
    VICTORY_CARD_BUFFERS_MB = '1280'
    VICTORY_CARD_HASH_WIDE = '1'
    VICTORY_CARD_HASH_LEN = '1'
    VICTORY_SHADER_OPT = '63'
    VICTORY_TEXTURE_STAGE_MB = '64'
    VICTORY_TEXTURE_STAGE_MS = '30'
    VICTORY_TEXTURE_CATCHUP = '0'
    VICTORY_FRAME_LOG = '1'
    VICTORY_CARD_METER = '1'
}
$settings['VICTORY_FRAME_RATE'] = if ($FrameRate -gt 0) { [string]$FrameRate } else { '' }
$frameRateLabel = if ($FrameRate -gt 0) { "$FrameRate FPS" } else { 'title default' }

Write-Output "Executable: $binary"
Write-Output "Game: $gamePath"
Write-Output "Console: $Console"
Write-Output "Frame rate override: $frameRateLabel"
Write-Output "Logs: $runDir"
if ($PrepareOnly) {
    Write-Output 'Paths validated; no game process started.'
    return
}

New-Item -ItemType Directory -Path $runDir | Out-Null
$previous = @{}
try {
    # Restore the invoking shell after the child inherits these settings.
    foreach ($name in $settings.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    $settings | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'settings.json')
    $process = Start-Process -FilePath $binary -WorkingDirectory $repo `
        -ArgumentList @('run', ('"' + $gamePath + '"'), '--execute', '--console', $Console) `
        -RedirectStandardOutput (Join-Path $runDir 'stdout.log') `
        -RedirectStandardError (Join-Path $runDir 'stderr.log') -PassThru
    Write-Output "Started PID $($process.Id). Play the same scene for at least 60 seconds, then close the game normally."
} finally {
    foreach ($name in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}
