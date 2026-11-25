[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [ValidateRange(1, 10000)]
    [int]$LastBatches = 12
)

$ErrorActionPreference = 'Stop'
# The frame logger emits each frame once in a five-second batch. Durations
# are rounded to milliseconds, so these percentiles have that resolution.
$batches = @(Get-Content -LiteralPath $LogPath | Where-Object {
    $_ -match 'frame times:.*; each: ([0-9 ]+)'
} | Select-Object -Last $LastBatches)
if ($batches.Count -eq 0) {
    throw 'No frame-time batches found. Enable VICTORY_FRAME_LOG=1 and play for at least five seconds.'
}
$frames = @($batches | ForEach-Object {
    if ($_ -match '; each: ([0-9 ]+)') {
        $Matches[1].Trim() -split '\s+' | ForEach-Object { [double]$_ }
    }
})
$sorted = @($frames | Sort-Object)
$totalMs = ($frames | Measure-Object -Sum).Sum
if ($totalMs -le 0) { throw 'No positive frame durations found.' }
function Percentile([double]$quantile) {
    $index = [Math]::Max(0, [Math]::Ceiling($quantile * $sorted.Count) - 1)
    return $sorted[$index]
}
[pscustomobject][ordered]@{
    Batches = $batches.Count
    Frames = $frames.Count
    RecordedSeconds = [Math]::Round($totalMs / 1000, 2)
    AverageFPS = [Math]::Round(1000 * $frames.Count / $totalMs, 2)
    P50Ms = Percentile 0.50
    P95Ms = Percentile 0.95
    P99Ms = Percentile 0.99
    WorstMs = $sorted[-1]
    FramesOver25Ms = @($frames | Where-Object { $_ -gt 25 }).Count
    FramesOver50Ms = @($frames | Where-Object { $_ -gt 50 }).Count
    FramesOver250Ms = @($frames | Where-Object { $_ -gt 250 }).Count
    DurationResolutionMs = 1
}
