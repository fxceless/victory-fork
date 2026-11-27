[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Path,
    [ValidateSet(30,60,90,120)][int]$TargetFps = 90,
    [ValidateRange(0,3600)][int]$WarmupSeconds = 5,
    [ValidateRange(1,36000)][int]$MinimumSeconds = 60,
    [string]$OutputJson
)
$ErrorActionPreference = 'Stop'
$intervals = [Collections.Generic.List[double]]::new()
$metadata = @{}
$lastSequence = [uint64]0; $lastTimestamp = [uint64]0; $gaps = 0
$headerSeen = $false
$footerSeen = $false
$reader = [IO.File]::OpenText((Resolve-Path -LiteralPath $Path).Path)
try {
    while ($null -ne ($line = $reader.ReadLine())) {
        if ($line.StartsWith('#')) {
            if ($line -match '^# ([a-z_]+)=(.*)$') {
                if ($metadata.ContainsKey($Matches[1])) { throw 'Duplicate frame-report metadata.' }
                if ($Matches[1] -in @('dropped_samples', 'rejected_presents', 'complete')) { $footerSeen = $true }
                $metadata[$Matches[1]] = $Matches[2]
            }
            continue
        }
        if (-not $headerSeen) {
            if ($line -ne 'Sequence,TimestampUs,FrameTimeUs') { throw 'Unrecognized frame-report columns.' }
            $headerSeen = $true; continue
        }
        if (-not $line) { continue }
        if ($footerSeen) { throw 'Frame data appears after the report footer.' }
        $fields = $line.Split(',')
        if ($fields.Count -ne 3) { throw 'Malformed frame-report row.' }
        $sequence = [uint64]::Parse($fields[0], [Globalization.CultureInfo]::InvariantCulture)
        $timestamp = [uint64]::Parse($fields[1], [Globalization.CultureInfo]::InvariantCulture)
        $interval = [uint64]::Parse($fields[2], [Globalization.CultureInfo]::InvariantCulture)
        if ($sequence -le $lastSequence -or $timestamp -le $lastTimestamp -or $interval -eq 0 -or $interval -gt $timestamp) { throw 'Nonmonotonic or invalid frame-report samples.' }
        if ($sequence -ne $lastSequence + 1) { $gaps++ }
        if ($lastSequence -gt 0 -and $sequence -eq $lastSequence + 1 -and $timestamp - $lastTimestamp -ne $interval) { throw 'Frame intervals disagree with timestamps.' }
        $lastSequence = $sequence; $lastTimestamp = $timestamp
        if ($timestamp - $interval -ge $WarmupSeconds * 1000000.0) { $intervals.Add($interval / 1000.0) }
    }
} finally { $reader.Dispose() }
if ($metadata['format'] -ne 'victory-present-v1') { throw 'Not a VIctory presentation report.' }
if ($intervals.Count -eq 0) { throw 'No frame intervals remain after warm-up.' }
$sorted = $intervals.ToArray()
[Array]::Sort($sorted)
$totalMs = 0.0; $over50 = 0; $overTwo = 0
$budget = 1000.0 / $TargetFps
foreach ($ms in $sorted) {
    $totalMs += $ms
    if ($ms -gt 50) { $over50++ }
    if ($ms -gt 2 * $budget) { $overTwo++ }
}
function Get-Percentile([double]$Percent) {
    return $sorted[[Math]::Max(0, [int][Math]::Ceiling($sorted.Length * $Percent) - 1)]
}
$slowCount = [Math]::Max(1, [int][Math]::Ceiling($sorted.Length / 100.0))
$slowMs = 0.0
for ($n = $sorted.Length - $slowCount; $n -lt $sorted.Length; $n++) { $slowMs += $sorted[$n] }
$average = $sorted.Length * 1000.0 / $totalMs
$lowOne = 1000.0 / ($slowMs / $slowCount)
$p99 = Get-Percentile 0.99
$valid = $metadata['complete'] -eq '1' -and $metadata['dropped_samples'] -eq '0' -and $metadata['rejected_presents'] -eq '0' -and $metadata['frame_generation_requested'] -eq '0' -and $metadata['frame_generation_seen'] -eq '0' -and $metadata['expected_samples'] -eq $lastSequence.ToString([Globalization.CultureInfo]::InvariantCulture) -and $gaps -eq 0
$duration = $totalMs / 1000.0
$meets = $valid -and $duration -ge $MinimumSeconds -and $average -ge $TargetFps -and $lowOne -ge $TargetFps * 0.9 -and $p99 -le $budget * 1.25 -and $over50 -eq 0
$result = [pscustomobject]@{
    Scope='host presentation intervals; graphics quality and scene require external verification'
    TargetFps=$TargetFps; TargetFrameMs=$budget; Samples=$sorted.Length; MeasuredSeconds=$duration
    AverageFps=$average; SlowestOnePercentFps=$lowOne
    P50Ms=(Get-Percentile 0.50); P95Ms=(Get-Percentile 0.95); P99Ms=$p99; P999Ms=(Get-Percentile 0.999)
    MaximumMs=$sorted[-1]; HitchesOver50Ms=$over50; FramesOverTwoBudgets=$overTwo
    SequenceGaps=$gaps; ValidTimingCapture=$valid; MeetsTimingTarget=[bool]$meets
    RequiredSeconds=$MinimumSeconds; WarmupSeconds=$WarmupSeconds
}
if ($OutputJson) { $result | ConvertTo-Json | Set-Content -LiteralPath $OutputJson -Encoding UTF8 }
$result
