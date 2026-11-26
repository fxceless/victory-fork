param(
    [Parameter(Mandatory=$true)][string]$Log,
    [long]$FirstFrame = 0,
    [long]$LastFrame = [long]::MaxValue
)
$ErrorActionPreference = 'Stop'
if ($FirstFrame -lt 0 -or $LastFrame -lt $FirstFrame) { throw 'Invalid frame window' }
$frames = [System.Collections.Generic.List[double]]::new()
$previous = $null
$reader=[IO.File]::OpenText((Resolve-Path -LiteralPath $Log).Path)
try {
while ($null -ne ($line=$reader.ReadLine())) {
    if ($line -notmatch 'frame_intervals_us first=(\d+) last=(\d+) values=([0-9,]+)') { continue }
    $first = [long]$Matches[1]
    $last = [long]$Matches[2]
    $values = $Matches[3].Split(',')
    if ($last -lt $first -or $values.Length -ne $last-$first+1) { throw 'Malformed frame block' }
    if ($null -ne $previous -and $first -ne $previous+1) { throw 'Discontinuous or mixed-run frame log' }
    for ($i=0;$i -lt $values.Length;$i++) {
        $us = [double]$values[$i]
        if ($us -le 0) { throw 'Invalid zero-length interval' }
        $frame = $first+$i
        if ($frame -ge $FirstFrame -and $frame -le $LastFrame) { $frames.Add($us/1000) }
    }
    $previous = $last
}
} finally { $reader.Dispose() }
if ($frames.Count -eq 0) { throw 'No exact frame intervals in selected window' }
$sorted = $frames.ToArray()
[Array]::Sort($sorted)
$n = $sorted.Length
$sum = ($sorted | Measure-Object -Sum).Sum
function Percentile([double]$p) {
    $sorted[[Math]::Max(0,[int][Math]::Ceiling($p*$n)-1)]
}
function SlowMeanFps([double]$fraction) {
    $k = [Math]::Max(1,[int][Math]::Ceiling($fraction*$n))
    $sumSlow = 0.0
    for ($i=$n-$k;$i -lt $n;$i++) { $sumSlow += $sorted[$i] }
    1000*$k/$sumSlow
}
[pscustomobject]@{
    Measurement = 'CPU present-call intervals, not display timestamps'
    Frames = $n
    Seconds = [Math]::Round($sum/1000,3)
    AverageFPS = [Math]::Round(1000*$n/$sum,3)
    Low1PercentFPS = [Math]::Round((SlowMeanFps 0.01),3)
    LowPoint1PercentFPS = [Math]::Round((SlowMeanFps 0.001),3)
    P50ms = [Math]::Round((Percentile 0.5),3)
    P95ms = [Math]::Round((Percentile 0.95),3)
    P99ms = [Math]::Round((Percentile 0.99),3)
    Maxms = $sorted[$n-1]
    SmallSampleWarning = ($n -lt 1000)
}
