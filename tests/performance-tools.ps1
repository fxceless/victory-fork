$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$launcher = Join-Path $root 'scripts\play-performance.ps1'
$analyzer = Join-Path $root 'scripts\analyze-frame-report.ps1'
$expected = @{ reference=90; high=120; mid=60; entry=30 }
foreach ($tier in $expected.Keys) {
    $plan = & $launcher -Exe 'unused' -Tier $tier -ShowSettings
    if ($plan.TargetFps -ne $expected[$tier] -or $plan.FrameCapFps -le $plan.TargetFps) { throw 'Target/cap mapping failed.' }
    if ($plan.Settings.VICTORY_TITLE_FPS -ne [string]$plan.FrameCapFps) { throw 'Title simulation rate was not set with the frame clock.' }
    $ms = [double]::Parse($plan.Settings.VICTORY_FRAME_INTERVAL_MS, [Globalization.CultureInfo]::InvariantCulture)
    if ([Math]::Abs(1000.0 / $ms - $plan.FrameCapFps) -gt 0.001) { throw 'Frame clock and simulation rate disagree.' }
    if (-not $plan.Warm -or $plan.Settings.VICTORY_FRAMEGEN -ne '0') { throw 'Qualification must warm known pipelines and disable frame generation.' }
    if ($plan.Settings.VICTORY_WATER_LIGHT -ne '1' -or $plan.Settings.VICTORY_LEAVE_OUT -ne '') { throw 'Full-pass profile silently disabled rendering passes.' }
}
try { & $launcher -Exe 'unused' -Tier reference -FrameCapFps 60 -ShowSettings | Out-Null; throw 'Low cap accepted.' }
catch { if ($_.Exception.Message -notmatch 'cannot be below') { throw } }
try { & $launcher -Exe 'unused' -Console series-s -ShowSettings | Out-Null; throw 'Invalid output/profile combination accepted.' }
catch { if ($_.Exception.Message -notmatch 'requires Series X') { throw } }
$plan = & $launcher -Exe 'unused' -Console series-s -Output stock -SkipWarm -ShowSettings
if ($plan.Warm) { throw 'SkipWarm was ignored.' }
$probe = Join-Path $PSScriptRoot 'launcher-probe.ps1'
$observed = @(& (Join-Path $root 'scripts\play.ps1') -Exe (Join-Path $root 'README.md') -Runner $probe -Warm -WarmJobs 2 -More @{VICTORY_TEST_EXIT_CODE='0'})
if ($observed.Count -ne 2 -or ($observed[0].Arguments -join '|') -ne 'warm|--used|--jobs|2' -or $observed[1].Arguments[0] -ne 'run') { throw 'Warm-up must finish before game launch.' }
$events = [Collections.Generic.List[object]]::new()
try {
    & (Join-Path $root 'scripts\play.ps1') -Exe (Join-Path $root 'README.md') -Runner $probe -Warm -More @{VICTORY_TEST_EXIT_CODE='9'} | ForEach-Object {$events.Add($_)}
    throw 'Failed warm-up was ignored.'
} catch { if ($_.Exception.Message -notmatch 'warm-up exited with code 9') { throw } }
if ($events.Count -ne 1 -or $events[0].Arguments[0] -ne 'warm') { throw 'Game started after warm-up failure.' }

$testDir = Join-Path ([IO.Path]::GetTempPath()) ('victory-perf-synthetic-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testDir | Out-Null
function Write-SyntheticReport([string]$Name, [int[]]$Times, [string[]]$Footer, [int]$GapAt = -1) {
    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add('# format=victory-present-v1'); $lines.Add('# frame_generation_requested=0')
    $lines.Add('Sequence,TimestampUs,FrameTimeUs')
    $timestamp = 0; $sequence = 0
    for ($n=0; $n -lt $Times.Length; $n++) {
        $timestamp += $Times[$n]; $sequence++
        if ($n -eq $GapAt) { $sequence++ }
        $lines.Add("$sequence,$timestamp,$($Times[$n])")
    }
    foreach ($line in $Footer) { $lines.Add($line) }
    $destination = Join-Path $testDir $Name
    [IO.File]::WriteAllLines($destination, $lines)
    return $destination
}
$footer = @('# dropped_samples=0','# rejected_presents=0','# expected_samples=200','# frame_generation_seen=0','# complete=1')
$good = Write-SyntheticReport 'good.csv' ((1..200 | ForEach-Object {10000})) $footer
$result = & $analyzer -Path $good -TargetFps 90 -WarmupSeconds 0 -MinimumSeconds 1
if (-not $result.MeetsTimingTarget -or [Math]::Abs($result.AverageFps - 100) -gt 0.001 -or $result.P99Ms -ne 10 -or $result.SlowestOnePercentFps -ne 100) { throw 'Constant-rate report analysis failed.' }
$result = & $analyzer -Path $good -TargetFps 120 -WarmupSeconds 0 -MinimumSeconds 1
if ($result.MeetsTimingTarget) { throw '100 FPS incorrectly met the 120 FPS target.' }
$result = & $analyzer -Path $good -TargetFps 90 -WarmupSeconds 0 -MinimumSeconds 3
if ($result.MeetsTimingTarget) { throw 'An undersized capture incorrectly qualified.' }
$hitch = Write-SyntheticReport 'hitch.csv' (@(1..199 | ForEach-Object {10000}) + @(100000)) $footer
$result = & $analyzer -Path $hitch -TargetFps 90 -WarmupSeconds 0 -MinimumSeconds 1
if ($result.MeetsTimingTarget -or $result.HitchesOver50Ms -ne 1 -or $result.SlowestOnePercentFps -ge 90) { throw 'Stutter was hidden by average FPS.' }
$gap = Write-SyntheticReport 'gap.csv' ((1..200 | ForEach-Object {10000})) $footer 100
$result = & $analyzer -Path $gap -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture -or $result.MeetsTimingTarget) { throw 'Missing samples incorrectly qualified.' }
$incomplete = Write-SyntheticReport 'incomplete.csv' ((1..200 | ForEach-Object {10000})) @()
$result = & $analyzer -Path $incomplete -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture -or $result.MeetsTimingTarget) { throw 'Incomplete capture incorrectly qualified.' }
$generated = Join-Path $testDir 'generated.csv'
([IO.File]::ReadAllText($good)).Replace('frame_generation_requested=0','frame_generation_requested=1') | Set-Content -LiteralPath $generated
$result = & $analyzer -Path $generated -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture -or $result.MeetsTimingTarget) { throw 'Generated-frame capture incorrectly qualified base FPS.' }
$runtimeGenerated = Join-Path $testDir 'runtime-generated.csv'
([IO.File]::ReadAllText($good)).Replace('frame_generation_seen=0','frame_generation_seen=1') | Set-Content -LiteralPath $runtimeGenerated
$result = & $analyzer -Path $runtimeGenerated -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture) { throw 'Runtime frame generation was ignored.' }
$rejected = Join-Path $testDir 'rejected-present.csv'
([IO.File]::ReadAllText($good)).Replace('rejected_presents=0','rejected_presents=1') | Set-Content -LiteralPath $rejected
$result = & $analyzer -Path $rejected -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture) { throw 'Rejected presentations incorrectly qualified.' }
$dropped = Write-SyntheticReport 'dropped.csv' ((1..200 | ForEach-Object {10000})) @('# dropped_samples=1','# rejected_presents=0','# expected_samples=200','# frame_generation_seen=0','# complete=1')
$result = & $analyzer -Path $dropped -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture) { throw 'Dropped final samples were ignored.' }
$tail = Write-SyntheticReport 'missing-tail.csv' ((1..200 | ForEach-Object {10000})) @('# dropped_samples=0','# rejected_presents=0','# expected_samples=201','# frame_generation_seen=0','# complete=1')
$result = & $analyzer -Path $tail -WarmupSeconds 0 -MinimumSeconds 1
if ($result.ValidTimingCapture) { throw 'A missing final sample incorrectly qualified.' }
$result = & $analyzer -Path $good -WarmupSeconds 1 -MinimumSeconds 1
if ($result.Samples -ne 100 -or $result.MeasuredSeconds -ne 1) { throw 'Warm-up exclusion is incorrect.' }
$global:LASTEXITCODE = 0
Write-Output 'PASS: four performance targets, coupled clock/simulation settings, warm-up ordering/failure, percentile/low-FPS/hitch analysis, incomplete/gapped/generated/dropped-report rejection and warm-up exclusion.'
