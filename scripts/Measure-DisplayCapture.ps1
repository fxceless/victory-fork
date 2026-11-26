<#
.SYNOPSIS
    Validate a completed, process-scoped PresentMon capture and report display FPS.
.DESCRIPTION
    Requires the launcher metadata, capture sidecar and system telemetry.
    Scene equivalence is intentionally not inferred from world-frame logs.
    Menu/loading batches, minimized windows, errors, insufficient duration
    and mixed process/swapchain CSVs reject the sample. No process is modified.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Metadata,
    [Parameter(Mandatory)][string]$Capture,
    [Parameter(Mandatory)][string]$PresentCsv,
    [Parameter(Mandatory)][string]$SystemCsv,
    [ValidateRange(1,3600)][double]$MinimumSeconds=120,
    [ValidateRange(1,10000)][int]$MinimumTelemetry=30,
    [switch]$RequireWarm,
    [switch]$RequireNoCompiler,
    [string]$OutputJson
)
$ErrorActionPreference='Stop'
$ci=[Globalization.CultureInfo]::InvariantCulture
function Number([string]$v){
    $n=0.0
    if(![double]::TryParse($v,[Globalization.NumberStyles]::Float,$ci,[ref]$n) -or
       [double]::IsInfinity($n) -or [double]::IsNaN($n)){throw "Invalid numeric field: $v"}
    return $n
}
function RequireFields($row,[string[]]$names){
    foreach($name in $names){if($name -notin $row.PSObject.Properties.Name){throw "Missing column: $name"}}
}
function Quantile($sorted,[double]$p){return $sorted[[int]([math]::Ceiling($p*$sorted.Count)-1)]}
function Low($sorted,[double]$p){
    $count=[math]::Max(1,[math]::Ceiling($sorted.Count*$p))
    return 1000/(($sorted|Select-Object -Last $count|Measure-Object -Average).Average)
}
$meta=Get-Content -LiteralPath $Metadata -Raw|ConvertFrom-Json
$cap=Get-Content -LiteralPath $Capture -Raw|ConvertFrom-Json
RequireFields $meta @('Trial','Commit','Executable','Log','Start')
RequireFields $cap @('PID','PresentMonExit','StartUtc','EndUtc')
if(!$meta.Trial -or !$meta.Commit -or !$meta.Executable -or !$meta.Log){throw 'Incomplete launch metadata'}
if($cap.PresentMonExit -ne 0 -or [long]$cap.PID -le 0){throw 'Invalid capture exit or PID'}
$gamePid=[long]$cap.PID
$start=[datetimeoffset]$cap.StartUtc;$end=[datetimeoffset]$cap.EndUtc
if($start -lt [datetimeoffset]$meta.Start -or $end -le $start -or ($end-$start).TotalSeconds -lt $MinimumSeconds){throw 'Invalid capture time range'}
$rows=@(Import-Csv -LiteralPath $PresentCsv)
if(!$rows.Count){throw 'Empty PresentMon CSV'}
RequireFields $rows[0] @('ProcessID','SwapChainAddress','MsUntilDisplayed','MsBetweenDisplayChange')
if(@($rows|Where-Object {$_.ProcessID -ne [string]$gamePid -or [string]::IsNullOrWhiteSpace($_.SwapChainAddress)}).Count){throw 'Invalid process or swapchain'}
if(@($rows|Group-Object SwapChainAddress).Count -ne 1){throw 'Multiple swapchains'}
$values=[Collections.Generic.List[double]]::new()
$shown=0
foreach($row in $rows){
    if($row.MsUntilDisplayed -eq 'NA' -or $row.MsBetweenDisplayChange -eq 'NA'){continue}
    if([string]::IsNullOrWhiteSpace($row.MsUntilDisplayed) -or [string]::IsNullOrWhiteSpace($row.MsBetweenDisplayChange)){throw 'Missing display timing'}
    $latency=Number $row.MsUntilDisplayed
    $ms=Number $row.MsBetweenDisplayChange
    if($ms -lt 0 -or $latency -lt 0){throw 'Negative display timing'}
    if($ms -eq 0){continue}
    $shown++
    # First displayed interval may include time preceding the trace.
    if($shown -gt 1){$values.Add($ms)}
}
if($values.Count -lt 2){throw 'Too few displayed intervals'}
$sorted=@($values|Sort-Object)
$seconds=($sorted|Measure-Object -Sum).Sum/1000
if($seconds -lt $MinimumSeconds -or $seconds -gt ($end-$start).TotalSeconds+5){throw "Invalid display duration: $seconds"}
$tele=@(Import-Csv -LiteralPath $SystemCsv|Where-Object {$_.UTC -or $_.Minimized -or $_.ProcessWorkingSetBytes})
if($tele.Count -lt $MinimumTelemetry){throw 'Insufficient telemetry'}
RequireFields $tele[0] @('UTC','Minimized','ProcessWorkingSetBytes')
$last=$start
$maxGap=0.0
foreach($t in $tele){
    if($t.Minimized -ne 'False' -or !$t.UTC -or !$t.ProcessWorkingSetBytes){throw 'Missing or minimized telemetry'}
    $at=[datetimeoffset]$t.UTC
    if($at -lt $last -or $at -gt $end.AddSeconds(5)){throw 'Telemetry time outside capture'}
    $maxGap=[math]::Max($maxGap,($at-$last).TotalSeconds);$last=$at
}
$maxGap=[math]::Max($maxGap,($end-$last).TotalSeconds)
if($maxGap -gt 15){throw "Telemetry gap exceeds 15 seconds: $maxGap"}
$compilerObserved='CompilerCount' -in $tele[0].PSObject.Properties.Name
$compilerSamples=$null
if($compilerObserved){
    $compilerSamples=0
    foreach($t in $tele){
        $n=Number $t.CompilerCount
        if($n -lt 0 -or $n -ne [math]::Floor($n)){throw 'Invalid compiler count'}
        if($n -gt 0){$compilerSamples++}
    }
}
if($RequireNoCompiler -and (!$compilerObserved -or $compilerSamples -gt 0)){throw 'Compiler isolation unavailable or violated'}
$lines=Get-Content -LiteralPath $meta.Log
if($lines|Select-String ' ERROR |first-chance exception|unhandled exception|device removed|panicked' -CaseSensitive|Select-Object -First 1){throw 'Title log contains errors'}
$batches=@($lines|Select-String 'frames by kind:'|Where-Object {$at=[datetimeoffset]::Parse(($_.Line -split ' ')[0]);$at -ge $start -and $at -le $end})
$worldMin=[math]::Max(1,[math]::Floor($MinimumSeconds/6))
if($batches.Count -lt $worldMin -or @($batches|Where-Object Line -NotMatch 'frames by kind: [1-9]\d* of the world.*; 0 menu or loading frames').Count){throw 'Not a valid all-world capture'}
$timings=@($lines|Select-String 'usual frame \(the mean of'|Where-Object {$at=[datetimeoffset]::Parse(($_.Line -split ' ')[0]);$at -ge $start -and $at -le $end})
$warm=@($timings|Where-Object Line -Match 'pipelines: 0 made, 0 loaded,').Count
if($RequireWarm -and ($timings.Count -lt $worldMin -or $warm -ne $timings.Count)){throw 'Pipeline activity or missing timing batches'}
# Count per-batch means; these are not per-frame percentiles and overlap.
$patterns=[ordered]@{
 PrepareMs='preparing ([\d.]+),'
 CopyMs='copies ([\d.]+) ms'
 CopyMB='copies [\d.]+ ms ([\d.]+) MB'
 BackMs='copy-back ([\d.]+) ms'
 BackMB='copy-back [\d.]+ ms ([\d.]+) MB'
 IdleMs='card: idle ([\d.]+) ms'
 RefreshMs='textures: refresh ([\d.]+) ms'
 Uploads='(\d+) uploads, staging'
 RemapMs='remapping: \d+ calls ([\d.]+) ms'
}
$stage=[ordered]@{}
foreach($key in $patterns.Keys){
 $series=@($timings|ForEach-Object{if($_.Line -match $patterns[$key]){Number $Matches[1]}}|Sort-Object)
 if($series.Count){$stage[$key]=[ordered]@{Median=(Quantile $series .5);Min=$series[0];Max=$series[-1];Batches=$series.Count}}
}
$result=[pscustomobject]@{
 Trial=$meta.Trial;Commit=$meta.Commit;PID=$gamePid;StartUtc=$start.ToString('o');EndUtc=$end.ToString('o')
 Seconds=$seconds;DisplayFPS=$sorted.Count/$seconds;Intervals=$sorted.Count;PresentRows=$rows.Count
 Low1=(Low $sorted .01);Low01=(Low $sorted .001);Low01SampleCount=[math]::Max(1,[math]::Ceiling($sorted.Count*.001))
 P50ms=(Quantile $sorted .5);P95ms=(Quantile $sorted .95);P99ms=(Quantile $sorted .99);MaxMs=$sorted[-1]
 WorldBatches=$batches.Count;TelemetrySamples=$tele.Count;MaxTelemetryGapSeconds=$maxGap
 PipelineZeroBatches=$warm;TimingBatches=$timings.Count;CompilerObserved=$compilerObserved;CompilerActiveSamples=$compilerSamples
 SceneMatched=$false;StageBatchMeans=$stage
}
if($OutputJson){$result|ConvertTo-Json -Depth 7|Set-Content -LiteralPath $OutputJson -Encoding utf8}
$result
