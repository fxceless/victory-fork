<#
.SYNOPSIS
    Offline regression cases for the PresentMon validator.
#>
$ErrorActionPreference='Stop'
$measure=Join-Path $PSScriptRoot 'Measure-DisplayCapture.ps1'
$dir=Join-Path ([IO.Path]::GetTempPath()) ('victory-display-test-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $dir|Out-Null
$start=[datetimeoffset]::Parse('2026-01-01T10:00:00Z')
$end=$start.AddSeconds(130)
$meta=Join-Path $dir 'meta.json'
$cap=Join-Path $dir 'capture.json'
$csv=Join-Path $dir 'present.csv'
$tele=Join-Path $dir 'system.csv'
$log=Join-Path $dir 'title.log'
$argsMap=@{Metadata=$meta;Capture=$cap;PresentCsv=$csv;SystemCsv=$tele;RequireWarm=$true;RequireNoCompiler=$true}
$passed=0
function Fixture {
 [pscustomobject]@{Trial='synthetic';Commit='synthetic-commit';Executable='synthetic.exe';Log=$log;Start=$start.AddSeconds(-10).ToString('o')}|ConvertTo-Json|Set-Content $meta
 [pscustomobject]@{PID=4242;PresentMonExit=0;StartUtc=$start.ToString('o');EndUtc=$end.ToString('o')}|ConvertTo-Json|Set-Content $cap
 # Deliberately use 40 ms intervals and 25 FPS for an exact check.
 $lines=@('ProcessID,SwapChainAddress,MsUntilDisplayed,MsBetweenDisplayChange','4242,0x1,4,40')
 $lines+=1..3250|ForEach-Object{'4242,0x1,4,40'}
 $lines|Set-Content $csv
 $rows=0..64|ForEach-Object{[pscustomobject]@{UTC=$start.AddSeconds(2*$_).ToString('o');Minimized='False';ProcessWorkingSetBytes=12345;CompilerCount=0}}
 $rows|Export-Csv $tele -NoTypeInformation
 $lines=1..26|ForEach-Object{
  $at=$start.AddSeconds(5*$_).ToString('o')
  "$at INFO victory_gpu::framelog: frames by kind: 125 of the world in 5.0 s; 0 menu or loading frames"
  "$at INFO victory_gpu::framelog: usual frame (the mean of 125 not long): 40 ms | preparing 4.0, waiting for the card 2.0; copies 3.0 ms 4.0 MB; copy-back 1.0 ms 2.0 MB | card: idle 5.0 ms | textures: refresh 3.0 ms, 2 uploads, staging | remapping: 3 calls 0.1 ms | pipelines: 0 made, 0 loaded,"
 }
 $lines|Set-Content $log
}
function Reject([string]$name,[scriptblock]$change,[string]$pattern){
 Fixture
 & $change
 $message=$null
 try { & $measure @argsMap|Out-Null } catch { $message=$_.Exception.Message }
 if(!$message -or $message -notmatch $pattern){throw "$name did not reject as expected: $message"}
 $script:passed++
 "$name : passed"
}
try {
 Fixture
 $m=& $measure @argsMap
 if($m.DisplayFPS -ne 25 -or $m.Seconds -ne 130 -or $m.Intervals -ne 3250 -or $m.P95ms -ne 40 -or $m.Low1 -ne 25 -or $m.SceneMatched -or $m.CompilerActiveSamples -ne 0){throw 'valid fixture metrics mismatch'}
 $passed++
 'valid fixture : passed'
 Reject 'foreign PID' {Add-Content $csv '9999,0x1,4,40'} 'Invalid process'
 Reject 'multiple swapchains' {Add-Content $csv '4242,0x2,4,40'} 'Multiple swapchains'
 Reject 'negative timing' {Add-Content $csv '4242,0x1,4,-40'} 'Negative'
 Reject 'nonfinite timing' {Add-Content $csv '4242,0x1,4,NaN'} 'Invalid numeric'
 Reject 'short display sample' {(Get-Content $csv -TotalCount 101)|Set-Content $csv} 'Invalid display duration'
 Reject 'minimized window' { (Get-Content $tele).Replace('"False"','"True"')|Set-Content $tele } 'minimized'
 Reject 'compiler overlap' { $rows=Import-Csv $tele;$rows[10].CompilerCount=1;$rows|Export-Csv $tele -NoTypeInformation } 'Compiler isolation'
 Reject 'telemetry gap' { $rows=Import-Csv $tele;$rows|Where-Object {([datetimeoffset]$_.UTC-$start).TotalSeconds -notin (40..64)}|Export-Csv $tele -NoTypeInformation } 'Telemetry gap'
 Reject 'menu frame' { (Get-Content $log).Replace('0 menu or loading frames','1 menu or loading frames')|Set-Content $log } 'all-world'
 Reject 'cold pipeline' { (Get-Content $log).Replace('0 made, 0 loaded,','1 made, 0 loaded,')|Set-Content $log } 'Pipeline activity'
 Reject 'title error' { Add-Content $log '2026-01-01T10:01:00Z ERROR victory_gpu::host: synthetic failure' } 'contains errors'
 Reject 'capture exit' { $c=Get-Content $cap -Raw|ConvertFrom-Json;$c.PresentMonExit=1;$c|ConvertTo-Json|Set-Content $cap } 'capture exit'
 "$passed tests passed"
}finally{Remove-Item -LiteralPath $dir -Recurse -Force}
