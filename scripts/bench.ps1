<#
.SYNOPSIS
Runs the title unattended with the diagnostics build, drives the menus into
the world and summarises frame times, so runs on different machines or with
different switches can be compared.

.DESCRIPTION
Each run gets a folder of its own (default: <cache_dir>\bench\<time>-<label>)
holding the log, frames captured every -CaptureEvery seconds, the run's copy
of the title's data, summary.md and summary.json.

Every run starts from the same saves: the title's data folder (the configured
data_dir, or -Seed) is copied for the run, so the run's autosaves never reach
it and two runs load the same checkpoint. The shader and pipeline caches are
shared: the first run of a build translates and compiles, later runs measure.

Until the world starts (a 5-second frame line with 500 draws or more) the
script looks at a captured frame every two seconds and presses what that
screen needs: Down to "Next" on the legal notices, A on each unticked box of
the terms and then A on "Confirm", and A (Enter) every few seconds on any
other screen, which passes the start and first-boot screens and continues
the save. With -FirstBoot the run starts with no saves at all (a fresh
profile), and its data folder then holds a save to start later runs from
(-Seed <run folder>\data).

Needs the diagnostics build (scripted keys, frame capture):
    cargo build --profile diag -p victory --features diagnostics
or pass -Build.

.EXAMPLE
.\scripts\bench.ps1 -Label baseline
.\scripts\bench.ps1 -Label no-opt -Switches @{ VICTORY_SHADER_OPT = '0' }
.\scripts\bench.ps1 -Compare <run folder>,<run folder>

.EXAMPLE
# No saves yet: make one, then measure from it.
.\scripts\bench.ps1 -FirstBoot -Label seed -Seconds 60
.\scripts\bench.ps1 -Seed <the seed run's folder>\data -Label baseline

.EXAMPLE
# Summarise an earlier run again, over another window.
.\scripts\bench.ps1 -From <run folder> -Warmup 60 -Seconds 120
#>
[CmdletBinding()]
param(
    # A name for the run folder and the summary.
    [string]$Label = 'run',
    # Switches for this run: @{ VICTORY_RELAY = '0' }, or 'VICTORY_RELAY=0'
    # (several as a list), which also passes through `powershell -File`.
    [object]$Switches = @{},
    # Seconds measured, after -Warmup, from the world's first frame.
    [int]$Seconds = 180,
    # Seconds of world frames left out first (pipelines still being made).
    [int]$Warmup = 30,
    # Give up if the world has not started after this many seconds.
    [int]$BootTimeout = 900,
    # Keep pressing A in the world (skips cutscenes; runs then differ more).
    [switch]$SkipCutscenes,
    # Seconds between captured frames (0: none).
    [int]$CaptureEvery = 20,
    # Take a PIX GPU capture of one frame this many seconds after the
    # world's first frame (needs PIX installed; the capturer slows every
    # frame, so the run's frame times are not comparable with others).
    [int]$PixAt = -1,
    # The data folder runs start from (default: the configured data_dir).
    [string]$Seed,
    # The run folder (default: <cache_dir>\bench\<time>-<label>).
    [string]$Out,
    # The title's executable (default: the one .exe in game_dir).
    [string]$Exe,
    # The diagnostics build to run (default: target\diag\victory.exe). A
    # copy elsewhere lets a run go on while the tree builds, or two builds
    # be compared.
    [string]$Victory,
    # Build the diagnostics build first.
    [switch]$Build,
    # Walk the first-boot screens of a profile with no saves.
    [switch]$FirstBoot,
    # Summarise this earlier run folder again instead of running.
    [string]$From,
    # Print a table of earlier runs' summaries instead of running.
    [string[]]$Compare
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$victory = if ($Victory) { (Resolve-Path $Victory).Path } else { Join-Path $root 'target\diag\victory.exe' }
$procName = [IO.Path]::GetFileNameWithoutExtension($victory)
$invariant = [Globalization.CultureInfo]::InvariantCulture
if ($Switches -isnot [hashtable]) {
    $table = @{}
    foreach ($pair in @($Switches) | ForEach-Object { "$_" -split '[,;]' } | Where-Object { $_ }) {
        $name, $value = $pair -split '=', 2
        if ($null -eq $value) { throw "a switch is NAME=VALUE, not '$pair'" }
        $table[$name.Trim()] = $value.Trim()
    }
    $Switches = $table
}

# The summary's columns: the frame line's fields, per frame.
$metrics = @(
    @{ key = 'fps'; head = 'fps' },
    @{ key = 'ms'; head = 'ms/frame' },
    @{ key = 'prep'; head = 'preparing' },
    @{ key = 'gpu'; head = 'GPU' },
    @{ key = 'back'; head = 'copying back' },
    @{ key = 'draws'; head = 'draws' },
    @{ key = 'dropped'; head = 'dropped' },
    @{ key = 'making'; head = 'being made' }
)
$fields = 'fps', 'ms', 'prep', 'copyms', 'texms', 'gpu', 'back', 'draws', 'dropped', 'making'
# The 5-second frame line (victory_gpu::host).
$lineRe = [regex]('^(?<ts>\S+Z)\s+INFO victory_gpu::host: (?<fps>[\d.]+) fps, (?<ms>[\d.]+) ms/frame ' +
    '\((?<prep>[\d.]+) preparing: copies (?<copyms>[\d.]+) ms [\d.]+ MB, textures (?<texms>[\d.]+) ms [\d.]+ MB; ' +
    '(?<gpu>[\d.]+) on the GPU; (?<back>[\d.]+) copying back [\d.]+ MB\), (?<draws>\d+) draws/frame, ' +
    'dropped (?<dropped>\d+) without a host pipeline \((?<making>\d+) being made\)')
$worldDraws = 500

function Format-Number($value) {
    if ($null -eq $value) { return '-' }
    return ([double]$value).ToString('0.#', $invariant)
}

function Show-Compare([string[]]$dirs) {
    $runs = foreach ($d in $dirs) {
        $path = Join-Path $d 'summary.json'
        if (-not (Test-Path $path)) { throw "no summary.json in $d" }
        Get-Content -Raw $path | ConvertFrom-Json
    }
    $heads = ($metrics | ForEach-Object { $_.head }) -join ' | '
    ('| run | frame lines | ' + $heads + ' | not translated | refused | crashes |')
    ('|' + ('---|' * ($metrics.Count + 5)))
    foreach ($r in $runs) {
        $cells = ($metrics | ForEach-Object { Format-Number $r.median.($_.key) }) -join ' | '
        ('| ' + $r.label + ' | ' + $r.lines + ' | ' + $cells + ' | ' + $r.untranslated + ' | ' + $r.refused + ' | ' + $r.crashes + ' |')
    }
    ''
    'Medians of the 5-second frame lines in each run''s window; the ms columns are per frame.'
}

function Get-Median([double[]]$values) {
    if ($values.Count -eq 0) { return $null }
    $sorted = [double[]]($values | Sort-Object)
    $mid = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$mid] }
    return ($sorted[$mid - 1] + $sorted[$mid]) / 2
}

# Reads a run folder's log and writes summary.json and summary.md.
function Write-Summary([string]$dir, [hashtable]$facts) {
    $log = Join-Path $dir 'victory.log'
    $rows = New-Object System.Collections.Generic.List[object]
    $untranslated = @{}
    $refused = @{}
    $notes = New-Object System.Collections.Generic.List[string]
    $slow = New-Object System.Collections.Generic.List[double]
    $first = $null
    foreach ($line in [System.IO.File]::ReadLines($log)) {
        if (-not $first -and $line -match '^(\S+Z)\s') { $first = [datetime]::Parse($Matches[1], $invariant).ToUniversalTime() }
        $m = $lineRe.Match($line)
        if ($m.Success) {
            $row = [ordered]@{ at = [datetime]::Parse($m.Groups['ts'].Value, $invariant).ToUniversalTime() }
            foreach ($k in $fields) { $row[$k] = [double]::Parse($m.Groups[$k].Value, $invariant) }
            $rows.Add([pscustomobject]$row)
        } elseif ($line -match 'shader does not translate.*?([0-9a-f]{16})') {
            $untranslated[$Matches[1]] = 1
        } elseif ($line -match 'host (mesh )?pipeline refused.*shaders=(\[.*?\])') {
            $refused[$Matches[2]] = 1
        } elseif ($line -match 'slow driver compile.*seconds=([\d.]+)') {
            $slow.Add([double]::Parse($Matches[1], $invariant))
        } elseif ($line -match 'device removed|DEVICE_REMOVED|DEVICE_HUNG|unhandled exception|PIX capture|VICTORY_PIX|VICTORY_DEBUG_NULL_VIEWS') {
            if ($notes.Count -lt 40) { $notes.Add(($line -replace '^\S+\s+', '')) }
        }
    }
    $world = $rows | Where-Object { $_.draws -ge $worldDraws } | Select-Object -First 1
    $window = @()
    if ($world) {
        $from = $world.at.AddSeconds($Warmup)
        $to = $from.AddSeconds($Seconds)
        $window = @($rows | Where-Object { $_.at -ge $from -and $_.at -le $to })
    } else {
        $notes.Add('the world never started (no frame line with 500 draws or more)')
    }
    $median = [ordered]@{}
    $mean = [ordered]@{}
    foreach ($k in $fields) {
        $values = [double[]]@($window | ForEach-Object { $_.$k })
        $median[$k] = Get-Median $values
        $mean[$k] = if ($values.Count) { ($values | Measure-Object -Average).Average } else { $null }
    }
    $crashes = Join-Path $dir 'data\crashes.log'
    $summary = [ordered]@{
        label           = $facts.label
        switches        = $facts.switches
        commit          = $facts.commit
        gpu             = $facts.gpu
        exit            = $facts.exit
        peak_private_gb = $facts.peak_private_gb
        world_after_s   = if ($world -and $first) { [math]::Round(($world.at - $first).TotalSeconds) } else { $null }
        window          = "$Warmup-$($Warmup + $Seconds) s after the world's first frame"
        lines           = $window.Count
        median          = $median
        mean            = $mean
        untranslated    = $untranslated.Count
        refused         = $refused.Count
        crashes         = if (Test-Path $crashes) { @(Get-Content $crashes).Count } else { 0 }
        slow_compiles   = $slow.Count
        slow_compile_s  = [math]::Round(($slow | Measure-Object -Sum).Sum, 1)
        slowest_compile_s = if ($slow.Count) { [math]::Round(($slow | Measure-Object -Maximum).Maximum, 1) } else { 0 }
        notes           = $notes
    }
    $summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $dir 'summary.json')

    $switchText = if ($facts.switches.Count) {
        ($facts.switches.Keys | Sort-Object | ForEach-Object { '`' + $_ + '=' + $facts.switches[$_] + '`' }) -join ' '
    } else { 'none' }
    $heads = ($metrics | ForEach-Object { $_.head }) -join ' | '
    $md = New-Object System.Collections.Generic.List[string]
    $md.Add('## ' + $facts.label)
    $md.Add('')
    $md.Add('- commit ' + $facts.commit + ', GPU ' + $facts.gpu)
    $md.Add('- switches: ' + $switchText)
    $md.Add('- world after ' + $summary.world_after_s + ' s; window ' + $summary.window + ': ' + $window.Count + ' frame lines')
    $md.Add('- shaders not translated: ' + $untranslated.Count + '; host pipelines refused: ' + $refused.Count +
        '; crashes.log lines: ' + $summary.crashes + '; exit: ' + $facts.exit + '; peak private memory: ' + $facts.peak_private_gb + ' GB')
    $md.Add('- driver compiles of 2 s or more: ' + $slow.Count + ', ' + $summary.slow_compile_s + ' s in all, the slowest ' + $summary.slowest_compile_s + ' s')
    $md.Add('')
    $md.Add('| | ' + $heads + ' |')
    $md.Add('|---|' + ('---|' * $metrics.Count))
    $md.Add('| median | ' + (($metrics | ForEach-Object { Format-Number $median[$_.key] }) -join ' | ') + ' |')
    $md.Add('| mean | ' + (($metrics | ForEach-Object { Format-Number $mean[$_.key] }) -join ' | ') + ' |')
    if ($notes.Count) {
        $md.Add('')
        $md.Add('Notes:')
        $md.Add('')
        foreach ($n in $notes) { $md.Add('- ' + $n) }
    }
    $md | Set-Content -Encoding UTF8 (Join-Path $dir 'summary.md')
    $md | ForEach-Object { Write-Host $_ }
    return [bool]$world
}

if ($Compare) {
    Show-Compare $Compare
    return
}

if ($From) {
    $old = Get-Content -Raw (Join-Path $From 'summary.json') | ConvertFrom-Json
    $switchTable = @{}
    if ($old.switches) { $old.switches.PSObject.Properties | ForEach-Object { $switchTable[$_.Name] = $_.Value } }
    $facts = @{
        label = $old.label; switches = $switchTable; commit = $old.commit; gpu = $old.gpu
        exit = $old.exit; peak_private_gb = $old.peak_private_gb
    }
    if (-not (Write-Summary $From $facts)) { exit 1 }
    return
}

if ($Build) {
    Push-Location $root
    try {
        cargo build --profile diag -p victory --features diagnostics
        if ($LASTEXITCODE -ne 0) { throw 'build failed' }
    } finally { Pop-Location }
}
if (-not (Test-Path $victory)) {
    throw "no diagnostics build at $victory (cargo build --profile diag -p victory --features diagnostics, or -Build)"
}

# The configuration as `victory config` resolves it (victory.toml and the
# VICTORY_*_DIR variables).
Push-Location $root
try { $config = & $victory config } finally { Pop-Location }
function Get-ConfigPath([string]$name) {
    foreach ($line in $config) {
        if ($line -match "^$name\s*:\s*(.+?)(\s+\(does not exist\))?$") { return $Matches[1] }
    }
    return $null
}
$gameDir = Get-ConfigPath 'game_dir'
$cacheDir = Get-ConfigPath 'cache_dir'
$dataDir = Get-ConfigPath 'data_dir'
if (-not $Exe) {
    if (-not $gameDir -or $gameDir -eq '(not set)') { throw 'no game_dir configured (victory.toml or VICTORY_GAME_DIR), and no -Exe' }
    $exes = @(Get-ChildItem -LiteralPath $gameDir -Filter '*.exe' -File)
    if ($exes.Count -ne 1) { throw "game_dir holds $($exes.Count) executables; name one with -Exe" }
    $Exe = $exes[0].FullName
}
if (-not $Seed) { $Seed = $dataDir }
if (-not $Out) {
    $Out = Join-Path $cacheDir ('bench\{0}-{1}' -f (Get-Date -Format 'yyyyMMdd-HHmmss'), ($Label -replace '[^\w.-]', '_'))
}
$frames = Join-Path $Out 'frames'
$runData = Join-Path $Out 'data'
$keys = Join-Path $Out 'keys.txt'
# Logs go to stderr; stdout has the loader's few plain lines.
$log = Join-Path $Out 'victory.log'
$stdout = Join-Path $Out 'victory.out.txt'
New-Item -ItemType Directory -Force -Path $frames | Out-Null
New-Item -ItemType File -Force -Path $keys | Out-Null
if ($FirstBoot) {
    New-Item -ItemType Directory -Force -Path $runData | Out-Null
} elseif (Test-Path $Seed) {
    Copy-Item -Recurse -Force -Path $Seed -Destination $runData
} else {
    throw "the seed data folder $Seed does not exist (no saves yet? run once with -FirstBoot)"
}

function Send-Keys([string[]]$names) {
    Add-Content -Encoding ASCII -Path $keys -Value $names
}

# What a captured menu frame shows. The two screens that A alone does not
# pass are told apart by their layout, measured on the title's own frames
# (coordinates on a 1920x1080 capture, scaled to the capture's size): the
# legal notices (Copyright, Photosensitivity, a Next button) and the terms
# (two boxes to tick, a Confirm button). Returns `notices:<focus>`,
# `terms:<focus>:<privacy ticked>:<terms ticked>`, or `other`.
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class BenchScreen {
    const double W0 = 1920, H0 = 1080;

    public static string Classify(string path) {
        using (var bmp = new Bitmap(path)) {
            int w = bmp.Width, h = bmp.Height;
            var data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            var px = new byte[data.Stride * h];
            Marshal.Copy(data.Scan0, px, 0, px.Length);
            bmp.UnlockBits(data);
            Func<int, int, double> lum = (x, y) => {
                int i = y * data.Stride + x * 3;
                return 0.114 * px[i] + 0.587 * px[i + 1] + 0.299 * px[i + 2];
            };
            Func<double, int> X = v => (int)Math.Round(v / W0 * w);
            Func<double, int> Y = v => (int)Math.Round(v / H0 * h);
            // Mean brightness of a row segment.
            Func<int, int, int, double> row = (y, x0, x1) => {
                double s = 0; for (int x = x0; x < x1; x++) s += lum(x, y); return s / Math.Max(1, x1 - x0);
            };
            // A horizontal outline at y: the rows around it differ.
            Func<double, double> edge = y0 => {
                int y = Y(y0), x0 = X(25), x1 = X(470); double best = 0;
                for (int d = -1; d <= 1; d++) best = Math.Max(best, Math.Abs(row(y + d + 1, x0, x1) - row(y + d - 1, x0, x1)));
                return best;
            };
            // The menu's focused entry is drawn white.
            Func<double, double, bool> focused = (a, b) => {
                double s = 0; int n = 0;
                for (int y = Y(a); y < Y(b); y++) { s += row(y, X(20), X(300)); n++; }
                return n > 0 && s / n > 180;
            };
            // A box is ticked when its inside holds a mark that stands out.
            Func<double, double, bool> ticked = (y0, y1) => {
                int x0 = X(50), x1 = X(71), a = Y(y0), b = Y(y1);
                var v = new System.Collections.Generic.List<double>();
                for (int y = a; y < b; y++) for (int x = x0; x < x1; x++) v.Add(lum(x, y));
                if (v.Count == 0) return false;
                v.Sort(); double median = v[v.Count / 2]; int far = 0;
                foreach (var p in v) if (Math.Abs(p - median) > 90) far++;
                return far > 0.05 * v.Count;
            };
            if (edge(374.5) > 100 && edge(437.5) > 100 && edge(543) < 10) {
                string f = focused(380, 432) ? "next" : focused(248, 304) ? "photo" : focused(180, 236) ? "copyright" : "none";
                return "notices:" + f;
            }
            if (edge(543) > 25 && edge(606) > 25 && edge(484) > 25 && edge(374.5) < 10) {
                string f = focused(549, 601) ? "confirm" : focused(448, 505) ? "terms" : focused(245, 340) ? "privacy" : "none";
                return "terms:" + f + ":" + (ticked(261, 282) ? 1 : 0) + ":" + (ticked(465, 487) ? 1 : 0);
            }
            return "other";
        }
    }
}
'@

# The key a menu screen needs, or $null to leave it to the A every few
# seconds. Ticking is a toggle, so the terms are only pressed on what the
# frame shows unticked.
function Get-MenuKey([string]$screen) {
    $part = $screen.Split(':')
    switch ($part[0]) {
        'notices' { if ($part[1] -eq 'next') { return 'Enter' } else { return 'Down' } }
        'terms' {
            $privacy = $part[2] -eq '1'
            $terms = $part[3] -eq '1'
            switch ($part[1]) {
                'privacy' { if ($privacy) { return 'Down' } else { return 'Enter' } }
                'terms' { if ($terms) { return 'Down' } else { return 'Enter' } }
                'confirm' { if ($privacy -and $terms) { return 'Enter' } else { return 'Up' } }
                default { return 'Down' }
            }
        }
    }
    return $null
}

# Asks for the next presented frame and returns its file (or $null).
function Get-Frame([double]$timeout = 5) {
    $since = Get-Date
    New-Item -ItemType File -Force -Path (Join-Path $frames 'now') | Out-Null
    while (((Get-Date) - $since).TotalSeconds -lt $timeout) {
        Start-Sleep -Milliseconds 250
        $new = Get-ChildItem -LiteralPath $frames -Filter 'frame-*.png' |
            Where-Object { $_.LastWriteTime -ge $since } | Sort-Object LastWriteTime | Select-Object -Last 1
        if ($new) {
            # The frame is written in one call; give it a moment to finish.
            Start-Sleep -Milliseconds 200
            return $new.FullName
        }
    }
    return $null
}

$saved = @{}
$pix = Join-Path $Out 'pix'
$set = @{
    VICTORY_DATA_DIR    = $runData
    VICTORY_KEYS_FILE   = $keys
    VICTORY_CAPTURE     = $frames
    VICTORY_OVERLAY_LOG = Join-Path $Out 'overlay.log'
}
if ($PixAt -ge 0) { $set['VICTORY_PIX'] = $pix }
foreach ($k in $Switches.Keys) { $set[$k] = [string]$Switches[$k] }
foreach ($k in $set.Keys) {
    $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
    [Environment]::SetEnvironmentVariable($k, $set[$k], 'Process')
}

# Watches the log for the world's first frame line (read as it grows).
$offset = 0
$pending = ''
$worldAt = $null
function Watch-Log {
    if ($script:worldAt -or -not (Test-Path $log)) { return }
    $stream = [System.IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
    try {
        if ($stream.Length -le $script:offset) { return }
        [void]$stream.Seek($script:offset, 'Begin')
        $text = $script:pending + (New-Object System.IO.StreamReader($stream)).ReadToEnd()
        $script:offset = $stream.Length
    } finally { $stream.Dispose() }
    $end = $text.LastIndexOf("`n")
    $script:pending = $text.Substring($end + 1)
    if ($end -lt 0) { return }
    foreach ($line in $text.Substring(0, $end).Split("`n")) {
        $m = $lineRe.Match($line.TrimEnd("`r"))
        if ($m.Success -and [int]$m.Groups['draws'].Value -ge $worldDraws) {
            $script:worldAt = Get-Date
            return
        }
    }
}

Write-Host "run folder: $Out"
Write-Host "title:      $Exe"
Write-Host ('switches:   ' + (($Switches.Keys | Sort-Object | ForEach-Object { "$_=$($Switches[$_])" }) -join ' '))
$launched = Get-Date
$proc = Start-Process -FilePath $victory -ArgumentList @('run', "`"$Exe`"", '--execute') -WorkingDirectory $root `
    -RedirectStandardOutput $stdout -RedirectStandardError $log -PassThru -NoNewWindow
# Taking the handle now keeps the exit code readable after the process ends.
$null = $proc.Handle
$exit = 'stopped by the script'
$peak = 0
try {
    $lastPress = 0
    $lastCapture = 0
    $lastLook = 0
    $lastScreen = ''
    $pixAsked = $false
    while ($true) {
        Start-Sleep -Milliseconds 500
        Watch-Log
        $t = ((Get-Date) - $launched).TotalSeconds
        if ($proc.HasExited) { $exit = $proc.ExitCode; break }
        # The title runs in a process `victory run` starts: the largest one.
        $mem = (Get-Process -Name $procName -ErrorAction SilentlyContinue | Measure-Object PrivateMemorySize64 -Maximum).Maximum
        if ($mem -gt $peak) { $peak = $mem }
        if (-not $worldAt) {
            if ($t -gt $BootTimeout) { Write-Warning "the world did not start within $BootTimeout s"; break }
            if ($t -lt 10 -or $t - $lastLook -lt 2) { continue }
            $lastLook = $t
            $frame = Get-Frame
            $screen = if ($frame) { [BenchScreen]::Classify($frame) } else { 'other' }
            if ($screen -ne $lastScreen) {
                Write-Host ('{0,5:0}s  {1}' -f $t, $screen)
                $lastScreen = $screen
            } elseif ($frame) {
                # One frame per screen is kept, to see where a run stopped.
                Remove-Item -LiteralPath $frame -ErrorAction SilentlyContinue
            }
            $key = Get-MenuKey $screen
            if ($key) {
                Send-Keys @($key); $lastPress = $t
            } elseif ($t - $lastPress -ge 4) {
                Send-Keys @('Enter'); $lastPress = $t
            }
            continue
        }
        if ($CaptureEvery -gt 0 -and $t - $lastCapture -ge $CaptureEvery) {
            New-Item -ItemType File -Force -Path (Join-Path $frames 'now') | Out-Null
            $lastCapture = $t
        }
        if ($SkipCutscenes -and $t - $lastPress -ge 8) { Send-Keys @('Enter'); $lastPress = $t }
        if ($PixAt -ge 0 -and -not $pixAsked -and ((Get-Date) - $worldAt).TotalSeconds -ge $PixAt) {
            New-Item -ItemType Directory -Force -Path $pix | Out-Null
            Set-Content -Path (Join-Path $pix 'capture') -Value '1'
            # The same frame as a PNG, to find the moment in the capture.
            New-Item -ItemType File -Force -Path (Join-Path $frames 'now') | Out-Null
            $pixAsked = $true
            Write-Host ('PIX capture asked for at {0:0} s in the world (into {1})' -f ((Get-Date) - $worldAt).TotalSeconds, $pix)
        }
        # A frame line lands every 5 s: run one past the window's end.
        if (((Get-Date) - $worldAt).TotalSeconds -ge $Warmup + $Seconds + 6) { break }
    }
} finally {
    if (-not $proc.HasExited) {
        Get-Process -Name $procName -ErrorAction SilentlyContinue |
            Where-Object { $_.Id -eq $proc.Id -or $_.StartTime -ge $launched } |
            Stop-Process -Force
        $proc.WaitForExit()
    }
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
}

$facts = @{
    label           = $Label
    switches        = $Switches
    # A build given with -Victory need not be the tree's: named by file.
    commit          = if ($Victory) { 'build ' + (Split-Path -Leaf $victory) + ' of ' + (Get-Item $victory).LastWriteTime.ToString('yyyy-MM-dd HH:mm') } else { (git -C $root rev-parse --short HEAD 2>$null) }
    gpu             = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '
    exit            = $exit
    peak_private_gb = [math]::Round($peak / 1GB, 1)
}
if (-not (Write-Summary $Out $facts)) { exit 1 }
