#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateRange(1, 2147483647)][int]$ProcessId,
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 2,
    [ValidateRange(0, 2147483647)][int]$Samples = 0,
    [switch]$DiagnosticsBuild
)

$ErrorActionPreference = 'Stop'
$target = [Diagnostics.Process]::GetProcessById($ProcessId)
$sampleNumber = 0

function Write-Record($kind, $fields) {
    [ordered]@{
        kind = $kind
        utc = [DateTime]::UtcNow.ToString('o')
        pid = $ProcessId
        data = $fields
    } | ConvertTo-Json -Depth 6 -Compress
}

# Keep other providers usable if a driver or a counter class is missing.
function Read-Counters($class, $filter, $properties, $unavailable) {
    $query = @{ ClassName = $class; Property = $properties }
    if ($filter) { $query.Filter = $filter }
    try {
        Get-CimInstance @query
    } catch {
        $unavailable.Add($class)
    }
}

function Read-Packages {
    $manifest = Join-Path $PSScriptRoot '../Cargo.toml'
    $options = @('tree', '--locked', '--offline', '--manifest-path', $manifest,
        '--package', 'victory', '--target', 'x86_64-pc-windows-msvc', '--edges', 'normal,build',
        '--prefix', 'none', '--format', '{p}')
    if ($DiagnosticsBuild) { $options += '--features', 'diagnostics' }
    try {
        $tree = & cargo @options 2>$null
        if ($LASTEXITCODE -eq 0) {
            $entries = foreach ($line in $tree) {
                # Keep only package and version, including for local path dependencies.
                if ($line -match '^(\S+) v(\S+)') { "$($Matches[1]) $($Matches[2])" }
            }
            return $entries | Sort-Object -Unique
        }
    } catch {
        # Cargo or its offline cache may be absent on a machine running a release.
    }
}

try {
    # Hold the process handle for the session so a reused PID cannot mix runs.
    [void]$target.Handle
    $unavailable = [Collections.Generic.List[string]]::new()
    $adapters = @(Read-Counters Win32_VideoController $null @('Name', 'DriverVersion') $unavailable |
        Select-Object Name, DriverVersion)
    $processors = @(Read-Counters Win32_Processor $null @('Name', 'NumberOfCores', 'NumberOfLogicalProcessors') $unavailable |
        Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
    $packages = @(Read-Packages)
    if (!$packages.Count) { $unavailable.Add('cargo tree') }
    Write-Record session ([ordered]@{
        process = $target.ProcessName
        processors = $processors
        adapters = $adapters
        cargo_features = if ($DiagnosticsBuild) { 'diagnostics' } else { 'default' }
        cargo_packages = $packages
        unavailable = $unavailable
    })

    $pidFilter = "Name LIKE 'pid[_]${ProcessId}[_]%'"
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $prevElapsed = $null
    $prevCpu = 0
    $nextModuleCheck = 0
    $reportedModules = @{}
    while (!$target.HasExited) {
        $target.Refresh()
        $elapsed = $timer.Elapsed.TotalMilliseconds
        $user = $target.UserProcessorTime.TotalMilliseconds
        $kernel = $target.PrivilegedProcessorTime.TotalMilliseconds
        $cpu = $user + $kernel
        $cores = $null
        if ($null -ne $prevElapsed -and $elapsed -gt $prevElapsed) {
            $cores = [Math]::Round(($cpu - $prevCpu) / ($elapsed - $prevElapsed), 3)
        }
        $prevElapsed = $elapsed
        $prevCpu = $cpu
        $processMetrics = [ordered]@{
            cpu_cores = $cores
            user_ms = $user
            kernel_ms = $kernel
            threads = $target.Threads.Count
            working_set_bytes = $target.WorkingSet64
            peak_working_set_bytes = $target.PeakWorkingSet64
            private_bytes = $target.PrivateMemorySize64
        }

        $unavailable.Clear()
        if ($elapsed -ge $nextModuleCheck) {
            $nextModuleCheck = $elapsed + 30000
            try {
                foreach ($module in $target.Modules) {
                    $name = $module.ModuleName
                    if ($name -notin @('dxgi.dll', 'd3d12.dll', 'D3D12Core.dll', 'dxcompiler.dll',
                            'dxil.dll', 'd3dcompiler_47.dll', 'xinput1_4.dll')) { continue }
                    $version = $module.FileVersionInfo.FileVersion
                    if (!$reportedModules.ContainsKey($name) -or $reportedModules[$name] -ne $version) {
                        Write-Record runtime_module @{ name = $name; file_version = $version }
                        $reportedModules[$name] = $version
                    }
                }
            } catch {
                $unavailable.Add('process modules')
            }
        }
        $memory = Read-Counters Win32_PerfFormattedData_PerfOS_Memory $null @(
            'AvailableBytes', 'CommittedBytes', 'CommitLimit', 'PagesInputPersec'
        ) $unavailable | Select-Object AvailableBytes, CommittedBytes, CommitLimit, PagesInputPersec
        if ($null -eq $memory) { $memory = $null } # Serialize an empty pipeline as JSON null on PS 5.1.
        $hostCpu = Read-Counters Win32_PerfFormattedData_PerfOS_Processor "Name='_Total'" @(
            'PercentProcessorTime'
        ) $unavailable
        $engines = @(Read-Counters Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine $pidFilter @(
            'Name', 'UtilizationPercentage'
        ) $unavailable)
        $gpuMemory = @(Read-Counters Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory $pidFilter @(
            'Name', 'DedicatedUsage', 'SharedUsage', 'TotalCommitted'
        ) $unavailable | Select-Object Name, DedicatedUsage, SharedUsage, TotalCommitted)

        Write-Record sample ([ordered]@{
            elapsed_ms = [Math]::Round($elapsed)
            process = $processMetrics
            host_cpu_percent = $hostCpu.PercentProcessorTime
            host_memory = $memory
            gpu_engine_count = $engines.Count
            busy_gpu_engines = @($engines | Where-Object UtilizationPercentage -gt 0 |
                Select-Object Name, UtilizationPercentage)
            gpu_memory = $gpuMemory
            unavailable = $unavailable
        })
        $sampleNumber++
        if ($Samples -and $sampleNumber -ge $Samples) { break }
        if ($target.WaitForExit(1000 * $IntervalSeconds)) { break }
    }
} catch {
    # The target can exit between Refresh and any of the process property reads.
    if (!$target.HasExited) { throw }
} finally {
    $target.Dispose()
}
