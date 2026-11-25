# Recording resource usage on Windows

Use the PowerShell collector to compare a slow frame, a loading pause or a
stalled run with Windows resource counters. It samples the running process
independently of the frame loop and works with release and diagnostics builds.
Use the 64-bit PowerShell on Windows 10 or 11.

Start the title as usual. In a second terminal at the repository root, find
the PID for that run and save the JSON Lines stream outside the checkout.

```powershell
Get-Process victory | Select-Object Id, StartTime
# Replace 1234 with the PID from the first command.
$log = Join-Path $env:TEMP ('victory-resources-{0:yyyyMMdd-HHmmss}.jsonl' -f (Get-Date))
.\tools\watch-resources.ps1 -ProcessId 1234 | Out-File -Encoding utf8 $log
```

The collector stops when the process exits or you press Ctrl+C. To capture a
bounded window, use `-Samples 30 -IntervalSeconds 2`. The interval is the
pause between polls (two seconds by default). Queries take additional time.
Each line has a UTC timestamp, the target PID and a record kind. This command
reads the process measurements back in PowerShell.

```powershell
Get-Content $log | ForEach-Object { $_ | ConvertFrom-Json } |
    Where-Object kind -eq sample | ForEach-Object { $_.data.process }
```

| Record | What to look for |
|--------|------------------|
| `session` | CPU model and core counts, installed display adapters and driver versions. The Cargo package list comes from the checkout's locked Windows build and normal dependencies. Pass `-DiagnosticsBuild` for a binary built with the diagnostics feature. Use the checkout that built the binary for a meaningful comparison. |
| `runtime_module` | Name and file version of a graphics or input DLL observed in the target. The collector checks on attachment and then every 30 seconds and emits newly observed names or versions. A short-lived module can escape observation. |
| `sample` process fields | Cumulative user and kernel CPU time in milliseconds, thread count, working set, peak working set and private committed bytes. `cpu_cores` is CPU time divided by actual elapsed time between samples. A value of 2 means two logical processors' worth of work. The first sample has no preceding interval and reports null. |
| `sample` host fields | Total CPU utilization in percent and system-wide memory counters. Compare committed bytes with the commit limit for allocation headroom. Low available physical memory together with rising pages read from disk per second can indicate paging pressure. |
| `sample` GPU fields | Per-process GPU memory in bytes (dedicated, shared and total committed), grouped by adapter instance. Busy engines carry their Windows instance names, including the adapter LUID and engine type, and their independently sampled utilization percentages. Engines can overlap in time. |

`gpu_engine_count` includes idle engines. A positive count with an empty busy list
means all reported engines showed zero percent. A zero count means Windows
returned no engine instances for this PID. Check `unavailable` for failed
queries. Driver support and permissions can limit the GPU counters. Windows
samples the counter classes separately, so peaks can fall between polls.

The Cargo inventory uses `cargo tree --locked --offline`. If Cargo or the local
cache is missing, the session marks it unavailable and continues sampling.
Runtime module records cover the DXGI, D3D12, shader compiler and XInput DLLs
listed in the script and report only their names and versions. Keep captures
outside the repository as described in the [asset policy](asset-policy.md).
