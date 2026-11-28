# D3D12 diagnostics overhead

This ignored Cargo test uses a synthetic compute shader and public Windows
D3D12 interfaces. The standalone fixture builds independently of the runtime
and the BCPack table. Windows with D3D12 feature level 12.0 and Graphics Tools
(for the debug layer) is required.

From the repository root in PowerShell:

```powershell
$manifest = 'tests/fixtures/d3d12-diagnostics/Cargo.toml'
$env:VICTORY_D3D12_BENCH_ADAPTER = '10de:2882'
cargo test --locked --release --manifest-path $manifest --test diagnostics -- --ignored --nocapture d3d12_diagnostics_overhead |
    Tee-Object "$env:TEMP\d3d12-hardware.log"
$env:VICTORY_D3D12_BENCH_ADAPTER = 'warp'
cargo test --locked --release --manifest-path $manifest --test diagnostics -- --ignored --nocapture d3d12_diagnostics_overhead |
    Tee-Object "$env:TEMP\d3d12-warp.log"
Remove-Item Env:VICTORY_D3D12_BENCH_ADAPTER
```

Use the hardware adapter's hexadecimal vendor and device IDs, or use
`hardware` to take the first non-software adapter in DXGI high-performance
order. The log records the adapter name, IDs, LUID, software flag and queue
timestamp frequency. It includes CSV records prefixed with `sample,`.

Each mode runs in a fresh process per round. The test varies the order of
modes over three rounds. Every mode runs 8 warm-up pairs and 31 measured
pairs per round. The test alternates which run in a pair uses timestamps.
Each run submits a single direct command list with 512 one-group compute
dispatches and verifies every output word after its fence.

| Mode | Configuration |
| --- | --- |
| `off` | DRED auto-breadcrumbs and page-fault reporting forced off, debug layer off |
| `breadcrumbs` | Auto-breadcrumbs only |
| `dred` | Auto-breadcrumbs and page-fault reporting |
| `debug` | CPU debug layer |
| `gbv` | Debug layer with GPU-based validation enabled |

All times are per list. `record_us` covers allocator/list reset through
list close. `submit_us` covers execute and fence signal. `completion_us`
covers reset through fence completion and includes synchronization and
the output copy. For timestamped runs, `gpu_us` divides the timestamp
difference bracketing the dispatch loop by the queue frequency. WARP
reports timestamps from its software implementation. Shader compilation,
resource creation, CPU-side readback and validation are outside the
measurement intervals.

Report the median and 25th–75th percentiles for each adapter and mode.
Compare CPU costs using the runs without timestamp queries and GPU costs
using the timestamped runs. For measurement overhead, report the paired
timestamp-on minus timestamp-off deltas for recording, submission and
completion. Completion is an end-to-end estimate of query perturbation.
The benchmark covers steady-state diagnostics. Device-removal handling,
VIctory's marker-text formatting and compact operation records, and game
frame times are outside its scope.
