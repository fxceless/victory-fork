# Testing VIctory under Proton

The Linux tests use the Windows target under Proton. Build it with
`cargo xwin build --locked --target x86_64-pc-windows-msvc --profile diag -p victory --features diagnostics`.

- `VICTORY_PIPELINE_WORKERS=all` allows one pipeline worker per logical CPU.
  You can also specify a worker count. The pipeline workers run below the
  title's thread priority.
- `VICTORY_NO_HOST_GPU=1` enables CPU-only diagnostics in a diagnostics
  build. It disables the host renderer and can prevent the title from
  reaching the same scene as a rendered run.
- Use a local, compatible Microsoft `d3dcompiler_47.dll` beside
  `victory.exe` and select it with Wine's `d3dcompiler_47=n,b` override.
  Proton's built-in compiler can fail to compile the compute shader that
  makes indirect draws safe to submit. For the test prefix, set
  `dxgi,d3d12,d3d12core=n` to select DXVK and vkd3d-proton.
- Keep the Wine prefix, title files, settings, and caches outside the
  source checkout.

## Wine mapped-view performance

In a diagnostic run under stock Proton, Wine's server was busy searching its
mapped-view list while VIctory waited for title memory remaps. A server with
a tree index substantially reduced the measured wait in the opening scene.
Ray-traced lighting and post-cutscene performance remain open.

See [the report and patch](https://github.com/fxceless/victory-fork/issues/184)
for the implementation and measurements. Build the server from the exact Wine
revision of the installed Proton release. The local test used Proton
Experimental 11.0-20261112 and Wine revision
[`5b55760`](https://github.com/ValveSoftware/wine/tree/5b557605b4e51f0853ec8e0e4f0eb536c669fae3).
Start the matching server in the test prefix before launching Proton and
verify that the client connects to it.
