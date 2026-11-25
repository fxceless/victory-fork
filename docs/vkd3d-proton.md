# vkd3d-proton 3.1 and the world's compute programs

Under Wine or Proton, Direct3D 12 goes through
[vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton), which turns
the DXIL we emit into SPIR-V with
[dxil-spirv](https://github.com/HansKristian-Work/dxil-spirv). vkd3d-proton
3.1 crashes the title as gameplay starts on NVIDIA. Use a build that has
dxil-spirv's new structurizer.

## What goes wrong

vkd3d-proton 3.1 bundles a dxil-spirv that still uses its legacy
structurizer. Some of the world's compute programs have a loop that is only
ever left by returning: console programs end a lane that way, and the
translator's loop limit (`abort_block`) does too. For such a loop the legacy
structurizer emits a merge block with no terminator, and
`spirv-val --target-env vulkan1.3` rejects the module.

NVIDIA's compiler does not reject that SPIR-V cleanly. It crashes or hangs
inside `CreateComputePipelineState`, and the process goes down with it, at the
first world compute pipelines after the intro.

dxil-spirv's new structurizer translates the same DXIL into valid SPIR-V. It
is dxil-spirv's default from `7c04863`, and vkd3d-proton `master` uses it from
`ec20719` (2026-11-24).

Giving such loops a single exit block in our emitter does not help: the legacy
structurizer then emits a phi whose incoming blocks do not match its
predecessors, which is invalid in another way and hangs the driver instead.

## What to use

- A vkd3d-proton release after 3.1, or a `master` build from its
  [Artifacts (Package)](https://github.com/HansKristian-Work/vkd3d-proton/actions/workflows/artifacts.yml)
  workflow, as the prefix's `d3d12.dll` and `d3d12core.dll`.
- A Proton build is fine if the vkd3d-proton it bundles includes `ec20719`.

## Checking a build

`VKD3D_SHADER_DUMP_PATH=<dir>` makes vkd3d-proton save each program it
translates as `.dxil` and `.spv`. `spirv-val --target-env vulkan1.3` on the
`.spv` files should report nothing. With a standalone dxil-spirv,
`dxil-spirv <file>.dxil --output out.spv` uses the new structurizer and
`--legacy-structurizer` reproduces 3.1's output.

## Tested on

Fedora 45, Wine 11.21 (Staging), NVIDIA RTX 5050 with driver 620.70.06,
DXVK 3.1.1 `dxgi`. With 3.1 the run died at the first world compute pipelines
every time. With `master` at `ec20719` the intro cutscene played with audio into
gameplay, binding 3,042 pipelines with none failing.
