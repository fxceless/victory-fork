# Findings on a 6 GB laptop GPU (2026-11-24 and 25)

What was run, read and changed while bringing the title up on a machine well
below the fork's test machine, and what is still open. Written for whoever
picks this up next; every number below comes from this machine's logs and
captures (none of which are in the repository: see
[asset-policy.md](asset-policy.md)).

| | Fork's test machine | This machine |
|---|---|---|
| GPU | RTX 4070, 12 GB | RTX 3060 Laptop, 6 GB (32-lane waves only, like every NVIDIA GPU) |
| CPU | i7-14700KF | i7-11800H, 8 cores |
| RAM | 32 GB | 16 GB (about 1 GB free during a world run) |
| Intel iGPU | none | present; the host device is made on the NVIDIA GPU (checked by its memory counters) |

## How to run it

```powershell
cargo build --release -p victory
# warm the shader cache as this GPU's runs read it (NVIDIA: 32-lane waves,
# a view heap capped at 1,000,000 views, as a run's log says)
$env:VICTORY_HOST_WAVE=32; $env:VICTORY_VIEW_LIMIT=999999
target\release\victory.exe translate <the title's shader archive> --warm all
target\release\victory.exe translate <game dir>\XboxBvhBuilder.dll --warm all
Remove-Item Env:VICTORY_HOST_WAVE, Env:VICTORY_VIEW_LIMIT
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

Warming (upstream's `translate --warm`) translates every shader container
into the cache a run reads; the whole pack (26,503 containers) takes about
2.5 minutes here. The cache is keyed on the translator's source, so a build
that does not touch the translator starts warm. The translation has to be
the one a run on this GPU makes: its wave width and view-heap limit, which
a run's log gives ("host wave lanes ... max=32"; "host view heaps hold
1000000 views", whose last slot, 999999, is the limit). This branch had
a warm-up of its own that made the host device first to get them; upstream's
replaced it. Driver pipelines cannot be made ahead of a run (they need the
render-target formats and other state the title passes when it creates
them); every pipeline a run uses is kept and made again at the start of the
next one.

## The runs

About twenty runs, Series X profile unless said otherwise; logs and captures
stayed outside the tree.

| Run | What it showed |
|---|---|
| First, cold caches | Menus at 30 fps; the opening cutscene as a blue screen with its sound: about 1,300 pipelines still being made, 800-4,000 draws a frame dropped for them, 0.2-1 fps. Ended after 8.5 minutes by the title itself: its object pools were full (instances 45,000 of 45,000, buildings 22,272 of 22,500, 3,000-6,000 streaming requests queued). Its crash reporter tried to upload a minidump; the network block refused it. |
| Second, warm pipelines | Same, ended the same way after 8 minutes. |
| Third, all pipelines made | 0.2-0.7 fps, 0 pipelines left to make but still 500-1,000 draws a frame dropped (see "What is dropped"); a frame took 1-6 s, most of it uploading textures. |
| Series S profile | Black from the start: the title presents two planes and only the first was shown (below). In the world, the host device was removed within seconds (below). |
| Captures (`VICTORY_CAPTURE`, `VICTORY_DUMP_TARGETS`) | The world's geometry is complete in the G-buffer's normals; walls and skin black in its albedo (below). |
| Texture experiments | Bigger budgets and the shared-memory spill (below): faces and walls get their colour. |
| Ray-tracing builder on | See "Ray tracing". |

## What is dropped, and why

The frame line's "dropped N without a host pipeline" counts dispatches too.
In a captured world frame 664 of the drops were the ray-tracing structure
builder's stages (off by default), a few were compute dispatches of the five
programs this GPU refuses (64-lane waves, below), and none were the world's
draws. So once the pipeline cache is warm, missing objects are not dropped
draws.

## Black walls, faces and clothes

The G-buffer of a world frame has every wall, door, arm and face in its
normals target, while its albedo target is black for skin and building
materials and coloured for clothes and hair. The materials that are black
pick their textures from the view heap by a computed index; the coloured ones
read theirs through descriptor tables. Textures picked by index are uploaded
by a queue (`draw.rs`, `heap_textures`) that stopped for two reasons on this
machine:

1. Throughput: 192 MB and 250 ms a pass. The world's first minutes had
   11,000-27,000 slots waiting, the oldest written 2,000-3,000 frames
   earlier.
2. Room: the queue stops once the card's memory in use reaches nine tenths of
   the budget Windows gives the process. On 6 GB that is reached in the
   world's first frames.

Measured with captures of the same shots:

| Setting | Albedo |
|---|---|
| Defaults (then) | walls, skin, uniforms black |
| `VICTORY_TEXTURE_BUDGET_MB=3000` alone | still black: the queue's own limits held it |
| plus 2 GB and 4 s a pass | walls, beds, rails, skin of most characters |
| plus the shared-memory spill | the main character's face, brows and lips; the concrete walls |

Upstream reached the same queue from another side at the same time: textures
uploaded before the title had filled them kept zeros (now looked at again
after their upload), and a catch-up budget (three times the bytes and time
while 1,024 or more slots are due). This branch keeps upstream's queue and
adds the spill: past the card's share, textures picked by index go on into
the PC memory the system shares with the card, up to half of that budget
(`VICTORY_TEXTURE_SPILL=0` turns it off). A texture there is read over the
bus: slower, not missing.

Still open: the queue does not know which slots the frame samples. The fix
that removes the problem instead of shortening it is GPU feedback: each
translated program that picks a texture by index sets the slot's bit in a
buffer, read back the next frame, and those slots go first.

## Light in shadow

Anything the sun does not reach directly looks like dusk: the title's bounce
light and its lamp and sun shadows are ray traced, and the ray-tracing
structure builder has never run in this fork (below). The depth bounds test
(upstream, #82) lit the first room by its lamp on the fork's machine.

## Ray tracing

The builder runs all or none: one builder program without a host pipeline
keeps the whole builder off (`pipeline.rs`, `Builder`). On this GPU three of
the 53 were missing, all three for the same reason: block-wide prefix sums
over a 512-thread group of 64-lane console waves. Each wave writes its total
to shared memory, the group meets at `s_barrier`, and each lane then sums the
totals of the waves before its own in a loop whose exit depends on its wave's
number, broadcasting the running sum with `v_readfirstlane`. Run as two
32-lane host waves per console wave, a first-lane read inside a branch only
some lanes take was refused (the halves cannot meet there).

The running sum is the same in every lane of the console wave: it is a scalar
plus a shared-memory read at a scalar address. `lane_uniform` now proves that
for shared-memory reads whose memory has settled (`lds_settled`: no write
follows the read on any path, and every write before it reaches it only
through an `s_barrier`), so both halves take their own first lane, which
holds the same value. Upstream had added `VICTORY_FIRSTLANE_OWN=1` for these
same scans as an opt-in shortcut; the proof makes it automatic where it
holds. All 52 builder programs of the title's DLL now translate, pass
`dxil.dll` and make pipelines on the RTX 3060 (49 before).

A test runs the same shape on the GPU (`wave64_tests.rs`, a scan over the eight
console waves of a 512-thread group, written for the test): on the RTX 3060,
as two 32-lane halves per console wave, every thread gets its waves'
predecessors' sum. Making it pass also fixed an over-cautious check in
`lane_uniform`: it took the destination's old value, which the IR records as
read, for one of the operand's registers, so an instruction whose
destination was not one of its sources was never uniform.

In a run (`VICTORY_BVH_BUILDER=1`), for the first time in this fork:
"the ray-tracing structure builder runs: all of its programs have host
pipelines pipelines=53". It is not usable yet:

- Builder on: 13 s after the builder's first frame the title faulted on
  pointers that held float bit patterns (`0x4084003e99`) and all-ones, as in
  the fork's run 98: a builder stage stores where it should not, over the
  title's own objects.
- Builder on with `VICTORY_BVH_QUARANTINE=1` (its stores stay in the layer's
  copies): the title's memory stays whole, and 7 s later the layer could not
  make its 4 MB host copies of title memory any more (4,889 refusals in 6 s,
  out of memory), then a command list failed to close. A stage that computes
  wild addresses makes the layer copy memory it was never meant to touch.

Both point at a builder stage computing wrong addresses. Not the three
scans: the GPU test above checks their form. The next step is to check each
of the builder's stages on the GPU against its expected output (the fork's
`bvh.rs` reference covers the ray test, not the builder), starting from the
stage whose dispatch first names memory outside the structures' buffers.
The builder stays off by default, so none of this touches a normal run.

## Series S profile

- The host device was removed seconds into the world: the title makes a
  1920x1080 R9G9B9E5 texture with render-target and unordered-access use,
  which the console GPU allows and the PC does not (it only samples that
  format). A stand-in format fixes it (`target_format`). This branch first
  used R11G11B10_FLOAT, as XCL does for the same texture; upstream's
  RGBA16F, which keeps every bit of R9G9B9E5's precision, replaced it.
- The screen stayed black because the title presents two planes: its 3D
  plane (1920x1080) and its UI plane (3840x2160), and the layer showed the
  first only. The plane description is 72 bytes, not the 24 the layer read
  (the second plane's count read as garbage). The layout is now right and
  logged when it changes; composing the planes is still to do.

## Shaders

Pack-wide, as for this GPU (`VICTORY_HOST_WAVE=32 victory translate
<the shader archive> --validate`): every one of the 26,084 programs is lowered, and
254 containers do not translate on the run's path.

Fixed while this was written, each checked with `--validate --pipelines` on
the RTX 3060: the deferred material pixel programs (a first interpolation
half read where no second half is), the puddle programs and a shallow-water
reduction (descriptors read only for an image's size or a buffer's record
count), two water programs (double-precision whole-wave sums), and the
programs whose only cross-lane code is `v_permlane16`/`x16`. This branch had
fixes of its own for the first two; upstream's (pairs evaluated as their
pair on a retry, `unused_descriptor_kind`) replaced them, and all those
programs still translate. What this branch keeps is the console-lane order
for `v_permlane16`/`x16` programs (upstream lowers the instructions but did
not ask for the console's lane order there).

Still failing, by cause (the pack census, 32-lane host):

| Programs | Cause |
|---|---|
| 5 compute (alpha prefilter, particle visibility, reflection tiling, two triangle-clip programs) | still 64-lane waves on a 32-lane GPU: shared memory read and written inside a branch only some lanes take, where the split halves cannot meet |
| 2 compute (water volumetrics) | translate and validate since the double-precision sums, but the driver refuses the pipeline |
| 1 compute (cascade shadow tiles) | a lane mask exchanged where not every lane runs (32-lane GPUs only) |
| 195 | an address in an SGPR that is no root parameter |
| 71 mesh | the validator: vertex plus primitive output over 32 KB |
| 72 | the validator: input or output signature not as the module expects |
| 70, mostly mesh | the validator: elements of different interpolation modes packed into one row |
| 40 | a descriptor chosen by control flow |

## Intro video stutter

The opening video drops to 3 fps while the pipeline workers make the cached
pipelines (8 workers on an 8-core CPU, below normal priority, but every core
busy) and holds 24-30 fps once they are done. A start-up cost only.

## Stability

- One GPU hang in about twenty runs (the driver reset itself); the runs on
  either side with the same shader changes did not hang.
- One access violation inside the NVIDIA driver's own code, in a run with
  the builder switch on but the builder off, on a machine with about 1 GB of
  RAM free. A driver failing an allocation crashes this way; the pipeline
  workers now wait while the system's commit is nearly used up (below).
- With the builder running, see "Ray tracing".
- Later launches stopped at the title's start screen until a key was
  pressed (earlier ones had someone at the keyboard): unattended runs need
  `VICTORY_KEYS` (diagnostics builds), e.g. Enter every 15-30 s from 45 s.

## XCL

[XCL](https://github.com/xcl-dev/xcl) runs the same kind of title another way:
every console lane runs every instruction (EXEC only masks writes), the
title's graphics memory sits in up to four committed windows that D3D12 opens
as heaps (shaders reach it by address), and translated programs go through
HLSL and DXC. It is GPL-2.0-only and this project GPL-3.0-or-later: ideas
only, written anew. Its notes on the same title (menus reached, not the
world) gave these, now in this branch:

- Device methods 57 and 67 are `Deserialize{Graphics,Compute}PipelineStateX`:
  the console driver's pipeline cache. The title makes the pipeline from its
  description when they fail, so failing is right; the layer's log called
  every call a lost pipeline (tens of thousands a run). Now one line.
- Compute is waited for, graphics is not: a dispatch whose pipeline is still
  being made after the frame's wait now waits for it instead of being
  dropped (`VICTORY_COMPUTE_WAIT=0` drops it as before). A dropped dispatch
  leaves what it computes unwritten for later passes and frames.
- Pipeline workers wait while less than 2 GB of the system's commit is left
  (`VICTORY_PIPELINE_COMMIT_MB`); XCL found the title dying in the driver's
  compiler under memory pressure and makes its largest pipelines one at a
  time with 3 GB free. A pipeline a title thread waits for never waits.

Considered and not taken:

- XCL's shared memory windows would end the copies of title buffers, but not
  the texture copies (XCL converts textures too), and they commit their
  windows whole, which this 16 GB machine cannot afford. This fork's own
  first stage of the same idea is `VICTORY_INPLACE`.
- XCL's split of 64-lane programs into two 32-lane halves that meet through
  shared-memory counters works because its control flow is the same in both
  halves; this fork's per-lane branches are faster but let the halves part,
  which is why the five programs above still need 64-lane waves. Porting it
  means a second way of lowering such programs.

## Switches added on this branch

| Switch | Default | What it does |
|---|---|---|
| `VICTORY_TEXTURE_SPILL` | on | textures picked by index may go on into shared memory |
| `VICTORY_COMPUTE_WAIT` | on | a dispatch waits for its pipeline |
| `VICTORY_PIPELINE_COMMIT_MB` | 2048 | pipeline workers wait below this much free commit |

## Next

1. GPU feedback for textures picked by index (which slots the frame samples).
2. Find the ray-tracing builder stage that stores outside the structures'
   buffers (above), then the trace programs end to end; the five shadow-mask
   trace programs.
3. The five 64-lane compute programs: a split form for shared memory inside
   per-lane branches.
4. Signature rows of mixed interpolation modes (70 programs) and mesh outputs over 32 KB (71).
5. Compose the Series S UI plane over the 3D plane.
6. GPU time per pass on this GPU: frames are 1-6 s, most of it uploads.
