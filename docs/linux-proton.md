# Running this fork on Linux, under Proton

Status, 2026-11-24. One machine: AMD RX 7900 XT (RDNA 3, native 64-lane waves),
Mesa 26.3.1 RADV, Vulkan 1.4, CachyOS kernel 7.3.5, KDE Plasma 6 on Wayland;
Proton Experimental's `wine` (experimental-11.0-20261112) run directly, with
its vkd3d-proton and DXVK `dxgi`; Series X profile; title 1.0.2.1. Numbers
only; no assets.

## What ten minutes gets you here

The first-boot screens, the Rockstar splash and the loading screen, with the
world assembling behind it. The frames saved at 512, 1024, 1536 and 2048 show
the splash and the loading art, so presentation works. What follows therefore
measures **the loading phase**, not a played frame: 700–6 300 draws per frame
of streaming, 4–10 s per frame.

Two things to keep apart, because both are in the log ten minutes in:

- The **saved render targets** at those frames are splash and loading art.
- From about two minutes in, the frames carry world-sized draw counts
  (700–6 300) while the GPU accounts for 54–241 ms of a 4–10 s frame, and the
  screen shows a scene lit in places and black elsewhere with geometry
  missing — the same picture the Windows reports describe (#70 and the
  missing-asset issues), i.e. the world's own frontier rather than a loading
  screen. A 60-minute run on `main`, unpatched, reached that state and stayed
  in it without a device loss.

(An earlier version of this note called all of these "world frames" and then
corrected itself to "loading only". Both were half right: check the saved frame
*and* the draw count, and watch the window.)

## What the run needs

- **Wine's own `d3d12` must be off.** With the builtin stub in the chain,
  device creation returns `E_INVALIDARG` and every frame submits 0 draws:
  `WINEDLLOVERRIDES="d3d12=n;d3d12core=n;dxgi=n"` (Proton's vkd3d-proton and
  DXVK then serve them).
- **HDR must be off on the output the window lands on.** With HDR enabled on
  that display, presentation blocks for about 400 ms per frame – 2.5 fps in
  the menus, and the 30 s watchdog then ends the run with `int3` at a
  frame-64 boundary, which reads exactly like a title crash. With HDR off the
  same build runs the menus at 15–30 fps. Measured while building upstream
  `main`, not re-tested on this fork; the mechanism is in the compositor and
  the present path, so the layer is unlikely to matter, but that is an
  inference.
- `VICTORY_NO_WATCHDOG=1` while experimenting, `VICTORY_SHADER_LOOP_LIMIT=256`
  as on Windows.
- **Microsoft's `d3dcompiler_47` is needed, as a native override.** The clamp
  and the card compare are compiled at run time from HLSL that uses
  `RWByteAddressBuffer.InterlockedMax`, `Load4` and `[RootSignature(...)]`.
  Wine's builtin compiler is the vendored vkd3d-shader, which has none of those:
  forcing it (`WINEDLLOVERRIDES="d3dcompiler_47=b"`) fails the fork's own
  `indirect::tests::argument_clamp_builds` with `E5005: Method 'InterlockedMax'
  is not defined on type 'RWByteAddressBuffer'`, and the same source compiled
  directly gives the same error. With `d3dcompiler_47=n` pointing at
  Microsoft's DLL the test passes and the container keeps its root signature.
  Install it with `winetricks d3dcompiler_47` (or point the override at any
  `fxc` build); without it the indirect clamp and the card compare are lost and
  the frame line says `no argument clamp`.

## Where the frame goes

Two ten-minute runs of the same tree, averaged over the frames past boot (more
than 800 draws/frame). The second had a warmer shader cache; the difference is
spread, not a change.

| | cold cache | warmer cache |
|---|---:|---:|
| frame | 7 892 ms | 5 173 ms |
| preparing | 6 417 ms | 4 261 ms |
| - copies | 2 793 ms / 172 MB | 1 969 ms / 197 MB |
| - textures | 2 787 ms / 38 MB | 1 581 ms / 65 MB |
| **on the GPU** | **134 ms** | **163 ms** |
| copying back | 1 091 ms / 38 MB | 637 ms / 65 MB |
| draws/frame | 2 990 | 2 783 |
| dropped for a missing pipeline | 1 140 per frame | 958 per frame |

The drops are pipelines still being made (peaks of 4 362 and 2 204 in a single
frame), so these are compilation runs, not steady state. The GPU is idle for
96 % of the frame. The rate the obvious division gives — copied bytes over the
lane's time, about 60–100 MB/s for buffers and 7–41 MB/s for textures — is not
the transfer speed, and the layer's own counters say why: to find 186 MB of
changes in one frame the refresh asks the kernel about **21.7 GB** of ranges
(`GetWriteWatch`, 3 552 calls, 1 169 ms of the 3 428 ms lane), and asks about
117–199 times more than it copies. The kernel reads themselves run at
0.45–1.45 GB/s in situ, and the write-watch askings at 3–19 GB/s of range, so
the lane is per-call-bound, not bandwidth-bound; 56–73 % of it is host
bookkeeping the current build does not attribute. That is the difference
against #10's roughly 1 GB/s: on Windows those per-call costs are cheaper.

## It is not the runtime underneath

A 128 MiB probe run in the same prefix, on the same device, doing only what
the layer does per frame – write through an upload mapping, copy on the card,
read back through a readback mapping:

| operation | rate |
|---|---:|
| writes into a mapped UPLOAD buffer | 21.3 GB/s (6.0 ms per 128 MB) |
| card copy, upload buffer to a default-heap buffer | 143-163 GB/s (0.9 ms per 128 MB) |
| card copy, default heap to a READBACK buffer | 24.6-25.1 GB/s (5.2 ms per 128 MB) |
| reading a mapped READBACK buffer | 12.6 GB/s (10.2 ms per 128 MB) |

The diagnostics build's own per-buffer statistics agree: its largest per-frame
copy is 156 MB and takes **2.46 ms** on the card (~63 GB/s), and the eight
largest are 0.3–2.5 ms each. The frame reports ~2.8 s under "copies" for
172 MB, so that figure is the host-side part — finding and gathering the
changed ranges — not the transfer.

So vkd3d-proton on RADV is two orders of magnitude faster than the rates a
frame shows. The 2–4 s of copies and textures per frame is the layer's own
per-frame work – the ranges it sends up, the pieces it hashes and compares on
the card, and the waits for those – not the memory paths beneath it.

## Not established yet

- Which part of that per-frame path costs the time: hashing and comparison of
  pieces, the waits for the card, or range bookkeeping. `VICTORY_CARD_BUFFERS=0`
  and `VICTORY_CARD_REHASH_RUNS=1` are the switches to try first.
- Which Wine/vkd3d release would make the builtin compiler enough. Nothing to
  pin: Wine has used vkd3d-shader for HLSL since 7.5, and no vkd3d release
  through 2.2 has these methods (`Load*` since 1.14, the `Interlocked*`
  intrinsics since 1.15, the `RWByteAddressBuffer.Interlocked*()` methods only
  upstream since 2026-11-04 and in no tag). `[RootSignature]` is not
  implemented at all; `RTS0` exists only as a standalone serializer. Wine 11.22
  vendors vkd3d 2.1, Proton Experimental runs wine-11.0, and neither has it.
- What the world costs here, since ten minutes of loading does not reach it.
- The AMD-specific pipeline skips from #3 (`CS_RaytracingResolve`,
  `fc5e66f8f873d3bb`) were not needed here: no device loss in about 30 minutes
  of running, and RADV never hands a program to `amdxc64.dll`.
