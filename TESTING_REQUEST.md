# Someone please test this fork

**No game files are provided.** Please use your own game files; do not include
game executables, dumps, assets or extracted shaders in test submissions.

Requested targets (not yet achieved or verified here):

| Hardware class | FPS goal |
|---|---:|
| Ryzen 5 7500X3D + RX 9070 GRE, or similar | 90+ |
| Faster hardware | 120+ |
| Slower hardware | 60+ |
| Lower-end hardware | 30+ |

Please test high visual quality, sustained gameplay and stutter. This Xbox build
does not expose a PC Ultra preset, so record the exact console profile, output
resolution, upscaling/internal resolution, ray tracing and enabled effects.

Please share CPU, GPU/VRAM, RAM, Windows/driver version, commit, settings, scene
and route, cold/warm-cache state, a gameplay video, frame CSV/summary and the
normal runtime log. Check for texture pop-in, missing draws, lighting/water
errors, crashes, memory growth and incorrect simulation speed as well as FPS.

Follow [the measurement instructions](docs/performance-targets.md). Rust build,
gameplay and the requested FPS results are still unverified on this development
host. The launcher/analyzer tests use synthetic data and are not game benchmarks.

Prepared by Cebo Ndlovu (Leemutachii).
