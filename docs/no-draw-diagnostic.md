# No-draw diagnostic (Kelvaroq 0061 adaptation)

Source: https://github.com/stellarr-studio/kelvaroq/blob/1270c422adaef1e0f369f670138d8d29d8f6f4c3/native/patches/0061-no-draw.patch (GPL-3.0; Kelvaroq contributors; original patch credits Cold Project VI). AI-assisted adaptation.

Set VICTORY_NO_DRAWS=1 in a fresh process for a diagnostic title-pipeline suppression run. Both the normal and render-target-specific host pipeline paths are disabled; the title pipeline creation path skips early builds and BVH builder registration. The builder is explicitly disabled. No defaults change.

This is not a normal gameplay mode and does not produce a valid rendered-FPS comparison. Dropping compute and graphics pipelines can alter streaming, simulation and GPU-dependent title behavior; other GPU and runtime operations can still execute. Do not compare it as an equivalent frame workload or assume a locked 30 FPS ceiling. Explicit offline precompile is not part of this test. Use an isolated cache and disable additional warmup when benchmarking.

A fresh-process regression checks a synthetic compute shader, both pipeline lookup entry points, builder suppression, no queued host pipelines and no shader translation. Local GPU and compilation tests deferred to avoid interfering with the active RTX 4090 benchmark; CI required before promotion.
