# Validating Display-FPS measurements

Use completed PresentMon 2.x captures. The launcher metadata must contain
Trial, Commit, Executable, Log and Start, and the capture sidecar PID,
StartUtc, EndUtc and PresentMonExit. Collect telemetry at approximately
two-second intervals with UTC, Minimized, ProcessWorkingSetBytes and
optionally CompilerCount.

Run:

    pwsh -File scripts/Measure-DisplayCapture.ps1 -Metadata launch.json -Capture capture.json -PresentCsv present.csv -SystemCsv system.csv -RequireWarm -RequireNoCompiler -OutputJson metrics.json

Legacy telemetry without CompilerCount is accepted only if
RequireNoCompiler is omitted; the output says CompilerObserved=false.
A concurrent compilation is a confounder, not evidence of renderer
regression. Keep GPU tests and compilation outside capture windows.

The validator rejects a failed capture, missing/nonfinite data, mixed
PID/swapchain, negative frame intervals, less than 120 seconds of actual
display intervals by default, telemetry gaps over 15 seconds, minimized
windows, insufficient world batches, mixed menu/loading and title errors.
RequireWarm also rejects any pipeline creation or loading in the logged
frame windows. The first displayed interval is excluded, since it can
include pre-capture time. P50/P95/P99 use nearest rank. Low1 and Low01
are the reciprocal mean of the slowest ceil(1%/0.1%) displayed intervals;
Low01SampleCount discloses tiny tails. Stage times are medians of batch
means, not per-frame percentiles. Stage times overlap.

World classification is not scene matching. Use the same saved starting
state, settings, resolution, camera route, streaming phase and warm-up
for each A/B, at least three repeats in alternating order. Verify
screenshots/scene independently and do not send input during capture.
SceneMatched is deliberately false: the data alone cannot prove it.
Avoid drawing a causal conclusion from a single unmatched run.

Offline regression:

    pwsh -File scripts/Test-Measure-DisplayCapture.ps1

13 synthetic cases, including exact 25 FPS and malformed/contaminated
captures. CI runs them without a game or GPU. The new script also
reproduced the 26 Nov 2026 2d48097 baseline and copy-queue measurements:
5.553932825 and 4.576092381 displayed FPS, respectively. Neither capture
proved identical scenes. No game assets or raw gameplay logs are tracked.
