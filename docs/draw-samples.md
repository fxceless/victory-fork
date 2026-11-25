# Counting samples for a draw

In a diagnostics build, set `VICTORY_CAPTURE` to a capture directory outside the source tree,
select frames with `VICTORY_CAPTURE_FRAMES` or `VICTORY_CAPTURE_WORLD`, and set
`VICTORY_DRAW_SAMPLES=1`. Use a fresh capture directory for each run.

Each graphics draw that receives a query has a `draw-query=N` field in its frame's ops file.
The corresponding `frame-N-draw-samples.tsv` is headerless. Its columns are the query ID,
submission number, zero-based list execution ordinal in that submission, and sample count.

A zero means the draw ran and no samples passed the depth and stencil tests. A missing row
means the coverage is unknown. A positive count can include samples later overwritten by
another draw. For indirect draws the count covers the whole indirect command.

When measuring a frame, submissions containing a sampled list take the synchronous path.
If the same list executes multiple times in one group without an intervening fence, its readback
contains the last execution only. Per-list GPU timing separates those executions. Capture's
recording frame numbers can differ from the time the GPU executes the commands.
