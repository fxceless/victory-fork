# Submission stage diagnostics (Kelvaroq 0053 subset)

Source concept: https://github.com/stellarr-studio/kelvaroq/blob/1270c422adaef1e0f369f670138d8d29d8f6f4c3/native/patches/0053-submit.patch (GPL-3.0, Kelvaroq contributors). AI-assisted adaptation.

Set VICTORY_SUBMIT_STAGES=1 on a fresh process. Reports title submission wall-clock time per frame for synchronous descriptor sync, waiting descriptor realization, join of the parallel descriptor pass, and the CPU duration of the direct queue ExecuteCommandLists call (both late and synchronous submissions). Reports calls/frame and lists/call. Signal, event and fence waits are not counted in the execute call. Parallel worker runtime is not measured by the join duration. Auxiliary copy queues and other GPU work are excluded. Existing preparation stats overlap these new measurements: do not sum them.

The default is off. No ordering, batching, guards, descriptor publication or GPU resource lifetime changes. Further 0053 batching and refresh-ahead changes require a separate correctness review. The CPU aggregate test checks durations, call/list counts and reset. Local compile/hardware testing deferred during the active benchmark; CI required before promotion.
