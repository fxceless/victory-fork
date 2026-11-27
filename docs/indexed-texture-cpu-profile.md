# Indexed texture CPU profiling

Set VICTORY_TEXTURE_CPU_PROFILE=1 to log three CPU phases of indexed
texture streaming: queue upkeep; due selection and accounting; and
slot processing (including pass setup after selection). Explicit bake
source-only passes are excluded. The timers include early returns,
and sum CPU wall time across threads, normalized by reported frames.
Concurrent intervals can overlap; these are not critical-path durations.
The flag defaults off and does not change texture scheduling.
