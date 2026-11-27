# Shared indexed-texture CPU budget (experimental)

VICTORY_TEXTURE_FRAME_BUDGET=1 shares a per-frame wall-time budget among
ordinary texture passes, including early streaming and the heap worker.
VICTORY_TEXTURE_SLICE_MS selects a positive cap (default 64 ms). The
previous observed queue backlog scales the budget from one quarter to
the full cap. Competing ordinary passes leave work queued rather than
waiting. A reserved older due entry is offered a turn every eight frames.
The existing per-pass byte/time limits and explicit bake budgets remain.
A single slot and queue upkeep can exceed the target; this is not a hard
frame-time cap. Delayed texture visibility is possible. Default off.
No validated gameplay FPS gain or visual parity claim.
