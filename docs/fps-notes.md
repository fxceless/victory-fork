# Frame rate notes (RTX 5070 Ti, 32 GB, Windows 11)

Measured in the first prison cutscene and the start of gameplay, five-minute
runs, the "N fps, M ms/frame" line of `victory_gpu::host`, card time from
`VICTORY_GPU_PROFILE=2` (one timestamp per run of one pipeline; mode 1's
timestamp per draw inflates per-call times).

| Point | ms/frame | fps | card ms/frame |
|---|---|---|---|
| Fork at 578d459, first runs | 378-509 | 2.0-2.6 | 88-123 |
| Before merging 9376fd6 | 217-225 | 4.5-4.6 | 117 |
| After merging 9376fd6, upstream features below off | 191-200 | 5.0-5.2 | 107 |
| + table constants on the card, windows in GPU upload heaps | 178-200 | 5.0-5.6 | 110 |
| + indirect clamps in runs, mesh draws clamped by an amplification stage | 156-170 | 5.9-6.4 | 65 (with 0b0074e) |
| + fewer view-memory askings, wide fills/flushes/texture checks | 150-161 | 6.2-6.7 | 65 |
| + landing fetches from card slots (`VICTORY_CARD_FETCH_INFLIGHT=2`) | 137-157 | 6.4-7.3 | 64 |
| + heap sync null skip | 133-138 | 7.2-7.5 | 64 |

The card now needs ~64 ms a frame; the frame is bound by the submitting
thread (~95-110 ms under the submission lock: the descriptor heap sync
~30-40 ms, asking + stale checks ~16 ms, the lists' steps and flushes ~25 ms)
plus the title's own time between submissions.

## The settings of the last row

```
VICTORY_DISPLAY_MODE=1920x1080
VICTORY_CHEAP_LOOKS=1
VICTORY_ASYNC_TEXTURES=4 VICTORY_ASYNC_TEXTURES_MB=256 VICTORY_STAGING_POOL_MB=128
VICTORY_HEAP_TEXTURE_MS=60 VICTORY_HEAP_ROLLING_RESCAN=1
VICTORY_LATE_WAIT=3 VICTORY_WIDE_BACK=1 VICTORY_JOIN_FETCH=1
VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_CARD_BUFFERS_MB=1024
VICTORY_COMMIT_FLOOR_MB=6144
VICTORY_DECLARE_ONCE=0
VICTORY_FINE_PAGES=1 VICTORY_HEAP_SLOT_CACHE=1 VICTORY_ASK_ONCE_VIEWS=1
VICTORY_WIDE_COPY=1 VICTORY_WIDE_MERGE=1 VICTORY_WIDE_FILLS=1
VICTORY_CARD_NO_PC_COPY=0 VICTORY_WAVE_WIDE=0
VICTORY_HEAP_RECHECK=0 VICTORY_REUPLOAD_WHEN_FULL=0 VICTORY_VIEW_GUARD=0
VICTORY_HEAP_SYNC_TABLES=2 VICTORY_TABLE_INDEXED=1
VICTORY_GPU_UPLOAD_COPIES=1 VICTORY_GPU_UPLOAD_COPIES_MB=4096
VICTORY_TABLE_ON_CARD=1
VICTORY_INDIRECT_BATCH=3
VICTORY_GUARD_PLACEHOLDERS=1 VICTORY_AHEAD_INDEX=1 VICTORY_VIEWS_TRIM=1 VICTORY_VIEW_KNOWN=1
VICTORY_DECLARED_WIDE=1 VICTORY_WIDE_TEXTURES=1 VICTORY_STALE_ONCE=1
VICTORY_CARD_FETCH_INFLIGHT=2
VICTORY_SYNC_NULL_SKIP=1
```

(`VICTORY_MESH_AMPLIFY=1` gives about the same card time once
`VICTORY_INDIRECT_BATCH=3` batches the mesh draws.)

## What made the big differences

- **Indirect clamp stalls.** Every indirect record was clamped by a compute
  pass between two waits: 7-12 us of card time per call, ~4,700 calls a
  frame. `VICTORY_MESH_AMPLIFY=1` clamps mesh records in an amplification
  stage; `VICTORY_INDIRECT_BATCH=3` clamps a run of calls in one pass, a run
  ending only at work that can store to a buffer. Card 105 -> 64 ms.
- **Landing fetches.** Each landing fetched the flights' changed card pieces
  with a copy list on the main queue, behind every later flight (~60-105 ms a
  frame of waiting). `VICTORY_CARD_FETCH_INFLIGHT=2` copies the flagged pieces
  into slots in card memory inside each flight and copies the slots down on a
  copy queue at landing. Shader writes into the PC's memory (mode 1) fall off a
  cliff on this card: 64 MB took ~600 ms against 0.74 ms into card memory.
- **Windows in GPU upload heaps** (`VICTORY_GPU_UPLOAD_COPIES=1`): draws stop
  reading vertex/index/constant windows over the bus.
- **Table constants on the card** (`VICTORY_TABLE_ON_CARD=1`): a table's
  constant view points at the buffer's card copy instead of a CPU fill of
  bytes an earlier pass wrote, removing a per-frame wait for the card.

## Upstream defaults that changed the picture here

With `VICTORY_HEAP_RECHECK`, `VICTORY_REUPLOAD_WHEN_FULL`, `VICTORY_VIEW_GUARD`
on and `VICTORY_HEAP_SYNC_TABLES=0`, the scene had a blocky strip along the
bottom and blocks by the camera, and the card was ~16 ms slower. The re-check
fingerprinted past a texture's own bytes into the next texture placed in its
large levels and uploaded it again with them (fixed here: it stops at the next
texture's start); refresh-when-full never ran with async texture workers (fixed:
done in the pass). `HEAP_SYNC_TABLES=0` leaves three gaps (root-indexed entries
in old tables, per-lane array indices, whole-heap picks of slots read through a
table); `=2` defers only constant-buffer and placed records one frame.

## Tried and rejected (wrong picture or no gain)

- `VICTORY_SETTLE_LISTS=2`, `VICTORY_RELAY_COPY_QUEUE`, `VICTORY_QUEUE2` with or
  without the picked-view guard, `VICTORY_SYNC_WIDE`, `VICTORY_SYNC_OVERLAP=1|2`,
  `VICTORY_HOT_BLOCKS`, `VICTORY_FLUSH_ONE_PART`, `VICTORY_INDIRECT_BATCH=2`,
  `VICTORY_TABLE_GPU_COPY=1` (before 33c01da): picture garbage. Most of these
  read card copies that shaders also write through UAVs picked from the heap
  by index (the skinning programs), which no flight declares.
- `VICTORY_RELAY_FREE_RAN=1`, `VICTORY_FENCE_EARLY=1`: the wait moves into the
  title (its fences are set at landing); no gain.
- `VICTORY_ASYNC_SUBMIT=1`: the room's walls are not drawn (the title reads an
  older frame's results); `VICTORY_FLIP_LANDED` (72dbd10) is the attempted fix,
  not yet measured.
- `VICTORY_LATE_WAIT_LISTS=256`, `VICTORY_DISPLAY_MODE` (card time unchanged:
  the title does not draw smaller for a 1080p mode), `VICTORY_INPLACE=1` (only
  392 MB of windows went in place; slower).

## Still open

- The run-to-run state of the scene (window wire mesh drawn or not, left wall,
  the top's colour) follows textures held back under the commit floor (a
  held-back slot reads as a null view). `VICTORY_LEVELS_ON_DEMAND=1` with
  `VICTORY_PRESSED_LEVELS_KB` gives such slots their small levels instead.
- `VICTORY_PICKED_EXACT=1` (c5ab405) declares the skinning programs' writes
  from their job records: not yet measured.
