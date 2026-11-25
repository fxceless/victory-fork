# Lazy copy-back: merge only what the CPU touches

Status: **stopped after step 2** (2026-11-25). Steps 1 and 2 are in the
code behind `VICTORY_LAZY_BACK=1`/`2`, off by default. Step 2 crashes the
game (see "What step 2 showed"); do not turn it on for play.

## Why

On an RTX 4070 / Ryzen 5 7500F in the first gameplay room, the relay's
copy-back costs 44-55 ms of a 112-123 ms frame, and the title's fences are
set only once it is done, so it adds to the frame almost one for one:

- about 1,250 pieces of 64 KiB a frame are fetched from card copies (78 MB),
  96 % of them hold bytes shaders wrote, 34 MB of bytes that changed;
- merging them on several threads gained nothing (the cost is the kernel's
  writes into title memory and the fetches, not the compare).

`VICTORY_BACK_READ_PROBE=8` (step 0) marked one page in eight of what the
copy-back wrote with `PAGE_GUARD` and counted the first touch:

| per frame | |
|---|---|
| pages marked | 650-1,230 |
| first touched by a CPU read | **0.0** |
| first touched by a CPU write | 2.5-3 |
| first touched by our own access (mostly the next merge) | 820-1,220 |

The title's CPU code does not read back what its shaders write to these
buffers. Almost all of the copy-back is bytes only the card uses.

The same run went to about 1 frame a second: changing a page's protection
(`VirtualProtect`) on the title's big view memory is expensive, and the
probe did it some 2,000 times a frame. **A design that protects and
unprotects pages every frame cannot win.**

## Design: protect once

A 64 KiB page of title memory that shaders write and the CPU does not
touch becomes **card-only**:

- its title views (every address that shows the page:
  `crate::inplace` knows them) are made `PAGE_NOACCESS` once;
- from then on, a flight that writes it costs nothing on the CPU: the
  piece is not fetched, not merged, and the flight's fences are not held
  for it; the copy in the card's memory is the page's truth;
- the page stays card-only, with no further protection change, for as
  long as nothing on the CPU side touches it.

A page leaves card-only (**materialize**) when anything touches it:

- the title's CPU code (an access violation on a `PAGE_NOACCESS` page,
  taken by the vectored handler);
- a kernel call on the title's behalf (file writes, DirectStorage) that
  fails on it;
- the layer's own reads of title memory (`host::read`, `copy_mapped`,
  `read_into`, fingerprints) and writes (`write_mapped`).

Materialize: fetch the page from the card copy (a copy list on the copy
queue and a wait), merge it as the copy-back does now
(`Shadow::card_merge_relay`, through our own view so that no title view's
protection must change), give the title views their protection back. The
page is then ordinary until a flight's compare flags it again, when it
may become card-only again (at most once per N frames per page, so that a
page the CPU does touch every frame does not pay a protection change
every frame).

## Hazards

- **Deadlock.** Materialize waits for the card. A fault on a layer thread
  that holds the submission lock, a queue's lock or the guards' lock would
  wait for itself. Layer threads must never touch a card-only page in
  process memory; every layer read of title memory goes through the
  kernel wrappers, which materialize before they read, outside those
  locks. Faults taken on a layer thread are counted and must be zero.
- **Aliases.** One physical page is shown at several title addresses;
  all of them must be protected and given back together.
- **The guards.** `crate::watch` protects view memory read-only to see
  the title's writes. A card-only page's protection belongs to this
  module until it is materialized; the guards must leave it alone (and
  must not count it as written).
- **Kernel calls.** A kernel call that fails on a card-only page on the
  title's behalf must be retried after materializing; the places that
  call the kernel with title addresses (HLE file and storage modules)
  must be found and wrapped.
- **Refreshes of other copies.** A texture or another buffer copy over
  the same memory reads title memory to send it to the card: through the
  wrappers, so it materializes first.

## Steps, each tested in the game before the next

1. **Registry and materialize, no protection.** Card-only pages are only
   *noted*: flagged pieces still fetched and merged as now, and a counter
   says how many would have been skipped, per frame, and how many layer
   reads touched a noted page. Proves the bookkeeping, changes nothing.
2. **Skip the merge, keep the fetch.** Card-only pages are protected once
   and not merged; the fetched bytes are kept until the next flight's, so
   materialize needs no wait for the card. Fences no longer wait for those
   merges. Measures the gain with the least risk.
3. **Skip the fetch too.** Materialize fetches the page itself.
4. **On by default**, after runs past the first rooms, a save, a load.

`VICTORY_LAZY_BACK=1` turns it on at every step until step 4.

## What step 2 showed (RTX 4070 / Ryzen 5 7500F)

Step 1 measured 575-612 of about 1,360 fetched pieces a frame lying wholly
on candidates (35-38 MB), and our own touches of candidates at 0.3-5 a
frame. Step 2 then made pages card-only as predicted, and:

- **Every give-back went through the kernel** (`way="kernel"`), never
  through our own view: `inplace::write_own` needs the runtime's mapping
  lock (`watch::title_memory_reading_now`, a try-lock), which is taken
  whenever a touch happens, while the world streams. The kernel way gives
  the views their protection back *before* the bytes are in: a title
  thread can read the page unmerged in that window. The game crashed
  every time (a write to 0 in its own code, `+0x23cf683`), within a
  second of the first pages going card-only; never with
  `VICTORY_LAZY_BACK=1` or without the switch.
- **Pages were given back 1-3 frames after they went card-only**, all for
  our own access (refreshes, fills, textures reading title memory): each
  new card-only page is touched almost at once, so pages churn between
  card-only and back, paying two protection changes each time, and the
  merges skipped would be few.

Two faults were found and fixed on the way (a page went card-only at its
first fetch; a give-back wrote the card's piece whole over title bytes the
card had not received). Neither was the last crash.

What it would take: find which of our own reads touch these pages every
few frames and whether they need to (step 1's "touches of other noted
pages", about 1,240 a frame, is the same question for the rest), and a
give-back that never opens the views before the bytes are in (waiting for
the mapping lock outside the fault handler, or a write path that needs
no lock).
