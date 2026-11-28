# Title memory layout (build 1.0.2.1): what is known

Status, 2026-11-26: **static analysis only, nothing here is verified at run time.** Each section says what was seen and what is still a guess; the checks at the end are how to confirm or refute it. Names of title objects below are ours, chosen for this document.

**AI-assisted, with human direction and review.** Method: static reading of the 1.0.2.1 image with a disassembler and small scanners. The image was never executed and I do not have the game data. Following `asset-policy.md`, this document holds no title code, listing, string or table: only addresses, field offsets and sizes, described in our own words. Addresses are image VAs with the image at `0x140000000` (subtract that for an RVA).

This is documentation, not a fix proposal. The resource-state errors seen on this buffer (upstream issue 199) were already handled on upstream `main` by `315cea1`.

## Summary

Three topics, in this order: the 174 MiB buffer (sections 1–5), the streaming state the title keeps in memory (section 6), and the ped skinning buffers (section 7). The 174 MiB buffer first:

The `title buffer X+0xae00000` (182,452,224 bytes) that shows up in many run reports is not one heap with one purpose. The title reserves one address range at graphics start-up, splits it into five parts, and registers the **last three parts as a single GPU resource**. Their sizes are constants:

| Part | Offset in the buffer | Size | What the title does with it |
|---|---|---|---|
| A | `0x0000000` – `0x1b00000` | 27 MiB (216 pages × 128 KiB) | Per-command-list dynamic data **written by the CPU**, then bound as vertex/constant data or indirect arguments |
| B | `0x1b00000` – `0x4a00000` | 47 MiB | Per-frame draw/instance data in 48-byte elements (contents unconfirmed) |
| C | `0x4a00000` – `0xae00000` | 100 MiB | Same system as B, separate budget (contents unconfirmed) |

27 + 47 + 100 = 174 MiB = `0xae00000`.

The title creates **both a whole-buffer SRV and a whole-buffer UAV** over this resource, and it also binds ranges of part A as vertex/constant data. So one resource is read as vertices, read bindlessly by draw shaders, and exposed for shader writes.

## 1. How the block is created

- `0x141e86440` takes five sizes. It reserves the sum with the title's `XMemVirtualAlloc` wrapper (`0x145f3fcc0`), releases the range piece by piece, and re-allocates each of the five pieces with its own attributes (`0x1004`, `0x1404`, `0x8404`, `0x2404`, `0x2004`).
- It then registers parts 3–5 as one resource through the device object (`*0x1475df048`, vtable `+0x428`), with size `p3 + p4 + p5`, and builds a raw 32-bit-element buffer object over it at `0x1475df0e8`.
- Its only caller is `0x145c5ff00`, called from the graphics start-up routine `0x142478440`.

Globals it fills:

| Global | Meaning |
|---|---|
| `0x1475df188` | base of the whole reservation |
| `0x1475df190` | size of parts 1 + 2 (descriptor tables) |
| `0x1475df138`, `0x1475df140` | base of part A (= base of the 174 MiB resource) |
| `0x1475df148` | base of part B |
| `0x1475df150` | base of part C |
| `0x1475df168`, `0x1475df170`, `0x1475df178` | sizes of A, B, C |
| `0x1475df158`, `0x1475df160` | page count and page size of part A |

## 2. Where the sizes come from

The sizes come from a `0x1e8`-byte graphics parameter block (default constructor `0x145eb1c60`, copied to `0x14a9437b8` by `0x1419f4df0`). The title fills it in `0x1443a47c0` with constants:

| Field | Value | Result |
|---|---|---|
| `+0x158` page size, `+0x15c` page count | `0x20000`, 216 | part A = `0x1b00000` |
| `+0x160` | `0x2f00000` | part B |
| `+0x168` | `0x6400000` | part C |
| `+0x184`, `+0x188`, `+0x18c` | `0x20000`, `0x30000`, `0x8000` | descriptor counts behind parts 1 and 2 |

Sizes are rounded up to 64 KiB. Two later callbacks get a chance to edit the block (`0x142241f10`, then `0x145955790`, which is empty); neither touches these fields, and I found no other store into them.
## 3. Part A: a ring of 128 KiB pages written by the CPU

- Right after the views are created, `0x14247ac20` is called (from `0x14247930d`) with the base of part A, 216 and `0x20000`. It builds a pool of 216 page nodes (`0x58` bytes each, array at `*0x1475df1c0`, free list at `0x1475df1d8`) and stores the base in `0x1475df1b8`.
- Each render context allocates from these pages with `0x144d9f3e0` (or `0x144e32420` for the second context type), 16- or 256-byte aligned.
- Callers such as `0x1400378c0`, `0x14004c100`, `0x14248d6d0`, `0x14248d780` and `0x1404f4210` then **copy CPU data into the allocation** and bind it on the context as an offset into the resource (`pointer − *0x1475df1b8`). `0x14248cbe0` issues an indirect call whose arguments live in the same range.

Consistent public data point: in upstream issue 59 a 112-byte SRV (stride 56) sits at offset `0x19ec730` of this buffer, which is inside part A.

## 4. Parts B and C: per-frame data in 48-byte elements

- `0x145eb2480` copies the bases and sizes of B and C to `0x1477aeb08` / `0x1477aeb18` (B) and `0x1477aeb10` / `0x1477aeb20` (C).
- Their only reader is `0x140cc8170`. Each frame it lays out several sub-ranges in both parts from budgets given in KiB, sums counts × `0x30` bytes, and traps (`int3`) if a layout exceeds the part's size.
- That function also selects the title's billboard and impostor draw programs. This suggests instance data for billboards/impostors, but I have **not** confirmed what the elements contain.

## 5. Views over the whole buffer

- `0x145bdb830` creates two views of the whole resource: an SRV stored at `0x1475df128` (device vtable `+0x90`) and a UAV stored at `0x1475df130` (device vtable `+0x98`, counter argument 0).
- `0x1450cc5b0` binds both on a context. The slots are looked up by hashed shader-variable name in `0x140cc8170`: `0x7b09bd38` (read) and `0x95557b50` (write). I could not recover the names; they are not in the image.
- `0x145eb2480` also stores the SRV's index in the title's descriptor table at `0x148d7ab7c` (index = (descriptor − `*0x14a943aa0`) >> 5). `0x14325c500` passes that index to draw shaders as root constants, together with the indices of the two ped-skinning output buffers.

## 6. Streaming: from a request to a loaded resource

Same caveat as above: static reading, not verified at run time.

### 6.1 Asset stores and the global resource index

- Each asset store object has: `+0x08` u32 base index, `+0x0c` u32 module id, `+0x18` name, and a slot pool at `+0x38` (`+0x38` storage, `+0x40` u32 slot size, `+0x44` u32 slot count, `+0x60` occupancy bitmap; free slots are chained through their first u32, `-2` ends the chain). The first u64 of a used slot points to the loaded object.
- The four large stores: texture dictionaries `0x14ab42e88`, drawables `0x14a92e7d0`, drawable dictionaries `0x14a92e700`, fragments `0x14a92e8a0`. About 28 more are built by one shared constructor, `0x1419ca0b0`.
- A resource's **global index** is `store base + slot`. The module registry at `*(u64*)0x14abcd200` (16-byte entries: module pointer, u32 exclusive end index; count u16 at `0x14abcd208`) maps a global index back to its store.

### 6.2 Per-resource state

| Global | Meaning |
|---|---|
| `*(u64*)0x14abcc5a0` | **loaded bitmap**, one bit per global index. Set in `0x14258e410` (resource placed), cleared in `0x140810bb0` (resource removed); query `0x144e685d0` |
| `*(u64*)0x14abcc598` | 4 bytes per resource: its archive file entry (`0xffffffff` = none). Not a request handle |
| `u32 0x14abcc594` | number of resources |

### 6.3 The streaming engine

The engine object is at `*(u64*)0x14764b240`; its per-frame update is `0x1409d9030`.

- **Requests from the game** go into 503 buckets of `0xa0` bytes starting at engine `+0x20`, chosen by `global index % 503`. Each bucket holds `0x1c`-byte request records (double-buffered by frame parity) and a "requested" bitmap at bucket `+0x48` (query `0x144e51f40`).
- A record's flags (u32 at `+0x08`) keep its **current step** in the low 5 bits. `0x142593560` is the state machine; it moves a record through steps 1 to 14. Step 8 issues the disk read (`0x142a8ca50`).
- Worker threads walk the buckets each frame (`0x1409d93f0` → `0x14258b150`) and turn records into **request objects** from a pool: `+0x00` global index, `+0x78` object pointer, `+0x90` sort key, `+0xa0` state.
- Request-object states seen: `1` issued, `9` deferred, `10` needs a disk read (queued at engine `+0x167a0`), `4` done (the loaded bit is set at this point), `2` failed. Finished requests go to engine `+0x167a8`. Per-step queues are at engine `+0x16718 + i*8` and `+0x167b0 + i*8`.
- The title's own count of streaming requests is the u32 at `0x147bd24f8`.

### 6.4 Disk reads in flight

The reader object is at `*(u64*)0x147bc7f88`:

- `0x800` read slots of `0x138` bytes: `+0x00` archive object, `+0x08` u32 file handle, `+0x28` file offset, `+0x30` chunk list (16 bytes each), `+0x130` u8 chunk count.
- Two lock-free rings: free slots (`+0x9c000` capacity, `+0x9c004` / `+0x9c008` head and tail) and submitted reads (`+0x9c028` capacity, `+0x9c02c` / `+0x9c030`).

A slot taken from the free ring and not yet returned is a read in flight.

### 6.5 The streaming address range

`*(u64*)0x149e1dac0` base, `*(u64*)0x149e1dac8` size, `*(u64*)0x149e1dad0` GPU-side base. Tile heap counters: used `u64 0x1498d26d8`, total `u64 0x149c3e060`.

## 7. Ped skinning buffers

- Set up in `0x1427c0400`, for the compute program that skins peds. Two output buffers with atomic cursors:

| Buffer | Capacity | Capacity field | Cursor |
|---|---|---|---|
| A | 25 MiB (`0x1900000`) | `u32 0x148562650` | `u32 0x148562654` |
| B | 50 MiB (`0x3200000`) | `u32 0x148562678` | `u32 0x14856267c` |

- `0x1427c2090` reserves space: sizes are rounded to 16 bytes, the cursor is advanced with an atomic add and compared to the capacity **afterwards**, so a cursor can exceed its capacity (read `min(cursor, capacity)`). The result is `offset >> 4`, with bit 24 set for buffer B.
- `0x1408dfb70` resets both cursors to 0; it is registered as a callback and looks like a once-per-frame reset.
- The buffers are bound as UAV by `0x140c5da40` and as SRV by `0x140d756e0`. I found no CPU read or write of their contents.
- `0x1427c15a0` builds one `0x90`-byte job record per mesh: `+0x44` vertex count, `+0x48` output offset and flags, and `+0x60` / `+0x64` / `+0x68` indices into the title's descriptor table (base `*(u64*)0x14a943aa0`, 32-byte entries). `+0x60` is the mesh's vertex buffer; `+0x64` is a per-entity buffer (probably bone matrices, **unconfirmed**); `+0x68` is optional (blend shapes). Queues are at `0x1478a9a50 + list*0x68 + queue*8`.

## What is not established

- Which pass writes through the UAV, and into which part. I saw the UAV created and bound to a write slot, not a dispatch.
- Whether the CPU also writes into parts B and C.
- What the 48-byte elements are.
- That the per-entity buffer at job `+0x64` holds bone matrices.
- Whether the title changes a loaded resource's memory later without clearing its loaded bit (it matters if the bit is ever used to skip checks).
- The exact meaning of each per-step queue in the streaming engine.
- Everything above at run time. See the checks below.

## Checks for someone with the title (read-only)

Reading process memory only; no title function needs to be called.

1. `*(u64*)0x1475df148 − *(u64*)0x1475df138` should be `0x1b00000`, and `*(u64*)0x1475df150 − *(u64*)0x1475df148` should be `0x2f00000`.
2. `u64` at `0x1475df168`, `0x1475df170`, `0x1475df178` should be `0x1b00000`, `0x2f00000`, `0x6400000`.
3. The `X` in the layer's `title buffer X+0xae00000` should equal `*(u64*)0x1475df138`.
4. Offsets of vertex/constant bindings inside this buffer should fall below `0x1b00000`.
5. `u32 0x147bd24f8` should be above 0 while moving through the world and near 0 when standing still.
6. The number of set bits in the bitmap at `*(u64*)0x14abcc5a0` should rise when a new area loads and fall when leaving it.
7. In the reader at `*(u64*)0x147bc7f88`, the distance between the free ring's head and tail (`+0x9c004`, `+0x9c008`) should track the number of reads in flight.
8. The skinning cursors (`u32 0x148562654`, `u32 0x14856267c`) should grow during a frame, stay far below 25 / 50 MiB, and return to 0 each frame.

If any of these fail, the corresponding section is wrong and I will correct it.

## Why this may be useful

The layer currently treats this buffer as one opaque 174 MiB range. If the layout holds, it tells the layer which part is rewritten by the CPU every frame (A, and only the pages handed out by the pool), and which parts are per-frame draw data (B, C). The streaming state (sections 6.2–6.4) would let the layer know which resources are requested, being read and loaded without guessing from page writes, and the skinning cursors give the exact number of bytes used per frame in buffers that only the GPU touches. I am not proposing a change here; I can follow up with a read-only, off-by-default diagnostic that logs the values above if that is wanted.
