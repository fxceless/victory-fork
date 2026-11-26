# Incremental descriptor buffer unions (Kelvaroq 0055 subset)

Source concept: https://github.com/stellarr-studio/kelvaroq/blob/1270c422adaef1e0f369f670138d8d29d8f6f4c3/native/patches/0055-descriptors.patch (GPL-3.0; Kelvaroq contributors). AI-assisted adaptation.

Set VICTORY_DESCRIPTOR_BUFFER_SPANS=1 in a fresh process. In the existing unmarked buffer pass, retain each live backing buffer's minimum start and maximum end in a per-heap index, updated only for the descriptors revalidated by the existing scan. Preserve its authoritative slot map, full/partitioned revalidation, heap identity, dirty drain, refresh cadence and enclosing range semantics. On index mismatch rebuild from the authoritative map; full scans and epoch resets invalidate the index. The marked pass is unchanged and remains independently selectable.

This does not skip unwatched bytes, relax dirty tracking or modify descriptor table cache keys. It reduces span aggregation from a walk of all live slots to a walk of distinct copies on unchanged passes, at the cost of ordered sets and additional retained metadata. Both modes issue the same enclosing ranges. Tests cover duplicate endpoints, shrinking, moving, deleting and 2048 mixed updates against a full-scan oracle. Local builds deferred during active benchmark; CI and live A/B required. No FPS claim.

Upstream adaptation: the index is part of the stable heap-ID cache introduced by #1372. It resets on full scans, marked-pass resets and heap destruction; there is no partition-spread path in this baseline. CI on the independent port is pending.
