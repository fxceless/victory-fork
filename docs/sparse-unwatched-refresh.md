# Sparse refresh for unwatched buffers (opt-in)

The existing submission batch unions all requested ranges per shadow. For an
unwatched shadow this can copy a large unrequested gap. With
VICTORY_SPARSE_REFRESH=1 a known-unwatched shadow retains up to eight sorted
intervals; overlapping ranges and gaps at most 64 KiB are joined. A ninth
disjoint interval collapses to the original bounding union. Unknown or watched
shadows keep the original union. The option defaults off.

The queue still schedules one job per shadow. Flush-one and full flush update
every retained span, and near-write partitioning flushes a group if any of its
spans intersect a recorded write. Current-range tracking records only refreshed
spans, not skipped gaps. Shadow update retains its existing pending-write
exclusion. The sparse refresh log reports logical omitted gap bytes, not actual
PCIe transfers or a measured frame-time improvement.

Validation: the source fork PR #38 passed workspace checks, the range oracle and explicit D3D12 tests. This independent upstream port retains the CPU range regression. The fork-specific staging test requires adaptation to upstream APIs; fresh compilation and hardware validation are pending. No gameplay improvement claimed.

AI-assisted implementation.
