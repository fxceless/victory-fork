# Exact conventional table CBV upload reuse

VICTORY_CBV_REUSE_UPLOAD=1 reuses exact source/size uploads only within
one HostList recording, for non-null addresses and sizes that are nonzero,
256-byte multiples and at most 64 KiB. Only ordinary Fill steps may
intervene; every other recorded step invalidates the cache. It does not
reuse card fills, change descriptor bounds or bind physical memory.
The original scratch path remains in use for all other requests.
This conservative independent port does not require the fork's UMA stack.
No gameplay speedup is claimed.
