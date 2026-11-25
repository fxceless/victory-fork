# Synchronous texture upload bands (experimental)

Based on [fxceless/victory-fork #80](https://github.com/fxceless/victory-fork/pull/80)
at `55d7fd3`, which records texture uploads after the submission-time refresh.
This branch is a pilot, not an upstream merge or a game performance claim.

Set `VICTORY_TEXTURE_UPLOAD_BANDS=1` to try a full-width boxed GPU copy after a
sparse write to an already-uploaded, fully mapped, write-watched 2D texture.
The band is rounded out to console tile rows and block-compression boundaries.
First, dense, untracked, unsupported, and on-demand updates still copy whole
subresources. Multiple command lists and submissions sent ahead of the CPU
also stay on the original full-copy path. The full CPU detile and staging
allocation are unchanged.

Run the synthetic read-back test on Windows from PowerShell:

```powershell
$env:VICTORY_TEXTURE_UPLOAD_BANDS = '1'
Remove-Item Env:VICTORY_D3D12_WARP -ErrorAction SilentlyContinue
cargo test -p victory-gpu --release --locked --features diagnostics boxed_upload_uses_only_a_watched_band -- --ignored --nocapture
```

The test checks all GPU-read bytes for linear RGBA8, tiled RGBA8 and tiled
BC3, including the initial upload, sparse update, two-list fallback and
dense update. It passed on an RTX 2070 Super and on WARP. No gameplay,
asynchronous overlap, or FPS claims are made.
