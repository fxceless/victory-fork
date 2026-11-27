# Comparing the skin-colour draw in two runs

Use the same binary, save, Series S profile, input and graphics settings for both runs. In the scene from issue #923, the first of three 1280×720 RGBA8 targets has a black neck in the bad frame and a coloured neck in the good frame. The neighbouring NPC and the normals target help locate the same shot. [RTX 5070 Ti report](https://github.com/fxceless/victory-fork/issues/923#issuecomment-7163671441)

## Select the draw

Build on Windows from the patched checkout.

```powershell
cargo build --locked --profile diag -p victory --features diagnostics
```

Locate the material pass using the existing frame/target dump and the operation listing. Set `VICTORY_CAPTURE` to a capture directory, `VICTORY_CAPTURE_FRAMES` to a frame visible in the scene and `VICTORY_DUMP_TARGETS=1`. In PIX, use pixel history on a neck pixel of the colour target. Identify the draw that writes that pixel, its pixel shader's 16-hex hash in the VIctory listing, and its 1-based occurrence among emitted graphics commands with that pixel shader in the frame. A pipeline can draw both the heroine and an NPC. The traced command gets a marker beginning with `VIctory traced draw`, including in `frame-N-ops.txt` when the frame is listed. Confirm the target pixel for each run. Lists can be recorded on several threads, so the ordinal or frame number may move. An indirect command can cover more than one draw record.

Enter the frame, hash and ordinal measured for the target scene. Use a fresh capture directory for each launch.

```powershell
$frame = Read-Host 'Frame number'
$pixel = Read-Host '16-hex pixel shader hash'
$occurrence = Read-Host 'Occurrence of this pixel shader in the frame (starts at 1)'
$env:VICTORY_CAPTURE = Join-Path $PWD (Read-Host 'New capture directory')
New-Item -ItemType Directory -Force $env:VICTORY_CAPTURE | Out-Null
$env:VICTORY_CAPTURE_FRAMES = "$frame"
$env:VICTORY_DUMP_TARGETS = '1'
$env:VICTORY_TRACE_DRAW = '{0}:{1}:{2}' -f $frame,$pixel,$occurrence
$game = Read-Host 'Full path to the title executable'
.\target\diag\victory.exe run $game --execute --console series-s
```

Keep the normal and black captures, their PNGs, and the TSV files together in their respective directories. Each execution of the selected list produces a separate `draw-*-f*-e*.tsv`. If the file is absent, check the tracer warnings, frame, pixel hash and occurrence, and whether the list was submitted. Choose the matching execution if the title replays a list.

## Compare

Pass the two draw files explicitly, good first. The Python 3 script finds each run's fence observations and later uploads alongside its draw file.

```powershell
$good = Read-Host 'Path to the good draw TSV'
$bad = Read-Host 'Path to the black draw TSV'
python scripts/compare-draw-trace.py "$good" "$bad"
```

Use `--role <exact role>` to inspect one binding even when its observations match. Static table roles include the stage hash, root parameter and dword offset. Computable root-indexed accesses also identify the constant that selects the view. The trace records the actual host slot after dynamic table copies. The detail lines include root-buffer bindings, samplers and the original material listing.

`word/M/0x…` and the other `word/` roles describe candidate heap indices in CPU-visible material records. The recorder rereads these words at queue submission, including the offset of the material record. A word can coincide with a valid slot by chance, and GPU-produced buffer contents can differ from their CPU copy. Use the matching PIX draw and shader access to confirm which candidate contributes the colour. Report the role, good/bad slot, resource description, upload sample and fence for the verified read.

## Read the timing

- `record_us` and `execute_us` are process-relative CPU times near recording and immediately before the native queue call. Object pointers and host indices are useful within one run. Compare roles, texture descriptions and sampled texels across runs.
- `upload_fence` and `upload_order` identify the last staged texture copy preceding the draw in queue order. Its 64-bit fingerprint covers samples from three unpadded rows per subresource in the upload buffer after deferred fills. Different hashes establish that the sampled upload bytes or footprint differed. Matching samples cover only those bytes. GPU rendering, alias copies and tile residency can change what a shader reads later.
- `completed_before` comes from the graphics fence immediately before submission. The separate observer polls that fence every 4 ms. The first observed value reaching a copy's fence gives an upper bound on GPU completion. A copy can be ordered before the draw in the same submission or still in flight when a later submission is enqueued. Both are valid queue ordering.
- The first later staged upload into each retained texture is recorded when it occurs. A missing upload record means the tracer has no observed copy for that point. A slot labelled `no-retained` has no resource in the host descriptor mirror. A missing texture, a scratch CBV and a diagnostic null view require different interpretations.
- The CPU observes descriptors at recording and at submission. PIX can confirm the view and mip sampled on the GPU. Capture work and fingerprinting can affect the timing of an intermittent bug.
