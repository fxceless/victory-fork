# Proton testing

The Linux measurements used GE-Proton11-13, an RTX 5060 Ti, and NVIDIA driver 595.99.02. They cover a short run from a prison checkpoint with camera control. Interior lighting and some geometry remain incorrect.

## Wine memory mapping

Streaming repeatedly maps and unmaps 64 KiB pages while many mapped views are live. The Wine server scans its per-process list for address lookups and overlap checks. [The external Wine patch](wineserver-view-index.patch.txt) adds a per-process red-black tree keyed by base address. It retains the existing list for ordered iteration and updates both structures on insertion and removal. An overlap check needs only the last view that begins before the end of the proposed range because the existing views cannot overlap.

Apply the patch in Wine source prepared for the Proton client you use. For this test, the Valve Wine base was `a47c73f52fd7b2efafc1c560710f3a5099ae51e0`, with the GE-Proton11-13 Wine preparation and staging snapshot `6cc805ea57132eeaf44764e9213823c9b8d0d300` applied. Keep the installed Proton files intact and use a separate test prefix. The patched server and client must use the same server protocol. This test used protocol 938.

From the prepared Wine source directory, with `VICTORY_SOURCE` pointing to this repository, build the server in a separate directory.

```sh
git apply --check "$VICTORY_SOURCE/docs/proton/wineserver-view-index.patch.txt"
git apply "$VICTORY_SOURCE/docs/proton/wineserver-view-index.patch.txt"
mkdir -p ../wine-view-index-build
cd ../wine-view-index-build
../wine/configure --enable-win64 --without-x --without-wayland --without-vulkan --disable-tests
make -j "$(nproc)" server/wineserver
```

The build directory in this example assumes the prepared source directory is named `wine`. The configure flags apply to the server-only build. The client still uses the GE-Proton libraries and the NVIDIA driver.

For the test, set `PROTON_FILES` to the GE-Proton `files` directory, `WINEPREFIX` to the separate prefix, and `TEST_RUNTIME` to a writable directory for the local server. Copy the server and the client's locale file into the local runtime. Start that server explicitly before launching Wine, since the Proton client can otherwise pick up its bundled server. Check the running server's executable path under `/proc` before attributing any result to the patch.

```sh
install -D -m755 server/wineserver "$TEST_RUNTIME/server/wineserver"
install -D -m644 "$PROTON_FILES/share/wine/nls/l_intl.nls" "$TEST_RUNTIME/nls/l_intl.nls"
export WINEPREFIX WINESERVER="$TEST_RUNTIME/server/wineserver"
"$WINESERVER" -p
```

The [synthetic benchmark](map-view-bench.c.txt) times 3,000 random unmap/remap pairs while holding a configurable number of views live. Compile it with MinGW and run it with the same Wine client and test prefix for both servers. Stop the first server for that prefix before starting the other one.

```sh
x86_64-w64-mingw32-gcc -O2 -x c "$VICTORY_SOURCE/docs/proton/map-view-bench.c.txt" -o map-view-bench.exe
"$PROTON_FILES/bin/wine" ./map-view-bench.exe 65536
```

| Live views | Bundled server | Indexed server |
| ---: | ---: | ---: |
| 1,024 | 56.0 µs | 46.9 µs |
| 32,768 | 204.0 µs | 38.8 µs |
| 65,536 | 371.0 µs | 47.8 µs |

Numbers are average time per map/unmap pair. The patch is distributed under LGPL-2.1-or-later as part of Wine. The [Wine license](LICENSE) accompanies the patch.

## Copy-back and rendering checks

With the indexed Wine server, `VICTORY_BACK_OWN_VIEW=1` uses the existing direct-view copy-back path. The runtime can still fall back to the regular path for pages it cannot write this way. `VICTORY_BACK_OWN_VIEW=alt` alternates every 20 seconds for measurements within a scene. `VICTORY_BACK_OWN_CHECK=1` adds a read-back check for diagnosis.

In the same prison scene, the median copy-back time fell from 279 to 232 ms across 9 regular and 10 direct-view windows with at least 2,500 draws and at most 10 pending pipelines. Median frame rate rose from 2.7 to 3.15 FPS. The separate verification run logged 131,072 checked writes, zero missing stores, and zero inconsistencies on pages with clean guards.

The comparison used `VICTORY_LEVELS_ON_DEMAND=0`. With sparse mip streaming enabled in a separate run, up to 896 placed texture slots sampled sparse textures with no resident levels. Loading flicker was reported during that run and did not recur in either short capture at 6 capture FPS. Sparse streaming remains disabled in the tested play configuration.

Two more short runs tested a 512 MiB budget for title buffers on the GPU and waiting for texture memory to settle before uploads. The smaller budget reduced resident title-buffer copies from 941 to 511 MiB, while peak local D3D12 usage still reached about 11.5 GiB. The movement prompt stayed readable after appearing with either texture upload setting. The runs retain the original buffer budget and the no-wait texture setting for play.

The shader suite passed 174 tests with two ignored. The Proton GPU tests passed three residency cases and one indirect compute pipeline case, with the optional tiled-resource hardware test ignored. No NVIDIA kernel fault was recorded in these bounded tests.
