//! The title's own programs, from container files the user extracted
//! (nothing of the title is in the repository): `VICTORY_TEST_CONTAINERS`
//! lists files (separated by `;`; or `@<file>` naming a file that lists
//! them a line each), each translated as on a host of
//! 32-lane waves and validated by `dxil.dll` when present; a program that
//! does not translate or validate fails the test with the reason. Ignored unless asked for:
//! `cargo test -p victory-shader title_containers -- --ignored --nocapture`.

#[test]
#[ignore = "needs VICTORY_TEST_CONTAINERS"]
fn title_containers_translate_for_32_lane_waves() {
    let Ok(list) = std::env::var("VICTORY_TEST_CONTAINERS") else {
        eprintln!("VICTORY_TEST_CONTAINERS unset; nothing to do");
        return;
    };
    super::set_host_max_wave(32);
    // `@file`: the list is in that file, a path per line.
    let list = match list.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file.trim()).unwrap_or_else(|e| panic!("{file}: {e}")),
        None => list,
    };
    let mut failed = Vec::new();
    for path in list.split([';', '\n']).map(str::trim).filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        match super::translate(&bytes) {
            Ok(shader) => {
                let checked = match super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
                    Ok(mut v) => {
                        let mut copy = shader.dxil.clone();
                        match v.validate(&mut copy) {
                            Ok(_) => "validated",
                            Err(e) => {
                                eprintln!("{path}: DXIL validation failed: {e}");
                                failed.push(path.to_owned());
                                continue;
                            }
                        }
                    }
                    Err(_) => "no validator",
                };
                eprintln!("{path}: translates ({} bytes of DXIL, {checked})", shader.dxil.len());
            }
            Err(e) => {
                eprintln!("{path}: {e}");
                failed.push(path.to_owned());
            }
        }
    }
    assert!(failed.is_empty(), "do not translate: {failed:?}");
}

/// The whole corpus, counted instead of asserted: each container of
/// `VICTORY_TEST_CONTAINERS` (as above) is translated as on the host the
/// title runs on (32-lane waves; `VICTORY_VIEW_LIMIT`, the null slot of a
/// capped view heap, when set), on `VICTORY_TEST_JOBS` threads (default
/// 2), and `VICTORY_TEST_PIPELINES=1` also creates a compute program's
/// host pipeline on the card (`VICTORY_TEST_HASHES=1`: see below). One line per container (`name: translates`,
/// `name: pipeline: <error>` or `name: <error>`) goes to
/// `VICTORY_TEST_STATUS` when set; the counts by reason are printed.
/// `cargo test -p victory-shader title_corpus -- --ignored --nocapture`.
#[test]
#[ignore = "needs VICTORY_TEST_CONTAINERS"]
fn title_corpus_counts() {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let Ok(list) = std::env::var("VICTORY_TEST_CONTAINERS") else {
        eprintln!("VICTORY_TEST_CONTAINERS unset; nothing to do");
        return;
    };
    super::set_host_max_wave(32);
    if let Some(n) = std::env::var("VICTORY_VIEW_LIMIT").ok().and_then(|v| v.parse().ok()) {
        super::set_view_heap_limit(n);
    }
    let list = match list.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file.trim()).unwrap_or_else(|e| panic!("{file}: {e}")),
        None => list,
    };
    let paths: Vec<&str> = list.split([';', '\n']).map(str::trim).filter(|p| !p.is_empty()).collect();
    let jobs = std::env::var("VICTORY_TEST_JOBS").ok().and_then(|v| v.parse().ok()).unwrap_or(2usize).max(1);
    let pipelines = std::env::var("VICTORY_TEST_PIPELINES").is_ok_and(|v| v == "1");
    // `VICTORY_TEST_HASHES=1`: a hash of the DXIL after "translates", to tell two runs'
    // translations apart (it is left out of the counts)
    let hashes = std::env::var("VICTORY_TEST_HASHES").is_ok_and(|v| v == "1");
    let next = AtomicUsize::new(0);
    let lines = Mutex::new(Vec::with_capacity(paths.len()));
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                let host = pipelines.then(|| super::host::Host::new(false).expect("a card for VICTORY_TEST_PIPELINES"));
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(i) else { break };
                    let name = std::path::Path::new(path).file_name().map_or(*path, |n| n.to_str().unwrap_or(path));
                    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
                    let outcome = match std::panic::catch_unwind(|| super::translate(&bytes)) {
                        Ok(Ok(shader)) => {
                            let ok = if hashes {
                                use std::hash::{Hash, Hasher};
                                let mut h = std::collections::hash_map::DefaultHasher::new();
                                shader.dxil.hash(&mut h);
                                format!("translates {:016x}", h.finish())
                            } else {
                                "translates".to_owned()
                            };
                            match &host {
                                Some(h) if shader.stage == crate::HwStage::Cs => match h.compute_pipeline(&shader) {
                                    Ok(_) => ok,
                                    Err(e) => format!("pipeline: {e}"),
                                },
                                _ => ok,
                            }
                        }
                        Ok(Err(e)) => e.to_string(),
                        Err(_) => "panics".to_owned(),
                    };
                    lines.lock().unwrap().push(format!("{name}: {outcome}"));
                }
            });
        }
    });
    let mut lines = lines.into_inner().unwrap();
    lines.sort();
    let mut reasons = std::collections::BTreeMap::<String, usize>::new();
    for line in &lines {
        let (_, outcome) = line.split_once(": ").unwrap_or(("", line));
        // The reason without the code offset.
        let reason = match outcome.split_once(": ") {
            Some((head, rest)) if head.starts_with("GFX10 code at") => rest.to_owned(),
            _ if outcome.starts_with("translates") => "translates".to_owned(),
            _ => outcome.to_owned(),
        };
        *reasons.entry(reason).or_default() += 1;
    }
    if let Ok(out) = std::env::var("VICTORY_TEST_STATUS") {
        std::fs::write(&out, lines.join("\n") + "\n").unwrap_or_else(|e| panic!("{out}: {e}"));
    }
    let mut by_count: Vec<_> = reasons.into_iter().collect();
    by_count.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("{} containers", lines.len());
    for (reason, n) in by_count {
        eprintln!("{n:7} {reason}");
    }
}

/// A mesh (or vertex) and pixel program of the title made into one host pipeline:
/// `VICTORY_TEST_PAIR=<first container>;<pixel container>`; the error (with the debug layer's
/// messages under `VICTORY_D3D12_DEBUG`) is printed.
/// `cargo test -p victory-shader title_pair -- --ignored --nocapture`.
#[test]
#[ignore = "needs VICTORY_TEST_PAIR"]
fn title_pair_makes_a_pipeline() {
    let Ok(pair) = std::env::var("VICTORY_TEST_PAIR") else {
        eprintln!("VICTORY_TEST_PAIR unset; nothing to do");
        return;
    };
    super::set_host_max_wave(32);
    let [first, ps] = pair.split(';').map(str::trim).collect::<Vec<_>>()[..] else {
        panic!("VICTORY_TEST_PAIR: two paths separated by ';'");
    };
    let shader = |path: &str| super::translate(&std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}")))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let (first, ps) = (shader(first), shader(ps));
    // `VICTORY_TEST_DUMP=<dir>`: both translations as `first.dxil` and `pixel.dxil` there.
    if let Ok(dir) = std::env::var("VICTORY_TEST_DUMP") {
        for (name, s) in [("first", &first), ("pixel", &ps)] {
            let path = std::path::Path::new(&dir).join(format!("{name}.dxil"));
            std::fs::write(&path, &s.dxil).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        }
    }
    let host = super::host::Host::new(false).expect("a card");
    let made = match first.stage {
        crate::HwStage::Ms => host.mesh_pipeline(&first, &ps).map(drop),
        _ => host.pipeline(&first, &ps).map(drop),
    };
    match made {
        Ok(()) => eprintln!("pipeline made"),
        Err(e) => panic!("pipeline refused: {e}"),
    }
}

/// The GFX10 listing of each container of `VICTORY_TEST_DISASM` (as `VICTORY_TEST_CONTAINERS`),
/// one instruction a line as `victory shader --disasm` prints it.
/// `cargo test -p victory-shader title_disasm -- --ignored --nocapture`.
#[test]
#[ignore = "needs VICTORY_TEST_DISASM"]
fn title_disasm() {
    let Ok(list) = std::env::var("VICTORY_TEST_DISASM") else {
        eprintln!("VICTORY_TEST_DISASM unset; nothing to do");
        return;
    };
    for path in list.split([';', '\n']).map(str::trim).filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let c = crate::dxbc::Container::parse(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
        let dxil = c.part(crate::dxbc::FourCC::DXIL).unwrap_or_else(|| panic!("{path}: no DXIL part"));
        for n in crate::native_programs(dxil.data).unwrap_or_else(|e| panic!("{path}: {e}")) {
            println!("{path}: {:?} {} B", n.stage, n.code.len());
            for i in crate::gfx10::decode(&n.code).unwrap_or_else(|e| panic!("{path}: {e}")) {
                let words: Vec<_> = i.words.iter().map(|w| format!("{w:08X}")).collect();
                let text = match i.name() {
                    Some(_) => i.text(n.wave64),
                    None => format!("<{:?} {:#x}>", i.encoding, i.opcode),
                };
                println!("    {:06x}: {:<44} {text}", i.offset, words.join(" "));
            }
        }
    }
}

/// Each container of `VICTORY_TEST_CONTAINERS` (as above) translated as on a host of 32-lane
/// waves, its DXIL written to `VICTORY_TEST_DUMP_DIR/<file stem>.dxil` (for `dxc -dumpbin`).
/// `cargo test -p victory-shader title_dump -- --ignored --nocapture`.
#[test]
#[ignore = "needs VICTORY_TEST_CONTAINERS and VICTORY_TEST_DUMP_DIR"]
fn title_dump() {
    let (Ok(list), Ok(dir)) = (std::env::var("VICTORY_TEST_CONTAINERS"), std::env::var("VICTORY_TEST_DUMP_DIR")) else {
        eprintln!("VICTORY_TEST_CONTAINERS or VICTORY_TEST_DUMP_DIR unset; nothing to do");
        return;
    };
    super::set_host_max_wave(32);
    let list = match list.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file.trim()).unwrap_or_else(|e| panic!("{file}: {e}")),
        None => list,
    };
    for path in list.split([';', '\n']).map(str::trim).filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let stem = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        match super::translate(&bytes) {
            Ok(shader) => {
                let out = std::path::Path::new(&dir).join(format!("{stem}.dxil"));
                std::fs::write(&out, &shader.dxil).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
                eprintln!("{path}: {} bytes of DXIL", shader.dxil.len());
            }
            Err(e) => eprintln!("{path}: {e}"),
        }
    }
}
