use super::Batch;
use std::sync::Arc;

#[test]
fn a_texture_is_checked_once_per_submission() {
    let mut batch = Batch::default();
    let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
    // The first list sampling each texture checks it ...
    assert!(batch.texture_due(&a));
    assert!(batch.texture_due(&b));
    // ... later lists of the same submission do not.
    assert!(!batch.texture_due(&a));
    assert!(!batch.texture_due(&a.clone()));
    assert!(!batch.texture_due(&b));
}

#[test]
fn title_writes_make_textures_due_again() {
    let mut batch = Batch::default();
    let a = Arc::new(1u32);
    assert!(batch.texture_due(&a));
    assert!(!batch.texture_due(&a));
    // A copy into title memory or work that may write it may have
    // changed the texture's memory.
    batch.title_written();
    assert!(batch.texture_due(&a));
    assert!(!batch.texture_due(&a));
    // A fresh submission starts over.
    let mut next = Batch::default();
    assert!(next.texture_due(&a));
}

/// The hashing a submission does for textures in physical-page memory,
/// with every list checking the textures it samples (before) and with
/// each checked once (after). Run with
/// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
#[test]
#[ignore = "benchmark"]
fn texture_checks_per_submission() {
    const TEXTURES: usize = 48;
    const SIZE: usize = 4 << 20;
    const LISTS: usize = 12;
    let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
        .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
        .collect();
    let submit = |dedupe: bool| {
        let mut batch = Batch::default();
        let mut hashed = 0usize;
        let started = std::time::Instant::now();
        for _ in 0..LISTS {
            for t in &textures {
                if !dedupe || batch.texture_due(t) {
                    std::hint::black_box(blake3::hash(t));
                    hashed += 1;
                }
            }
        }
        (started.elapsed(), hashed)
    };
    submit(false); // warm up
    let (before, n_before) = submit(false);
    let (after, n_after) = submit(true);
    assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
    println!(
        "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
        SIZE >> 20,
        before.as_secs_f64() / after.as_secs_f64()
    );
}

#[test]
fn vram_idle_ms_scales_with_pressure() {
    use super::vram_idle_ms;
    let gib = 1024 * 1024 * 1024;
    // Normal (< 75%)
    assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
    assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
    assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
    // Moderate (75%..85%)
    assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
    assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
    // High (>= 85%, including over-budget)
    assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
    assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
    // Missing or zero budget falls back to normal
    assert_eq!(vram_idle_ms(None), 5_000);
    assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
}

#[test]
fn ram_idle_ms_scales_with_pressure() {
    use super::ram_idle_ms;
    let gib = 1024 * 1024 * 1024;
    // Normal (>= 4 GiB)
    assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
    assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
    // Moderate (2 GiB..4 GiB)
    assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
    assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
    // High (< 2 GiB)
    assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
    assert_eq!(ram_idle_ms(Some(0)), 2_000);
    // Missing falls back to normal
    assert_eq!(ram_idle_ms(None), 10_000);
}
