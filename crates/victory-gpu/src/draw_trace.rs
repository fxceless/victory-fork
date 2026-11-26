//! Per-draw descriptor and staged-upload diagnostics for the host graphics queue.

use crate::pipeline::host::{HostPipeline, Stage};
use crate::{capture, host::HostList, texture::Staging};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows::Win32::Graphics::Direct3D12::{ID3D12Fence, ID3D12Resource};
use windows::core::{GUID, Interface};

struct Selector {
    frame: u64,
    hash: String,
    occurrence: usize,
    folder: PathBuf,
}

impl Selector {
    fn parse(raw: &str) -> Option<Self> {
        let mut parts = raw.trim().split(':');
        let frame = parts.next()?.parse::<u64>().ok()?;
        let hash = parts.next()?.to_ascii_lowercase();
        let occurrence = parts.next()?.parse::<usize>().ok()?;
        if parts.next().is_some()
            || frame == 0
            || occurrence == 0
            || hash.len() != 16
            || !hash.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return None;
        }
        Some(Self {
            frame,
            hash,
            occurrence,
            folder: capture::folder()?.clone(),
        })
    }
}

fn selector() -> Option<&'static Selector> {
    static SELECT: OnceLock<Option<Selector>> = OnceLock::new();
    SELECT.get_or_init(|| {
        let raw = std::env::var("VICTORY_TRACE_DRAW").ok()?;
        let config = Selector::parse(&raw);
        if config.is_none() {
            tracing::warn!("VICTORY_TRACE_DRAW needs frame:16-hex-pixel-hash:occurrence (1-based) and VICTORY_CAPTURE");
        }
        config
    }).as_ref()
}

pub(crate) fn enabled() -> bool {
    selector().is_some()
}

pub(crate) fn pixel(p: &HostPipeline) -> Option<&Stage> {
    let s = selector()?;
    if crate::queue::presented() + 1 != s.frame {
        return None;
    }
    p.stages.iter().find(|stage| {
        stage.translated().host.stage == victory_shader::HwStage::Ps && stage.shader.hash == s.hash
    })
}

pub(crate) fn take_occurrence() -> bool {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    selector().is_some_and(|s| COUNT.fetch_add(1, Ordering::Relaxed) + 1 == s.occurrence)
}

fn micros() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_micros() as u64
}

fn run_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let start = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        format!("{}-{}", std::process::id(), start.as_nanos())
    })
}

fn fence_file() -> String {
    format!("draw-fences-{}.tsv", run_id())
}

/// A separate observer never waits on the submission thread or queues a GPU command.
/// Each timestamp is an upper bound on completion of all values up to the one read.
fn observe(fence: &ID3D12Fence, folder: &std::path::Path) {
    static START: std::sync::Once = std::sync::Once::new();
    START.call_once(|| {
        let fence = fence.clone();
        let path = folder.join(fence_file());
        let run = move || -> io::Result<()> {
            std::fs::create_dir_all(path.parent().unwrap())?;
            let mut file = File::create(path)?;
            writeln!(file, "completed\tobserved_us")?;
            let mut last = None;
            loop {
                // SAFETY: the observer owns a live fence reference. MAX means removal.
                let value = unsafe { fence.GetCompletedValue() };
                if value == u64::MAX {
                    return Ok(());
                }
                if last != Some(value) {
                    writeln!(file, "{value}\t{}", micros())?;
                    last = Some(value);
                }
                std::thread::sleep(Duration::from_millis(4));
            }
        };
        if let Err(e) = std::thread::Builder::new()
            .name("draw fence trace".into())
            .spawn(move || {
                if let Err(e) = run() {
                    tracing::warn!(%e, "could not write the draw fence timeline");
                }
            })
        {
            tracing::warn!(%e, "could not start the draw fence observer");
        }
    });
}

pub(crate) type Slots = Arc<Vec<Mutex<Vec<Option<ID3D12Resource>>>>>;

#[derive(Clone)]
pub(crate) enum Event {
    Upload(ID3D12Resource, Arc<Staging>),
    Draw(Arc<Draw>),
}

/// The material index can itself change between recording and deferred fills.
pub(crate) enum WordSource {
    Fixed(u64),
    Material {
        base: u64,
        len: u64,
        index_at: u64,
        skip: u64,
    },
}

impl WordSource {
    fn read(&self, bytes: usize) -> Option<Vec<u32>> {
        let address = match *self {
            Self::Fixed(address) => address,
            Self::Material {
                base,
                len,
                index_at,
                skip,
            } => {
                let mut index = [0; 4];
                if !crate::host::read_into(index_at as *const u8, &mut index) {
                    return None;
                }
                let start = u64::from(u32::from_le_bytes(index)).checked_add(skip)?;
                if start.checked_add(bytes as u64)? > len {
                    return None;
                }
                base.checked_add(start)?
            }
        };
        let mut raw = vec![0; bytes];
        crate::host::read_into(address as *const u8, &mut raw).then(|| {
            raw.chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect()
        })
    }
}

struct Binding {
    title_slot: Option<u32>,
    host_slot: Option<u32>,
    title: String,
    word: Option<(usize, usize)>,
    recorded: Slot,
}

#[derive(Clone)]
enum Slot {
    Unresolved,
    Outside,
    NoRetained,
    Resource(ID3D12Resource),
}

impl Slot {
    fn read(slots: Option<&Slots>, index: Option<u32>) -> Self {
        let (Some(slots), Some(index)) = (slots, index) else {
            return Self::Unresolved;
        };
        let i = index as usize;
        let piece = slots[i % slots.len()].lock().unwrap_or_else(|e| e.into_inner());
        match piece.get(i / slots.len()) {
            None => Self::Outside,
            Some(None) => Self::NoRetained,
            Some(Some(r)) => Self::Resource(r.clone()),
        }
    }

    fn columns(&self) -> String {
        match self {
            Self::Unresolved => "unresolved\t-\t-".into(),
            Self::Outside => "outside\t-\t-".into(),
            Self::NoRetained => "no-retained\t-\t-".into(),
            Self::Resource(r) => {
                // SAFETY: the snapshot holds the resource.
                let d = unsafe { r.GetDesc() };
                format!(
                    "resource\t{:p}\td{}:{}x{}x{}:f{}:m{}:s{}",
                    r.as_raw(),
                    d.Dimension.0,
                    d.Width,
                    d.Height,
                    d.DepthOrArraySize,
                    d.Format.0,
                    d.MipLevels,
                    d.SampleDesc.Count
                )
            }
        }
    }
}

pub(crate) struct Draw {
    signature: String,
    pub(crate) details: String,
    slots: Option<Slots>,
    recorded_us: u64,
    bindings: BTreeMap<String, Binding>,
    blocks: Vec<(WordSource, usize)>,
}

impl Draw {
    pub(crate) fn new(signature: String, slots: Option<Slots>) -> Self {
        Self {
            signature,
            slots,
            recorded_us: micros(),
            details: String::new(),
            bindings: BTreeMap::new(),
            blocks: Vec::new(),
        }
    }

    pub(crate) fn signature(&self) -> &str {
        &self.signature
    }

    pub(crate) fn word_block(&mut self, source: WordSource, bytes: usize) -> usize {
        let block = self.blocks.len();
        self.blocks.push((source, bytes));
        block
    }

    pub(crate) fn binding(
        &mut self,
        role: String,
        title_slot: Option<u32>,
        host_slot: Option<u32>,
        title: String,
        word: Option<(usize, usize)>,
    ) {
        let recorded = Slot::read(self.slots.as_ref(), host_slot);
        self.bindings.insert(
            role,
            Binding {
                title_slot,
                host_slot,
                title,
                recorded,
                word,
            },
        );
    }

    fn snapshot(&self) -> Vec<(Option<u32>, Slot)> {
        let blocks: Vec<_> = self
            .blocks
            .iter()
            .map(|(source, bytes)| source.read(*bytes))
            .collect();
        self.bindings
            .values()
            .map(|b| {
                let index = match b.word {
                    Some((block, offset)) => {
                        blocks[block].as_ref().and_then(|w| w.get(offset)).copied()
                    }
                    None => b.host_slot,
                };
                (
                    index,
                    Slot::read(self.slots.as_ref(), index),
                )
            })
            .collect()
    }
}

// This record lives on the actual D3D resource and disappears with it. A later
// resource at the same address cannot inherit its predecessor's upload.
const UPLOAD: GUID = GUID::from_u128(0x9fbfdc49_d83f_47d7_8c40_209fbb1a9d6d);
const WATCH: GUID = GUID::from_u128(0xd861ac3d_a776_4fce_9d6e_33c17060c8bd);
static WATCHING: AtomicBool = AtomicBool::new(false);

fn private<const N: usize>(r: &ID3D12Resource, key: &GUID) -> Option<[u64; N]> {
    let mut data = [0; N];
    let mut size = size_of_val(&data) as u32;
    // SAFETY: the output has the requested size and consists solely of integers.
    unsafe { r.GetPrivateData(key, &mut size, Some(data.as_mut_ptr().cast())) }.ok()?;
    (size as usize == size_of_val(&data)).then_some(data)
}

fn store(r: &ID3D12Resource, key: &GUID, data: &[u64]) -> bool {
    // SAFETY: D3D copies these integers during the call.
    match unsafe { r.SetPrivateData(key, size_of_val(data) as u32, Some(data.as_ptr().cast())) } {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(%e, "could not attach draw trace data to the resource");
            false
        }
    }
}

fn watch(r: &ID3D12Resource) -> Option<u64> {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER;
    // SAFETY: the snapshot owns a live resource.
    if unsafe { r.GetDesc() }.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    if let Some([id]) = private(r, &WATCH) {
        return Some(id);
    }
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    WATCHING.store(true, Ordering::Relaxed);
    store(r, &WATCH, &[id]).then_some(id)
}

fn follow_file() -> String {
    format!("draw-next-{}.tsv", run_id())
}

// Record just the first later staging copy for each texture in the draw.
// The token follows the D3D object without keeping it alive.
fn follow_upload(s: &Selector, r: &ID3D12Resource, data: [u64; 4]) {
    if !WATCHING.load(Ordering::Relaxed) {
        return;
    }
    let Some([id]) = private(r, &WATCH) else {
        return;
    };
    let [fence, order, us, sample] = data;
    let write = || -> io::Result<()> {
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(s.folder.join(follow_file()))?;
        writeln!(out, "{id}\t{fence}\t{order}\t{us}\t{sample:016x}")
    };
    match write() {
        Ok(()) => {
            // SAFETY: a zero size and a null pointer remove our private token.
            let _ = unsafe { r.SetPrivateData(&WATCH, 0, None) };
        }
        Err(e) => tracing::warn!(%e, "could not write a later staging copy"),
    }
}

enum Pending {
    Upload(ID3D12Resource, u64),
    Draw(Arc<Draw>, Vec<(Option<u32>, Slot)>),
}

pub(crate) struct Batch {
    events: Vec<Pending>,
    completed: u64,
    execute_us: u64,
}

impl Batch {
    pub(crate) fn before(lists: &[&HostList], fence: &ID3D12Fence) -> Option<Self> {
        let s = selector()?;
        observe(fence, &s.folder);
        let mut events = Vec::new();
        for list in lists {
            for event in list.trace_events() {
                events.push(match event {
                    Event::Upload(r, staging) => Pending::Upload(r, staging.trace_sample()),
                    Event::Draw(draw) => {
                        let snapshot = draw.snapshot();
                        Pending::Draw(draw, snapshot)
                    }
                });
            }
        }
        if events.is_empty() {
            return None;
        }
        // SAFETY: called under the submission lock, immediately before ExecuteCommandLists.
        let completed = unsafe { fence.GetCompletedValue() };
        Some(Self {
            events,
            completed,
            execute_us: micros(),
        })
    }

    pub(crate) fn sent(self, fence: u64, signalled: bool) {
        let Some(s) = selector() else { return };
        let submission = Submission {
            fence,
            signalled,
            execute_us: self.execute_us,
            completed: self.completed,
        };
        for (order, event) in self.events.into_iter().enumerate() {
            match event {
                Pending::Upload(r, sample) => {
                    let data = [
                        if signalled { fence } else { 0 },
                        order as u64,
                        self.execute_us,
                        sample,
                    ];
                    store(&r, &UPLOAD, &data);
                    follow_upload(s, &r, data);
                }
                Pending::Draw(draw, bound) => {
                    if let Err(e) = draw.write(s, bound, &submission, order) {
                        tracing::warn!(%e, "could not write the draw resource trace");
                    }
                }
            }
        }
    }
}

struct Submission {
    fence: u64,
    signalled: bool,
    execute_us: u64,
    completed: u64,
}

impl Draw {
    fn write(
        &self,
        s: &Selector,
        bound: Vec<(Option<u32>, Slot)>,
        submission: &Submission,
        order: usize,
    ) -> io::Result<()> {
        let Submission {
            fence, signalled, ..
        } = submission;
        std::fs::create_dir_all(&s.folder)?;
        let path = s.folder.join(format!(
            "draw-{}-{}-{}-{}-f{fence}-e{order}.tsv",
            s.frame,
            s.hash,
            s.occurrence,
            run_id()
        ));
        let mut out = BufWriter::new(File::create(&path)?);
        writeln!(
            out,
            "version\t1\nsignature\t{}\nframe\t{}\noccurrence\t{}\nrecord_us\t{}\nexecute_us\t{}\ndraw_fence\t{fence}\nevent_order\t{order}\nsignal_ok\t{signalled}\ncompleted_before\t{}\nfence_file\t{}\nfollow_file\t{}",
            self.signature,
            s.frame,
            s.occurrence,
            self.recorded_us,
            submission.execute_us,
            submission.completed,
            fence_file(),
            follow_file()
        )?;
        for line in self.details.lines() {
            writeln!(out, "detail\t{}", line.replace('\t', " "))?;
        }
        writeln!(
            out,
            "role\ttitle_slot\thost_slot\tsubmit_slot\ttitle\trecord_state\trecord_object\trecord_desc\tsubmit_state\tsubmit_object\tsubmit_desc\tupload_fence\tupload_order\tupload_us\tupload_sample\twatch"
        )?;
        let number = |n: Option<u32>| n.map_or_else(|| "-".into(), |v| v.to_string());
        for ((role, b), (submit_slot, slot)) in self.bindings.iter().zip(bound) {
            let (upload, watched) = match &slot {
                Slot::Resource(r) => (private(r, &UPLOAD), watch(r)),
                _ => (None, None),
            };
            let provenance = upload.map_or_else(
                || "-\t-\t-\t-".into(),
                |[f, k, time, hash]| format!("{f}\t{k}\t{time}\t{hash:016x}"),
            );
            writeln!(
                out,
                "{role}\t{}\t{}\t{}\t{}\t{}\t{}\t{provenance}\t{}",
                number(b.title_slot),
                number(b.host_slot),
                number(submit_slot),
                b.title.replace(['\t', '\n'], " "),
                b.recorded.columns(),
                slot.columns(),
                watched.map_or_else(|| "-".into(), |id| id.to_string())
            )?;
        }
        out.flush()?;
        tracing::info!(?path, "wrote draw resource trace");
        Ok(())
    }
}
