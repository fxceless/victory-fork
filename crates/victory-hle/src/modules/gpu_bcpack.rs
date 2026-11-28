//! Opt-in GPU BCPack offload backed by `bcpack-gpu-sdk` (`bcpack_gpu::ffi`, DX12 Async Compute).
//!
//! Enabled only when built with `--features gpu-bcpack` AND toggled at runtime
//! with `VICTORY_GPU_BCPACK=1` (off by default). Uploads `victory-formats`'s
//! `bcpack-tables.bin` tables at worker startup via runtime C-ABI setters,
//! routes `BC1`, `BC3`, `BC4`, `BC5`, `BC6H`, and `BC7` (Modes 0–7) to GPU
//! Async Compute (`bcpack-gpu-sdk v0.15.1`), and automatically coalesces
//! concurrent worker jobs via `BcPackDecodeBatch`. Chunks smaller than
//! `VICTORY_GPU_BCPACK_MIN_KB` (default `32` KiB) or invalid/unsupported
//! sizes fall back transparently to `victory-formats`'s CPU decoder.

use std::sync::OnceLock;
use std::sync::mpsc::{self, Receiver, Sender};

use bcpack_gpu::ffi::{
    BCPACK_FORMAT_BC1, BCPACK_FORMAT_BC3, BCPACK_FORMAT_BC4, BCPACK_FORMAT_BC5,
    BCPACK_FORMAT_BC6H, BCPACK_FORMAT_BC7, BCPACK_SWIZZLE_LINEAR, BcPackBatchItem,
    BcPackContext, BcPackDecodeAsync, BcPackDecodeBatch, bcpack_gpu_create, bcpack_gpu_destroy,
    bcpack_gpu_set_bc1_tables, bcpack_gpu_set_bc6h_tables, bcpack_gpu_set_bc7_tables,
};

const TABLES_BIN: &[u8] = include_bytes!("../../../victory-formats/src/bcpack-tables.bin");

const MAX_BATCH_CHUNKS: u32 = 64;
const MAX_BATCH_BLOCKS: u32 = 2_097_152; // 32 MiB output budget per submission
const MAX_BATCH_STREAM_BYTES: u32 = 24 * 1024 * 1024;

struct Job {
    bc: u8,
    src: Vec<u8>,
    out_len: usize,
    reply: mpsc::SyncSender<Result<Vec<u8>, &'static str>>,
}

struct FfiDecoder(*mut BcPackContext);

fn bc_format_layout(bc: u8) -> Option<(u32, usize)> {
    match bc {
        1 => Some((BCPACK_FORMAT_BC1, 8usize)),
        3 => Some((BCPACK_FORMAT_BC3, 16usize)),
        4 => Some((BCPACK_FORMAT_BC4, 8usize)),
        5 => Some((BCPACK_FORMAT_BC5, 16usize)),
        6 => Some((BCPACK_FORMAT_BC6H, 16usize)),
        7 => Some((BCPACK_FORMAT_BC7, 16usize)),
        _ => None,
    }
}

impl FfiDecoder {
    fn new() -> Option<Self> {
        // SAFETY: `bcpack_gpu_create` is thread-safe and returns null on adapter failure.
        let ctx = unsafe {
            bcpack_gpu_create(MAX_BATCH_CHUNKS, MAX_BATCH_BLOCKS, MAX_BATCH_STREAM_BYTES)
        };
        if ctx.is_null() {
            return None;
        }
        let dec = Self(ctx);
        if !upload_victory_tables(dec.0) {
            return None;
        }
        Some(dec)
    }

    fn decode(&mut self, bc: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, &'static str> {
        let Some((fmt, blk_bytes)) = bc_format_layout(bc) else {
            return Err("unsupported BC format");
        };
        if !out_len.is_multiple_of(blk_bytes) || out_len == 0 {
            return Err("invalid output size");
        }
        let num_blocks = (out_len / blk_bytes) as u32;
        let mut out = vec![0u8; out_len];
        let mut fence = 0u64;
        // SAFETY: `self.0` is valid for `&mut self`; `src` and `out` point to live slices of their lengths.
        let rc = unsafe {
            BcPackDecodeAsync(
                self.0,
                src.as_ptr(),
                src.len() as u32,
                num_blocks,
                fmt,
                BCPACK_SWIZZLE_LINEAR,
                0,
                0,
                out.as_mut_ptr(),
                out_len as u32,
                &mut fence,
            )
        };
        if rc == 0 {
            Ok(out)
        } else {
            Err("GPU BCPack decode error")
        }
    }

    fn decode_batch(&mut self, jobs: &[Job]) -> Vec<Result<Vec<u8>, &'static str>> {
        let ctx = self.0;
        decode_batch_with(jobs, |items| {
            let mut kernel_ms = 0.0f64;
            // SAFETY: `ctx` is live for `&mut self`; valid items point into `job.src` and exclusive `out_bufs` entries.
            unsafe {
                BcPackDecodeBatch(
                    ctx,
                    items.as_mut_ptr(),
                    items.len() as u32,
                    &mut kernel_ms,
                )
            }
        })
    }
}

fn decode_batch_with<F>(jobs: &[Job], mut invoke: F) -> Vec<Result<Vec<u8>, &'static str>>
where
    F: FnMut(&mut [BcPackBatchItem]) -> i32,
{
    let mut out_bufs: Vec<Vec<u8>> = Vec::with_capacity(jobs.len());
    let mut items: Vec<BcPackBatchItem> = Vec::with_capacity(jobs.len());

    for job in jobs {
        if let Some((fmt, blk_bytes)) = bc_format_layout(job.bc)
            && job.out_len > 0
            && job.out_len.is_multiple_of(blk_bytes)
            && !job.src.is_empty()
        {
            let mut buf = vec![0u8; job.out_len];
            let out_ptr = buf.as_mut_ptr();
            items.push(BcPackBatchItem {
                compressed_data: job.src.as_ptr(),
                compressed_size: job.src.len() as u32,
                num_blocks: (job.out_len / blk_bytes) as u32,
                format: fmt,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: out_ptr,
                out_capacity: job.out_len as u32,
                status: -1,
            });
            out_bufs.push(buf);
        } else {
            items.push(BcPackBatchItem {
                compressed_data: std::ptr::null(),
                compressed_size: 0,
                num_blocks: 0,
                format: BCPACK_FORMAT_BC1,
                swizzle_mode: BCPACK_SWIZZLE_LINEAR,
                width_blocks: 0,
                height_blocks: 0,
                out_host_buffer: std::ptr::null_mut(),
                out_capacity: 0,
                status: -1,
            });
            out_bufs.push(Vec::new());
        }
    }

    let rc = invoke(&mut items);
    if rc < 0 {
        return vec![Err("GPU BCPack batch decode error"); jobs.len()];
    }

    out_bufs
        .into_iter()
        .zip(items)
        .map(|(buf, item)| {
            if item.status == 0 {
                Ok(buf)
            } else {
                Err("GPU BCPack batch decode error")
            }
        })
        .collect()
}

impl Drop for FfiDecoder {
    fn drop(&mut self) {
        // SAFETY: `self.0` was created by `bcpack_gpu_create` and is destroyed once here.
        unsafe { bcpack_gpu_destroy(self.0) };
    }
}

fn worker_loop(rx: Receiver<Job>) {
    let mut dec = FfiDecoder::new();
    while let Ok(first) = rx.recv() {
        let mut batch = Vec::with_capacity(MAX_BATCH_CHUNKS as usize);
        batch.push(first);
        while batch.len() < MAX_BATCH_CHUNKS as usize {
            match rx.try_recv() {
                Ok(job) => batch.push(job),
                Err(_) => break,
            }
        }

        if batch.len() == 1 {
            let job = batch.pop().expect("single job");
            let res = match dec.as_mut() {
                Some(d) => d.decode(job.bc, &job.src, job.out_len),
                None => Err("no DX12 compute adapter"),
            };
            let _ = job.reply.send(res);
        } else {
            let results = match dec.as_mut() {
                Some(d) => d.decode_batch(&batch),
                None => vec![Err("no DX12 compute adapter"); batch.len()],
            };
            for (job, res) in batch.into_iter().zip(results) {
                let _ = job.reply.send(res);
            }
        }
    }
}

fn sender() -> Option<&'static Sender<Job>> {
    static TX: OnceLock<Option<Sender<Job>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("victory-gpu-bcpack".into())
            .spawn(move || worker_loop(rx))
            .ok()
            .map(|_| tx)
    })
    .as_ref()
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_GPU_BCPACK").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

fn min_bytes() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_GPU_BCPACK_MIN_KB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(32)
            * 1024
    })
}

fn upload_victory_tables(ctx: *mut BcPackContext) -> bool {
    let rank1_off = 2304;
    let rank1: &[u8; 256] = TABLES_BIN[rank1_off..rank1_off + 256]
        .try_into()
        .expect("rank1 slice");

    let ctrl7_off = rank1_off + 256 + 64 + 256 + 64 + 64;
    let mut ctrl7 = [0u16; 512];
    for (i, c) in ctrl7.iter_mut().enumerate() {
        let o = ctrl7_off + i * 2;
        *c = u16::from_le_bytes([TABLES_BIN[o], TABLES_BIN[o + 1]]);
    }

    let bc1_off = ctrl7_off + 1024;
    let mut bc1_ep = vec![0u8; 255 * 6];
    let mut bc1_ix = vec![0u8; 255 * 16];
    for k in 0..255 {
        let o = bc1_off + k * 22;
        bc1_ep[k * 6..(k + 1) * 6].copy_from_slice(&TABLES_BIN[o..o + 6]);
        bc1_ix[k * 16..(k + 1) * 16].copy_from_slice(&TABLES_BIN[o + 6..o + 22]);
    }

    let bc6_off = bc1_off + 255 * (22 + 40 + 18 + 36);
    let mut bc6_ep16 = vec![0u16; 255 * 12];
    let mut bc6_ix_mode = [0u8; 255];
    let mut bc6_ix16 = vec![0i16; 255 * 16];
    for k in 0..255 {
        let o = bc6_off + k * 58;
        for j in 0..12 {
            let p = o + 1 + j * 2;
            bc6_ep16[k * 12 + j] = u16::from_le_bytes([TABLES_BIN[p], TABLES_BIN[p + 1]]);
        }
        bc6_ix_mode[k] = TABLES_BIN[o + 25];
        for p in 0..16 {
            let q = o + 26 + p * 2;
            bc6_ix16[k * 16 + p] = i16::from_le_bytes([TABLES_BIN[q], TABLES_BIN[q + 1]]);
        }
    }

    let bc7_off = bc6_off + 255 * 58;
    let mut bc7_subs = [0u8; 255];
    let mut bc7_ep8 = vec![0u8; 255 * 24];
    let mut bc7_ix4 = vec![0u8; 255 * 32];
    for k in 0..255 {
        let o = bc7_off + k * 116;
        let ctrl = u16::from_le_bytes([TABLES_BIN[o], TABLES_BIN[o + 1]]);
        bc7_subs[k] = ((ctrl & 0xff) >> 6) as u8;
        for j in 0..24 {
            let p = o + 2 + j * 2;
            bc7_ep8[k * 24 + j] = (u16::from_le_bytes([TABLES_BIN[p], TABLES_BIN[p + 1]]) >> 8) as u8;
        }
        let b0 = TABLES_BIN[o + 50];
        if b0 > 0 && b0 <= 4 {
            for p in 0..16 {
                let q = o + 54 + p * 2;
                let v = i16::from_le_bytes([TABLES_BIN[q], TABLES_BIN[q + 1]]);
                bc7_ix4[k * 32 + p] = ((v << (4 - b0)) & 0xF) as u8;
            }
        }
    }

    // SAFETY: `ctx` is live; all arrays match the sizes required by `bcpack_gpu.h`.
    unsafe {
        bcpack_gpu_set_bc1_tables(ctx, rank1.as_ptr(), bc1_ep.as_ptr(), bc1_ix.as_ptr()) == 0
            && bcpack_gpu_set_bc7_tables(
                ctx,
                ctrl7.as_ptr(),
                bc7_subs.as_ptr(),
                bc7_ep8.as_ptr(),
                bc7_ix4.as_ptr(),
            ) == 0
            && bcpack_gpu_set_bc6h_tables(
                ctx,
                bc6_ep16.as_ptr(),
                bc6_ix_mode.as_ptr(),
                bc6_ix16.as_ptr(),
            ) == 0
    }
}

/// Attempts GPU BCPack decode when `VICTORY_GPU_BCPACK=1`. Returns `None` to
/// fall back to VIctory's CPU decoder (small chunks, invalid layout, or error).
pub(super) fn try_decode(bc: u8, src: &[u8], out: &mut [u8]) -> Option<Result<(), &'static str>> {
    if !enabled() {
        return None;
    }
    decode_via_gpu_worker(bc, src, out)
}

fn decode_via_gpu_worker(bc: u8, src: &[u8], out: &mut [u8]) -> Option<Result<(), &'static str>> {
    let blk = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return None,
    };
    submit_gpu_worker(bc, src, out, blk)
}

fn submit_with_sender(
    tx: Option<&Sender<Job>>,
    bc: u8,
    src: &[u8],
    out: &mut [u8],
    blk: usize,
) -> Option<Result<(), &'static str>> {
    let tx = tx?;
    if out.len() < min_bytes()
        || !out.len().is_multiple_of(blk)
        || src.len() + 256 > MAX_BATCH_STREAM_BYTES as usize
        || out.len() > (MAX_BATCH_BLOCKS as usize) * 8
    {
        return None;
    }
    let (reply_tx, reply_rx) = mpsc::sync_channel(1);
    let job = Job {
        bc,
        src: src.to_vec(),
        out_len: out.len(),
        reply: reply_tx,
    };
    if tx.send(job).is_err() {
        return None;
    }
    match reply_rx.recv() {
        Ok(Ok(bytes)) => {
            out.copy_from_slice(&bytes);
            Some(Ok(()))
        }
        Ok(Err(e)) => {
            tracing::debug!("gpu-bcpack fallback to CPU: {e}");
            None
        }
        Err(_) => None,
    }
}

fn submit_gpu_worker(
    bc: u8,
    src: &[u8],
    out: &mut [u8],
    blk: usize,
) -> Option<Result<(), &'static str>> {
    submit_with_sender(sender(), bc, src, out, blk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use victory_formats::bcpack::{self, Tables};

    const RANS_L: u32 = 1 << 24;

    struct ModelFreq {
        freq: [[u32; 9]; 128],
        start: [[u32; 9]; 128],
    }

    impl ModelFreq {
        fn load() -> Self {
            let mut freq = [[0u32; 9]; 128];
            let mut start = [[0u32; 9]; 128];
            for m in 0..128 {
                let used = m / 16 + 2;
                let mut at = 0u32;
                for b in 0..9 {
                    let off = (m * 9 + b) * 2;
                    let f = u16::from_le_bytes([TABLES_BIN[off], TABLES_BIN[off + 1]]) as u32;
                    freq[m][b] = f;
                    if b < used {
                        start[m][b] = at;
                        at += f << b.saturating_sub(1);
                    }
                }
            }
            Self { freq, start }
        }

        fn sym_freq_cf(&self, model: usize, s: u16) -> (u32, u32) {
            if s == 0 {
                return (self.freq[model][0] - 1, 1);
            }
            let b = (16 - s.leading_zeros()) as usize;
            let f = self.freq[model][b];
            let st_s = 1u32 << (b - 1);
            let cf = (self.start[model][b] + f * (u32::from(s) - st_s)) & 0xFFFF;
            if cf == 0 {
                (f - 1, 1)
            } else {
                (f, cf)
            }
        }
    }

    fn encode_frag(mf: &ModelFreq, eab: u8, q: u8, syms: &[u16]) -> Vec<u8> {
        let model = usize::from(eab) * 16 + usize::from(q);
        let mut x: u64 = u64::from(RANS_L);
        let mut rev = Vec::new();
        let mut before_last = 0usize;
        for (k, &s) in syms.iter().enumerate().rev() {
            let (f, cf) = mf.sym_freq_cf(model, s);
            let x_max = u64::from(f) << 16;
            while x >= x_max {
                rev.push(x as u8);
                x >>= 8;
            }
            if k == syms.len() - 1 {
                before_last = rev.len();
            }
            x = ((x / u64::from(f)) << 16) + (x % u64::from(f)) + u64::from(cf);
        }
        let mut trim = 0usize;
        while trim < 2 && trim < rev.len() && rev[trim] == 0 && before_last > trim {
            trim += 1;
        }
        let mut payload = (x as u32).to_be_bytes().to_vec();
        payload.extend(rev[trim..].iter().rev());
        let len = payload.len();
        let hdr3 = len > 256;
        let b0 = ((hdr3 as u8) << 5) | (1 << 4) | (q & 15);
        let mut out = if hdr3 {
            let v = len - 257;
            vec![b0, v as u8, (v >> 8) as u8]
        } else {
            vec![b0, (len - 1) as u8]
        };
        out.extend(payload);
        out
    }

    #[derive(Clone, Copy)]
    enum Phase {
        Hdr0,
        Hdr1,
        Hdr2,
        Proc,
        Drain,
    }

    fn interleave_lanes(mf: &ModelFreq, ab: &[u8], target: usize, lane_bytes: &[Vec<u8>]) -> Vec<u8> {
        let n = ab.len();
        let mut queues: Vec<VecDeque<u8>> = lane_bytes.iter().map(|v| v.iter().copied().collect()).collect();
        let mut phase = vec![Phase::Hdr0; n];
        let mut ib: Vec<Option<u8>> = vec![None; n];
        let mut in_rem = vec![0usize; n];
        let mut out_rem = vec![target; n];
        let mut sym_latch = vec![false; n];
        let mut done = vec![false; n];
        let mut p0 = vec![0u32; n];
        let mut model = vec![0usize; n];
        let mut flush = vec![false; n];
        let mut hdr3 = vec![false; n];
        let mut slots = 0u32;
        let mut stream = Vec::new();

        loop {
            let mut active = 0u32;
            for l in 0..n {
                if !(done[l] && !sym_latch[l] && ((slots >> l) & 1 == 0)) {
                    active |= 1 << l;
                }
            }
            if active == 0 {
                break;
            }
            let mut fed = 0usize;
            let mut stall = vec![false; n];
            let mut got_byte = vec![None; n];
            for l in 0..n {
                let need = (!done[l]
                    && ib[l].is_none()
                    && matches!(phase[l], Phase::Hdr0 | Phase::Hdr1 | Phase::Hdr2))
                    || (!done[l] && ib[l].is_none() && matches!(phase[l], Phase::Proc) && in_rem[l] > 0);
                let blk_out = sym_latch[l] && ((slots >> l) & 1 != 0);
                if need && !blk_out {
                    if fed < 8 {
                        if let Some(b) = queues[l].pop_front() {
                            stream.push(b);
                            got_byte[l] = Some(b);
                            fed += 1;
                        } else {
                            stall[l] = true;
                        }
                    } else {
                        stall[l] = true;
                    }
                } else if blk_out {
                    stall[l] = true;
                }
                if !stall[l] {
                    if let Some(b) = got_byte[l] {
                        ib[l] = Some(b);
                    }
                    if sym_latch[l] {
                        slots |= 1 << l;
                        sym_latch[l] = false;
                    }
                }
            }
            for l in 0..n {
                if done[l] || stall[l] {
                    continue;
                }
                match phase[l] {
                    Phase::Hdr0 => {
                        if let Some(b) = ib[l].take() {
                            let q = b & 15;
                            hdr3[l] = b & 0x20 != 0;
                            flush[l] = b & 0x10 != 0;
                            let e = ((i32::from(ab[l]) - i32::from(b >> 6)) & 15) as usize;
                            model[l] = e * 16 + usize::from(q);
                            p0[l] = 0;
                            phase[l] = Phase::Hdr1;
                        }
                    }
                    Phase::Hdr1 => {
                        if let Some(b) = ib[l].take() {
                            if hdr3[l] {
                                in_rem[l] = usize::from(b);
                                phase[l] = Phase::Hdr2;
                            } else {
                                in_rem[l] = usize::from(b) + 1;
                                phase[l] = Phase::Proc;
                            }
                        }
                    }
                    Phase::Hdr2 => {
                        if let Some(b) = ib[l].take() {
                            in_rem[l] = ((usize::from(b) << 8) | in_rem[l]) + 257;
                            phase[l] = Phase::Proc;
                        }
                    }
                    Phase::Proc => {
                        let will = p0[l] < RANS_L && in_rem[l] > 0 && ib[l].is_some();
                        let x = if will {
                            in_rem[l] -= 1;
                            ((p0[l] & 0xFFFFFF) << 8) | u32::from(ib[l].take().unwrap())
                        } else {
                            p0[l]
                        };
                        let nsf = x > RANS_L && out_rem[l] > 0;
                        if nsf {
                            out_rem[l] -= 1;
                            let cf = x & 0xFFFF;
                            let m = model[l];
                            let b = (1..=(m / 16 + 1)).rfind(|&k| mf.start[m][k] <= cf).unwrap_or(0);
                            let s = if b == 0 {
                                0u16
                            } else {
                                ((1u32 << (b - 1)) + (cf - mf.start[m][b]) / mf.freq[m][b]) as u16
                            };
                            let (f, c) = mf.sym_freq_cf(m, s);
                            p0[l] = f * (x >> 16) + cf - c;
                            sym_latch[l] = true;
                        } else {
                            p0[l] = x;
                        }
                        if in_rem[l] == 0 {
                            if out_rem[l] > 0 {
                                phase[l] = if x == RANS_L {
                                    if flush[l] { p0[l] = 0; }
                                    Phase::Hdr0
                                } else {
                                    Phase::Drain
                                };
                            } else {
                                done[l] = true;
                            }
                        }
                    }
                    Phase::Drain => {
                        let x = p0[l];
                        let nsf = x > RANS_L && out_rem[l] > 0;
                        if nsf {
                            out_rem[l] -= 1;
                            let cf = x & 0xFFFF;
                            let m = model[l];
                            let b = (1..=(m / 16 + 1)).rfind(|&k| mf.start[m][k] <= cf).unwrap_or(0);
                            let s = if b == 0 {
                                0u16
                            } else {
                                ((1u32 << (b - 1)) + (cf - mf.start[m][b]) / mf.freq[m][b]) as u16
                            };
                            let (f, c) = mf.sym_freq_cf(m, s);
                            p0[l] = f * (x >> 16) + cf - c;
                            sym_latch[l] = true;
                        }
                        if x <= RANS_L {
                            if out_rem[l] > 0 {
                                phase[l] = Phase::Hdr0;
                                if flush[l] { p0[l] = 0; }
                            } else {
                                done[l] = true;
                            }
                        }
                    }
                }
            }
            if slots != 0 && (slots & active) == active {
                slots = 0;
            }
        }
        stream
    }

    #[derive(Clone, Copy)]
    enum RefMode {
        Zero,
        Full,
    }

    fn encode_test_chunk(
        mf: &ModelFreq,
        bc: u8,
        blocks: usize,
        seed_in: u64,
        valid_ctrl7: &[u16],
        ref_mode: RefMode,
    ) -> Vec<u8> {
        let ab: &[u8] = match bc {
            1 => &[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7],
            3 => &[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7],
            4 => &[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5],
            5 => &[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5],
            6 => &[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4],
            _ => &[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4],
        };
        let per = if bc == 1 || bc == 4 { 1 } else { 2 };
        let rows = blocks * per;
        let mut s = seed_in | 1;
        let mut next = || -> u32 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 32) as u32
        };
        let mut lane_bytes = Vec::with_capacity(ab.len());
        let mut ctrl_codes = Vec::with_capacity(blocks);
        for i in 0..blocks {
            ctrl_codes.push(valid_ctrl7[(i + seed_in as usize) % valid_ctrl7.len()]);
        }
        for (l, &a) in ab.iter().enumerate() {
            let lim = 1u32 << a;
            let syms: Vec<u16> = (0..rows)
                .map(|r| {
                    if matches!(ref_mode, RefMode::Zero) && l < 2 {
                        0
                    } else if bc == 7 && l == 15 {
                        let c = ctrl_codes[r / 2];
                        if r % 2 == 0 { c & 31 } else { (c >> 5) & 15 }
                    } else if bc == 6 && l == 15 {
                        if r % 2 == 0 {
                            ((r / 2) % 14) as u16 // cycles through all BC6H modes 1..=14
                        } else {
                            (next() & 31) as u16
                        }
                    } else {
                        (next() % lim) as u16
                    }
                })
                .collect();
            lane_bytes.push(encode_frag(mf, a, 0, &syms));
        }
        interleave_lanes(mf, ab, rows, &lane_bytes)
    }

    #[test]
    fn gpu_bcpack_matches_victory_cpu() {
        let mf = ModelFreq::load();
        let tables = Tables::builtin().expect("builtin tables");
        let ctrl_off = 2304 + 256 + 64 + 256 + 64 + 64;
        let valid_ctrl7: Vec<u16> = (0..512u16)
            .filter(|&c| {
                let o = ctrl_off + usize::from(c) * 2;
                u16::from_le_bytes([TABLES_BIN[o], TABLES_BIN[o + 1]]) != 0
            })
            .collect();
        assert!(!valid_ctrl7.is_empty());

        let mut small = [0u8; 16 * 1024];
        assert!(decode_via_gpu_worker(7, &[0u8; 64], &mut small).is_none());

        let mut failures = Vec::new();
        for &(ref_mode, label) in &[
            (RefMode::Zero, "no-ref"),
            (RefMode::Full, "full-ref+dict"),
        ] {
            for &(bc, blk_bytes, blocks) in &[
                (1u8, 8usize, 4096usize), // 32 KiB
                (3u8, 16usize, 2048usize),
                (4u8, 8usize, 4096usize),
                (5u8, 16usize, 2048usize),
                (6u8, 16usize, 2048usize),
                (7u8, 16usize, 2048usize),
            ] {
                let chunk = encode_test_chunk(
                    &mf,
                    bc,
                    blocks,
                    u64::from(bc) * 991 + label.len() as u64,
                    &valid_ctrl7,
                    ref_mode,
                );
                let mut cpu_out = vec![0u8; blocks * blk_bytes];
                bcpack::decode(bc, &chunk, &mut cpu_out, &tables)
                    .expect("victory CPU decode failed");

                let mut gpu_out = vec![0u8; blocks * blk_bytes];
                decode_via_gpu_worker(bc, &chunk, &mut gpu_out)
                    .expect("GPU worker fell back unexpectedly")
                    .expect("GPU decode returned error");

                if gpu_out != cpu_out {
                    let diff_bytes = gpu_out.iter().zip(&cpu_out).filter(|(a, b)| a != b).count();
                    let first = gpu_out
                        .iter()
                        .zip(&cpu_out)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    let msg = format!(
                        "BC{bc} [{label}]: MISMATCH {diff_bytes}/{} bytes (first at {first}: gpu={:02x} cpu={:02x})",
                        gpu_out.len(),
                        gpu_out[first],
                        cpu_out[first]
                    );
                    println!("[victory-gpu-bcpack] {msg}");
                    failures.push(msg);
                } else {
                    println!(
                        "[victory-gpu-bcpack] BC{bc} [{label}] ({} KiB, {blocks} blocks): 0 byte mismatches vs victory_formats::bcpack",
                        gpu_out.len() / 1024
                    );
                }
            }
        }
        assert!(failures.is_empty(), "Mismatches:\n{}", failures.join("\n"));
    }

    #[test]
    fn gpu_bcpack_batch_mixed_formats_matches_victory_cpu() {
        let mf = ModelFreq::load();
        let tables = Tables::builtin().expect("builtin tables");
        let ctrl_off = 2304 + 256 + 64 + 256 + 64 + 64;
        let valid_ctrl7: Vec<u16> = (0..512u16)
            .filter(|&c| {
                let o = ctrl_off + usize::from(c) * 2;
                u16::from_le_bytes([TABLES_BIN[o], TABLES_BIN[o + 1]]) != 0
            })
            .collect();
        assert!(!valid_ctrl7.is_empty());

        let mut dec = FfiDecoder::new().expect("DX12 compute adapter");
        let mut expected_list: Vec<Vec<u8>> = Vec::new();
        let mut jobs: Vec<Job> = Vec::new();

        for &(ref_mode, label) in &[
            (RefMode::Zero, "no-ref"),
            (RefMode::Full, "full-ref+dict"),
        ] {
            for &(bc, blk_bytes, blocks) in &[
                (1u8, 8usize, 4096usize),
                (3u8, 16usize, 2048usize),
                (4u8, 8usize, 4096usize),
                (5u8, 16usize, 2048usize),
                (6u8, 16usize, 2048usize),
                (7u8, 16usize, 2048usize),
            ] {
                let chunk = encode_test_chunk(
                    &mf,
                    bc,
                    blocks,
                    u64::from(bc) * 1993 + label.len() as u64,
                    &valid_ctrl7,
                    ref_mode,
                );
                let mut cpu_out = vec![0u8; blocks * blk_bytes];
                bcpack::decode(bc, &chunk, &mut cpu_out, &tables)
                    .expect("victory CPU decode failed");
                expected_list.push(cpu_out);
                let (tx, _rx) = mpsc::sync_channel(1);
                jobs.push(Job {
                    bc,
                    src: chunk,
                    out_len: blocks * blk_bytes,
                    reply: tx,
                });
            }
        }

        let (bad_tx, _bad_rx) = mpsc::sync_channel(1);
        jobs.insert(
            6,
            Job {
                bc: 7,
                src: vec![0u8; 32],
                out_len: 4,
                reply: bad_tx,
            },
        );

        let results = dec.decode_batch(&jobs);
        assert_eq!(results.len(), 13);
        assert!(results[6].is_err(), "Invalid batch item must fail isolated");

        let mut exp_iter = expected_list.iter();
        for (idx, res) in results.into_iter().enumerate() {
            if idx == 6 {
                continue;
            }
            let gpu_out = res.expect("valid batch item failed");
            let cpu_out = exp_iter.next().unwrap();
            assert_eq!(&gpu_out, cpu_out, "Batch mismatch at item {idx}");
        }

        let cases: Vec<(u8, usize, Vec<u8>, Vec<u8>)> = [
            (1u8, 8usize, 4096usize),
            (3u8, 16usize, 2048usize),
            (4u8, 8usize, 4096usize),
            (5u8, 16usize, 2048usize),
            (6u8, 16usize, 2048usize),
            (7u8, 16usize, 2048usize),
        ]
        .iter()
        .map(|&(bc, blk, blks)| {
            let chunk = encode_test_chunk(
                &mf,
                bc,
                blks,
                u64::from(bc) * 4099,
                &valid_ctrl7,
                RefMode::Full,
            );
            let mut cpu_out = vec![0u8; blks * blk];
            bcpack::decode(bc, &chunk, &mut cpu_out, &tables).unwrap();
            (bc, blks * blk, chunk, cpu_out)
        })
        .collect();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(cases.len());
            for (bc, out_len, chunk, cpu_out) in &cases {
                handles.push(s.spawn(move || {
                    let mut gpu_out = vec![0u8; *out_len];
                    decode_via_gpu_worker(*bc, chunk, &mut gpu_out)
                        .expect("GPU worker fell back unexpectedly")
                        .expect("GPU decode returned error");
                    assert_eq!(&gpu_out, cpu_out, "Concurrent BC{bc} mismatch");
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        println!("[victory-gpu-bcpack] v0.15.1 mixed 12/13 batch + 6-thread coalesced decode: 0 byte mismatches");
    }

    #[test]
    fn batch_early_error_return_rejects_items() {
        let mf = ModelFreq::load();
        let tables = Tables::builtin().expect("builtin tables");
        let ctrl_off = 2304 + 256 + 64 + 256 + 64 + 64;
        let valid_ctrl7: Vec<u16> = (0..512u16)
            .filter(|&c| {
                let o = ctrl_off + usize::from(c) * 2;
                u16::from_le_bytes([TABLES_BIN[o], TABLES_BIN[o + 1]]) != 0
            })
            .collect();
        let chunk = encode_test_chunk(&mf, 1, 4096, 9001, &valid_ctrl7, RefMode::Zero);
        let out_len = 4096 * 8;
        let mut cpu_expected = vec![0u8; out_len];
        bcpack::decode(1, &chunk, &mut cpu_expected, &tables).expect("CPU decode failed");

        let (tx, _rx) = mpsc::sync_channel(1);
        let jobs = [Job {
            bc: 1,
            src: chunk.clone(),
            out_len,
            reply: tx,
        }];
        let err_res = decode_batch_with(&jobs, |items| {
            assert_eq!(items[0].status, -1);
            -99
        });
        assert!(err_res[0].is_err());

        let tampered_res = decode_batch_with(&jobs, |items| {
            items[0].status = 0;
            -99
        });
        assert!(tampered_res[0].is_err());

        let mut cpu_fallback = vec![0u8; out_len];
        bcpack::decode(1, &chunk, &mut cpu_fallback, &tables).expect("CPU fallback decode failed");
        assert_eq!(cpu_fallback, cpu_expected);
    }

    #[test]
    fn unavailable_worker_falls_back_without_panicking() {
        let mf = ModelFreq::load();
        let tables = Tables::builtin().expect("builtin tables");
        let ctrl_off = 2304 + 256 + 64 + 256 + 64 + 64;
        let valid_ctrl7: Vec<u16> = (0..512u16)
            .filter(|&c| {
                let o = ctrl_off + usize::from(c) * 2;
                u16::from_le_bytes([TABLES_BIN[o], TABLES_BIN[o + 1]]) != 0
            })
            .collect();
        let chunk = encode_test_chunk(&mf, 1, 4096, 9002, &valid_ctrl7, RefMode::Zero);
        let out_len = 4096 * 8;
        let mut cpu_expected = vec![0u8; out_len];
        bcpack::decode(1, &chunk, &mut cpu_expected, &tables).expect("CPU decode failed");

        let mut out = vec![0xAAu8; out_len];
        assert!(submit_with_sender(None, 1, &chunk, &mut out, 8).is_none());
        let (dead_tx, dead_rx) = mpsc::channel();
        drop(dead_rx);
        assert!(submit_with_sender(Some(&dead_tx), 1, &chunk, &mut out, 8).is_none());

        bcpack::decode(1, &chunk, &mut out, &tables).expect("CPU fallback decode failed");
        assert_eq!(out, cpu_expected);
    }
}
