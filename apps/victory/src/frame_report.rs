//! Frame-report writer. The rendering path only tries a bounded channel send.
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{RecvTimeoutError, sync_channel};
use std::thread::JoinHandle;
use std::time::Duration;
use anyhow::{Context, Result};

pub(crate) struct Report {
    stopping: Arc<AtomicBool>,
    writer: Option<JoinHandle<Result<()>>>,
}

fn output_path(requested: &Path) -> PathBuf {
    let stem = requested.file_stem().unwrap_or_else(|| std::ffi::OsStr::new("frames"));
    let mut name = stem.to_os_string();
    name.push(format!("-{}.csv", std::process::id()));
    requested.with_file_name(name)
}

pub(crate) fn start() -> Result<Option<Report>> {
    let Some(requested) = std::env::var_os("VICTORY_FRAME_REPORT") else { return Ok(None) };
    let path = output_path(Path::new(&requested));
    victory_core::ensure_outside_repo(&path)?;
    if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
    let file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)
        .with_context(|| format!("creating frame report {}", path.display()))?;
    let mut output = BufWriter::with_capacity(64 * 1024, file);
    writeln!(output, "# format=victory-present-v1")?;
    writeln!(output, "# frame_generation_requested={}", u8::from(std::env::var("VICTORY_FRAMEGEN").as_deref() == Ok("1")))?;
    writeln!(output, "Sequence,TimestampUs,FrameTimeUs")?;
    let (tx, rx) = sync_channel::<victory_hle::frame_report::FrameSample>(1024);
    let stopping = Arc::new(AtomicBool::new(false));
    let stop = stopping.clone();
    let writer = std::thread::Builder::new().name("victory-frame-report".into()).spawn(move || -> Result<()> {
        loop {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(sample) => writeln!(output, "{},{},{}", sample.sequence, sample.timestamp_us, sample.interval_us)?,
                Err(RecvTimeoutError::Timeout) if stop.load(Ordering::Acquire) => break,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        writeln!(output, "# dropped_samples={}", victory_hle::frame_report::dropped())?;
        writeln!(output, "# rejected_presents={}", victory_hle::frame_report::rejected())?;
        writeln!(output, "# expected_samples={}", victory_hle::frame_report::expected())?;
        writeln!(output, "# frame_generation_seen={}", u8::from(victory_hle::frame_report::framegen_seen()))?;
        writeln!(output, "# complete=1")?;
        output.flush()?;
        Ok(())
    })?;
    anyhow::ensure!(victory_hle::frame_report::install(tx), "frame reporting was already installed");
    println!("frame report   {} (host presentation intervals; frame generation must be off for qualification)", path.display());
    Ok(Some(Report { stopping, writer: Some(writer) }))
}

impl Drop for Report {
    fn drop(&mut self) {
        victory_hle::frame_report::stop();
        self.stopping.store(true, Ordering::Release);
        if let Some(writer) = self.writer.take() {
            match writer.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => tracing::warn!(%error, "frame report incomplete; it cannot qualify performance"),
                Err(_) => tracing::warn!("frame report writer panicked; report is incomplete"),
            }
        }
    }
}
