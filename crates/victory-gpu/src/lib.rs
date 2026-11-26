//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod backreads;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(windows)]
mod cardmeter;
#[cfg(all(windows, feature = "diagnostics"))]
mod renderdoc;
#[cfg(all(windows, feature = "diagnostics"))]
mod shot;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod drvwatch;
#[cfg(windows)]
mod fpsbox;
#[cfg(windows)]
mod stats;
#[cfg(windows)]
mod spread;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
pub mod frame;
#[cfg_attr(not(windows), allow(dead_code))]
mod heap_pass;
#[cfg(windows)]
pub mod framelog;
#[cfg(windows)]
mod heap_worker;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod plaincopy;
#[cfg(windows)]
mod primitive;
#[cfg(all(windows, feature = "diagnostics"))]
mod profile;
pub mod query;
pub mod queue;
#[cfg(windows)]
mod queue2;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod tiles;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
mod texback;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use watch::{
    title_memory_committed, title_memory_reading, title_memory_remapped, title_memory_remapping, title_memory_written,
    writing as kernel_will_write,
};

/// For switches set to `alt` (twenty seconds with, twenty without): which
/// half of each forty seconds is "with". 1 by default, the second half, as
/// before; `VICTORY_ALT_FLIP=1` makes it the first.
///
/// Why: the phases start with the process, so in a scripted scene (a
/// cutscene from a save, driven by a key file) each half falls on the same
/// stretches of the scene in every launch, and the scene's own difference
/// between those stretches is read as the switch's. Measured on an RTX
/// 5090 in the opening cutscene: `VICTORY_SYNC_BESIDE=alt` read +11.3 % in
/// a launch as before and +0.9 % flipped; `VICTORY_REAPER=alt` +14.4 % and
/// +1.9 %. One launch each way puts "with" on every stretch once: the mean
/// of the two has the scene taken out.
pub fn alt_on_phase() -> u64 {
    static PHASE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PHASE.get_or_init(|| u64::from(std::env::var_os("VICTORY_ALT_FLIP").is_none_or(|v| v != "1")))
}
