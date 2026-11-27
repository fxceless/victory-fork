//! PlayStation controllers (DualShock 4, DualSense over USB) read as HID
//! devices. Windows gives them to XInput only through extra software (Steam
//! Input, DS4Windows), so without it a player with one had no pad. A thread
//! reads the first one found (input report 0x01; a DualShock 4 over Bluetooth
//! sends the same first bytes as over USB) and `Host` uses it as slot 0 while
//! XInput has nothing there. No rumble or light bar.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use windows_sys::core::GUID;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Layout {
    DualShock4,
    /// over USB (its Bluetooth reports are laid out differently)
    DualSense,
}

/// Sony's controllers by USB product id (vendor 054C).
const PADS_KNOWN: [(&str, Layout, &str); 5] = [
    ("05c4", Layout::DualShock4, "DualShock 4"),
    ("09cc", Layout::DualShock4, "DualShock 4"),
    (
        "0ba0",
        Layout::DualShock4,
        "DualShock 4 USB wireless adaptor",
    ),
    ("0ce6", Layout::DualSense, "DualSense"),
    ("0df2", Layout::DualSense, "DualSense Edge"),
];

/// The pad's state now; `None` while no PlayStation controller is read.
static STATE: Mutex<Option<PadState>> = Mutex::new(None);

/// A PlayStation controller's state, when one is connected; the reading
/// thread starts on the first call. `VICTORY_PLAYSTATION_PAD=0` turns it off.
pub(super) fn pad() -> Option<PadState> {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if !STARTED.swap(true, Ordering::Relaxed)
        && std::env::var("VICTORY_PLAYSTATION_PAD").map_or(true, |v| v.trim() != "0")
        && let Err(e) = std::thread::Builder::new()
            .name("victory playstation pad".into())
            .spawn(reader)
    {
        tracing::warn!(error = %e, "no thread for PlayStation controllers");
    }
    *lock(&STATE)
}

/// Looks for a controller every second; reads one until it is gone.
fn reader() {
    loop {
        if let Some((path, layout, name)) = find() {
            read_until_gone(&path, layout, name);
            *lock(&STATE) = None;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

#[link(name = "cfgmgr32")]
unsafe extern "system" {
    fn CM_Get_Device_Interface_List_SizeW(
        len: *mut u32,
        guid: *const GUID,
        device: *const u16,
        flags: u32,
    ) -> u32;
    fn CM_Get_Device_Interface_ListW(
        guid: *const GUID,
        device: *const u16,
        buf: *mut u16,
        len: u32,
        flags: u32,
    ) -> u32;
}

/// GUID_DEVINTERFACE_HID
const HID: GUID = GUID::from_u128(0x4d1e55b2_f16f_11cf_88cb_001111000030);

/// The first present HID interface of a PlayStation controller.
fn find() -> Option<(String, Layout, &'static str)> {
    let mut len = 0u32;
    // SAFETY: a valid out pointer and GUID; flags 0 = present interfaces only.
    if unsafe { CM_Get_Device_Interface_List_SizeW(&mut len, &HID, std::ptr::null(), 0) } != 0
        || len < 2
    {
        return None;
    }
    let mut buf = vec![0u16; len as usize];
    // SAFETY: `buf` holds `len` characters.
    if unsafe { CM_Get_Device_Interface_ListW(&HID, std::ptr::null(), buf.as_mut_ptr(), len, 0) }
        != 0
    {
        return None;
    }
    buf.split(|&c| c == 0)
        .filter(|p| !p.is_empty())
        .find_map(|p| {
            let path = String::from_utf16(p).ok()?;
            let (layout, name) = known(&path)?;
            Some((path, layout, name))
        })
}

/// The controller a HID interface path names, if it is one read here.
pub(super) fn known(path: &str) -> Option<(Layout, &'static str)> {
    let p = path.to_ascii_lowercase();
    // USB: vid_054c&pid_xxxx; Bluetooth: vid&0002054c_pid&xxxx
    if !p.contains("vid_054c") && !p.contains("vid&0002054c") {
        return None;
    }
    // Bluetooth HID (service 1124): a DualSense's reports there are not the USB ones.
    let bluetooth = p.contains("00001124-0000-1000-8000-00805f9b34fb");
    PADS_KNOWN
        .iter()
        .find(|(pid, layout, _)| {
            (p.contains(&format!("pid_{pid}")) || p.contains(&format!("pid&{pid}")))
                && !(bluetooth && *layout == Layout::DualSense)
        })
        .map(|&(_, layout, name)| (layout, name))
}

fn read_until_gone(path: &str, layout: Layout, name: &str) {
    use std::io::Read;
    // (std opens with read access and every share mode, as HID needs)
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            static ONCE: AtomicBool = AtomicBool::new(false);
            if !ONCE.swap(true, Ordering::Relaxed) {
                tracing::warn!(name, error = %e, "a PlayStation controller could not be opened");
            }
            return;
        }
    };
    tracing::info!(
        name,
        "a PlayStation controller is read as pad slot 0 while XInput has none there"
    );
    let mut buf = [0u8; 128];
    let mut odd = false;
    while let Ok(n @ 1..) = file.read(&mut buf) {
        match parse(layout, &buf[..n]) {
            Some(s) => *lock(&STATE) = Some(s),
            None if !odd => {
                odd = true;
                tracing::info!(
                    name,
                    id = buf[0],
                    bytes = n,
                    "a PlayStation controller report not read"
                );
            }
            None => {}
        }
    }
    tracing::info!(name, "the PlayStation controller was disconnected");
}

/// Input report 0x01 as a pad state (HID sticks: 0 left / up, 128 centre).
pub(super) fn parse(layout: Layout, r: &[u8]) -> Option<PadState> {
    // (face buttons and hat, left trigger, right trigger)
    let (b, l2, r2) = match layout {
        Layout::DualShock4 if r.len() >= 10 => (5, 8, 9),
        Layout::DualSense if r.len() >= 11 => (8, 5, 6),
        _ => return None,
    };
    if r[0] != 0x01 {
        return None;
    }
    const HAT: [u32; 8] = [
        UP,
        UP | RIGHT,
        RIGHT,
        DOWN | RIGHT,
        DOWN,
        DOWN | LEFT,
        LEFT,
        UP | LEFT,
    ];
    const BITS: [(usize, u8, u32); 10] = [
        (0, 0x10, X), // square
        (0, 0x20, A), // cross
        (0, 0x40, B), // circle
        (0, 0x80, Y), // triangle
        (1, 0x01, LEFT_SHOULDER),
        (1, 0x02, RIGHT_SHOULDER),
        (1, 0x10, VIEW), // share / create
        (1, 0x20, MENU), // options
        (1, 0x40, LEFT_STICK),
        (1, 0x80, RIGHT_STICK),
    ];
    let hat = HAT.get(usize::from(r[b] & 0xF)).copied().unwrap_or(0);
    let buttons = BITS
        .iter()
        .filter(|&&(k, m, _)| r[b + k] & m != 0)
        .fold(hat, |s, &(_, _, g)| s | g);
    let x = |v: u8| ((f32::from(v) - 128.0) / 127.0).clamp(-1.0, 1.0);
    let y = |v: u8| ((128.0 - f32::from(v)) / 127.0).clamp(-1.0, 1.0);
    Some(PadState {
        buttons,
        left_trigger: f32::from(r[l2]) / 255.0,
        right_trigger: f32::from(r[r2]) / 255.0,
        left_x: x(r[1]),
        left_y: y(r[2]),
        right_x: x(r[3]),
        right_y: y(r[4]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DualShock 4 report and a DualSense USB report with the same input
    /// give the same state; idle sticks are centred and nothing is pressed.
    #[test]
    fn playstation_reports_read_as_a_pad() {
        let mut ds4 = [0u8; 64];
        ds4[..10].copy_from_slice(&[0x01, 128, 128, 128, 128, 0x08, 0, 0, 0, 0]);
        let idle = parse(Layout::DualShock4, &ds4).unwrap();
        assert_eq!(idle.buttons, 0);
        assert_eq!(
            (idle.left_x, idle.left_y, idle.right_x, idle.right_y),
            (0.0, 0.0, 0.0, 0.0)
        );
        // cross + circle, hat down-left, L1 + options + R3, left stick full
        // left and up, right stick full right and down, L2 full, R2 half
        ds4[..10].copy_from_slice(&[
            0x01,
            0,
            0,
            255,
            255,
            0x60 | 5,
            0x01 | 0x20 | 0x80,
            0,
            255,
            128,
        ]);
        let s = parse(Layout::DualShock4, &ds4).unwrap();
        assert_eq!(
            s.buttons,
            A | B | DOWN | LEFT | LEFT_SHOULDER | MENU | RIGHT_STICK
        );
        assert_eq!(
            (s.left_x, s.left_y, s.right_x, s.right_y),
            (-1.0, 1.0, 1.0, -1.0)
        );
        assert_eq!(s.left_trigger, 1.0);
        assert!((s.right_trigger - 0.502).abs() < 0.001);
        // the same on a DualSense over USB: triggers at 5 and 6, buttons from 8
        let mut ds = [0u8; 64];
        ds[..11].copy_from_slice(&[
            0x01,
            0,
            0,
            255,
            255,
            255,
            128,
            0,
            0x60 | 5,
            0x01 | 0x20 | 0x80,
            0,
        ]);
        assert_eq!(parse(Layout::DualSense, &ds), Some(s));
        // other reports and short ones are not read
        assert_eq!(parse(Layout::DualShock4, &[0x11; 64]), None);
        assert_eq!(parse(Layout::DualShock4, &[0x01; 9]), None);
    }

    #[test]
    fn playstation_controllers_are_known_by_their_hid_path() {
        let usb =
            r"\\?\hid#vid_054c&pid_05c4#6&75b5218&4&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}";
        assert_eq!(known(usb), Some((Layout::DualShock4, "DualShock 4")));
        assert_eq!(
            known(&usb.to_uppercase()),
            Some((Layout::DualShock4, "DualShock 4"))
        );
        assert_eq!(
            known(r"\\?\hid#vid_054c&pid_0ce6#7&1#{4d1e55b2-f16f-11cf-88cb-001111000030}"),
            Some((Layout::DualSense, "DualSense"))
        );
        let bt = r"\\?\hid#{00001124-0000-1000-8000-00805f9b34fb}_vid&0002054c_pid&";
        assert_eq!(
            known(&format!(
                "{bt}09cc#9&1#{{4d1e55b2-f16f-11cf-88cb-001111000030}}"
            ))
            .map(|k| k.0),
            Some(Layout::DualShock4)
        );
        assert_eq!(
            known(&format!(
                "{bt}0ce6#9&1#{{4d1e55b2-f16f-11cf-88cb-001111000030}}"
            )),
            None
        );
        assert_eq!(
            known(r"\\?\hid#vid_045e&pid_0b13#1#{4d1e55b2-f16f-11cf-88cb-001111000030}"),
            None
        );
    }
}
