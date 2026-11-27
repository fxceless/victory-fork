//! System dialogs a title opens (XGameUi). A profile card has nothing to
//! show and closes at once. Text entry opens a small host dialog (title,
//! description, default text, the length limit) on a thread of its own and
//! completes the call with what the user typed, or as cancelled (`E_ABORT`)
//! when the dialog is dismissed or cannot open.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};

use super::xasync::{self, AsyncBlock, E_ABORT};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};

pub const PROVIDER: Guid = Guid::new(
    0xdfcd_4649,
    0x4ff8,
    0x4043,
    [0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0],
);
/// As the title (2026) asks for it; the slots it calls (0x38,
/// 0x40, 0x88-0x98) take the public API's arguments in the 2020 order.
pub const IID_2026: Guid = Guid::new(
    0xeaf6_69df,
    0x5542,
    0x4590,
    [0x99, 0xa3, 0x8d, 0xc0, 0x61, 0xf8, 0x37, 0xcc],
);

pub static OBJECT: Object = Object::new("game ui", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID_2026, &OBJECT)];

type This = *const Object;

static PROFILE: u8 = 0;
static TEXT: u8 = 0;

extern "system" fn profile_card_async(
    _: This,
    block: *mut AsyncBlock,
    _user: *const c_void,
    xuid: u64,
) -> HRESULT {
    tracing::info!(xuid, "profile card: nothing to show");
    xasync::finished(block, (&raw const PROFILE).cast(), S_OK)
}
extern "system" fn profile_card_result(_: This, block: *mut AsyncBlock) -> HRESULT {
    let none = std::ptr::null_mut();
    xasync::XAsyncGetResult(block, (&raw const PROFILE).cast(), 0, none, none.cast())
}

/// GDK `XGameUiShowTextEntryAsync` treats `maxTextLength == 0` as 1024 characters.
const DEFAULT_MAX_TEXT_LENGTH: u32 = 1024;

/// Normalize `maxTextLength`: `0` maps to the 1024-character GDK default.
fn effective_max_length(max: u32) -> u32 {
    if max == 0 {
        DEFAULT_MAX_TEXT_LENGTH
    } else {
        max
    }
}

/// The title treats `E_ABORT` as cancelled and goes on; any
/// other failure as an error.
extern "system" fn text_entry_async(
    _: This,
    block: *mut AsyncBlock,
    title: *const c_char,
    description: *const c_char,
    default: *const c_char,
    _scope: u32,
    max_length: u32,
) -> HRESULT {
    let [title, description, default] = [title, description, default].map(|t| text(t).unwrap_or_default());
    let max_length = effective_max_length(max_length);
    tracing::info!(title, description, max_length, "text entry");
    xasync::pending_with(block, (&raw const TEXT).cast(), move || {
        match ask(&title, &description, &default, max_length) {
            Some(typed) => {
                tracing::info!(length = typed.chars().count(), "text entry: done");
                let mut bytes = limit(&typed, max_length).into_bytes();
                bytes.push(0);
                Ok(bytes)
            }
            None => {
                tracing::info!("text entry: cancelled");
                Err(E_ABORT)
            }
        }
    })
}
/// Bytes of the result, its NUL included.
extern "system" fn text_entry_size(_: This, block: *mut AsyncBlock, out: *mut u32) -> HRESULT {
    let mut size = 0;
    let hr = xasync::XAsyncGetResultSize(block, &mut size);
    if hr < 0 {
        return hr;
    }
    put(out, size as u32)
}
extern "system" fn text_entry_result(
    _: This,
    block: *mut AsyncBlock,
    size: u32,
    buffer: *mut c_char,
    used: *mut u32,
) -> HRESULT {
    let mut written = 0;
    let identity = (&raw const TEXT).cast();
    let hr = xasync::XAsyncGetResult(block, identity, size as usize, buffer.cast(), &mut written);
    if hr < 0 || used.is_null() {
        return hr;
    }
    put(used, written as u32)
}

/// At most `effective_max_length(max)` characters (0: 1024-character default).
fn limit(typed: &str, max: u32) -> String {
    typed
        .chars()
        .take(effective_max_length(max) as usize)
        .collect()
}

/// Asks the user in a modal host dialog; `None` if dismissed.
#[cfg(not(test))]
fn ask(title: &str, description: &str, default: &str, max: u32) -> Option<String> {
    dialog::ask(title, description, default, max)
}

/// Tests answer for the user: `default` + "typed" for a title starting with
/// "Type" when `max` has been normalized (non-zero), cancelled otherwise.
#[cfg(test)]
fn ask(title: &str, _: &str, default: &str, max: u32) -> Option<String> {
    (max != 0 && title.starts_with("Type")).then(|| format!("{default}typed"))
}

mod dialog {
    use windows_sys::Win32::Foundation::{HWND, LPARAM, WPARAM};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::SetFocus;
    use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, DLGTEMPLATE};

    const EDIT: i32 = 100;
    // Edit control messages (the controls headers).
    const EM_SETSEL: u32 = 0xb1;
    const EM_LIMITTEXT: u32 = 0xc5;

    /// The dialog's state, behind its user data while it runs.
    struct Entry {
        max: u32,
        typed: Option<String>,
    }

    /// `EM_LIMITTEXT` parameter for `max` (normalizing `0` to the 1024 default).
    pub(super) fn limit_text_wparam(max: u32) -> WPARAM {
        super::effective_max_length(max) as WPARAM
    }

    /// The description, an edit box with the default text, OK and Cancel,
    /// from an in-memory template (dialog units, the system font); Enter
    /// accepts, Esc cancels.
    #[allow(dead_code)] // unused in test builds
    pub(super) fn ask(title: &str, description: &str, default: &str, max: u32) -> Option<String> {
        let mut t: Vec<u16> = Vec::new();
        fn dword(t: &mut Vec<u16>, v: u32) {
            t.extend([v as u16, (v >> 16) as u16]);
        }
        fn text(t: &mut Vec<u16>, s: &str) {
            t.extend(s.encode_utf16().filter(|&c| c != 0));
            t.push(0);
        }
        let style = wm::WS_POPUP | wm::WS_CAPTION | wm::WS_SYSMENU;
        let flags = (wm::DS_MODALFRAME | wm::DS_CENTER | wm::DS_SETFOREGROUND) as u32;
        dword(&mut t, style | flags);
        dword(&mut t, wm::WS_EX_TOPMOST);
        t.extend([4, 0, 0, 240, 62, 0, 0]); // items, x, y, cx, cy, no menu, default class
        text(&mut t, title);
        let mut item = |style: u32, rect: [u16; 4], id: u16, class: u16, caption: &str| {
            if t.len() % 2 == 1 {
                t.push(0); // items start on a DWORD
            }
            dword(&mut t, wm::WS_CHILD | wm::WS_VISIBLE | style);
            dword(&mut t, 0);
            t.extend(rect);
            t.extend([id, 0xffff, class]);
            text(&mut t, caption);
            t.push(0); // no creation data
        };
        let (button, edit, label) = (0x80, 0x81, 0x82);
        let tab = wm::WS_TABSTOP;
        item(0, [7, 7, 226, 10], 0xffff, label, description);
        let edit_style = wm::WS_BORDER | tab | wm::ES_AUTOHSCROLL as u32;
        item(edit_style, [7, 20, 226, 12], EDIT as u16, edit, default);
        let ok = wm::BS_DEFPUSHBUTTON as u32 | tab;
        item(ok, [129, 40, 50, 14], wm::IDOK as u16, button, "OK");
        let cancel = wm::BS_PUSHBUTTON as u32 | tab;
        item(cancel, [183, 40, 50, 14], wm::IDCANCEL as u16, button, "Cancel");
        // The template must start on a DWORD.
        let template: Vec<u32> = t
            .chunks(2)
            .map(|c| u32::from(c[0]) | u32::from(c.get(1).copied().unwrap_or(0)) << 16)
            .collect();
        let mut entry = Entry {
            max: super::effective_max_length(max),
            typed: None,
        };
        // SAFETY: a complete template and our dialog procedure; `entry`
        // outlives the modal dialog.
        unsafe {
            wm::DialogBoxIndirectParamW(
                GetModuleHandleW(std::ptr::null()),
                template.as_ptr().cast::<DLGTEMPLATE>(),
                std::ptr::null_mut(),
                Some(procedure),
                (&raw mut entry) as LPARAM,
            );
        }
        entry.typed
    }

    extern "system" fn procedure(dialog: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> isize {
        // SAFETY: our dialog; its user data is the `Entry` from `ask`,
        // stored at WM_INITDIALOG.
        unsafe {
            let edit = wm::GetDlgItem(dialog, EDIT);
            match message {
                wm::WM_INITDIALOG => {
                    wm::SetWindowLongPtrW(dialog, wm::GWLP_USERDATA, lparam);
                    let entry = &*(lparam as *const Entry);
                    wm::SendMessageW(edit, EM_LIMITTEXT, limit_text_wparam(entry.max), 0);
                    wm::SendMessageW(edit, EM_SETSEL, 0, -1);
                    SetFocus(edit);
                    0 // focus set here
                }
                wm::WM_COMMAND => {
                    let entry = &mut *(wm::GetWindowLongPtrW(dialog, wm::GWLP_USERDATA) as *mut Entry);
                    match (wparam & 0xffff) as i32 {
                        wm::IDOK => {
                            let mut buffer = vec![0u16; wm::GetWindowTextLengthW(edit) as usize + 1];
                            let n = wm::GetWindowTextW(edit, buffer.as_mut_ptr(), buffer.len() as i32);
                            entry.typed = Some(String::from_utf16_lossy(&buffer[..n.max(0) as usize]));
                            wm::EndDialog(dialog, 1);
                        }
                        wm::IDCANCEL => {
                            wm::EndDialog(dialog, 0);
                        }
                        _ => return 0,
                    }
                    1
                }
                _ => 0,
            }
        }
    }
}

static VTABLE: Methods<20> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),        // 0x18 message dialog
    missing::<4> as *const (),        // 0x20
    missing::<5> as *const (),        // 0x28 send invite
    missing::<6> as *const (),        // 0x30
    profile_card_async as *const (),  // 0x38
    profile_card_result as *const (), // 0x40
    missing::<9> as *const (),        // 0x48 achievements
    missing::<10> as *const (),       // 0x50
    missing::<11> as *const (),       // 0x58 player picker
    missing::<12> as *const (),       // 0x60
    missing::<13> as *const (),       // 0x68
    missing::<14> as *const (),       // 0x70 error dialog
    missing::<15> as *const (),       // 0x78
    missing::<16> as *const (),       // 0x80 notification position
    text_entry_async as *const (),    // 0x88
    text_entry_size as *const (),     // 0x90
    text_entry_result as *const (),   // 0x98
]);

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use super::*;
    use crate::modules::xgameruntime::method;

    type Entry = extern "system" fn(
        This,
        *mut AsyncBlock,
        *const c_char,
        *const c_char,
        *const c_char,
        u32,
        u32,
    ) -> HRESULT;
    type Size = extern "system" fn(This, *mut AsyncBlock, *mut u32) -> HRESULT;
    type Result_ = extern "system" fn(This, *mut AsyncBlock, u32, *mut c_char, *mut u32) -> HRESULT;

    fn block() -> AsyncBlock {
        AsyncBlock {
            queue: std::ptr::null(),
            context: std::ptr::null_mut(),
            callback: None,
            internal: [0; 4],
        }
    }

    fn run_text_entry(default: &str, max_length: u32) -> (u32, String, u32) {
        let this: This = &OBJECT;
        let mut b = block();
        let default = CString::new(default).unwrap();
        let entry = method::<Entry>(&OBJECT, 0x88);
        assert_eq!(
            entry(
                this,
                &mut b,
                c"Type".as_ptr(),
                std::ptr::null(),
                default.as_ptr(),
                0,
                max_length,
            ),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        let mut size = 0;
        assert_eq!(method::<Size>(&OBJECT, 0x90)(this, &mut b, &mut size), S_OK);
        let mut out = vec![0 as c_char; size as usize];
        let mut used = 0;
        assert_eq!(
            method::<Result_>(&OBJECT, 0x98)(this, &mut b, size, out.as_mut_ptr(), &mut used),
            S_OK
        );
        let returned = text(out.as_ptr()).expect("result should be valid UTF-8 with NUL");
        (size, returned, used)
    }

    #[test]
    fn dialogs_end_with_what_was_typed() {
        let this: This = &OBJECT;
        let mut b = block();
        type Card = extern "system" fn(This, *mut AsyncBlock, *const c_void, u64) -> HRESULT;
        type Done = extern "system" fn(This, *mut AsyncBlock) -> HRESULT;
        assert_eq!(
            method::<Card>(&OBJECT, 0x38)(this, &mut b, std::ptr::null(), 1),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        assert_eq!(method::<Done>(&OBJECT, 0x40)(this, &mut b), S_OK);

        let null = std::ptr::null();
        let entry = method::<Entry>(&OBJECT, 0x88);
        // Dismissed: cancelled.
        let mut b = block();
        assert_eq!(entry(this, &mut b, c"Name".as_ptr(), null, null, 0, 16), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), E_ABORT);
        let mut size = 0;
        assert_eq!(method::<Size>(&OBJECT, 0x90)(this, &mut b, &mut size), E_ABORT);
        // Typed: the text, cut to the limit, with its NUL.
        let mut b = block();
        let default = c"abc".as_ptr();
        assert_eq!(entry(this, &mut b, c"Type".as_ptr(), null, default, 0, 6), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        assert_eq!(method::<Size>(&OBJECT, 0x90)(this, &mut b, &mut size), S_OK);
        assert_eq!(size, 7);
        let mut out = [0 as c_char; 7];
        let mut used = 0;
        let result = method::<Result_>(&OBJECT, 0x98);
        assert_eq!(result(this, &mut b, 7, out.as_mut_ptr(), &mut used), S_OK);
        assert_eq!((text(out.as_ptr()).as_deref(), used), (Some("abctyp"), 7));
        assert_eq!(limit("\u{e9}t\u{e9}s", 3), "\u{e9}t\u{e9}");
    }

    #[test]
    fn max_length_zero_applies_1024_default() {
        assert_eq!(effective_max_length(0), 1024);
        assert_eq!(dialog::limit_text_wparam(0), 1024);

        let over_limit = "a".repeat(1025);
        let expected = "a".repeat(1024);
        assert_eq!(limit(&over_limit, 0), expected);
        assert_eq!(limit(&expected, 0), expected);

        let (size, returned, used) = run_text_entry(&over_limit, 0);
        assert_eq!((size, used), (1025, 1025));
        assert_eq!(returned, expected);

        let (short_size, short_returned, short_used) = run_text_entry("hi", 0);
        assert_eq!((short_size, short_used), (8, 8));
        assert_eq!(short_returned, "hityped");
    }

    #[test]
    fn explicit_positive_limit_and_utf8_boundaries() {
        assert_eq!(effective_max_length(16), 16);
        assert_eq!(dialog::limit_text_wparam(16), 16);

        assert_eq!(limit(&"b".repeat(20), 16), "b".repeat(16));
        assert_eq!(limit("short", 16), "short");

        let (size, returned, used) = run_text_entry(&"b".repeat(20), 16);
        assert_eq!((size, used), (17, 17));
        assert_eq!(returned, "b".repeat(16));

        let (short_size, short_returned, short_used) = run_text_entry("hi", 16);
        assert_eq!((short_size, short_used), (8, 8));
        assert_eq!(short_returned, "hityped");

        // Multi-byte UTF-8 characters are bounded by scalar count, not byte count,
        // and the result buffer still includes the trailing NUL byte.
        let utf8_over = "\u{e9}".repeat(1025);
        let utf8_expected = "\u{e9}".repeat(1024);
        assert_eq!(limit(&utf8_over, 0), utf8_expected);
        let (utf8_size, utf8_returned, utf8_used) = run_text_entry(&utf8_over, 0);
        assert_eq!((utf8_size, utf8_used), (1024 * 2 + 1, 1024 * 2 + 1));
        assert_eq!(utf8_returned, utf8_expected);
    }
}
