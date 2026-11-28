//! Loading a prepared image as a real Windows module and binding it.

use std::collections::HashSet;
use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Context, Result, bail};
use victory_pe::{ImageInfo, ImportSymbol};
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::LoadLibraryExW;
use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};

use crate::thunks::{ImportSite, ThunkBlock, UnimplementedPolicy};
use crate::{
    DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH, ImportResolver, PreparedImage, Provenance, Resolution,
    prepare,
};

/// A title image mapped into this process. A clone names the same mapping
/// (no reference of its own): unload one copy only.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    module: HMODULE,
    path: PathBuf,
    info: ImageInfo,
}

// SAFETY: HMODULE is a process-wide handle (the base address); using it from
// any thread is fine.
unsafe impl Send for LoadedImage {}
// SAFETY: as above; `LoadedImage` has no interior mutability.
unsafe impl Sync for LoadedImage {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingImport {
    pub dll: String,
    pub symbol: String,
}

/// Outcome of [`LoadedImage::bind_imports`].
#[derive(Debug, Default, Clone)]
pub struct BindReport {
    pub hle: usize,
    pub host: usize,
    pub title: usize,
    pub missing: Vec<MissingImport>,
}

/// Writes the prepared copy into `dir` and maps it with the Windows loader.
///
/// `dir` receives derived game data, so it must be outside the repository.
pub fn load(prepared: &PreparedImage, dir: &Path) -> Result<LoadedImage> {
    victory_core::ensure_outside_repo(dir)?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(prepared.file_name());
    write_if_changed(&path, &prepared.bytes)?;

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: `wide` is a NUL-terminated path. The image has no entry point
    // and no imports, so loading it runs no title code.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0) };
    if module.is_null() {
        bail!(
            "LoadLibraryExW({}) failed: {}{}",
            path.display(),
            std::io::Error::last_os_error(),
            if prepared.info.has_relocations {
                ""
            } else {
                " (image has no relocations; its preferred base may be taken)"
            }
        );
    }
    let loaded = LoadedImage {
        module,
        path,
        info: prepared.info.clone(),
    };
    // A DLL indexes its TLS properly; only an exe assumes index 0.
    if !loaded.info.is_dll() {
        // SAFETY: just mapped, and no title code has run.
        if let Err(err) = unsafe { crate::tls::redirect(loaded.base(), &loaded.info) } {
            // SAFETY: `module` was just mapped by `LoadLibraryExW` and not yet registered.
            unsafe { FreeLibrary(loaded.module) };
            return Err(err);
        }
    }
    tracing::info!(
        base = format_args!("{:#x}", loaded.base()),
        preferred = format_args!("{:#x}", loaded.info.image_base),
        path = %loaded.path.display(),
        "image mapped"
    );
    crate::threads::mapped(&loaded);
    Ok(loaded)
}

/// Maps the DLLs the title ships in `title_dir` that `main` imports,
/// directly or through each other, in initialisation order (dependencies
/// first). Only DLL names `wanted` accepts are considered, so DLLs the host
/// or our runtime provide stay with them. Imports are not bound.
pub fn load_title_dlls(
    main: &ImageInfo,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    if let Err(err) = map_imports(main, dirs, wanted, &mut HashSet::new(), &mut loaded) {
        unload_all(loaded);
        return Err(err);
    }
    Ok(loaded)
}

/// Maps the DLL `name` from `title_dir` like [`load_title_dlls`] does, with
/// the title DLLs it imports (dependencies first, `name` last); empty if
/// `wanted` refuses it or the folder has no such file. For DLLs a title
/// loads itself; `wanted` must refuse those already mapped.
pub fn load_title_dll(
    name: &str,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    if let Err(err) = map_dll(name, dirs, wanted, &mut HashSet::new(), &mut loaded) {
        unload_all(loaded);
        return Err(err);
    }
    Ok(loaded)
}

fn unload_all(loaded: Vec<(String, LoadedImage)>) {
    for (_, image) in loaded.into_iter().rev() {
        // SAFETY: newly mapped images whose code has never run.
        unsafe { image.unload() };
    }
}

/// Validates that a PE DLL reference is a single plain filename (no path
/// separators, drive prefixes, or `.`/`..` traversal) and normalizes it to
/// lowercase.
fn safe_dll_name(name: &str) -> Option<String> {
    if name.is_empty() || name.contains(['/', '\\', ':', '\0']) || name == "." || name == ".." {
        return None;
    }
    let mut comps = Path::new(name).components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(_)), None) => Some(name.to_ascii_lowercase()),
        _ => None,
    }
}

/// Parses a PE export forwarder (`"dll.Symbol"` or `"dll.#123"`) into its
/// target DLL filename (appending `.dll` when omitted) and [`ImportSymbol`].
fn parse_forward(spec: &str) -> Option<(String, ImportSymbol)> {
    let (module, target) = spec.rsplit_once('.')?;
    if module.is_empty() || target.is_empty() {
        return None;
    }
    let dll = if module.to_ascii_lowercase().ends_with(".dll") {
        module.to_owned()
    } else {
        format!("{module}.dll")
    };
    let symbol = if let Some(ord) = target.strip_prefix('#') {
        ImportSymbol::Ordinal(ord.parse::<u16>().ok()?)
    } else {
        ImportSymbol::Name {
            hint: 0,
            name: target.to_owned(),
        }
    };
    Some((dll, symbol))
}

fn map_imports(
    info: &ImageInfo,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    for module in info.imports.iter().chain(&info.delay_imports) {
        map_dll(&module.dll, dirs, wanted, seen, loaded)?;
    }
    for export in &info.exports {
        if let Some(forward) = &export.forward
            && let Some((dll, _)) = parse_forward(forward)
        {
            map_dll(&dll, dirs, wanted, seen, loaded)?;
        }
    }
    Ok(())
}

fn map_dll(
    name: &str,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    let Some(name) = safe_dll_name(name) else {
        return Ok(());
    };
    let path = dirs.0.join(&name);
    if !seen.insert(name.clone()) || !wanted(&name) || !path.is_file() {
        return Ok(());
    }
    let original = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let prepared = prepare(&original).with_context(|| format!("preparing {}", path.display()))?;
    let image = load(&prepared, dirs.1).with_context(|| format!("loading {}", path.display()))?;
    if let Err(err) = map_imports(&image.info, dirs, wanted, seen, loaded) {
        // SAFETY: `image` was just mapped and never attached.
        unsafe { image.unload() };
        return Err(err);
    }
    loaded.push((name, image));
    Ok(())
}

/// Resolves through `inner`, then through the exports of the title's DLLs.
pub struct WithTitleDlls<'a> {
    pub inner: &'a dyn ImportResolver,
    pub dlls: &'a [(String, LoadedImage)],
}

const MAX_FORWARD_DEPTH: usize = 16;

impl WithTitleDlls<'_> {
    fn resolve_depth(&self, dll: &str, symbol: &ImportSymbol, depth: usize) -> Resolution {
        if depth > MAX_FORWARD_DEPTH {
            return Resolution::Missing;
        }
        let found = self.inner.resolve(dll, symbol);
        if found != Resolution::Missing {
            return found;
        }
        let Some((_, image)) = self
            .dlls
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(dll))
        else {
            return Resolution::Missing;
        };
        match image.lookup_export(symbol) {
            Some(ExportTarget::Address(address)) => Resolution::Found {
                address,
                provenance: Provenance::Title,
            },
            Some(ExportTarget::Forward(next_dll, next_symbol)) => {
                self.resolve_depth(&next_dll, &next_symbol, depth + 1)
            }
            None => Resolution::Missing,
        }
    }
}

impl ImportResolver for WithTitleDlls<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        self.resolve_depth(dll, symbol, 0)
    }
}

enum ExportTarget {
    Address(usize),
    Forward(String, ImportSymbol),
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::publish::write_if_changed(path, bytes)
        .with_context(|| format!("publishing {}", path.display()))
}

impl LoadedImage {
    pub fn base(&self) -> usize {
        self.module as usize
    }

    /// Parsed from the original image (imports, entry point, TLS callbacks).
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    fn lookup_export(&self, symbol: &ImportSymbol) -> Option<ExportTarget> {
        let entry = self.info.exports.iter().find(|e| match symbol {
            ImportSymbol::Name { name, .. } => e.name.as_deref() == Some(name.as_str()),
            ImportSymbol::Ordinal(o) => e.ordinal == *o,
        })?;
        if let Some(rva) = entry.rva {
            if rva >= self.info.size_of_image {
                return None;
            }
            return self
                .base()
                .checked_add(rva as usize)
                .map(ExportTarget::Address);
        }
        let forward = entry.forward.as_deref()?;
        parse_forward(forward).map(|(dll, sym)| ExportTarget::Forward(dll, sym))
    }

    /// Address of a non-forwarded export of the image, if it has one by that name.
    pub fn export(&self, name: &str) -> Option<usize> {
        self.export_symbol(&ImportSymbol::Name {
            hint: 0,
            name: name.to_owned(),
        })
    }

    /// Address of a non-forwarded export, by name or ordinal. Forwarded
    /// exports resolve through [`WithTitleDlls`] so they stay within HLE and
    /// mapped title DLLs instead of asking the host loader.
    pub fn export_symbol(&self, symbol: &ImportSymbol) -> Option<usize> {
        match self.lookup_export(symbol) {
            Some(ExportTarget::Address(addr)) => Some(addr),
            Some(ExportTarget::Forward(..)) | None => None,
        }
    }

    /// Fills every IAT slot of the image.
    ///
    /// Each import goes to `resolver`; whatever it cannot resolve gets an
    /// unimplemented-import stub that behaves according to `policy`.
    pub fn bind_imports(
        &self,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        let mut report = BindReport::default();
        let mut slots: Vec<(u32, usize)> = Vec::with_capacity(self.info.import_count());
        let mut unresolved: Vec<(u32, &str, &ImportSymbol)> = Vec::new();

        // Delay-load slots are bound eagerly too, so the title's own
        // delay-load helper (which would ask the host) never runs.
        for module in self.info.imports.iter().chain(&self.info.delay_imports) {
            for entry in &module.entries {
                match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found {
                        address,
                        provenance,
                    } => {
                        match provenance {
                            Provenance::Hle => report.hle += 1,
                            Provenance::Host => report.host += 1,
                            Provenance::Title => report.title += 1,
                        }
                        slots.push((entry.iat_slot_rva, address));
                    }
                    Resolution::Missing => {
                        unresolved.push((entry.iat_slot_rva, &module.dll, &entry.symbol));
                    }
                }
            }
        }

        if !unresolved.is_empty() {
            let mut block = ThunkBlock::new(unresolved.len())?;
            for (slot, dll, symbol) in unresolved {
                let site: &'static ImportSite = Box::leak(Box::new(ImportSite {
                    dll: dll.to_owned(),
                    symbol: symbol.to_string(),
                    policy,
                    calls: AtomicU64::new(0),
                }));
                slots.push((slot, block.emit(site)?));
                report.missing.push(MissingImport {
                    dll: site.dll.clone(),
                    symbol: site.symbol.clone(),
                });
            }
            block.seal()?;
        }

        self.write_slots(&slots)?;
        tracing::info!(
            hle = report.hle,
            host = report.host,
            missing = report.missing.len(),
            "imports bound"
        );
        Ok(report)
    }

    /// Protects and restores each slot on its own: import tables can sit in
    /// sections with different protections (delay-load tables in `.didat`).
    fn write_slots(&self, slots: &[(u32, usize)]) -> Result<()> {
        for &(rva, address) in slots {
            let slot = (self.base() + rva as usize) as *mut u64;
            let mut old = 0;
            // SAFETY: the parser only yields slots inside the image.
            if unsafe { VirtualProtect(slot as *const c_void, 8, PAGE_READWRITE, &mut old) } == 0 {
                bail!(
                    "VirtualProtect(IAT slot {rva:#x}) failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            // SAFETY: inside the image, now writable, 8-byte aligned by the PE format.
            unsafe { slot.write_volatile(address as u64) };
            let mut ignored = 0;
            // SAFETY: restoring the protection we just changed.
            unsafe { VirtualProtect(slot as *const c_void, 8, old, &mut ignored) };
        }
        Ok(())
    }

    /// Calls the image's TLS callbacks with `reason` (e.g. `DLL_PROCESS_ATTACH`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub unsafe fn run_tls_callbacks(&self, reason: u32) {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, std::ptr::null_mut()) }
    }

    /// # Safety
    /// See [`Self::run_tls_callbacks`].
    unsafe fn tls_callbacks(&self, reason: u32, reserved: *mut c_void) {
        let Some(tls) = &self.info.tls else { return };
        for &rva in &tls.callback_rvas {
            type TlsCallback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
            // SAFETY: the RVA came from the image's own TLS directory.
            let callback: TlsCallback = unsafe { std::mem::transmute(self.base() + rva as usize) };
            // SAFETY: the caller guarantees imports are bound.
            unsafe { callback(self.module, reason, reserved) };
        }
    }

    /// Initialises a DLL as the Windows loader would: its TLS callbacks,
    /// then `DllMain(DLL_PROCESS_ATTACH)`. From then on it hears of every
    /// thread start and end (see `threads`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first, and the DLLs this one
    /// depends on attached.
    pub unsafe fn attach(&self) -> Result<()> {
        // SAFETY: forwarded from the caller.
        if unsafe { self.notify(DLL_PROCESS_ATTACH, true, std::ptr::null_mut()) } == Some(0) {
            bail!("DllMain returned FALSE");
        }
        crate::threads::attached(self.base());
        Ok(())
    }

    /// `DLL_PROCESS_DETACH` as `FreeLibrary` sends it (TLS callbacks, then
    /// `DllMain`); the image hears of no more threads.
    ///
    /// # Safety
    /// Runs title code. The image must have been attached (or its attach
    /// failed), and no other thread may be using it.
    pub unsafe fn detach(&self) {
        crate::threads::detached(self.base());
        // SAFETY: forwarded from the caller.
        unsafe { self.notify(DLL_PROCESS_DETACH, true, std::ptr::null_mut()) };
    }

    /// Runs the image's TLS callbacks with `reason`, then, if `dll_main` and
    /// the image is a DLL with an entry point, its `DllMain`, whose result
    /// is returned. `reserved` is the entry points' third argument.
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub(crate) unsafe fn notify(
        &self,
        reason: u32,
        dll_main: bool,
        reserved: *mut c_void,
    ) -> Option<i32> {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, reserved) };
        if !dll_main || !self.info.is_dll() || self.info.entry_rva == 0 {
            return None;
        }
        type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
        // SAFETY: the entry RVA came from the original image header.
        let main: DllMain =
            unsafe { std::mem::transmute(self.base() + self.info.entry_rva as usize) };
        // SAFETY: forwarded from the caller.
        Some(unsafe { main(self.module, reason, reserved) })
    }

    /// Starts the title: on a new thread with the stack size from the image
    /// header, attaches `dlls` in order (see [`Self::attach`]), runs the
    /// image's TLS callbacks (`DLL_PROCESS_ATTACH`), then its original entry
    /// point. Returns the thread so the caller can join it.
    ///
    /// # Safety
    /// Runs title code. Imports of the image and of `dlls` must be bound.
    pub unsafe fn spawn_main(
        self: Arc<Self>,
        dlls: Vec<(String, LoadedImage)>,
    ) -> Result<std::thread::JoinHandle<Result<u32>>> {
        if self.info.entry_rva == 0 {
            bail!("image has no entry point");
        }
        let stack = usize::try_from(self.info.stack_reserve)
            .unwrap_or(usize::MAX)
            .max(1 << 20);
        std::thread::Builder::new()
            .name("title-main".into())
            .stack_size(stack)
            .spawn(move || {
                for (name, dll) in &dlls {
                    // SAFETY: forwarded from the caller of `spawn_main`.
                    unsafe { dll.attach() }.with_context(|| format!("attaching {name}"))?;
                }
                // SAFETY: forwarded from the caller of `spawn_main`.
                unsafe { self.run_tls_callbacks(DLL_PROCESS_ATTACH) };
                crate::threads::attached(self.base());
                type Entry = unsafe extern "system" fn(*mut c_void) -> u32;
                // SAFETY: the entry RVA came from the original image header.
                let entry: Entry =
                    unsafe { std::mem::transmute(self.base() + self.info.entry_rva as usize) };
                // SAFETY: forwarded from the caller of `spawn_main`.
                Ok(unsafe { entry(std::ptr::null_mut()) })
            })
            .context("spawning title main thread")
    }

    /// Unmaps the image.
    ///
    /// # Safety
    /// No title code may be running or run later, and no pointer into the
    /// image may be used afterwards.
    pub unsafe fn unload(self) {
        crate::threads::unmapped(self.base());
        // SAFETY: guaranteed by the caller.
        unsafe { FreeLibrary(self.module) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_pe::ExportInfo;

    struct FakeResolver {
        hits: Vec<(&'static str, &'static str, usize, Provenance)>,
    }

    impl ImportResolver for FakeResolver {
        fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
            let sym_name = match symbol {
                ImportSymbol::Name { name, .. } => name.as_str(),
                ImportSymbol::Ordinal(_) => return Resolution::Missing,
            };
            self.hits
                .iter()
                .find(|(d, s, _, _)| d.eq_ignore_ascii_case(dll) && *s == sym_name)
                .map_or(Resolution::Missing, |&(_, _, address, provenance)| {
                    Resolution::Found {
                        address,
                        provenance,
                    }
                })
        }
    }

    fn fake_image(base: usize, size_of_image: u32, exports: Vec<ExportInfo>) -> LoadedImage {
        LoadedImage {
            module: base as HMODULE,
            path: PathBuf::new(),
            info: ImageInfo {
                machine: victory_pe::MACHINE_AMD64,
                characteristics: 0x2000,
                timestamp: 0,
                subsystem: 0,
                dll_characteristics: 0,
                image_base: base as u64,
                size_of_image,
                entry_rva: 0,
                stack_reserve: 0,
                stack_commit: 0,
                has_relocations: true,
                has_exception_table: false,
                has_load_config: false,
                sections: Vec::new(),
                imports: Vec::new(),
                delay_imports: Vec::new(),
                exports,
                tls: None,
                codeview: None,
            },
        }
    }

    #[test]
    fn safe_dll_name_rejects_traversal_and_paths() {
        assert_eq!(safe_dll_name("Title_Foo.DLL").as_deref(), Some("title_foo.dll"));
        assert_eq!(safe_dll_name(""), None);
        assert_eq!(safe_dll_name("."), None);
        assert_eq!(safe_dll_name(".."), None);
        assert_eq!(safe_dll_name("../evil.dll"), None);
        assert_eq!(safe_dll_name(r"..\evil.dll"), None);
        assert_eq!(safe_dll_name(r"sub\evil.dll"), None);
        assert_eq!(safe_dll_name(r"C:\Windows\System32\kernel32.dll"), None);
        assert_eq!(safe_dll_name("foo.dll\0bar"), None);
    }

    #[test]
    fn parse_forward_handles_names_ordinals_and_dotted_dlls() {
        assert_eq!(
            parse_forward("KERNELBASE.CreateEventW"),
            Some((
                "KERNELBASE.dll".to_owned(),
                ImportSymbol::Name {
                    hint: 0,
                    name: "CreateEventW".to_owned(),
                }
            ))
        );
        assert_eq!(
            parse_forward("title_b.#42"),
            Some(("title_b.dll".to_owned(), ImportSymbol::Ordinal(42)))
        );
        assert_eq!(
            parse_forward("api-ms-win-core-synch-l1-2-0.dll.WaitOnAddress"),
            Some((
                "api-ms-win-core-synch-l1-2-0.dll".to_owned(),
                ImportSymbol::Name {
                    hint: 0,
                    name: "WaitOnAddress".to_owned(),
                }
            ))
        );
        assert_eq!(parse_forward("nodot"), None);
        assert_eq!(parse_forward("bad.#notanumber"), None);
    }

    #[test]
    fn with_title_dlls_follows_forwards_to_hle_and_title_dlls_and_breaks_cycles() {
        let inner = FakeResolver {
            hits: vec![("kernelbase.dll", "CreateEventW", 0x1111_0000, Provenance::Hle)],
        };
        let dll_a = fake_image(
            0x2000_0000,
            0x4000,
            vec![
                ExportInfo {
                    name: Some("Direct".into()),
                    ordinal: 1,
                    rva: Some(0x1200),
                    forward: None,
                },
                ExportInfo {
                    name: Some("OutOfBounds".into()),
                    ordinal: 2,
                    rva: Some(0x5000),
                    forward: None,
                },
                ExportInfo {
                    name: Some("FwdToHle".into()),
                    ordinal: 3,
                    rva: None,
                    forward: Some("KERNELBASE.CreateEventW".into()),
                },
                ExportInfo {
                    name: Some("FwdToTitleOrd".into()),
                    ordinal: 4,
                    rva: None,
                    forward: Some("title_b.#7".into()),
                },
                ExportInfo {
                    name: Some("Cycle".into()),
                    ordinal: 5,
                    rva: None,
                    forward: Some("title_b.CycleBack".into()),
                },
            ],
        );
        let dll_b = fake_image(
            0x3000_0000,
            0x4000,
            vec![
                ExportInfo {
                    name: None,
                    ordinal: 7,
                    rva: Some(0x2340),
                    forward: None,
                },
                ExportInfo {
                    name: Some("CycleBack".into()),
                    ordinal: 8,
                    rva: None,
                    forward: Some("title_a.Cycle".into()),
                },
            ],
        );
        let dlls = vec![("title_a.dll".into(), dll_a), ("title_b.dll".into(), dll_b)];
        let resolver = WithTitleDlls {
            inner: &inner,
            dlls: &dlls,
        };

        let name = |s: &str| ImportSymbol::Name {
            hint: 0,
            name: s.into(),
        };

        // Direct in-bounds export resolves within title DLL; OOB RVA is rejected.
        assert_eq!(
            resolver.resolve("title_a.dll", &name("Direct")),
            Resolution::Found {
                address: 0x2000_1200,
                provenance: Provenance::Title,
            }
        );
        assert_eq!(
            resolver.resolve("title_a.dll", &name("OutOfBounds")),
            Resolution::Missing
        );

        // Forwarded export into HLE resolves through `inner` (never host GetProcAddress).
        assert_eq!(
            resolver.resolve("title_a.dll", &name("FwdToHle")),
            Resolution::Found {
                address: 0x1111_0000,
                provenance: Provenance::Hle,
            }
        );

        // Forwarded export by ordinal into another mapped title DLL resolves cleanly.
        assert_eq!(
            resolver.resolve("title_a.dll", &name("FwdToTitleOrd")),
            Resolution::Found {
                address: 0x3000_2340,
                provenance: Provenance::Title,
            }
        );

        // Cyclic forwarders terminate cleanly instead of recursing forever.
        assert_eq!(
            resolver.resolve("title_a.dll", &name("Cycle")),
            Resolution::Missing
        );
    }
}
