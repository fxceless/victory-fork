//! Hand-written replacements for translated shaders.
//!
//! `VICTORY_SHADER_OVERRIDES=<folder>`: a translated shader whose console
//! container hashes to `<hash>` (the 16 hex digits runs name shaders by:
//! "first bind", "shader does not translate", the GPU timing lines) runs the
//! DXIL container in `<folder>\<hash>.dxil` instead of the translator's.
//!
//! Only the code is replaced. Everything the host builds around a shader
//! comes from the translation as before: its root signature, the user-data
//! dwords it reads, its immediate samplers, its root views and the heap
//! views it picks. So a replacement is written against the translation's
//! own register layout, which its disassembly shows (`victory translate
//! --list` with `VICTORY_EMIT_DUMP=<dir>`, then `dxc -dumpbin`), and does the
//! same work, only faster or more exactly than the generic translation of
//! the console program (no wave64 emulation, no lane-mask bookkeeping, the
//! compiler's own scheduling). The container must be a validated (signed)
//! DXIL container, as `dxc` writes it when its validator is found.
//!
//! A replacement that does not load as a DXIL container is ignored, with a
//! warning, and the translation runs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use victory_shader::emit::HostShader;

/// The overrides folder, if one is set.
fn folder() -> Option<&'static PathBuf> {
    static FOLDER: OnceLock<Option<PathBuf>> = OnceLock::new();
    FOLDER
        .get_or_init(|| {
            std::env::var_os("VICTORY_SHADER_OVERRIDES")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        })
        .as_ref()
}

/// Whether `bytes` look like a DXIL container: the `DXBC` magic, a size that
/// matches, and a `DXIL` part.
fn is_dxil_container(bytes: &[u8]) -> bool {
    if bytes.len() < 32 || &bytes[..4] != b"DXBC" {
        return false;
    }
    let size = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]) as usize;
    if size != bytes.len() {
        return false;
    }
    let parts = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]) as usize;
    (0..parts).any(|i| {
        let at = 32 + 4 * i;
        bytes
            .get(at..at + 4)
            .map(|o| u32::from_le_bytes([o[0], o[1], o[2], o[3]]) as usize)
            .and_then(|o| bytes.get(o..o + 4))
            .is_some_and(|fourcc| fourcc == b"DXIL")
    })
}

/// The replacement for the shader `hash` in `dir`, if there is a usable one.
fn load(dir: &std::path::Path, hash: &str) -> Option<Vec<u8>> {
    let path = dir.join(format!("{hash}.dxil"));
    let bytes = std::fs::read(&path).ok()?;
    if is_dxil_container(&bytes) {
        tracing::warn!(%hash, path = %path.display(), bytes = bytes.len(), "shader override: the replacement runs");
        Some(bytes)
    } else {
        tracing::warn!(%hash, path = %path.display(), "shader override is not a DXIL container; the translation runs");
        None
    }
}

/// `shader`, the translation of the console container `container`, with its
/// code replaced if the overrides folder holds a replacement for it.
pub(crate) fn apply(container: &[u8], mut shader: HostShader) -> HostShader {
    let Some(dir) = folder() else {
        return shader;
    };
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    // Looked up once per hash: a run binds the same shaders over and over.
    static SEEN: Mutex<Option<HashMap<String, Option<Vec<u8>>>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let entry = seen.get_or_insert_with(HashMap::new).entry(hash.clone()).or_insert_with(|| load(dir, &hash));
    if let Some(dxil) = entry {
        shader.dxil = dxil.clone();
    }
    shader
}

#[cfg(test)]
mod tests {
    use super::{is_dxil_container, load};

    fn container(parts: &[&[u8; 4]]) -> Vec<u8> {
        let header = 32 + 4 * parts.len();
        let mut v = b"DXBC".to_vec();
        v.extend([0u8; 16]); // hash
        v.extend(1u32.to_le_bytes()); // version
        let size_at = v.len();
        v.extend(0u32.to_le_bytes()); // size, set below
        v.extend((parts.len() as u32).to_le_bytes());
        for i in 0..parts.len() {
            v.extend(((header + 8 * i) as u32).to_le_bytes());
        }
        for p in parts {
            v.extend_from_slice(&p[..]);
            v.extend(0u32.to_le_bytes());
        }
        let n = v.len() as u32;
        v[size_at..size_at + 4].copy_from_slice(&n.to_le_bytes());
        v
    }

    #[test]
    fn a_container_with_a_dxil_part_is_taken() {
        assert!(is_dxil_container(&container(&[b"SFI0", b"DXIL"])));
    }

    #[test]
    fn anything_else_is_refused() {
        assert!(!is_dxil_container(&container(&[b"SFI0", b"SHEX"])));
        assert!(!is_dxil_container(b"not a container at all, not even close"));
        let mut short = container(&[b"DXIL"]);
        short.pop();
        assert!(!is_dxil_container(&short));
    }

    #[test]
    fn a_replacement_is_found_by_hash_and_checked() {
        let dir = std::env::temp_dir().join(format!("victory-overrides-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = container(&[b"SFI0", b"DXIL"]);
        std::fs::write(dir.join("0123456789abcdef.dxil"), &good).unwrap();
        std::fs::write(dir.join("fedcba9876543210.dxil"), b"DXBC but nothing more").unwrap();
        assert_eq!(load(&dir, "0123456789abcdef"), Some(good));
        // Not a container: the translation runs.
        assert_eq!(load(&dir, "fedcba9876543210"), None);
        // No file for that hash.
        assert_eq!(load(&dir, "1111111111111111"), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
