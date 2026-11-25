//! Opt-in omission of consumers and producers of an unbuilt ray-tracing BVH.

use std::ffi::CStr;

#[cfg(windows)]
pub(super) fn skip(container: &[u8]) -> bool {
    use std::sync::OnceLock;
    use victory_shader::{Container, FourCC};

    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        requested(
            std::env::var("VICTORY_BVH_PASSES").ok().as_deref(),
            std::env::var("VICTORY_BVH_BUILDER").ok().as_deref(),
        )
    });
    enabled
        && Container::parse(container).is_ok_and(|shader| {
            shader
                .part(FourCC::PSV0)
                .is_some_and(|psv| structure_pass(psv.data))
        })
}

fn requested(passes: Option<&str>, builder: Option<&str>) -> bool {
    passes == Some("0") && builder != Some("1")
}

fn structure_pass(psv: &[u8]) -> bool {
    let Some(name) = compute_entry(psv) else {
        return false;
    };
    // In this title the CSComputeSkinning family skins the BVH vertices.
    // Screen-probe fallback and voxel passes remain active without structures.
    name.starts_with("CSComputeSkinning")
        || name
            .strip_prefix("CS_ScreenProbes_TraceRays_")
            .is_some_and(|variant| {
                variant.split('_').any(|field| field == "BVH")
                    && !variant
                        .split('_')
                        .any(|field| matches!(field, "Fallback" | "Voxels"))
            })
}

// PSVRuntimeInfo3 adds EntryFunctionName at byte 48, an offset into the
// string table following the resource bindings. Older or malformed metadata
// leaves the pipeline enabled. Other strings in the table are irrelevant.
fn compute_entry(psv: &[u8]) -> Option<&str> {
    fn read_u32(bytes: &mut &[u8]) -> Option<usize> {
        let (word, rest) = bytes.split_first_chunk::<4>()?;
        *bytes = rest;
        Some(u32::from_le_bytes(*word) as usize)
    }

    let mut bytes = psv;
    let info_size = read_u32(&mut bytes)?;
    let (info, mut rest) = bytes.split_at_checked(info_size)?;
    if info.get(24) != Some(&5) {
        return None;
    }
    let offset = read_u32(&mut info.get(48..)?)?;
    let count = read_u32(&mut rest)?;
    if count > 0 {
        let stride = read_u32(&mut rest)?;
        if stride < 16 {
            return None;
        }
        rest = rest.get(count.checked_mul(stride)?..)?;
    }
    let size = read_u32(&mut rest)?;
    let table = rest.get(..size)?;
    CStr::from_bytes_until_nul(table.get(offset..)?)
        .ok()?
        .to_str()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn psv(stage: u8, entry: &str) -> Vec<u8> {
        let mut bytes = 52u32.to_le_bytes().to_vec();
        let mut info = [0u8; 52];
        info[24] = stage;
        info[48..52].copy_from_slice(&1u32.to_le_bytes());
        bytes.extend(info);
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(24u32.to_le_bytes());
        bytes.extend([0; 24]);
        let strings = format!("\0{entry}\0CSComputeSkinningUnused\0");
        bytes.extend((strings.len() as u32).to_le_bytes());
        bytes.extend(strings.bytes());
        bytes
    }

    #[test]
    fn only_explicit_opt_in_with_builder_off_skips_passes() {
        assert!(requested(Some("0"), None));
        assert!(requested(Some("0"), Some("0")));
        assert!(!requested(Some("0"), Some("1")));
        for value in [None, Some("1"), Some(""), Some("typo")] {
            assert!(!requested(value, None));
        }
    }

    #[test]
    fn skip_only_bvh_compute_entries() {
        for name in [
            "CSComputeSkinningDefaultVelocity",
            "CS_ScreenProbes_TraceRays_BVH_Example",
            "CS_ScreenProbes_TraceRays_RayBinning_BVH_Example",
        ] {
            assert!(structure_pass(&psv(5, name)));
            assert!(!structure_pass(&psv(0, name)));
            assert!(!structure_pass(&psv(13, name)));
        }
        for name in [
            "CS_Skinning",
            "CS_ScreenProbes_TraceRays_Fallback",
            "CS_ScreenProbes_TraceRays_Voxels",
            "CS_ScreenProbes_TraceRays_Fallback_BVH_Test",
            "CS_ScreenProbes_TraceRays_BVHish",
            "other_BVH_Use",
            "ExampleCSComputeSkinning",
        ] {
            assert!(!structure_pass(&psv(5, name)));
        }
        let fixture = psv(5, "CSComputeSkinningExample");
        for len in 0..fixture.len() {
            assert!(!structure_pass(&fixture[..len]));
        }
    }
}
