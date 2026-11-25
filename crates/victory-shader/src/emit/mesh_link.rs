//! Mesh pipelines that can fit the output limit by dropping whole unread varyings.

use super::container::{MESH_OUTPUT_LIMIT, SigElement, mesh_output_bytes};
use super::shader::Shader;
use super::{HostShader, signature, translate_with_pruning};
use crate::dxbc::{Container, FourCC, u16_at, u32_at};
use crate::native::HwStage;

/// Returns replacement DXIL for an oversized mesh shader and its actual pixel
/// consumer. Both variants use the original translations' binding metadata.
pub fn prune_mesh_pair(
    mesh: (&[u8], &HostShader),
    pixel: (&[u8], &HostShader),
) -> Option<(Vec<u8>, Vec<u8>)> {
    if mesh.1.stage != HwStage::Ms || pixel.1.stage != HwStage::Ps {
        return None;
    }
    let (outputs, prims, max_vertices, max_primitives) = mesh_signature(&mesh.1.dxil)?;
    if mesh_output_bytes(&outputs, &prims, max_vertices, max_primitives) <= MESH_OUTPUT_LIMIT {
        return None;
    }
    let ps = Container::parse(&pixel.1.dxil).ok()?;
    let inputs = input_usage(ps.part(FourCC::ISG1)?.data)?;
    let removed = prune_plan(&outputs, &inputs, &prims, max_vertices, max_primitives)?;
    Some((recompile(mesh, &removed)?, recompile(pixel, &removed)?))
}

fn recompile(
    (container, baseline): (&[u8], &HostShader),
    removed: &[(String, u32)],
) -> Option<Vec<u8>> {
    let mut variant = translate_with_pruning(container, removed).ok()?;
    let dxil = std::mem::replace(&mut variant.dxil, baseline.dxil.clone());
    (variant == *baseline).then_some(dxil)
}

fn mesh_signature(dxil: &[u8]) -> Option<(Vec<SigElement>, Vec<SigElement>, u32, u32)> {
    let c = Container::parse(dxil).ok()?;
    let psv = c.part(FourCC::PSV0)?.data;
    // Our emitted PSV0: runtime block size, then the mesh's maxima and stage.
    if u32_at(psv, 0).ok()? < 36 || *psv.get(28)? != 13 {
        return None;
    }
    let vertices = u32::from(u16_at(psv, 16).ok()?);
    let primitives = u32::from(u16_at(psv, 18).ok()?);
    if !(1..=256).contains(&vertices) || primitives > 256 {
        return None;
    }
    let outputs = signature(c.part(FourCC::OSG1)?.data).ok()?;
    let prims = signature(c.part(FourCC::PSG1)?.data).ok()?;
    Some((outputs, prims, vertices, primitives))
}

fn input_usage(data: &[u8]) -> Option<Vec<SigElement>> {
    let mut elements = signature(data).ok()?;
    let records = data.get(u32_at(data, 4).ok()? as usize..)?;
    for (element, record) in elements.iter_mut().zip(records.chunks_exact(32)) {
        let usage = record[25];
        if usage & !record[24] != 0 {
            return None;
        }
        element.usage = usage >> element.col;
    }
    Some(elements)
}

fn prune_plan(
    outputs: &[SigElement],
    inputs: &[SigElement],
    prims: &[SigElement],
    vertices: u32,
    primitives: u32,
) -> Option<Vec<(String, u32)>> {
    let mut bytes = mesh_output_bytes(outputs, prims, vertices, primitives);
    let bytes_per_column = 4 * u64::from(vertices).next_multiple_of(32);
    let mut removed = Vec::new();
    for output in outputs.iter().filter(|e| e.kind == 0 && e.row.is_some()) {
        let key = |e: &&SigElement| {
            e.kind == 0 && e.index == output.index && e.name.eq_ignore_ascii_case(&output.name)
        };
        let mut matches = inputs.iter().filter(key);
        let Some(input) = matches.next() else {
            continue;
        };
        if matches.next().is_some()
            || outputs.iter().filter(key).count() != 1
            || prims.iter().any(|e| key(&e))
            || input.usage != 0
            || (output.row, output.col, output.cols, output.comp)
                != (input.row, input.col, input.cols, input.comp)
        {
            continue;
        }
        removed.push((output.name.clone(), output.index));
        bytes -= u64::from(output.cols) * bytes_per_column;
        if bytes <= MESH_OUTPUT_LIMIT {
            return Some(removed);
        }
    }
    None
}

pub(super) fn prune(shader: &mut Shader, removed: &[(String, u32)]) -> Option<()> {
    let mesh = shader.mesh.is_some();
    let elements = if mesh {
        &mut shader.outputs
    } else if shader.pixel {
        &mut shader.inputs
    } else {
        return None;
    };
    let mut next = 0;
    let mut dropped = 0;
    let mut remap = Vec::with_capacity(elements.len());
    for element in elements.iter() {
        let remove = element.kind == 0
            && removed.iter().any(|(name, index)| {
                element.name.eq_ignore_ascii_case(name) && element.index == *index
            });
        if remove {
            if !mesh && element.usage != 0 {
                return None;
            }
            remap.push(None);
            dropped += 1;
        } else {
            remap.push(Some(next));
            next += 1;
        }
    }
    if dropped != removed.len() {
        return None;
    }
    shader.m.prune_signature(&remap, mesh)?;
    let mut ids = remap.into_iter();
    elements.retain(|_| ids.next().unwrap().is_some());
    if let Some(mesh) = shader.mesh
        && mesh_output_bytes(
            elements,
            &shader.prims,
            mesh.max_vertices,
            mesh.max_primitives,
        ) > MESH_OUTPUT_LIMIT
    {
        return None;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_only_a_unique_whole_unread_varying_that_fits_the_pair() {
        let element = |row| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let outputs: Vec<_> = (0..17).map(element).collect();
        let mut inputs = outputs.clone();
        assert_eq!(prune_plan(&outputs, &inputs, &[], 128, 128), None);

        inputs[5].usage = 0;
        assert_eq!(
            prune_plan(&outputs, &inputs, &[], 128, 128),
            Some(vec![("TEXCOORD".into(), 5)]),
        );
        inputs[5].row = Some(6);
        assert_eq!(prune_plan(&outputs, &inputs, &[], 128, 128), None);
        inputs[5].row = Some(5);
        inputs.push(inputs[5].clone());
        assert_eq!(prune_plan(&outputs, &inputs, &[], 128, 128), None);
    }
}
