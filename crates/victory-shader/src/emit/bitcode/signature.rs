use super::{Const, Module, Op, TypeDef, V};

impl Module {
    /// Update the signature IDs in the mesh stores or pixel input reads made
    /// by the lowerer. A surviving pixel read of a removed element is an error.
    pub(crate) fn prune_signature(&mut self, remap: &[Option<u32>], mesh: bool) -> Option<()> {
        let mut edits = Vec::new();
        let mut removed = std::collections::HashSet::new();
        for block in &self.blocks {
            for &i in &block.body {
                let inst = &self.insts[i as usize];
                let Op::Call(function, args) = &inst.op else {
                    continue;
                };
                let name = &self.funcs[*function as usize].name;
                let signature_call = if mesh {
                    name.starts_with("dx.op.storeVertexOutput.")
                } else {
                    ["dx.op.loadInput.", "dx.op.attributeAtVertex."]
                        .iter()
                        .any(|prefix| name.starts_with(prefix))
                };
                if !signature_call {
                    continue;
                }
                let V::Const(sig) = *args.get(1)? else {
                    return None;
                };
                let Const::Int(sig) = self.consts[sig as usize].1 else {
                    return None;
                };
                match *remap.get(usize::try_from(sig).ok()?)? {
                    Some(new) if i64::from(new) != sig => edits.push((i, new)),
                    Some(_) => {}
                    None if mesh && matches!(self.types[inst.ty.0 as usize], TypeDef::Void) => {
                        removed.insert(i);
                    }
                    None => return None,
                }
            }
        }
        for (i, new) in edits {
            let new = self.i32(new);
            let Op::Call(_, args) = &mut self.insts[i as usize].op else {
                unreachable!()
            };
            args[1] = new;
        }
        for block in &mut self.blocks {
            block.body.retain(|i| !removed.contains(i));
        }
        Some(())
    }
}
