//! Which heap slots a program stores through when it picks buffer views
//! from the view heap by an index it computes (SM 6.6 dynamic resources).
//!
//! The title's skinning programs (`bvh_skinning`: CSComputeSkinning*,
//! cable, rope, tree animation; `CSUnpackIB*`) write their vertex and
//! index streams through unordered access views they pick from the heap.
//! No table or root view of the dispatch names those views, so the host
//! cannot tell from the binding what the dispatch writes. Their index is
//! simple, though, and can be worked out when the dispatch is bound:
//!
//! - `CSUnpackIB*`: a root constant, times 32 (one descriptor);
//! - the others: a 24-bit field of a job record, times 32. The records are
//!   a structured buffer in a table (`t117`, 176 bytes each); the dispatch
//!   runs jobs `[c0, c0 + c1)` of it (`c0`, `c1`: root constants), each
//!   thread searching them for its own (`s_add_i32 counter, c0` in a loop
//!   that ends at `s_cmp_ge_u32 counter + 1, c1`).
//!
//! [`heap_picks`] reads that recipe out of the program's SSA form: per
//! buffer store whose descriptor is loaded from the heap, the shift, the
//! mask and where the index comes from ([`HeapPick`]). The host evaluates
//! it with the dispatch's root arguments and the title memory the records
//! lie in, and declares the views it finds written.

use std::collections::HashMap;

use crate::RootType;
use crate::gfx10::{Fields, Operand, Valu};
use crate::ir::{Program, Reg, Value};
use crate::native::RootParam;
use crate::resource::{Resources, Role, Site, Source};

/// A root parameter's dword: layout index and dword within the parameter.
pub type ParamDword = (usize, u32);

/// How a program finds the heap slot a buffer store writes through: the
/// descriptor is read from the heap at byte `offset + ((x & mask) << shift)`
/// for each value `x` of `source`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HeapPick {
    pub offset: u32,
    pub shift: u32,
    pub mask: u32,
    pub source: PickSource,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PickSource {
    /// None: the slot at `offset` itself.
    Fixed,
    /// A root constant.
    Constant(ParamDword),
    /// The dword at byte `field` of records of the buffer view at byte
    /// `entry` of the table parameter `table` (layout index; the view's
    /// address is the table's plus `entry`), the record size being the
    /// view's element size. Records `[first, first + count)` when the
    /// program bounds them by root constants, else all of the view's.
    Records {
        table: usize,
        entry: u32,
        field: u32,
        first: Option<ParamDword>,
        count: Option<ParamDword>,
    },
}

/// The heap picks of `p`'s buffer stores (`store` says which instructions
/// store), one per distinct recipe; `None` if a store's descriptor comes
/// from the heap in a way this does not follow (the caller then has to
/// assume the store may write any view of the heap). An empty list: no
/// buffer store picks from the heap.
pub fn heap_picks(
    p: &Program,
    layout: &[RootParam],
    res: &Resources,
    store: impl Fn(&crate::ir::Inst) -> bool,
) -> Option<Vec<HeapPick>> {
    let walk = Walk::new(p, layout, res);
    let mut out: Vec<HeapPick> = Vec::new();
    for d in &res.descriptors {
        if d.role != Role::Resource {
            continue;
        }
        let inst = p.blocks.get(d.site.0)?.insts.get(d.site.1)?;
        if !store(inst) {
            continue;
        }
        for leaf in d.source.leaves() {
            let load = match leaf {
                Source::Loaded { load, .. } | Source::Fetched { load, .. } => load,
                _ => continue,
            };
            let Some(address) = walk.scalar.get(load) else {
                continue;
            };
            if layout.get(address.param).is_none_or(|rp| rp.root != RootType::ResourceHeap) {
                continue;
            }
            let pick = match address.dynamic {
                None => HeapPick { offset: address.offset, shift: 0, mask: u32::MAX, source: PickSource::Fixed },
                Some(v) => walk.pick(v, address.offset)?,
            };
            if !out.contains(&pick) {
                out.push(pick);
            }
        }
    }
    Some(out)
}

struct Walk<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    res: &'a Resources,
    defs: HashMap<Value, Site>,
    phis: HashMap<Value, Vec<Value>>,
    scalar: HashMap<Site, crate::resource::Address>,
}

/// An inline constant's value (operand codes 128-208).
fn inline(code: u8) -> Option<i32> {
    match code {
        128..=192 => Some(i32::from(code) - 128),
        193..=208 => Some(192 - i32::from(code)),
        _ => None,
    }
}

impl<'a> Walk<'a> {
    fn new(p: &'a Program, layout: &'a [RootParam], res: &'a Resources) -> Self {
        let mut defs = HashMap::new();
        let mut phis = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                phis.insert(phi.value, phi.args.clone());
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for w in &inst.writes {
                    defs.insert(w.1, (b, i));
                }
            }
        }
        let scalar = res.scalar.iter().map(|(s, a)| (*s, *a)).collect();
        Walk { p, layout, res, defs, phis, scalar }
    }

    fn inst(&self, v: Value) -> Option<(&'a crate::ir::Inst, Site)> {
        let &(b, i) = self.defs.get(&v)?;
        Some((self.p.blocks.get(b)?.insts.get(i)?, (b, i)))
    }

    /// The value an instruction reads for `operand` (a single register).
    fn operand(inst: &crate::ir::Inst, operand: &Operand) -> Option<Value> {
        match *operand {
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            Operand::Scalar { code, .. } => inst.read(Reg::S(code)),
            _ => None,
        }
    }

    /// A constant operand's value (inline or literal).
    fn constant(operand: &Operand) -> Option<u32> {
        match *operand {
            Operand::Const(c) => inline(c).map(|x| x as u32),
            Operand::Literal(x) => Some(x),
            _ => None,
        }
    }

    /// The root parameter dword `v` holds on entry, if it is one of a
    /// root constant.
    fn root_constant(&self, v: Value) -> Option<ParamDword> {
        let &(_, param, dword) = self.res.params.iter().find(|x| x.0 == v)?;
        (self.layout.get(param)?.root == RootType::Constants).then_some((param, dword))
    }

    /// `v` through moves.
    fn moved(&self, mut v: Value) -> Value {
        for _ in 0..16 {
            let Some((inst, _)) = self.inst(v) else {
                return v;
            };
            if !matches!(inst.name(), "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32") || inst.writes.first().map(|w| w.1) != Some(v) {
                return v;
            }
            let src = match &inst.fields {
                Fields::Valu(valu) => valu.src.first(),
                Fields::Sop { src, .. } => src.first(),
                _ => None,
            };
            match src.and_then(|s| Self::operand(inst, s)) {
                Some(next) => v = next,
                None => return v,
            }
        }
        v
    }

    /// The recipe of heap byte offset `v` (plus `offset`).
    fn pick(&self, v: Value, offset: u32) -> Option<HeapPick> {
        let (mut shift, mut mask) = (0u32, u32::MAX);
        let mut v = self.moved(v);
        for _ in 0..8 {
            if let Some(param) = self.root_constant(v) {
                return Some(HeapPick { offset, shift, mask, source: PickSource::Constant(param) });
            }
            let (inst, _) = self.inst(v)?;
            if inst.writes.first().map(|w| w.1) != Some(v) && !inst.name().starts_with("buffer_load") {
                return None;
            }
            match (inst.name(), &inst.fields) {
                ("v_lshlrev_b32", Fields::Valu(valu)) if mask == u32::MAX && shift == 0 => {
                    shift = Self::constant(valu.src.first()?)?;
                    v = self.moved(Self::operand(inst, valu.src.get(1)?)?);
                }
                ("s_lshl_b32", Fields::Sop { src, .. }) if mask == u32::MAX && shift == 0 => {
                    shift = Self::constant(src.get(1)?)?;
                    v = self.moved(Self::operand(inst, src.first()?)?);
                }
                ("v_and_b32" | "s_and_b32", Fields::Valu(Valu { src, .. }) | Fields::Sop { src, .. }) if mask == u32::MAX => {
                    let (a, b) = (src.first()?, src.get(1)?);
                    let (m, x) = match (Self::constant(a), Self::constant(b)) {
                        (Some(m), None) => (m, b),
                        (None, Some(m)) => (m, a),
                        _ => return None,
                    };
                    mask = m;
                    v = self.moved(Self::operand(inst, x)?);
                }
                (name, Fields::Buffer { data: Some(Operand::Vector { reg: data, .. }), addr: Operand::Vector { reg: index, .. }, rsrc: Operand::Scalar { code: rsrc, .. }, soffset, offset: at, offen: false, idxen: true, .. })
                    if name.starts_with("buffer_load_dword") =>
                {
                    if Self::constant(soffset) != Some(0) {
                        return None;
                    }
                    let Some(&(Reg::V(r), _)) = inst.writes.iter().find(|w| w.1 == v) else {
                        return None;
                    };
                    let field = u32::from(*at) + 4 * u32::from(r.checked_sub(*data)?);
                    // The descriptor: loaded from a table.
                    let desc = inst.read(Reg::S(*rsrc))?;
                    let (load, load_site) = self.inst(desc)?;
                    if !load.name().starts_with("s_load_dwordx") {
                        return None;
                    }
                    let address = self.scalar.get(&load_site)?;
                    if address.dynamic.is_some() || self.layout.get(address.param)?.root != RootType::Table {
                        return None;
                    }
                    let (first, count) = inst.read(Reg::V(*index)).map_or((None, None), |i| self.record_range(i));
                    return Some(HeapPick {
                        offset,
                        shift,
                        mask,
                        source: PickSource::Records { table: address.param, entry: address.offset, field, first, count },
                    });
                }
                _ => return None,
            }
        }
        None
    }

    /// Of a record index: the root constants bounding it, if it is
    /// `counter + c0` in a loop over `counter` from 0 that ends at
    /// `counter + 1 >= c1` (constants, the "not found" sentinel, aside).
    fn record_range(&self, index: Value) -> (Option<ParamDword>, Option<ParamDword>) {
        let mut leaves = Vec::new();
        let mut todo = vec![index];
        let mut seen = Vec::new();
        while let Some(v) = todo.pop() {
            if seen.contains(&v) || seen.len() > 64 {
                continue;
            }
            seen.push(v);
            let v = self.moved(v);
            if let Some(args) = self.phis.get(&v) {
                todo.extend(args.iter().copied());
                continue;
            }
            // `v_mov_b32 vN, -1`: no record.
            if let Some((inst, _)) = self.inst(v)
                && matches!(inst.name(), "v_mov_b32" | "s_mov_b32")
                && let Fields::Valu(Valu { src, .. }) | Fields::Sop { src, .. } = &inst.fields
                && src.first().and_then(Self::constant).is_some()
            {
                continue;
            }
            leaves.push(v);
        }
        let mut range: Option<(ParamDword, Option<ParamDword>)> = None;
        for leaf in leaves {
            let Some(found) = self.counted(leaf) else {
                return (None, None);
            };
            if range.is_some_and(|r| r != found) {
                return (None, None);
            }
            range = Some(found);
        }
        match range {
            Some((first, count)) => (Some(first), count),
            None => (None, None),
        }
    }

    /// `leaf` = `s_add_i32 counter, c0` with `counter` a loop counter from 0
    /// by 1: `c0`, and `c1` if the loop's compare is found.
    fn counted(&self, leaf: Value) -> Option<(ParamDword, Option<ParamDword>)> {
        let (inst, _) = self.inst(leaf)?;
        if inst.name() != "s_add_i32" && inst.name() != "s_add_u32" {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let a = Self::operand(inst, src.first()?)?;
        let b = Self::operand(inst, src.get(1)?)?;
        let (first, counter) = match (self.root_constant(a), self.root_constant(b)) {
            (Some(c), None) => (c, b),
            (None, Some(c)) => (c, a),
            _ => return None,
        };
        // counter = phi(0, counter + 1).
        let args = self.phis.get(&counter)?;
        let mut next = None;
        for &arg in args {
            let (def, _) = self.inst(arg)?;
            let Fields::Sop { src, .. } = &def.fields else {
                return None;
            };
            match def.name() {
                "s_mov_b32" if src.first().and_then(Self::constant) == Some(0) => {}
                "s_add_i32" | "s_add_u32"
                    if src.iter().filter_map(Self::constant).eq([1]) && src.iter().any(|s| Self::operand(def, s) == Some(counter)) =>
                {
                    next = Some(arg);
                }
                _ => return None,
            }
        }
        let next = next?;
        // The compare that ends the loop: `counter + 1` against a constant.
        let count = self.p.blocks.iter().flat_map(|b| &b.insts).find_map(|i| {
            if !matches!(i.name(), "s_cmp_ge_u32" | "s_cmp_lt_u32" | "s_cmp_ge_i32" | "s_cmp_lt_i32") {
                return None;
            }
            let Fields::Sop { src, .. } = &i.fields else {
                return None;
            };
            let x = Self::operand(i, src.first()?)?;
            let y = Self::operand(i, src.get(1)?)?;
            (x == next).then(|| self.root_constant(y)).flatten()
        });
        Some((first, count))
    }
}

#[cfg(test)]
mod tests {
    use super::inline;

    #[test]
    fn inline_constants_decode() {
        assert_eq!(inline(128), Some(0));
        assert_eq!(inline(133), Some(5));
        assert_eq!(inline(192), Some(64));
        assert_eq!(inline(193), Some(-1));
        assert_eq!(inline(208), Some(-16));
        assert_eq!(inline(240), None);
    }
}
