//! Offline: for each shader container given (as `victory translate`
//! dumps them with `VICTORY_DUMP_MATCH`), the heap slots its buffer stores
//! pick (`victory_shader::picked`), one line per recipe, or "NOT FOLLOWED".
//! With `--chains` instead: each such store and the instructions its heap
//! index is made of.
//!
//! ```text
//! cargo run --release -p victory-shader --example picked_index -- [--chains] <file.cdxil>...
//! ```

use std::collections::HashMap;
use victory_shader::ir::{Program, Reg, Value};
use victory_shader::resource::{Role, Source};
use victory_shader::{Container, FourCC, RootType};

fn defs(p: &Program) -> HashMap<Value, (usize, usize)> {
    let mut m = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            for w in &inst.writes {
                m.insert(w.1, (b, i));
            }
        }
    }
    m
}

fn show(p: &Program, d: &HashMap<Value, (usize, usize)>, v: Value, depth: usize, out: &mut String, seen: &mut Vec<Value>) {
    let pad = "  ".repeat(depth + 2);
    if seen.contains(&v) || depth > 9 {
        out.push_str(&format!("{pad}%{} ...\n", v.0));
        return;
    }
    seen.push(v);
    if let Some(&(b, i)) = d.get(&v) {
        let inst = &p.blocks[b].insts[i];
        let reads: Vec<String> = inst.reads.iter().filter(|r| r.0 != Reg::S(126) && r.0 != Reg::S(127)).map(|r| format!("{:?}=%{}", r.0, r.1.0)).collect();
        out.push_str(&format!("{pad}%{} = {} @{:#x} [{}] {:?}\n", v.0, inst.name(), inst.machine.offset, reads.join(" "), inst.fields));
        for r in &inst.reads {
            if r.0 == Reg::S(126) || r.0 == Reg::S(127) {
                continue;
            }
            show(p, d, r.1, depth + 1, out, seen);
        }
    } else if let Some((reg, _)) = p.inputs.iter().find(|x| x.1 == v) {
        out.push_str(&format!("{pad}%{} = input {:?}\n", v.0, reg));
    } else {
        // a phi
        for (b, block) in p.blocks.iter().enumerate() {
            if let Some(phi) = block.phis.iter().find(|ph| ph.value == v) {
                out.push_str(&format!("{pad}%{} = phi b{} {:?}\n", v.0, b, phi.args));
                for a in phi.args.clone() {
                    show(p, d, a, depth + 1, out, seen);
                }
                return;
            }
        }
        out.push_str(&format!("{pad}%{} = ?\n", v.0));
    }
}

fn main() {
    let chains = std::env::args().any(|a| a == "--chains");
    for path in std::env::args().skip(1).filter(|a| a != "--chains") {
        let bytes = std::fs::read(&path).unwrap();
        let c = Container::parse(&bytes).unwrap();
        let Some(dxil) = c.part(FourCC::DXIL) else { continue };
        let programs = victory_shader::native_programs(dxil.data).unwrap();
        for np in programs {
            if !np.layout.iter().any(|r| r.root == RootType::ResourceHeap) {
                continue;
            }
            let Ok(p) = victory_shader::spill::build(&np) else { println!("{path}: no build"); continue };
            let st = victory_shader::structure::structure(&p).ok();
            let res = victory_shader::resource::track(&p, &np.layout, np.stage, None)
                .or_else(|_| victory_shader::resource::track(&p, &np.layout, np.stage, st.as_ref()));
            let Ok(res) = res else { println!("{path}: no track"); continue };
            if !chains {
                let picks = victory_shader::picked::heap_picks(&p, &np.layout, &res, |i| {
                    let n = i.name();
                    (n.starts_with("buffer_") || n.starts_with("tbuffer_")) && (n.contains("store") || n.contains("atomic"))
                });
                let name = path.rsplit('/').next().unwrap();
                match picks {
                    Some(v) => {
                        for pk in v {
                            let desc = |pd: Option<(usize, u32)>| pd.map(|(i, d)| format!("slot{} dw{}", np.layout[i].slot, d));
                            match &pk.source {
                                victory_shader::picked::PickSource::Records { table, entry, field, first, count } => println!(
                                    "{name}: off {} shift {} mask {:#x} records table slot {} (offs {}) entry {} field {} first {:?} count {:?}",
                                    pk.offset, pk.shift, pk.mask, np.layout[*table].slot, np.layout[*table].table_offset_dwords, entry, field, desc(*first), desc(*count)
                                ),
                                other => println!("{name}: off {} shift {} mask {:#x} {:?}", pk.offset, pk.shift, pk.mask, other),
                            }
                        }
                    }
                    None => println!("{name}: NOT FOLLOWED"),
                }
                continue;
            }
            let scalar: HashMap<_, _> = res.scalar.iter().map(|(s, a)| (*s, *a)).collect();
            let d = defs(&p);
            let mut out = String::new();
            out.push_str(&format!("== {} stage {:?}\n", path.rsplit('/').next().unwrap(), np.stage));
            for (i, rp) in np.layout.iter().enumerate() {
                out.push_str(&format!("  param {i}: {:?} sgprs {} user_data {} slot {} base_reg {}\n", rp.root, rp.sgprs, rp.user_data, rp.slot, rp.base_register));
            }
            for desc in &res.descriptors {
                if desc.role != Role::Resource {
                    continue;
                }
                let inst = &p.blocks[desc.site.0].insts[desc.site.1];
                let n = inst.name();
                if !(n.starts_with("buffer_") || n.starts_with("tbuffer_")) || !(n.contains("store") || n.contains("atomic")) {
                    continue;
                }
                out.push_str(&format!(" store {} @{:#x}\n", n, inst.machine.offset));
                for leaf in desc.source.leaves() {
                    match leaf {
                        Source::Loaded { load, first } | Source::Fetched { load, first } => {
                            let a = scalar.get(load);
                            let li = &p.blocks[load.0].insts[load.1];
                            out.push_str(&format!("   from {} @{:#x} first {} addr {:?} kind {}\n", li.name(), li.machine.offset, first, a, if matches!(leaf, Source::Fetched { .. }) { "per-lane" } else { "uniform" }));
                            if let Some(a) = a
                                && let Some(v) = a.dynamic
                            {
                                let mut seen = Vec::new();
                                show(&p, &d, v, 0, &mut out, &mut seen);
                            }
                        }
                        other => out.push_str(&format!("   other {other:?}\n")),
                    }
                }
            }
            print!("{out}");
        }
    }
}
