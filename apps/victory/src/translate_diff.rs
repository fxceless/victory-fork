//! `victory translate-diff`: two `translate --list` outputs (two builds over
//! the same shaders) compared program by program.
//!
//! A program is its container hash and stage (`#<16 hex> ~<8 hex> Ps`).
//! The comparison names the programs one list validates and the other
//! does not, grouped by the other list's reason, and, where both lists
//! carry `--hashes`, the programs both validate whose DXIL differs.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};

/// What one list says of one program.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Outcome {
    validated: bool,
    /// The first reason it stopped for, if it did.
    failed: Option<String>,
    /// `dxil=` of a `--hashes` list.
    dxil: Option<String>,
}

/// The program a list line names (`#<container> <stage>`), its DXIL hash
/// if the line has one, and the outcome after the two-space separator.
/// `None` for lines that name no program (summaries, duplicates).
fn parse_line(line: &str) -> Option<(String, Option<String>, &str)> {
    // The outcome follows the first double space after the stage.
    let mut words = line.split(' ');
    let container = words
        .by_ref()
        .find(|w| w.len() == 17 && w.starts_with('#') && w[1..].bytes().all(|b| b.is_ascii_hexdigit()))?;
    let code = words.next()?;
    if !code.starts_with('~') {
        return None;
    }
    let stage = words.next()?;
    let stage_ok = stage.len() == 2
        && stage.as_bytes()[0].is_ascii_uppercase()
        && stage.as_bytes()[1].is_ascii_lowercase();
    if !stage_ok {
        return None;
    }
    let at = line.find(&format!("{container} {code} {stage}"))? + container.len() + code.len() + stage.len() + 2;
    let rest = &line[at..];
    let (label, outcome) = rest.split_once("  ")?;
    let dxil = label
        .split(' ')
        .find_map(|w| w.strip_prefix("dxil="))
        .map(str::to_owned);
    Some((format!("{container} {stage}"), dxil, outcome.trim()))
}

/// Every program of a list and what the list says of it.
fn parse(text: &str) -> BTreeMap<String, Outcome> {
    let mut out: BTreeMap<String, Outcome> = BTreeMap::new();
    for line in text.lines() {
        let Some((key, dxil, outcome)) = parse_line(line) else {
            continue;
        };
        let o = out.entry(key).or_default();
        if o.dxil.is_none() {
            o.dxil = dxil;
        }
        match outcome {
            "validated" => o.validated = true,
            // (a pipeline line comes with the program's other lines)
            "pipeline created" | "emitted" => {}
            reason => {
                if o.failed.is_none() {
                    o.failed = Some(reason.to_owned());
                }
            }
        }
    }
    out
}

/// The comparison of two lists.
#[derive(Debug, Default)]
struct Diff {
    old_programs: usize,
    new_programs: usize,
    old_validated: usize,
    new_validated: usize,
    /// Validated in the new list only: the old list's reason (numbers left
    /// out, [`shape`]) → programs.
    gained: BTreeMap<String, Vec<String>>,
    /// Validated in the old list only: the new list's reason → programs.
    lost: BTreeMap<String, Vec<String>>,
    /// Validated in both, with hashes in both, and those whose DXIL differs.
    hashed_both: usize,
    changed: Vec<String>,
    only_old: usize,
    only_new: usize,
}

/// A reason with its numbers left out, so that one kind of stop is one
/// group (the validator names a signature row's location, say).
fn shape(reason: &str) -> String {
    let mut out = String::with_capacity(reason.len());
    let mut in_number = false;
    for ch in reason.chars() {
        if ch.is_ascii_digit() {
            if !in_number {
                out.push('#');
            }
            in_number = true;
        } else {
            out.push(ch);
            in_number = false;
        }
    }
    out
}

fn diff(old: &BTreeMap<String, Outcome>, new: &BTreeMap<String, Outcome>) -> Diff {
    let mut d = Diff {
        old_programs: old.len(),
        new_programs: new.len(),
        old_validated: old.values().filter(|o| o.validated).count(),
        new_validated: new.values().filter(|o| o.validated).count(),
        ..Diff::default()
    };
    let reason = |o: Option<&Outcome>| match o {
        None => "not in the list".to_owned(),
        Some(o) => o.failed.as_deref().map_or_else(|| "not validated".to_owned(), shape),
    };
    for (key, n) in new {
        let o = old.get(key);
        if o.is_none() {
            d.only_new += 1;
        }
        if n.validated && !o.is_some_and(|o| o.validated) {
            d.gained.entry(reason(o)).or_default().push(key.clone());
        }
    }
    for (key, o) in old {
        let n = new.get(key);
        if n.is_none() {
            d.only_old += 1;
        }
        if o.validated && !n.is_some_and(|n| n.validated) {
            d.lost.entry(reason(n)).or_default().push(key.clone());
        }
        if let Some(n) = n
            && o.validated
            && n.validated
            && let (Some(a), Some(b)) = (&o.dxil, &n.dxil)
        {
            d.hashed_both += 1;
            if a != b {
                d.changed.push(key.clone());
            }
        }
    }
    d
}

fn print(d: &Diff, keys: bool) {
    println!("old          {} programs, {} validated", d.old_programs, d.old_validated);
    println!("new          {} programs, {} validated", d.new_programs, d.new_validated);
    if d.only_old + d.only_new > 0 {
        println!("             {} programs only in the old list, {} only in the new", d.only_old, d.only_new);
    }
    let groups = |title: &str, g: &BTreeMap<String, Vec<String>>| {
        let n: usize = g.values().map(Vec::len).sum();
        println!("{title} {n}");
        let mut by_size: Vec<_> = g.iter().collect();
        by_size.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));
        for (reason, programs) in by_size {
            println!("  {:>7}  {reason}", programs.len());
            if keys {
                for p in programs {
                    println!("             {p}");
                }
            }
        }
    };
    groups("gained       (validated in the new list only; the old list's reason)", &d.gained);
    groups("lost         (validated in the old list only; the new list's reason)", &d.lost);
    if d.hashed_both > 0 {
        println!(
            "DXIL changed {} of the {} programs both validate (both lists with --hashes)",
            d.changed.len(),
            d.hashed_both
        );
        if keys {
            for p in &d.changed {
                println!("             {p}");
            }
        }
    } else {
        println!("DXIL changed not compared: run both lists with --hashes");
    }
}

pub fn run(old: &Path, new: &Path, keys: bool) -> Result<()> {
    let read = |p: &Path| std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()));
    let d = diff(&parse(&read(old)?), &parse(&read(new)?));
    print(&d, keys);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "#0123456789abcdef";
    const B: &str = "#fedcba9876543210";
    const C: &str = "#00000000000000aa";

    #[test]
    fn a_list_line_gives_its_program_hash_and_outcome() {
        let line = format!("pack.rpf@0x10 PS_Example_1 {A} ~deadbeef Ps dxil=1111222233334444  validated");
        let (key, dxil, outcome) = parse_line(&line).unwrap();
        assert_eq!(key, format!("{A} Ps"));
        assert_eq!(dxil.as_deref(), Some("1111222233334444"));
        assert_eq!(outcome, "validated");
        // A stop in an earlier stage: the address before the reason.
        let line = format!("pack.rpf@0x10 CS_X {B} ~0badf00d Cs at 0x1eb4  emit: a lane mask read as data");
        let (key, dxil, outcome) = parse_line(&line).unwrap();
        assert_eq!((key.as_str(), dxil, outcome), (format!("{B} Cs").as_str(), None, "emit: a lane mask read as data"));
        // A validator message keeps its own double spaces.
        let line = format!("p {C} ~12345678 Vs dxil=aaaabbbbccccdddd  validator: error: a  b");
        assert_eq!(parse_line(&line).unwrap().2, "validator: error: a  b");
        // Summaries and duplicates name no program.
        assert!(parse_line("programs     26084 translated of 26084").is_none());
        assert!(parse_line(&format!("pack.rpf@0x10 {A} ~deadbeef  duplicate")).is_none());
    }

    #[test]
    fn the_diff_groups_gains_and_losses_by_the_other_lists_reason() {
        let old = format!(
            "p {A} ~11111111 Ps dxil=aaaaaaaaaaaaaaaa  validated\n\
             p {B} ~22222222 Ps  emit: a descriptor read only as data\n\
             p {C} ~33333333 Cs dxil=cccccccccccccccc  validated\n"
        );
        let new = format!(
            "p {A} ~11111111 Ps dxil=aaaaaaaaaaaaaaaa  validated\n\
             p {B} ~22222222 Ps dxil=bbbbbbbbbbbbbbbb  validated\n\
             p {C} ~33333333 Cs dxil=dddddddddddddddd  validator: error: something\n"
        );
        let d = diff(&parse(&old), &parse(&new));
        assert_eq!((d.old_validated, d.new_validated), (2, 2));
        assert_eq!(d.gained["emit: a descriptor read only as data"], [format!("{B} Ps")]);
        assert_eq!(d.lost["validator: error: something"], [format!("{C} Cs")]);
        // Only A is validated in both, with the same DXIL.
        assert_eq!((d.hashed_both, d.changed.len()), (1, 0));
    }

    #[test]
    fn reasons_that_differ_in_numbers_are_one_group() {
        let old = format!(
            "p {A} ~11111111 Ps  validator: error: element at location (7,1) differs
             p {B} ~22222222 Ps  validator: error: element at location (9,12) differs
"
        );
        let new = format!("p {A} ~11111111 Ps  validated
p {B} ~22222222 Ps  validated
");
        let d = diff(&parse(&old), &parse(&new));
        assert_eq!(d.gained.len(), 1);
        assert_eq!(d.gained["validator: error: element at location (#,#) differs"].len(), 2);
    }

    #[test]
    fn a_program_both_validate_with_other_dxil_is_changed() {
        let old = format!("p {A} ~11111111 Ms dxil=aaaaaaaaaaaaaaaa  validated\np {A} ~11111111 Ms dxil=aaaaaaaaaaaaaaaa  pipeline created\n");
        let new = format!("p {A} ~11111111 Ms dxil=bbbbbbbbbbbbbbbb  validated\n");
        let d = diff(&parse(&old), &parse(&new));
        assert_eq!(d.changed, [format!("{A} Ms")]);
        assert!(d.gained.is_empty() && d.lost.is_empty());
        // Without hashes in one list nothing is compared.
        let plain = format!("p {A} ~11111111 Ms  validated\n");
        assert_eq!(diff(&parse(&old), &parse(&plain)).hashed_both, 0);
    }
}
