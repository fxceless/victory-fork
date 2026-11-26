//! Capture the bindings of one emitted graphics command on its recording thread.

use super::*;
use crate::draw_trace::{self, Draw, Event};
use crate::pipeline::host::Stage;
use std::fmt::Write;

impl State {
    pub(super) fn trace_draw(&self, list: &HostList, operation: impl FnOnce() -> String) {
        if !draw_trace::enabled() {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return;
        };
        let Some(pixel) = draw_trace::pixel(p) else {
            return;
        };
        if !draw_trace::take_occurrence() {
            return;
        }

        let operation = operation();
        let targets: Vec<_> = self
            .targets
            .iter()
            .map(|&t| {
                crate::resource::host_made(t as This).map_or_else(
                    || "-".into(),
                    |r| {
                        // SAFETY: a live render target.
                        let d = unsafe { r.GetDesc() };
                        format!("{}x{}f{}", d.Width, d.Height, d.Format.0)
                    },
                )
            })
            .collect();
        let stages: Vec<_> = p.stages.iter().map(|s| s.shader.hash.as_str()).collect();
        let signature = format!(
            "{operation} stages={} targets={} topology={}",
            stages.join(","),
            targets.join(","),
            self.topology
        );
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let slots = heap.and_then(|h| h.host()).map(|h| h.kept.clone());
        let mut draw = Draw::new(signature, slots);
        draw.details = self.op_text(operation, false);
        for stage in &p.stages {
            self.trace_tables(stage, &mut draw);
        }
        // Preserve raw words too. A valid index in one run may become zero,
        // out of range, or refer to a buffer in the other run.
        let mut descriptions = std::collections::HashMap::new();
        let material = self.slots_text(pixel, |label, source, bytes, words| {
            let block = draw.word_block(source, bytes);
            for offset in 0..bytes / 4 {
                let index = words.get(offset).copied();
                let view = index
                    .and_then(|i| heap.and_then(|h| h.handle(i)))
                    .and_then(crate::descriptor::view);
                let description = match (index, view.as_ref()) {
                    (None, _) => "unreadable".into(),
                    (_, None) => "no-title-view".into(),
                    (Some(index), Some(v)) if crate::descriptor::host::is_texture_view(v) => {
                        descriptions
                            .entry(index)
                            .or_insert_with(|| format!("title-texture {}", view_state(Some(v))))
                            .clone()
                    }
                    _ => "other-title-view".into(),
                };
                draw.binding(
                    format!("word/{label}/{:#x}", offset * 4),
                    index,
                    index,
                    description,
                    Some((block, offset)),
                );
            }
        });
        draw.details.push_str(&material);
        let label = format!(
            "VIctory traced draw {} {}",
            pixel.shader.hash,
            draw.signature()
        );
        crate::capture::op(std::ptr::from_ref(list) as usize, || label.clone());
        let marker: Vec<u16> = label.encode_utf16().chain([0]).collect();
        // SAFETY: same Unicode marker representation as HostList::note.
        unsafe {
            list.list
                .SetMarker(0, Some(marker.as_ptr().cast()), 2 * marker.len() as u32)
        };
        list.trace_event(Event::Draw(std::sync::Arc::new(draw)));
    }

    fn trace_tables(&self, stage: &Stage, draw: &mut Draw) {
        let t = stage.translated();
        let mut entry = |slot: u8, offset: u32, role: String, sampler: bool| {
            let handle = match self.graphics.get(usize::from(slot)) {
                Some(Root::Table(handle)) => Some(*handle),
                _ => None,
            };
            let heap = handle.and_then(|h| self.heap_of(h));
            let at = handle.and_then(|h| h.checked_add(u64::from(offset)));
            let view = at.and_then(crate::descriptor::view);
            let title = heap.zip(at).and_then(|(h, a)| h.index(a));
            let name = format!("{}/{role}", stage.shader.hash);
            if sampler {
                let _ = writeln!(draw.details, "\n    {name} sampler {title:?} {view:?}");
                return;
            }
            let base = self
                .trace_bases
                .iter()
                .find(|&&(s, root, _)| s == t.host.stage && root == slot);
            let bytes = base.map_or_else(
                || self.user_value(false, slot, RootType::Table, 0),
                |item| item.2,
            );
            let host = Some(bytes.wrapping_add(offset) / 32);
            draw.binding(name, title, host, view_state(view.as_ref()), None);
        };
        for u in t.used.iter().filter(|u| u.root == RootType::Table) {
            entry(
                u.slot,
                u.table_offset_dwords * 4,
                format!("table/{}/{}/{:?}", u.slot, u.table_offset_dwords, u.kind),
                u.kind == ResourceType::Sampler,
            );
        }
        for idx in &t.host.root_indexed_views {
            let constant = self.user_value(
                false,
                idx.constant_slot,
                RootType::Constants,
                idx.constant_dword,
            );
            let offset = idx.byte_offset.wrapping_add(constant << idx.shift);
            let role = format!(
                "indexed/{}/{}/{}/{}/{}",
                idx.table_slot, idx.constant_slot, idx.constant_dword, idx.byte_offset, idx.shift
            );
            entry(idx.table_slot, offset, role, false);
        }
    }
}
