//! Partial thread-group masking for console `DispatchX` with
//! `COMPUTE_DISPATCH_INITIATOR::USE_THREAD_DIMENSIONS` (`flags & 0x20`).
//!
//! Kept as a child module of [`super`] so it shares `Lower`'s state without
//! adding bulk to `lower.rs`.

use super::super::bitcode::{Bin, Term, V, pred};
use super::super::shader::USER_DISPATCH_THREADS;
use super::{Lower, Val, Why};
use crate::native::RootType;

impl<'a> Lower<'a> {
    /// Guard for a memory or side-effecting instruction: `guard(b, i)` AND
    /// `partial_active` when partial-thread-dimension masking stays in
    /// predicated mode (for group barriers or wave64-over-wave32 exchange).
    pub(super) fn effect_guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        let g = self.guard(b, i)?;
        match (g, self.partial_active) {
            (Some(g), Some(a)) => Ok(Some(self.s.m.bin(self.cur, Bin::And, g, a))),
            (g, a) => Ok(g.or(a)),
        }
    }

    /// Partial thread-group masking for `DispatchX` with
    /// `USE_THREAD_DIMENSIONS` (`flags & 0x20`): when the host passes a
    /// non-zero thread count along an axis whose group size exceeds 1,
    /// lanes at or past that count behave as inactive in initial hardware
    /// `EXEC`.
    ///
    /// When the program has neither a group barrier nor wave64-over-wave32
    /// LDS exchange (and block 0 has no back-edge), inactive lanes return
    /// immediately after entry inputs are computed, so hardware `EXEC`,
    /// wave ballots and wave reductions see the exact active lane set with
    /// zero per-instruction guard overhead. Otherwise every thread stays
    /// alive for group barriers and `partial_active` predicates memory
    /// accesses, wave ballots and wave reductions.
    pub(super) fn dispatch_threads_prologue(&mut self, order: &[usize]) {
        let Some(threads) = self.s.threads else {
            return;
        };
        let e = self.s.entry;
        let zero = self.s.m.i32(0);
        let mut active: Option<V> = None;
        for (c, &dim) in threads.iter().enumerate() {
            if dim <= 1 {
                continue;
            }
            let limit = self.host_word(RootType::Other(USER_DISPATCH_THREADS), 0, c as u32);
            let Val::I32(tid) = self.thread_id(c as u8) else {
                unreachable!()
            };
            let Val::I32(gid) = self.compute_id("groupId.i32", 94, c as u8) else {
                unreachable!()
            };
            let g = self.s.m.i32(dim);
            let base = self.s.m.bin(e, Bin::Mul, gid, g);
            let global = self.s.m.bin(e, Bin::Add, base, tid);
            let no_limit = self.s.m.cmp(e, pred::EQ, limit, zero);
            let in_range = self.s.m.cmp(e, pred::IULT, global, limit);
            let ok = self.s.m.bin(e, Bin::Or, no_limit, in_range);
            active = Some(match active {
                None => ok,
                Some(prev) => self.s.m.bin(e, Bin::And, prev, ok),
            });
        }
        let Some(active) = active else {
            return;
        };
        let has_barrier = self.x.p.blocks.iter().enumerate().any(|(b, blk)| {
            !self.no_barrier.contains(&b) && blk.insts.iter().any(|i| i.name() == "s_barrier")
        });
        let can_early_exit = self.exchange.is_none()
            && !has_barrier
            && self.x.p.blocks[order[0]].preds.is_empty();
        if can_early_exit {
            let body = self.s.m.new_block();
            let exit = self.s.m.new_block();
            self.s.m.terminate(e, Term::CondBr(active, body, exit));
            self.s.m.terminate(exit, Term::Ret);
            self.body_entry = body;
            self.block[order[0]] = Some(body);
            self.cur = body;
        } else {
            self.partial_active = Some(active);
        }
    }
}
