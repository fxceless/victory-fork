//! Compare-exchange lowering for RDNA2 integer LDS operations.

use super::{Bin, Lower, Rmw, Term, V, pred};

impl Lower<'_> {
    pub(super) fn lds_integer_update(&mut self, p: V, op: &str, v0: V, mask_value: Option<V>) -> V {
        self.lds_update(p, |l, old| {
            let c = l.cur;
            let zero = l.s.m.i32(0);
            let one = l.s.m.i32(1);
            match op {
                "inc_u32" => {
                    let wrap = l.s.m.cmp(c, pred::IUGE, old, v0);
                    let next = l.s.m.bin(c, Bin::Add, old, one);
                    l.s.m.select(c, wrap, zero, next)
                }
                "dec_u32" => {
                    let empty = l.s.m.cmp(c, pred::EQ, old, zero);
                    let above = l.s.m.cmp(c, pred::IUGT, old, v0);
                    let wrap = l.s.m.bin(c, Bin::Or, empty, above);
                    let next = l.s.m.bin(c, Bin::Sub, old, one);
                    l.s.m.select(c, wrap, v0, next)
                }
                "rsub_u32" => l.s.m.bin(c, Bin::Sub, v0, old),
                "mskor_b32" => {
                    let ones = l.s.m.i32(u32::MAX);
                    let clear = l.s.m.bin(c, Bin::Xor, v0, ones);
                    let keep = l.s.m.bin(c, Bin::And, old, clear);
                    l.s.m.bin(c, Bin::Or, keep, mask_value.unwrap())
                }
                _ => unreachable!(),
            }
        })
    }

    /// Atomically update LDS, returning the value from the successful CAS.
    fn lds_update(&mut self, p: V, update: impl FnOnce(&mut Self, V) -> V) -> V {
        let pre = self.cur;
        // An atomic zero-add supplies the first snapshot even under contention.
        let zero = self.s.m.i32(0);
        let init = self.s.m.atomic(pre, Rmw::Add, p, zero);
        let (lp, exit) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(pre, Term::Br(lp));
        let old = self.s.m.phi(lp, self.s.t.i32);
        self.cur = lp;
        let new = update(self, old);
        let prev = self.s.m.cmpxchg(lp, p, old, new);
        let done = self.s.m.cmp(lp, pred::EQ, prev, old);
        self.s.m.set_phi(old, vec![(init, pre), (prev, lp)]);
        self.s.m.terminate(lp, Term::CondBr(done, exit, lp));
        self.cur = exit;
        old
    }
}
