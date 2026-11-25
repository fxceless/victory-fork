//! The host's amplification shader (`VICTORY_MESH_AMPLIFY=1`): put in front of every
//! translated mesh shader, it launches the mesh groups of one draw. A direct dispatch's
//! group count comes in a root constant; an indirect record is read here, in the graphics
//! pipe, and clamped as `victory-gpu`'s `indirect.rs` clamps mesh records (65,535 groups at
//! most, one-dimensional, the list's budget of 2^27 threads with a mesh group as 128).
//!
//! Without it each indirect mesh record (the title draws its meshes one record a call,
//! some 3,500-4,600 a frame) was clamped by a compute pass of its own between two waits for
//! every earlier shader write: the card drained before and after every draw. Measured on an
//! RTX 5070 Ti (`mesh_bench_tests`), the title's cascade-shadow mesh draw of 1 to 16 groups
//! took 7-12 us that way and 0.3-0.6 us launched by this shader.
//!
//! As HLSL (the bindings are visible to the amplification stage only, so they may share
//! registers with the title's):
//!
//! ```hlsl
//! cbuffer Launch : register(b0) { uint flags; uint direct; uint stride; uint unused; };
//! ByteAddressBuffer records : register(t0, space1); // record 0 at byte 0
//! ByteAddressBuffer count : register(t1, space1);   // the record count at byte 0
//! RWByteAddressBuffer budget : register(u0, space1); // word 0 the budget, 1-2 notes
//! struct Payload { uint unused; };
//! groupshared Payload payload;
//! [numthreads(1, 1, 1)]
//! void main(uint3 id : SV_GroupID) {
//!     uint groups = 0;
//!     if ((flags & 1) == 0) groups = min(direct, 65535);
//!     else if ((flags & 2) == 0 || id.x < count.Load(0)) {
//!         uint3 g = records.Load3(id.x * stride);
//!         budget.InterlockedMax(4, g.x); budget.InterlockedAdd(8, min(g.x, 1 << 24));
//!         g = uint3(min(g.x, 65535), min(g.y, 1), min(g.z, 1));
//!         uint cost = g.x * g.y * g.z * 128;
//!         // the clamp's compare-exchange loop on word 0 against 2^27
//!         groups = over ? 0 : g.x * g.y * g.z;
//!     }
//!     DispatchMesh(groups, 1, 1, payload);
//! }
//! ```

use super::bitcode::{Attr, Bin, Cast, Term, pred};
use super::shader::Shader;

/// The payload bytes the amplification shader hands its mesh groups, which the mesh
/// shaders declare under the switch (`shader::mesh_payload`).
pub const MESH_PAYLOAD: u32 = 4;

/// The list's budget of threads (`victory-gpu` `indirect.rs`, `BUDGET`).
const BUDGET: u32 = 1 << 27;

/// Root constants of the amplification shader (`b0`, 4 dwords): bit 0 of the first, its
/// groups draw records (else the direct count, the second dword); bit 1, a count bounds
/// them; the third, the bytes from one record to the next.
pub const AMPLIFY_RECORDS: u32 = 1;
/// See [`AMPLIFY_RECORDS`].
pub const AMPLIFY_COUNTED: u32 = 2;
/// Dwords of its root constants.
pub const AMPLIFY_CONSTANTS: u32 = 4;

/// Whether mesh shaders are translated for the host's amplification shader in front
/// (`VICTORY_MESH_AMPLIFY=1`; see [`amplification_shader`]).
pub fn mesh_amplify() -> bool {
    super::shader::mesh_payload() != 0
}

/// The amplification shader as a DXIL container (`as_6_6`, validated by the tests). Its
/// root bindings: [`AMPLIFY_CONSTANTS`] constants at `b0` (space 0), the records' raw
/// buffer at `t0` and the count's at `t1`, the budget's (read-write) at `u0`, those three in
/// space 1; one AS thread group per record, of one thread.
pub fn amplification_shader() -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([1, 1, 1]);
    s.amplify = Some(MESH_PAYLOAD);
    s.user_data = AMPLIFY_CONSTANTS;
    s.flags |= super::shader::flags::UAVS_EVERY_STAGE;
    let i32t = s.t.i32;
    let k = |s: &mut Shader, v: u32| s.m.i32(v);
    let entry = s.entry;

    // The payload, never read: thread-group shared, as DXIL declares it.
    let payload_ty = s.m.named_struct("struct.Payload", vec![i32t]);
    let payload = s.m.global("payload", payload_ty, 3);

    let launch = s.user_data_handle();
    let records = s.root_srv_handle(0);
    let count = s.root_srv_handle(1);
    let budget = s.root_uav_handle(0);
    let zero = k(&mut s, 0);
    let group = s.op(entry, "groupId.i32", Attr::ReadNone, i32t, 94, &[zero]);
    let row = s.op(
        entry,
        "cbufferLoadLegacy.i32",
        Attr::ReadOnly,
        s.t.cbuf_ret,
        59,
        &[launch, zero],
    );
    let flags = s.m.extract(entry, row, 0);
    let direct = s.m.extract(entry, row, 1);
    let stride = s.m.extract(entry, row, 2);

    let direct_b = s.m.new_block();
    let counted_b = s.m.new_block();
    let record_b = s.m.new_block();
    let charge_b = s.m.new_block();
    let head_b = s.m.new_block();
    let swap_b = s.m.new_block();
    let won_b = s.m.new_block();
    let end_b = s.m.new_block();

    let umin = |s: &mut Shader, b: usize, a, x: u32| {
        let x = s.m.i32(x);
        s.op(b, "binary.i32", Attr::ReadNone, i32t, 40, &[a, x])
    };
    let iu = s.m.undef(i32t);
    let align = s.m.i32(4);
    // the first dword of a raw buffer load, as an integer
    let word = |s: &mut Shader, b: usize, ret, n: u32| {
        let v = s.m.extract(b, ret, n);
        s.m.cast(b, Cast::Bitcast, v, i32t)
    };

    // entry: direct or records
    let one = k(&mut s, AMPLIFY_RECORDS);
    let bit = s.m.bin(entry, Bin::And, flags, one);
    let is_direct = s.m.cmp(entry, pred::EQ, bit, zero);
    s.m.terminate(entry, Term::CondBr(is_direct, direct_b, counted_b));

    // direct: the count, at most what one dimension runs
    let direct_groups = umin(&mut s, direct_b, direct, 65535);
    s.m.terminate(direct_b, Term::Br(end_b));

    // counted: groups at or past the record count draw nothing
    let two = k(&mut s, AMPLIFY_COUNTED);
    let bit = s.m.bin(counted_b, Bin::And, flags, two);
    let uncounted = s.m.cmp(counted_b, pred::EQ, bit, zero);
    let mask1 = s.m.i8(1);
    let c = s.op(
        counted_b,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        s.t.res_ret,
        139,
        &[count, zero, iu, mask1, align],
    );
    let c = word(&mut s, counted_b, c, 0);
    let inside = s.m.cmp(counted_b, pred::IULT, group, c);
    let runs = s.m.bin(counted_b, Bin::Or, uncounted, inside);
    s.m.terminate(counted_b, Term::CondBr(runs, record_b, end_b));

    // record: read, note, clamp
    let at = s.m.bin(record_b, Bin::Mul, group, stride);
    let mask3 = s.m.i8(7);
    let r = s.op(
        record_b,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        s.t.res_ret,
        139,
        &[records, at, iu, mask3, align],
    );
    let (gx, gy, gz) = (
        word(&mut s, record_b, r, 0),
        word(&mut s, record_b, r, 1),
        word(&mut s, record_b, r, 2),
    );
    // the clamp's notes: the largest count asked for, and the groups asked in all
    let (umax_op, add_op) = (k(&mut s, 7), k(&mut s, 0));
    let (four, eight) = (k(&mut s, 4), k(&mut s, 8));
    s.op(
        record_b,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32t,
        78,
        &[budget, umax_op, four, iu, iu, gx],
    );
    let asked = umin(&mut s, record_b, gx, 1 << 24);
    s.op(
        record_b,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32t,
        78,
        &[budget, add_op, eight, iu, iu, asked],
    );
    let (cx, cy, cz) = (
        umin(&mut s, record_b, gx, 65535),
        umin(&mut s, record_b, gy, 1),
        umin(&mut s, record_b, gz, 1),
    );
    let xy = s.m.bin(record_b, Bin::Mul, cx, cy);
    let groups = s.m.bin(record_b, Bin::Mul, xy, cz);
    let seven = k(&mut s, 7);
    let cost = s.m.bin(record_b, Bin::Shl, groups, seven);
    let free = s.m.cmp(record_b, pred::EQ, cost, zero);
    s.m.terminate(record_b, Term::CondBr(free, end_b, charge_b));

    // charge: the budget's word as it is
    let or_op = k(&mut s, 2);
    let first = s.op(
        charge_b,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32t,
        78,
        &[budget, or_op, zero, iu, iu, zero],
    );
    s.m.terminate(charge_b, Term::Br(head_b));

    // head: while below the budget, try to add the cost (saturating at it)
    let before = s.m.phi(head_b, i32t);
    let limit = k(&mut s, BUDGET);
    let below = s.m.cmp(head_b, pred::IULT, before, limit);
    s.m.terminate(head_b, Term::CondBr(below, swap_b, end_b));

    let left = s.m.bin(swap_b, Bin::Sub, limit, before);
    let fills = s.m.cmp(swap_b, pred::IULE, left, cost);
    let more = s.m.bin(swap_b, Bin::Add, before, cost);
    let next = s.m.select(swap_b, fills, limit, more);
    let prev = s.op(
        swap_b,
        "atomicCompareExchange.i32",
        Attr::NoUnwind,
        i32t,
        79,
        &[budget, zero, iu, iu, before, next],
    );
    let swapped = s.m.cmp(swap_b, pred::EQ, prev, before);
    s.m.terminate(swap_b, Term::CondBr(swapped, won_b, head_b));
    s.m.set_phi(before, vec![(first, charge_b), (prev, swap_b)]);

    // won: over the budget when less was left than the cost
    let over = s.m.cmp(won_b, pred::IULT, left, cost);
    let kept = s.m.select(won_b, over, zero, groups);
    s.m.terminate(won_b, Term::Br(end_b));

    // end: launch
    let launched = s.m.phi(end_b, i32t);
    s.m.set_phi(
        launched,
        vec![
            (direct_groups, direct_b),
            (zero, counted_b),
            (zero, record_b),
            (zero, head_b),
            (kept, won_b),
        ],
    );
    let one_i = k(&mut s, 1);
    let void = s.t.void;
    s.op(
        end_b,
        "dispatchMesh.struct.Payload",
        Attr::NoUnwind,
        void,
        173,
        &[launched, one_i, one_i, payload],
    );
    s.m.terminate(end_b, Term::Ret);
    s.finish()
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn the_amplification_shader_validates() {
        let mut dxil = super::amplification_shader();
        match super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            Ok(mut v) => {
                if let Err(e) = v.validate(&mut dxil) {
                    panic!("the amplification shader does not validate: {e}");
                }
            }
            Err(e) => eprintln!("no validator ({e}); not checked"),
        }
    }
}
