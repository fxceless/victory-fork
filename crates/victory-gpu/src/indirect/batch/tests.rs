use super::*;
use crate::host::Host;

const WORKER: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Weight : register(b0) { uint weight; };
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { uint old; hits.InterlockedAdd(0, weight, old); }
"#;

const BUDGET_READER: &str = r#"
#define RS "UAV(u0), UAV(u1)"
RWByteAddressBuffer budget : register(u0);
RWByteAddressBuffer result : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { result.Store3(0, budget.Load3(0)); }
"#;

const PRODUCER: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Args : register(b0) { uint groups; };
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { dst.Store3(0, uint3(groups, 1, 1)); }
"#;

fn produce(list: &HostList, producer: &Clamp, dst: &ID3D12Resource, groups: u32) {
    list.stop_indirect_batch();
    // SAFETY: an open list, the producer's root layout and a writable buffer.
    unsafe {
        list.list.ResourceBarrier(&[uav_barrier()]);
        list.list.SetComputeRootSignature(&producer.root);
        list.list.SetPipelineState(&producer.pso);
        list.list
            .SetComputeRoot32BitConstants(0, 1, (&raw const groups).cast(), 0);
        list.list
            .SetComputeRootUnorderedAccessView(1, dst.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
    }
}

fn device() -> &'static Host {
    host().expect("a D3D12 device is required for the batch regression tests")
}

fn signature(host: &Host) -> ID3D12CommandSignature {
    let arg = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 12,
        NumArgumentDescs: 1,
        pArgumentDescs: &arg,
        NodeMask: 0,
    };
    let mut signature = None;
    // SAFETY: a valid one-argument command signature, with no root changes.
    unsafe {
        host.device
            .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
            .unwrap()
    };
    signature.unwrap()
}

fn set_worker(list: &HostList, worker: &Clamp, output: &ID3D12Resource) {
    // SAFETY: an open native consumer list and live objects.
    unsafe {
        list.list.SetComputeRootSignature(&worker.root);
        list.list.SetPipelineState(&worker.pso);
        list.list
            .SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
    }
}

fn run_record(
    list: &HostList,
    signature: &ID3D12CommandSignature,
    args: &ID3D12Resource,
    at: u64,
    weight: u32,
) {
    // SAFETY: the consumer's pipeline is already bound; the prelude leaves
    // its scratch readable before this list runs. Only the root value changes.
    unsafe {
        list.list
            .SetComputeRoot32BitConstants(0, 1, (&raw const weight).cast(), 0);
        list.list.ExecuteIndirect(signature, 1, args, at, None, 0);
    }
}

fn record_budget(list: &HostList, reader: &Clamp, output: &ID3D12Resource, at: u64) {
    let budget = list.budget().unwrap();
    // SAFETY: the budget stays UAV through the consumer list. The barrier
    // orders its preceding charge before the read; the output is mapped.
    unsafe {
        list.list.ResourceBarrier(&[uav_barrier()]);
        list.list.SetComputeRootSignature(&reader.root);
        list.list.SetPipelineState(&reader.pso);
        list.list.SetComputeRootUnorderedAccessView(0, budget);
        list.list
            .SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress() + at);
        list.list.Dispatch(1, 1, 1);
    }
}

fn copy_record(
    list: &HostList,
    args: &ID3D12Resource,
    at: u64,
    output: &ID3D12Resource,
    out_at: u64,
) {
    // SAFETY: a live scratch buffer, readable as arguments after either
    // clamp, and an output buffer large enough for this 32-byte record.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            args,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.list.CopyBufferRegion(output, out_at, args, at, 32);
        list.list.ResourceBarrier(&[transition(
            args,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
    }
}

#[test]
fn only_actual_single_record_nonwriting_draws_join_a_prefix() {
    for kind in [0, 1, 10] {
        assert!(accepts(kind, 1, false, false));
        assert!(!accepts(kind, 0, false, false));
        assert!(!accepts(kind, 2, false, false));
        assert!(!accepts(kind, 1, true, false));
        assert!(!accepts(kind, 1, false, true));
    }
    for kind in [2, 3, 105, u32::MAX] {
        assert!(!accepts(kind, 1, false, false));
    }
}

#[test]
fn gpu_prefix_keeps_each_roots_value_and_reclamps_on_replay_and_reset() {
    let host = device();
    let _ = host.log_messages();
    let list = host.new_list().unwrap();
    let worker = make_from(&host.device, WORKER).unwrap();
    let signature = signature(host);
    let (src_a, a) = host.shared_buffer(64).unwrap();
    let (src_b, b) = host.shared_buffer(32).unwrap();
    let (output, hits) = host.shared_buffer(64).unwrap();
    // SAFETY: each mapping holds these complete padded records.
    unsafe {
        std::ptr::copy_nonoverlapping(
            [2u32, 1, 1, 77, 0, 0, 0, 0, 5, 1, 1, 99]
                .as_ptr()
                .cast::<u8>(),
            a,
            48,
        );
        std::ptr::copy_nonoverlapping([7u32, 1, 1, 88].as_ptr().cast::<u8>(), b, 16);
    }
    set_worker(&list, &worker, &output);
    for (src, offset, weight) in [
        (&src_a, 0, 1),
        (&src_b, 0, 3),
        (&src_a, 32, 11),
        (&src_b, 0, 5),
    ] {
        let (args, at) = list.append_indirect_batch(src, offset, 10, 12).unwrap();
        run_record(&list, &signature, &args, at, weight);
    }
    list.close();
    assert_eq!(host.log_messages(), None);
    for (first, expected) in [(2u32, 2 + 7 * 8 + 5 * 11), (13, 13 + 7 * 8 + 5 * 11)] {
        // SAFETY: the previous execution has finished; no reset or new
        // recording separates these two executions of the same closed list.
        unsafe {
            a.cast::<u32>().write_volatile(first);
            hits.cast::<u32>().write_volatile(0);
        }
        host.execute(&[&list]);
        // SAFETY: mapped result after the device fence.
        assert_eq!(unsafe { hits.cast::<u32>().read_volatile() }, expected);
    }
    list.reset();
    set_worker(&list, &worker, &output);
    let (args, at) = list.append_indirect_batch(&src_a, 0, 10, 12).unwrap();
    run_record(&list, &signature, &args, at, 7);
    list.close();
    // SAFETY: the list is not executing.
    unsafe { hits.cast::<u32>().write_volatile(0) };
    host.execute(&[&list]);
    // SAFETY: after its fence.
    assert_eq!(unsafe { hits.cast::<u32>().read_volatile() }, 13 * 7);
    list.reset();
}

#[test]
fn gpu_ordered_batch_matches_inline_counts_padding_and_budget_across_groups() {
    let host = device();
    let _ = host.log_messages();
    let inline = host.new_list().unwrap();
    let batched = host.new_list().unwrap();
    let reader = make_from(&host.device, BUDGET_READER).unwrap();
    const N: usize = 129;
    let (source, data) = host.shared_buffer((N * 32) as u64).unwrap();
    let (left, left_ptr) = host.shared_buffer((N * 32 + 16) as u64).unwrap();
    let (right, right_ptr) = host.shared_buffer((N * 32 + 16) as u64).unwrap();
    let mut original = Vec::new();
    for i in 0..N {
        let kind = [0u32, 1, 10][i % 3];
        let mut words = [
            0u32,
            1,
            1,
            0xFFFF_FFF9,
            i as u32,
            0x12345678,
            0x80000000,
            0xABCDEF01,
        ];
        words[0] = if i % 7 == 0 {
            u32::MAX
        } else {
            i as u32 % 17 + 1
        };
        if kind <= 1 {
            words[1] = if i % 5 == 0 { 0 } else { 1 };
        }
        if i == 0 {
            words[0] = 1 << 23;
        }
        if i == 63 || i == 64 {
            words[0] = 1;
            words[1] = 1 << 26;
        }
        if i == 66 {
            words[0] = 97;
            words[1] = 0;
        }
        // SAFETY: the source mapping has N complete records.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), data.add(i * 32), 32) };
        original.extend_from_slice(&words);
        let args =
            super::super::bound(&inline, (&source, (i * 32) as u64), None, (kind, 32, 1)).unwrap();
        copy_record(&inline, &args, 0, &left, (i * 32) as u64);
        let (args, at) = batched
            .append_indirect_batch(&source, (i * 32) as u64, kind, 32)
            .unwrap();
        copy_record(&batched, &args, at, &right, (i * 32) as u64);
    }
    record_budget(&inline, &reader, &left, (N * 32) as u64);
    record_budget(&batched, &reader, &right, (N * 32) as u64);
    inline.close();
    batched.close();
    assert_eq!(host.log_messages(), None);
    host.execute(&[&inline, &batched]);
    // SAFETY: mappings cover every output record and the three budget words;
    // both GPU executions finished. The source must remain untouched.
    unsafe {
        let left = std::slice::from_raw_parts(left_ptr.cast::<u32>(), N * 8 + 3);
        let right = std::slice::from_raw_parts(right_ptr.cast::<u32>(), N * 8 + 3);
        assert_eq!(
            left, right,
            "batching must preserve all output bytes and budget notes"
        );
        assert_eq!(right[N * 8], 1 << 27, "the original list budget saturates");
        assert_eq!(
            std::slice::from_raw_parts(data.cast::<u32>(), N * 8),
            original
        );
    }
    inline.reset();
    batched.reset();
}

#[test]
fn gpu_query_boundary_seals_the_prefix_and_uses_inline_clamps_inside() {
    let host = device();
    let _ = host.log_messages();
    let list = host.new_list().unwrap();
    let worker = make_from(&host.device, WORKER).unwrap();
    let signature = signature(host);
    let (source, data) = host.shared_buffer(32).unwrap();
    let (output, _) = host.shared_buffer(64).unwrap();
    let (result, mapped) = host.shared_buffer(128).unwrap();
    // SAFETY: complete records in mapped source memory.
    unsafe {
        std::ptr::copy_nonoverlapping([2u32, 1, 1, 0, 3, 1, 1].as_ptr().cast::<u8>(), data, 28)
    };
    set_worker(&list, &worker, &output);
    let (args, at) = list.append_indirect_batch(&source, 0, 10, 12).unwrap();
    run_record(&list, &signature, &args, at, 1);
    let mut query = None;
    // SAFETY: an ordinary pipeline statistics heap.
    unsafe {
        host.device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
                    Count: 1,
                    NodeMask: 0,
                },
                &mut query,
            )
            .unwrap()
    };
    let query: ID3D12QueryHeap = query.unwrap();
    list.stop_indirect_batch();
    assert!(list.append_indirect_batch(&source, 16, 10, 12).is_none());
    // SAFETY: the query begins and ends in the same native list.
    unsafe {
        list.list
            .BeginQuery(&query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0)
    };
    let args = super::super::bound(&list, (&source, 16), None, (2, 12, 1)).unwrap();
    set_worker(&list, &worker, &output);
    run_record(&list, &signature, &args, 0, 1);
    // SAFETY: the matching query end and a sufficiently large result buffer.
    unsafe {
        list.list
            .EndQuery(&query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
        list.list.ResolveQueryData(
            &query,
            D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            0,
            1,
            &result,
            0,
        );
    }
    list.close();
    assert_eq!(host.log_messages(), None);
    host.execute(&[&list]);
    // SAFETY: query data after the device fence. The prefix's 1-lane
    // prepare and 64-lane charge are outside the query; its inline clamp
    // (64 lanes) and three worker groups remain inside, as in the old path.
    let stats = unsafe {
        mapped
            .cast::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>()
            .read_volatile()
    };
    assert_eq!(stats.CSInvocations, 64 + 3);
    list.reset();
}

#[test]
fn gpu_capacity_and_short_tail_fall_back_without_restarting_a_prefix() {
    let host = device();
    let list = host.new_list().unwrap();
    let (source, data) = host.shared_buffer(16).unwrap();
    // SAFETY: a complete argument record in the mapping.
    unsafe { std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), data, 12) };
    for _ in 0..CAPACITY {
        list.append_indirect_batch(&source, 0, 10, 12).unwrap();
    }
    assert!(list.append_indirect_batch(&source, 0, 10, 12).is_none());
    assert!(list.append_indirect_batch(&source, 0, 10, 12).is_none());
    // The inline fallback shares the already charged list budget.
    assert!(super::super::bound(&list, (&source, 0), None, (2, 12, 1)).is_some());
    list.close();
    host.execute(&[&list]);
    list.reset();
    assert!(list.append_indirect_batch(&source, 4, 10, 16).is_none());
    assert!(list.append_indirect_batch(&source, 0, 10, 12).is_none());
    list.close();
    list.reset();
    assert!(list.append_indirect_batch(&source, 0, 10, 12).is_some());
    list.close();
    host.execute(&[&list]);
    list.reset();
}

#[test]
fn gpu_copy_before_the_first_draw_blocks_a_prefix() {
    let host = device();
    let list = host.new_list().unwrap();
    let (source, data) = host.shared_buffer(16).unwrap();
    let (_other, other) = host.shared_buffer(16).unwrap();
    list.defer_copy(data, other, 12);
    assert!(list.append_indirect_batch(&source, 0, 10, 12).is_none());
    list.close();
    list.reset();
}

#[test]
fn gpu_a_guard_during_preparation_keeps_the_pending_consumer_readable() {
    let host = device();
    let list = host.new_list().unwrap();
    let worker = make_from(&host.device, WORKER).unwrap();
    let signature = signature(host);
    let (source, data) = host.shared_buffer(16).unwrap();
    let (output, hits) = host.shared_buffer(64).unwrap();
    // SAFETY: mapped source and output before execution.
    unsafe {
        std::ptr::copy_nonoverlapping([5u32, 1, 1].as_ptr().cast::<u8>(), data, 12);
        hits.cast::<u32>().write_volatile(0);
    }
    let (args, at) = list.append_indirect_batch(&source, 0, 10, 12).unwrap();
    // A texture preparation helper can seal the prefix before the current
    // draw records its native execution. Scratch must stay readable until
    // the owner's close, rather than returning to COMMON at this guard.
    list.stop_indirect_batch();
    set_worker(&list, &worker, &output);
    run_record(&list, &signature, &args, at, 3);
    list.close();
    host.execute(&[&list]);
    // SAFETY: output after the device fence.
    assert_eq!(unsafe { hits.cast::<u32>().read_volatile() }, 15);
    list.reset();
}

#[test]
fn gpu_producer_after_a_prefix_is_seen_by_the_inline_fallback() {
    let host = device();
    let _ = host.log_messages();
    let list = host.new_list().unwrap();
    let producer = make_from(&host.device, PRODUCER).unwrap();
    let worker = make_from(&host.device, WORKER).unwrap();
    let signature = signature(host);
    let (source, data) = host.shared_buffer(16).unwrap();
    let (output, hits) = host.shared_buffer(64).unwrap();
    // SAFETY: mapped source and output, not executing yet.
    unsafe {
        std::ptr::copy_nonoverlapping([2u32, 1, 1].as_ptr().cast::<u8>(), data, 12);
        hits.cast::<u32>().write_volatile(0);
    }
    set_worker(&list, &worker, &output);
    let (args, at) = list.append_indirect_batch(&source, 0, 10, 12).unwrap();
    run_record(&list, &signature, &args, at, 1);
    produce(&list, &producer, &source, 7);
    assert!(list.append_indirect_batch(&source, 0, 10, 12).is_none());
    let args = super::super::bound(&list, (&source, 0), None, (2, 12, 1)).unwrap();
    set_worker(&list, &worker, &output);
    run_record(&list, &signature, &args, 0, 1);
    list.close();
    assert_eq!(host.log_messages(), None);
    host.execute(&[&list]);
    // SAFETY: mapped output after the device fence: the second execution
    // must use 7, not the prefix's stale value 2.
    assert_eq!(unsafe { hits.cast::<u32>().read_volatile() }, 2 + 7);
    list.reset();
}

#[test]
fn gpu_prefix_observes_a_producer_in_the_preceding_list() {
    let host = device();
    let _ = host.log_messages();
    let before = host.new_list().unwrap();
    let consumer = host.new_list().unwrap();
    let producer = make_from(&host.device, PRODUCER).unwrap();
    let worker = make_from(&host.device, WORKER).unwrap();
    let signature = signature(host);
    let (source, _) = host.shared_buffer(16).unwrap();
    let (output, hits) = host.shared_buffer(64).unwrap();
    // SAFETY: the mapped result is not executing yet.
    unsafe { hits.cast::<u32>().write_volatile(0) };
    produce(&before, &producer, &source, 11);
    before.close();
    set_worker(&consumer, &worker, &output);
    let (args, at) = consumer.append_indirect_batch(&source, 0, 10, 12).unwrap();
    run_record(&consumer, &signature, &args, at, 3);
    consumer.close();
    assert_eq!(host.log_messages(), None);
    host.execute(&[&before, &consumer]);
    // SAFETY: mapped result after both lists and their fence.
    assert_eq!(unsafe { hits.cast::<u32>().read_volatile() }, 11 * 3);
    before.reset();
    consumer.reset();
}

#[test]
#[ignore = "a synthetic timestamp measurement; run explicitly with --ignored --nocapture"]
fn measure_indirect_prefix_against_inline_with_equal_work() {
    let host = device();
    let _ = host.log_messages();
    let worker = make_from(&host.device, WORKER).unwrap();
    let signature = signature(host);
    let (source, data) = host.shared_buffer(16).unwrap();
    // SAFETY: a complete single-group record in mapped source memory.
    unsafe { std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), data, 12) };
    for n in [1u32, 16, 64, CAPACITY] {
        let mut times = [Vec::new(), Vec::new()];
        for (mode, samples) in times.iter_mut().enumerate() {
            let begin = host.new_list().unwrap();
            let work = host.new_list().unwrap();
            let end = host.new_list().unwrap();
            let (output, hits) = host.shared_buffer(64).unwrap();
            let (back, mapped) = host.shared_buffer(16).unwrap();
            let mut heap = None;
            // SAFETY: a two-timestamp heap, read back after both native
            // lists and any batch prelude. Neither timestamp excludes it.
            unsafe {
                host.device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut heap,
                    )
                    .unwrap();
            }
            let heap: ID3D12QueryHeap = heap.unwrap();
            // SAFETY: open wrapper lists and a 16-byte readback mapping.
            unsafe {
                begin.list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                end.list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                end.list
                    .ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            }
            let mut expected = 0u32;
            set_worker(&work, &worker, &output);
            for i in 0..n {
                let weight = i % 11 + 1;
                expected += weight;
                let (args, at) = if mode == 0 {
                    let args = super::super::bound(&work, (&source, 0), None, (10, 12, 1)).unwrap();
                    set_worker(&work, &worker, &output);
                    (args, 0)
                } else {
                    work.append_indirect_batch(&source, 0, 10, 12).unwrap()
                };
                run_record(&work, &signature, &args, at, weight);
            }
            begin.close();
            work.close();
            end.close();
            assert_eq!(host.log_messages(), None);
            for iteration in 0..15 {
                // SAFETY: previous executions finished before the next reset
                // of the counter; all commands and root values are identical.
                unsafe { hits.cast::<u32>().write_volatile(0) };
                host.execute(&[&begin, &work, &end]);
                // SAFETY: timestamps and output after the device fence.
                let (start, stop, count) = unsafe {
                    (
                        mapped.cast::<u64>().read_volatile(),
                        mapped.cast::<u64>().add(1).read_volatile(),
                        hits.cast::<u32>().read_volatile(),
                    )
                };
                assert_eq!(
                    count, expected,
                    "both modes must execute the same weighted work"
                );
                assert!(stop > start);
                if iteration >= 5 {
                    samples.push((stop - start) as f64 * 1e6 / host.timestamp_frequency() as f64);
                }
            }
            begin.reset();
            work.reset();
            end.reset();
        }
        for samples in &mut times {
            samples.sort_by(f64::total_cmp);
        }
        println!(
            "prefix-bench calls={n} inline_median_us={:.3} batch_median_us={:.3} ratio={:.3} same_weighted_work=true",
            times[0][5],
            times[1][5],
            times[0][5] / times[1][5]
        );
    }
}
