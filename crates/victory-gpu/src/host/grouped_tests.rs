/// What the search per step gives (`declared_writes` without the
/// switch), for keys that stand for themselves.
fn searched(given: &[(usize, (usize, usize))]) -> Vec<(usize, Vec<(usize, usize)>)> {
    let mut back: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for &(key, range) in given {
        match back.iter_mut().find(|(k, _)| *k == key) {
            Some((_, ranges)) => {
                if !ranges.contains(&range) {
                    ranges.push(range);
                }
            }
            None => back.push((key, vec![range])),
        }
    }
    back
}

/// The index gives what the search gives: the same keys in the same
/// order, each with the same ranges in the same order, whatever comes
/// twice and however the keys take turns.
#[test]
fn the_index_gives_what_the_search_gives() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |below: usize| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as usize % below
    };
    assert!(super::grouped(std::iter::empty::<(usize, usize, (usize, usize))>()).is_empty());
    for (keys, ranges, steps) in [(1, 1, 5), (3, 4, 200), (40, 6, 3000), (500, 50, 3000)] {
        for _ in 0..20 {
            let given: Vec<(usize, (usize, usize))> = (0..steps)
                .map(|_| {
                    let from = next(ranges) * 64;
                    (0x1000 + next(keys) * 8, (from, from + 16 + next(2) * 16))
                })
                .collect();
            let by_index = super::grouped(given.iter().map(|&(key, range)| (key, key, range)));
            assert_eq!(by_index, searched(&given), "{keys} keys, {ranges} ranges, {steps} steps");
        }
    }
}
