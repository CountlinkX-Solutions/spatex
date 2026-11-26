use super::*;
use crate::graph::Csr;

/// 0-1, 0-2, 1-2 triangle with distinct weights, a node 3 linked only to 2 (and itself with a
/// self weight), and island 4.
fn graph() -> Csr {
    let edges = [
        (0, 1, 0.5),
        (0, 2, 1.5),
        (1, 0, 2.0),
        (1, 2, 3.0),
        (2, 0, 0.25),
        (2, 1, 0.75),
        (2, 3, 4.0),
        (3, 2, 5.0),
        (3, 3, 7.0),
    ];
    let focal: Vec<u32> = edges.iter().map(|e| e.0).collect();
    let neighbor: Vec<u32> = edges.iter().map(|e| e.1).collect();
    let weight: Vec<f64> = edges.iter().map(|e| e.2).collect();
    Csr::from_arrays(5, &focal, &neighbor, &weight).unwrap()
}

fn settings(permutations: usize, keep: bool) -> Settings {
    Settings {
        permutations,
        seed: 99,
        alternative: Alternative::Directed,
        island_weight: 0.0,
        keep,
    }
}

/// `self_w + sum_c other_w[c] * draw[c]`, a statistic that exposes the neighbor mapping.
fn probe(_i: usize, self_w: f64, other_w: &[f64], draw: &[f64]) -> f64 {
    self_w + other_w.iter().zip(draw).map(|(w, d)| w * d).sum::<f64>()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn explicit_ids_pin_the_neighbor_column_mapping_and_the_skip_of_i() {
    let g = graph();
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    // max_card is 3 (node 2: neighbors 0, 1, 3); ids index the n - 1 = 4 other sites.
    let ids: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 1, 0, 3];
    let observed = vec![0.0; 5];
    let out = crand_with_ids(&g, &values, &observed, &ids, 3, &settings(3, true), probe).unwrap();
    let sims = out.sims.unwrap();
    let skip = |i: usize, m: u32| {
        values[if (m as usize) < i {
            m as usize
        } else {
            m as usize + 1
        }]
    };
    // Site 0: other weights [0.5, 1.5]; replicate r uses ids[r][0], ids[r][1].
    for r in 0..3 {
        let row = &ids[r * 3..r * 3 + 3];
        let expected = 0.5 * skip(0, row[0]) + 1.5 * skip(0, row[1]);
        assert_eq!(sims[r], expected, "site 0 replicate {r}");
    }
    // Site 2: other weights [0.25, 0.75, 4.0] in column order (neighbors 0, 1, 3).
    for r in 0..3 {
        let row = &ids[r * 3..r * 3 + 3];
        let expected = 0.25 * skip(2, row[0]) + 0.75 * skip(2, row[1]) + 4.0 * skip(2, row[2]);
        assert_eq!(sims[2 * 3 + r], expected, "site 2 replicate {r}");
    }
    // Site 3: the self weight 7 is separated from the other weight [5.0] (neighbor 2).
    for r in 0..3 {
        let row = &ids[r * 3..r * 3 + 3];
        let expected = 7.0 + 5.0 * skip(3, row[0]);
        assert_eq!(sims[3 * 3 + r], expected, "site 3 replicate {r}");
    }
}

#[test]
fn island_uses_a_single_pseudo_neighbor_with_island_weight_and_no_self_weight() {
    let g = graph();
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    let ids: Vec<u32> = vec![3, 0, 1, 2, 2, 1, 0, 3, 1];
    let mut s = settings(3, true);
    s.island_weight = 0.5;
    let out = crand_with_ids(&g, &values, &[0.0; 5], &ids, 3, &s, probe).unwrap();
    let sims = out.sims.unwrap();
    for r in 0..3 {
        // Site 4 is the island: weights [0 (self), 0.5], first id column, i = 4 skips nothing.
        let expected = 0.5 * values[ids[r * 3] as usize];
        assert_eq!(sims[4 * 3 + r], expected);
    }
}

#[test]
fn permuted_ids_are_distinct_in_range_and_prefix_stable() {
    let ids = permuted_ids(10, 4, 200, 5);
    assert_eq!(ids.len(), 200 * 4);
    for row in ids.chunks(4) {
        let mut sorted = row.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "row {row:?} repeats an id");
        assert!(row.iter().all(|&m| m < 9));
    }
    let fewer = permuted_ids(10, 4, 50, 5);
    assert_eq!(fewer, ids[..50 * 4]);
    assert_ne!(permuted_ids(10, 4, 50, 6), fewer);
}

#[test]
fn full_width_rows_are_permutations_of_the_other_sites() {
    let ids = permuted_ids(6, 5, 30, 1);
    for row in ids.chunks(5) {
        let mut sorted = row.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}

#[test]
fn results_are_bit_identical_for_any_thread_count() {
    let g = graph();
    let values = [1.0, -2.0, 0.5, 3.0, 4.0];
    let observed = [0.3, -0.1, 0.2, 0.9, 0.0];
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| crand(&g, &values, &observed, &settings(500, true), probe).unwrap())
    };
    let one = run(1);
    for threads in [2, 4, 7] {
        let many = run(threads);
        assert_eq!(bits(&one.p_sim), bits(&many.p_sim), "threads = {threads}");
        assert_eq!(
            bits(one.sims.as_ref().unwrap()),
            bits(many.sims.as_ref().unwrap()),
            "threads = {threads}"
        );
    }
}

#[test]
fn p_sim_is_in_range_and_independent_of_keep() {
    let g = graph();
    let values = [1.0, -2.0, 0.5, 3.0, 4.0];
    let observed = [0.3, -0.1, 0.2, 0.9, 0.0];
    let kept = crand(&g, &values, &observed, &settings(99, true), probe).unwrap();
    let dropped = crand(&g, &values, &observed, &settings(99, false), probe).unwrap();
    assert!(dropped.sims.is_none());
    assert_eq!(bits(&kept.p_sim), bits(&dropped.p_sim));
    assert!(kept.p_sim.iter().all(|&p| (0.01..=1.0).contains(&p)));
    assert_eq!(kept.sims.unwrap().len(), 5 * 99);
}

#[test]
fn a_site_whose_statistic_never_varies_gets_the_smallest_p_value() {
    // The island's reference distribution is all zeros and so is its observed value:
    // directed folds `larger = P` to 0, p = 1 / (P + 1) (an esda quirk, replicated).
    let g = graph();
    let values = [1.0, -2.0, 0.5, 3.0, 4.0];
    let out = crand(
        &g,
        &values,
        &[0.0; 5],
        &settings(99, false),
        |_, s, w, d| probe(0, 0.0 * s, w, d),
    )
    .unwrap();
    assert_eq!(out.p_sim[4], 1.0 / 100.0);
}

#[test]
fn alternatives_follow_significance() {
    let g = graph();
    let values = [1.0, -2.0, 0.5, 3.0, 4.0];
    let observed = [0.3, -0.1, 0.2, 0.9, 0.0];
    for alt in [
        Alternative::Directed,
        Alternative::TwoSided,
        Alternative::Greater,
        Alternative::Lesser,
        Alternative::Folded,
    ] {
        let mut s = settings(199, true);
        s.alternative = alt;
        let out = crand(&g, &values, &observed, &s, probe).unwrap();
        let sims = out.sims.unwrap();
        for i in 0..5 {
            let row = &sims[i * 199..(i + 1) * 199];
            assert_eq!(
                out.p_sim[i],
                significance(alt, observed[i], row),
                "{alt:?} site {i}"
            );
            let sum = summarize(observed[i], row);
            assert_eq!(out.summary[i].ei_sim.to_bits(), sum.ei_sim.to_bits());
        }
    }
}

#[test]
fn invalid_input_is_rejected() {
    let g = graph();
    let v = [1.0; 5];
    assert!(crand(&g, &v[..4], &[0.0; 5], &settings(9, true), probe).is_err());
    assert!(crand(&g, &v, &[0.0; 4], &settings(9, true), probe).is_err());
    assert!(crand(&g, &v, &[0.0; 5], &settings(0, true), probe).is_err());
}

#[test]
fn observed_with_evaluates_the_statistic_on_the_actual_neighbors() {
    let g = graph();
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    let observed = observed_with(&g, &values, 0.0, probe);
    // Site 0 neighbors 1, 2 with weights 0.5, 1.5; site 3 self weight 7 and neighbor 2 (5.0).
    assert_eq!(observed[0], 0.5 * 20.0 + 1.5 * 30.0);
    assert_eq!(observed[2], 0.25 * 10.0 + 0.75 * 20.0 + 4.0 * 40.0);
    assert_eq!(observed[3], 7.0 + 5.0 * 30.0);
    // The island has no neighbors: weights [0 (self), island_weight] on a zero draw.
    assert_eq!(observed[4], 0.0);
}

#[test]
fn oversized_index_matrix_is_an_error_before_allocating() {
    let g = graph();
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    let observed = vec![0.0; 5];
    let err = crand(&g, &values, &observed, &settings(400_000_000, false), probe).unwrap_err();
    assert!(err.contains("permutations"), "{err}");
    let err = crand(&g, &values, &observed, &settings(usize::MAX, false), probe).unwrap_err();
    assert!(err.contains("permutations"), "{err}");
}
