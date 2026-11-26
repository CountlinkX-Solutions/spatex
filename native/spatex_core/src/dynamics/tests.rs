use super::chain::{self, mfpt, sojourn_time, steady_state, structure};
use super::markov::{chi2_test, count_transitions, homogeneity, kullback, row_normalize};
use super::rank::{spatial_tau, tau, theta};
use super::spatial::{self, SpatialMarkovSettings};
use super::special::chi2_sf;
use crate::graph::Csr;

const OZ: [f64; 9] = [0.5, 0.25, 0.25, 0.5, 0.0, 0.5, 0.25, 0.25, 0.5];
const RED: [f64; 9] = [0.5, 0.5, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 1.0];

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        let ok = (x.is_infinite() && x == y) || (x - y).abs() <= tol;
        assert!(ok, "{a:?} vs {b:?}");
    }
}

fn path_graph(n: usize) -> Csr {
    let mut f = Vec::new();
    let mut nb = Vec::new();
    for i in 0..n - 1 {
        f.extend([i as u32, (i + 1) as u32]);
        nb.extend([(i + 1) as u32, i as u32]);
    }
    let w = vec![1.0; f.len()];
    Csr::from_arrays(n, &f, &nb, &w).unwrap()
}

#[test]
fn chi2_sf_matches_critical_values() {
    assert!((chi2_sf(3.841458820694124, 1.0) - 0.05).abs() < 1e-13);
    assert!((chi2_sf(5.991464547107979, 2.0) - 0.05).abs() < 1e-13);
    assert!((chi2_sf(18.307038053275146, 10.0) - 0.05).abs() < 1e-12);
    assert_eq!(chi2_sf(0.0, 3.0), 1.0);
    assert!(chi2_sf(1.0, 0.0).is_nan());
    assert!(chi2_sf(1000.0, 3.0) < 1e-200);
}

#[test]
fn transitions_count_consecutive_periods() {
    // two locations, three periods, classes 0..2
    let ids = [0, 1, 1, 1, 0, 0];
    let t = count_transitions(&ids, 2, 3, 2).unwrap();
    // loc 0: 0->1, 1->1; loc 1: 1->0, 0->0
    close(&t, &[1.0, 1.0, 1.0, 1.0], 0.0);
    assert!(count_transitions(&[0, 5], 1, 2, 2).is_err());
    assert!(count_transitions(&[0, 1, 0], 2, 2, 2).is_err());
}

#[test]
fn row_normalize_leaves_empty_rows_zero() {
    let p = row_normalize(&[2.0, 2.0, 0.0, 0.0], 2);
    close(&p, &[0.5, 0.5, 0.0, 0.0], 1e-15);
}

#[test]
fn steady_state_of_oz_is_known() {
    let ss = steady_state(&OZ, 3).unwrap();
    assert_eq!(ss.len(), 1);
    close(&ss[0], &[0.4, 0.2, 0.4], 1e-14);
}

#[test]
fn steady_state_reducible_lists_each_recurrent_class() {
    let p = [0.5, 0.5, 0.0, 0.4, 0.6, 0.0, 0.0, 0.0, 1.0];
    let ss = steady_state(&p, 3).unwrap();
    assert_eq!(ss.len(), 2);
    close(&ss[0], &[4.0 / 9.0, 5.0 / 9.0, 0.0], 1e-14);
    close(&ss[1], &[0.0, 0.0, 1.0], 1e-14);
}

#[test]
fn mfpt_of_oz_matches_kemeny() {
    let m = mfpt(&OZ, 3).unwrap();
    close(
        &m,
        &[
            2.5,
            4.0,
            10.0 / 3.0,
            8.0 / 3.0,
            5.0,
            8.0 / 3.0,
            10.0 / 3.0,
            4.0,
            2.5,
        ],
        1e-13,
    );
}

#[test]
fn mfpt_of_reducible_chain_is_infinite_across_classes() {
    let m = mfpt(&RED, 3).unwrap();
    let inf = f64::INFINITY;
    close(&m, &[2.25, 2.0, inf, 2.5, 1.8, inf, inf, inf, 1.0], 1e-13);
}

#[test]
fn mfpt_with_a_recurrent_cycle_is_finite_where_passage_is_certain() {
    // {0,1} is a closed 2-cycle; state 2 leaves to it. Passage 2 -> 0 is certain? no: it may
    // land on 1 first and then reaches 0 surely, so it is finite; 0 -> 2 is never.
    let p = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0];
    let m = mfpt(&p, 3).unwrap();
    assert_eq!(m[2], f64::INFINITY); // 0 -> 2
    assert!((m[1] - 1.0).abs() < 1e-13); // 0 -> 1
    assert!(m[2 * 3].is_finite()); // 2 -> 0
}

#[test]
fn mfpt_ignores_states_only_reachable_through_the_destination() {
    // 0 -> {0, 1}, 1 -> {1, 2}, 2 absorbing: 2 is reached only through 1, so 0 -> 1 is finite.
    let p = [0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0];
    let m = mfpt(&p, 3).unwrap();
    assert!((m[1] - 2.0).abs() < 1e-13, "{m:?}"); // 0 -> 1: geometric(1/2)
    assert!((m[5] - 2.0).abs() < 1e-13); // 1 -> 2
    assert!((m[2] - 4.0).abs() < 1e-13); // 0 -> 2
    assert_eq!(m[3], f64::INFINITY); // 1 -> 0 never
}

#[test]
fn structure_finds_classes_and_absorbing_states() {
    let s = structure(&RED, 3);
    assert_eq!(s.communication, vec![vec![0, 1], vec![2]]);
    assert_eq!(s.recurrent, vec![vec![0, 1], vec![2]]);
    assert!(s.transient.is_empty());
    assert_eq!(s.absorbing, vec![2]);
    let t = structure(&[0.5, 0.25, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3);
    assert_eq!(t.recurrent, vec![vec![1], vec![2]]);
    assert_eq!(t.transient, vec![vec![0]]);
    assert_eq!(t.absorbing, vec![1, 2]);
    let o = structure(&OZ, 3);
    assert_eq!(o.communication, vec![vec![0, 1, 2]]);
    assert!(o.absorbing.is_empty());
}

#[test]
fn sojourn_time_is_infinite_for_absorbing_states() {
    let s = sojourn_time(&RED, 3);
    close(&s, &[2.0, 2.5, f64::INFINITY], 1e-14);
}

#[test]
fn fill_empty_diagonals_makes_rows_stochastic() {
    let mut p = vec![0.5, 0.5, 0.0, 0.0];
    chain::fill_empty_diagonals(&mut p, 2);
    close(&p, &[0.5, 0.5, 0.0, 1.0], 0.0);
}

#[test]
fn chi2_test_of_identical_counts_is_zero() {
    let t = [10.0, 5.0, 4.0, 11.0];
    let (stat, p, dof) = chi2_test(&t, &t, 2);
    assert!(stat.abs() < 1e-12);
    assert!((p - 1.0).abs() < 1e-12);
    assert_eq!(dof, 1.0);
}

#[test]
fn homogeneity_of_identical_regimes_is_zero() {
    let a = vec![10.0, 5.0, 4.0, 11.0];
    let h = homogeneity(&[a.clone(), a], 2);
    assert!(h.q.abs() < 1e-12 && h.lr.abs() < 1e-12);
    assert_eq!(h.dof, 2);
}

#[test]
fn tau_perfect_reversed_and_ties() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let t = tau(&x, &[2.0, 4.0, 6.0, 8.0]);
    assert!((t.tau - 1.0).abs() < 1e-14);
    assert!(t.p < 0.1);
    let r = tau(&x, &[4.0, 3.0, 2.0, 1.0]);
    assert!((r.tau + 1.0).abs() < 1e-14);
    let tied = tau(&[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0]);
    assert_eq!((tied.concordant, tied.discordant), (2.0, 0.0));
    assert_eq!((tied.extra_x, tied.extra_y), (0.0, 1.0));
    assert!((tied.tau - 2.0 / 6.0_f64.sqrt()).abs() < 1e-14);
}

#[test]
fn spatial_tau_counts_neighbor_pairs_only() {
    let g = path_graph(4);
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [1.0, 3.0, 2.0, 4.0];
    let s = spatial_tau(&x, &y, &g, 0, 0).unwrap();
    // pairs (0,1) conc, (1,2) disc, (2,3) conc: iS = 1, n1 = n2 = 3
    assert!((s.tau_spatial - 1.0 / 3.0).abs() < 1e-14);
    assert_eq!(s.pairs_spatial, 3);
    assert_eq!((s.concordant_spatial, s.discordant_spatial), (2.0, 1.0));
    assert!(s.taus.is_empty());
}

#[test]
fn spatial_tau_permutations_are_thread_independent() {
    let g = path_graph(30);
    let x: Vec<f64> = (0..30).map(f64::from).collect();
    let y: Vec<f64> = (0..30).map(|i| f64::from((i * 7) % 11)).collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| spatial_tau(&x, &y, &g, 200, 42).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.taus, b.taus);
    assert_eq!(a.psim, b.psim);
    assert!(a.psim > 0.0 && a.psim <= 1.0);
}

#[test]
fn theta_is_one_when_regimes_move_together() {
    // 4 locations, 2 periods; ranks go 1,2,3,4 -> 4,3,2,1: ranks_d = 3,1,-1,-3
    let y = [1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
    let r = theta(&y, 4, 2, &[0, 0, 1, 1], 0, 0).unwrap();
    // regime0: |3+1| = 4, regime1: |-1-3| = 4, total 8
    assert!((r.theta[0] - 1.0).abs() < 1e-14);
    assert_eq!(r.total, vec![8.0]);
    assert_eq!(r.max_total, 8.0);
    let mixed = theta(&y, 4, 2, &[0, 1, 0, 1], 0, 0).unwrap();
    // regime0: |3-1| = 2, regime1: |1-3| = 2
    assert!((mixed.theta[0] - 0.5).abs() < 1e-14);
}

#[test]
fn theta_permutation_pvalues_are_valid_and_deterministic() {
    let y: Vec<f64> = (0..40)
        .map(|i| f64::from((i * 13) % 17) + f64::from(i % 3))
        .collect();
    let regime: Vec<u32> = (0..10).map(|i| i % 3).collect();
    let a = theta(&y, 10, 4, &regime, 99, 7).unwrap();
    let b = theta(&y, 10, 4, &regime, 99, 7).unwrap();
    assert_eq!(a.pvalue_left, b.pvalue_left);
    for (l, r) in a.pvalue_left.iter().zip(&a.pvalue_right) {
        assert!(*l > 0.0 && *r >= 0.0 && l + r >= 1.0 - 1e-12);
    }
}

#[test]
fn lag_categorical_takes_the_weighted_mode_and_breaks_ties_by_self_then_lowest() {
    // path 0-1-2-3-4, classes 0,1,1,0,1 with binary weights
    let g = path_graph(5);
    let out = spatial::lag_categorical(&g, &[0, 1, 1, 0, 1], 2);
    // node0: nbr {1}->1; node1: nbrs {0,2} -> tie(0:1,1:1) -> self 1 wins;
    // node2: nbrs {1,3} -> tie -> self 1; node3: nbrs {2,4} -> 1; node4: nbr {3} -> 0
    assert_eq!(out, vec![1, 1, 1, 1, 0]);
}

#[test]
fn spatial_markov_conditional_counts_partition_the_transitions() {
    let g = path_graph(6).transform(crate::graph::Transform::parse("r").unwrap());
    let y: Vec<f64> = (0..6 * 4)
        .map(|i| f64::from((i * 5 + i / 3) % 11))
        .collect();
    let s = SpatialMarkovSettings {
        k: 3,
        m: 2,
        fixed: true,
        discrete: false,
        cutoffs: None,
        lag_cutoffs: None,
        permutations: 0,
        seed: 0,
    };
    let r = spatial::spatial_markov(&y, 6, 4, &g, &s).unwrap();
    let total: f64 = r.t_cond.iter().sum();
    assert_eq!(total, (6 * 3) as f64);
    assert_eq!(r.class_ids.len(), 24);
    assert!(r.x2_rpvalue.is_none());
    let perm = SpatialMarkovSettings {
        permutations: 99,
        seed: 3,
        ..s
    };
    let a = spatial::spatial_markov(&y, 6, 4, &g, &perm).unwrap();
    let b = spatial::spatial_markov(&y, 6, 4, &g, &perm).unwrap();
    assert_eq!(a.x2_sims, b.x2_sims);
    assert!(a.x2_rpvalue.unwrap() > 0.0);
}

// -- Kendall tau: Knight's O(n log n) algorithm against the O(n^2) definition ---------------

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

fn same_tau(a: &super::rank::TauResult, b: &super::rank::TauResult) {
    assert_eq!(a.concordant, b.concordant);
    assert_eq!(a.discordant, b.discordant);
    assert_eq!(a.extra_x, b.extra_x);
    assert_eq!(a.extra_y, b.extra_y);
    assert!(
        (a.tau.is_nan() && b.tau.is_nan()) || (a.tau - b.tau).abs() <= 1e-14,
        "{} vs {}",
        a.tau,
        b.tau
    );
    assert!((a.p.is_nan() && b.p.is_nan()) || (a.p - b.p).abs() <= 1e-12);
}

#[test]
fn knight_tau_matches_the_quadratic_definition_on_random_data_with_ties() {
    let mut seed = 7;
    for round in 0..300 {
        let n = 2 + (lcg(&mut seed) % 60) as usize;
        // Few distinct values force many ties in x, y, and both.
        let (vx, vy) = (1 + (round % 7) as u64, 1 + (round % 5) as u64);
        let x: Vec<f64> = (0..n).map(|_| (lcg(&mut seed) % vx) as f64).collect();
        let y: Vec<f64> = (0..n).map(|_| (lcg(&mut seed) % vy) as f64 - 0.5).collect();
        same_tau(&tau(&x, &y), &super::rank::tau_naive(&x, &y));
    }
}

#[test]
fn knight_tau_matches_on_continuous_data_and_signed_zero() {
    let mut seed = 11;
    let x: Vec<f64> = (0..200).map(|_| lcg(&mut seed) as f64 / 1e6).collect();
    let y: Vec<f64> = (0..200).map(|_| lcg(&mut seed) as f64 / 1e6).collect();
    same_tau(&tau(&x, &y), &super::rank::tau_naive(&x, &y));
    let (xz, yz) = ([0.0, -0.0, 1.0, 1.0], [1.0, 2.0, -0.0, 0.0]);
    same_tau(&tau(&xz, &yz), &super::rank::tau_naive(&xz, &yz));
}

#[test]
fn knight_tau_handles_a_hundred_thousand_points() {
    let mut seed = 3;
    let x: Vec<f64> = (0..100_000)
        .map(|_| (lcg(&mut seed) % 5000) as f64)
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|v| v + (lcg(&mut seed) % 100) as f64)
        .collect();
    let t = tau(&x, &y);
    assert!(t.tau > 0.9 && t.tau <= 1.0, "{}", t.tau);
}

// -- allocation guards ------------------------------------------------------------------------

#[test]
fn transitions_reject_too_many_classes() {
    let err = count_transitions(&[0, 0], 1, 2, 1_001).unwrap_err();
    assert!(err.contains("classes") && err.contains("1000"), "{err}");
    assert!(count_transitions(&[0, 0], 1, 2, 1_000).is_ok());
}

fn guard_settings(k: usize, m: usize, discrete: bool) -> SpatialMarkovSettings {
    SpatialMarkovSettings {
        k,
        m,
        fixed: true,
        discrete,
        cutoffs: None,
        lag_cutoffs: None,
        permutations: 0,
        seed: 0,
    }
}

#[test]
fn spatial_markov_rejects_too_many_classes_before_allocating() {
    let g = path_graph(4);
    let y = [0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 3.0, 3.0];
    let err =
        spatial::spatial_markov(&y, 4, 2, &g, &guard_settings(1_001, 1_001, true)).unwrap_err();
    assert!(err.contains("classes"), "{err}");
    let err = spatial::spatial_markov(&y, 4, 2, &g, &guard_settings(4, 1_001, false)).unwrap_err();
    assert!(err.contains("classes"), "{err}");
}

#[test]
fn spatial_markov_rejects_an_oversized_conditional_table() {
    let g = path_graph(4);
    let y = [0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 3.0, 3.0];
    let err = spatial::spatial_markov(&y, 4, 2, &g, &guard_settings(400, 400, false)).unwrap_err();
    assert!(
        err.contains("m * k * k") || err.contains("conditional"),
        "{err}"
    );
}

const KULLBACK_S1: [f64; 36] = [
    22.0, 11.0, 24.0, 2.0, 2.0, 7.0, 5.0, 23.0, 15.0, 3.0, 42.0, 6.0, 4.0, 21.0, 190.0, 25.0, 20.0,
    34.0, 0.0, 2.0, 14.0, 56.0, 14.0, 28.0, 32.0, 15.0, 20.0, 10.0, 56.0, 14.0, 5.0, 22.0, 31.0,
    18.0, 13.0, 134.0,
];
const KULLBACK_S2: [f64; 36] = [
    3.0, 6.0, 9.0, 3.0, 0.0, 8.0, 1.0, 9.0, 3.0, 12.0, 27.0, 5.0, 2.0, 9.0, 208.0, 32.0, 5.0, 18.0,
    0.0, 14.0, 32.0, 108.0, 40.0, 40.0, 22.0, 14.0, 9.0, 26.0, 224.0, 14.0, 1.0, 5.0, 13.0, 53.0,
    13.0, 116.0,
];

#[test]
fn kullback_matches_the_giddy_docstring_table() {
    let f: Vec<f64> = KULLBACK_S1.iter().chain(&KULLBACK_S2).copied().collect();
    for giddy in [false, true] {
        let k = kullback(&f, 2, 6, giddy).unwrap();
        assert!((k.statistic - 160.960_600_311_718_74).abs() < 1e-9, "{k:?}");
        assert_eq!(k.dof, 30);
        assert!(k.p_value < 1e-12);
    }
}

#[test]
fn kullback_of_identical_strata_is_zero_with_row_totals() {
    let f = [10.0, 5.0, 4.0, 12.0, 10.0, 5.0, 4.0, 12.0];
    let k = kullback(&f, 2, 2, false).unwrap();
    assert!(k.statistic.abs() < 1e-12);
    assert_eq!(k.dof, 2);
    assert!((k.p_value - 1.0).abs() < 1e-12);
    // giddy sums the column totals instead, which is negative here
    let g = kullback(&f, 2, 2, true).unwrap();
    assert!(
        (g.statistic - -0.517_027_564_798_581_8).abs() < 1e-9,
        "{g:?}"
    );
}

#[test]
fn kullback_with_one_stratum_has_no_degrees_of_freedom() {
    let k = kullback(&[10.0, 5.0, 4.0, 12.0], 1, 2, false).unwrap();
    assert!(k.statistic.abs() < 1e-12);
    assert_eq!(k.dof, 0);
    assert!(k.p_value.is_nan());
}

#[test]
fn kullback_survives_empty_rows_and_rejects_bad_input() {
    let f = [
        5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 4.0, 6.0, 3.0, 1.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0,
    ];
    let k = kullback(&f, 2, 3, false).unwrap();
    assert!(k.statistic.is_finite() && k.statistic >= 0.0);
    assert!(kullback(&f, 3, 3, false).is_err());
    assert!(kullback(&[1.0, -1.0, 2.0, 3.0], 1, 2, false).is_err());
    assert!(kullback(&[], 0, 2, false).is_err());
}
