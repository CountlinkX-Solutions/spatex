use super::{gini, spatial_gini, theil, theil_d, theil_d_sim};
use crate::graph::Csr;

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
fn gini_of_known_distributions() {
    assert_eq!(gini(&[1.0, 1.0, 1.0, 1.0]), 0.0);
    // one holder of everything among four: (2 * 4 * 1 - 5 * 1) / (4 * 1)
    assert!((gini(&[0.0, 0.0, 0.0, 1.0]) - 0.75).abs() < 1e-15);
    assert!((gini(&[1.0, 2.0, 3.0, 4.0]) - 0.25).abs() < 1e-15);
    assert!(gini(&[0.0, 0.0]).is_nan());
}

#[test]
fn gini_ignores_input_order() {
    assert_eq!(gini(&[4.0, 1.0, 3.0, 2.0]), gini(&[1.0, 2.0, 3.0, 4.0]));
}

#[test]
fn theil_of_two_values_and_equal_values() {
    let t = theil(&[1.0, 3.0], 2, 1);
    let expected = 0.25 * (0.5_f64).ln() + 0.75 * (1.5_f64).ln();
    assert!((t[0] - expected).abs() < 1e-15);
    assert!(theil(&[2.0, 2.0, 2.0], 3, 1)[0].abs() < 1e-15);
}

#[test]
fn theil_works_per_column_and_survives_zeros() {
    // columns: [1, 3] and [0, 4]
    let t = theil(&[1.0, 0.0, 3.0, 4.0], 2, 2);
    assert_eq!(t.len(), 2);
    assert!((t[1] - 2.0_f64.ln()).abs() < 1e-12); // all mass on one of two: ln 2
}

#[test]
fn theil_d_splits_total_into_between_and_within() {
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0];
    let part = [0, 0, 0, 1, 1, 1];
    let d = theil_d(&y, 6, 1, &part).unwrap();
    assert!((d.t[0] - (d.bg[0] + d.wg[0])).abs() < 1e-15);
    assert!(d.bg[0] > 0.0 && d.wg[0] > 0.0);
    // every observation its own group: all inequality is between groups
    let own: Vec<u32> = (0..6).collect();
    let all = theil_d(&y, 6, 1, &own).unwrap();
    assert!((all.bg[0] - all.t[0]).abs() < 1e-12);
    assert!(all.wg[0].abs() < 1e-12);
    // a single group has no between-group inequality
    let one = theil_d(&y, 6, 1, &[0; 6]).unwrap();
    assert!(one.bg[0].abs() < 1e-12);
    assert!(theil_d(&y, 6, 1, &[0, 1]).is_err());
}

#[test]
fn theil_d_sim_pvalues_are_valid_and_thread_independent() {
    let y: Vec<f64> = (0..24).map(|i| 1.0 + f64::from((i * 7) % 13)).collect();
    let part: Vec<u32> = (0..24).map(|i| i / 6).collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| theil_d_sim(&y, 24, 1, &part, 199, 5).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a, b);
    assert!(a[0] > 0.0 && a[0] <= 1.0);
}

#[test]
fn spatial_gini_splits_inequality_into_neighbor_and_distant_pairs() {
    // path 0-1-2-3 with x = 1, 2, 4, 8: neighbor pair gaps 1, 2, 4 (counted both ways)
    let g = path_graph(4);
    let x = [1.0, 2.0, 4.0, 8.0];
    let s = spatial_gini(&x, &g, 0, 0).unwrap();
    assert!((s.g - gini(&x)).abs() < 1e-15);
    assert!((s.wg - 14.0).abs() < 1e-12);
    // all ordered pairs: 2 * (1 + 3 + 7 + 2 + 6 + 4) = 46
    assert!((s.dtotal - 46.0).abs() < 1e-12);
    assert!((s.wcg - 32.0).abs() < 1e-12);
    let den = 15.0 / 4.0 * 2.0 * 16.0;
    assert!((s.wcg_share - 32.0 / den).abs() < 1e-12);
    // n_n = 3 neighbor pairs, n_d = 6 - 3
    assert!((s.polarization - (32.0 / 14.0) * 1.0).abs() < 1e-12);
    assert!(s.p_sim.is_nan() && s.wcgp.is_empty());
}

#[test]
fn spatial_gini_permutations_are_seeded_and_thread_independent() {
    let g = path_graph(20);
    let x: Vec<f64> = (0..20).map(|i| 1.0 + f64::from((i * 5) % 9)).collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| spatial_gini(&x, &g, 199, 11).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.wcgp, b.wcgp);
    assert_eq!(a.p_sim, b.p_sim);
    assert!(a.p_sim > 0.0 && a.p_sim <= 1.0);
    assert!(a.s_wcg > 0.0 && (a.z_wcg.is_finite()));
    assert_eq!(a.polarization_sim.len(), 199);
    assert!(spatial_gini(&x, &path_graph(5), 0, 0).is_err());
}
