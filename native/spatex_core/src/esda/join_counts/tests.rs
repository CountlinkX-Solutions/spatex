use super::*;
use crate::graph::Csr;

fn path_graph(n: usize, islands: usize) -> Csr {
    let mut focal = Vec::new();
    let mut neighbor = Vec::new();
    for i in 0..n.saturating_sub(1) {
        focal.extend([i as u32, i as u32 + 1]);
        neighbor.extend([i as u32 + 1, i as u32]);
    }
    let weight = vec![1.0; focal.len()];
    Csr::from_arrays(n + islands, &focal, &neighbor, &weight).unwrap()
}

/// 4x4 rook lattice.
fn lattice() -> Csr {
    let (mut f, mut t) = (vec![], vec![]);
    for r in 0..4u32 {
        for c in 0..4u32 {
            let i = r * 4 + c;
            if c < 3 {
                f.extend([i, i + 1]);
                t.extend([i + 1, i]);
            }
            if r < 3 {
                f.extend([i, i + 4]);
                t.extend([i + 4, i]);
            }
        }
    }
    Csr::from_arrays(16, &f, &t, &vec![1.0; f.len()]).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * 1.0_f64.max(b.abs())
}

fn checker() -> Vec<f64> {
    (0..16).map(|i| ((i / 4 + i % 4) % 2) as f64).collect()
}

#[test]
fn checkerboard_lattice_matches_esda() {
    let a = join_counts(&checker(), &lattice(), true).unwrap();
    assert_eq!((a.j, a.bb, a.ww, a.bw), (24.0, 0.0, 0.0, 24.0));
    assert_eq!(a.crosstab, [0.0, 12.0, 12.0, 0.0]);
    assert_eq!(a.expected, [6.0, 6.0, 6.0, 6.0]);
    assert!(close(a.chi2, 20.166666666666668, 1e-12));
    assert!(close(a.chi2_p, 7.09790833090828e-06, 1e-9));
    assert_eq!(a.chi2_dof, 1.0);
    assert_eq!((a.autocorr_pos, a.autocorr_neg), (0.0, 24.0));
}

#[test]
fn half_lattice_matches_esda() {
    let y: Vec<f64> = (0..16).map(|i| if i >= 8 { 1.0 } else { 0.0 }).collect();
    let a = join_counts(&y, &lattice(), true).unwrap();
    assert_eq!((a.bb, a.ww, a.bw), (10.0, 10.0, 4.0));
    assert_eq!(a.crosstab, [10.0, 2.0, 2.0, 10.0]);
    assert!(close(a.chi2, 8.166666666666666, 1e-12));
    assert!(close(a.chi2_p, 0.004266724822176128, 1e-9));
}

#[test]
fn island_rows_are_dropped_or_kept_like_esda() {
    let g = path_graph(4, 4);
    let y = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let drop = join_counts(&y, &g, true).unwrap();
    assert_eq!((drop.j, drop.bb, drop.ww, drop.bw), (3.0, 1.0, 0.0, 2.0));
    assert_eq!(drop.crosstab, [0.0, 1.0, 1.0, 1.0]);
    assert!(close(drop.expected[0], 1.0 / 3.0, 1e-12));
    let keep = join_counts(&y, &g, false).unwrap();
    assert_eq!((keep.j, keep.bb, keep.ww, keep.bw), (3.0, 2.0, 1.0, 2.0));
    assert_eq!(keep.crosstab, [1.0, 1.0, 1.0, 2.0]);
}

#[test]
fn non_binary_constant_or_mismatched_input_is_rejected() {
    let g = path_graph(4, 0);
    assert!(join_counts(&[0.0, 1.0, 2.0, 1.0], &g, true)
        .unwrap_err()
        .contains("binary"));
    assert!(join_counts(&[0.0, 1.0, f64::NAN, 1.0], &g, true).is_err());
    assert!(join_counts(&[1.0; 4], &g, true).is_err());
    assert!(join_counts(&[0.0, 1.0, 0.0], &g, true).is_err());
    let empty = Csr::from_arrays(3, &[], &[], &[]).unwrap();
    assert!(join_counts(&[0.0, 1.0, 0.0], &empty, true).is_err());
}

#[test]
fn chi2_contingency_applies_yates_only_through_half() {
    // Probed with scipy: [[10, 2], [2, 10]] -> 8.166666666666666 (uncorrected 10.6667).
    let (chi2, _p, dof, expected) = chi2_contingency([10.0, 2.0, 2.0, 10.0]).unwrap();
    assert!(close(chi2, 8.166666666666666, 1e-12));
    assert_eq!(dof, 1.0);
    assert_eq!(expected, [6.0, 6.0, 6.0, 6.0]);
    // Observed equals expected: no correction pushes past it.
    assert_eq!(chi2_contingency([6.0, 6.0, 6.0, 6.0]).unwrap().0, 0.0);
    assert!(chi2_contingency([0.0, 0.0, 3.0, 4.0]).is_err());
}

#[test]
fn permutations_are_seeded_and_thread_count_independent() {
    let g = lattice();
    let y: Vec<f64> = (0..16)
        .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
        .collect();
    let a = join_counts_permutations(&y, &g, true, 199, 5).unwrap();
    let b = join_counts_permutations(&y, &g, true, 199, 5).unwrap();
    assert_eq!(a.sim_bb, b.sim_bb);
    assert_eq!(a.sim_chi2, b.sim_chi2);
    for threads in [1, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let t = pool.install(|| join_counts_permutations(&y, &g, true, 199, 5).unwrap());
        assert_eq!(t.sim_bb, a.sim_bb, "{threads} threads");
        assert_eq!(t.sim_chi2, a.sim_chi2, "{threads} threads");
    }
    for p in [
        a.p_sim_bb,
        a.p_sim_bw,
        a.p_sim_chi2,
        a.p_sim_autocorr_pos,
        a.p_sim_autocorr_neg,
    ] {
        assert!((1.0 / 200.0..=1.0).contains(&p));
    }
    assert_eq!(a.sim_bb.len(), 199);
}

#[test]
fn p_values_are_upper_tail_and_unfolded() {
    let g = lattice();
    let y = checker();
    let obs = join_counts(&y, &g, true).unwrap();
    let t = join_counts_permutations(&y, &g, true, 99, 1).unwrap();
    // Checkerboard has the maximum possible bw: every permutation is <= it, p_sim_bw is large.
    let larger = t.sim_bw.iter().filter(|&&s| s >= obs.bw).count();
    assert_eq!(t.p_sim_bw, (larger as f64 + 1.0) / 100.0);
    assert_eq!(
        t.min_bb,
        t.sim_bb.iter().cloned().fold(f64::INFINITY, f64::min)
    );
    assert!(close(t.mean_bb, t.sim_bb.iter().sum::<f64>() / 99.0, 1e-12));
}

#[test]
fn permutations_redraw_when_a_table_has_a_zero_expected_cell() {
    // Black nodes can land on the island so every joined node is white: the draw is discarded.
    let g = path_graph(3, 1);
    let y = [1.0, 0.0, 1.0, 1.0];
    let t = join_counts_permutations(&y, &g, true, 300, 9).unwrap();
    assert_eq!(t.sim_chi2.len(), 300);
    assert!(t.sim_chi2.iter().all(|c| c.is_finite()));
}

#[test]
fn autocorr_chi2_uses_the_pearson_statistic() {
    // pos 10 vs mean 8, neg 14 vs mean 16: (4/8 + 4/16) = 0.75.
    let p = autocorr_chi2_p(10.0, 14.0, 8.0, 16.0);
    assert!(close(p, crate::perm::erfc((0.75_f64 / 2.0).sqrt()), 1e-12));
    assert!(autocorr_chi2_p(1.0, 1.0, 0.0, 2.0).is_nan());
}
