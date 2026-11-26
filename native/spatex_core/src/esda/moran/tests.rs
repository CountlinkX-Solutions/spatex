use super::*;
use crate::graph::{Csr, Transform};

fn path_graph(n: usize, extra_islands: usize) -> Csr {
    let mut focal = Vec::new();
    let mut neighbor = Vec::new();
    for i in 0..n.saturating_sub(1) {
        focal.extend([i as u32, i as u32 + 1]);
        neighbor.extend([i as u32 + 1, i as u32]);
    }
    let weight = vec![1.0; focal.len()];
    Csr::from_arrays(n + extra_islands, &focal, &neighbor, &weight).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * 1.0_f64.max(b.abs())
}

#[test]
fn path_with_island_matches_esda_probe() {
    let g = path_graph(3, 1).transform(Transform::R);
    let m = moran(&[1.0, 2.0, 4.0, 7.0], &g, true).unwrap();
    assert!(close(m.i, 0.2857142857142857, 1e-12));
    assert!(close(m.ei, -1.0 / 3.0, 1e-12));
    assert!(close(m.vi_norm, 0.2222222222222222, 1e-12));
    assert!(close(m.vi_rand, 0.2222222222222222, 1e-12));
    assert!(close(m.z_norm, 1.313198307917874, 1e-9));
    assert!(close(m.p_norm, 0.18911613402108862, 1e-9));
    assert!(close(m.p_rand, 0.18911613402108862, 1e-9));
}

#[test]
fn one_tailed_p_is_half_of_two_tailed() {
    let g = path_graph(6, 0).transform(Transform::R);
    let y = [1.0, 3.0, 2.0, 8.0, 5.0, 9.0];
    let two = moran(&y, &g, true).unwrap();
    let one = moran(&y, &g, false).unwrap();
    assert!(close(two.p_norm, 2.0 * one.p_norm, 1e-12));
    assert!(close(two.p_rand, 2.0 * one.p_rand, 1e-12));
    assert_eq!(two.i, one.i);
}

#[test]
fn two_nodes_give_nan_randomization_moments() {
    let g = path_graph(2, 0).transform(Transform::R);
    let m = moran(&[1.0, 3.0], &g, true).unwrap();
    assert!(close(m.i, -1.0, 1e-12));
    assert!(close(m.ei, -1.0, 1e-12));
    assert!(close(m.vi_norm, 0.0, 1e-12));
    assert!(m.vi_rand.is_nan() && m.se_i_rand.is_nan());
    assert!(m.z_rand.is_nan() && m.p_rand.is_nan());
}

#[test]
fn three_nodes_give_nan_randomization_moments() {
    let g = path_graph(3, 0).transform(Transform::R);
    let m = moran(&[1.0, 5.0, 2.0], &g, true).unwrap();
    assert!(m.vi_rand.is_nan() && m.p_rand.is_nan());
    assert!(m.vi_norm.is_finite());
}

#[test]
fn invalid_inputs_are_rejected() {
    let g = path_graph(4, 0).transform(Transform::R);
    assert!(moran(&[1.0, 2.0, 3.0], &g, true)
        .unwrap_err()
        .contains("length"));
    assert!(moran(&[2.0; 4], &g, true).unwrap_err().contains("constant"));
    assert!(moran(&[1.0, f64::NAN, 3.0, 4.0], &g, true)
        .unwrap_err()
        .contains("finite"));
    let lone = Csr::from_arrays(1, &[], &[], &[]).unwrap();
    assert!(moran(&[1.0], &lone, true).is_err());
    let islands = Csr::from_arrays(3, &[], &[], &[]).unwrap();
    assert!(moran(&[1.0, 2.0, 3.0], &islands, true)
        .unwrap_err()
        .contains("no weights"));
}

#[test]
fn identity_permutation_reproduces_the_observed_statistic_exactly() {
    let g = path_graph(7, 1).transform(Transform::R);
    let y = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];
    let m = moran(&y, &g, true).unwrap();
    let ctx = MoranStat::new(&y, &g).unwrap();
    assert_eq!(ctx.statistic(&ctx.z).to_bits(), m.i.to_bits());
}

#[test]
fn permutation_test_is_deterministic_and_bounded() {
    let g = path_graph(12, 0).transform(Transform::R);
    let y: Vec<f64> = (0..12).map(|i| ((i * 5) % 7) as f64).collect();
    let a = moran_permutations(&y, &g, 199, 5).unwrap();
    let b = moran_permutations(&y, &g, 199, 5).unwrap();
    assert_eq!(a.sim.len(), 199);
    assert_eq!(
        a.sim.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.sim.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!((1.0 / 200.0..=1.0).contains(&a.p_sim));
    let c = moran_permutations(&y, &g, 199, 6).unwrap();
    assert_ne!(a.sim, c.sim);
}

#[test]
fn simulated_mean_is_close_to_the_analytic_expectation() {
    let g = path_graph(30, 0).transform(Transform::R);
    let y: Vec<f64> = (0..30).map(|i| ((i * 11) % 13) as f64).collect();
    let m = moran(&y, &g, true).unwrap();
    let s = moran_permutations(&y, &g, 4999, 1).unwrap();
    let band = 5.0 * s.summary.se_sim / (4999.0_f64).sqrt();
    assert!((s.summary.ei_sim - m.ei).abs() < band);
}

#[test]
fn strongly_clustered_values_get_the_smallest_p() {
    let g = path_graph(20, 0).transform(Transform::R);
    let y: Vec<f64> = (0..20).map(f64::from).collect();
    let s = moran_permutations(&y, &g, 999, 3).unwrap();
    assert_eq!(s.p_sim, 1.0 / 1000.0);
}

#[test]
fn bivariate_of_a_variable_with_itself_equals_row_standardised_moran() {
    let g = path_graph(9, 0).transform(Transform::R);
    let y = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0, 5.5];
    let bv = moran_bv(&y, &y, &g).unwrap();
    let m = moran(&y, &g, true).unwrap();
    assert!(close(bv, m.i, 1e-12));
}

#[test]
fn bivariate_matches_hand_computation() {
    // Path 0-1-2, R: W zy = [zy1, (zy0 + zy2) / 2, zy1].
    let g = path_graph(3, 0).transform(Transform::R);
    let x = [1.0, 2.0, 4.0];
    let y = [3.0, 1.0, 2.0];
    let zx = [-4.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0];
    let sx = (zx.iter().map(|v| v * v).sum::<f64>() / 2.0).sqrt();
    let zy = [1.0, -1.0, 0.0];
    let lag = [zy[1], (zy[0] + zy[2]) / 2.0, zy[1]];
    let expected = (0..3).map(|i| zx[i] / sx * lag[i]).sum::<f64>() / 2.0;
    assert!(close(moran_bv(&x, &y, &g).unwrap(), expected, 1e-12));
}

#[test]
fn bivariate_rejects_constant_and_mismatched_inputs() {
    let g = path_graph(4, 0).transform(Transform::R);
    assert!(moran_bv(&[1.0; 4], &[1.0, 2.0, 3.0, 4.0], &g)
        .unwrap_err()
        .contains("constant"));
    assert!(moran_bv(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0], &g).is_err());
}

#[test]
fn bivariate_permutations_permute_only_y() {
    let g = path_graph(10, 0).transform(Transform::R);
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    let y: Vec<f64> = (0..10).map(|i| ((i * 3) % 7) as f64).collect();
    let a = moran_bv_permutations(&x, &y, &g, 99, 8).unwrap();
    let b = moran_bv_permutations(&x, &y, &g, 99, 8).unwrap();
    assert_eq!(a.sim, b.sim);
    assert_eq!(a.sim.len(), 99);
}

#[test]
fn assuncao_rate_matches_hand_computation() {
    let y = assuncao_rate(&[1.0, 4.0], &[10.0, 20.0]).unwrap();
    assert!(close(y[0], -0.7559289460184543, 1e-12));
    assert!(close(y[1], 0.365148371670111, 1e-12));
}

#[test]
fn assuncao_rate_validates_inputs() {
    assert!(assuncao_rate(&[1.0], &[1.0, 2.0]).is_err());
    assert!(assuncao_rate(&[1.0, 2.0], &[0.0, 2.0])
        .unwrap_err()
        .contains("base"));
    assert!(assuncao_rate(&[-1.0, 2.0], &[1.0, 2.0])
        .unwrap_err()
        .contains("events"));
}

#[test]
fn raw_rate_is_events_over_base() {
    assert_eq!(
        rate(&[1.0, 3.0], &[2.0, 4.0], false).unwrap(),
        vec![0.5, 0.75]
    );
    assert_eq!(
        rate(&[1.0, 4.0], &[10.0, 20.0], true).unwrap(),
        assuncao_rate(&[1.0, 4.0], &[10.0, 20.0]).unwrap()
    );
    assert!(rate(&[1.0], &[0.0], false).is_err());
}
