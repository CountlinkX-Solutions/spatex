use super::*;
use crate::graph::{Csr, Transform};

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

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * 1.0_f64.max(b.abs())
}

const Y8: [f64; 8] = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];

#[test]
fn path_with_islands_matches_esda() {
    let g = path_graph(4, 4);
    let a = general_g(&Y8, &g).unwrap();
    assert!(close(a.g, 0.03178484107579462, 1e-12));
    assert!(close(a.eg, 0.10714285714285714, 1e-12));
    assert!(close(a.vg, 0.003553183069247571, 1e-10));
    assert!(close(a.z_norm, -1.2642142006042831, 1e-10));
    assert!(close(a.p_norm, 0.10307657760853384, 1e-10));
    assert!(close(a.den_sum, 818.0, 1e-12));
}

#[test]
fn weights_are_binarized_internally() {
    let g = path_graph(4, 4);
    let row = g.transform(Transform::R);
    let a = general_g(&Y8, &g).unwrap();
    let b = general_g(&Y8, &row).unwrap();
    assert_eq!(a.g, b.g);
    assert_eq!(a.vg, b.vg);
}

#[test]
fn negative_or_degenerate_input_is_rejected() {
    let g = path_graph(5, 0);
    assert!(general_g(&[1.0, -2.0, 3.0, 4.0, 5.0], &g)
        .unwrap_err()
        .contains("non-negative"));
    assert!(general_g(&[0.0; 5], &g).is_err());
    assert!(general_g(&[1.0, f64::INFINITY, 3.0, 4.0, 5.0], &g).is_err());
    assert!(general_g(&[1.0, 2.0], &g).unwrap_err().contains("length"));
}

#[test]
fn complete_graph_has_non_positive_variance_and_nan_inference() {
    let n = 6;
    let (mut f, mut t) = (vec![], vec![]);
    for i in 0..n {
        for j in 0..n {
            if i != j {
                f.push(i as u32);
                t.push(j as u32);
            }
        }
    }
    let g = Csr::from_arrays(n, &f, &t, &vec![1.0; f.len()]).unwrap();
    let a = general_g(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &g).unwrap();
    assert!(close(a.g, 1.0, 1e-12));
    assert!(a.z_norm.is_nan() && a.p_norm.is_nan());
}

#[test]
fn small_n_gives_nan_variance() {
    let g = path_graph(3, 0);
    let a = general_g(&[1.0, 5.0, 2.0], &g).unwrap();
    assert!(a.vg.is_nan() && a.z_norm.is_nan());
}

#[test]
fn permutations_are_seeded_and_thread_count_independent() {
    let g = path_graph(4, 4);
    let a = general_g_permutations(&Y8, &g, 199, 7).unwrap();
    assert_eq!(a.sim, general_g_permutations(&Y8, &g, 199, 7).unwrap().sim);
    for threads in [1, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let t = pool.install(|| general_g_permutations(&Y8, &g, 199, 7).unwrap());
        assert_eq!(t.sim, a.sim, "{threads} threads");
    }
    assert!(a.p_sim >= 1.0 / 200.0 && a.p_sim <= 1.0);
}

#[test]
fn identity_permutation_reproduces_g_and_p_z_sim_uses_one_minus_cdf() {
    let g = path_graph(4, 4);
    let ctx = GStat::new(&Y8, &g).unwrap();
    assert_eq!(ctx.statistic(&Y8), general_g(&Y8, &g).unwrap().g);
    let t = general_g_permutations(&Y8, &g, 199, 3).unwrap();
    let expected = 1.0 - crate::perm::norm_cdf(t.summary.z_sim.abs());
    assert!(close(t.summary.p_z_sim, expected, 1e-15));
}
