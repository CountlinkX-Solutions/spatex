use super::*;
use crate::graph::{Csr, Transform};

/// Path 0-1-2-3 plus `islands` isolated nodes.
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
fn path_with_islands_matches_esda_row_standardized() {
    let g = path_graph(4, 4).transform(Transform::R);
    let a = geary(&Y8, &g).unwrap();
    assert!(close(a.c, 0.42470441817050403, 1e-12));
    assert_eq!(a.ec, 1.0);
    assert!(close(a.vc_norm, 0.4583333333333333, 1e-12));
    assert!(close(a.vc_rand, 0.494568343847255, 1e-12));
    assert!(close(a.z_norm, -0.8497678910545847, 1e-12));
    assert!(close(a.z_rand, -0.8180462937397502, 1e-12));
    assert!(close(a.p_norm, 0.19772707226622627, 1e-12));
    assert!(close(a.p_rand, 0.20666537685703268, 1e-12));
}

#[test]
fn path_with_islands_matches_esda_binary() {
    let g = path_graph(4, 4).transform(Transform::B);
    let a = geary(&Y8, &g).unwrap();
    assert!(close(a.c, 0.4472101223812487, 1e-12));
    assert!(close(a.vc_norm, 0.4691358024691358, 1e-12));
    assert!(close(a.vc_rand, 0.5100679439756028, 1e-12));
    assert!(close(a.p_norm, 0.209813297598756, 1e-12));
    assert!(close(a.p_rand, 0.21946270497531745, 1e-12));
}

#[test]
fn small_n_gives_nan_randomization_moments() {
    let g = path_graph(3, 0).transform(Transform::R);
    let a = geary(&[1.0, 5.0, 2.0], &g).unwrap();
    assert!(a.vc_rand.is_nan() && a.p_rand.is_nan());
    assert!(a.vc_norm.is_finite());
}

#[test]
fn constant_or_invalid_input_is_rejected() {
    let g = path_graph(4, 0).transform(Transform::R);
    assert!(geary(&[2.0; 4], &g).unwrap_err().contains("constant"));
    assert!(geary(&[1.0, f64::NAN, 2.0, 3.0], &g).is_err());
    assert!(geary(&[1.0, 2.0, 3.0], &g).unwrap_err().contains("length"));
    let empty = Csr::from_arrays(3, &[], &[], &[]).unwrap();
    assert!(geary(&[1.0, 2.0, 3.0], &empty).is_err());
}

#[test]
fn permutations_are_seeded_and_thread_count_independent() {
    let g = path_graph(4, 4).transform(Transform::R);
    let a = geary_permutations(&Y8, &g, 199, 7).unwrap();
    let b = geary_permutations(&Y8, &g, 199, 7).unwrap();
    assert_eq!(a.sim, b.sim);
    assert_ne!(a.sim, geary_permutations(&Y8, &g, 199, 8).unwrap().sim);
    for threads in [1, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let t = pool.install(|| geary_permutations(&Y8, &g, 199, 7).unwrap());
        assert_eq!(t.sim, a.sim, "{threads} threads");
    }
    assert!(a.p_sim >= 1.0 / 200.0 && a.p_sim <= 1.0);
}

#[test]
fn identity_permutation_reproduces_the_statistic_bit_for_bit() {
    let g = path_graph(4, 4).transform(Transform::R);
    let ctx = GearyStat::new(&Y8, &g).unwrap();
    assert_eq!(ctx.statistic(&Y8), geary(&Y8, &g).unwrap().c);
}

#[test]
fn p_z_sim_is_one_tailed_on_abs_z() {
    let g = path_graph(4, 4).transform(Transform::R);
    let t = geary_permutations(&Y8, &g, 199, 3).unwrap();
    let expected = crate::perm::norm_sf(t.summary.z_sim.abs());
    assert!(close(t.summary.p_z_sim, expected, 1e-15));
}

#[test]
fn zero_permutations_are_rejected() {
    let g = path_graph(4, 0).transform(Transform::R);
    assert!(geary_permutations(&[1.0, 2.0, 4.0, 3.0], &g, 0, 1).is_err());
}
