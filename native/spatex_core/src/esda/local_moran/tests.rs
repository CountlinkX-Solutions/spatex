use super::*;
use crate::graph::{Csr, Transform};
use crate::perm::crand::Settings;
use crate::perm::Alternative;

/// Path 0-1-2-3-4 plus island 5 (an esda probe with an island).
fn path_with_island() -> Csr {
    let focal = [0, 1, 1, 2, 2, 3, 3, 4];
    let neighbor = [1, 0, 2, 1, 3, 2, 4, 3];
    Csr::from_arrays(6, &focal, &neighbor, &[1.0; 8]).unwrap()
}

const Y: [f64; 6] = [1.0, 5.0, 2.0, 8.0, 3.0, 4.0];

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * 1.0_f64.max(b.abs())
}

fn settings(permutations: usize, keep: bool) -> Settings {
    Settings {
        permutations,
        seed: 12345,
        alternative: Alternative::Directed,
        island_weight: 0.0,
        keep,
    }
}

#[test]
fn island_probe_matches_esda() {
    let g = path_with_island().transform(Transform::R);
    let m = local_moran(&Y, &g, false).unwrap();
    let expected = [
        -0.53603604,
        -0.44144144,
        -0.79279279,
        -0.9009009,
        -0.56306306,
        0.0,
    ];
    for (a, e) in m.is.iter().zip(expected) {
        assert!(close(*a, e, 1e-8), "{:?}", m.is);
    }
    assert_eq!(m.q, vec![2, 4, 2, 4, 2, 4]);
    assert_eq!(m.lag[5], 0.0);
}

#[test]
fn geoda_quads_swap_low_high_and_low_low() {
    let g = path_with_island().transform(Transform::R);
    let m = local_moran(&Y, &g, true).unwrap();
    assert_eq!(m.q, vec![3, 4, 3, 4, 3, 4]);
}

#[test]
fn analytic_moments_follow_sokal() {
    let g = path_with_island().transform(Transform::R);
    let m = local_moran(&Y, &g, false).unwrap();
    let mo = m.moments.as_ref().unwrap();
    // Total randomization: EI = -wi / (n - 1); an island has wi = 0.
    for &ei in &mo.ei[..5] {
        assert!(close(ei, -1.0 / 5.0, 1e-12));
    }
    assert_eq!(mo.ei[5], 0.0);
    assert_eq!(mo.vi[5], 0.0);
    assert_eq!(mo.vic[5], 0.0);
    // Conditional: EIc = -(z^2 wi) / ((n-1) m2).
    let n = 6.0;
    let m2 = m.z.iter().map(|z| z * z).sum::<f64>() / n;
    for i in 0..5 {
        assert!(close(mo.eic[i], -(m.z[i] * m.z[i]) / (5.0 * m2), 1e-12));
    }
    // Hand-evaluated VIc of site 0 (wi = 1, wi2 = 1): (z/m2)^2 n/(n-2) (1 - 1/5) (m2 - z^2/5).
    let z0 = m.z[0];
    let vic0 = (z0 / m2).powi(2) * (n / 4.0) * (1.0 - 1.0 / 5.0) * (m2 - z0 * z0 / 5.0);
    assert!(close(mo.vic[0], vic0, 1e-12));
}

#[test]
fn self_weights_add_the_diagonal_term() {
    let plain = Csr::from_arrays(3, &[0, 1, 1, 2], &[1, 0, 2, 1], &[1.0; 4]).unwrap();
    let selfw =
        Csr::from_arrays(3, &[0, 0, 1, 1, 1, 2, 2], &[0, 1, 0, 1, 2, 1, 2], &[1.0; 7]).unwrap();
    let y = [1.0, 4.0, 2.0];
    let a = local_moran(&y, &plain, false).unwrap();
    let b = local_moran(&y, &selfw, false).unwrap();
    let den: f64 = a.z.iter().map(|z| z * z).sum();
    for i in 0..3 {
        let extra = 2.0 * a.z[i] * a.z[i] / den;
        assert!(close(b.is[i], a.is[i] + extra, 1e-12));
    }
}

#[test]
fn bivariate_with_x_equal_y_is_the_univariate_statistic() {
    let g = path_with_island().transform(Transform::R);
    let uni = local_moran(&Y, &g, false).unwrap();
    let bv = local_moran_bv(&Y, &Y, &g, false).unwrap();
    assert_eq!(uni.is, bv.is);
    assert_eq!(uni.q, bv.q);
    assert!(bv.moments.is_none());
}

#[test]
fn bivariate_quadrant_uses_x_and_the_lag_of_y() {
    let g = path_with_island().transform(Transform::R);
    let x = [9.0, 1.0, 9.0, 1.0, 9.0, 1.0];
    let bv = local_moran_bv(&x, &Y, &g, false).unwrap();
    let zy_lag = g.lag(&bv.zy.clone().unwrap()).unwrap();
    for (i, &zl) in zy_lag.iter().enumerate() {
        let high_x = bv.z[i] > 0.0;
        let high_lag = zl > 0.0;
        let q = match (high_x, high_lag) {
            (true, true) => 1,
            (false, true) => 2,
            (false, false) => 3,
            (true, false) => 4,
        };
        assert_eq!(bv.q[i], q);
        assert_eq!(bv.lag[i], zl);
    }
}

#[test]
fn invalid_input_is_rejected() {
    let g = path_with_island().transform(Transform::R);
    assert!(local_moran(&[3.0; 6], &g, false)
        .unwrap_err()
        .contains("constant"));
    assert!(local_moran(&Y[..5], &g, false)
        .unwrap_err()
        .contains("length"));
    let mut bad = Y;
    bad[2] = f64::NAN;
    assert!(local_moran(&bad, &g, false)
        .unwrap_err()
        .contains("non-finite"));
    assert!(local_moran_bv(&[1.0; 6], &Y, &g, false).is_err());
    let tiny = Csr::from_arrays(2, &[0, 1], &[1, 0], &[1.0; 2]).unwrap();
    assert!(local_moran(&[1.0, 2.0], &tiny, false).is_err());
}

#[test]
fn island_gets_the_smallest_p_value_and_nan_z_sim() {
    let g = path_with_island().transform(Transform::R);
    let out = local_moran_permutations(&Y, &g, &settings(99, true)).unwrap();
    assert_eq!(out.p_sim[5], 0.01);
    assert!(out.summary[5].z_sim.is_nan());
    assert!(out.p_sim.iter().all(|&p| (0.01..=1.0).contains(&p)));
}

#[test]
fn permutations_are_deterministic_and_keep_does_not_change_p() {
    let g = path_with_island().transform(Transform::R);
    let a = local_moran_permutations(&Y, &g, &settings(199, true)).unwrap();
    let b = local_moran_permutations(&Y, &g, &settings(199, false)).unwrap();
    assert_eq!(a.p_sim, b.p_sim);
    assert!(b.sims.is_none());
    let c = local_moran_permutations(&Y, &g, &settings(199, true)).unwrap();
    assert_eq!(a.sims, c.sims);
    let mut other = settings(199, true);
    other.seed = 1;
    assert_ne!(
        a.sims,
        local_moran_permutations(&Y, &g, &other).unwrap().sims
    );
}

#[test]
fn simulated_mean_matches_the_conditional_expectation() {
    // Neighbors drawn without replacement from the other sites have mean -z_i / (n - 1), so
    // E[rstat_i] = -z_i^2 wi / sum(z^2) for row-standardised weights.
    let g = path_with_island().transform(Transform::R);
    let m = local_moran(&Y, &g, false).unwrap();
    let out = local_moran_permutations(&Y, &g, &settings(20_000, false)).unwrap();
    let den: f64 = m.z.iter().map(|z| z * z).sum();
    for i in 0..5 {
        let expected = -m.z[i] * m.z[i] / den;
        let s = &out.summary[i];
        assert!(
            (s.ei_sim - expected).abs() <= 4.0 * s.se_sim / (20_000f64).sqrt(),
            "site {i}: {} vs {expected}",
            s.ei_sim
        );
    }
}

#[test]
fn bivariate_permutations_run_and_island_is_minimal() {
    let g = path_with_island().transform(Transform::R);
    let x = [2.0, 7.0, 1.0, 3.0, 9.0, 5.0];
    let out = local_moran_bv_permutations(&x, &Y, &g, &settings(99, false)).unwrap();
    assert_eq!(out.p_sim[5], 0.01);
}

#[test]
fn cluster_labels_use_a_strict_threshold() {
    let q = [1u8, 2, 3, 4, 1];
    let p = [0.01, 0.049, 0.05, 0.001, 0.2];
    assert_eq!(
        cluster_labels(&q, &p, 0.05),
        vec![
            "High-High",
            "Low-High",
            "Insignificant",
            "High-Low",
            "Insignificant"
        ]
    );
}

#[test]
fn exact_ties_with_the_observed_statistic_do_not_depend_on_rounding() {
    // Site 1 of the path has neighbors 0 and 2, the two lowest values, so its observed
    // statistic is the minimum of every reference distribution and ties it whenever that pair
    // is drawn. Counting ties as >= must give the smallest p for every seed.
    let g = path_with_island().transform(Transform::R);
    for seed in 0..20 {
        let mut s = settings(99, false);
        s.seed = seed;
        let out = local_moran_permutations(&Y, &g, &s).unwrap();
        assert_eq!(out.p_sim[1], 0.01, "seed {seed}");
    }
}
