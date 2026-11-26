use super::*;
use crate::esda::fdr::fdr;
use crate::graph::{Csr, Transform};
use crate::perm::crand::Settings;
use crate::perm::Alternative;

/// Path 0-1-2-3 with unit weights.
fn path4() -> Csr {
    Csr::from_arrays(4, &[0, 1, 1, 2, 2, 3], &[1, 0, 2, 1, 3, 2], &[1.0; 6]).unwrap()
}

/// Path 0-1-2-3-4 plus island 5.
fn path_island() -> Csr {
    Csr::from_arrays(
        6,
        &[0, 1, 1, 2, 2, 3, 3, 4],
        &[1, 0, 2, 1, 3, 2, 4, 3],
        &[1.0; 8],
    )
    .unwrap()
}

const Y4: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
const Y6: [f64; 6] = [1.0, 5.0, 2.0, 8.0, 3.0, 4.0];

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * 1.0_f64.max(b.abs())
}

fn opts(star: Star, transform: GTransform) -> Options {
    Options {
        star,
        transform,
        input_binary: false,
    }
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

/// Hand computation of the non-star statistic on the path: node 0 has the single neighbor 1.
#[test]
fn non_star_row_standardized_matches_closed_form() {
    let r = g_local(&Y4, &path4(), &opts(Star::Off, GTransform::R)).unwrap();
    assert!(!r.star);
    assert!(close(r.gs[0], 2.0 / 9.0));
    assert!(close(r.gs[1], 4.0 / 2.0 / 8.0));
    // N = n - 1 = 3, card = 1: E = 1/3, V = card (N - card) / (N - 1) / N^2 * var / mean^2.
    assert!(close(r.egs[0], 1.0 / 3.0));
    let var0 = (30.0 - 1.0) / 3.0 - 3.0 * 3.0;
    assert!(close(r.vgs[0], 1.0 * 2.0 / 2.0 / 9.0 * var0 / 9.0));
    assert!(close(r.zs[0], (r.gs[0] - r.egs[0]) / r.vgs[0].sqrt()));
    assert!(close(r.p_norm[0], norm_sf(r.zs[0].abs())));
    assert_eq!(r.y_sum, 10.0);
}

#[test]
fn star_true_on_zero_diagonal_uses_row_max_as_self_weight() {
    let r = g_local(&Y4, &path4(), &opts(Star::On, GTransform::R)).unwrap();
    assert!(r.star);
    // Row 0 becomes [1/2 self, 1/2 neighbor]; G* = (y0 + y1) / 2 / sum(y).
    assert!(close(r.gs[0], 1.5 / 10.0));
    assert!(close(r.gs[1], 2.0 / 10.0));
    // N = n, card = 1 (R): E = 1/4.
    assert!(close(r.egs[0], 0.25));
}

#[test]
fn explicit_self_weight_is_assigned_then_transformed() {
    let r = g_local(&Y4, &path4(), &opts(Star::Weight(0.5), GTransform::R)).unwrap();
    assert!(r.star);
    // Row 0: [0.5, 1] -> [1/3, 2/3].
    assert!(close(r.gs[0], (1.0 / 3.0 + 2.0 * 2.0 / 3.0) / 10.0));
    let b = g_local(&Y4, &path4(), &opts(Star::Weight(0.5), GTransform::B)).unwrap();
    // Binary: every nonzero weight becomes 1, so the self weight is 1 as well.
    assert!(close(b.gs[0], (1.0 + 2.0) / 10.0));
    assert!(close(b.egs[0], 2.0 / 4.0));
}

#[test]
fn binary_transform_star_true_assigns_unit_self_weight() {
    let r = g_local(&Y4, &path4(), &opts(Star::On, GTransform::B)).unwrap();
    assert!(close(r.gs[1], (1.0 + 2.0 + 3.0) / 10.0));
    assert!(close(r.egs[1], 3.0 / 4.0));
}

#[test]
fn binary_input_graph_gets_unit_self_weight_even_for_row_standardization() {
    // A binary graph that was already row-standardized (labelled B by the caller)
    // would have row max 1/2; esda assigns 1 whenever the input transform is binary.
    let g = path4().transform(Transform::R);
    let mut o = opts(Star::On, GTransform::R);
    o.input_binary = true;
    let r = g_local(&Y4, &g, &o).unwrap();
    assert!(close(r.gs[0], (1.0 + 2.0) / 2.0 / 10.0));
    o.input_binary = false;
    let r = g_local(&Y4, &g, &o).unwrap();
    // Row max of the R graph is 1 for the end nodes, 1/2 for the inner nodes.
    assert!(close(r.gs[0], (1.0 + 2.0) / 2.0 / 10.0));
    assert!(close(
        r.gs[1],
        (0.5 * 2.0 + 0.5 * 1.0 + 0.5 * 3.0) / 1.5 / 10.0
    ));
}

#[test]
fn infer_follows_the_diagonal() {
    let plain = g_local(&Y4, &path4(), &opts(Star::Infer, GTransform::R)).unwrap();
    assert!(!plain.star);
    // Graph with a nonzero diagonal: Infer -> star, Off removes it again.
    let g = Csr::from_arrays(
        4,
        &[0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        &[1.0; 10],
    )
    .unwrap();
    let inferred = g_local(&Y4, &g, &opts(Star::Infer, GTransform::R)).unwrap();
    assert!(inferred.star);
    let off = g_local(&Y4, &g, &opts(Star::Off, GTransform::R)).unwrap();
    assert!(!off.star);
    for (a, b) in off.gs.iter().zip(&plain.gs) {
        assert!(close(*a, *b));
    }
    let on = g_local(&Y4, &g, &opts(Star::On, GTransform::R)).unwrap();
    for (a, b) in on.gs.iter().zip(&inferred.gs) {
        assert!(close(*a, *b));
    }
}

#[test]
fn zero_weight_star_removes_the_diagonal() {
    let r = g_local(&Y4, &path4(), &opts(Star::Weight(0.0), GTransform::R)).unwrap();
    assert!(!r.star);
    assert!(close(r.gs[0], 2.0 / 9.0));
}

#[test]
fn island_has_zero_statistic_and_nan_z() {
    let r = g_local(&Y6, &path_island(), &opts(Star::Off, GTransform::R)).unwrap();
    assert_eq!(r.gs[5], 0.0);
    assert_eq!(r.egs[5], 0.0);
    assert!(r.zs[5].is_nan() && r.p_norm[5].is_nan());
    let star = g_local(&Y6, &path_island(), &opts(Star::On, GTransform::R)).unwrap();
    assert_eq!(star.gs[5], 0.0, "row max of an island is 0");
}

#[test]
fn invalid_input_is_rejected() {
    let o = opts(Star::Off, GTransform::R);
    assert!(g_local(&[1.0, 2.0, -3.0, 4.0], &path4(), &o)
        .unwrap_err()
        .contains("non-negative"));
    assert!(g_local(&[0.0; 4], &path4(), &o)
        .unwrap_err()
        .contains("sum"));
    assert!(g_local(&[1.0, 2.0, f64::NAN, 4.0], &path4(), &o)
        .unwrap_err()
        .contains("non-finite"));
    assert!(g_local(&[1.0, 2.0], &path4(), &o)
        .unwrap_err()
        .contains("length"));
}

#[test]
fn transform_parse_accepts_r_and_b_only() {
    assert_eq!(GTransform::parse("R").unwrap(), GTransform::R);
    assert_eq!(GTransform::parse("b").unwrap(), GTransform::B);
    assert!(GTransform::parse("d").is_err());
    assert_eq!(Star::parse("infer", 0.0).unwrap(), Star::Infer);
    assert_eq!(Star::parse("weight", 0.5).unwrap(), Star::Weight(0.5));
    assert!(Star::parse("maybe", 0.0).is_err());
}

#[test]
fn permutations_are_reproducible_and_keep_flag_only_drops_sims() {
    let r = g_local(&Y6, &path_island(), &opts(Star::Off, GTransform::R)).unwrap();
    let a = g_local_permutations(&Y6, &r, &settings(99, true)).unwrap();
    let b = g_local_permutations(&Y6, &r, &settings(99, false)).unwrap();
    assert_eq!(a.p_sim, b.p_sim);
    assert!(b.sims.is_none());
    assert_eq!(a.sims.as_ref().unwrap().len(), 6 * 99);
    for p in &a.p_sim {
        assert!((1.0 / 100.0..=1.0).contains(p));
    }
    // PySAL quirk: an island's replicates are all 0, so it gets the smallest p-value.
    assert_eq!(a.p_sim[5], 0.01);
}

/// Under conditional randomization the focal value is fixed and each neighbor has mean
/// `(sum - y_i)/(n - 1)`, so the replicate mean is known in closed form.
#[test]
fn replicate_means_match_the_conditional_expectation() {
    let y: Vec<f64> = (0..12).map(|i| 1.0 + f64::from(i % 5) * 1.7).collect();
    let focal: Vec<u32> = (0..12).flat_map(|i| [i, i]).collect();
    let neighbor: Vec<u32> = (0..12)
        .flat_map(|i| [(i + 1) % 12, (i + 11) % 12])
        .collect();
    let ring = Csr::from_arrays(12, &focal, &neighbor, &[1.0; 24]).unwrap();
    for star in [Star::Off, Star::On] {
        let r = g_local(&y, &ring, &opts(star, GTransform::R)).unwrap();
        let c = g_local_permutations(&y, &r, &settings(20_000, true)).unwrap();
        let sum: f64 = y.iter().sum();
        for (i, s) in c.summary.iter().enumerate() {
            let w = &r.weights;
            let (mut own, mut others) = (0.0, 0.0);
            for k in w.offsets[i]..w.offsets[i + 1] {
                if w.neighbors[k] as usize == i {
                    own = w.weights[k];
                } else {
                    others += w.weights[k];
                }
            }
            let expected = if r.star {
                (others * (sum - y[i]) / 11.0 + own * y[i]) / sum
            } else {
                others / 11.0
            };
            assert!(
                (s.ei_sim - expected).abs() < 0.004,
                "{i}: {} vs {expected}",
                s.ei_sim
            );
        }
    }
}

#[test]
fn island_weight_changes_island_replicates_only() {
    let r = g_local(&Y6, &path_island(), &opts(Star::Off, GTransform::R)).unwrap();
    let a = g_local_permutations(&Y6, &r, &settings(99, true)).unwrap();
    let mut s = settings(99, true);
    s.island_weight = 1.0;
    let b = g_local_permutations(&Y6, &r, &s).unwrap();
    assert_eq!(
        a.sims.as_ref().unwrap()[..5 * 99],
        b.sims.as_ref().unwrap()[..5 * 99]
    );
    assert!(b.sims.as_ref().unwrap()[5 * 99..].iter().any(|v| *v != 0.0));
}

#[test]
fn fdr_matches_esda() {
    assert_eq!(
        fdr(&[0.001, 0.01, 0.02, 0.04, 0.3, 0.5], 0.05).unwrap(),
        0.025000000000000005
    );
    assert_eq!(fdr(&[0.5, 0.6], 0.05).unwrap(), 0.025);
    // All significant: the first (largest) p-value already satisfies p < index*alpha/n.
    let all = [0.001, 0.002, 0.003, 0.004];
    assert_eq!(fdr(&all, 0.05).unwrap(), 4.0 * 0.05 / 4.0);
    // None significant: Bonferroni.
    assert_eq!(fdr(&[0.5, 0.7, 0.9, 0.99], 0.05).unwrap(), 0.05 / 4.0);
    // Ties and order independence.
    assert_eq!(
        fdr(&[0.01, 0.01, 0.9, 0.9], 0.1).unwrap(),
        fdr(&[0.9, 0.01, 0.9, 0.01], 0.1).unwrap()
    );
}

#[test]
fn fdr_rejects_bad_input() {
    assert!(fdr(&[], 0.05).is_err());
    assert!(fdr(&[0.1, f64::NAN], 0.05).is_err());
    assert!(fdr(&[0.1, 0.2], 0.0).is_err());
    assert!(fdr(&[0.1, 0.2], 1.5).is_err());
    assert!(fdr(&[0.1, -0.2], 0.05).is_err());
}
