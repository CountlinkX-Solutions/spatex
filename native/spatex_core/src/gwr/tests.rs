use super::kernel::{check_bandwidth, weights, Workspace};
use super::*;
use crate::regression::linalg;

/// Jittered 8x8 lattice with a smooth spatially varying slope.
pub(super) fn sample() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut coords = Vec::new();
    let mut y = Vec::new();
    let mut x = Vec::new();
    let mut state = 12345_u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    for r in 0..8 {
        for c in 0..8 {
            let (px, py) = (c as f64 + 0.3 * rnd(), r as f64 + 0.3 * rnd());
            let xv = rnd() * 4.0 - 2.0;
            coords.extend([px, py]);
            x.extend([1.0, xv]);
            y.push(1.0 + 0.2 * px + (0.5 + 0.15 * py) * xv + 0.05 * (rnd() - 0.5));
        }
    }
    (coords, y, x)
}

fn model<'a>(d: &'a (Vec<f64>, Vec<f64>, Vec<f64>), kernel: Kernel, fixed: bool) -> Model<'a> {
    Model::new(&d.0, &d.1, &d.2, 2, kernel, fixed, false).unwrap()
}

#[test]
fn adaptive_weights_keep_the_tied_neighbors_at_a_tiny_weight() {
    // 6x6 unit lattice, focal (2, 2), bw = 5: itself plus 4 neighbors at distance 1.
    let mut coords = Vec::new();
    for r in 0..6 {
        for c in 0..6 {
            coords.extend([c as f64, r as f64]);
        }
    }
    let y = vec![0.0; 36];
    let x: Vec<f64> = (0..36).flat_map(|i| [1.0, f64::from(i)]).collect();
    let m = Model::new(&coords, &y, &x, 2, Kernel::Bisquare, false, false).unwrap();
    let mut ws = Workspace::new(36);
    weights(&m, 14, 5.0, &mut ws).unwrap();
    let nonzero: Vec<f64> = ws.w.iter().copied().filter(|w| *w > 0.0).collect();
    assert_eq!(nonzero.len(), 5);
    assert_eq!(ws.w[14], 1.0);
    for w in nonzero.iter().filter(|w| **w < 1.0) {
        assert!((w - 3.999_998_805_630_16e-14).abs() < 1e-24, "{w}");
    }
}

#[test]
fn gaussian_and_exponential_are_not_truncated() {
    let d = sample();
    for kernel in [Kernel::Gaussian, Kernel::Exponential] {
        let m = model(&d, kernel, true);
        let mut ws = Workspace::new(64);
        weights(&m, 0, 1.5, &mut ws).unwrap();
        assert!(ws.w.iter().all(|w| *w > 0.0));
        assert_eq!(ws.w[0], 1.0);
    }
}

#[test]
fn fractional_adaptive_bandwidth_truncates() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let (mut a, mut b) = (Workspace::new(64), Workspace::new(64));
    weights(&m, 3, 12.0, &mut a).unwrap();
    weights(&m, 3, 12.9, &mut b).unwrap();
    assert_eq!(a.w, b.w);
}

#[test]
fn bandwidth_validation() {
    let d = sample();
    let adaptive = model(&d, Kernel::Bisquare, false);
    assert!(check_bandwidth(&adaptive, 1.0).is_err());
    assert!(check_bandwidth(&adaptive, 65.0).is_err());
    assert!(check_bandwidth(&adaptive, f64::NAN).is_err());
    assert!(check_bandwidth(&adaptive, 2.0).is_ok());
    assert!(check_bandwidth(&adaptive, f64::INFINITY).is_ok());
    let fixed = model(&d, Kernel::Gaussian, true);
    assert!(check_bandwidth(&fixed, 0.0).is_err());
    assert!(check_bandwidth(&fixed, -1.0).is_err());
    assert!(check_bandwidth(&fixed, 0.5).is_ok());
}

#[test]
fn infinite_bandwidth_reproduces_ols() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let full = fit_full(&m, f64::INFINITY, true).unwrap();
    let cols = vec![vec![1.0; 64], d.2.chunks(2).map(|r| r[1]).collect()];
    let ols = linalg::fit(&d.1, &cols).unwrap();
    for i in 0..64 {
        assert!((full.params[2 * i] - ols.betas[0]).abs() < 1e-12);
        assert!((full.params[2 * i + 1] - ols.betas[1]).abs() < 1e-12);
        assert!((full.predy[i] - ols.predy[i]).abs() < 1e-12);
    }
    // Global model: S is the OLS hat matrix, so tr(S) = tr(S'S) = k.
    assert!((full.tr_s - 2.0).abs() < 1e-10);
    assert!((full.tr_sts - 2.0).abs() < 1e-10);
    // The coefficient variance is the OLS one: sigma2 (X'X)^-1 diagonal.
    let sigma2 = ols.utu / (64.0 - 2.0);
    for i in 0..64 {
        for l in 0..2 {
            let expected = (sigma2 * ols.xtxi[(l, l)]).sqrt();
            assert!((full.bse[2 * i + l] - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn fit_invariants() {
    let d = sample();
    for kernel in [Kernel::Bisquare, Kernel::Gaussian, Kernel::Exponential] {
        let m = model(&d, kernel, false);
        let f = fit_full(&m, 20.0, true).unwrap();
        assert!(f.influ.iter().all(|h| (0.0..=1.0).contains(h)));
        assert!(f.tr_s >= 2.0 - 1e-9 && f.tr_s <= 64.0);
        assert!(f.tr_sts <= f.tr_s + 1e-9);
        assert!(f.local_r2.iter().all(|r| *r <= 1.0 + 1e-12));
        assert_eq!(f.enp, f.tr_s);
        let lite = fit_lite(&m, 20.0).unwrap();
        assert_eq!(lite.params, f.params);
        assert_eq!(lite.criteria, f.criteria);
        let v2 = fit_full(&m, 20.0, false).unwrap();
        assert_eq!(v2.enp, 2.0 * f.tr_s - f.tr_sts);
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let d = sample();
    let m = model(&d, Kernel::Gaussian, false);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| fit_full(&m, 25.0, true).unwrap())
    };
    let (one, many) = (run(1), run(4));
    assert_eq!(one.params, many.params);
    assert_eq!(one.tr_sts.to_bits(), many.tr_sts.to_bits());
    assert_eq!(one.bse, many.bse);
}

#[test]
fn singular_design_is_reported() {
    let d = sample();
    let x: Vec<f64> = d.2.chunks(2).flat_map(|r| [r[0], r[0]]).collect();
    let m = Model::new(&d.0, &d.1, &x, 2, Kernel::Bisquare, false, false).unwrap();
    let err = fit_lite(&m, 30.0).unwrap_err();
    assert!(err.contains("singular"), "{err}");
}

#[test]
fn model_rejects_bad_shapes() {
    let d = sample();
    assert!(Model::new(&d.0[..10], &d.1, &d.2, 2, Kernel::Bisquare, false, false).is_err());
    assert!(Model::new(&d.0, &d.1, &d.2[..10], 2, Kernel::Bisquare, false, false).is_err());
    assert!(Kernel::parse("triangular").is_err());
}

#[test]
fn t_critical_matches_scipy() {
    // scipy.stats.t.ppf(0.975, 158) and the georgia corrected-alpha value.
    assert!((t_critical(0.05, 158.0) - 1.975_092_072_712_084_4).abs() < 1e-9);
    assert!((t_critical(0.013_923_55, 158.0) - 2.486_946_447_635_862_7).abs() < 1e-9);
    assert!((t_critical(1.0, 10.0)).abs() < 1e-12);
    assert!((t_critical(1.5, 10.0) + t_critical(0.5, 10.0)).abs() < 1e-9);
    assert_eq!(t_critical(0.0, 10.0), f64::INFINITY);
}

fn parabola(center: f64) -> impl FnMut(f64) -> Result<f64, String> {
    move |bw| Ok((bw - center) * (bw - center))
}

#[test]
fn golden_section_finds_the_integer_minimum() {
    let s = golden_section(50.0, 200.0, parabola(97.0), 1e-6, 200, true, false).unwrap();
    assert_eq!(s.bw, 97.0);
    assert_eq!(s.score, 0.0);
    assert!(s.history.iter().all(|(b, _)| b.fract() == 0.0));
}

#[test]
fn golden_section_rounds_a_fixed_bandwidth_to_two_decimals() {
    let s = golden_section(0.1, 10.0, parabola(3.217), 1e-12, 200, false, false).unwrap();
    assert!((s.bw - 3.217).abs() <= 0.011, "{}", s.bw);
    assert_eq!(s.bw, (s.bw * 100.0).round() / 100.0);
}

#[test]
fn golden_section_memoizes_evaluations() {
    let mut calls = Vec::new();
    let s = golden_section(
        50.0,
        200.0,
        |bw| {
            calls.push(bw.to_bits());
            Ok((bw - 120.0).abs())
        },
        1e-6,
        200,
        true,
        false,
    )
    .unwrap();
    let mut unique = calls.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(calls.len(), unique.len());
    assert_eq!(s.bw, 120.0);
}

#[test]
fn golden_section_global_check_wins_ties_and_loses_otherwise() {
    let flat = golden_section(50.0, 100.0, |_| Ok(1.0), 1e-6, 200, true, true).unwrap();
    assert_eq!(flat.bw, f64::INFINITY);
    assert_eq!(flat.history.last().unwrap().0, f64::INFINITY);
    let bowl = golden_section(
        50.0,
        100.0,
        |bw| {
            Ok(if bw.is_infinite() {
                9.0
            } else {
                (bw - 70.0).abs()
            })
        },
        1e-6,
        200,
        true,
        true,
    )
    .unwrap();
    assert!((bowl.bw - 70.0).abs() <= 1.0, "{}", bowl.bw);
}

#[test]
fn golden_section_propagates_errors() {
    let err =
        golden_section(1.0, 2.0, |_| Err("boom".to_string()), 1e-6, 5, false, false).unwrap_err();
    assert_eq!(err, "boom");
}

#[test]
fn equal_interval_keeps_the_first_of_equal_scores() {
    let s = equal_interval(1.0, 9.0, 2.0, |bw| Ok((bw - 4.0).abs()), false).unwrap();
    // 3 and 5 tie at distance 1: the earlier one is kept (strict improvement).
    assert_eq!((s.bw, s.score), (3.0, 1.0));
    let bws: Vec<f64> = s.history.iter().map(|h| h.0).collect();
    assert_eq!(bws, vec![1.0, 3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn select_bandwidth_adaptive_rejects_small_samples_and_bad_bounds() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let opts = SelectOptions {
        criterion: Criterion::Aicc,
        search: Search::Golden,
        bw_min: None,
        bw_max: None,
        n_loc: 1,
        constant: true,
        tol: 1e-6,
        max_iter: 200,
    };
    // 40 observations: the default lower bound 40 + 2k = 44 exceeds n.
    let small = Model::new(
        &d.0[..80],
        &d.1[..40],
        &d.2[..80],
        2,
        Kernel::Bisquare,
        false,
        false,
    )
    .unwrap();
    let err = select_bandwidth(&small, &opts).unwrap_err();
    assert!(err.contains("n >= 40 + 2k"), "{err}");
    let ok = SelectOptions {
        bw_min: Some(12.0),
        bw_max: Some(60.0),
        ..opts
    };
    let s = select_bandwidth(&m, &ok).unwrap();
    assert!(s.bw >= 12.0 && s.bw <= 60.0 && s.bw.fract() == 0.0);
    let bad = SelectOptions {
        bw_min: Some(12.0),
        bw_max: Some(70.0),
        ..opts
    };
    assert!(select_bandwidth(&m, &bad).is_err());
}

#[test]
fn select_bandwidth_fixed_matches_a_manual_scan() {
    let d = sample();
    let m = model(&d, Kernel::Gaussian, true);
    let opts = SelectOptions {
        criterion: Criterion::Aicc,
        search: Search::Golden,
        bw_min: None,
        bw_max: None,
        n_loc: 1,
        constant: true,
        tol: 1e-6,
        max_iter: 200,
    };
    let s = select_bandwidth(&m, &opts).unwrap();
    let at = |bw: f64| fit_lite(&m, bw).unwrap().criteria.aicc;
    assert!(at(s.bw) <= at(s.bw * 1.5) && at(s.bw) <= at(s.bw / 1.5));
    for crit in [Criterion::Aic, Criterion::Bic, Criterion::Cv] {
        let o = SelectOptions {
            criterion: crit,
            ..opts
        };
        assert!(select_bandwidth(&m, &o).unwrap().bw > 0.0);
    }
}

#[test]
fn interval_search_validates_its_arguments() {
    let d = sample();
    let m = model(&d, Kernel::Gaussian, true);
    let base = SelectOptions {
        criterion: Criterion::Aicc,
        search: Search::Interval(1.0),
        bw_min: Some(1.0),
        bw_max: Some(4.0),
        n_loc: 1,
        constant: true,
        tol: 1e-6,
        max_iter: 200,
    };
    let s = select_bandwidth(&m, &base).unwrap();
    assert_eq!(s.history.len(), 4);
    let zero = SelectOptions {
        search: Search::Interval(0.0),
        ..base
    };
    assert!(select_bandwidth(&m, &zero).is_err());
    let open = SelectOptions {
        bw_max: None,
        ..base
    };
    assert!(select_bandwidth(&m, &open).is_err());
}
