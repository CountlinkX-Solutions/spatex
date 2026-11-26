//! Unit tests for the opt-in hat matrix and the scipy bandwidth search. The mgwr oracle checks
//! live in the Elixir suites (`gwr_hat_oracle_test.exs`, `gwr_search_oracle_test.exs`).

use super::tests::sample;
use super::tests_glm::data;
use super::*;

fn model<'a>(d: &'a (Vec<f64>, Vec<f64>, Vec<f64>), kernel: Kernel, fixed: bool) -> Model<'a> {
    Model::new(&d.0, &d.1, &d.2, 2, kernel, fixed, false).unwrap()
}

fn options(search: Search) -> SelectOptions {
    SelectOptions {
        criterion: Criterion::Aicc,
        search,
        bw_min: None,
        bw_max: None,
        n_loc: 1,
        constant: true,
        tol: 1e-6,
        max_iter: 200,
    }
}

#[test]
fn the_hat_matrix_is_kept_only_when_requested() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    assert!(fit_full(&m, 20.0, true).unwrap().hat.is_empty());
    assert!(fit_full_with(&m, 20.0, true, false).unwrap().hat.is_empty());
    assert_eq!(
        fit_full_with(&m, 20.0, true, true).unwrap().hat.len(),
        64 * 64
    );
}

#[test]
fn the_gaussian_hat_matrix_reproduces_the_fit() {
    let d = sample();
    for (kernel, fixed, bw) in [
        (Kernel::Bisquare, false, 20.0),
        (Kernel::Gaussian, true, 1.8),
        (Kernel::Exponential, false, 15.0),
    ] {
        let m = model(&d, kernel, fixed);
        let plain = fit_full(&m, bw, true).unwrap();
        let f = fit_full_with(&m, bw, true, true).unwrap();
        let n = 64;
        // Everything else is unchanged by keeping S.
        assert_eq!(f.params, plain.params);
        assert_eq!(f.tr_sts, plain.tr_sts);
        let mut sum_sq = 0.0;
        for i in 0..n {
            let row = &f.hat[i * n..(i + 1) * n];
            // S y = fitted values; diag(S) = influence.
            let fitted: f64 = row.iter().zip(&d.1).map(|(s, y)| s * y).sum();
            assert!((fitted - f.predy[i]).abs() < 1e-10, "row {i}");
            assert!((row[i] - f.influ[i]).abs() < 1e-12, "diag {i}");
            sum_sq += row.iter().map(|s| s * s).sum::<f64>();
        }
        assert!((sum_sq - f.tr_sts).abs() < 1e-10);
        let trace: f64 = (0..n).map(|i| f.hat[i * n + i]).sum();
        assert!((trace - f.tr_s).abs() < 1e-12);
    }
}

#[test]
fn the_glm_hat_matrix_has_the_fit_trace_and_only_finite_entries() {
    let d = data();
    let m = Model::new(
        &d.coords,
        &d.counts,
        &d.x,
        2,
        Kernel::Bisquare,
        false,
        false,
    )
    .unwrap()
    .with_family(Family::Poisson, &d.offset, 1.0e-5, 20)
    .unwrap();
    let f = fit_full_with(&m, 25.0, true, true).unwrap();
    assert_eq!(f.hat.len(), 64 * 64);
    assert!(f.hat.iter().all(|v| v.is_finite()));
    let trace: f64 = (0..64).map(|i| f.hat[i * 64 + i]).sum();
    assert!(trace > 0.0 && trace.is_finite());
}

#[test]
fn the_scipy_search_minimizes_the_criterion_inside_the_bounds() {
    let d = sample();
    for (kernel, fixed) in [(Kernel::Bisquare, false), (Kernel::Gaussian, true)] {
        let m = model(&d, kernel, fixed);
        let s = select_bandwidth(&m, &options(Search::Scipy)).unwrap();
        // mgwr keeps the optimizer's float and records no history.
        assert!(s.history.is_empty());
        let at = |bw: f64| fit_lite(&m, bw).unwrap().criteria.aicc;
        assert_eq!(s.score, at(s.bw));
        let opt = s.optimizer.expect("scipy exposes its optimizer result");
        assert!(opt.nfev > 3 && opt.status == 0);
        if !fixed {
            assert!(s.bw >= 44.0 && s.bw <= 64.0);
            assert!(
                s.bw.fract() != 0.0,
                "the scipy search returns a float count"
            );
        }
    }
}

#[test]
fn the_scipy_search_honours_explicit_bounds_and_rejects_equal_ones() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let bounded = SelectOptions {
        bw_min: Some(20.0),
        bw_max: Some(40.0),
        ..options(Search::Scipy)
    };
    let s = select_bandwidth(&m, &bounded).unwrap();
    assert!(s.bw >= 20.0 && s.bw <= 40.0);
    let same = SelectOptions {
        bw_min: Some(30.0),
        bw_max: Some(30.0),
        ..options(Search::Scipy)
    };
    let err = select_bandwidth(&m, &same).unwrap_err();
    assert!(err.contains("distinct"), "{err}");
    // `bw_max = +inf` (mgwr's include-global switch) is not meaningful for this search.
    let inf = SelectOptions {
        bw_max: Some(f64::INFINITY),
        ..options(Search::Scipy)
    };
    assert!(select_bandwidth(&m, &inf).unwrap_err().contains("scipy"));
}

#[test]
fn more_columns_raise_the_default_adaptive_lower_bound() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let few = select_bandwidth(&m, &options(Search::Scipy)).unwrap();
    let many = SelectOptions {
        n_loc: 6,
        ..options(Search::Scipy)
    };
    let s = select_bandwidth(&m, &many).unwrap();
    // 40 + 2 * (6 + 1) = 54 neighbors minimum: a different, narrower search.
    assert!(s.bw >= 54.0);
    assert_ne!(s.bw, few.bw);
}
