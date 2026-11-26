//! Unit tests for GWR prediction and local collinearity. The mgwr oracle checks live in the
//! Elixir suites (`gwr_predict_oracle_test.exs`, `gwr_collinearity_oracle_test.exs`).

use super::tests::sample;
use super::tests_glm::data;
use super::*;

fn close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "entry {i}: {x} vs {y}"
        );
    }
}

fn model<'a>(d: &'a (Vec<f64>, Vec<f64>, Vec<f64>), kernel: Kernel, fixed: bool) -> Model<'a> {
    Model::new(&d.0, &d.1, &d.2, 2, kernel, fixed, false).unwrap()
}

#[test]
fn predicting_at_the_calibration_points_reproduces_the_gaussian_fit() {
    let d = sample();
    for (kernel, fixed, bw) in [
        (Kernel::Bisquare, false, 20.0),
        (Kernel::Gaussian, true, 1.8),
        (Kernel::Exponential, false, 15.0),
    ] {
        let m = model(&d, kernel, fixed);
        let fit = fit_full(&m, bw, true).unwrap();
        let p = predict(&m, bw, true, &d.0, &d.2, None).unwrap();
        assert_eq!((p.m, p.k), (64, 2));
        close(&p.params, &fit.params, 1e-10);
        close(&p.bse, &fit.bse, 1e-10);
        close(&p.tvalues, &fit.tvalues, 1e-9);
        close(&p.predictions, &fit.predy, 1e-10);
        close(&p.local_r2, &fit.local_r2, 1e-10);
        assert!((p.scale - fit.scale).abs() < 1e-14);
    }
}

#[test]
fn the_global_bandwidth_gives_the_ols_surface_everywhere() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let fit = fit_full(&m, f64::INFINITY, true).unwrap();
    // Two points far outside the sample.
    let points = [100.0, -50.0, -7.0, 3.0];
    let p = [1.0, 2.0, 1.0, -1.0];
    let pred = predict(&m, f64::INFINITY, true, &points, &p, None).unwrap();
    close(&pred.params[..2], &fit.params[..2], 1e-10);
    close(&pred.params[2..], &fit.params[..2], 1e-10);
    let b = &fit.params[..2];
    close(&pred.predictions, &[b[0] + 2.0 * b[1], b[0] - b[1]], 1e-10);
}

#[test]
fn supplied_exog_scale_and_residuals_are_used() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let base = predict(&m, 20.0, true, &d.0, &d.2, None).unwrap();
    let zeros = vec![0.0; 64];
    let exog = Exog {
        scale: 4.0 * base.scale,
        resid: &zeros,
    };
    let p = predict(&m, 20.0, true, &d.0, &d.2, Some(exog)).unwrap();
    close(&p.params, &base.params, 1e-14);
    for (a, b) in p.bse.iter().zip(&base.bse) {
        assert!((a - 2.0 * b).abs() < 1e-10 * (1.0 + b.abs()));
    }
    // No residual variation: the local R2 is one.
    close(&p.local_r2, &vec![1.0; 64], 1e-12);
    assert_eq!(p.scale, 4.0 * base.scale);
}

#[test]
fn glm_predictions_are_the_linear_predictor_and_match_the_fit_at_the_data() {
    let d = data();
    for (family, y, off, bw) in [
        (Family::Poisson, &d.counts, &d.offset, 24.0),
        (Family::Binomial, &d.binary, &Vec::new(), 30.0),
    ] {
        let m = Model::new(&d.coords, y, &d.x, 2, Kernel::Bisquare, false, false)
            .unwrap()
            .with_family(family, off, 1.0e-5, 20)
            .unwrap();
        let fit = fit_full(&m, bw, true).unwrap();
        let p = predict(&m, bw, true, &d.coords, &d.x, None).unwrap();
        close(&p.params, &fit.params, 1e-10);
        close(&p.bse, &fit.bse, 1e-10);
        let eta: Vec<f64> = (0..64)
            .map(|i| d.x[2 * i] * p.params[2 * i] + d.x[2 * i + 1] * p.params[2 * i + 1])
            .collect();
        close(&p.predictions, &eta, 1e-12);
        assert!(p.local_r2.is_empty());
        assert_eq!(p.scale, 1.0);
    }
}

#[test]
fn more_prediction_points_than_observations_are_fine() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let points: Vec<f64> = (0..70).flat_map(|i| [f64::from(i) * 0.1, 3.0]).collect();
    let p: Vec<f64> = (0..70).flat_map(|i| [1.0, f64::from(i) * 0.01]).collect();
    let pred = predict(&m, 20.0, true, &points, &p, None).unwrap();
    assert_eq!(pred.predictions.len(), 70);
    assert!(pred.predictions.iter().all(|v| v.is_finite()));
}

#[test]
fn prediction_validates_shapes_and_bandwidth() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let err = |r: Result<Prediction, String>| r.unwrap_err();
    assert!(err(predict(&m, 20.0, true, &[1.0, 2.0, 3.0], &[1.0, 1.0], None)).contains("points"));
    assert!(err(predict(&m, 20.0, true, &[1.0, 2.0], &[1.0], None)).contains("rows"));
    let short = Exog {
        scale: 1.0,
        resid: &[0.0; 3],
    };
    assert!(err(predict(
        &m,
        20.0,
        true,
        &[1.0, 2.0],
        &[1.0, 1.0],
        Some(short)
    ))
    .contains("resid"));
    assert!(err(predict(&m, 1.0, true, &[1.0, 2.0], &[1.0, 1.0], None)).contains("bandwidth"));
}

#[test]
fn a_prediction_point_without_neighbors_reports_the_point() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, true);
    let exog = Exog {
        scale: 1.0,
        resid: &[0.0; 64],
    };
    let err = predict(&m, 0.5, true, &[500.0, 500.0], &[1.0, 1.0], Some(exog)).unwrap_err();
    assert!(err.contains("prediction point 0"), "{err}");
}

#[test]
fn prediction_does_not_depend_on_the_thread_count() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| predict(&m, 20.0, true, &d.0, &d.2, None).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.params, b.params);
    assert_eq!(a.bse, b.bse);
    assert_eq!(a.local_r2, b.local_r2);
}

// -- local collinearity -------------------------------------------------------------------

fn two_columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 6;
    let coords: Vec<f64> = (0..n).flat_map(|i| [f64::from(i), 0.0]).collect();
    let x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x2 = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
    let x: Vec<f64> = (0..6).flat_map(|i| [x1[i], x2[i]]).collect();
    (coords, vec![0.0; n as usize], x)
}

#[test]
fn global_collinearity_of_two_columns_has_a_closed_form() {
    let d = two_columns();
    let m = model(&d, Kernel::Bisquare, false);
    let c = local_collinearity(&m, f64::INFINITY, false).unwrap();
    assert_eq!((c.n, c.k, c.ncor, c.nvif), (6, 2, 1, 2));
    let (x1, x2): (Vec<f64>, Vec<f64>) = (0..6).map(|i| (d.2[2 * i], d.2[2 * i + 1])).unzip();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    // Uncentered cosine between the (weighted, column-normalized) columns drives the SVD.
    let rho = dot(&x1, &x2) / (dot(&x1, &x1) * dot(&x2, &x2)).sqrt();
    let cn = ((1.0 + rho) / (1.0 - rho)).sqrt();
    // Pearson correlation drives the correlation and the VIF.
    let mean = |v: &[f64]| v.iter().sum::<f64>() / 6.0;
    let (m1, m2) = (mean(&x1), mean(&x2));
    let cov: f64 = x1.iter().zip(&x2).map(|(a, b)| (a - m1) * (b - m2)).sum();
    let v1: f64 = x1.iter().map(|a| (a - m1) * (a - m1)).sum();
    let v2: f64 = x2.iter().map(|a| (a - m2) * (a - m2)).sum();
    let r = cov / (v1 * v2).sqrt();
    for i in 0..6 {
        assert!((c.condition_number[i] - cn).abs() < 1e-10 * cn);
        assert!((c.corr[i] - r).abs() < 1e-12);
        assert!((c.vif[2 * i] - 1.0 / (1.0 - r * r)).abs() < 1e-10);
        assert!((c.vif[2 * i + 1] - 1.0 / (1.0 - r * r)).abs() < 1e-10);
        assert!((c.vdp[2 * i] - (1.0 + rho) / 2.0).abs() < 1e-10);
        assert!((c.vdp[2 * i + 1] - (1.0 + rho) / 2.0).abs() < 1e-10);
    }
}

#[test]
fn with_a_constant_the_correlations_skip_the_intercept() {
    let d = sample();
    // Add a third column so one explanatory pair exists next to the intercept.
    let x3: Vec<f64> = (0..64)
        .flat_map(|i| {
            [
                d.2[2 * i],
                d.2[2 * i + 1],
                d.2[2 * i + 1] * 0.5 + (i % 5) as f64,
            ]
        })
        .collect();
    let m = Model::new(&d.0, &d.1, &x3, 3, Kernel::Bisquare, false, false).unwrap();
    let c = local_collinearity(&m, 25.0, true).unwrap();
    assert_eq!((c.ncor, c.nvif), (1, 2));
    assert_eq!(c.corr.len(), 64);
    assert_eq!(c.vif.len(), 128);
    assert_eq!(c.condition_number.len(), 64);
    assert_eq!(c.vdp.len(), 64 * 3);
    // Each variance-decomposition proportion is a share of one variable's variance.
    for i in 0..64 {
        assert!(c.vdp[3 * i..3 * i + 3]
            .iter()
            .all(|p| (0.0..=1.0).contains(p)));
        assert!(c.condition_number[i] >= 1.0);
        assert!(c.vif[2 * i] >= 1.0);
    }
}

#[test]
fn collinearity_needs_two_explanatory_variables() {
    let d = sample();
    let m = model(&d, Kernel::Bisquare, false);
    let err = local_collinearity(&m, 20.0, true).unwrap_err();
    assert!(err.contains("two explanatory"), "{err}");
    let one = Model::new(&d.0, &d.1, &d.2[..64], 1, Kernel::Bisquare, false, false).unwrap();
    assert!(local_collinearity(&one, 20.0, false)
        .unwrap_err()
        .contains("two explanatory"));
}

#[test]
fn mgwr_collinearity_with_equal_bandwidths_matches_the_single_bandwidth_one() {
    let d = sample();
    let x3: Vec<f64> = (0..64)
        .flat_map(|i| [d.2[2 * i], d.2[2 * i + 1], (i % 7) as f64 - d.2[2 * i + 1]])
        .collect();
    let m = Model::new(&d.0, &d.1, &x3, 3, Kernel::Bisquare, false, false).unwrap();
    let g = local_collinearity(&m, 25.0, true).unwrap();
    let mg = local_collinearity_mgwr(&m, &[25.0, 25.0, 25.0]).unwrap();
    close(&mg.condition_number, &g.condition_number, 1e-10);
    close(&mg.vdp, &g.vdp, 1e-10);
    assert!(mg.corr.is_empty() && mg.vif.is_empty());
    // Distinct bandwidths change the answer.
    let other = local_collinearity_mgwr(&m, &[25.0, 15.0, 40.0]).unwrap();
    assert!(other
        .condition_number
        .iter()
        .zip(&g.condition_number)
        .any(|(a, b)| (a - b).abs() > 1e-6));
    assert!(local_collinearity_mgwr(&m, &[25.0, 25.0])
        .unwrap_err()
        .contains("bandwidth"));
}

#[test]
fn collinearity_does_not_depend_on_the_thread_count() {
    let d = two_columns();
    let m = model(&d, Kernel::Gaussian, true);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| local_collinearity(&m, 2.0, false).unwrap())
    };
    let (a, b) = (run(1), run(3));
    assert_eq!(a.condition_number, b.condition_number);
    assert_eq!(a.vdp, b.vdp);
}
