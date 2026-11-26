//! Unit tests for the Poisson/Binomial families and MGWR.

use super::family::{binomial_resid_dev_sq, poisson_resid_dev_sq, Family};
use super::mgwr::{fit_mgwr, MgwrOptions};
use super::*;

pub(super) struct Data {
    pub(super) coords: Vec<f64>,
    pub(super) x: Vec<f64>,
    pub(super) counts: Vec<f64>,
    pub(super) binary: Vec<f64>,
    pub(super) offset: Vec<f64>,
}

/// Jittered 8x8 lattice; counts and 0/1 outcomes with spatially varying effects.
pub(super) fn data() -> Data {
    let mut state = 987_654_321_u64;
    let mut rnd = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1_u64 << 53) as f64)
    };
    let (mut coords, mut x, mut counts, mut binary, mut offset) =
        (vec![], vec![], vec![], vec![], vec![]);
    for r in 0..8 {
        for c in 0..8 {
            let (px, py) = (c as f64 + 0.3 * rnd(), r as f64 + 0.3 * rnd());
            let xv = rnd() * 2.0 - 1.0;
            let off = 0.5 + rnd();
            let eta = 0.8 + 0.1 * px + (0.4 + 0.05 * py) * xv;
            // Deterministic pseudo-counts: floor of the mean plus a coin flip.
            let mean = off * eta.exp();
            counts.push((mean + rnd() * 2.0 - 1.0).round().max(1.0));
            let p = 1.0 / (1.0 + (-(0.3 * xv + 0.05 * px - 0.2)).exp());
            binary.push(if rnd() < p { 1.0 } else { 0.0 });
            coords.extend([px, py]);
            x.extend([1.0, xv]);
            offset.push(off);
        }
    }
    Data {
        coords,
        x,
        counts,
        binary,
        offset,
    }
}

fn glm<'a>(d: &'a Data, y: &'a [f64], family: Family, off: &'a [f64]) -> Model<'a> {
    Model::new(&d.coords, y, &d.x, 2, Kernel::Bisquare, false, false)
        .unwrap()
        .with_family(family, off, 1.0e-5, 20)
        .unwrap()
}

#[test]
fn family_parsing() {
    assert_eq!(Family::parse("gaussian").unwrap(), Family::Gaussian);
    assert_eq!(Family::parse("poisson").unwrap(), Family::Poisson);
    assert_eq!(Family::parse("binomial").unwrap(), Family::Binomial);
    assert!(Family::parse("gamma").is_err());
}

#[test]
fn deviance_residual_formulas() {
    // y = 3, mu = 2: 2 (3 ln 1.5 - 1).
    let expected = 2.0 * (3.0 * 1.5_f64.ln() - 1.0);
    assert!((poisson_resid_dev_sq(3.0, 2.0) - expected).abs() < 1e-14);
    assert_eq!(poisson_resid_dev_sq(2.0, 2.0), 0.0);
    // A zero count has deviance 2 mu (0 ln 0 = 0 after the clip).
    assert!((poisson_resid_dev_sq(0.0, 1.5) - 3.0).abs() < 1e-12);
    // Binomial: -2 ln(mu) for y = 1 and -2 ln(1 - mu) for y = 0.
    assert!((binomial_resid_dev_sq(1.0, 0.8) + 2.0 * 0.8_f64.ln()).abs() < 1e-14);
    assert!((binomial_resid_dev_sq(0.0, 0.8) + 2.0 * 0.2_f64.ln()).abs() < 1e-14);
}

#[test]
fn model_validates_family_inputs() {
    let d = data();
    let ok = Model::new(
        &d.coords,
        &d.counts,
        &d.x,
        2,
        Kernel::Bisquare,
        false,
        false,
    )
    .unwrap();
    let mut bad = d.counts.clone();
    bad[3] = -1.0;
    let m = Model::new(&d.coords, &bad, &d.x, 2, Kernel::Bisquare, false, false).unwrap();
    assert!(m
        .with_family(Family::Poisson, &d.offset, 1e-5, 20)
        .unwrap_err()
        .contains("non-negative"));
    let zero_off = vec![0.0; 64];
    assert!(ok
        .with_family(Family::Poisson, &zero_off, 1e-5, 20)
        .unwrap_err()
        .contains("offset"));
    assert!(ok
        .with_family(Family::Poisson, &d.offset[..5], 1e-5, 20)
        .unwrap_err()
        .contains("offset"));
    assert!(ok
        .with_family(Family::Binomial, &d.offset, 1e-5, 20)
        .unwrap_err()
        .contains("0 or 1"));
    assert!(ok
        .with_family(Family::Poisson, &d.offset, -1.0, 20)
        .is_err());
    assert!(ok.with_family(Family::Poisson, &d.offset, 1e-5, 0).is_err());
}

#[test]
fn global_glm_solves_the_score_equations() {
    let d = data();
    for (family, y) in [
        (Family::Poisson, d.counts.clone()),
        (Family::Binomial, d.binary.clone()),
    ] {
        let m = Model::new(&d.coords, &y, &d.x, 2, Kernel::Bisquare, false, false)
            .unwrap()
            .with_family(family, &d.offset, 1e-12, 100)
            .unwrap();
        let f = fit_full(&m, f64::INFINITY, true).unwrap();
        // Every site fits the same global model.
        for i in 1..64 {
            assert!((f.params[2 * i] - f.params[0]).abs() < 1e-9);
            assert!((f.params[2 * i + 1] - f.params[1]).abs() < 1e-9);
        }
        // Canonical links: sum_j x_j (y_j - mu_j) = 0 at the maximum likelihood estimate.
        let mut score = [0.0, 0.0];
        for (j, yj) in y.iter().enumerate() {
            let r = yj - f.predy[j];
            score[0] += r;
            score[1] += d.x[2 * j + 1] * r;
        }
        assert!(score[0].abs() < 1e-7 && score[1].abs() < 1e-7, "{score:?}");
        assert!((f.tr_s - 2.0).abs() < 1e-6, "{}", f.tr_s);
        assert_eq!(f.scale, 1.0);
        assert!((f.d2 - (1.0 - f.global_deviance / f.null_deviance)).abs() < 1e-14);
        assert!(f.iterations.iter().all(|&i| i >= 1));
    }
}

#[test]
fn glm_criteria_follow_the_deviance() {
    let d = data();
    let m = glm(&d, &d.counts, Family::Poisson, &d.offset);
    let f = fit_full(&m, 30.0, true).unwrap();
    let dev = f.global_deviance;
    assert!((f.criteria.aic - (dev + 2.0 * f.tr_s)).abs() < 1e-9);
    let corr = 2.0 * f.tr_s * (f.tr_s + 1.0) / (64.0 - f.tr_s - 1.0);
    assert!((f.criteria.aicc - (f.criteria.aic + corr)).abs() < 1e-9);
    assert!((f.criteria.bic - (dev + f.tr_s * 64.0_f64.ln())).abs() < 1e-9);
    let lite = fit_lite(&m, 30.0).unwrap();
    assert_eq!(lite.criteria, f.criteria);
    assert_eq!(lite.params, f.params);
}

#[test]
fn glm_fits_do_not_depend_on_the_thread_count() {
    let d = data();
    let m = glm(&d, &d.binary, Family::Binomial, &d.offset);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| fit_full(&m, 40.0, true).unwrap())
    };
    let (one, many) = (run(1), run(4));
    assert_eq!(one.params, many.params);
    assert_eq!(one.bse, many.bse);
    assert_eq!(one.criteria, many.criteria);
    assert_eq!(one.p_dev.len(), 64);
}

#[test]
fn glm_selection_runs_on_the_family_criterion() {
    let d = data();
    let m = glm(&d, &d.counts, Family::Poisson, &d.offset);
    let o = SelectOptions {
        criterion: Criterion::Aicc,
        search: Search::Golden,
        bw_min: None,
        bw_max: None,
        n_loc: 1,
        constant: true,
        tol: 1e-6,
        max_iter: 200,
    };
    let s = select_bandwidth(&m, &o).unwrap();
    let at_best = fit_lite(&m, s.bw).unwrap().criteria.aicc;
    assert!((s.score - at_best).abs() < 1e-9);
    assert!(s.history.iter().all(|(_, v)| v.is_finite()));
}

fn mgwr_options(k: usize) -> MgwrOptions {
    MgwrOptions {
        criterion: Criterion::Aicc,
        search: Search::Golden,
        multi_bw_min: vec![None; k],
        multi_bw_max: vec![None; k],
        init_bw: None,
        tol: 1e-6,
        max_iter: 200,
        tol_multi: 1e-5,
        max_iter_multi: 200,
        rss_score: false,
        bws_same_times: 5,
        inference: true,
        chunk_size: 256,
    }
}

fn standardized(d: &Data) -> (Vec<f64>, Vec<f64>) {
    let n = 64;
    let col: Vec<f64> = d.x.chunks(2).map(|r| r[1]).collect();
    let mean = col.iter().sum::<f64>() / n as f64;
    let sd = (col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let xs: Vec<f64> = col.iter().flat_map(|v| [1.0, (v - mean) / sd]).collect();
    let y: Vec<f64> =
        d.x.chunks(2)
            .zip(d.coords.chunks(2))
            .map(|(r, c)| 0.2 * c[0] + (0.5 + 0.1 * c[1]) * (r[1] - mean) / sd)
            .collect();
    (xs, y)
}

#[test]
fn mgwr_chunking_does_not_change_inference() {
    let d = data();
    let (xs, y) = standardized(&d);
    let m = Model::new(&d.coords, &y, &xs, 2, Kernel::Bisquare, true, false).unwrap();
    let whole = fit_mgwr(&m, &mgwr_options(2)).unwrap();
    let mut small = mgwr_options(2);
    small.chunk_size = 5;
    let chunked = fit_mgwr(&m, &small).unwrap();
    assert_eq!(whole.params, chunked.params);
    assert_eq!(whole.bws, chunked.bws);
    let (a, b) = (whole.inference.unwrap(), chunked.inference.unwrap());
    for (u, v) in a.enp_j.iter().zip(&b.enp_j) {
        assert!((u - v).abs() < 1e-10);
    }
    for (u, v) in a.cct.iter().zip(&b.cct) {
        assert!((u - v).abs() <= 1e-10 * u.abs().max(1.0));
    }
}

#[test]
fn mgwr_without_inference_returns_the_backfitted_surface_only() {
    let d = data();
    let (xs, y) = standardized(&d);
    let m = Model::new(&d.coords, &y, &xs, 2, Kernel::Bisquare, true, false).unwrap();
    let mut o = mgwr_options(2);
    o.inference = false;
    let fit = fit_mgwr(&m, &o).unwrap();
    assert!(fit.inference.is_none());
    assert_eq!(fit.params.len(), 128);
    assert_eq!(fit.bws.len(), 2);
    assert_eq!(fit.bws_history.len(), fit.scores.len());
    let rss: f64 = fit.resid.iter().map(|u| u * u).sum();
    assert!((rss - fit.resid_ss).abs() < 1e-12);
    for i in 0..64 {
        let p = fit.params[2 * i] * xs[2 * i] + fit.params[2 * i + 1] * xs[2 * i + 1];
        assert!((fit.predy[i] - p).abs() < 1e-12);
    }
}

#[test]
fn mgwr_validates_bounds_lengths() {
    let d = data();
    let (xs, y) = standardized(&d);
    let m = Model::new(&d.coords, &y, &xs, 2, Kernel::Bisquare, true, false).unwrap();
    let mut o = mgwr_options(2);
    o.multi_bw_min = vec![None; 3];
    assert!(fit_mgwr(&m, &o).unwrap_err().contains("multi_bw_min"));
    let mut o = mgwr_options(2);
    o.chunk_size = 0;
    assert!(fit_mgwr(&m, &o).is_err());
    let mut o = mgwr_options(2);
    o.tol_multi = 0.0;
    assert!(fit_mgwr(&m, &o).is_err());
}
