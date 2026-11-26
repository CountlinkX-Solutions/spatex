//! GWR prediction at new locations (`GWR.predict` of mgwr 2.2.1).
//!
//! For every prediction point the kernel weights come from its distances to the calibration
//! locations (an adaptive bandwidth is the distance to the `bw`-th nearest calibration
//! location), the local coefficients from the weighted fit of the calibration sample at that
//! point, and the standard errors from the same local fit scaled by the calibration residual
//! variance (`exog_scale`). Like the fit, nothing of size `m x n` is stored: weights are
//! recomputed per point, so memory is `O(m k + n)` per worker thread.

use rayon::prelude::*;

use super::family::{iwls, normal_equations, Family, IwlsInput};
use super::fit::{cct_diagonal, fit_full};
use super::kernel::{check_bandwidth, weights_from, Workspace};
use super::lu::Lu;
use super::Model;

/// Scale and residuals of the calibration sample, mgwr's `exog_scale` and `exog_resid`.
#[derive(Debug, Clone, Copy)]
pub struct Exog<'a> {
    pub scale: f64,
    /// One response residual per calibration observation.
    pub resid: &'a [f64],
}

/// Local coefficients and predictions at the new locations; matrices are row-major `m x k`.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub m: usize,
    pub k: usize,
    pub params: Vec<f64>,
    pub bse: Vec<f64>,
    pub tvalues: Vec<f64>,
    /// `sum_j P_ij params_ij`: the fitted mean for Gaussian models and, as in mgwr, the
    /// linear predictor (not the mean) for Poisson and Binomial.
    pub predictions: Vec<f64>,
    /// Local R-squared from the calibration residuals (Gaussian only; empty otherwise).
    pub local_r2: Vec<f64>,
    /// The scale the covariances were multiplied by.
    pub scale: f64,
}

struct Point {
    beta: Vec<f64>,
    cct: Vec<f64>,
    local_r2: f64,
}

fn local_point(
    model: &Model,
    j: usize,
    xy: (f64, f64),
    bw: f64,
    resid: &[f64],
    ws: &mut Workspace,
) -> Result<Point, String> {
    let k = model.k;
    weights_from(model, xy.0, xy.1, bw, ws).map_err(|e| format!("{e} at prediction point {j}"))?;
    let at = |e: String| format!("{e} at prediction point {j}");
    let (beta, lu, s): (Vec<f64>, Lu, Option<Vec<f64>>) = if model.family == Family::Gaussian {
        let (a, xtwy) = normal_equations(model.x, k, &ws.w, model.y);
        let lu = Lu::factor(a, k).map_err(at)?;
        let beta = lu.solve(&xtwy);
        (beta, lu, None)
    } else {
        let fit = iwls(&IwlsInput {
            y: model.y,
            x: model.x,
            k,
            family: model.family,
            offset: model.offset,
            w: &ws.w,
            tol: model.iwls_tol,
            max_iter: model.iwls_max_iter,
        })
        .map_err(at)?;
        (fit.betas, fit.lu, Some(fit.s))
    };
    // B = X' C^2 X with c_j = w_j s_j, the middle of diag(C C').
    let mut b = vec![0.0; k * k];
    for (jj, &wj) in ws.w.iter().enumerate() {
        if wj == 0.0 {
            continue;
        }
        let cj = wj * s.as_ref().map_or(1.0, |s| s[jj]);
        let c2 = cj * cj;
        let xj = model.row(jj);
        for p in 0..k {
            let wx = c2 * xj[p];
            for q in 0..k {
                b[p * k + q] += wx * xj[q];
            }
        }
    }
    let cct = cct_diagonal(&lu, &b, k);
    let local_r2 = if model.family == Family::Gaussian {
        let sum_w: f64 = ws.w.iter().sum();
        let sum_yw: f64 = ws.w.iter().zip(model.y).map(|(w, y)| w * y).sum();
        let y_bar = sum_yw / sum_w;
        let mut tss = 0.0;
        let mut rss = 0.0;
        for ((w, y), u) in ws.w.iter().zip(model.y).zip(resid) {
            tss += w * (y - y_bar) * (y - y_bar);
            rss += w * u * u;
        }
        (tss - rss) / tss
    } else {
        f64::NAN
    };
    Ok(Point {
        beta,
        cct,
        local_r2,
    })
}

/// Predicts at `points` (interleaved `x, y`, `m` of them) with regressors `p` (row-major
/// `m x k`, including the intercept column when the model has one).
///
/// `exog` supplies the calibration scale and residuals; without it a Gaussian model is fitted
/// at the calibration locations first (`sigma2_v1` picks its residual variance, as in
/// [`fit_full`]) and a GLM uses scale one. Errors when a local design matrix is singular.
pub fn predict(
    model: &Model,
    bw: f64,
    sigma2_v1: bool,
    points: &[f64],
    p: &[f64],
    exog: Option<Exog>,
) -> Result<Prediction, String> {
    let k = model.k;
    if !points.len().is_multiple_of(2) {
        return Err("points must hold two coordinates per prediction point".to_string());
    }
    let m = points.len() / 2;
    if p.len() != m * k {
        return Err(format!(
            "p has {} values, expected {m} rows x {k} columns",
            p.len()
        ));
    }
    check_bandwidth(model, bw)?;
    let gaussian = model.family == Family::Gaussian;
    let (scale, resid): (f64, Vec<f64>) = match exog {
        Some(e) => {
            if e.resid.len() != model.n {
                return Err(format!(
                    "resid has {} values, expected {} (one per observation)",
                    e.resid.len(),
                    model.n
                ));
            }
            (e.scale, e.resid.to_vec())
        }
        None if gaussian => {
            let fit = fit_full(model, bw, sigma2_v1)?;
            (fit.scale, fit.resid)
        }
        None => (1.0, Vec::new()),
    };
    let results: Vec<Result<Point, String>> = (0..m)
        .into_par_iter()
        .map_init(
            || Workspace::new(model.n),
            |ws, j| local_point(model, j, (points[2 * j], points[2 * j + 1]), bw, &resid, ws),
        )
        .collect();
    let rows: Vec<Point> = results.into_iter().collect::<Result<_, _>>()?;
    let mut params = Vec::with_capacity(m * k);
    let mut bse = Vec::with_capacity(m * k);
    for r in &rows {
        params.extend_from_slice(&r.beta);
        bse.extend(r.cct.iter().map(|c| (c * scale).sqrt()));
    }
    let tvalues = params.iter().zip(&bse).map(|(b, s)| b / s).collect();
    let predictions = (0..m)
        .map(|j| (0..k).map(|c| p[j * k + c] * params[j * k + c]).sum())
        .collect();
    Ok(Prediction {
        m,
        k,
        params,
        bse,
        tvalues,
        predictions,
        local_r2: if gaussian {
            rows.iter().map(|r| r.local_r2).collect()
        } else {
            Vec::new()
        },
        scale,
    })
}
