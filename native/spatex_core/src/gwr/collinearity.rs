//! Local collinearity diagnostics of the geographically weighted design matrix
//! (`GWRResults.local_collinearity` and `MGWRResults.local_collinearity` of mgwr 2.2.1).
//!
//! For every location the kernel weights are normalized to sum to one and
//!
//! * the local correlations between the explanatory variables and their variance inflation
//!   factors (`diag(R^{-1})`) come from the weighted covariance (GWR only);
//! * the local condition number and the variance-decomposition proportions (Belsley, Kuh and
//!   Welsch) come from the singular value decomposition of the weighted design matrix whose
//!   columns are scaled to unit length.
//!
//! Weights are recomputed per location, so memory is `O(n k)` per worker thread.

use faer::Mat;
use rayon::prelude::*;

use super::kernel::{check_bandwidth, weights, Workspace};
use super::Model;
use crate::regression::linalg;

/// Local collinearity indicators. Matrices are row-major with one row per location.
#[derive(Debug, Clone)]
pub struct Collinearity {
    pub n: usize,
    pub k: usize,
    /// Pairwise correlations per location: `ncor = q (q - 1) / 2` for `q` explanatory
    /// variables, in the order `(0, 1), (0, 2), ..., (1, 2), ...` (empty for MGWR).
    pub ncor: usize,
    /// Variance inflation factors per location: one per explanatory variable (empty for MGWR).
    pub nvif: usize,
    pub corr: Vec<f64>,
    pub vif: Vec<f64>,
    /// Largest over smallest singular value of the scaled weighted design, per location.
    pub condition_number: Vec<f64>,
    /// `n x k`: the variance-decomposition proportions of the smallest singular value.
    pub vdp: Vec<f64>,
}

struct Row {
    corr: Vec<f64>,
    vif: Vec<f64>,
    cn: f64,
    vdp: Vec<f64>,
}

/// Scales the columns of `xw` (`n x k`, row-major) to unit length and returns the condition
/// number and the variance-decomposition proportions of the smallest singular value.
fn decompose(xw: &[f64], n: usize, k: usize) -> (f64, Vec<f64>) {
    let norms: Vec<f64> = (0..k)
        .map(|c| {
            (0..n)
                .map(|r| xw[r * k + c] * xw[r * k + c])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let a = Mat::from_fn(n, k, |r, c| xw[r * k + c] / norms[c]);
    let finite = (0..n).all(|r| (0..k).all(|c| a[(r, c)].is_finite()));
    let svd = if finite { a.thin_svd().ok() } else { None };
    let Some(svd) = svd else {
        return (f64::NAN, vec![f64::NAN; k]);
    };
    let s: Vec<f64> = (0..k).map(|i| svd.S().column_vector()[i]).collect();
    let v = svd.V();
    // phi[kk][j] = V[j][kk]^2 / s[kk]^2; pi is phi normalized over the components kk.
    let phi = |kk: usize, j: usize| v[(j, kk)] * v[(j, kk)] / (s[kk] * s[kk]);
    let vdp = (0..k)
        .map(|j| phi(k - 1, j) / (0..k).map(|kk| phi(kk, j)).sum::<f64>())
        .collect();
    (s[0] / s[k - 1], vdp)
}

fn check_columns(explanatory: usize) -> Result<(), String> {
    if explanatory < 2 {
        return Err(format!(
            "local collinearity needs at least two explanatory variables, got {explanatory}"
        ));
    }
    Ok(())
}

fn local_row(
    m: &Model,
    i: usize,
    bw: f64,
    first: usize,
    ws: &mut Workspace,
) -> Result<Row, String> {
    let (n, k) = (m.n, m.k);
    weights(m, i, bw, ws)?;
    let sw: f64 = ws.w.iter().sum();
    let wi: Vec<f64> = ws.w.iter().map(|w| w / sw).collect();
    let q = k - first;
    // Weighted covariance of the explanatory columns (the normalizing factor cancels in the
    // correlation).
    let mean: Vec<f64> = (0..q)
        .map(|a| (0..n).map(|r| wi[r] * m.x[r * k + first + a]).sum())
        .collect();
    let mut cov = vec![0.0; q * q];
    for (r, &w) in wi.iter().enumerate().take(n) {
        let row = &m.x[r * k + first..(r + 1) * k];
        for a in 0..q {
            let da = w * (row[a] - mean[a]);
            for b in a..q {
                cov[a * q + b] += da * (row[b] - mean[b]);
            }
        }
    }
    let mut cor = vec![0.0; q * q];
    for a in 0..q {
        for b in a..q {
            let v = cov[a * q + b] / (cov[a * q + a] * cov[b * q + b]).sqrt();
            cor[a * q + b] = v;
            cor[b * q + a] = v;
        }
    }
    let mut corr = Vec::with_capacity(q * (q - 1) / 2);
    for a in 0..q {
        for b in a + 1..q {
            corr.push(cor[a * q + b]);
        }
    }
    let vif = if cor.iter().all(|v| v.is_finite()) {
        let inv = linalg::inverse(&Mat::from_fn(q, q, |a, b| cor[a * q + b]))
            .map_err(|e| format!("{e}: collinear regressors at location {i}"))?;
        (0..q).map(|a| inv[(a, a)]).collect()
    } else {
        vec![f64::NAN; q]
    };
    let mut xw = vec![0.0; n * k];
    for r in 0..n {
        for c in 0..k {
            xw[r * k + c] = m.x[r * k + c] * wi[r];
        }
    }
    let (cn, vdp) = decompose(&xw, n, k);
    Ok(Row { corr, vif, cn, vdp })
}

fn assemble(n: usize, k: usize, rows: Vec<Row>) -> Collinearity {
    let ncor = rows.first().map_or(0, |r| r.corr.len());
    let nvif = rows.first().map_or(0, |r| r.vif.len());
    let mut out = Collinearity {
        n,
        k,
        ncor,
        nvif,
        corr: Vec::with_capacity(n * ncor),
        vif: Vec::with_capacity(n * nvif),
        condition_number: Vec::with_capacity(n),
        vdp: Vec::with_capacity(n * k),
    };
    for r in rows {
        out.corr.extend(r.corr);
        out.vif.extend(r.vif);
        out.condition_number.push(r.cn);
        out.vdp.extend(r.vdp);
    }
    out
}

/// `GWRResults.local_collinearity()` for the bandwidth `bw`.
///
/// `constant` says that the first column of `x` is the intercept: it is left out of the
/// correlations and the variance inflation factors but not of the decomposition. At least two
/// explanatory variables are needed (mgwr crashes inside numpy with fewer).
pub fn local_collinearity(m: &Model, bw: f64, constant: bool) -> Result<Collinearity, String> {
    let first = usize::from(constant);
    check_columns(m.k - first)?;
    check_bandwidth(m, bw)?;
    let rows: Vec<Result<Row, String>> = (0..m.n)
        .into_par_iter()
        .map_init(
            || Workspace::new(m.n),
            |ws, i| local_row(m, i, bw, first, ws),
        )
        .collect();
    let rows: Vec<Row> = rows.into_iter().collect::<Result<_, _>>()?;
    Ok(assemble(m.n, m.k, rows))
}

/// `MGWRResults.local_collinearity()`: column `j` of the design is weighted with its own
/// bandwidth `bws[j]`. Only the condition number and the decomposition proportions exist
/// (`corr` and `vif` are empty).
pub fn local_collinearity_mgwr(m: &Model, bws: &[f64]) -> Result<Collinearity, String> {
    let (n, k) = (m.n, m.k);
    if bws.len() != k {
        return Err(format!(
            "expected one bandwidth per design column ({k}), got {}",
            bws.len()
        ));
    }
    check_columns(k)?;
    for &bw in bws {
        check_bandwidth(m, bw)?;
    }
    let rows: Vec<Result<Row, String>> = (0..n)
        .into_par_iter()
        .map_init(
            || Workspace::new(n),
            |ws, i| {
                let mut xw = vec![0.0; n * k];
                for (c, &bw) in bws.iter().enumerate() {
                    weights(m, i, bw, ws)?;
                    let sw: f64 = ws.w.iter().sum();
                    for r in 0..n {
                        xw[r * k + c] = m.x[r * k + c] * (ws.w[r] / sw);
                    }
                }
                let (cn, vdp) = decompose(&xw, n, k);
                Ok(Row {
                    corr: Vec::new(),
                    vif: Vec::new(),
                    cn,
                    vdp,
                })
            },
        )
        .collect();
    let rows: Vec<Row> = rows.into_iter().collect::<Result<_, _>>()?;
    Ok(assemble(n, k, rows))
}
