//! Local weighted least squares (Gaussian) or IWLS (Poisson, Binomial) for every location
//! (`GWR._local_fit`) and the global diagnostics built from them (`GWRResults`,
//! `GWRResultsLite`).

use rayon::prelude::*;

use super::family::{iwls, normal_equations, null_mu, Family, IwlsInput};
use super::kernel::{check_bandwidth, weights, Workspace};
use super::lu::Lu;
use super::{t_critical, Model};

/// One location's local fit.
struct Row {
    beta: Vec<f64>,
    influ: f64,
    predy: f64,
    /// `sum_j S_ij^2 s_i^2` (full fits only).
    tr_sts: f64,
    /// Diagonal of `C C'` (full fits only).
    cct: Vec<f64>,
    /// IWLS iterations (1 for the Gaussian family).
    iterations: u32,
    /// Row `i` of the hat matrix (only when requested).
    hat_row: Vec<f64>,
}

/// What a pass over the locations keeps: `full` the inference quantities, `hat` also the whole
/// hat matrix row.
#[derive(Clone, Copy)]
struct Mode {
    full: bool,
    hat: bool,
}

fn local_row(m: &Model, i: usize, bw: f64, ws: &mut Workspace, mode: Mode) -> Result<Row, String> {
    let k = m.k;
    weights(m, i, bw, ws)?;
    let at = |e: String| format!("{e} at location {i}");
    // `s` are the square roots of the IRLS weights (all ones for the Gaussian family).
    let (beta, lu, predy, s, iterations) = if m.family == Family::Gaussian {
        let (a, xtwy) = normal_equations(m.x, k, &ws.w, m.y);
        let lu = Lu::factor(a, k).map_err(at)?;
        let beta = lu.solve(&xtwy);
        let predy: f64 = m.row(i).iter().zip(&beta).map(|(a, b)| a * b).sum();
        (beta, lu, predy, vec![1.0; m.n], 1)
    } else {
        let fit = iwls(&IwlsInput {
            y: m.y,
            x: m.x,
            k,
            family: m.family,
            offset: m.offset,
            w: &ws.w,
            tol: m.iwls_tol,
            max_iter: m.iwls_max_iter,
        })
        .map_err(at)?;
        let predy = fit.mu[i];
        (fit.betas, fit.lu, predy, fit.s, fit.iterations)
    };
    let xi = m.row(i);
    // x_i' A^{-1}; A is symmetric, so this is (A^{-1} x_i)'.
    let v = lu.solve(xi);
    let xv: f64 = xi.iter().zip(&v).map(|(a, b)| a * b).sum();
    let influ = ws.w[i] * (s[i] * s[i]) * xv;
    if !mode.full {
        return Ok(Row {
            beta,
            influ,
            predy,
            tr_sts: 0.0,
            cct: Vec::new(),
            iterations,
            hat_row: Vec::new(),
        });
    }
    // Row i of the hat matrix: S_ij = c_j (v . x_j) with c_j = w_j s_j; and B = X' C^2 X for
    // diag(C C').
    let mut tr_sts = 0.0;
    let mut b = vec![0.0; k * k];
    let mut hat_row = if mode.hat { vec![0.0; m.n] } else { Vec::new() };
    for (j, &wj) in ws.w.iter().enumerate() {
        if wj == 0.0 {
            continue;
        }
        let xj = m.row(j);
        let cj = wj * s[j];
        let sij: f64 = cj * v.iter().zip(xj).map(|(a, b)| a * b).sum::<f64>();
        tr_sts += sij * sij;
        if mode.hat {
            hat_row[j] = sij;
        }
        let c2 = cj * cj;
        for p in 0..k {
            let wx = c2 * xj[p];
            for q in 0..k {
                b[p * k + q] += wx * xj[q];
            }
        }
    }
    let tr_sts = tr_sts * (s[i] * s[i]);
    let cct = cct_diagonal(&lu, &b, k);
    Ok(Row {
        beta,
        influ,
        predy,
        tr_sts,
        cct,
        iterations,
        hat_row,
    })
}

/// Diagonal of `A^{-1} B A^{-1}` (`A = X' W s^2 X`, `B = X' C^2 X`): the unscaled coefficient
/// variances `diag(C C')` of one local fit. `b` is row-major `k x k`.
pub(crate) fn cct_diagonal(lu: &Lu, b: &[f64], k: usize) -> Vec<f64> {
    let cols: Vec<Vec<f64>> = (0..k)
        .map(|c| {
            let mut e = vec![0.0; k];
            e[c] = 1.0;
            lu.solve(&e)
        })
        .collect();
    (0..k)
        .map(|l| {
            let mut total = 0.0;
            for p in 0..k {
                for q in 0..k {
                    total += cols[p][l] * b[p * k + q] * cols[q][l];
                }
            }
            total
        })
        .collect()
}

fn rows(m: &Model, bw: f64, mode: Mode) -> Result<Vec<Row>, String> {
    check_bandwidth(m, bw)?;
    let results: Vec<Result<Row, String>> = (0..m.n)
        .into_par_iter()
        .map_init(
            || Workspace::new(m.n),
            |ws, i| local_row(m, i, bw, ws, mode),
        )
        .collect();
    results.into_iter().collect()
}

/// Information criteria of a fit with `tr_s = tr(S)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Criteria {
    pub llf: f64,
    pub aic: f64,
    pub aicc: f64,
    pub bic: f64,
}

impl Criteria {
    /// `get_AIC`, `get_AICc`, `get_BIC` of mgwr for the Gaussian family.
    pub fn gaussian(n: usize, tr_s: f64, resid_ss: f64) -> Self {
        let nf = n as f64;
        let half = nf / 2.0;
        let llf = -half * resid_ss.ln() - half * (1.0 + (std::f64::consts::PI / half).ln());
        Self {
            llf,
            aic: -2.0 * llf + 2.0 * (tr_s + 1.0),
            aicc: -2.0 * llf + 2.0 * nf * (tr_s + 1.0) / (nf - tr_s - 2.0),
            bic: -2.0 * llf + (tr_s + 1.0) * nf.ln(),
        }
    }

    /// The Poisson/Binomial versions: deviance based, `AIC = D + 2 tr(S)`,
    /// `AICc = AIC + 2 tr(S) (tr(S) + 1) / (n - tr(S) - 1)`, `BIC = D + tr(S) ln n`.
    pub fn glm(n: usize, tr_s: f64, deviance: f64, llf: f64) -> Self {
        let nf = n as f64;
        let aic = deviance + 2.0 * tr_s;
        Self {
            llf,
            aic,
            aicc: aic + 2.0 * tr_s * (tr_s + 1.0) / (nf - tr_s - 1.0),
            bic: deviance + tr_s * nf.ln(),
        }
    }
}

/// `sum(resid_dev(y, mu)^2)` of the model family.
fn global_deviance(m: &Model, mu: &[f64]) -> f64 {
    m.y.iter()
        .zip(mu)
        .map(|(&y, &mu)| m.family.resid_dev_sq(y, mu))
        .sum()
}

/// Criteria of the family for the fitted means `mu`.
fn criteria(m: &Model, tr_s: f64, resid_ss: f64, mu: &[f64]) -> Criteria {
    match m.family {
        Family::Gaussian => Criteria::gaussian(m.n, tr_s, resid_ss),
        family => Criteria::glm(m.n, tr_s, global_deviance(m, mu), family.loglike(m.y, mu)),
    }
}

/// `GWRResultsLite`: the minimum needed by the bandwidth search.
#[derive(Debug, Clone)]
pub struct Lite {
    /// `n x k`, row-major.
    pub params: Vec<f64>,
    pub influ: Vec<f64>,
    pub predy: Vec<f64>,
    pub resid: Vec<f64>,
    pub tr_s: f64,
    pub resid_ss: f64,
    pub criteria: Criteria,
    /// `sum((u / (1 - influ))^2) / n`.
    pub cv: f64,
}

fn flatten(rows: &[Row], k: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows.len() * k);
    for r in rows {
        out.extend_from_slice(&r.beta);
    }
    out
}

fn cross_validation(resid: &[f64], influ: &[f64], n: usize) -> f64 {
    resid
        .iter()
        .zip(influ)
        .map(|(u, h)| {
            let a = u / (1.0 - h);
            a * a
        })
        .sum::<f64>()
        / n as f64
}

/// Fits all locations keeping only coefficients, hat diagonal and criteria.
pub fn fit_lite(m: &Model, bw: f64) -> Result<Lite, String> {
    let rows = rows(
        m,
        bw,
        Mode {
            full: false,
            hat: false,
        },
    )?;
    let influ: Vec<f64> = rows.iter().map(|r| r.influ).collect();
    let predy: Vec<f64> = rows.iter().map(|r| r.predy).collect();
    let resid: Vec<f64> = m.y.iter().zip(&predy).map(|(y, p)| y - p).collect();
    let tr_s = influ.iter().sum();
    let resid_ss = resid.iter().map(|u| u * u).sum();
    let cv = cross_validation(&resid, &influ, m.n);
    let criteria = criteria(m, tr_s, resid_ss, &predy);
    Ok(Lite {
        params: flatten(&rows, m.k),
        influ,
        predy,
        resid,
        tr_s,
        resid_ss,
        criteria,
        cv,
    })
}

/// `GWRResults`. Per-location matrices are row-major `n x k`.
///
/// `local_r2` is empty for GLM families; `deviance`, `resid_deviance` and `p_dev` are empty
/// for the Gaussian family. `r2` and `adj_r2` are NaN for GLM families (use `d2`, `adj_d2`).
#[derive(Debug, Clone)]
pub struct Fit {
    pub n: usize,
    pub k: usize,
    pub family: Family,
    pub params: Vec<f64>,
    pub bse: Vec<f64>,
    pub tvalues: Vec<f64>,
    pub local_r2: Vec<f64>,
    pub predy: Vec<f64>,
    pub resid: Vec<f64>,
    pub influ: Vec<f64>,
    pub std_res: Vec<f64>,
    pub cooks_d: Vec<f64>,
    pub resid_ss: f64,
    pub tr_s: f64,
    pub tr_sts: f64,
    pub enp: f64,
    pub sigma2: f64,
    /// `sigma2` for Gaussian, `1` for Poisson and Binomial.
    pub scale: f64,
    pub criteria: Criteria,
    pub cv: f64,
    pub r2: f64,
    pub adj_r2: f64,
    /// `1 - global_deviance / null_deviance` and its adjusted form.
    pub d2: f64,
    pub adj_d2: f64,
    pub null_deviance: f64,
    pub global_deviance: f64,
    /// Local deviance of the weighted mean model, local residual deviance and the local percent
    /// deviance `1 - resid_deviance / deviance` (GLM families only).
    pub deviance: Vec<f64>,
    pub resid_deviance: Vec<f64>,
    pub p_dev: Vec<f64>,
    /// IWLS iterations per location.
    pub iterations: Vec<u32>,
    /// The full hat matrix `S`, row-major `n x n`; empty unless requested (`hat_matrix`).
    pub hat: Vec<f64>,
    /// `n - 2 tr(S) + tr(S'S)`.
    pub df_resid: f64,
    /// Multiple-testing corrected alpha for the 0.1, 0.05 and 0.001 levels.
    pub adj_alpha: [f64; 3],
}

impl Fit {
    /// mgwr's `critical_tval`: with `alpha` the plain level, otherwise the corrected 0.05
    /// level. The degrees of freedom are `n - 1`, as in mgwr.
    pub fn critical_tval(&self, alpha: Option<f64>) -> f64 {
        let a = alpha.map_or(self.adj_alpha[1].abs(), f64::abs);
        t_critical(a, self.n as f64 - 1.0)
    }
}

/// Full fit at bandwidth `bw` (`GWR(...).fit()`).
///
/// `sigma2_v1` picks `n - tr(S)` (mgwr's default) instead of `n - 2 tr(S) + tr(S'S)` for the
/// residual variance, and `tr(S)` instead of `2 tr(S) - tr(S'S)` for the effective number of
/// parameters.
pub fn fit_full(m: &Model, bw: f64, sigma2_v1: bool) -> Result<Fit, String> {
    fit_full_with(m, bw, sigma2_v1, false)
}

/// [`fit_full`] that also keeps the full hat matrix `S` in [`Fit::hat`] when `hat` is true
/// (mgwr's `hat_matrix=True`). That costs `O(n^2)` memory, so it is opt-in.
pub fn fit_full_with(m: &Model, bw: f64, sigma2_v1: bool, hat: bool) -> Result<Fit, String> {
    let rows = rows(m, bw, Mode { full: true, hat })?;
    let (n, k) = (m.n, m.k);
    let nf = n as f64;
    let glm = m.family != Family::Gaussian;
    let influ: Vec<f64> = rows.iter().map(|r| r.influ).collect();
    let predy: Vec<f64> = rows.iter().map(|r| r.predy).collect();
    let resid: Vec<f64> = m.y.iter().zip(&predy).map(|(y, p)| y - p).collect();
    let tr_s: f64 = influ.iter().sum();
    let tr_sts: f64 = rows.iter().map(|r| r.tr_sts).sum();
    let resid_ss: f64 = resid.iter().map(|u| u * u).sum();
    let enp = if sigma2_v1 { tr_s } else { 2.0 * tr_s - tr_sts };
    let sigma2 = if sigma2_v1 {
        resid_ss / (nf - tr_s)
    } else {
        resid_ss / (nf - 2.0 * tr_s + tr_sts)
    };
    let scale = if glm { 1.0 } else { sigma2 };
    let params = flatten(&rows, k);
    let mut bse = Vec::with_capacity(n * k);
    for r in &rows {
        bse.extend(r.cct.iter().map(|c| (c * scale).sqrt()));
    }
    let tvalues: Vec<f64> = params.iter().zip(&bse).map(|(p, s)| p / s).collect();
    let std_res: Vec<f64> = resid
        .iter()
        .zip(&influ)
        .map(|(u, h)| u / (scale * (1.0 - h)).sqrt())
        .collect();
    let cooks_d: Vec<f64> = std_res
        .iter()
        .zip(&influ)
        .map(|(s, h)| s * s * h / (tr_s * (1.0 - h)))
        .collect();
    let cv = cross_validation(&resid, &influ, n);
    let null = null_mu(m.y, m.family, m.offset)?;
    let null_deviance = m.family.deviance(m.y, &null);
    let global_dev = if glm {
        global_deviance(m, &predy)
    } else {
        resid_ss
    };
    let d2 = 1.0 - global_dev / null_deviance;
    let adj_d2 = 1.0 - (1.0 - d2) * (nf - 1.0) / (nf - enp - 1.0);
    let kf = k as f64;
    let (local_r2, deviance, resid_deviance, p_dev) = if glm {
        let (dev, rdev) = local_deviances(m, bw, &predy)?;
        let p = dev.iter().zip(&rdev).map(|(d, r)| 1.0 - r / d).collect();
        (Vec::new(), dev, rdev, p)
    } else {
        (local_r2(m, bw, &resid)?, Vec::new(), Vec::new(), Vec::new())
    };
    Ok(Fit {
        n,
        k,
        family: m.family,
        params,
        bse,
        tvalues,
        local_r2,
        criteria: criteria(m, tr_s, resid_ss, &predy),
        predy,
        resid,
        influ,
        std_res,
        cooks_d,
        resid_ss,
        tr_s,
        tr_sts,
        enp,
        sigma2,
        scale,
        cv,
        r2: if glm { f64::NAN } else { d2 },
        adj_r2: if glm { f64::NAN } else { adj_d2 },
        d2,
        adj_d2,
        null_deviance,
        global_deviance: global_dev,
        deviance,
        resid_deviance,
        p_dev,
        iterations: rows.iter().map(|r| r.iterations).collect(),
        hat: rows
            .iter()
            .flat_map(|r| r.hat_row.iter().copied())
            .collect(),
        df_resid: nf - 2.0 * tr_s + tr_sts,
        adj_alpha: [0.1 * kf / enp, 0.05 * kf / enp, 0.001 * kf / enp],
    })
}

/// Geographically weighted R-squared per location: `(TSS_i - RSS_i) / TSS_i`.
fn local_r2(m: &Model, bw: f64, resid: &[f64]) -> Result<Vec<f64>, String> {
    let results: Vec<Result<f64, String>> = (0..m.n)
        .into_par_iter()
        .map_init(
            || Workspace::new(m.n),
            |ws, i| {
                weights(m, i, bw, ws)?;
                let sum_w: f64 = ws.w.iter().sum();
                let sum_yw: f64 = ws.w.iter().zip(m.y).map(|(w, y)| w * y).sum();
                let y_bar = sum_yw / sum_w;
                let mut tss = 0.0;
                let mut rss = 0.0;
                for ((w, y), u) in ws.w.iter().zip(m.y).zip(resid) {
                    tss += w * (y - y_bar) * (y - y_bar);
                    rss += w * u * u;
                }
                Ok((tss - rss) / tss)
            },
        )
        .collect();
    results.into_iter().collect()
}

/// `y ln(ratio)` with the convention `0 ln(0) = 0` (mgwr yields NaN for a zero count).
fn xlogy(y: f64, ratio: f64) -> f64 {
    if y == 0.0 {
        0.0
    } else {
        y * ratio.ln()
    }
}

/// Local deviance (of the local weighted-mean model) and local residual deviance per location
/// for the GLM families.
///
#[allow(clippy::type_complexity)]
fn local_deviances(m: &Model, bw: f64, mu: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let results: Vec<Result<(f64, f64), String>> = (0..m.n)
        .into_par_iter()
        .map_init(
            || Workspace::new(m.n),
            |ws, i| {
                weights(m, i, bw, ws)?;
                let sum_yw: f64 = ws.w.iter().zip(m.y).map(|(w, y)| w * y).sum();
                let sum_wo: f64 =
                    ws.w.iter()
                        .enumerate()
                        .map(|(j, w)| w * m.offset_at(j))
                        .sum();
                let y_bar = sum_yw / sum_wo;
                let dev = match m.family {
                    Family::Poisson => (0..m.n)
                        .map(|j| {
                            let fitted = y_bar * m.offset_at(j);
                            2.0 * ws.w[j] * (xlogy(m.y[j], m.y[j] / fitted) - (m.y[j] - fitted))
                        })
                        .sum(),
                    _ => {
                        let hit = (y_bar + 1e-200).ln();
                        let miss = (1.0 - y_bar + 1e-200).ln();
                        -2.0 * (0..m.n)
                            .map(|j| ws.w[j] * if m.y[j] == 1.0 { hit } else { miss })
                            .sum::<f64>()
                    }
                };
                let rdev = (0..m.n)
                    .map(|j| ws.w[j] * m.family.resid_dev_sq(m.y[j], mu[j]))
                    .sum();
                Ok((dev, rdev))
            },
        )
        .collect();
    let rows: Vec<(f64, f64)> = results.into_iter().collect::<Result<_, _>>()?;
    Ok(rows.into_iter().unzip())
}
