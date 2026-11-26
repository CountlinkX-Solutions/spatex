//! Multiscale GWR (`mgwr.gwr.MGWR` with `mgwr.search.multi_bw`): one bandwidth per covariate
//! found by GAM backfitting, then inference from per-covariate hat matrices.
//!
//! Only the Gaussian family is supported (as in mgwr). The backfitting runs univariate local
//! fits, each `O(n^2)`, so it never stores an `n x n` matrix. The inference needs the hat
//! matrices `R_j` (`n x n` each); they are computed in blocks of `chunk_size` columns, so memory
//! is `O(n * chunk_size * k)` while time is `O(iterations * k * n^3)`. Callers should guard `n`.

use rayon::prelude::*;

use super::family::Family;
use super::fit::{fit_lite, Criteria};
use super::kernel::{weights, Workspace};
use super::lu::Lu;
use super::select::{select_bandwidth, Criterion, Search, SelectOptions};
use super::{t_critical, Model};

/// Options of [`fit_mgwr`].
#[derive(Debug, Clone)]
pub struct MgwrOptions {
    /// Criterion, search method and tolerances of every (univariate and initial) bandwidth search.
    pub criterion: Criterion,
    pub search: Search,
    pub tol: f64,
    pub max_iter: usize,
    /// Per-column search bounds (`k` entries each); `None` uses the default bracket.
    pub multi_bw_min: Vec<Option<f64>>,
    pub multi_bw_max: Vec<Option<f64>>,
    /// Starting bandwidth of the backfitting; searched with a GWR on all columns when `None`.
    pub init_bw: Option<f64>,
    /// Backfitting convergence tolerance and iteration cap.
    pub tol_multi: f64,
    pub max_iter_multi: usize,
    /// Converge on the relative change of the residual sum of squares instead of mgwr's SOC.
    pub rss_score: bool,
    /// Stop re-searching bandwidths after this many identical iterations.
    pub bws_same_times: usize,
    /// Compute the hat-matrix based inference.
    pub inference: bool,
    /// Columns of each hat-matrix block.
    pub chunk_size: usize,
}

/// Inference of an MGWR fit (`MGWRResults`).
#[derive(Debug, Clone)]
pub struct Inference {
    /// Effective number of parameters per covariate (trace of each `R_j`).
    pub enp_j: Vec<f64>,
    /// `n x k`, row-major: `sigma2 * sum_c (R_j[i, c] / x_ij)^2` (mgwr's scaled `CCT`).
    pub cct: Vec<f64>,
    pub tr_s: f64,
    /// `resid_ss / (n - tr(S))`.
    pub sigma2: f64,
    pub bse: Vec<f64>,
    pub tvalues: Vec<f64>,
    pub criteria: Criteria,
    pub r2: f64,
    pub adj_r2: f64,
    /// `k x 3`, row-major: `[0.1, 0.05, 0.001] / enp_j`.
    pub adj_alpha_j: Vec<f64>,
    /// Per covariate `t.ppf(1 - adj_alpha_j[:, 1] / 2, n - 1)`.
    pub critical_tval: Vec<f64>,
}

/// Result of [`fit_mgwr`].
#[derive(Debug, Clone)]
pub struct MgwrFit {
    pub n: usize,
    pub k: usize,
    pub bws: Vec<f64>,
    pub bw_init: f64,
    /// One row of `k` bandwidths per backfitting iteration.
    pub bws_history: Vec<Vec<f64>>,
    /// The evaluations `(bw, score)` of every bandwidth search of the backfitting, in order
    /// (mgwr's `Sel_BW.sel_hist` of an MGWR selector).
    pub sel_hist: Vec<Vec<(f64, f64)>>,
    pub scores: Vec<f64>,
    pub converged: bool,
    /// `n x k`, row-major.
    pub params: Vec<f64>,
    pub predy: Vec<f64>,
    pub resid: Vec<f64>,
    pub resid_ss: f64,
    pub inference: Option<Inference>,
}

fn column(m: &Model, j: usize) -> Vec<f64> {
    (0..m.n).map(|i| m.x[i * m.k + j]).collect()
}

fn check_options(m: &Model, o: &MgwrOptions) -> Result<(), String> {
    if m.family != Family::Gaussian {
        return Err("MGWR supports only the gaussian family".to_string());
    }
    if o.multi_bw_min.len() != m.k {
        return Err(format!(
            "multi_bw_min must have {} entries (one per column), got {}",
            m.k,
            o.multi_bw_min.len()
        ));
    }
    if o.multi_bw_max.len() != m.k {
        return Err(format!(
            "multi_bw_max must have {} entries (one per column), got {}",
            m.k,
            o.multi_bw_max.len()
        ));
    }
    if !(o.tol_multi > 0.0 && o.tol_multi.is_finite()) {
        return Err("tol_multi must be positive".to_string());
    }
    if o.max_iter_multi == 0 {
        return Err("max_iter_multi must be at least 1".to_string());
    }
    if o.chunk_size == 0 {
        return Err("chunk_size must be at least 1".to_string());
    }
    Ok(())
}

/// Multiscale GWR on `m` (whose `x` already holds the intercept, if any): bandwidth search by
/// backfitting, then (optionally) inference. `m` must be a Gaussian model.
pub fn fit_mgwr(m: &Model, o: &MgwrOptions) -> Result<MgwrFit, String> {
    check_options(m, o)?;
    let (n, k) = (m.n, m.k);
    let search = |model: &Model,
                  n_loc: usize,
                  bw_min: Option<f64>,
                  bw_max: Option<f64>|
     -> Result<(f64, Vec<(f64, f64)>), String> {
        let options = SelectOptions {
            criterion: o.criterion,
            search: o.search,
            bw_min,
            bw_max,
            n_loc,
            constant: false,
            tol: o.tol,
            max_iter: o.max_iter,
        };
        let found = select_bandwidth(model, &options)?;
        Ok((found.bw, found.history))
    };
    let bw_init = match o.init_bw {
        Some(bw) => bw,
        None => search(m, k, None, None)?.0,
    };
    let init = fit_lite(m, bw_init)?;
    let mut err: Vec<f64> = init.resid.clone();
    let mut xb: Vec<f64> = init.params.iter().zip(m.x).map(|(p, x)| p * x).collect();
    let mut rss: f64 = err.iter().map(|e| e * e).sum();
    let mut bws = vec![0.0; k];
    let mut history: Vec<Vec<f64>> = Vec::new();
    let mut sel_hist: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    let mut params = vec![0.0; n * k];
    let mut stable = 0_usize;
    let mut converged = false;
    for iter in 1..=o.max_iter_multi {
        let mut new_xb = vec![0.0; n * k];
        for j in 0..k {
            let temp_y: Vec<f64> = (0..n).map(|i| xb[i * k + j] + err[i]).collect();
            let temp_x = column(m, j);
            let sub = Model::new(
                m.coords,
                &temp_y,
                &temp_x,
                1,
                m.kernel,
                m.fixed,
                m.spherical,
            )?;
            let bw = if stable >= o.bws_same_times {
                bws[j]
            } else {
                let (bw, searched) = search(&sub, 1, o.multi_bw_min[j], o.multi_bw_max[j])?;
                sel_hist.push(searched);
                bw
            };
            let fit = fit_lite(&sub, bw)?;
            for i in 0..n {
                err[i] = temp_y[i] - fit.predy[i];
                new_xb[i * k + j] = fit.predy[i];
                params[i * k + j] = fit.params[i];
            }
            bws[j] = bw;
        }
        if iter > 1 && history.last().is_some_and(|prev| *prev == bws) {
            stable += 1;
        } else {
            stable = 0;
        }
        let num = xb
            .iter()
            .zip(&new_xb)
            .map(|(a, b)| (b - a) * (b - a))
            .sum::<f64>()
            / n as f64;
        let den: f64 = new_xb
            .chunks(k)
            .map(|r| {
                let s: f64 = r.iter().sum();
                s * s
            })
            .sum();
        let mut score = (num / den).sqrt();
        xb = new_xb;
        if o.rss_score {
            let new_rss: f64 = (0..n)
                .map(|i| {
                    let p: f64 = (0..k).map(|j| params[i * k + j] * m.x[i * k + j]).sum();
                    (m.y[i] - p) * (m.y[i] - p)
                })
                .sum();
            score = ((new_rss - rss) / new_rss).abs();
            rss = new_rss;
        }
        scores.push(score);
        history.push(bws.clone());
        if score < o.tol_multi {
            converged = true;
            break;
        }
    }
    let predy: Vec<f64> = (0..n)
        .map(|i| (0..k).map(|j| params[i * k + j] * m.x[i * k + j]).sum())
        .collect();
    let resid: Vec<f64> = m.y.iter().zip(&predy).map(|(y, p)| y - p).collect();
    let resid_ss: f64 = resid.iter().map(|u| u * u).sum();
    let inference = if o.inference {
        Some(inference(
            m,
            bw_init,
            &history,
            o.chunk_size,
            &params,
            resid_ss,
        )?)
    } else {
        None
    };
    Ok(MgwrFit {
        n,
        k,
        bws,
        bw_init,
        bws_history: history,
        sel_hist,
        scores,
        converged,
        params,
        predy,
        resid,
        resid_ss,
        inference,
    })
}

/// Applies the backfitting smoother of covariate `j` at bandwidth `bw` to an `n x cs` matrix:
/// row `i` of the result is `x_ij / sum_l w_il x_lj^2 * sum_l w_il x_lj old[l, :]`.
fn smooth(m: &Model, j: usize, bw: f64, old: &[f64], cs: usize) -> Result<Vec<f64>, String> {
    let (n, k) = (m.n, m.k);
    let mut out = vec![0.0; n * cs];
    let status: Vec<Result<(), String>> = out
        .par_chunks_mut(cs)
        .enumerate()
        .map_init(
            || (Workspace::new(n), vec![0.0; cs]),
            |(ws, acc), (i, row)| {
                weights(m, i, bw, ws)?;
                acc.fill(0.0);
                let mut den = 0.0;
                for l in 0..n {
                    let w = ws.w[l];
                    if w == 0.0 {
                        continue;
                    }
                    let xl = m.x[l * k + j];
                    let wx = w * xl;
                    den += wx * xl;
                    let src = &old[l * cs..(l + 1) * cs];
                    for (a, s) in acc.iter_mut().zip(src) {
                        *a += wx * s;
                    }
                }
                let coef = m.x[i * k + j] / den;
                for (r, a) in row.iter_mut().zip(acc.iter()) {
                    *r = coef * a;
                }
                Ok(())
            },
        )
        .collect();
    status.into_iter().collect::<Result<(), String>>()?;
    Ok(out)
}

/// Initial partial hat matrices of the columns `start..start + cs`: for every location `i`
/// the GWR at `bw_init`, `pR[i, c, j] = x_ij (A_i^{-1} x_c)_j w_ic`, laid out per covariate.
fn initial_blocks(
    m: &Model,
    bw_init: f64,
    start: usize,
    cs: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let (n, k) = (m.n, m.k);
    let zeros = vec![0.0; n];
    let rows: Vec<Result<Vec<f64>, String>> = (0..n)
        .into_par_iter()
        .map_init(
            || Workspace::new(n),
            |ws, i| {
                weights(m, i, bw_init, ws)?;
                let (a, _) = super::family::normal_equations(m.x, k, &ws.w, &zeros);
                let lu = Lu::factor(a, k).map_err(|e| format!("{e} at location {i}"))?;
                let mut out = vec![0.0; k * cs];
                for c in 0..cs {
                    let v = lu.solve(m.row(start + c));
                    let w = ws.w[start + c];
                    for j in 0..k {
                        out[j * cs + c] = m.x[i * k + j] * (v[j] * w);
                    }
                }
                Ok(out)
            },
        )
        .collect();
    let mut blocks = vec![vec![0.0; n * cs]; k];
    for (i, row) in rows.into_iter().enumerate() {
        let row = row?;
        for j in 0..k {
            blocks[j][i * cs..(i + 1) * cs].copy_from_slice(&row[j * cs..(j + 1) * cs]);
        }
    }
    Ok(blocks)
}

/// `MGWR._chunk_compute_R` over all column blocks, plus the derived statistics.
fn inference(
    m: &Model,
    bw_init: f64,
    history: &[Vec<f64>],
    chunk_size: usize,
    params: &[f64],
    resid_ss: f64,
) -> Result<Inference, String> {
    let (n, k) = (m.n, m.k);
    let mut enp_j = vec![0.0; k];
    let mut cct = vec![0.0; n * k];
    let mut start = 0;
    while start < n {
        let cs = chunk_size.min(n - start);
        let mut pr = initial_blocks(m, bw_init, start, cs)?;
        // err = E0 - sum_j pR_j, with E0 the identity columns of the block.
        let mut err = vec![0.0; n * cs];
        for c in 0..cs {
            err[(start + c) * cs + c] = 1.0;
        }
        for block in &pr {
            for (e, p) in err.iter_mut().zip(block) {
                *e -= p;
            }
        }
        for bws in history {
            for j in 0..k {
                let old: Vec<f64> = pr[j].iter().zip(&err).map(|(p, e)| p + e).collect();
                let new = smooth(m, j, bws[j], &old, cs)?;
                for (e, (o, nw)) in err.iter_mut().zip(old.iter().zip(&new)) {
                    *e = o - nw;
                }
                pr[j] = new;
            }
        }
        for j in 0..k {
            for c in 0..cs {
                enp_j[j] += pr[j][(start + c) * cs + c];
            }
            for i in 0..n {
                let xij = m.x[i * k + j];
                let row = &pr[j][i * cs..(i + 1) * cs];
                cct[i * k + j] += row.iter().map(|v| (v / xij) * (v / xij)).sum::<f64>();
            }
        }
        start += cs;
    }
    let nf = n as f64;
    let tr_s: f64 = enp_j.iter().sum();
    let sigma2 = resid_ss / (nf - tr_s);
    let cct: Vec<f64> = cct.iter().map(|c| c * sigma2).collect();
    let bse: Vec<f64> = cct.iter().map(|c| c.sqrt()).collect();
    let y_bar = m.y.iter().sum::<f64>() / nf;
    let tss: f64 = m.y.iter().map(|y| (y - y_bar) * (y - y_bar)).sum();
    let r2 = 1.0 - resid_ss / tss;
    let adj_r2 = 1.0 - (1.0 - r2) * (nf - 1.0) / (nf - tr_s - 1.0);
    let adj_alpha_j: Vec<f64> = enp_j
        .iter()
        .flat_map(|e| [0.1 / e, 0.05 / e, 0.001 / e])
        .collect();
    let critical_tval = (0..k)
        .map(|j| t_critical(adj_alpha_j[3 * j + 1].abs(), nf - 1.0))
        .collect();
    let tvalues = params.iter().zip(&bse).map(|(p, s)| p / s).collect();
    Ok(Inference {
        enp_j,
        cct,
        tr_s,
        sigma2,
        bse,
        tvalues,
        criteria: Criteria::gaussian(n, tr_s, resid_ss),
        r2,
        adj_r2,
        adj_alpha_j,
        critical_tval,
    })
}
