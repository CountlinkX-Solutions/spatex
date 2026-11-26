//! Ordinary least squares with spreg's diagnostics (`spreg.OLS`, spreg 1.9.1).
//!
//! The design matrix arrives as columns that already include the constant; adding it and
//! removing constant columns is the caller's job (as in `spreg.OLS`). Results come back as named
//! `f64` vectors (`Entries`) so the NIF layer stays a thin transport: scalars have one element,
//! matrices are row-major, pairs are `[statistic, p]`.

use std::collections::BTreeMap;

use faer::Mat;

use super::dist::{chi2_sf, f_sf, norm_two_sided, t_two_sided};
use super::linalg::{
    col_mat, column, fit, inverse, lag_transpose, mat_from_cols, mean, row_major, row_range,
    symmetric_eigenvalues, trace, trace_wtw_ww,
};
use crate::graph::Csr;

/// Named result vectors.
pub type Entries = BTreeMap<&'static str, Vec<f64>>;

/// Variance estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Robust {
    None,
    White,
    Hac,
}

/// What to compute.
pub struct Options<'a> {
    pub robust: Robust,
    /// Estimate sigma^2 with `n - k` (spreg's OLS default) rather than `n`.
    pub sig2n_k: bool,
    /// Kernel weights for HAC.
    pub gwk: Option<&'a Csr>,
    pub nonspat_diag: bool,
    pub white_test: bool,
    pub vif: bool,
    /// Spatial weights, used exactly as given (spreg never transforms them).
    pub w: Option<&'a Csr>,
    pub spat_diag: bool,
    pub moran: bool,
}

fn sq(v: f64) -> f64 {
    v * v
}

fn pair(stat: f64, p: f64) -> Vec<f64> {
    vec![stat, p]
}

/// Drops later columns equal to an earlier one (spreg compares with `sum |a - b| == 0`).
fn drop_duplicates(cols: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut omit = vec![false; cols.len()];
    for i in 0..cols.len() {
        for j in (i + 1)..cols.len() {
            if cols[i] == cols[j] {
                omit[j] = true;
            }
        }
    }
    cols.into_iter()
        .zip(omit)
        .filter_map(|(c, o)| (!o).then_some(c))
        .collect()
}

/// `g'Z (Z'Z)^-1 Z'g` for `Z` given as columns.
fn quad_form(z: &[Vec<f64>], g: &[f64]) -> Result<f64, String> {
    let zm = mat_from_cols(z, g.len());
    let ztz = zm.transpose() * &zm;
    let ztzi = inverse(&ztz)?;
    let ztg = zm.transpose() * col_mat(g);
    let part = ztg.transpose() * (&ztzi * &ztg);
    Ok(part[(0, 0)])
}

/// Breusch-Pagan / Koenker-Bassett share the squared-regressor design.
fn heteroskedasticity(x: &[Vec<f64>], u: &[f64], utu: f64) -> (Vec<f64>, Vec<f64>) {
    let n = u.len() as f64;
    let z = drop_duplicates(
        x.iter()
            .map(|c| c.iter().map(|v| v * v).collect())
            .collect(),
    );
    let df = (z.len() - 1) as f64;
    let e2: Vec<f64> = u.iter().map(|v| v * v).collect();
    let den = utu / n;
    let g_bp: Vec<f64> = e2.iter().map(|v| v / den - 1.0).collect();
    let g_kb: Vec<f64> = e2.iter().map(|v| v - den).collect();
    let v = g_kb.iter().map(|d| d * d).sum::<f64>() / n;
    let bp = quad_form(&z, &g_bp).map_or(f64::NAN, |q| 0.5 * q);
    let kb = quad_form(&z, &g_kb).map_or(f64::NAN, |q| q / v);
    (vec![df, bp, chi2_sf(bp, df)], vec![df, kb, chi2_sf(kb, df)])
}

fn condition_number(xtx: &Mat<f64>) -> Result<f64, String> {
    let k = xtx.nrows();
    let d: Vec<f64> = (0..k).map(|i| xtx[(i, i)]).collect();
    let scaled = Mat::from_fn(k, k, |i, j| xtx[(i, j)] / (d[i] * d[j]).sqrt());
    let ev = symmetric_eigenvalues(&scaled)?;
    Ok((ev[ev.len() - 1] / ev[0]).sqrt())
}

/// White's test; `None` when spreg would report "Not computed due to multicollinearity".
fn white(x: &[Vec<f64>], u: &[f64], cond: f64) -> Option<Vec<f64>> {
    if cond > 30.0 {
        return None;
    }
    let n = u.len();
    let k = x.len();
    let e: Vec<f64> = u.iter().map(|v| v * v).collect();
    let mut a: Vec<Vec<f64>> = x.to_vec();
    for i in 0..k {
        for j in i..k {
            a.push(x[i].iter().zip(&x[j]).map(|(p, q)| p * q).collect());
        }
    }
    let total = a.len();
    let mut omit = vec![false; total];
    for i in 0..total {
        let (lo, hi) = a[i]
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        if lo == hi {
            omit[i] = true;
        }
        for j in (i + 1)..total {
            if a[i] == a[j] {
                omit[j] = true;
            }
        }
    }
    let mut cols = vec![vec![1.0; n]];
    cols.extend(
        a.into_iter()
            .zip(omit)
            .filter_map(|(c, o)| (!o).then_some(c)),
    );
    let df = (cols.len() - 1) as f64;
    let wh = match fit(&e, &cols) {
        Ok(f) => {
            let m = mean(&e);
            let ss_tot: f64 = e.iter().map(|v| sq(v - m)).sum();
            (1.0 - f.utu / ss_tot) * n as f64
        }
        Err(_) => f64::NAN,
    };
    Some(vec![df, wh, chi2_sf(wh, df)])
}

fn vif(x: &[Vec<f64>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * x.len());
    for j in 0..x.len() {
        let y = &x[j];
        let others: Vec<Vec<f64>> = x
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != j)
            .map(|(_, c)| c.clone())
            .collect();
        let m = mean(y);
        let ss_tot: f64 = y.iter().map(|v| sq(v - m)).sum();
        if ss_tot == 0.0 {
            out.extend([f64::NAN, f64::NAN]);
            continue;
        }
        match fit(y, &others) {
            Ok(f) => {
                let tol = 1.0 - (1.0 - f.utu / ss_tot);
                out.extend([1.0 / tol, tol]);
            }
            Err(_) => out.extend([f64::NAN, f64::NAN]),
        }
    }
    out
}

/// `psi0 = xu' (G xu)` for HAC, or `xu' xu` for White.
fn meat(x: &[Vec<f64>], u: &[f64], gwk: Option<&Csr>) -> Result<Mat<f64>, String> {
    let n = u.len();
    let xu: Vec<Vec<f64>> = x
        .iter()
        .map(|c| c.iter().zip(u).map(|(a, b)| a * b).collect())
        .collect();
    let xum = mat_from_cols(&xu, n);
    let right = match gwk {
        Some(g) => {
            let lagged: Result<Vec<Vec<f64>>, String> = xu.iter().map(|c| g.lag(c)).collect();
            mat_from_cols(&lagged?, n)
        }
        None => xum.clone(),
    };
    Ok(xum.transpose() * right)
}

struct Fitted<'a> {
    x: &'a [Vec<f64>],
    y: &'a [f64],
    xtxi: Mat<f64>,
    predy: Vec<f64>,
    u: Vec<f64>,
    utu: f64,
    sig2n: f64,
    n: usize,
    k: usize,
}

fn check_graph(w: &Csr, n: usize, name: &str) -> Result<(), String> {
    if w.n != n {
        return Err(format!("{name} has {} nodes but y has {n} rows", w.n));
    }
    Ok(())
}

fn check_zero_diagonal(w: &Csr) -> Result<(), String> {
    for i in 0..w.n {
        for k in row_range(w, i) {
            if w.neighbors[k] as usize == i && w.weights[k] != 0.0 {
                return Err("graph must have a zero diagonal (no self-weights)".to_string());
            }
        }
    }
    Ok(())
}

fn check_kernel(g: &Csr) -> Result<(), String> {
    for i in 0..g.n {
        let mut diag = 0.0;
        for k in row_range(g, i) {
            let w = g.weights[k];
            if g.neighbors[k] as usize == i {
                diag = w;
            } else if !(0.0..=1.0).contains(&w) {
                return Err("gwk off-diagonal weights must lie in [0, 1]".to_string());
            }
        }
        if diag != 1.0 {
            return Err("gwk diagonal entries must all equal 1".to_string());
        }
    }
    Ok(())
}

/// LM, Koley-Bera and Moran residual diagnostics on the fitted model.
fn spatial(f: &Fitted, w: &Csr, moran: bool, out: &mut Entries) -> Result<(), String> {
    let (n, k) = (f.n, f.k);
    let nf = n as f64;
    check_zero_diagonal(w)?;
    let t = trace_wtw_ww(w);
    if t == 0.0 {
        return Err("graph has no links".to_string());
    }
    let sig2n = f.sig2n;
    let wu = w.lag(&f.u)?;
    let wy = w.lag(f.y)?;
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    let utwu_ds = dot(&f.u, &wu) / sig2n;
    let utwy_ds = dot(&f.u, &wy) / sig2n;

    let xm = mat_from_cols(f.x, n);
    let wx_cols: Vec<Vec<f64>> = f.x.iter().map(|c| w.lag(c)).collect::<Result<_, _>>()?;
    let wxm = mat_from_cols(&wx_cols, n);
    let wxb = w.lag(&f.predy)?;
    let wxb_m = col_mat(&wxb);
    let xwxb = xm.transpose() * &wxb_m;
    let num1 = dot(&wxb, &wxb) - (xwxb.transpose() * (&f.xtxi * &xwxb))[(0, 0)];
    let j = (num1 + t * sig2n) / (nf * sig2n);
    let nj = nf * j;

    let lm_error = sq(utwu_ds) / t;
    let lm_lag = sq(utwy_ds) / nj;
    let rlm_error = sq(utwu_ds - t * utwy_ds / nj) / (t * (1.0 - t / nj));
    let rlm_lag = sq(utwy_ds - utwu_ds) / (nj - t);
    let lm_sarma = sq(utwy_ds - utwu_ds) / (nf * j - t) + sq(utwu_ds) / t;
    out.insert("lm_error", pair(lm_error, chi2_sf(lm_error, 1.0)));
    out.insert("lm_lag", pair(lm_lag, chi2_sf(lm_lag, 1.0)));
    out.insert("rlm_error", pair(rlm_error, chi2_sf(rlm_error, 1.0)));
    out.insert("rlm_lag", pair(rlm_lag, chi2_sf(rlm_lag, 1.0)));
    out.insert("lm_sarma", pair(lm_sarma, chi2_sf(lm_sarma, 2.0)));

    if let Some(kb) = koley_bera(f, w, &wy, &wxb, &wx_cols, t) {
        let kf = k as f64;
        let (lm_wx, lm_spdurbin) = (kb.0, kb.1);
        let rlm_wx = lm_spdurbin - lm_lag;
        let rlm_durlag = lm_spdurbin - lm_wx;
        let lm_slxerr = lm_error + lm_wx;
        out.insert("lm_wx", pair(lm_wx, chi2_sf(lm_wx, kf - 1.0)));
        out.insert("lm_spdurbin", pair(lm_spdurbin, chi2_sf(lm_spdurbin, kf)));
        out.insert("rlm_wx", pair(rlm_wx, chi2_sf(rlm_wx, kf - 1.0)));
        out.insert("rlm_durlag", pair(rlm_durlag, chi2_sf(rlm_durlag, 1.0)));
        out.insert("lm_slxerr", pair(lm_slxerr, chi2_sf(lm_slxerr, kf)));
    }

    if moran {
        let s0 = w.s0();
        let s1 = w.s1();
        let kf = k as f64;
        let i_stat = nf * dot(&f.u, &wu) / (s0 * f.utu);
        let xwx = xm.transpose() * &wxm;
        let tr_a = trace(&(&f.xtxi * xwx));
        let e_i = -(nf * tr_a) / (s0 * (nf - kf));
        // Z = (W + W') X / 2
        let z_cols: Vec<Vec<f64>> =
            f.x.iter()
                .zip(&wx_cols)
                .map(|(c, wc)| {
                    lag_transpose(w, c)
                        .into_iter()
                        .zip(wc)
                        .map(|(a, b)| (a + b) / 2.0)
                        .collect()
                })
                .collect();
        let zm = mat_from_cols(&z_cols, n);
        let a = &f.xtxi * (xm.transpose() * &zm);
        let b = &f.xtxi * (zm.transpose() * &zm);
        let tr_a2 = trace(&(&a * &a));
        let tr_b = trace(&b) * 4.0;
        let v_i = sq(nf) / (sq(s0) * (nf - kf) * (nf - kf + 2.0))
            * (s1 + 2.0 * tr_a2 - tr_b - 2.0 * sq(tr_a) / (nf - kf));
        let z = (i_stat - e_i) / v_i.sqrt();
        out.insert("moran_res", vec![i_stat, z, norm_two_sided(z)]);
    }
    Ok(())
}

/// `(lm_wx, lm_spdurbin)`, or `None` where spreg's `try` block fails (singular matrices, no
/// regressor besides the constant).
fn koley_bera(
    f: &Fitted,
    w: &Csr,
    wy: &[f64],
    wxb: &[f64],
    wx_cols: &[Vec<f64>],
    t: f64,
) -> Option<(f64, f64)> {
    let (n, k) = (f.n, f.k);
    if k < 2 {
        return None;
    }
    let _ = w;
    let kx = k - 1;
    let sig2n = f.sig2n;
    let xm = mat_from_cols(f.x, n);
    // The first column is the constant; spreg lags only the 'x' variables.
    let wx = mat_from_cols(&wx_cols[1..], n);
    let u = col_mat(&f.u);
    let xtx = xm.transpose() * &xm;
    let wxtu = wx.transpose() * &u;
    let mx1 = wx.transpose() * &xm;
    let mx = &mx1 * (&f.xtxi * mx1.transpose());
    let xwwx = wx.transpose() * &wx;
    let xqx = Mat::from_fn(kx, kx, |i, j| xwwx[(i, j)] - mx[(i, j)]);
    let xqxi = inverse(&xqx).ok()?;
    let rsg = (wxtu.transpose() * (&xqxi * &wxtu))[(0, 0)] / sig2n;

    let wxb_m = col_mat(wxb);
    let drho = wy.iter().zip(&f.u).map(|(a, b)| a * b).sum::<f64>() / sig2n;
    let dgam: Vec<f64> = (0..kx).map(|i| wxtu[(i, 0)] / sig2n).collect();
    let jj11 = Mat::from_fn(k + 1, k + 1, |i, j| {
        if i < k && j < k {
            xtx[(i, j)]
        } else if i == k && j == k {
            nf(n) / (2.0 * sig2n)
        } else {
            0.0
        }
    });
    let x1twxb = xm.transpose() * &wxb_m;
    let x1twx = xm.transpose() * &wx;
    let jj12 = Mat::from_fn(k + 1, 1 + kx, |i, j| {
        if i == k {
            0.0
        } else if j == 0 {
            x1twxb[(i, 0)]
        } else {
            x1twx[(i, j - 1)]
        }
    });
    let wxbtwx = wxb_m.transpose() * &wx;
    let wxtwx = wx.transpose() * &wx;
    let wxb2: f64 = wxb.iter().map(|v| v * v).sum();
    let jj22 = Mat::from_fn(1 + kx, 1 + kx, |i, j| match (i, j) {
        (0, 0) => wxb2 + t * sig2n,
        (0, c) => wxbtwx[(0, c - 1)],
        (r, 0) => wxbtwx[(0, r - 1)],
        (r, c) => wxtwx[(r - 1, c - 1)],
    });
    let jj11i = inverse(&jj11).ok()?;
    let sub = jj12.transpose() * (&jj11i * &jj12);
    let schur = Mat::from_fn(1 + kx, 1 + kx, |i, j| jj22[(i, j)] - sub[(i, j)]);
    let jj22i = inverse(&schur).ok()?;
    let dd: Vec<f64> = std::iter::once(drho).chain(dgam).collect();
    let mut joint = 0.0;
    for (i, di) in dd.iter().enumerate() {
        for (j, dj) in dd.iter().enumerate() {
            joint += di * jj22i[(i, j)] * sig2n * dj;
        }
    }
    Some((rsg, joint))
}

fn nf(n: usize) -> f64 {
    n as f64
}

/// Fits OLS and computes the requested diagnostics.
pub fn ols(y: &[f64], x: &[Vec<f64>], opts: &Options) -> Result<Entries, String> {
    let n = y.len();
    let k = x.len();
    if k == 0 {
        return Err("x has no columns".to_string());
    }
    if x.iter().any(|c| c.len() != n) {
        return Err("every x column must have the length of y".to_string());
    }
    if n <= k {
        return Err(format!(
            "need more observations than columns, got n = {n}, k = {k}"
        ));
    }
    if (opts.spat_diag || opts.moran) && opts.w.is_none() {
        return Err("spat_diag and moran require a graph".to_string());
    }
    if let Some(w) = opts.w {
        check_graph(w, n, "graph")?;
    }
    if opts.robust == Robust::Hac {
        let g = opts.gwk.ok_or("robust: :hac requires gwk")?;
        check_graph(g, n, "gwk")?;
        check_kernel(g)?;
    }

    let xm = mat_from_cols(x, n);
    let xtx = xm.transpose() * &xm;
    let xtxi = inverse(&xtx).map_err(|_| "X'X is singular (collinear columns)".to_string())?;
    let xty = xm.transpose() * col_mat(y);
    let betas_m = &xtxi * xty;
    let predy = column(&(&xm * &betas_m), 0);
    let u: Vec<f64> = y.iter().zip(&predy).map(|(a, b)| a - b).collect();
    let utu: f64 = u.iter().map(|v| v * v).sum();
    let (nfl, kfl) = (n as f64, k as f64);
    let sig2n = utu / nfl;
    let sig2n_k = utu / (nfl - kfl);
    let sig2 = if opts.sig2n_k { sig2n_k } else { sig2n };
    let mean_y = mean(y);
    let std_y = (y.iter().map(|v| sq(v - mean_y)).sum::<f64>() / (nfl - 1.0)).sqrt();

    let vm = match opts.robust {
        Robust::None => Mat::from_fn(k, k, |i, j| sig2 * xtxi[(i, j)]),
        Robust::White | Robust::Hac => {
            let gwk = if opts.robust == Robust::Hac {
                opts.gwk
            } else {
                None
            };
            let mut psi0 = meat(x, &u, gwk)?;
            if opts.robust == Robust::White && opts.sig2n_k {
                psi0 = Mat::from_fn(k, k, |i, j| psi0[(i, j)] * (nfl / (nfl - kfl)));
            }
            &xtxi * (psi0 * &xtxi)
        }
    };
    let std_err: Vec<f64> = (0..k).map(|i| vm[(i, i)].sqrt()).collect();
    let df = nfl - kfl;
    let t_stat: Vec<f64> = (0..k)
        .flat_map(|i| {
            let t = betas_m[(i, 0)] / vm[(i, i)].sqrt();
            [t, t_two_sided(t, df)]
        })
        .collect();
    let ss_tot: f64 = y.iter().map(|v| sq(v - mean_y)).sum();
    let r2 = 1.0 - utu / ss_tot;
    let ar2 = 1.0 - (1.0 - r2) * (nfl - 1.0) / (nfl - kfl);

    let mut out = Entries::new();
    out.insert("betas", column(&betas_m, 0));
    out.insert("vm", row_major(&vm));
    out.insert("xtx", row_major(&xtx));
    out.insert("xtxi", row_major(&xtxi));
    out.insert("std_err", std_err);
    out.insert("t_stat", t_stat);
    out.insert("predy", predy.clone());
    out.insert("u", u.clone());
    for (name, v) in [
        ("utu", utu),
        ("sig2", sig2),
        ("sig2n", sig2n),
        ("sig2n_k", sig2n_k),
        ("mean_y", mean_y),
        ("std_y", std_y),
        ("r2", r2),
        ("ar2", ar2),
    ] {
        out.insert(name, vec![v]);
    }

    if opts.nonspat_diag {
        let cond = condition_number(&xtx)?;
        let u_reg: f64 = predy.iter().map(|p| sq(p - mean_y)).sum();
        let f = (u_reg / (kfl - 1.0)) / (utu / df);
        let logll = -0.5
            * (nfl * (2.0 * std::f64::consts::PI).ln()
                + nfl * (utu / nfl).ln()
                + utu / (utu / nfl));
        out.insert("f_stat", pair(f, f_sf(f, kfl - 1.0, df)));
        out.insert("logll", vec![logll]);
        out.insert("aic", vec![2.0 * kfl - 2.0 * logll]);
        out.insert("schwarz", vec![kfl * nfl.ln() - 2.0 * logll]);
        out.insert("sig2ml", vec![sig2n]);
        out.insert("condition_number", vec![cond]);

        let mu = |p: i32| u.iter().map(|v| v.powi(p)).sum::<f64>() / nfl;
        let (mu2, mu3, mu4) = (mu(2), mu(3), mu(4));
        let s = mu3 / mu2.powf(1.5);
        let kurt = mu4 / sq(mu2);
        let jb = nfl * (sq(s) / 6.0 + sq(kurt - 3.0) / 24.0);
        out.insert("jarque_bera", vec![2.0, jb, chi2_sf(jb, 2.0)]);

        let (bp, kb) = heteroskedasticity(x, &u, utu);
        out.insert("breusch_pagan", bp);
        out.insert("koenker_bassett", kb);

        if opts.white_test && opts.robust == Robust::None {
            match white(x, &u, cond) {
                Some(w) => {
                    out.insert("white", w);
                }
                None => {
                    out.insert("white_not_computed", vec![1.0]);
                }
            }
        }
    }
    if opts.vif {
        out.insert("vif", vif(x));
    }
    if opts.spat_diag && opts.robust != Robust::Hac {
        if let Some(w) = opts.w {
            let fitted = Fitted {
                x,
                y,
                xtxi,
                predy,
                u,
                utu,
                sig2n,
                n,
                k,
            };
            spatial(&fitted, w, opts.moran, &mut out)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts<'a>(w: Option<&'a Csr>, spat: bool) -> Options<'a> {
        Options {
            robust: Robust::None,
            sig2n_k: true,
            gwk: None,
            nonspat_diag: true,
            white_test: false,
            vif: false,
            w,
            spat_diag: spat,
            moran: spat,
        }
    }

    fn close(a: f64, e: f64) {
        assert!(
            (a - e).abs() <= 1e-9 * e.abs().max(1.0),
            "got {a}, expected {e}"
        );
    }

    fn lattice3() -> Csr {
        // 3x3 rook lattice, row-standardized (spreg fixture `ols_lattice`).
        let mut focal = Vec::new();
        let mut nbr = Vec::new();
        let mut wt = Vec::new();
        for r in 0..3i32 {
            for c in 0..3i32 {
                let nb: Vec<(i32, i32)> = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    .iter()
                    .map(|(dr, dc)| (r + dr, c + dc))
                    .filter(|(a, b)| (0..3).contains(a) && (0..3).contains(b))
                    .collect();
                for (a, b) in &nb {
                    focal.push((r * 3 + c) as u32);
                    nbr.push((a * 3 + b) as u32);
                    wt.push(1.0 / nb.len() as f64);
                }
            }
        }
        Csr::from_arrays(9, &focal, &nbr, &wt).unwrap()
    }

    const Y9: [f64; 9] = [3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 5.0];
    const X9: [f64; 9] = [1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 7.5];

    fn design() -> Vec<Vec<f64>> {
        vec![vec![1.0; 9], X9.to_vec()]
    }

    #[test]
    fn lattice_matches_spreg() {
        let w = lattice3();
        let out = ols(&Y9, &design(), &opts(Some(&w), true)).unwrap();
        close(out["betas"][0], 4.1795031055900616);
        close(out["betas"][1], -0.0037267080745345058);
        close(out["lm_error"][0], 0.7277849972213929);
        close(out["lm_lag"][0], 0.7238163352864323);
        close(out["rlm_error"][0], 2.143479769628242);
        close(out["rlm_lag"][0], 2.1395111076886066);
        close(out["lm_sarma"][0], 2.8672961049099994);
        close(out["lm_wx"][0], 2.1395111085263565);
        close(out["lm_spdurbin"][0], 2.8672961057477444);
        close(out["moran_res"][0], 0.24929149556840496);
        close(out["moran_res"][2], 0.05204558332738606);
    }

    #[test]
    fn exact_line_has_unit_r2() {
        let x: Vec<f64> = (0..8).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.0 - 0.5 * v).collect();
        let out = ols(&y, &[vec![1.0; 8], x], &opts(None, false)).unwrap();
        close(out["betas"][0], 3.0);
        close(out["betas"][1], -0.5);
        assert!((out["r2"][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn residuals_are_orthogonal_to_x() {
        let out = ols(&Y9, &design(), &opts(None, false)).unwrap();
        let u = &out["u"];
        for col in design() {
            let dot: f64 = col.iter().zip(u).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-10, "dot {dot}");
        }
    }

    #[test]
    fn white_scaling_follows_sig2n_k() {
        let mut o = opts(None, false);
        o.robust = Robust::White;
        let scaled = ols(&Y9, &design(), &o).unwrap();
        o.sig2n_k = false;
        let plain = ols(&Y9, &design(), &o).unwrap();
        for (a, b) in scaled["vm"].iter().zip(&plain["vm"]) {
            close(*a, b * 9.0 / 7.0);
        }
    }

    #[test]
    fn rejects_invalid_input() {
        let w = lattice3();
        assert!(ols(&Y9[..2], &design(), &opts(None, false)).is_err());
        assert!(ols(
            &[1.0, 2.0],
            &[vec![1.0; 2], vec![0.0, 1.0]],
            &opts(None, false)
        )
        .unwrap_err()
        .contains("more observations"));
        let dup = vec![vec![1.0; 9], X9.to_vec(), X9.to_vec()];
        assert!(ols(&Y9, &dup, &opts(None, false))
            .unwrap_err()
            .contains("singular"));
        assert!(ols(&Y9, &design(), &opts(None, true)).is_err());
        let mut o = opts(Some(&w), false);
        o.robust = Robust::Hac;
        assert!(ols(&Y9, &design(), &o).unwrap_err().contains("gwk"));
        let kernel = Csr::from_arrays(9, &[0], &[1], &[0.5]).unwrap();
        o.gwk = Some(&kernel);
        assert!(ols(&Y9, &design(), &o).unwrap_err().contains("diagonal"));
    }

    #[test]
    fn graph_without_links_is_rejected() {
        let empty = Csr::from_arrays(9, &[], &[], &[]).unwrap();
        let err = ols(&Y9, &design(), &opts(Some(&empty), true)).unwrap_err();
        assert!(err.contains("no links"), "{err}");
    }

    #[test]
    fn vif_is_nan_for_the_constant() {
        let mut o = opts(None, false);
        o.vif = true;
        let out = ols(&Y9, &design(), &o).unwrap();
        assert!(out["vif"][0].is_nan() && out["vif"][1].is_nan());
        assert!((out["vif"][2] - 1.0).abs() < 1e-12);
    }
}
