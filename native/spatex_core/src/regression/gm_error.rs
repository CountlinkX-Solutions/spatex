//! GMM spatial error models `y = X beta + u`, `u = lambda W u + e` and their spatial-lag
//! ("combo") and endogenous-regressor variants (spreg 1.9.1):
//!
//! * [`Estimator::Kp98`] - `GM_Error`, `GM_Endog_Error`, `GM_Combo` (Kelejian-Prucha 1998/1999),
//! * [`Estimator::Het`] - `GM_Error_Het`, `GM_Endog_Error_Het`, `GM_Combo_Het` (Arraiz, Drukker,
//!   Kelejian and Prucha 2010),
//! * [`Estimator::Hom`] - `GM_Error_Hom`, `GM_Endog_Error_Hom`, `GM_Combo_Hom` (Drukker, Egger and
//!   Prucha 2013).
//!
//! `lambda` minimizes a quadratic moment objective; see [`super::gm_common::optim_moments`] for
//! the exact optimizer that replaces spreg's L-BFGS-B. With [`Options::at`] the optimizer is
//! bypassed and the given lambdas are consumed in call order (the first-step estimate, then the
//! estimate of each iteration), which verifies every other formula independently of the
//! optimizer.
//!
//! With [`Options::regimes`] the Het and Hom estimators fit spreg's pooled regimes models
//! (`GM_*_Het_Regimes`, `GM_*_Hom_Regimes`): the inputs are the base columns, the regime design is
//! built here, the second stage filters the base columns and then expands them, and the
//! variances filter the regime design itself, as spreg does. The regimes classes filter with the
//! first-step lambda in every iteration, except `GM_Error_Het_Regimes` (previous lambda), and
//! never clamp the coefficient of the last endogenous column.
//!
//! Deviation: spreg's `GM_Endog_Error_Het` clamps the last endogenous coefficient to +-0.9
//! (meant for `rho` in the combo models); Spatex clamps only when that coefficient is `rho`.

use std::collections::VecDeque;

use faer::Mat;

use super::gm_common::{
    dot, filter, filter_cols, optim_moments, power_expansion_t, set_endog, sp_att, Moments,
    Optimum, Sp, INVERSE_THRESHOLD,
};
use super::linalg::{col_mat, column, fit, inverse, mat_from_cols, row_major};
use super::ml_common::pearson_r;
use super::ols::Entries;
use super::tsls::{base_tsls, inference, Instruments, Tsls, VarOptions};
use crate::graph::Csr;

/// Which GMM estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    Kp98,
    Het,
    Hom,
}

/// Moment matrix `A1` of the homoskedastic estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum A1Kind {
    /// `W'W - tr(W'W)/n I`, scaled by `1 / (1 + (tr(W'W)/n)^2)` (spreg `hom_sc`).
    HomSc,
    /// `W'W - tr(W'W)/n I`.
    Hom,
    /// `W'W - diag(W'W)`.
    Het,
}

/// Spatial-lag ("combo") settings: `W y` is added as an endogenous regressor.
#[derive(Clone, Copy, Debug)]
pub struct Combo {
    pub w_lags: usize,
    pub lag_q: bool,
    /// The first design column is the constant (excluded from the spatial instruments).
    pub skip_const: bool,
}

/// Regime design of the pooled regimes models (spreg `Regimes_Frame`): the regime-specific
/// columns of every regime in turn, then the global ones.
#[derive(Clone, Debug)]
pub struct Regimes {
    /// Position of the regime of every row.
    pub index: Vec<usize>,
    pub nr: usize,
    /// One flag per column of `x` (constant first) and of `yend`: the coefficient varies by
    /// regime. The instruments always do.
    pub x_flags: Vec<bool>,
    pub y_flags: Vec<bool>,
}

impl Regimes {
    /// The regime design of `cols`.
    pub fn expand(&self, cols: &[Vec<f64>], flags: &[bool]) -> Vec<Vec<f64>> {
        let masked = |col: &[f64], r: usize| -> Vec<f64> {
            col.iter()
                .zip(&self.index)
                .map(|(v, &i)| if i == r { *v } else { 0.0 })
                .collect()
        };
        let mut out = Vec::with_capacity(cols.len() * self.nr);
        for r in 0..self.nr {
            for (col, _) in cols.iter().zip(flags).filter(|(_, f)| **f) {
                out.push(masked(col, r));
            }
        }
        for (col, _) in cols.iter().zip(flags).filter(|(_, f)| !**f) {
            out.push(col.clone());
        }
        out
    }

    fn check(&self, n: usize, kx: usize, ky: usize) -> Result<(), String> {
        if self.index.len() != n || self.index.iter().any(|&r| r >= self.nr) {
            return Err("the regimes must give one position per observation".to_string());
        }
        if self.x_flags.len() != kx || self.y_flags.len() != ky {
            return Err("the regimes need one flag per column of x and yend".to_string());
        }
        Ok(())
    }
}

/// Options of [`gm_error`].
pub struct Options<'a> {
    pub w: &'a Csr,
    pub estimator: Estimator,
    pub max_iter: usize,
    pub epsilon: f64,
    pub step1c: bool,
    pub a1: A1Kind,
    pub hard_bound: bool,
    pub combo: Option<Combo>,
    /// Lambdas to consume instead of optimizing (test hook), in call order.
    pub at: Option<Vec<f64>>,
    /// Pooled regimes model (Het and Hom only; the `x`, `yend` and `q` given are the base
    /// columns and are expanded here). The second stage filters the base columns and then
    /// expands them, as spreg does.
    pub regimes: Option<Regimes>,
}

const LAM_ERR: &str = "Spatial parameter was outside the bounds of -0.99 and 0.99";
const LAM_ERR_HET: &str = "Spatial error parameter was outside the bounds of -0.99 and 0.99";

/// Source of lambdas: the optimizer, or a queue of given values.
struct Lam {
    at: Option<VecDeque<f64>>,
    used: Vec<f64>,
    degenerate: bool,
    hard_bound: bool,
}

impl Lam {
    fn next(
        &mut self,
        mom: &Moments,
        vc: Option<&Mat<f64>>,
        check_bound: bool,
    ) -> Result<f64, String> {
        let lam = match self.at.as_mut() {
            Some(q) => q
                .pop_front()
                .ok_or_else(|| "not enough lambdas supplied".to_string())?,
            None => {
                let o = optim_moments(mom, vc)?;
                self.degenerate |= o.degenerate;
                o.lam
            }
        };
        if check_bound && self.hard_bound && lam.abs() >= 0.99 {
            return Err(LAM_ERR.to_string());
        }
        self.used.push(lam);
        Ok(lam)
    }
}

struct Ctx<'a> {
    w: &'a Csr,
    n: usize,
    y: &'a [f64],
    x: &'a [Vec<f64>],
    yend: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
    combo: bool,
    /// The regime design and the base columns it was built from.
    reg: Option<RegCtx<'a>>,
}

struct RegCtx<'a> {
    spec: &'a Regimes,
    x0: &'a [Vec<f64>],
    y0: &'a [Vec<f64>],
}

impl Ctx<'_> {
    /// The filtered regressors of a second stage: in the regimes models the base columns are
    /// filtered and then put on the regime design.
    fn filtered_x(&self, lam: f64) -> Result<Vec<Vec<f64>>, String> {
        match &self.reg {
            Some(r) => Ok(r
                .spec
                .expand(&filter_cols(self.w, r.x0, lam)?, &r.spec.x_flags)),
            None => filter_cols(self.w, self.x, lam),
        }
    }

    fn filtered_yend(&self, lam: f64) -> Result<Vec<Vec<f64>>, String> {
        match &self.reg {
            Some(r) => Ok(r
                .spec
                .expand(&filter_cols(self.w, r.y0, lam)?, &r.spec.y_flags)),
            None => filter_cols(self.w, &self.yend, lam),
        }
    }

    /// The lambda the second stage filters with: the previous one, except in the regimes
    /// models whose spreg code reuses the first-step lambda in every iteration.
    fn stage_lambda(&self, first: f64, previous: f64) -> f64 {
        if self.reg.is_some() {
            first
        } else {
            previous
        }
    }

    fn endog(&self) -> bool {
        !self.yend.is_empty()
    }

    fn z(&self) -> Vec<Vec<f64>> {
        self.x.iter().chain(&self.yend).cloned().collect()
    }

    fn nf(&self) -> f64 {
        self.n as f64
    }
}

/// Everything the entry assembly needs from one estimator.
struct Fit {
    betas: Vec<f64>,
    vm: Mat<f64>,
    sig2: Option<f64>,
    u: Vec<f64>,
    predy: Vec<f64>,
    iteration: Option<usize>,
    /// Het combo: `|rho| >= 0.99` in the last second-stage regression.
    rho_near_bound: bool,
}

fn xb(cols: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = cols[0].len();
    (0..n)
        .map(|i| cols.iter().zip(b).map(|(c, bj)| c[i] * bj).sum())
        .collect()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(p, q)| p - q).collect()
}

fn quad(m: &Sp, a: &[f64], b: &[f64]) -> f64 {
    dot(a, &m.matvec(b))
}

fn mat2(a: f64, b: f64, c: f64, d: f64) -> Mat<f64> {
    Mat::from_fn(2, 2, |i, j| [[a, b], [c, d]][i][j])
}

// -- moment equations -----------------------------------------------------------------------

/// Kelejian-Prucha moments (`_momentsGM_Error`): three equations in `(lambda, lambda^2, sigma^2)`.
fn moments_kp98(w: &Csr, u: &[f64], n: f64) -> Result<Moments, String> {
    let wu = w.lag(u)?;
    let wwu = w.lag(&wu)?;
    let tr_wtw: f64 = w.weights.iter().map(|v| v * v).sum();
    let (u2, wu2, uwu) = (dot(u, u), dot(&wu, &wu), dot(u, &wu));
    let (uwwu, wwu2, wuwwu) = (dot(u, &wwu), dot(&wwu, &wwu), dot(&wu, &wwu));
    let rows = [
        [2.0 * uwu, -wu2, n],
        [2.0 * wuwwu, -wwu2, tr_wtw],
        [uwwu + wu2, -wuwwu, 0.0],
    ];
    Ok(Moments {
        g_mat: Mat::from_fn(3, 3, |i, j| rows[i][j] / n),
        g: vec![u2 / n, wu2 / n, uwu / n],
    })
}

/// Heteroskedastic moments (`_moments2eqs`).
fn moments_het(a1: &Sp, w: &Sp, u: &[f64], n: f64) -> Moments {
    let wu = w.matvec(u);
    let a1u = a1.matvec(u);
    let a1t = a1.transpose();
    let g11 = dot(u, &a1.matvec(&wu)) + dot(u, &a1t.matvec(&wu));
    let g12 = -quad(a1, &wu, &wu);
    let g21 = dot(u, &w.matvec(&wu)) + dot(u, &w.transpose().matvec(&wu));
    let g22 = -quad(w, &wu, &wu);
    Moments {
        g_mat: mat2(g11 / n, g12 / n, g21 / n, g22 / n),
        g: vec![dot(u, &a1u) / n, dot(u, &wu) / n],
    }
}

/// Homoskedastic moments (`moments_hom`).
fn moments_hom(w: &Sp, a1: &Sp, a2: &Sp, u: &[f64], n: f64) -> Moments {
    let wu = w.matvec(u);
    let (a1u, a2u) = (a1.matvec(u), a2.matvec(u));
    Moments {
        g_mat: mat2(
            2.0 * dot(&wu, &a1u) / n,
            -quad(a1, &wu, &wu) / n,
            2.0 * dot(&wu, &a2u) / n,
            -quad(a2, &wu, &wu) / n,
        ),
        g: vec![dot(u, &a1u) / n, dot(u, &a2u) / n],
    }
}

// -- shared small-matrix pieces ----------------------------------------------------------------

/// `G [1, 2 lambda]'`.
fn jvec(g: &Mat<f64>, lam: f64) -> [f64; 2] {
    [
        g[(0, 0)] + 2.0 * lam * g[(0, 1)],
        g[(1, 0)] + 2.0 * lam * g[(1, 1)],
    ]
}

/// `(j' psi^-1 j)^-1` for a 2-vector `j`, and `psi^-1 j`.
fn j_psi_j(j: [f64; 2], psi: &Mat<f64>) -> Result<(f64, [f64; 2], Mat<f64>), String> {
    let psii = inverse(psi)?;
    let pj = [
        psii[(0, 0)] * j[0] + psii[(0, 1)] * j[1],
        psii[(1, 0)] * j[0] + psii[(1, 1)] * j[1],
    ];
    let jpj = j[0] * pj[0] + j[1] * pj[1];
    Ok((1.0 / jpj, pj, psii))
}

fn block_diag_square(top: &Mat<f64>, corner: f64) -> Mat<f64> {
    let k = top.nrows();
    Mat::from_fn(k + 1, k + 1, |i, j| {
        if i < k && j < k {
            top[(i, j)]
        } else if i == k && j == k {
            corner
        } else {
            0.0
        }
    })
}

// -- Kelejian-Prucha ---------------------------------------------------------------------------

/// The Kelejian-Prucha estimate of lambda from the residuals `u` of a first-stage fit: the
/// moments of `u` and the optimizer of `GM_Error`, for models (regimes) whose second stage is
/// built outside [`gm_error`].
pub fn kp98_lambda(w: &Csr, u: &[f64]) -> Result<Optimum, String> {
    if w.n != u.len() {
        return Err(format!(
            "graph has {} nodes but u has {} rows",
            w.n,
            u.len()
        ));
    }
    optim_moments(&moments_kp98(w, u, u.len() as f64)?, None)
}

fn kp98(ctx: &Ctx, lam: &mut Lam) -> Result<Fit, String> {
    let (w, n) = (ctx.w, ctx.nf());
    let (u0, first) = if ctx.endog() {
        let t = base_tsls(
            ctx.y,
            ctx.x,
            &ctx.yend,
            Instruments::Q(&ctx.q),
            VarOptions::PLAIN,
        )?;
        (t.u.clone(), Some(t))
    } else {
        (fit(ctx.y, ctx.x)?.u, None)
    };
    let mom = moments_kp98(w, &u0, n)?;
    let lam1 = lam.next(&mom, None, true)?;
    let xs = filter_cols(w, ctx.x, lam1)?;
    let ys = filter(w, ctx.y, lam1)?;
    let (b, predy, sig2, vm) = match first {
        Some(t) => {
            let yend_s = filter_cols(w, &ctx.yend, lam1)?;
            let t2 = base_tsls(&ys, &xs, &yend_s, Instruments::H(&t.h), VarOptions::PLAIN)?;
            let predy = xb(&ctx.z(), &t2.betas);
            let sig2 = dot(&t2.u, &t2.u) / n;
            (t2.betas.clone(), predy, sig2, &t2.varb * sig2)
        }
        None => {
            let f = fit(&ys, &xs)?;
            let sig2 = f.utu / n;
            (f.betas.clone(), xb(ctx.x, &f.betas), sig2, &f.xtxi * sig2)
        }
    };
    let u = sub(ctx.y, &predy);
    let mut betas = b;
    betas.push(lam1);
    Ok(Fit {
        betas,
        vm,
        sig2: Some(sig2),
        u,
        predy,
        iteration: None,
        rho_near_bound: false,
    })
}

// -- heteroskedastic ---------------------------------------------------------------------------

/// `diag((u - lambda W u)^2)` (`get_psi_sigma`).
fn psi_sigma(w: &Csr, u: &[f64], lam: f64) -> Result<Vec<f64>, String> {
    Ok(filter(w, u, lam)?.into_iter().map(|v| v * v).collect())
}

/// Moment covariance together with the vectors `a1`, `a2`.
type HetVc = (Mat<f64>, Vec<f64>, Vec<f64>);

struct HetOps {
    w: Sp,
    a1: Sp,
    /// `W + W'`.
    s: Sp,
}

/// The 2x2 covariance of the moments (`get_vc_het`).
fn vc_het(ops: &HetOps, e: &[f64], n: f64) -> Mat<f64> {
    let p11 = 4.0 * ops.a1.trace_prod(&ops.a1, e);
    let p12 = 2.0 * ops.a1.trace_prod(&ops.s, e);
    let p22 = ops.s.trace_prod(&ops.s, e);
    let d = 2.0 * n;
    mat2(p11 / d, p12 / d, p12 / d, p22 / d)
}

/// OLS-based variance of `(beta, lambda)` (`get_vm_het`).
fn vm_het(
    ctx: &Ctx,
    g: &Mat<f64>,
    lam: f64,
    u: &[f64],
    psi: &Mat<f64>,
) -> Result<Mat<f64>, String> {
    let n = ctx.n;
    let j = jvec(g, lam);
    let zs = mat_from_cols(&filter_cols(ctx.w, ctx.x, lam)?, n);
    let e = psi_sigma(ctx.w, u, lam)?;
    let zte = Mat::from_fn(zs.ncols(), n, |r, i| zs[(i, r)] * e[i]);
    let zsi = inverse(&(zs.transpose() * &zs))?;
    let omega11 = &zsi * (&zte * &zs) * &zsi * ctx.nf();
    let (omega22, _, _) = j_psi_j(j, psi)?;
    let vm = block_diag_square(&omega11, omega22);
    Ok(Mat::from_fn(vm.nrows(), vm.ncols(), |i, k| {
        vm[(i, k)] / ctx.nf()
    }))
}

fn het_ols(ctx: &Ctx, ops: &HetOps, opts: &Options, lam: &mut Lam) -> Result<Fit, String> {
    let (w, n) = (ctx.w, ctx.nf());
    let ols = fit(ctx.y, ctx.x)?;
    let mom = moments_het(&ops.a1, &ops.w, &ols.u, n);
    let lam1 = lam.next(&mom, None, false)?;
    let lam2 = if opts.step1c {
        let vc1 = vc_het(ops, &psi_sigma(w, &ols.u, lam1)?, n);
        lam.next(&mom, Some(&vc1), false)?
    } else {
        lam1
    };
    let mut lam_old = lam2;
    let (mut iteration, mut eps) = (0usize, 1.0_f64);
    let mut last = None;
    while iteration < opts.max_iter && eps > opts.epsilon {
        // spreg's GM_Error_Het_Regimes filters with the previous lambda, as GM_Error_Het
        let xs = ctx.filtered_x(lam_old)?;
        let ys = filter(w, ctx.y, lam_old)?;
        let b = fit(&ys, &xs)?.betas;
        let predy = xb(ctx.x, &b);
        let u = sub(ctx.y, &predy);
        let vc_i = vc_het(ops, &psi_sigma(w, &u, lam_old)?, n);
        let mom_i = moments_het(&ops.a1, &ops.w, &u, n);
        let lam3 = lam.next(&mom_i, Some(&vc_i), false)?;
        eps = (lam3 - lam_old).abs();
        lam_old = lam3;
        iteration += 1;
        last = Some((b, predy, u, mom_i));
    }
    let (b, predy, u, mom_i) = last.ok_or("max_iter must be at least 1")?;
    let lam3 = lam_old;
    if lam.hard_bound && lam3.abs() >= 0.99 {
        return Err(LAM_ERR_HET.to_string());
    }
    let vc3 = vc_het(ops, &psi_sigma(w, &u, lam3)?, n);
    let vm = vm_het(ctx, &mom_i.g_mat, lam3, &u, &vc3)?;
    let mut betas = b;
    betas.push(lam3);
    Ok(Fit {
        betas,
        vm,
        sig2: None,
        u,
        predy,
        iteration: Some(iteration),
        rho_near_bound: false,
    })
}

/// Clamps the coefficient of the last `Z` column to +-0.9 and recomputes the residuals.
fn clamp_rho(y: &[f64], t: &mut Tsls) {
    let last = t.betas.len() - 1;
    if t.betas[last].abs() > 0.9 {
        t.betas[last] = 0.9_f64.copysign(t.betas[last]);
        t.u = sub(y, &xb(&t.z, &t.betas));
    }
}

/// `a1, a2` of the IV moment covariance (`get_a1a2`).
#[allow(clippy::too_many_arguments)]
fn a1a2(
    ctx: &Ctx,
    ops: &HetOps,
    h: &Mat<f64>,
    u: &[f64],
    lam: f64,
    p: &Mat<f64>,
    zs: &Mat<f64>,
    filt: bool,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = ctx.nf();
    let us = filter(ctx.w, u, lam)?;
    let zst = zs.transpose().to_owned();
    let alpha = |m: &Sp, c: f64| -> Mat<f64> {
        let v = m.matvec(&us);
        &zst * col_mat(&v) * (c / n)
    };
    let alpha1 = alpha(&ops.a1, -2.0);
    let alpha2 = alpha(&ops.s, -1.0);
    let hp = h * p;
    let mut a1 = column(&(&hp * &alpha1), 0);
    let mut a2 = column(&(&hp * &alpha2), 0);
    if !filt {
        a1 = power_expansion_t(ctx.w, &a1, lam, INVERSE_THRESHOLD)?;
        a2 = power_expansion_t(ctx.w, &a2, lam, INVERSE_THRESHOLD)?;
    }
    Ok((a1, a2))
}

/// Moment covariance with the first-stage estimation error (`get_vc_het_tsls`).
#[allow(clippy::too_many_arguments)]
fn vc_het_tsls(
    ctx: &Ctx,
    ops: &HetOps,
    h: &Mat<f64>,
    u: &[f64],
    lam: f64,
    p: &Mat<f64>,
    zs: &Mat<f64>,
    filt: bool,
) -> Result<HetVc, String> {
    let n = ctx.nf();
    let e = psi_sigma(ctx.w, u, lam)?;
    let vc1 = vc_het(ops, &e, n);
    let (a1, a2) = a1a2(ctx, ops, h, u, lam, p, zs, filt)?;
    let wdot = |a: &[f64], b: &[f64]| -> f64 {
        a.iter().zip(b).zip(&e).map(|((x, y), s)| x * s * y).sum()
    };
    let psi0 = mat2(
        wdot(&a1, &a1) / n,
        wdot(&a1, &a2) / n,
        wdot(&a2, &a1) / n,
        wdot(&a2, &a2) / n,
    );
    Ok((
        Mat::from_fn(2, 2, |i, j| vc1[(i, j)] + psi0[(i, j)]),
        a1,
        a2,
    ))
}

/// `P_hat` from Appendix B of the paper (`get_P_hat`).
fn p_hat(h: &Mat<f64>, hthi: &Mat<f64>, zf: &Mat<f64>, n: f64) -> Result<Mat<f64>, String> {
    let htzf = h.transpose() * zf;
    let p1 = hthi * &htzf;
    let p2 = htzf.transpose() * &p1;
    Ok(&p1 * inverse(&p2)? * n)
}

/// Variance of the GS2SLS estimator (`get_Omega_GS2SLS`).
#[allow(clippy::too_many_arguments)]
fn omega_gs2sls(
    ctx: &Ctx,
    h: &Mat<f64>,
    u: &[f64],
    lam: f64,
    g: &Mat<f64>,
    psi: &Mat<f64>,
    a1: &[f64],
    a2: &[f64],
    p: &Mat<f64>,
) -> Result<Mat<f64>, String> {
    let (n, nn) = (ctx.nf(), ctx.n);
    let e = psi_sigma(ctx.w, u, lam)?;
    let hh = h.ncols();
    let he = Mat::from_fn(hh, nn, |r, i| h[(i, r)] * e[i] / n);
    let psi_dd = &he * h;
    let a12 = Mat::from_fn(nn, 2, |i, c| if c == 0 { a1[i] } else { a2[i] });
    let psi_dl = &he * &a12;
    let size = hh + 2;
    let psi_o = Mat::from_fn(size, size, |i, j| match (i < hh, j < hh) {
        (true, true) => psi_dd[(i, j)],
        (true, false) => psi_dl[(i, j - hh)],
        (false, true) => psi_dl[(j, i - hh)],
        (false, false) => psi[(i - hh, j - hh)],
    });
    let j = jvec(g, lam);
    let (jpj_inv, pj, psii) = j_psi_j(j, psi)?;
    // om1 = (j' psi^-1 j)^-1 j' psi^-1 (1 x 2); om2 = psi^-1 j (j' psi^-1 j)^-1 (2 x 1)
    let om1 = [
        jpj_inv * (j[0] * psii[(0, 0)] + j[1] * psii[(1, 0)]),
        jpj_inv * (j[0] * psii[(0, 1)] + j[1] * psii[(1, 1)]),
    ];
    let om2 = [pj[0] * jpj_inv, pj[1] * jpj_inv];
    let k = p.ncols();
    let left = Mat::from_fn(k + 1, size, |r, c| match (r < k, c < hh) {
        (true, true) => p[(c, r)],
        (false, false) => om1[c - hh],
        _ => 0.0,
    });
    let right = Mat::from_fn(size, k + 1, |r, c| match (r < hh, c < k) {
        (true, true) => p[(r, c)],
        (false, false) => om2[r - hh],
        _ => 0.0,
    });
    let omega = &left * &psi_o * &right;
    Ok(Mat::from_fn(k + 1, k + 1, |i, j2| omega[(i, j2)] / n))
}

fn het_iv(ctx: &Ctx, ops: &HetOps, opts: &Options, lam: &mut Lam) -> Result<Fit, String> {
    let (w, n) = (ctx.w, ctx.nf());
    let mut tsls = base_tsls(
        ctx.y,
        ctx.x,
        &ctx.yend,
        Instruments::Q(&ctx.q),
        VarOptions::PLAIN,
    )?;
    if ctx.combo {
        clamp_rho(ctx.y, &mut tsls);
    }
    let h = mat_from_cols(&tsls.h, ctx.n);
    let z = ctx.z();
    let mom = moments_het(&ops.a1, &ops.w, &tsls.u, n);
    let lam1 = lam.next(&mom, None, false)?;
    let lam2 = if opts.step1c {
        let zs = mat_from_cols(&filter_cols(w, &z, lam1)?, ctx.n);
        let (vc1, _, _) = vc_het_tsls(ctx, ops, &h, &tsls.u, lam1, &tsls.pfora1a2(), &zs, false)?;
        lam.next(&mom, Some(&vc1), false)?
    } else {
        lam1
    };
    let mut lam_old = lam2;
    let (mut iteration, mut eps) = (0usize, 1.0_f64);
    let mut last = None;
    while iteration < opts.max_iter && eps > opts.epsilon {
        let lf = ctx.stage_lambda(lam1, lam_old);
        let xs = ctx.filtered_x(lf)?;
        let ys = filter(w, ctx.y, lf)?;
        let yend_s = ctx.filtered_yend(lf)?;
        let mut ts = base_tsls(
            &ys,
            &xs,
            &yend_s,
            Instruments::H(&tsls.h),
            VarOptions::PLAIN,
        )?;
        if ctx.combo {
            clamp_rho(&ys, &mut ts);
        }
        let predy = xb(&z, &ts.betas);
        let u = sub(ctx.y, &predy);
        let zs_loop = mat_from_cols(&ts.z, ctx.n);
        let (vc2, _, _) = vc_het_tsls(ctx, ops, &h, &u, lam_old, &ts.pfora1a2(), &zs_loop, true)?;
        let mom_i = moments_het(&ops.a1, &ops.w, &u, n);
        let lam3 = lam.next(&mom_i, Some(&vc2), false)?;
        eps = (lam3 - lam_old).abs();
        lam_old = lam3;
        iteration += 1;
        last = Some((ts.betas.clone(), predy, u, mom_i));
    }
    let (b, predy, u, mom_i) = last.ok_or("max_iter must be at least 1")?;
    let lam3 = lam_old;
    let rho_near = ctx.combo && b[b.len() - 1].abs() >= 0.99;
    if lam.hard_bound {
        if lam3.abs() >= 0.99 {
            return Err(LAM_ERR_HET.to_string());
        }
        if ctx.endog() && b[b.len() - 1].abs() >= 0.99 {
            return Err(
                "Spatial lag parameter was outside the bounds of -0.99 and 0.99".to_string(),
            );
        }
    }
    let zs = mat_from_cols(&filter_cols(w, &z, lam3)?, ctx.n);
    let p = p_hat(&h, &tsls.hthi, &zs, n)?;
    let (vc3, a1, a2) = vc_het_tsls(ctx, ops, &h, &u, lam3, &p, &zs, true)?;
    let vm = omega_gs2sls(ctx, &h, &u, lam3, &mom_i.g_mat, &vc3, &a1, &a2, &p)?;
    let mut betas = b;
    betas.push(lam3);
    Ok(Fit {
        betas,
        vm,
        sig2: None,
        u,
        predy,
        iteration: Some(iteration),
        rho_near_bound: rho_near,
    })
}

// -- homoskedastic -----------------------------------------------------------------------------

struct HomOps {
    w: Sp,
    a1: Sp,
    a2: Sp,
    d1: Vec<f64>,
}

fn hom_ops(w: &Csr, kind: A1Kind) -> HomOps {
    let ws = Sp::from_csr(w);
    let wt = ws.transpose();
    let wtw = wt.matmul(&ws);
    let a2 = ws.combine(0.5, &wt, 0.5);
    let n = w.n as f64;
    let a1 = match kind {
        A1Kind::Het => {
            let d = wtw.diag();
            let diag =
                Sp::from_triplets(w.n, d.iter().enumerate().map(|(i, v)| (i, i, *v)).collect());
            wtw.combine(1.0, &diag, -1.0)
        }
        A1Kind::Hom | A1Kind::HomSc => {
            let tr: f64 = wtw.diag().iter().sum();
            let num = wtw.combine(1.0, &Sp::scaled_identity(w.n, tr / n), -1.0);
            if kind == A1Kind::HomSc {
                let den = 1.0 + (tr / n) * (tr / n);
                num.scaled(1.0 / den)
            } else {
                num
            }
        }
    };
    let d1 = a1.diag();
    HomOps { w: ws, a1, a2, d1 }
}

struct HomVc {
    psi: Mat<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    /// `P`: `h x k` with instruments, `k x k` (`(xs'xs/n)^-1`) for the OLS variance.
    p: Option<Mat<f64>>,
}

/// What `get_vc_hom` needs besides the residuals.
enum HomZ<'a> {
    /// No endogenous variables, covariance for the moment weighting.
    Weight,
    /// OLS variance: `P = (xs'xs / n)^-1`.
    OlsOmega,
    /// Instruments `h` and filtered regressors `zs`.
    Iv { h: &'a Mat<f64>, zs: &'a Mat<f64> },
}

fn vc_hom(ctx: &Ctx, ops: &HomOps, u: &[f64], lam: f64, z: HomZ) -> Result<HomVc, String> {
    let n = ctx.nf();
    let us = filter(ctx.w, u, lam)?;
    let s2 = dot(&us, &us) / n;
    let mu3 = us.iter().map(|v| v.powi(3)).sum::<f64>() / n;
    let mu4 = us.iter().map(|v| v.powi(4)).sum::<f64>() / n;
    let ones = vec![1.0; ctx.n];
    let tr11 = ops.a1.trace_prod(&ops.a1, &ones);
    let tr12 = 2.0 * ops.a1.trace_prod(&ops.a2, &ones);
    let tr22 = 2.0 * ops.a2.trace_prod(&ops.a2, &ones);
    let d1d1 = dot(&ops.d1, &ops.d1);
    let mut psi11 = 2.0 * s2 * s2 * tr11 + (mu4 - 3.0 * s2 * s2) * d1d1;
    let mut psi12 = s2 * s2 * tr12;
    let mut psi22 = s2 * s2 * tr22;
    let (mut a1, mut a2) = (vec![0.0; ctx.n], vec![0.0; ctx.n]);
    let mut p = None;
    match z {
        HomZ::Weight => {}
        HomZ::OlsOmega => {
            let xs = mat_from_cols(&filter_cols(ctx.w, ctx.x, lam)?, ctx.n);
            p = Some(inverse(&(xs.transpose() * &xs * (1.0 / n)))?);
        }
        HomZ::Iv { h, zs } => {
            let zst = zs.transpose().to_owned();
            let alpha = |m: &Sp| -> Mat<f64> { &zst * col_mat(&m.matvec(&us)) * (-2.0 / n) };
            let (alpha1, alpha2) = (alpha(&ops.a1), alpha(&ops.a2));
            let hthn = inverse(&(h.transpose() * h * (1.0 / n)))?;
            let htzsn = h.transpose() * zs * (1.0 / n);
            let p0 = &hthn * &htzsn;
            let pm = &p0 * inverse(&(htzsn.transpose() * &p0))?;
            let hp = h * &pm;
            a1 = column(&(&hp * &alpha1), 0);
            a2 = column(&(&hp * &alpha2), 0);
            psi11 += s2 * dot(&a1, &a1) + 2.0 * mu3 * dot(&a1, &ops.d1);
            psi12 += s2 * dot(&a1, &a2) + mu3 * dot(&a2, &ops.d1);
            psi22 += s2 * dot(&a2, &a2);
            p = Some(pm);
        }
    }
    Ok(HomVc {
        psi: mat2(psi11 / n, psi12 / n, psi12 / n, psi22 / n),
        a1,
        a2,
        p,
    })
}

/// `p' psiDL psi^-1 j oLL` with `psiDL` given as `rows x 2`.
fn o_dl(p: &Mat<f64>, psi_dl: &Mat<f64>, pj: [f64; 2], oll: f64) -> Mat<f64> {
    let t = p.transpose() * psi_dl;
    Mat::from_fn(t.nrows(), 1, |i, _| {
        (t[(i, 0)] * pj[0] + t[(i, 1)] * pj[1]) * oll
    })
}

fn assemble_hom(odd: &Mat<f64>, odl: &Mat<f64>, oll: f64) -> Mat<f64> {
    let k = odd.nrows();
    Mat::from_fn(k + 1, k + 1, |i, j| match (i < k, j < k) {
        (true, true) => odd[(i, j)],
        (true, false) => odl[(i, 0)],
        (false, true) => odl[(j, 0)],
        (false, false) => oll,
    })
}

fn hom_ols(ctx: &Ctx, ops: &HomOps, opts: &Options, lam: &mut Lam) -> Result<Fit, String> {
    let (w, n) = (ctx.w, ctx.nf());
    let ols = fit(ctx.y, ctx.x)?;
    let mom = moments_hom(&ops.w, &ops.a1, &ops.a2, &ols.u, n);
    let lam1 = lam.next(&mom, None, true)?;
    let mut lam_old = lam1;
    let (mut iteration, mut eps) = (0usize, 1.0_f64);
    let mut last = None;
    while iteration < opts.max_iter && eps > opts.epsilon {
        let lf = ctx.stage_lambda(lam1, lam_old);
        let xs = ctx.filtered_x(lf)?;
        let ys = filter(w, ctx.y, lf)?;
        let b = fit(&ys, &xs)?.betas;
        let predy = xb(ctx.x, &b);
        let u = sub(ctx.y, &predy);
        let mom_i = moments_hom(&ops.w, &ops.a1, &ops.a2, &u, n);
        let psi = vc_hom(ctx, ops, &u, lam_old, HomZ::Weight)?.psi;
        let lam2 = lam.next(&mom_i, Some(&psi), true)?;
        eps = (lam2 - lam_old).abs();
        lam_old = lam2;
        iteration += 1;
        last = Some((b, predy, u, mom_i));
    }
    let (b, predy, u, mom_i) = last.ok_or("max_iter must be at least 1")?;
    let lam2 = lam_old;
    // omega_hom_ols
    let xs = mat_from_cols(&filter_cols(w, ctx.x, lam2)?, ctx.n);
    let us = filter(w, &u, lam2)?;
    let s2 = dot(&us, &us) / n;
    let vc = vc_hom(ctx, ops, &u, lam2, HomZ::OlsOmega)?;
    let j = jvec(&mom_i.g_mat, lam2);
    let (oll_inv, pj, _) = j_psi_j(j, &vc.psi)?;
    let oll = oll_inv / n;
    let odd = inverse(&(xs.transpose() * &xs))? * s2;
    let mu3 = us.iter().map(|v| v.powi(3)).sum::<f64>() / n;
    let xm = mat_from_cols(ctx.x, ctx.n);
    let psi_dl = Mat::from_fn(xm.ncols(), 2, |r, c| {
        if c == 0 {
            mu3 * (0..ctx.n).map(|i| xm[(i, r)] * ops.d1[i]).sum::<f64>() / n
        } else {
            0.0
        }
    });
    let odl = o_dl(vc.p.as_ref().expect("OLS P"), &psi_dl, pj, oll);
    let mut betas = b;
    betas.push(lam2);
    Ok(Fit {
        betas,
        vm: assemble_hom(&odd, &odl, oll),
        sig2: Some(s2),
        u,
        predy,
        iteration: Some(iteration),
        rho_near_bound: false,
    })
}

fn hom_iv(ctx: &Ctx, ops: &HomOps, opts: &Options, lam: &mut Lam) -> Result<Fit, String> {
    let (w, n) = (ctx.w, ctx.nf());
    let tsls = base_tsls(
        ctx.y,
        ctx.x,
        &ctx.yend,
        Instruments::Q(&ctx.q),
        VarOptions::PLAIN,
    )?;
    let h = mat_from_cols(&tsls.h, ctx.n);
    let z = ctx.z();
    let mom = moments_hom(&ops.w, &ops.a1, &ops.a2, &tsls.u, n);
    let lam1 = lam.next(&mom, None, true)?;
    let mut lam_old = lam1;
    let (mut iteration, mut eps) = (0usize, 1.0_f64);
    let mut last = None;
    while iteration < opts.max_iter && eps > opts.epsilon {
        let lf = ctx.stage_lambda(lam1, lam_old);
        let xs = ctx.filtered_x(lf)?;
        let ys = filter(w, ctx.y, lf)?;
        let yend_s = ctx.filtered_yend(lf)?;
        let ts = base_tsls(
            &ys,
            &xs,
            &yend_s,
            Instruments::H(&tsls.h),
            VarOptions::PLAIN,
        )?;
        let predy = xb(&z, &ts.betas);
        let u = sub(ctx.y, &predy);
        let mom_i = moments_hom(&ops.w, &ops.a1, &ops.a2, &u, n);
        let zs_loop = mat_from_cols(&ts.z, ctx.n);
        let psi = vc_hom(
            ctx,
            ops,
            &u,
            lam_old,
            HomZ::Iv {
                h: &h,
                zs: &zs_loop,
            },
        )?
        .psi;
        let lam2 = lam.next(&mom_i, Some(&psi), true)?;
        eps = (lam2 - lam_old).abs();
        lam_old = lam2;
        iteration += 1;
        last = Some((ts.betas.clone(), predy, u, mom_i));
    }
    let (b, predy, u, mom_i) = last.ok_or("max_iter must be at least 1")?;
    let lam2 = lam_old;
    // omega_hom
    let zs = mat_from_cols(&filter_cols(w, &z, lam2)?, ctx.n);
    let us = filter(w, &u, lam2)?;
    let s2 = dot(&us, &us) / n;
    let mu3 = us.iter().map(|v| v.powi(3)).sum::<f64>() / n;
    let vc = vc_hom(ctx, ops, &u, lam2, HomZ::Iv { h: &h, zs: &zs })?;
    let j = jvec(&mom_i.g_mat, lam2);
    let (oll_inv, pj, _) = j_psi_j(j, &vc.psi)?;
    let oll = oll_inv / n;
    let hth_i_htz = &tsls.hthi * (h.transpose() * &zs);
    let odd = inverse(&(zs.transpose() * &h * &hth_i_htz))? * s2;
    let hh = h.ncols();
    let psi_dl = Mat::from_fn(hh, 2, |r, c| {
        let ht = |v: &dyn Fn(usize) -> f64| -> f64 { (0..ctx.n).map(|i| h[(i, r)] * v(i)).sum() };
        let (d_part, a_part) = if c == 0 {
            (ht(&|i| ops.d1[i]), ht(&|i| vc.a1[i]))
        } else {
            (0.0, ht(&|i| vc.a2[i]))
        };
        (mu3 * d_part + s2 * a_part) / n
    });
    let odl = o_dl(vc.p.as_ref().expect("IV P"), &psi_dl, pj, oll);
    let mut betas = b;
    betas.push(lam2);
    Ok(Fit {
        betas,
        vm: assemble_hom(&odd, &odl, oll),
        sig2: Some(s2),
        u,
        predy,
        iteration: Some(iteration),
        rho_near_bound: false,
    })
}

// -- entry point -------------------------------------------------------------------------------

/// Fits a GMM spatial error model. `x` includes the constant; `yend` / `q` are the user's extra
/// endogenous variables and instruments (with `combo`, `W y` and its instruments are added).
pub fn gm_error(
    y: &[f64],
    x: &[Vec<f64>],
    yend: &[Vec<f64>],
    q: &[Vec<f64>],
    opts: &Options,
) -> Result<Entries, String> {
    let n = y.len();
    let w = opts.w;
    if w.n != n {
        return Err(format!("graph has {} nodes but y has {n} rows", w.n));
    }
    if yend.is_empty() && !q.is_empty() {
        return Err("q needs a matching yend".to_string());
    }
    if opts.max_iter == 0 && opts.estimator != Estimator::Kp98 {
        return Err("max_iter must be at least 1".to_string());
    }
    if let Some(r) = &opts.regimes {
        if opts.estimator == Estimator::Kp98 || opts.combo.is_some() {
            return Err("regimes are supported for the Het and Hom estimators only".to_string());
        }
        r.check(n, x.len(), yend.len())?;
    }
    let (yend2, q2) = match opts.combo {
        Some(c) => {
            let x_nc = if c.skip_const && !x.is_empty() {
                &x[1..]
            } else {
                x
            };
            set_endog(w, y, x_nc, yend, q, c.w_lags, c.lag_q)?
        }
        None => (yend.to_vec(), q.to_vec()),
    };
    let (design_x, design_yend, design_q);
    let ctx = match &opts.regimes {
        Some(r) => {
            design_x = r.expand(x, &r.x_flags);
            design_yend = r.expand(&yend2, &r.y_flags);
            design_q = r.expand(&q2, &vec![true; q2.len()]);
            Ctx {
                w,
                n,
                y,
                x: &design_x,
                yend: design_yend,
                q: design_q,
                combo: false,
                reg: Some(RegCtx {
                    spec: r,
                    x0: x,
                    y0: &yend2,
                }),
            }
        }
        None => Ctx {
            w,
            n,
            y,
            x,
            yend: yend2,
            q: q2,
            combo: opts.combo.is_some(),
            reg: None,
        },
    };
    let mut lam = Lam {
        at: opts.at.clone().map(VecDeque::from),
        used: Vec::new(),
        degenerate: false,
        hard_bound: opts.hard_bound,
    };
    let f = match (opts.estimator, ctx.endog()) {
        (Estimator::Kp98, _) => kp98(&ctx, &mut lam)?,
        (Estimator::Het, endog) => {
            let ws = Sp::from_csr(w);
            let wt = ws.transpose();
            let a1 = {
                let wtw = wt.matmul(&ws);
                let d = wtw.diag();
                let diag =
                    Sp::from_triplets(n, d.iter().enumerate().map(|(i, v)| (i, i, *v)).collect());
                wtw.combine(1.0, &diag, -1.0)
            };
            let s = ws.combine(1.0, &wt, 1.0);
            let ops = HetOps { w: ws, a1, s };
            if endog {
                het_iv(&ctx, &ops, opts, &mut lam)?
            } else {
                het_ols(&ctx, &ops, opts, &mut lam)?
            }
        }
        (Estimator::Hom, endog) => {
            let ops = hom_ops(w, opts.a1);
            if endog {
                hom_iv(&ctx, &ops, opts, &mut lam)?
            } else {
                hom_ols(&ctx, &ops, opts, &mut lam)?
            }
        }
    };
    finish(&ctx, opts, f, lam)
}

fn finish(ctx: &Ctx, opts: &Options, f: Fit, lam: Lam) -> Result<Entries, String> {
    let w = ctx.w;
    let kb = f.betas.len();
    let lam_hat = f.betas[kb - 1];
    let wu = w.lag(&f.u)?;
    let e_filtered: Vec<f64> = f.u.iter().zip(&wu).map(|(a, b)| a - lam_hat * b).collect();
    let (std_err, z_stat) = inference(&f.betas[..f.vm.nrows()], &f.vm);
    let mut out = Entries::new();
    out.insert("lam", vec![lam_hat]);
    out.insert("pr2", vec![pearson_r(ctx.y, &f.predy).powi(2)]);
    if lam_hat.abs() >= 0.99 {
        out.insert("warn_lam", vec![1.0]);
    }
    if lam.degenerate {
        out.insert("warn_degenerate", vec![1.0]);
    }
    if f.rho_near_bound {
        out.insert("warn_rho_near", vec![1.0]);
    }
    if ctx.combo {
        let rho = f.betas[kb - 2];
        let wy = ctx.yend.last().expect("Wy is the last endogenous column");
        let att = sp_att(w, ctx.y, &f.predy, wy, rho, opts.hard_bound)?;
        out.insert("rho", vec![rho]);
        if att.out_of_bounds {
            out.insert("warn_rho", vec![1.0]);
        } else {
            out.insert("pr2_e", vec![pearson_r(ctx.y, &att.predy_e).powi(2)]);
        }
        out.insert("predy_e", att.predy_e);
        out.insert("e_pred", att.e_pred);
    }
    if let Some(s) = f.sig2 {
        out.insert("sig2", vec![s]);
    }
    if let Some(i) = f.iteration {
        out.insert("iteration", vec![i as f64]);
        out.insert(
            "iter_stop_max",
            vec![f64::from(u8::from(i == opts.max_iter))],
        );
    }
    out.insert("vm", row_major(&f.vm));
    out.insert("std_err", std_err);
    out.insert("z_stat", z_stat);
    out.insert("lambdas", lam.used);
    out.insert("u", f.u);
    out.insert("predy", f.predy);
    out.insert("e_filtered", e_filtered);
    out.insert("betas", f.betas);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::linalg::fit as ols_fit;
    use super::super::ml_lag::tests::{data, lattice};
    use super::*;

    fn opts(w: &Csr, estimator: Estimator) -> Options<'_> {
        Options {
            w,
            estimator,
            max_iter: 1,
            epsilon: 1e-5,
            step1c: false,
            a1: A1Kind::HomSc,
            hard_bound: false,
            combo: None,
            at: None,
            regimes: None,
        }
    }

    fn endog(x: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let yend = vec![x[1]
            .iter()
            .zip(&x[2])
            .map(|(a, b)| a - 0.5 * b + 0.1 * a * a)
            .collect()];
        let q = vec![x[2].iter().map(|v| v * v).collect()];
        (yend, q)
    }

    #[test]
    fn every_estimator_reports_a_consistent_fit() {
        let w = lattice(6);
        let (y, x) = data(36);
        for est in [Estimator::Kp98, Estimator::Het, Estimator::Hom] {
            let e = gm_error(&y, &x, &[], &[], &opts(&w, est)).unwrap();
            let lam = e["lam"][0];
            assert!((-0.99..=0.99).contains(&lam), "{est:?}");
            assert_eq!(e["betas"].len(), 4);
            assert_eq!(e["betas"][3], lam);
            let side = if est == Estimator::Kp98 { 3 } else { 4 };
            assert_eq!(e["vm"].len(), side * side, "{est:?}");
            assert_eq!(e["std_err"].len(), side);
            assert_eq!(e["z_stat"].len(), 2 * side);
            assert_eq!(e.contains_key("sig2"), est != Estimator::Het);
            assert_eq!(e.contains_key("iteration"), est != Estimator::Kp98);
            for i in 0..36 {
                assert!((e["predy"][i] - (y[i] - e["u"][i])).abs() < 1e-10);
            }
            // vm is symmetric
            for i in 0..side {
                for j in 0..side {
                    let (a, b) = (e["vm"][i * side + j], e["vm"][j * side + i]);
                    assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{est:?} {i}{j}");
                }
            }
        }
    }

    #[test]
    fn lambda_zero_is_ols_for_every_estimator() {
        let w = lattice(6);
        let (y, x) = data(36);
        let ols = ols_fit(&y, &x).unwrap();
        for (est, at) in [
            (Estimator::Kp98, vec![0.0]),
            (Estimator::Het, vec![0.0, 0.0]),
            (Estimator::Hom, vec![0.0, 0.0]),
        ] {
            let mut o = opts(&w, est);
            o.at = Some(at);
            let e = gm_error(&y, &x, &[], &[], &o).unwrap();
            for (b, o) in e["betas"].iter().zip(&ols.betas) {
                assert!((b - o).abs() < 1e-9, "{est:?}");
            }
        }
    }

    #[test]
    fn kp98_lambda_is_the_estimate_of_the_ols_residuals() {
        let w = lattice(6);
        let (y, x) = data(36);
        let e = gm_error(&y, &x, &[], &[], &opts(&w, Estimator::Kp98)).unwrap();
        let u = ols_fit(&y, &x).unwrap().u;
        let o = kp98_lambda(&w, &u).unwrap();
        assert_eq!(o.lam, e["lam"][0]);
        assert!(!o.degenerate);
        assert!(kp98_lambda(&w, &u[1..]).is_err());
    }

    #[test]
    fn at_hook_consumes_lambdas_in_order() {
        let w = lattice(6);
        let (y, x) = data(36);
        let mut o = opts(&w, Estimator::Het);
        o.step1c = true;
        o.max_iter = 2;
        o.at = Some(vec![0.1, 0.2, 0.3, 0.4]);
        let e = gm_error(&y, &x, &[], &[], &o).unwrap();
        assert_eq!(e["lambdas"], vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(e["lam"][0], 0.4);
        assert_eq!(e["iteration"][0], 2.0);
        assert_eq!(e["iter_stop_max"][0], 1.0);
        o.at = Some(vec![0.1]);
        assert!(gm_error(&y, &x, &[], &[], &o)
            .unwrap_err()
            .contains("not enough lambdas"));
    }

    #[test]
    fn convergence_stops_before_max_iter() {
        let w = lattice(6);
        let (y, x) = data(36);
        let mut o = opts(&w, Estimator::Hom);
        o.max_iter = 50;
        o.epsilon = 1e-3;
        let e = gm_error(&y, &x, &[], &[], &o).unwrap();
        assert!(e["iteration"][0] < 50.0);
        assert_eq!(e["iter_stop_max"][0], 0.0);
        o.max_iter = 0;
        assert!(gm_error(&y, &x, &[], &[], &o).is_err());
    }

    #[test]
    fn hom_a1_variants_differ_in_the_weighting_only() {
        let w = lattice(6);
        let (y, x) = data(36);
        let run = |kind| {
            let mut o = opts(&w, Estimator::Hom);
            o.a1 = kind;
            gm_error(&y, &x, &[], &[], &o).unwrap()
        };
        let (a, b, c) = (run(A1Kind::HomSc), run(A1Kind::Hom), run(A1Kind::Het));
        assert_ne!(a["lam"][0], b["lam"][0]);
        assert_ne!(b["lam"][0], c["lam"][0]);
    }

    #[test]
    fn endogenous_and_combo_variants() {
        let w = lattice(6);
        let (y, x) = data(36);
        let (yend, q) = endog(&x);
        for est in [Estimator::Kp98, Estimator::Het, Estimator::Hom] {
            let e = gm_error(&y, &x, &yend, &q, &opts(&w, est)).unwrap();
            assert_eq!(e["betas"].len(), 5, "{est:?}");
            assert!(!e.contains_key("rho"));
            let mut o = opts(&w, est);
            o.combo = Some(Combo {
                w_lags: 1,
                lag_q: true,
                skip_const: true,
            });
            let c = gm_error(&y, &x, &[], &[], &o).unwrap();
            assert_eq!(c["betas"].len(), 5);
            assert_eq!(c["rho"][0], c["betas"][3]);
            assert_eq!(c["predy_e"].len(), 36);
            assert!(c.contains_key("pr2_e"));
            let c2 = gm_error(&y, &x, &yend, &q, &o).unwrap();
            assert_eq!(c2["betas"].len(), 6);
        }
        assert!(gm_error(&y, &x, &[], &q, &opts(&w, Estimator::Kp98))
            .unwrap_err()
            .contains("q needs"));
    }

    #[test]
    fn combo_with_lambda_zero_is_gm_lag() {
        use super::super::gm_lag::{gm_lag, Options as LagOptions};
        use super::super::ml_common::Impacts;
        use super::super::ols::Robust;
        let w = lattice(6);
        let (y, x) = data(36);
        let mut o = opts(&w, Estimator::Kp98);
        o.at = Some(vec![0.0]);
        o.combo = Some(Combo {
            w_lags: 1,
            lag_q: true,
            skip_const: true,
        });
        let c = gm_error(&y, &x, &[], &[], &o).unwrap();
        let l = gm_lag(
            &y,
            &x,
            &[],
            &[],
            &LagOptions {
                w: &w,
                w_lags: 1,
                lag_q: true,
                robust: Robust::None,
                gwk: None,
                sig2n_k: false,
                spat_diag: false,
                impacts: Impacts::None,
                hard_bound: false,
                skip_const: true,
                slx_lags: 0,
                kx_base: 0,
            },
        )
        .unwrap();
        for j in 0..4 {
            assert!((c["betas"][j] - l["betas"][j]).abs() < 1e-9);
        }
        assert!((c["pr2_e"][0] - l["pr2_e"][0]).abs() < 1e-9);
    }

    #[test]
    fn hard_bound_raises_and_otherwise_warns() {
        let w = lattice(6);
        let (y, x) = data(36);
        for est in [Estimator::Kp98, Estimator::Het, Estimator::Hom] {
            let mut o = opts(&w, est);
            o.at = Some(vec![0.995; 4]);
            let e = gm_error(&y, &x, &[], &[], &o).unwrap();
            assert!(e.contains_key("warn_lam"), "{est:?}");
            o.hard_bound = true;
            assert!(gm_error(&y, &x, &[], &[], &o)
                .unwrap_err()
                .contains("outside the bounds"));
        }
    }

    fn spec(n: usize, nr: usize, x_flags: &[bool], y_flags: &[bool]) -> Regimes {
        Regimes {
            index: (0..n).map(|i| i * nr / n).collect(),
            nr,
            x_flags: x_flags.to_vec(),
            y_flags: y_flags.to_vec(),
        }
    }

    fn assert_same(a: &Entries, b: &Entries, what: &str) {
        for key in ["betas", "vm", "lam", "u", "predy", "lambdas"] {
            assert_eq!(a[key].len(), b[key].len(), "{what} {key}");
            for (p, q) in a[key].iter().zip(&b[key]) {
                assert!((p - q).abs() <= 1e-9 * q.abs().max(1.0), "{what} {key}");
            }
        }
    }

    #[test]
    fn regime_design_lists_the_regime_columns_of_every_regime_then_the_globals() {
        let r = spec(4, 2, &[true, false, true], &[]);
        let cols = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 1.0, 2.0, 3.0],
        ];
        let d = r.expand(&cols, &r.x_flags);
        assert_eq!(d.len(), 5);
        assert_eq!(d[0], vec![1.0, 2.0, 0.0, 0.0]);
        assert_eq!(d[1], vec![9.0, 1.0, 0.0, 0.0]);
        assert_eq!(d[2], vec![0.0, 0.0, 3.0, 4.0]);
        assert_eq!(d[3], vec![0.0, 0.0, 2.0, 3.0]);
        assert_eq!(d[4], vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn a_single_regime_or_only_global_columns_is_the_base_model() {
        let w = lattice(6);
        let (y, x) = data(36);
        let (yend, q) = endog(&x);
        for est in [Estimator::Het, Estimator::Hom] {
            for with_endog in [false, true] {
                let (ye, qe) = if with_endog {
                    (yend.clone(), q.clone())
                } else {
                    (vec![], vec![])
                };
                let base = gm_error(&y, &x, &ye, &qe, &opts(&w, est)).unwrap();
                // the instruments always vary by regime: global columns only match without them
                let designs: &[(usize, bool)] = if with_endog {
                    &[(1, true)]
                } else {
                    &[(1, true), (3, false)]
                };
                for &(nr, flag) in designs {
                    let mut o = opts(&w, est);
                    o.regimes = Some(spec(36, nr, &vec![flag; x.len()], &vec![flag; ye.len()]));
                    let got = gm_error(&y, &x, &ye, &qe, &o).unwrap();
                    assert_same(&got, &base, &format!("{est:?} endog {with_endog} nr {nr}"));
                }
            }
        }
    }

    #[test]
    fn regime_fits_have_a_coefficient_block_per_regime() {
        let w = lattice(6);
        let (y, x) = data(36);
        let (yend, q) = endog(&x);
        for est in [Estimator::Het, Estimator::Hom] {
            let mut o = opts(&w, est);
            o.regimes = Some(spec(36, 2, &[true, false, true], &[true]));
            let e = gm_error(&y, &x, &yend, &q, &o).unwrap();
            // 2 varying x columns and the endogenous one in two regimes, one global, lambda
            assert_eq!(e["betas"].len(), 2 * 3 + 1 + 1, "{est:?}");
            assert_eq!(e["vm"].len(), 8 * 8);
            assert_eq!(e["std_err"].len(), 8);
            for i in 0..36 {
                assert!((e["predy"][i] - (y[i] - e["u"][i])).abs() < 1e-10);
            }
        }
        let mut o = opts(&w, Estimator::Kp98);
        o.regimes = Some(spec(36, 2, &[true; 3], &[]));
        assert!(gm_error(&y, &x, &[], &[], &o)
            .unwrap_err()
            .contains("Het and Hom"));
        let mut o = opts(&w, Estimator::Het);
        o.regimes = Some(spec(36, 2, &[true; 2], &[]));
        assert!(gm_error(&y, &x, &[], &[], &o)
            .unwrap_err()
            .contains("one flag per column"));
    }

    // spreg's regimes models filter the second-stage regressors with the first-step lambda in
    // every iteration, except GM_Error_Het_Regimes, which uses the previous lambda as the base
    // models do.
    #[test]
    fn iterations_filter_with_the_first_step_lambda_except_in_error_het() {
        let w = lattice(6);
        let (y, x) = data(36);
        let (yend, q) = endog(&x);
        let cases = [
            (Estimator::Hom, false, false),
            (Estimator::Hom, true, false),
            (Estimator::Het, true, false),
            (Estimator::Het, false, true),
        ];
        for (est, with_endog, uses_previous) in cases {
            let (ye, qe) = if with_endog {
                (yend.clone(), q.clone())
            } else {
                (vec![], vec![])
            };
            let r = spec(36, 1, &vec![true; x.len()], &vec![true; ye.len()]);
            let mut one = opts(&w, est);
            one.at = Some(vec![0.1, 0.2]);
            let base = gm_error(&y, &x, &ye, &qe, &one).unwrap();
            let mut two = opts(&w, est);
            two.max_iter = 2;
            two.epsilon = 0.0;
            two.at = Some(vec![0.1, 0.2, 0.3]);
            let plain = gm_error(&y, &x, &ye, &qe, &two).unwrap();
            two.regimes = Some(r);
            two.at = Some(vec![0.1, 0.2, 0.3]);
            let reg = gm_error(&y, &x, &ye, &qe, &two).unwrap();
            assert_eq!(reg["iteration"][0], 2.0);
            let same = |a: &Entries, b: &Entries| {
                a["betas"][..x.len()]
                    .iter()
                    .zip(&b["betas"])
                    .all(|(p, q)| (p - q).abs() < 1e-9)
            };
            if uses_previous {
                assert!(same(&reg, &plain), "{est:?} {with_endog}");
                assert!(!same(&reg, &base), "{est:?} {with_endog}");
            } else {
                assert!(same(&reg, &base), "{est:?} {with_endog}");
                assert!(!same(&reg, &plain), "{est:?} {with_endog}");
            }
        }
    }
}
