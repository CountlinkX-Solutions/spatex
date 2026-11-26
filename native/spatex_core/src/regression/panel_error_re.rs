//! The random-effects spatial error panel (`spreg.ML_ErrorRE`, spreg 1.9.1 `panel.py`).
//!
//! `y = X b + mu_i + u`, `u = lambda W u + e`, data stacked by period (`y[s * n + i]` is area `i`
//! in period `s`), `x` including the constant. spreg maximizes a concentrated log-likelihood over
//! `(lambda, phi)` (`phi` the square root of the variance ratio `sigma_u^2 / sigma^2` it reports)
//! with L-BFGS-B, using the eigen-decomposition `W = V diag(w) V^-1` of `numpy.linalg.eig`:
//!
//! * `c_j = t phi^2 + 1 / (1 - lambda w_j)^2`, `P = diag(c^-1/2) V'`, and the transformed data
//!   `ys = y + (P - B) ybar - lambda W y` with `B = I - lambda W` and `ybar` the area means;
//! * objective `nt/2 ln(e'e / nt) + 1/2 sum ln(1 + t phi^2 (1 - lambda w_j)^2) - t sum ln|1 - lambda w_j|`.
//!
//! Because the within part `(I_t (x) B) ytilde` and the between part `(I_t (x) P) ybar` are
//! orthogonal, the objective only depends on `P'P = V diag(1/c) V'`, which is invariant to
//! eigenvector signs. For a symmetric `W` with a simple spectrum it equals the exact likelihood and
//! `numpy.linalg.eig` returns an orthonormal basis, so the symmetric eigensolver reproduces spreg.
//! With repeated eigenvalues `numpy.linalg.eig` returns a non-orthogonal basis, the objective is
//! no longer the likelihood and depends on LAPACK's choice: for a symmetric `W` this port uses the
//! orthonormal basis (the exact likelihood), for a non-symmetric `W` it refuses. Complex spectra
//! are refused (spreg's complex arithmetic leaves the real line).
//!
//! The final estimates (betas, residuals, variances) use the exact GLS transform
//! `P = Omega^-1/2`, `Omega = t phi^2 I + (B'B)^-1`, as spreg does, so they only depend on the
//! optimum. This port replaces L-BFGS-B by a nested bounded search (`phi` on a log scale inside
//! `lambda`), which finds the optimum to about `1e-10` where spreg stops at `ftol = 1e-9`.

use std::cell::Cell;

use faer::{Mat, Side};

use super::brent::minimize_bounded;
use super::linalg::{inverse, row_major};
use super::ml_common::dense;
use super::ols::Entries;
use super::panel::{coefficients, dot, fitted, lag_nt, ln2pi, z_stats};
use crate::graph::Csr;

/// `lambda` bounds of spreg's search.
const LAM_BOUNDS: (f64, f64) = (-0.999, 0.999);
/// Bounds of spreg's `phi` parameter (`phi^2` is the reported ratio).
const PHI_BOUNDS: (f64, f64) = (1.0e-8, 1.0e9);
/// Absolute tolerance of the nested bounded searches.
const TOL: f64 = 1.0e-10;
/// Eigenvalues closer than this are repeated.
const REPEATED: f64 = 1.0e-8;

/// Eigenvalues and eigenvectors of `W` as the objective uses them.
struct Spectral {
    evals: Vec<f64>,
    vecs: Mat<f64>,
}

fn is_symmetric(w: &Mat<f64>) -> bool {
    let scale = (0..w.nrows())
        .flat_map(|i| (0..w.ncols()).map(move |j| (i, j)))
        .map(|ij| w[ij].abs())
        .fold(1.0, f64::max);
    (0..w.nrows()).all(|i| (0..i).all(|j| (w[(i, j)] - w[(j, i)]).abs() <= 1e-12 * scale))
}

impl Spectral {
    /// Orthonormal eigenvectors of a symmetric `W`.
    fn symmetric(w: &Mat<f64>) -> Result<Spectral, String> {
        let e = w
            .self_adjoint_eigen(Side::Lower)
            .map_err(|e| format!("eigenvalue decomposition failed: {e:?}"))?;
        let evals = (0..w.nrows()).map(|i| e.S().column_vector()[i]).collect();
        Ok(Spectral {
            evals,
            vecs: e.U().to_owned(),
        })
    }

    /// Unit-norm real eigenvectors of a general `W` (`numpy.linalg.eig`'s normalization, up to
    /// sign). Complex or repeated eigenvalues are refused.
    fn general(w: &Mat<f64>) -> Result<Spectral, String> {
        let n = w.nrows();
        let e = w
            .eigen()
            .map_err(|e| format!("eigenvalue decomposition failed: {e:?}"))?;
        let s = e.S().column_vector();
        let u = e.U();
        let mut evals = Vec::with_capacity(n);
        for j in 0..n {
            let z = s[j];
            if z.im.abs() > 1e-9 * z.re.abs().max(1.0) {
                return Err(
                    "the weights matrix has complex eigenvalues; the spreg likelihood of \
                     ML_ErrorRE is not defined for it"
                        .into(),
                );
            }
            evals.push(z.re);
        }
        let mut sorted = evals.clone();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|p| p[1] - p[0] < REPEATED) {
            return Err(
                "the non-symmetric weights matrix has repeated eigenvalues; spreg's likelihood \
                 then depends on LAPACK's choice of eigenvectors inside each eigenspace and is not \
                 reproducible"
                    .into(),
            );
        }
        let mut vecs = Mat::<f64>::zeros(n, n);
        for j in 0..n {
            let pivot = (0..n)
                .max_by(|&a, &b| u[(a, j)].norm().total_cmp(&u[(b, j)].norm()))
                .unwrap_or(0);
            let phase = u[(pivot, j)] / u[(pivot, j)].norm();
            let col: Vec<f64> = (0..n).map(|i| (u[(i, j)] / phase).re).collect();
            let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            for i in 0..n {
                vecs[(i, j)] = col[i] / norm;
            }
        }
        Ok(Spectral { evals, vecs })
    }

    fn new(w: &Mat<f64>) -> Result<Spectral, String> {
        if is_symmetric(w) {
            Spectral::symmetric(w)
        } else {
            Spectral::general(w)
        }
    }
}

/// The concentrated problem for one panel.
struct Problem<'a> {
    n: usize,
    t: usize,
    nt: f64,
    y: &'a [f64],
    x: &'a [Vec<f64>],
    ylag: Vec<f64>,
    xlag: Vec<Vec<f64>>,
    /// Area means of `y` and of every column of `x`.
    ybar: Vec<f64>,
    xbar: Vec<Vec<f64>>,
    /// `W ybar`, `W xbar`.
    wybar: Vec<f64>,
    wxbar: Vec<Vec<f64>>,
    spec: Spectral,
    /// `V' ybar`, `V' xbar`.
    qy: Vec<f64>,
    qx: Vec<Vec<f64>>,
    evaluations: Cell<usize>,
}

fn area_means(v: &[f64], n: usize, t: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..t).map(|s| v[s * n + i]).sum::<f64>() / t as f64)
        .collect()
}

fn mat_vec(m: &Mat<f64>, v: &[f64], transpose: bool) -> Vec<f64> {
    let n = m.nrows();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if transpose { m[(j, i)] } else { m[(i, j)] } * v[j])
                .sum()
        })
        .collect()
}

impl<'a> Problem<'a> {
    fn new(y: &'a [f64], x: &'a [Vec<f64>], w: &Csr, t: usize, spec: Spectral) -> Problem<'a> {
        let n = w.n;
        let wd = dense(w);
        let ylag = lag_nt(w, y, n).unwrap_or_default();
        let xlag = x
            .iter()
            .map(|c| lag_nt(w, c, n).unwrap_or_default())
            .collect();
        let ybar = area_means(y, n, t);
        let xbar: Vec<Vec<f64>> = x.iter().map(|c| area_means(c, n, t)).collect();
        let wybar = mat_vec(&wd, &ybar, false);
        let wxbar = xbar.iter().map(|c| mat_vec(&wd, c, false)).collect();
        let qy = mat_vec(&spec.vecs, &ybar, true);
        let qx = xbar.iter().map(|c| mat_vec(&spec.vecs, c, true)).collect();
        Problem {
            n,
            t,
            nt: (n * t) as f64,
            y,
            x,
            ylag,
            xlag,
            ybar,
            xbar,
            wybar,
            wxbar,
            spec,
            qy,
            qx,
            evaluations: Cell::new(0),
        }
    }

    /// Test hook: flips the sign of the given eigenvector columns (and `V' ybar` with them).
    #[cfg(test)]
    fn flip_signs(&mut self, cols: &[usize]) {
        for &j in cols {
            for i in 0..self.n {
                self.spec.vecs[(i, j)] = -self.spec.vecs[(i, j)];
            }
            self.qy[j] = -self.qy[j];
            for q in &mut self.qx {
                q[j] = -q[j];
            }
        }
    }

    /// `ys` and the columns of `xs` for the between part `pybar = (P - B) ybar` etc.
    fn transform(&self, lam: f64, pr_y: &[f64], pr_x: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = self.n;
        let ys = (0..self.y.len())
            .map(|r| self.y[r] + pr_y[r % n] - lam * self.ylag[r])
            .collect();
        let xs = (0..self.x.len())
            .map(|j| {
                (0..self.y.len())
                    .map(|r| self.x[j][r] + pr_x[j][r % n] - lam * self.xlag[j][r])
                    .collect()
            })
            .collect();
        (ys, xs)
    }

    /// spreg's `err_c_loglik_ord` (the negative concentrated log-likelihood, no constants).
    fn objective(&self, lam: f64, phi: f64) -> f64 {
        self.evaluations.set(self.evaluations.get() + 1);
        let tf = self.t as f64;
        let phi2 = phi * phi;
        let n = self.n;
        let (mut d, mut phi_jacob, mut jacob) = (Vec::with_capacity(n), 0.0, 0.0);
        for &w in &self.spec.evals {
            let b = 1.0 - lam * w;
            d.push((tf * phi2 + 1.0 / (b * b)).powf(-0.5));
            phi_jacob += 0.5 * (1.0 + tf * phi2 * b * b).ln();
            jacob += b.abs().ln();
        }
        let between = |q: &[f64], bar: &[f64], wbar: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| d[i] * q[i] - bar[i] + lam * wbar[i])
                .collect()
        };
        let pr_y = between(&self.qy, &self.ybar, &self.wybar);
        let pr_x: Vec<Vec<f64>> = (0..self.x.len())
            .map(|j| between(&self.qx[j], &self.xbar[j], &self.wxbar[j]))
            .collect();
        let (ys, xs) = self.transform(lam, &pr_y, &pr_x);
        let Ok((b, _)) = coefficients(&xs, &ys) else {
            return f64::NAN;
        };
        let xsys: Vec<f64> = xs.iter().map(|c| dot(c, &ys)).collect();
        let ee = dot(&ys, &ys) - dot(&xsys, &b);
        let sig2 = ee / self.nt;
        self.nt / 2.0 * sig2.ln() + phi_jacob - tf * jacob
    }

    /// `phi` (log scale) and objective value minimizing the objective for a given `lambda`.
    fn profile(&self, lam: f64) -> (f64, f64) {
        let f = |u: f64| self.objective(lam, u.exp());
        let (lo, hi) = (PHI_BOUNDS.0.ln(), PHI_BOUNDS.1.ln());
        let (a, b) = bracket(&f, lo, hi, 24);
        match minimize_bounded(f, a, b, TOL, 500) {
            Ok(m) => (m.x.exp(), m.fun),
            Err(_) => (f64::NAN, f64::NAN),
        }
    }

    /// Nested search for the maximum likelihood `(lambda, phi)`.
    fn optimize(&self) -> Result<(f64, f64, bool), String> {
        let g = |lam: f64| self.profile(lam).1;
        let (a, b) = bracket(&g, LAM_BOUNDS.0, LAM_BOUNDS.1, 24);
        let m = minimize_bounded(g, a, b, TOL, 500)?;
        let (phi, fun) = self.profile(m.x);
        if !fun.is_finite() {
            return Err("the likelihood could not be evaluated at the optimum".into());
        }
        Ok((m.x, phi, m.converged()))
    }
}

/// The grid interval around the best of `g + 1` equally spaced values (NaN counts as infinite).
fn bracket<F: Fn(f64) -> f64>(f: &F, lo: f64, hi: f64, g: usize) -> (f64, f64) {
    let step = (hi - lo) / g as f64;
    let vals: Vec<f64> = (0..=g)
        .map(|i| {
            let v = f(lo + step * i as f64);
            if v.is_nan() {
                f64::INFINITY
            } else {
                v
            }
        })
        .collect();
    let best = (0..=g)
        .min_by(|&a, &b| vals[a].total_cmp(&vals[b]))
        .unwrap_or(0);
    (
        lo + step * best.saturating_sub(1) as f64,
        lo + step * (best + 1).min(g) as f64,
    )
}

fn matmul(a: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    a * b
}

/// `trace(a b)`.
fn trace_prod(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let n = a.nrows();
    (0..n)
        .map(|i| (0..n).map(|j| a[(i, j)] * b[(j, i)]).sum::<f64>())
        .sum()
}

fn trace(a: &Mat<f64>) -> f64 {
    (0..a.nrows()).map(|i| a[(i, i)]).sum()
}

/// `ML_ErrorRE`. `at = Some((lambda, phi))` evaluates everything at that point (with `phi` the
/// reported ratio `sigma_u^2 / sigma^2`) instead of optimizing.
pub fn ml_error_re(
    y: &[f64],
    x: &[Vec<f64>],
    w: &Csr,
    t: usize,
    at: Option<(f64, f64)>,
) -> Result<Entries, String> {
    let n = w.n;
    if t < 2 {
        return Err("a panel needs at least two periods".into());
    }
    if y.len() != n * t {
        return Err(format!(
            "y has {} values but the panel has {n} areas x {t} periods (length mismatch)",
            y.len()
        ));
    }
    if x.is_empty() || x.iter().any(|c| c.len() != n * t) {
        return Err("every x column needs n * t values (length mismatch)".into());
    }
    let wd = dense(w);
    let problem = Problem::new(y, x, w, t, Spectral::new(&wd)?);
    let (lam, phi_opt, converged) = match at {
        Some((lam, phi)) if phi >= 0.0 => (lam, phi.sqrt(), true),
        Some(_) => return Err("phi must be non-negative".into()),
        None => problem.optimize()?,
    };
    let fun = problem.objective(lam, phi_opt);
    let (nf, tf, k) = (n as f64, t as f64, x.len());
    let nt = problem.nt;
    let logll = -fun - nt / 2.0 * ln2pi() - nt / 2.0;
    let phi = phi_opt * phi_opt;

    // Exact GLS transform: P = Omega^-1/2, Omega = t phi I + (B'B)^-1.
    let b = Mat::from_fn(n, n, |i, j| f64::from(u8::from(i == j)) - lam * wd[(i, j)]);
    let wtw = matmul(&wd.transpose().to_owned(), &wd);
    let ata = Mat::from_fn(n, n, |i, j| {
        f64::from(u8::from(i == j)) - lam * (wd[(i, j)] + wd[(j, i)]) + lam * lam * wtw[(i, j)]
    });
    let atai = inverse(&ata).map_err(|_| "I - lambda W is singular at the optimum".to_string())?;
    let omega = Mat::from_fn(n, n, |i, j| {
        atai[(i, j)] + if i == j { tf * phi } else { 0.0 }
    });
    let eo = omega
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("eigenvalue decomposition failed: {e:?}"))?;
    let (uo, so) = (eo.U(), eo.S().column_vector());
    let p_apply = |v: &[f64]| -> Vec<f64> {
        let q: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|i| uo[(i, j)] * v[i]).sum::<f64>() * so[j].powf(-0.5))
            .collect();
        (0..n)
            .map(|i| (0..n).map(|j| uo[(i, j)] * q[j]).sum())
            .collect()
    };
    let between = |bar: &[f64]| -> Vec<f64> {
        let pv = p_apply(bar);
        let bv = mat_vec(&b, bar, false);
        (0..n).map(|i| pv[i] - bv[i]).collect()
    };
    let pr_y = between(&problem.ybar);
    let pr_x: Vec<Vec<f64>> = problem.xbar.iter().map(|c| between(c)).collect();
    let (ys, xs) = problem.transform(lam, &pr_y, &pr_x);
    let (bhat, xsxsi) = coefficients(&xs, &ys)?;
    let u: Vec<f64> = y.iter().zip(fitted(x, &bhat)).map(|(a, f)| a - f).collect();
    let predy: Vec<f64> = y.iter().zip(&u).map(|(a, c)| a - c).collect();
    let e_filtered: Vec<f64> = ys
        .iter()
        .zip(fitted(&xs, &bhat))
        .map(|(a, f)| a - f)
        .collect();
    let sig2 = dot(&e_filtered, &e_filtered) / nt;
    let sig2_u = phi * sig2;

    // Variance of lambda and phi (spreg's information matrix, kept as written).
    let wt_a = Mat::from_fn(n, n, |i, j| wd[(j, i)] - lam * wtw[(i, j)]);
    let wa_aw = Mat::from_fn(n, n, |i, j| wt_a[(i, j)] + wt_a[(j, i)]);
    let gamma = matmul(&wa_aw, &atai);
    let vi = inverse(&Mat::from_fn(n, n, |i, j| {
        atai[(i, j)] + if i == j { tf * phi_opt } else { 0.0 }
    }))
    .map_err(|_| "the information matrix is singular".to_string())?;
    let sigma = matmul(&vi, &atai);
    let sigma_gamma = matmul(&sigma, &gamma);
    let (tr1, tr2, tr3) = (trace(&gamma), trace(&vi), trace(&sigma));
    let tr4 = trace(&sigma_gamma);
    let tr5 = trace_prod(&sigma, &vi);
    let tr6 = trace_prod(&sigma_gamma, &vi);
    let tr7 = trace_prod(&sigma_gamma, &sigma);
    let mut v = Mat::<f64>::zeros(3, 3);
    v[(0, 0)] = (tf - 1.0) / 2.0 * tr1 * tr1 + 0.5 * tr4 * tr4;
    v[(1, 0)] = tf / (2.0 * sig2) * tr6;
    v[(2, 0)] = (tf - 1.0) / (2.0 * sig2) * tr1 + tr7 / (2.0 * sig2);
    v[(0, 1)] = v[(1, 0)];
    v[(1, 1)] = tf * tf / (2.0 * sig2 * sig2) * tr2 * tr2;
    v[(2, 1)] = tf / (2.0 * sig2 * sig2) * tr5;
    v[(0, 2)] = v[(2, 0)];
    v[(1, 2)] = v[(2, 1)];
    v[(2, 2)] = ((tf - 1.0) * nf + tr3 * tr3) / (2.0 * sig2 * sig2);
    let vm1 = inverse(&v).map_err(|_| "the information matrix is singular".to_string())?;
    let vm = Mat::from_fn(k + 2, k + 2, |i, j| {
        if i < k && j < k {
            sig2 * xsxsi[(i, j)]
        } else if i >= k && j >= k {
            vm1[(i - k, j - k)]
        } else {
            0.0
        }
    });
    let mut betas = bhat;
    betas.extend([lam, sig2_u]);
    let std_err: Vec<f64> = (0..k + 2).map(|i| vm[(i, i)].sqrt()).collect();
    let kf = k as f64;

    let mut out = Entries::new();
    out.insert("lam", vec![lam]);
    out.insert("phi", vec![phi]);
    out.insert("sig2_u", vec![sig2_u]);
    out.insert("vm", row_major(&vm));
    out.insert("z_stat", z_stats(&betas, &std_err));
    out.insert("std_err", std_err);
    out.insert("logll", vec![logll]);
    out.insert("aic", vec![2.0 * kf - 2.0 * logll]);
    out.insert("schwarz", vec![kf * nt.ln() - 2.0 * logll]);
    out.insert("sig2", vec![sig2]);
    out.insert("iterations", vec![problem.evaluations.get() as f64]);
    out.insert("converged", vec![f64::from(u8::from(converged))]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("e_filtered", e_filtered);
    out.insert("betas", betas);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::ml_common::dense;
    use super::*;
    use crate::graph::Csr;
    use faer::Mat;

    fn csr(n: usize, edges: &[(usize, usize, f64)]) -> Csr {
        let mut sorted = edges.to_vec();
        sorted.sort_by_key(|e| (e.0, e.1));
        let mut offsets = vec![0usize; n + 1];
        for e in &sorted {
            offsets[e.0 + 1] += 1;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        Csr {
            n,
            offsets,
            neighbors: sorted.iter().map(|e| e.1 as u32).collect(),
            weights: sorted.iter().map(|e| e.2).collect(),
        }
    }

    fn ring(n: usize) -> Csr {
        let mut e = Vec::new();
        for i in 0..n {
            e.push((i, (i + 1) % n, 0.5));
            e.push((i, (i + n - 1) % n, 0.5));
        }
        csr(n, &e)
    }

    fn panel(n: usize, t: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut seed = 7u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64 - 0.5
        };
        let x1: Vec<f64> = (0..n * t).map(|_| next() * 4.0).collect();
        let ones = vec![1.0; n * t];
        let y: Vec<f64> = (0..n * t)
            .map(|r| 1.0 + 2.0 * x1[r] + (r % n) as f64 * 0.3 + next())
            .collect();
        (y, vec![ones, x1])
    }

    #[test]
    fn complex_spectrum_is_rejected() {
        let cycle = csr(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]);
        let (y, x) = panel(3, 3);
        let err = ml_error_re(&y, &x, &cycle, 3, None).unwrap_err();
        assert!(err.contains("complex eigenvalues"), "{err}");
    }

    #[test]
    fn repeated_eigenvalues_of_a_non_symmetric_w_are_rejected() {
        let nilpotent = csr(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        let (y, x) = panel(4, 3);
        let err = ml_error_re(&y, &x, &nilpotent, 3, None).unwrap_err();
        assert!(err.contains("repeated eigenvalues"), "{err}");
    }

    #[test]
    fn a_symmetric_w_with_repeated_eigenvalues_is_fitted() {
        let (y, x) = panel(6, 4);
        let out = ml_error_re(&y, &x, &ring(6), 4, None).unwrap();
        assert!(out["lam"][0].abs() < 1.0);
        assert_eq!(out["betas"].len(), 4);
    }

    #[test]
    fn fitted_optimum_is_stationary_and_consistent_with_the_hook() {
        let (y, x) = panel(8, 4);
        let w = ring(8);
        let fit = ml_error_re(&y, &x, &w, 4, None).unwrap();
        let (lam, phi) = (fit["lam"][0], fit["phi"][0]);
        let at = |l: f64, p: f64| ml_error_re(&y, &x, &w, 4, Some((l, p))).unwrap()["logll"][0];
        let best = at(lam, phi);
        assert!((best - fit["logll"][0]).abs() < 1e-9);
        for (dl, dp) in [(1e-3, 0.0), (-1e-3, 0.0), (0.0, 1e-3), (0.0, -1e-3)] {
            assert!(at(lam + dl, phi + dp) <= best + 1e-9, "({dl}, {dp})");
        }
    }

    fn problem<'a>(
        w: &Csr,
        data: &'a (Vec<f64>, Vec<Vec<f64>>),
        spec: fn(&Mat<f64>) -> Result<Spectral, String>,
    ) -> Problem<'a> {
        Problem::new(&data.0, &data.1, w, 4, spec(&dense(w)).unwrap())
    }

    /// 2 x 3 rook lattice: symmetric with a simple spectrum.
    fn lattice() -> Csr {
        let mut e = Vec::new();
        for (a, b) in [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)] {
            e.push((a, b, 1.0));
            e.push((b, a, 1.0));
        }
        csr(6, &e)
    }

    #[test]
    fn objective_ignores_eigenvector_signs() {
        // spreg's statistic only depends on V diag(1/c) V', so flipping signs changes nothing.
        let w = lattice();
        let data = panel(6, 4);
        let mut p = problem(&w, &data, Spectral::symmetric);
        let before: Vec<f64> = [(0.1, 0.5), (-0.05, 1.5)]
            .iter()
            .map(|&(l, f)| p.objective(l, f))
            .collect();
        p.flip_signs(&[1, 3, 4]);
        for (&(l, f), b) in [(0.1, 0.5), (-0.05, 1.5)].iter().zip(&before) {
            assert!((p.objective(l, f) - b).abs() < 1e-9 * b.abs().max(1.0));
        }
    }

    #[test]
    fn general_eigenvectors_reproduce_the_symmetric_objective() {
        // For a symmetric W with a simple spectrum, la.eig (spreg) and eigh agree.
        let w = lattice();
        let data = panel(6, 4);
        let sym = problem(&w, &data, Spectral::symmetric);
        let gen = problem(&w, &data, Spectral::general);
        for (l, f) in [(0.1, 0.5), (-0.05, 1.5), (0.0, 0.1)] {
            let (a, b) = (sym.objective(l, f), gen.objective(l, f));
            assert!((a - b).abs() < 1e-9 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
}
