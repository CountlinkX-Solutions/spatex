//! Joint maximum-likelihood step of the SUR spatial error model (`spreg.SURerrorML`, spreg 1.9.1,
//! `sur_error.py`): the concentrated log-likelihood of the spatial error parameters given the
//! residuals of all equations, its minimization and the trace terms of the asymptotic variance.
//!
//! spreg minimizes `-clik(lambda)` with L-BFGS-B from the previous lambdas; this module minimizes
//! the same objective with a projected BFGS method using the analytic gradient, so the optimum is
//! determined far more tightly than spreg's (see docs/parity.md, "spreg spatial SUR notes").

use faer::Mat;

use super::linalg::inverse;
use super::ml_common::{check_inputs, log_abs_det_lu, traces, wai, Jacobian, Method};
use crate::graph::Csr;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Residual columns of the equations, their spatial lags and the log-Jacobian evaluator.
pub struct Problem<'a> {
    w: &'a Csr,
    e: &'a [Vec<f64>],
    we: Vec<Vec<f64>>,
    jac: Jacobian,
    n: f64,
}

/// Value, and when asked the gradient, of `-clik`.
struct Eval {
    f: f64,
    grad: Vec<f64>,
}

impl<'a> Problem<'a> {
    pub fn new(w: &'a Csr, e: &'a [Vec<f64>], max_dense_n: usize) -> Result<Self, String> {
        let n = e.first().map_or(0, Vec::len);
        if e.len() < 2 || e.iter().any(|c| c.len() != n) {
            return Err("need at least two residual columns of the same length".to_string());
        }
        check_inputs(w, n, 1.0, max_dense_n)?;
        let we = e.iter().map(|c| w.lag(c)).collect::<Result<Vec<_>, _>>()?;
        Ok(Problem {
            w,
            e,
            we,
            jac: Jacobian::new(w, Method::Full)?,
            n: n as f64,
        })
    }

    /// `sig = F'F / n` for the filtered residuals `F = E - W E diag(lambda)`, and `F`.
    fn sigma(&self, lam: &[f64]) -> (Mat<f64>, Vec<Vec<f64>>) {
        let f: Vec<Vec<f64>> = self
            .e
            .iter()
            .zip(&self.we)
            .zip(lam)
            .map(|((e, we), l)| e.iter().zip(we).map(|(a, b)| a - l * b).collect())
            .collect();
        let k = f.len();
        let sig = Mat::from_fn(k, k, |i, j| dot(&f[i], &f[j]) / self.n);
        (sig, f)
    }

    /// spreg's `clik`, negated: `n/2 ln|sig| - sum_r ln|det(I - lambda_r W)|`.
    pub fn objective(&self, lam: &[f64]) -> f64 {
        let (sig, _) = self.sigma(lam);
        let ldet = log_abs_det_lu(sig).0;
        let logjac: f64 = lam.iter().map(|&l| self.jac.log_abs_det(l)).sum();
        self.n / 2.0 * ldet - logjac
    }

    fn eval(&self, lam: &[f64]) -> Result<Eval, String> {
        let (sig, f) = self.sigma(lam);
        let k = lam.len();
        let ldet = log_abs_det_lu(sig.clone()).0;
        let logjac: f64 = lam.iter().map(|&l| self.jac.log_abs_det(l)).sum();
        let val = self.n / 2.0 * ldet - logjac;
        if !val.is_finite() {
            return Ok(Eval {
                f: f64::INFINITY,
                grad: vec![0.0; k],
            });
        }
        let sigi = inverse(&sig)?;
        let mut grad = Vec::with_capacity(k);
        for r in 0..k {
            // d sig = e_r g' + g e_r' with g_j = -(W e_r . f_j) / n
            let g: Vec<f64> = (0..k).map(|j| -dot(&self.we[r], &f[j]) / self.n).collect();
            let tr_sig: f64 = 2.0 * (0..k).map(|j| g[j] * sigi[(j, r)]).sum::<f64>();
            let tr1 = traces(self.w, lam[r])?.tr1;
            grad.push(self.n / 2.0 * tr_sig + tr1);
        }
        Ok(Eval { f: val, grad })
    }
}

/// Result of [`minimize`].
#[derive(Clone, Debug)]
pub struct Fit {
    pub lam: Vec<f64>,
    /// The minimum of `-clik`.
    pub fun: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn clamp(v: f64) -> f64 {
    v.clamp(-1.0, 1.0)
}

/// Minimizes `-clik` over `[-1, 1]^k` from `lam0` (projected BFGS, backtracking line search).
pub fn minimize(p: &Problem, lam0: &[f64]) -> Result<Fit, String> {
    let k = lam0.len();
    let mut x: Vec<f64> = lam0.iter().map(|&v| clamp(v)).collect();
    let mut cur = p.eval(&x)?;
    if !cur.f.is_finite() {
        return Err("the objective is not finite at the starting lambdas".to_string());
    }
    let mut h = Mat::<f64>::identity(k, k);
    let mut converged = false;
    let mut iterations = 0;
    while iterations < 500 {
        // projected gradient: free coordinates only
        let free: Vec<bool> = (0..k)
            .map(|i| !((x[i] <= -1.0 && cur.grad[i] > 0.0) || (x[i] >= 1.0 && cur.grad[i] < 0.0)))
            .collect();
        let pg: Vec<f64> = (0..k)
            .map(|i| if free[i] { cur.grad[i] } else { 0.0 })
            .collect();
        if pg.iter().all(|g| g.abs() < 1.0e-10) {
            converged = true;
            break;
        }
        iterations += 1;
        let mut d: Vec<f64> = (0..k)
            .map(|i| -(0..k).map(|j| h[(i, j)] * pg[j]).sum::<f64>())
            .collect();
        for i in 0..k {
            if !free[i] {
                d[i] = 0.0;
            }
        }
        if dot(&d, &pg) >= 0.0 {
            h = Mat::identity(k, k);
            d = pg.iter().map(|g| -g).collect();
        }
        // backtracking with projection
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..60 {
            let cand: Vec<f64> = (0..k).map(|i| clamp(x[i] + t * d[i])).collect();
            let step: Vec<f64> = (0..k).map(|i| cand[i] - x[i]).collect();
            let e = p.eval(&cand)?;
            if e.f.is_finite() && e.f <= cur.f + 1.0e-4 * dot(&cur.grad, &step) {
                accepted = Some((cand, step, e));
                break;
            }
            t *= 0.5;
        }
        let Some((cand, s, e)) = accepted else {
            // no decrease possible: at the limit of the floating point precision
            converged = pg.iter().all(|g| g.abs() < 1.0e-6);
            break;
        };
        let y: Vec<f64> = (0..k).map(|i| e.grad[i] - cur.grad[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1.0e-14 {
            // BFGS update of the inverse Hessian
            let hy: Vec<f64> = (0..k)
                .map(|i| (0..k).map(|j| h[(i, j)] * y[j]).sum())
                .collect();
            let yhy = dot(&y, &hy);
            h = Mat::from_fn(k, k, |i, j| {
                h[(i, j)] + (1.0 + yhy / sy) * s[i] * s[j] / sy - (hy[i] * s[j] + s[i] * hy[j]) / sy
            });
        }
        x = cand;
        cur = e;
    }
    Ok(Fit {
        lam: x,
        fun: cur.f,
        iterations,
        converged,
    })
}

/// Trace terms of `D_i = W (I - lambda_i W)^{-1}` for the asymptotic variance (`surerrvm`):
/// `tr(D_i)`, `tr(D_i D_i)` and the matrix `tr(D_i' D_j)` (its diagonal is `tr(D_i' D_i)`).
pub struct TraceTerms {
    pub tr_d: Vec<f64>,
    pub tr_dd: Vec<f64>,
    pub cross: Vec<Vec<f64>>,
}

pub fn trace_terms(w: &Csr, lam: &[f64], max_dense_n: usize) -> Result<TraceTerms, String> {
    check_inputs(w, w.n, 1.0, max_dense_n)?;
    let d: Vec<Mat<f64>> = lam
        .iter()
        .map(|&l| wai(w, l))
        .collect::<Result<Vec<_>, _>>()?;
    let n = w.n;
    let mut tr_d = Vec::new();
    let mut tr_dd = Vec::new();
    for a in &d {
        let t = traces_of(a, n);
        tr_d.push(t.0);
        tr_dd.push(t.1);
    }
    let k = lam.len();
    let cross = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let mut s = 0.0;
                    for a in 0..n {
                        for b in 0..n {
                            s += d[i][(a, b)] * d[j][(a, b)];
                        }
                    }
                    s
                })
                .collect()
        })
        .collect();
    Ok(TraceTerms { tr_d, tr_dd, cross })
}

fn traces_of(d: &Mat<f64>, n: usize) -> (f64, f64) {
    let mut tr = 0.0;
    let mut tr2 = 0.0;
    for i in 0..n {
        tr += d[(i, i)];
        for j in 0..n {
            tr2 += d[(i, j)] * d[(j, i)];
        }
    }
    (tr, tr2)
}

#[cfg(test)]
mod tests {
    use super::super::ml_lag::tests::lattice;
    use super::*;

    fn residuals(n: usize) -> Vec<Vec<f64>> {
        let a: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| 0.5 * a[i] + ((i * 5 + 1) % 7) as f64 - 3.0)
            .collect();
        let c: Vec<f64> = (0..n).map(|i| ((i * 3 + 2) % 9) as f64 - 4.0).collect();
        vec![a, b, c]
    }

    #[test]
    fn the_gradient_matches_finite_differences() {
        let w = lattice(5);
        let e = residuals(25);
        let p = Problem::new(&w, &e, 100).unwrap();
        let lam = [0.2, -0.1, 0.3];
        let g = p.eval(&lam).unwrap().grad;
        for r in 0..3 {
            let h = 1.0e-6;
            let mut up = lam;
            let mut dn = lam;
            up[r] += h;
            dn[r] -= h;
            let num = (p.objective(&up) - p.objective(&dn)) / (2.0 * h);
            assert!(
                (g[r] - num).abs() < 1.0e-5 * (1.0 + num.abs()),
                "{r}: {} {num}",
                g[r]
            );
        }
    }

    #[test]
    fn the_minimum_has_a_vanishing_gradient_and_beats_its_neighbours() {
        let w = lattice(5);
        let e = residuals(25);
        let p = Problem::new(&w, &e, 100).unwrap();
        let fit = minimize(&p, &[0.0, 0.0, 0.0]).unwrap();
        assert!(fit.converged);
        let g = p.eval(&fit.lam).unwrap().grad;
        assert!(g.iter().all(|v| v.abs() < 1.0e-8), "{g:?}");
        assert!((fit.fun - p.objective(&fit.lam)).abs() < 1.0e-12);
        for r in 0..3 {
            for d in [1.0e-3, -1.0e-3] {
                let mut l = fit.lam.clone();
                l[r] += d;
                assert!(p.objective(&l) > fit.fun);
            }
        }
    }

    #[test]
    fn invalid_input_is_rejected() {
        let w = lattice(5);
        let e = residuals(25);
        assert!(Problem::new(&w, &e[..1], 100).is_err());
        assert!(Problem::new(&w, &e, 3)
            .err()
            .unwrap()
            .contains("max_dense_n"));
    }

    #[test]
    fn trace_terms_match_the_single_equation_traces() {
        let w = lattice(5);
        let t = trace_terms(&w, &[0.2, 0.4], 100).unwrap();
        for (i, l) in [0.2, 0.4].into_iter().enumerate() {
            let s = traces(&w, l).unwrap();
            assert!((t.tr_d[i] - s.tr1).abs() < 1e-10);
            assert!((t.tr_dd[i] - s.tr2).abs() < 1e-10);
            assert!((t.cross[i][i] - s.tr3).abs() < 1e-10);
        }
        assert!((t.cross[0][1] - t.cross[1][0]).abs() < 1e-12);
    }
}
