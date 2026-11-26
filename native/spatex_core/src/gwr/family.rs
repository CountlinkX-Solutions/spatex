//! Exponential families for GWR (Gaussian, Poisson with log link, Binomial with logit link) and
//! the per-location iteratively re-weighted least squares of `spglm.iwls`.
//!
//! The starting values, the clipping of means and the stopping rule are ported literally,
//! including the quirks that decide the final coefficients (`spglm/iwls.py`):
//! the convergence measure is the *smallest* coefficient change, the Poisson linear predictor
//! starts from `y / offset` while the mean starts from the raw `y`, and the weights that
//! produced the final coefficients (not weights at the final mean) define the hat matrix.

use super::lu::Lu;

/// `f64::EPSILON`, the clip used by spglm's `_clean`.
pub(crate) const EPS: f64 = f64::EPSILON;

/// Response distribution of a GWR model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    Gaussian,
    Poisson,
    Binomial,
}

impl Family {
    /// Parses `"gaussian"`, `"poisson"` or `"binomial"`.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "gaussian" => Ok(Self::Gaussian),
            "poisson" => Ok(Self::Poisson),
            "binomial" => Ok(Self::Binomial),
            other => Err(format!(
                "unsupported family {other:?}; expected gaussian, poisson or binomial"
            )),
        }
    }
}

fn clip_p(p: f64) -> f64 {
    p.clamp(EPS, 1.0 - EPS)
}

/// Squared deviance residual of a Poisson observation: `2 (y ln(clip(y/mu)) - (y - mu))`.
pub fn poisson_resid_dev_sq(y: f64, mu: f64) -> f64 {
    let t = y * (y / mu).max(EPS).ln() - (y - mu);
    let v = 2.0 * t;
    if v < 0.0 {
        f64::NAN
    } else {
        v
    }
}

/// Squared deviance residual of a Bernoulli observation with the mean clipped to `(0, 1)`.
pub fn binomial_resid_dev_sq(y: f64, mu: f64) -> f64 {
    let mu = clip_p(mu);
    let p = if y == 1.0 { mu } else { 1.0 - mu };
    -2.0 * p.ln()
}

impl Family {
    /// Squared deviance residual of one observation (`resid_dev(y, mu)^2`).
    pub fn resid_dev_sq(self, y: f64, mu: f64) -> f64 {
        match self {
            Self::Gaussian => (y - mu) * (y - mu),
            Self::Poisson => poisson_resid_dev_sq(y, mu),
            Self::Binomial => binomial_resid_dev_sq(y, mu),
        }
    }

    /// `family.deviance(y, mu)` as mgwr evaluates it for the null model: the Poisson one omits
    /// the `-(y - mu)` term (sic).
    pub fn deviance(self, y: &[f64], mu: &[f64]) -> f64 {
        match self {
            Self::Gaussian => y.iter().zip(mu).map(|(y, m)| (y - m) * (y - m)).sum(),
            Self::Poisson => {
                2.0 * y
                    .iter()
                    .zip(mu)
                    .map(|(y, m)| y * (y / m).max(EPS).ln())
                    .sum::<f64>()
            }
            Self::Binomial => {
                -2.0 * y
                    .iter()
                    .zip(mu)
                    .map(|(&y, &m)| {
                        if y == 1.0 {
                            (m + 1e-200).ln()
                        } else {
                            (1.0 - m + 1e-200).ln()
                        }
                    })
                    .sum::<f64>()
            }
        }
    }

    /// `family.loglike(y, mu)` with scale one (Gaussian is handled by the caller).
    pub fn loglike(self, y: &[f64], mu: &[f64]) -> f64 {
        match self {
            Self::Gaussian => f64::NAN,
            Self::Poisson => y
                .iter()
                .zip(mu)
                .map(|(&y, &m)| y * m.ln() - m - crate::dynamics::special::ln_gamma(y + 1.0))
                .sum(),
            Self::Binomial => y
                .iter()
                .zip(mu)
                .map(|(&y, &m)| y * (m / (1.0 - m) + 1e-200).ln() + (1.0 - m).ln())
                .sum(),
        }
    }

    /// IRLS weight and working response for the mean `mu`: `1 / (g'(mu)^2 V(mu))` and
    /// `v + g'(mu) (y - mu)`.
    fn weight_and_response(self, y: f64, mu: f64, v: f64) -> (f64, f64) {
        match self {
            Self::Poisson => {
                let d = 1.0 / mu.max(EPS);
                (1.0 / (d * d * mu.abs()), v + d * (y - mu))
            }
            Self::Binomial => {
                let p = clip_p(mu);
                let d = 1.0 / (p * (1.0 - p));
                (1.0 / (d * d * (p * (1.0 - p))), v + d * (y - mu))
            }
            Self::Gaussian => (1.0, y),
        }
    }

    fn inverse_link(self, v: f64, offset: f64) -> f64 {
        match self {
            Self::Poisson => v.exp() * offset,
            Self::Binomial => 1.0 / (1.0 + (-v).exp()),
            Self::Gaussian => v,
        }
    }
}

/// Weighted normal equations `A = sum_j ww_j x_j x_j'` and `b = sum_j ww_j z_j x_j`, skipping
/// zero weights.
pub(crate) fn normal_equations(x: &[f64], k: usize, ww: &[f64], z: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; k * k];
    let mut b = vec![0.0; k];
    for (j, &wj) in ww.iter().enumerate() {
        if wj == 0.0 {
            continue;
        }
        let xj = &x[j * k..(j + 1) * k];
        for p in 0..k {
            let wx = wj * xj[p];
            b[p] += wx * z[j];
            for q in p..k {
                a[p * k + q] += wx * xj[q];
            }
        }
    }
    for p in 0..k {
        for q in 0..p {
            a[p * k + q] = a[q * k + p];
        }
    }
    (a, b)
}

/// Result of [`iwls`]: the final coefficients and the quantities of the last iteration.
pub(crate) struct Iwls {
    pub betas: Vec<f64>,
    /// Fitted means for every observation (including the offset for Poisson).
    pub mu: Vec<f64>,
    /// Square roots of the IRLS weights that produced `betas`.
    pub s: Vec<f64>,
    /// Factorization of `X' diag(w s^2) X` of the last iteration.
    pub lu: Lu,
    pub iterations: u32,
}

/// Inputs of one IWLS fit: the design, the response and the local kernel weights.
pub(crate) struct IwlsInput<'a> {
    pub y: &'a [f64],
    /// Row-major `n x k`.
    pub x: &'a [f64],
    pub k: usize,
    pub family: Family,
    /// Per-observation offset (only Poisson uses it); empty means ones.
    pub offset: &'a [f64],
    /// Kernel weights of the focal location (all ones for a global fit).
    pub w: &'a [f64],
    pub tol: f64,
    pub max_iter: usize,
}

/// Port of `spglm.iwls.iwls` with `wi` given (starting coefficients are zero).
pub(crate) fn iwls(inp: &IwlsInput) -> Result<Iwls, String> {
    let IwlsInput {
        y,
        x,
        k,
        family,
        offset,
        w,
        tol,
        max_iter,
    } = *inp;
    let n = y.len();
    let off = |j: usize| if offset.is_empty() { 1.0 } else { offset[j] };
    let mut betas = vec![0.0; k];
    let y_used: Vec<f64> = if family == Family::Binomial {
        y.iter().map(|&v| clip_p(v)).collect()
    } else {
        y.to_vec()
    };
    let (mut v, mut mu): (Vec<f64>, Vec<f64>) = match family {
        Family::Poisson => {
            let y_off: Vec<f64> = (0..n).map(|j| y[j] / off(j)).collect();
            let m_off = y_off.iter().sum::<f64>() / n as f64;
            let v = y_off
                .iter()
                .map(|&t| ((t + m_off) / 2.0).max(EPS).ln())
                .collect();
            let m_y = y.iter().sum::<f64>() / n as f64;
            (v, y.iter().map(|&t| (t + m_y) / 2.0).collect())
        }
        _ => {
            let mu: Vec<f64> = y_used.iter().map(|&t| (t + 0.5) / 2.0).collect();
            let v = mu
                .iter()
                .map(|&m| {
                    let p = clip_p(m);
                    (p / (1.0 - p)).ln()
                })
                .collect();
            (v, mu)
        }
    };
    let mut s = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut ww = vec![0.0; n];
    let mut diff = 1.0e6_f64;
    let mut iterations = 0_u32;
    let mut last: Option<Lu> = None;
    while diff > tol && (iterations as usize) < max_iter {
        iterations += 1;
        for j in 0..n {
            let (wt, zj) = family.weight_and_response(y_used[j], mu[j], v[j]);
            s[j] = wt.sqrt();
            z[j] = zj;
            ww[j] = w[j] * (s[j] * s[j]);
        }
        let (a, b) = normal_equations(x, k, &ww, &z);
        let lu = Lu::factor(a, k)?;
        let new = lu.solve(&b);
        for j in 0..n {
            let xj = &x[j * k..(j + 1) * k];
            v[j] = xj.iter().zip(&new).map(|(a, b)| a * b).sum();
            mu[j] = family.inverse_link(v[j], off(j));
        }
        diff = new
            .iter()
            .zip(&betas)
            .map(|(a, b)| (a - b).abs())
            .fold(f64::INFINITY, f64::min);
        betas = new;
        last = Some(lu);
    }
    let Some(lu) = last else {
        return Err("IWLS did not run an iteration".to_string());
    };
    Ok(Iwls {
        betas,
        mu,
        s,
        lu,
        iterations,
    })
}

/// Fitted means of the intercept-only model (`GLM(y, ones, family, offset).fit()`, tolerance
/// `1e-6`, at most 200 iterations), used for the null deviance.
pub(crate) fn null_mu(y: &[f64], family: Family, offset: &[f64]) -> Result<Vec<f64>, String> {
    let n = y.len();
    if family == Family::Gaussian {
        return Ok(vec![y.iter().sum::<f64>() / n as f64; n]);
    }
    let ones = vec![1.0; n];
    let fit = iwls(&IwlsInput {
        y,
        x: &ones,
        k: 1,
        family,
        offset,
        w: &ones,
        tol: 1.0e-6,
        max_iter: 200,
    })?;
    Ok(fit.mu)
}
