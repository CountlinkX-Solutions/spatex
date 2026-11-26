//! Spatial panel models with individual fixed effects (`spreg.panel.ML_LagFE`, `ML_ErrorFE`) and
//! the random-effects lag model (`ML_LagRE`).
//!
//! Data are stacked by period (`y[s * n + i]` is area `i` in period `s`), as spreg's long format.
//! Both models demean every variable by area over time (`demean_panel`), then maximize the
//! concentrated likelihood with SciPy's bounded Brent search, the log-Jacobian being
//! `T ln|I - a W|` (a dense LU per evaluation, like spreg's sparse LU). The asymptotic variance
//! follows spreg literally, including the lag model's cross term `x' (I_T (x) W A^-1) X b`, which
//! spreg evaluates with the *raw* (not demeaned) regressors.

use faer::Mat;

use super::brent::minimize_bounded;
use super::dist::norm_two_sided;
use super::linalg::{inverse, row_major};
use super::ml_common::{dense, impact_methods, multipliers, Impacts, Jacobian, Method};
use super::ols::Entries;
use crate::graph::Csr;

/// Options of the panel estimators.
pub struct Options<'a> {
    pub w: &'a Csr,
    /// Number of periods.
    pub t: usize,
    pub epsilon: f64,
    pub impacts: Impacts,
}

/// `v` minus its mean over time for every area.
pub fn demean(v: &[f64], n: usize, t: usize) -> Vec<f64> {
    quasi_demean(v, n, t, 0.0)
}

/// spreg's `demean_panel(arr, n, t, phi)`: `v - (1 - phi) * mean_t(v)` for every area.
pub fn quasi_demean(v: &[f64], n: usize, t: usize, phi: f64) -> Vec<f64> {
    let means: Vec<f64> = (0..n)
        .map(|i| (0..t).map(|s| v[s * n + i]).sum::<f64>() / t as f64)
        .collect();
    v.iter()
        .enumerate()
        .map(|(e, x)| x - (1.0 - phi) * means[e % n])
        .collect()
}

/// `(I_T (x) W) v`.
pub(super) fn lag_nt(w: &Csr, v: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(v.len());
    for block in v.chunks(n) {
        out.extend(w.lag(block)?);
    }
    Ok(out)
}

/// `(I_T (x) M) v` for a dense `n x n` matrix `M`.
fn dense_nt(m: &Mat<f64>, v: &[f64], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len());
    for block in v.chunks(n) {
        for i in 0..n {
            out.push((0..n).map(|j| m[(i, j)] * block[j]).sum());
        }
    }
    out
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// OLS coefficients of `y` on the columns `x` (normal equations, as spreg's `xtxi xty`).
pub(super) fn coefficients(x: &[Vec<f64>], y: &[f64]) -> Result<(Vec<f64>, Mat<f64>), String> {
    let k = x.len();
    let xtx = Mat::from_fn(k, k, |i, j| dot(&x[i], &x[j]));
    let xtxi = inverse(&xtx)?;
    let xty: Vec<f64> = x.iter().map(|c| dot(c, y)).collect();
    let b = (0..k)
        .map(|i| (0..k).map(|j| xtxi[(i, j)] * xty[j]).sum())
        .collect();
    Ok((b, xtxi))
}

pub(super) fn fitted(x: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    (0..x[0].len())
        .map(|r| x.iter().zip(b).map(|(c, bj)| c[r] * bj).sum())
        .collect()
}

fn check(y: &[f64], x: &[Vec<f64>], opts: &Options) -> Result<(usize, usize), String> {
    let n = opts.w.n;
    let t = opts.t;
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
    Ok((n, t))
}

/// `A^-1 = (I - a W)^-1` and `W A^-1`, dense.
fn inverses(w: &Csr, a: f64) -> Result<(Mat<f64>, Mat<f64>), String> {
    let n = w.n;
    let wd = dense(w);
    let am = Mat::from_fn(n, n, |i, j| f64::from(u8::from(i == j)) - a * wd[(i, j)]);
    let ai = inverse(&am)?;
    let wai = Mat::from_fn(n, n, |i, j| (0..n).map(|k| wd[(i, k)] * ai[(k, j)]).sum());
    Ok((ai, wai))
}

fn traces(wai: &Mat<f64>) -> (f64, f64, f64) {
    let n = wai.nrows();
    let (mut tr1, mut tr2, mut tr3) = (0.0, 0.0, 0.0);
    for i in 0..n {
        tr1 += wai[(i, i)];
        for j in 0..n {
            tr2 += wai[(i, j)] * wai[(j, i)];
            tr3 += wai[(i, j)] * wai[(i, j)];
        }
    }
    (tr1, tr2, tr3)
}

/// `(I_T (x) (I - a W))^-1 v` by the power series of spreg's `inverse_prod(.., 'power_exp')`
/// over the whole stacked vector.
fn power_expansion_nt(
    w: &Csr,
    v: &[f64],
    a: f64,
    threshold: f64,
    n: usize,
) -> Result<Vec<f64>, String> {
    let mut total = v.to_vec();
    let mut inc = v.to_vec();
    let mut test = 1.0e7_f64;
    let mut count = 1u64;
    while test > threshold && count <= 10_000_000 {
        inc = lag_nt(w, &inc, n)?.into_iter().map(|x| x * a).collect();
        for (s, d) in total.iter_mut().zip(&inc) {
            *s += d;
        }
        let old = test;
        test = inc.iter().map(|x| x * x).sum::<f64>().sqrt();
        if test > old {
            return Err("power expansion will not converge".into());
        }
        count += 1;
    }
    Ok(total)
}

pub(super) fn z_stats(betas: &[f64], std_err: &[f64]) -> Vec<f64> {
    betas
        .iter()
        .zip(std_err)
        .flat_map(|(b, s)| {
            let z = b / s;
            [z, norm_two_sided(z)]
        })
        .collect()
}

pub(super) fn ln2pi() -> f64 {
    (2.0 * std::f64::consts::PI).ln()
}

/// `ML_LagFE`: `y = rho W y + X b + mu_i + e` on the demeaned panel. `x` are the raw regressors
/// (no constant; time dummies and SLX terms included by the caller).
pub fn ml_lag_fe(y: &[f64], x: &[Vec<f64>], opts: &Options) -> Result<Entries, String> {
    let (n, t) = check(y, x, opts)?;
    let w = opts.w;
    let nt = (n * t) as f64;
    let yd = demean(y, n, t);
    let xd: Vec<Vec<f64>> = x.iter().map(|c| demean(c, n, t)).collect();
    if let Some(j) = xd.iter().position(|c| c.iter().all(|v| v.abs() < 1e-12)) {
        return Err(format!(
            "x column {j} is time-invariant; fixed effects absorb it"
        ));
    }
    let ylag = lag_nt(w, &yd, n)?;
    let (b0, _) = coefficients(&xd, &yd)?;
    let (b1, _) = coefficients(&xd, &ylag)?;
    let fit0 = fitted(&xd, &b0);
    let fit1 = fitted(&xd, &b1);
    let e0: Vec<f64> = yd.iter().zip(&fit0).map(|(a, b)| a - b).collect();
    let e1: Vec<f64> = ylag.iter().zip(&fit1).map(|(a, b)| a - b).collect();
    let jac = Jacobian::new(w, Method::Full)?;
    let tf = t as f64;
    let objective = |rho: f64| {
        let ss: f64 = e0.iter().zip(&e1).map(|(a, b)| (a - rho * b).powi(2)).sum();
        nt / 2.0 * (ss / nt).ln() - tf * jac.log_det(rho)
    };
    let m = minimize_bounded(objective, -0.99, 0.99, opts.epsilon, 500)?;
    let rho = m.x;
    let logll = -m.fun - nt / 2.0 * ln2pi() - nt / 2.0;
    let b: Vec<f64> = b0.iter().zip(&b1).map(|(a, c)| a - rho * c).collect();
    let u: Vec<f64> = e0.iter().zip(&e1).map(|(a, c)| a - rho * c).collect();
    let predy: Vec<f64> = yd.iter().zip(&u).map(|(a, c)| a - c).collect();
    let xb = fitted(&xd, &b);
    let predy_e = power_expansion_nt(w, &xb, rho, opts.epsilon, n)?;
    let e_pred: Vec<f64> = yd.iter().zip(&predy_e).map(|(a, c)| a - c).collect();
    let sig2 = dot(&u, &u) / nt;

    let (_, wai) = inverses(w, rho)?;
    let (tr1, tr2, tr3) = traces(&wai);
    let wpredy = dense_nt(&wai, &xb, n);
    // spreg: xTwpy = x.T @ wpredy with the raw regressors.
    let xtwpy: Vec<f64> = x.iter().map(|c| dot(c, &wpredy)).collect();
    let waitwai = Mat::from_fn(n, n, |i, j| (0..n).map(|r| wai[(r, i)] * wai[(r, j)]).sum());
    let wpywpy = dot(&xb, &dense_nt(&waitwai, &xb, n));
    let k = x.len();
    let xtx = Mat::from_fn(k, k, |i, j| dot(&xd[i], &xd[j]));
    let mut info = Mat::<f64>::zeros(k + 2, k + 2);
    for i in 0..k {
        for j in 0..k {
            info[(i, j)] = xtx[(i, j)] / sig2;
        }
        info[(i, k)] = xtwpy[i] / sig2;
        info[(k, i)] = xtwpy[i] / sig2;
    }
    info[(k, k)] = tf * (tr2 + tr3) + wpywpy / sig2;
    info[(k, k + 1)] = tf * tr1 / sig2;
    info[(k + 1, k)] = tf * tr1 / sig2;
    info[(k + 1, k + 1)] = nt / (2.0 * sig2 * sig2);
    let vm1 = inverse(&info)?;
    let vm = Mat::from_fn(k + 1, k + 1, |i, j| vm1[(i, j)]);
    let mut betas = b;
    betas.push(rho);
    let std_err: Vec<f64> = (0..=k).map(|i| vm[(i, i)].sqrt()).collect();
    let kc = (k + 1) as f64;

    // Fixed effects: mean_t(y) - mean_t(X) b - rho W mean_t(y), on the raw data.
    let area_mean = |v: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| (0..t).map(|s| v[s * n + i]).sum::<f64>() / tf)
            .collect()
    };
    let ym = area_mean(y);
    let wym = w.lag(&ym)?;
    let xm: Vec<Vec<f64>> = x.iter().map(|c| area_mean(c)).collect();
    let mu: Vec<f64> = (0..n)
        .map(|i| ym[i] - (0..k).map(|j| xm[j][i] * betas[j]).sum::<f64>() - rho * wym[i])
        .collect();

    let mut out = Entries::new();
    out.insert("rho", vec![rho]);
    out.insert("vm", row_major(&vm));
    out.insert("vm1", row_major(&vm1));
    out.insert("z_stat", z_stats(&betas, &std_err));
    out.insert("std_err", std_err);
    out.insert("logll", vec![logll]);
    out.insert("aic", vec![2.0 * kc - 2.0 * logll]);
    out.insert("schwarz", vec![kc * nt.ln() - 2.0 * logll]);
    out.insert("sig2", vec![sig2]);
    out.insert("iterations", vec![m.nfev as f64]);
    out.insert("converged", vec![f64::from(u8::from(m.converged()))]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("predy_e", predy_e);
    out.insert("e_pred", e_pred);
    out.insert("mu_i", mu);
    for method in impact_methods(opts.impacts) {
        let mm = multipliers(w, rho, method)?;
        let key = match *method {
            "simple" => "mult_simple",
            "full" => "mult_full",
            _ => "mult_power",
        };
        out.insert(key, vec![mm.adi, mm.aii, mm.ati]);
    }
    out.insert("betas", betas);
    Ok(out)
}

/// `ML_ErrorFE`: `y = X b + mu_i + u`, `u = lambda W u + e`, on the demeaned panel.
pub fn ml_error_fe(y: &[f64], x: &[Vec<f64>], opts: &Options) -> Result<Entries, String> {
    let (n, t) = check(y, x, opts)?;
    let w = opts.w;
    let nt = (n * t) as f64;
    let tf = t as f64;
    let yd = demean(y, n, t);
    let xd: Vec<Vec<f64>> = x.iter().map(|c| demean(c, n, t)).collect();
    if let Some(j) = xd.iter().position(|c| c.iter().all(|v| v.abs() < 1e-12)) {
        return Err(format!(
            "x column {j} is time-invariant; fixed effects absorb it"
        ));
    }
    let ylag = lag_nt(w, &yd, n)?;
    let xlag: Vec<Vec<f64>> = xd
        .iter()
        .map(|c| lag_nt(w, c, n))
        .collect::<Result<_, _>>()?;
    let jac = Jacobian::new(w, Method::Full)?;
    let star = |lam: f64| -> (Vec<f64>, Vec<Vec<f64>>) {
        let ys = yd.iter().zip(&ylag).map(|(a, b)| a - lam * b).collect();
        let xs = xd
            .iter()
            .zip(&xlag)
            .map(|(c, l)| c.iter().zip(l).map(|(a, b)| a - lam * b).collect())
            .collect();
        (ys, xs)
    };
    let objective = |lam: f64| {
        let (ys, xs) = star(lam);
        let Ok((b, _)) = coefficients(&xs, &ys) else {
            return f64::NAN;
        };
        let xsys: Vec<f64> = xs.iter().map(|c| dot(c, &ys)).collect();
        let ee = dot(&ys, &ys) - dot(&xsys, &b);
        nt / 2.0 * ee.ln() - tf * jac.log_det(lam)
    };
    let m = minimize_bounded(objective, -1.0, 1.0, opts.epsilon, 500)?;
    let lam = m.x;
    let logll = -m.fun - nt / 2.0 * ln2pi() - nt / 2.0 + nt / 2.0 * nt.ln();
    let (ys, xs) = star(lam);
    let (b, xsxsi) = coefficients(&xs, &ys)?;
    let u: Vec<f64> = yd.iter().zip(fitted(&xd, &b)).map(|(a, f)| a - f).collect();
    let predy: Vec<f64> = yd.iter().zip(&u).map(|(a, c)| a - c).collect();
    let wu = lag_nt(w, &u, n)?;
    let e_filtered: Vec<f64> = u.iter().zip(&wu).map(|(a, c)| a - lam * c).collect();
    let sig2 = dot(&e_filtered, &e_filtered) / nt;
    let k = x.len();
    let (_, wai) = inverses(w, lam)?;
    let (tr1, tr2, tr3) = traces(&wai);
    let mut v = Mat::<f64>::zeros(2, 2);
    v[(0, 0)] = tf * (tr2 + tr3);
    v[(0, 1)] = tf * tr1 / sig2;
    v[(1, 0)] = tf * tr1 / sig2;
    v[(1, 1)] = nt / (2.0 * sig2 * sig2);
    let vm1 = inverse(&v)?;
    let vm = Mat::from_fn(k + 1, k + 1, |i, j| {
        if i < k && j < k {
            sig2 * xsxsi[(i, j)]
        } else if i == k && j == k {
            vm1[(0, 0)]
        } else {
            0.0
        }
    });
    let mut betas = b;
    betas.push(lam);
    let std_err: Vec<f64> = (0..=k).map(|i| vm[(i, i)].sqrt()).collect();
    // spreg: k (and the information criteria) do not count lambda.
    let kf = k as f64;
    let raw_u: Vec<f64> = y
        .iter()
        .zip(fitted(x, &betas[..k]))
        .map(|(a, f)| a - f)
        .collect();
    let mu: Vec<f64> = (0..n)
        .map(|i| (0..t).map(|s| raw_u[s * n + i]).sum::<f64>() / tf)
        .collect();

    let mut out = Entries::new();
    out.insert("lam", vec![lam]);
    out.insert("vm", row_major(&vm));
    out.insert("vm1", row_major(&vm1));
    out.insert("z_stat", z_stats(&betas, &std_err));
    out.insert("std_err", std_err);
    out.insert("logll", vec![logll]);
    out.insert("aic", vec![2.0 * kf - 2.0 * logll]);
    out.insert("schwarz", vec![kf * nt.ln() - 2.0 * logll]);
    out.insert("sig2", vec![sig2]);
    out.insert("iterations", vec![m.nfev as f64]);
    out.insert("converged", vec![f64::from(u8::from(m.converged()))]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("e_filtered", e_filtered);
    out.insert("mu_i", mu);
    out.insert("betas", betas);
    Ok(out)
}

/// `ML_LagRE`: `y = rho W y + X b + mu_i + e` with random effects. `x` includes the constant.
/// spreg alternates a bounded Brent search for the variance ratio `phi` (quasi-demeaning
/// `v - (1 - phi) mean_t(v)`) with one for `rho`, until `phi` moves less than `1e-7`
/// (at most 100 rounds).
pub fn ml_lag_re(y: &[f64], x: &[Vec<f64>], opts: &Options) -> Result<Entries, String> {
    let (n, t) = check(y, x, opts)?;
    let w = opts.w;
    let nt = (n * t) as f64;
    let tf = t as f64;
    let nf = n as f64;
    let k = x.len();
    let jac = Jacobian::new(w, Method::Full)?;
    let (mut b, _) = coefficients(x, y)?;
    let (mut rho, mut phi) = (0.1, 0.1);
    let mut converge = 1.0;
    let mut rounds = 0;
    let mut state = None;
    while converge > 1e-7 && rounds < 100 {
        let phi_old = phi;
        let phi_objective = |p: f64| -> f64 {
            let yq = quasi_demean(y, n, t, p);
            let xq: Vec<Vec<f64>> = x.iter().map(|c| quasi_demean(c, n, t, p)).collect();
            let Ok(ylag) = lag_nt(w, &yq, n) else {
                return f64::NAN;
            };
            let xb = fitted(&xq, &b);
            let ss: f64 = (0..yq.len())
                .map(|r| (yq[r] - rho * ylag[r] - xb[r]).powi(2))
                .sum();
            nt / 2.0 * ss.ln() - nf / 2.0 * (p * p).ln()
        };
        phi = minimize_bounded(phi_objective, 1e-8, 0.999_999_99, opts.epsilon, 500)?.x;
        let yq = quasi_demean(y, n, t, phi);
        let xq: Vec<Vec<f64>> = x.iter().map(|c| quasi_demean(c, n, t, phi)).collect();
        let ylag = lag_nt(w, &yq, n)?;
        let (b0, _) = coefficients(&xq, &yq)?;
        let (b1, _) = coefficients(&xq, &ylag)?;
        let fit0 = fitted(&xq, &b0);
        let fit1 = fitted(&xq, &b1);
        let e0: Vec<f64> = yq.iter().zip(&fit0).map(|(a, c)| a - c).collect();
        let e1: Vec<f64> = ylag.iter().zip(&fit1).map(|(a, c)| a - c).collect();
        let objective = |r: f64| {
            let ss: f64 = e0.iter().zip(&e1).map(|(a, c)| (a - r * c).powi(2)).sum();
            nt / 2.0 * (ss / nt).ln() - tf * jac.log_det(r)
        };
        let m = minimize_bounded(objective, -0.99, 0.99, opts.epsilon, 500)?;
        rho = m.x;
        b = b0.iter().zip(&b1).map(|(a, c)| a - rho * c).collect();
        rounds += 1;
        converge = (phi_old - phi).abs();
        state = Some((m, yq, xq, e0, e1));
    }
    let (m, yq, xq, e0, e1) = state.ok_or("the random-effects iterations did not run")?;
    let logll = -m.fun + nf / 2.0 * (phi * phi).ln() - nt / 2.0 * ln2pi() - nt / 2.0;
    let u: Vec<f64> = e0.iter().zip(&e1).map(|(a, c)| a - rho * c).collect();
    let predy: Vec<f64> = yq.iter().zip(&u).map(|(a, c)| a - c).collect();
    let xb = fitted(&xq, &b);
    let predy_e = power_expansion_nt(w, &xb, rho, opts.epsilon, n)?;
    let e_pred: Vec<f64> = yq.iter().zip(&predy_e).map(|(a, c)| a - c).collect();
    let sig2 = dot(&u, &u) / nt;

    let (_, wai) = inverses(w, rho)?;
    let (tr1, tr2, tr3) = traces(&wai);
    let wpredy = dense_nt(&wai, &xb, n);
    let xtwpy: Vec<f64> = xq.iter().map(|c| dot(c, &wpredy)).collect();
    let waitwai = Mat::from_fn(n, n, |i, j| (0..n).map(|r| wai[(r, i)] * wai[(r, j)]).sum());
    let wpywpy = dot(&xb, &dense_nt(&waitwai, &xb, n));
    let xtx = Mat::from_fn(k, k, |i, j| dot(&xq[i], &xq[j]));
    // spreg's np.hstack((v1, v2, v3, v4)) of the columns below, kept as written (the phi / sigma^2
    // cross entries differ: -n / sig2 and -n / sig2^2).
    let (cr, cp, cs) = (k, k + 1, k + 2);
    let mut v = Mat::<f64>::zeros(k + 3, k + 3);
    for i in 0..k {
        for j in 0..k {
            v[(i, j)] = xtx[(i, j)] / sig2;
        }
        v[(cr, i)] = xtwpy[i] / sig2;
        v[(i, cr)] = xtwpy[i] / sig2;
    }
    v[(cr, cr)] = tf * (tr2 + tr3) + wpywpy / sig2;
    v[(cp, cr)] = -tr1 / sig2;
    v[(cs, cr)] = tf * tr1 / sig2;
    v[(cr, cp)] = -tr1 / sig2;
    v[(cp, cp)] = nf * (1.0 + 1.0 / (phi * phi));
    v[(cs, cp)] = -nf / sig2;
    v[(cr, cs)] = tf * tr1 / sig2;
    v[(cp, cs)] = -nf / (sig2 * sig2);
    v[(cs, cs)] = nt / (2.0 * sig2 * sig2);
    let vm1 = inverse(&v)?;
    let vm = Mat::from_fn(k + 2, k + 2, |i, j| vm1[(i, j)]);
    let mut betas = b;
    betas.extend([rho, phi]);
    let std_err: Vec<f64> = (0..k + 2).map(|i| vm[(i, i)].sqrt()).collect();
    let kc = (k + 2) as f64;

    let mut out = Entries::new();
    out.insert("rho", vec![rho]);
    out.insert("phi", vec![phi]);
    out.insert("vm", row_major(&vm));
    out.insert("vm1", row_major(&vm1));
    out.insert("z_stat", z_stats(&betas, &std_err));
    out.insert("std_err", std_err);
    out.insert("logll", vec![logll]);
    out.insert("aic", vec![2.0 * kc - 2.0 * logll]);
    out.insert("schwarz", vec![kc * nt.ln() - 2.0 * logll]);
    out.insert("sig2", vec![sig2]);
    out.insert("iterations", vec![rounds as f64]);
    out.insert("converged", vec![f64::from(u8::from(converge <= 1e-7))]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("predy_e", predy_e);
    out.insert("e_pred", e_pred);
    for method in impact_methods(opts.impacts) {
        let mm = multipliers(w, rho, method)?;
        let key = match *method {
            "simple" => "mult_simple",
            "full" => "mult_full",
            _ => "mult_power",
        };
        out.insert(key, vec![mm.adi, mm.aii, mm.ati]);
    }
    out.insert("betas", betas);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demeaning_removes_the_area_means() {
        // n = 2, t = 3, stacked by period.
        let v = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        assert_eq!(demean(&v, 2, 3), vec![-1.0, -10.0, 0.0, 0.0, 1.0, 10.0]);
    }
}
