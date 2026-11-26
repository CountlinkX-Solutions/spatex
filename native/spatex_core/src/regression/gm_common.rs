//! Building blocks shared by the 2SLS and GMM estimators (`spreg.utils`, spreg 1.9.1): a small
//! sparse square matrix for `W'W`-type products, spatial filters and lags, the power-series
//! inverse in both directions, `sp_att` (the reduced-form prediction) and the GMM moment
//! optimizer.
//!
//! Deviation (recorded in `test/reference/known_deviations.exs`): spreg minimizes the moment
//! objective with L-BFGS-B, which stops early (lambda accurate to ~1e-6). [`optim_moments`] finds
//! the exact minimizer of the same objective instead.

use faer::Mat;

use super::brent::minimize_bounded;
use super::linalg::{col_mat, column, inverse, lag_transpose, row_range};
use super::ml_common::power_expansion;
use crate::graph::Csr;

/// Square sparse matrix in CSR form with sorted, merged rows.
#[derive(Clone, Debug)]
pub struct Sp {
    pub n: usize,
    offsets: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl Sp {
    /// Builds a matrix from `(row, col, value)` triplets; duplicates are summed.
    pub fn from_triplets(n: usize, mut t: Vec<(usize, usize, f64)>) -> Sp {
        t.sort_by_key(|a| (a.0, a.1));
        let mut offsets = vec![0usize; n + 1];
        let mut cols: Vec<usize> = Vec::with_capacity(t.len());
        let mut vals: Vec<f64> = Vec::with_capacity(t.len());
        let mut last: Option<(usize, usize)> = None;
        for (i, j, v) in t {
            if last == Some((i, j)) {
                *vals.last_mut().expect("merged entry exists") += v;
            } else {
                cols.push(j);
                vals.push(v);
                offsets[i + 1] += 1;
                last = Some((i, j));
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        Sp {
            n,
            offsets,
            cols,
            vals,
        }
    }

    /// The weights as given (explicit zeros dropped).
    pub fn from_csr(w: &Csr) -> Sp {
        let mut t = Vec::with_capacity(w.nnz());
        for i in 0..w.n {
            for k in row_range(w, i) {
                if w.weights[k] != 0.0 {
                    t.push((i, w.neighbors[k] as usize, w.weights[k]));
                }
            }
        }
        Sp::from_triplets(w.n, t)
    }

    /// The `n x n` identity times `scale`.
    pub fn scaled_identity(n: usize, scale: f64) -> Sp {
        Sp::from_triplets(n, (0..n).map(|i| (i, i, scale)).collect())
    }

    fn row(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i]..self.offsets[i + 1]
    }

    /// `M v`.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| self.row(i).map(|k| self.vals[k] * v[self.cols[k]]).sum())
            .collect()
    }

    /// `M'`.
    pub fn transpose(&self) -> Sp {
        let mut t = Vec::with_capacity(self.vals.len());
        for i in 0..self.n {
            for k in self.row(i) {
                t.push((self.cols[k], i, self.vals[k]));
            }
        }
        Sp::from_triplets(self.n, t)
    }

    /// The entry `(i, j)`, zero when absent.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let r = self.row(i);
        match self.cols[r.clone()].binary_search(&j) {
            Ok(p) => self.vals[r.start + p],
            Err(_) => 0.0,
        }
    }

    /// Every stored entry `(row, col, value)`.
    pub fn entries(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.n).flat_map(move |i| self.row(i).map(move |k| (i, self.cols[k], self.vals[k])))
    }

    /// The diagonal.
    pub fn diag(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.get(i, i)).collect()
    }

    /// `a * self + b * other`.
    pub fn combine(&self, a: f64, other: &Sp, b: f64) -> Sp {
        let mut t = Vec::with_capacity(self.vals.len() + other.vals.len());
        for i in 0..self.n {
            for k in self.row(i) {
                t.push((i, self.cols[k], a * self.vals[k]));
            }
            for k in other.row(i) {
                t.push((i, other.cols[k], b * other.vals[k]));
            }
        }
        Sp::from_triplets(self.n, t)
    }

    /// `f * self`.
    pub fn scaled(&self, f: f64) -> Sp {
        Sp {
            n: self.n,
            offsets: self.offsets.clone(),
            cols: self.cols.clone(),
            vals: self.vals.iter().map(|v| v * f).collect(),
        }
    }

    /// `self * other`.
    pub fn matmul(&self, other: &Sp) -> Sp {
        let n = self.n;
        let mut acc = vec![0.0; n];
        let mut seen = vec![false; n];
        let mut touched: Vec<usize> = Vec::new();
        let mut t = Vec::new();
        for i in 0..n {
            for k in self.row(i) {
                let a = self.vals[k];
                let m = self.cols[k];
                for p in other.row(m) {
                    let j = other.cols[p];
                    if !seen[j] {
                        seen[j] = true;
                        touched.push(j);
                    }
                    acc[j] += a * other.vals[p];
                }
            }
            for &j in &touched {
                t.push((i, j, acc[j]));
                acc[j] = 0.0;
                seen[j] = false;
            }
            touched.clear();
        }
        Sp::from_triplets(n, t)
    }

    /// `tr(M E N E)` for the diagonal `E = diag(e)`: `sum_ij M_ij e_j N_ji e_i`.
    pub fn trace_prod(&self, other: &Sp, e: &[f64]) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n {
            for k in self.row(i) {
                let j = self.cols[k];
                total += self.vals[k] * e[j] * other.get(j, i) * e[i];
            }
        }
        total
    }
}

/// Dot product.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// `v - lambda W v` (`get_spFilter`).
pub fn filter(w: &Csr, v: &[f64], lam: f64) -> Result<Vec<f64>, String> {
    let wv = w.lag(v)?;
    Ok(v.iter().zip(&wv).map(|(a, b)| a - lam * b).collect())
}

/// Spatial filter of every column.
pub fn filter_cols(w: &Csr, cols: &[Vec<f64>], lam: f64) -> Result<Vec<Vec<f64>>, String> {
    cols.iter().map(|c| filter(w, c, lam)).collect()
}

/// `W` applied `lags` times in turn to every column: `[W c_1, .., W c_p, W^2 c_1, ..]`
/// (`get_lags`).
pub fn get_lags(w: &Csr, cols: &[Vec<f64>], lags: usize) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(cols.len() * lags);
    let mut current: Vec<Vec<f64>> = cols.to_vec();
    for _ in 0..lags {
        current = current
            .iter()
            .map(|c| w.lag(c))
            .collect::<Result<Vec<_>, _>>()?;
        out.extend(current.iter().cloned());
    }
    Ok(out)
}

/// `(yend, q)` columns.
pub type Endog = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// The endogenous variables and instruments of a spatial lag model (`set_endog`): returns
/// `(yend, q)` with `W y` as the last endogenous column. `x_nc` excludes the constant.
pub fn set_endog(
    w: &Csr,
    y: &[f64],
    x_nc: &[Vec<f64>],
    yend: &[Vec<f64>],
    q: &[Vec<f64>],
    w_lags: usize,
    lag_q: bool,
) -> Result<Endog, String> {
    if w_lags == 0 {
        return Err("w_lags must be at least 1".to_string());
    }
    let yl = w.lag(y)?;
    if yend.is_empty() {
        return Ok((vec![yl], get_lags(w, x_nc, w_lags)?));
    }
    let mut base: Vec<Vec<f64>> = x_nc.to_vec();
    if lag_q {
        base.extend(q.iter().cloned());
    }
    let mut q2 = q.to_vec();
    q2.extend(get_lags(w, &base, w_lags)?);
    let mut yend2 = yend.to_vec();
    yend2.push(yl);
    Ok((yend2, q2))
}

/// [`set_endog`] for a model with `slx` spatially lagged regressors (spreg's `slx_lags`): `x_base`
/// holds the original non-constant columns. Without `yend` the instruments are their lags
/// `slx + 1 ..= slx + w_lags`; with `yend` they are `q` and the lags `1..=w_lags` of
/// `W^(slx + 1) x_base` (and of `q` with `lag_q`), as spreg's `get_lags_split` path builds them.
#[allow(clippy::too_many_arguments)]
pub fn set_endog_slx(
    w: &Csr,
    y: &[f64],
    x_base: &[Vec<f64>],
    yend: &[Vec<f64>],
    q: &[Vec<f64>],
    w_lags: usize,
    lag_q: bool,
    slx: usize,
) -> Result<Endog, String> {
    if w_lags == 0 {
        return Err("w_lags must be at least 1".to_string());
    }
    let yl = w.lag(y)?;
    let kx = x_base.len();
    if yend.is_empty() {
        let lags = get_lags(w, x_base, slx + w_lags)?;
        return Ok((vec![yl], lags[slx * kx..].to_vec()));
    }
    let mut base: Vec<Vec<f64>> = get_lags(w, x_base, slx + 1)?[slx * kx..].to_vec();
    if lag_q {
        base.extend(q.iter().cloned());
    }
    let mut q2 = q.to_vec();
    q2.extend(get_lags(w, &base, w_lags)?);
    let mut yend2 = yend.to_vec();
    yend2.push(yl);
    Ok((yend2, q2))
}

/// `(I - scalar W')^{-1} data`: the post-multiplied power series `data' (I - scalar W)^{-1}`
/// transposed (`inverse_prod(.., post_multiply=True)`), with the same stopping rule as
/// [`power_expansion`].
pub fn power_expansion_t(
    w: &Csr,
    data: &[f64],
    scalar: f64,
    threshold: f64,
) -> Result<Vec<f64>, String> {
    let mut total = data.to_vec();
    let mut inc = data.to_vec();
    let mut test = 1.0e7_f64;
    let mut count = 1u64;
    while test > threshold && count <= 10_000_000 {
        inc = lag_transpose(w, &inc)
            .into_iter()
            .map(|v| v * scalar)
            .collect();
        for (t, d) in total.iter_mut().zip(&inc) {
            *t += d;
        }
        let test_old = test;
        test = inc.iter().map(|v| v * v).sum::<f64>().sqrt();
        if test > test_old {
            return Err("power expansion will not converge, check model specification and that weights are less than 1".to_string());
        }
        count += 1;
    }
    Ok(total)
}

/// Threshold of `utils.inverse_prod`.
pub const INVERSE_THRESHOLD: f64 = 1e-10;

/// Reduced-form prediction of a model with a spatial lag (`sp_att`).
pub struct SpAtt {
    pub predy_e: Vec<f64>,
    pub e_pred: Vec<f64>,
    /// `|rho| >= 1`: the prediction was not computed (zeros) and spreg warns.
    pub out_of_bounds: bool,
}

/// `predy_e = (I - rho W)^{-1} (predy - rho W y)` for `|rho| < 1`.
pub fn sp_att(
    w: &Csr,
    y: &[f64],
    predy: &[f64],
    wy: &[f64],
    rho: f64,
    hard_bound: bool,
) -> Result<SpAtt, String> {
    if rho.abs() < 1.0 {
        let xb: Vec<f64> = predy.iter().zip(wy).map(|(p, l)| p - rho * l).collect();
        let predy_e = power_expansion(w, &xb, rho, INVERSE_THRESHOLD)?;
        let e_pred = y.iter().zip(&predy_e).map(|(a, b)| a - b).collect();
        Ok(SpAtt {
            predy_e,
            e_pred,
            out_of_bounds: false,
        })
    } else if hard_bound {
        Err("Spatial autoregressive parameter is outside the maximum/minimum bounds.".to_string())
    } else {
        Ok(SpAtt {
            predy_e: vec![0.0; y.len()],
            e_pred: vec![0.0; y.len()],
            out_of_bounds: true,
        })
    }
}

/// The moment equations `g = G v` of a GMM spatial error model: `G` is `m x m`
/// (`m = 2`: `v = [lambda, lambda^2]`; `m = 3`: `v = [lambda, lambda^2, sigma^2]`).
pub struct Moments {
    pub g_mat: Mat<f64>,
    pub g: Vec<f64>,
}

/// Lower-triangular Cholesky factor of a symmetric positive definite matrix.
pub(super) fn cholesky(a: &Mat<f64>) -> Result<Mat<f64>, String> {
    let m = a.nrows();
    let mut l = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if s <= 0.0 || s.is_nan() {
                    return Err("weighting matrix is not positive definite".to_string());
                }
                l[(i, i)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }
    Ok(l)
}

/// Lower bound, upper bound and step of the lambda grid.
const GRID_MAX: f64 = 0.99;
const GRID_STEP: f64 = 0.01;
const GRID_POINTS: usize = 199;

/// Minimizes `f` on `[-0.99, 0.99]`: walk downhill on a 0.01 grid from 0 to the first local
/// minimum (the basin L-BFGS-B reaches from its start at 0) and refine it with bounded Brent.
fn minimize_lambda<F: Fn(f64) -> f64>(f: F) -> f64 {
    minimize_lambda_from(f, 0.0)
}

/// [`minimize_lambda`] walking downhill from the grid point nearest to `start` (spreg's
/// L-BFGS-B start value, e.g. the first-step lambda of the panel GM estimator).
pub(super) fn minimize_lambda_from<F: Fn(f64) -> f64>(f: F, start: f64) -> f64 {
    let mid = GRID_POINTS / 2;
    let x_at = |i: usize| (i as f64 - mid as f64) * GRID_STEP;
    let vals: Vec<f64> = (0..GRID_POINTS).map(|i| f(x_at(i))).collect();
    if vals.iter().all(|v| v.is_nan()) {
        return 0.0;
    }
    let from = (mid as f64 + (start / GRID_STEP).round()).clamp(0.0, (GRID_POINTS - 1) as f64);
    let mut i = from as usize;
    loop {
        let left = if i > 0 { vals[i - 1] } else { f64::INFINITY };
        let right = if i + 1 < GRID_POINTS {
            vals[i + 1]
        } else {
            f64::INFINITY
        };
        if left < vals[i] && left <= right {
            i -= 1;
        } else if right < vals[i] {
            i += 1;
        } else {
            break;
        }
    }
    let lo = x_at(i.saturating_sub(1)).max(-GRID_MAX);
    let hi = x_at((i + 1).min(GRID_POINTS - 1)).min(GRID_MAX);
    let best =
        minimize_bounded(&f, lo, hi, 1e-12, 500).map_or((x_at(i), vals[i]), |m| (m.x, m.fun));
    // A grid minimum on the boundary may also be the bound itself.
    if i == 0 || i == GRID_POINTS - 1 {
        let bound = x_at(i);
        if vals[i] <= best.1 || best.1.is_nan() {
            return bound;
        }
    }
    if best.1 <= vals[i] {
        best.0
    } else {
        x_at(i)
    }
}

/// Outcome of [`optim_moments`].
#[derive(Clone, Copy, Debug)]
pub struct Optimum {
    pub lam: f64,
    /// The moment scale was zero (or not finite): spreg's objective is NaN and L-BFGS-B returns
    /// its start; Spatex returns zero and flags it.
    pub degenerate: bool,
}

/// The `lambda` minimizing `|g - G v|^2` over `[-0.99, 0.99]` (`utils.optim_moments`), after
/// optionally weighting by `chol(inv(vc))'`.
pub fn optim_moments(mom: &Moments, vc: Option<&Mat<f64>>) -> Result<Optimum, String> {
    let m = mom.g.len();
    let mut gm = mom.g_mat.clone();
    let mut g = col_mat(&mom.g);
    if let Some(vc) = vc {
        if (0..vc.nrows()).any(|i| (0..vc.ncols()).any(|j| vc[(i, j)] != 0.0)) {
            let ec = cholesky(&inverse(vc)?)?.transpose().to_owned();
            gm = &ec * &gm;
            g = &ec * &g;
        }
    }
    let gv = column(&g, 0);
    let all: Vec<f64> = (0..m)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| gm[(i, j)])
        .chain(gv.iter().copied())
        .collect();
    let scale = all.iter().copied().fold(f64::INFINITY, f64::min);
    if all.iter().any(|v| !v.is_finite()) || scale == 0.0 {
        return Ok(Optimum {
            lam: 0.0,
            degenerate: true,
        });
    }
    let lam = if m == 2 {
        minimize_lambda(|lam| {
            (0..2)
                .map(|r| {
                    let e = gv[r] - gm[(r, 0)] * lam - gm[(r, 1)] * lam * lam;
                    e * e
                })
                .sum()
        })
    } else if m == 3 {
        let c: Vec<f64> = (0..3).map(|r| gm[(r, 2)]).collect();
        let cc = dot(&c, &c);
        minimize_lambda(|lam| {
            let r: Vec<f64> = (0..3)
                .map(|i| gv[i] - gm[(i, 0)] * lam - gm[(i, 1)] * lam * lam)
                .collect();
            let s2 = (dot(&c, &r) / cc).max(0.0);
            r.iter()
                .zip(&c)
                .map(|(ri, ci)| (ri - ci * s2) * (ri - ci * s2))
                .sum()
        })
    } else {
        return Err(format!("unsupported number of moment equations: {m}"));
    };
    Ok(Optimum {
        lam,
        degenerate: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path3() -> Csr {
        Csr::from_arrays(3, &[0, 1, 1, 2], &[1, 0, 2, 1], &[1.0, 0.5, 0.5, 1.0]).unwrap()
    }

    #[test]
    fn sparse_products_traces_and_transpose() {
        let w = Sp::from_csr(&path3());
        let wt = w.transpose();
        let wtw = wt.matmul(&w);
        // W'W for rows [0,1,0], [.5,0,.5], [0,1,0]: diag = [.25, 2, .25]
        assert_eq!(wtw.diag(), vec![0.25, 2.0, 0.25]);
        assert_eq!(wtw.get(0, 2), 0.25);
        assert_eq!(w.get(1, 0), 0.5);
        assert_eq!(w.get(0, 2), 0.0);
        assert_eq!(w.matvec(&[1.0, 2.0, 3.0]), vec![2.0, 2.0, 2.0]);
        let a = wtw.combine(1.0, &Sp::scaled_identity(3, 1.0), -2.0);
        assert_eq!(a.get(1, 1), 2.0 - 2.0);
        // tr(W E W E) with E = I is tr(W W) = sum_ij w_ij w_ji = 0.5 + 0.5 + 0.5 + 0.5
        let ones = [1.0; 3];
        assert!((w.trace_prod(&w, &ones) - 2.0).abs() < 1e-14);
        let e = [1.0, 2.0, 3.0];
        // sum_ij w_ij e_j w_ji e_i: (0,1): 1*2*.5*1; (1,0): .5*1*1*2; (1,2): .5*3*1*2; (2,1): 1*2*.5*3
        assert!((w.trace_prod(&w, &e) - (1.0 + 1.0 + 3.0 + 3.0)).abs() < 1e-14);
    }

    #[test]
    fn lags_and_instruments() {
        let w = path3();
        let x = vec![vec![1.0, 2.0, 3.0]];
        let lags = get_lags(&w, &x, 2).unwrap();
        assert_eq!(lags[0], vec![2.0, 2.0, 2.0]);
        assert_eq!(lags[1], vec![2.0, 2.0, 2.0]);
        let (yend, q) = set_endog(&w, &[1.0, 2.0, 3.0], &x, &[], &[], 1, true).unwrap();
        assert_eq!((yend.len(), q.len()), (1, 1));
        let (yend, q) = set_endog(
            &w,
            &[1.0, 2.0, 3.0],
            &x,
            &[vec![0.0, 1.0, 0.0]],
            &[vec![1.0, 0.0, 1.0]],
            1,
            false,
        )
        .unwrap();
        // user instruments first, then the lag of X only (lag_q is false)
        assert_eq!((yend.len(), q.len()), (2, 2));
        assert_eq!(q[0], vec![1.0, 0.0, 1.0]);
        assert!(set_endog(&w, &[1.0, 2.0, 3.0], &x, &[], &[], 0, true).is_err());
    }

    #[test]
    fn post_multiplied_series_inverts_the_transpose() {
        let w = path3();
        let lam = 0.3;
        let d = [1.0, -2.0, 0.5];
        let r = power_expansion_t(&w, &d, lam, 1e-14).unwrap();
        // (I - lam W') r = d
        let wtr = lag_transpose(&w, &r);
        for i in 0..3 {
            assert!((r[i] - lam * wtr[i] - d[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn sp_att_branches() {
        let w = path3();
        let y = [1.0, 2.0, 3.0];
        let wy = w.lag(&y).unwrap();
        let ok = sp_att(&w, &y, &y, &wy, 0.0, false).unwrap();
        assert_eq!(ok.predy_e, y.to_vec());
        let oob = sp_att(&w, &y, &y, &wy, 1.2, false).unwrap();
        assert!(oob.out_of_bounds && oob.predy_e == vec![0.0; 3]);
        assert!(sp_att(&w, &y, &y, &wy, 1.2, true).is_err());
    }

    fn two_eq(lam: f64) -> Moments {
        // g = G [lam, lam^2] exactly at `lam`
        let g_mat = Mat::from_fn(2, 2, |i, j| [[1.0, 0.5], [0.3, 2.0]][i][j]);
        let g = (0..2)
            .map(|i| g_mat[(i, 0)] * lam + g_mat[(i, 1)] * lam * lam)
            .collect();
        Moments { g_mat, g }
    }

    #[test]
    fn optimizer_recovers_exact_roots() {
        for lam in [-0.7, -0.2, 0.1, 0.35, 0.8] {
            let o = optim_moments(&two_eq(lam), None).unwrap();
            assert!((o.lam - lam).abs() < 1e-7, "{lam}: {}", o.lam);
            assert!(!o.degenerate);
        }
        // a positive-definite weighting does not move an exact solution
        let vc = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 0.5 });
        let o = optim_moments(&two_eq(0.4), Some(&vc)).unwrap();
        assert!((o.lam - 0.4).abs() < 1e-7);
    }

    #[test]
    fn three_equation_profile_and_bounds() {
        // g = G [lam, lam^2, s2] at lam = 0.25, s2 = 3
        let g_mat = Mat::from_fn(3, 3, |i, j| {
            [[2.0, -1.0, 1.0], [0.5, -0.3, 2.0], [1.0, 0.2, 0.0]][i][j]
        });
        let v = [0.25, 0.0625, 3.0];
        let g: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| g_mat[(i, j)] * v[j]).sum())
            .collect();
        let o = optim_moments(&Moments { g_mat, g }, None).unwrap();
        assert!((o.lam - 0.25).abs() < 1e-7);
        // a minimum beyond the bound returns the bound
        let g_mat = Mat::from_fn(2, 2, |i, j| f64::from(u8::from(i == j)) + 1.0);
        let o = optim_moments(
            &Moments {
                g_mat,
                g: vec![5.0, 5.0],
            },
            None,
        )
        .unwrap();
        assert!((o.lam - 0.99).abs() < 1e-12);
    }

    #[test]
    fn zero_scale_is_flagged() {
        let m = Moments {
            g_mat: Mat::from_fn(2, 2, |i, j| f64::from(u8::from(i == 0 && j == 0))),
            g: vec![0.0, 1.0],
        };
        let o = optim_moments(&m, None).unwrap();
        assert!(o.degenerate && o.lam == 0.0);
    }
}
