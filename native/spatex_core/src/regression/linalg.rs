//! Dense and sparse linear-algebra helpers shared by the regression estimators.
//!
//! Dense work goes through `faer` (pure Rust, no system BLAS). The weights matrix always stays
//! sparse: it is a [`Csr`], and only its products with vectors or thin dense matrices are formed.

use crate::graph::Csr;
use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, Side};

/// An `n x cols.len()` matrix from column vectors.
pub fn mat_from_cols(cols: &[Vec<f64>], n: usize) -> Mat<f64> {
    Mat::from_fn(n, cols.len(), |i, j| cols[j][i])
}

/// An `n x 1` matrix from a vector.
pub fn col_mat(v: &[f64]) -> Mat<f64> {
    Mat::from_fn(v.len(), 1, |i, _| v[i])
}

/// Column `j` of `m` as a vector.
pub fn column(m: &Mat<f64>, j: usize) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, j)]).collect()
}

/// Row-major flattening, the layout the NIF boundary uses for matrices.
pub fn row_major(m: &Mat<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)]);
        }
    }
    out
}

/// Sum of the diagonal.
pub fn trace(m: &Mat<f64>) -> f64 {
    (0..m.nrows().min(m.ncols())).map(|i| m[(i, i)]).sum()
}

/// Inverse by LU with partial pivoting (the LAPACK `getrf`/`getri` route numpy takes).
///
/// Errors when a pivot is exactly zero or the result is not finite.
pub fn inverse(a: &Mat<f64>) -> Result<Mat<f64>, String> {
    if a.nrows() != a.ncols() {
        return Err(format!(
            "cannot invert a {}x{} matrix",
            a.nrows(),
            a.ncols()
        ));
    }
    if a.nrows() == 0 {
        return Ok(Mat::zeros(0, 0));
    }
    let lu = a.partial_piv_lu();
    let u = lu.U();
    if (0..u.nrows()).any(|i| u[(i, i)] == 0.0 || u[(i, i)].is_nan()) {
        return Err("matrix is singular".to_string());
    }
    let inv = lu.inverse();
    if (0..inv.nrows()).any(|i| (0..inv.ncols()).any(|j| !inv[(i, j)].is_finite())) {
        return Err("matrix is singular".to_string());
    }
    Ok(inv)
}

/// Eigenvalues of a symmetric matrix, ascending.
pub fn symmetric_eigenvalues(a: &Mat<f64>) -> Result<Vec<f64>, String> {
    a.self_adjoint_eigenvalues(Side::Lower)
        .map_err(|e| format!("eigenvalue decomposition failed: {e:?}"))
}

/// Index range of row `i` in the CSR arrays.
pub fn row_range(w: &Csr, i: usize) -> std::ops::Range<usize> {
    w.offsets[i]..w.offsets[i + 1]
}

/// `W' v` for a CSR `W`: the spatial lag with the transposed matrix.
pub fn lag_transpose(w: &Csr, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; w.n];
    for (i, vi) in v.iter().enumerate().take(w.n) {
        for k in row_range(w, i) {
            out[w.neighbors[k] as usize] += w.weights[k] * vi;
        }
    }
    out
}

/// The weight `w_ij`, or zero when the edge is absent (rows are sorted by neighbor).
pub fn entry(w: &Csr, i: usize, j: usize) -> f64 {
    let range = w.offsets[i]..w.offsets[i + 1];
    match w.neighbors[range.clone()].binary_search(&(j as u32)) {
        Ok(pos) => w.weights[range.start + pos],
        Err(_) => 0.0,
    }
}

/// `tr((W' + W) W) = sum_ij w_ij^2 + sum_ij w_ij w_ji`.
pub fn trace_wtw_ww(w: &Csr) -> f64 {
    let mut total = 0.0;
    for i in 0..w.n {
        for k in row_range(w, i) {
            let wij = w.weights[k];
            total += wij * wij + wij * entry(w, w.neighbors[k] as usize, i);
        }
    }
    total
}

/// Result of an OLS fit on raw columns (normal equations, LU inverse).
pub struct Fit {
    pub betas: Vec<f64>,
    pub xtxi: Mat<f64>,
    pub predy: Vec<f64>,
    pub u: Vec<f64>,
    pub utu: f64,
}

/// OLS of `y` on the columns `cols` exactly as `spreg.ols.BaseOLS` computes it.
pub fn fit(y: &[f64], cols: &[Vec<f64>]) -> Result<Fit, String> {
    let n = y.len();
    let x = mat_from_cols(cols, n);
    let xtx = x.transpose() * &x;
    let xtxi = inverse(&xtx)?;
    let xty = x.transpose() * col_mat(y);
    let betas_m = &xtxi * xty;
    let predy_m = &x * &betas_m;
    let predy = column(&predy_m, 0);
    let u: Vec<f64> = y.iter().zip(&predy).map(|(a, b)| a - b).collect();
    let utu = u.iter().map(|v| v * v).sum();
    Ok(Fit {
        betas: column(&betas_m, 0),
        xtxi,
        predy,
        u,
        utu,
    })
}

/// Arithmetic mean.
pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrip_and_singular() {
        let a = Mat::from_fn(3, 3, |i, j| if i == j { 2.0 + i as f64 } else { 0.5 });
        let inv = inverse(&a).unwrap();
        let id = &a * &inv;
        for i in 0..3 {
            for j in 0..3 {
                assert!((id[(i, j)] - f64::from(u8::from(i == j))).abs() < 1e-14);
            }
        }
        let s = Mat::from_fn(2, 2, |i, _| (i + 1) as f64);
        assert!(inverse(&s).unwrap_err().contains("singular"));
    }

    #[test]
    fn eigenvalues_of_symmetric_matrix() {
        let a = Mat::from_fn(2, 2, |i, j| if i == j { 2.0 } else { 1.0 });
        let ev = symmetric_eigenvalues(&a).unwrap();
        assert!((ev[0] - 1.0).abs() < 1e-14 && (ev[1] - 3.0).abs() < 1e-14);
    }

    fn path3() -> Csr {
        Csr::from_arrays(3, &[0, 1, 1, 2], &[1, 0, 2, 1], &[1.0, 0.5, 2.0, 1.5]).unwrap()
    }

    #[test]
    fn transpose_lag_and_trace() {
        let w = path3();
        // W' v: column sums weighted by v.
        let lt = lag_transpose(&w, &[1.0, 2.0, 3.0]);
        assert_eq!(lt, vec![1.0, 5.5, 4.0]);
        // sum w^2 = 1 + .25 + 4 + 2.25 ; sum w_ij w_ji = 1*.5 + .5*1 + 2*1.5 + 1.5*2
        let expected = 7.5 + 7.0;
        assert!((trace_wtw_ww(&w) - expected).abs() < 1e-14);
        assert_eq!(entry(&w, 1, 2), 2.0);
        assert_eq!(entry(&w, 0, 2), 0.0);
    }

    #[test]
    fn fit_recovers_exact_line() {
        let x: Vec<f64> = (0..6).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v).collect();
        let f = fit(&y, &[vec![1.0; 6], x]).unwrap();
        assert!((f.betas[0] - 1.0).abs() < 1e-12 && (f.betas[1] - 2.0).abs() < 1e-12);
        assert!(f.utu < 1e-20);
    }
}
