//! Small dense LU solver shared by the local fits.

/// `LU` factorization with partial pivoting of a small dense matrix (LAPACK `gesv` route).
pub(crate) struct Lu {
    k: usize,
    a: Vec<f64>,
    perm: Vec<usize>,
}

impl Lu {
    /// Factors the row-major `k x k` matrix; an exactly zero or NaN pivot is an error.
    pub(crate) fn factor(mut a: Vec<f64>, k: usize) -> Result<Self, String> {
        let mut perm: Vec<usize> = (0..k).collect();
        for c in 0..k {
            let mut p = c;
            for r in c + 1..k {
                if a[r * k + c].abs() > a[p * k + c].abs() {
                    p = r;
                }
            }
            let pivot = a[p * k + c];
            if pivot == 0.0 || pivot.is_nan() {
                return Err("singular local design matrix".to_string());
            }
            if p != c {
                for j in 0..k {
                    a.swap(p * k + j, c * k + j);
                }
                perm.swap(p, c);
            }
            for r in c + 1..k {
                let f = a[r * k + c] / a[c * k + c];
                a[r * k + c] = f;
                for j in c + 1..k {
                    a[r * k + j] -= f * a[c * k + j];
                }
            }
        }
        Ok(Self { k, a, perm })
    }

    /// Solves `A x = b`.
    pub(crate) fn solve(&self, b: &[f64]) -> Vec<f64> {
        let k = self.k;
        let mut x: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        for r in 1..k {
            for c in 0..r {
                x[r] -= self.a[r * k + c] * x[c];
            }
        }
        for r in (0..k).rev() {
            for c in r + 1..k {
                x[r] -= self.a[r * k + c] * x[c];
            }
            x[r] /= self.a[r * k + r];
        }
        x
    }
}
