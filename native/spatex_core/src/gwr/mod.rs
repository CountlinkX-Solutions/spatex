//! Geographically weighted regression (Gaussian, Poisson and Binomial), bandwidth selection and
//! multiscale GWR (MGWR), after `mgwr` 2.2.1.
//!
//! The hat matrix `S` and the kernel weight matrix are never materialized: for every focal
//! location the local weights are recomputed on the fly, and the diagonal of `S`, the row sums
//! `sum_j S_ij^2` (so `tr(S'S)`) and the covariance diagonal are accumulated row by row. Memory
//! is `O(n k)` and a full fit costs `O(n^2 k^2)` time, so a fit is feasible for tens of
//! thousands of observations but a bandwidth search (a few dozen fits) is the practical limit.
//!
//! Poisson and Binomial models are fitted by IWLS at every location ([`family`]). MGWR
//! ([`mgwr`]) backfits one bandwidth per covariate; its inference chunks the per-covariate hat
//! matrices by columns, so memory is `O(n * chunk * k)` while time is `O(iterations * k * n^3)`.
//!
//! Locations are processed in parallel with rayon, but every scalar reduction runs sequentially
//! in index order, so results do not depend on the thread count.

pub mod collinearity;
pub mod family;
pub mod fit;
pub mod kernel;
pub(crate) mod lu;
pub mod mgwr;
pub mod predict;
pub mod select;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_glm;
#[cfg(test)]
mod tests_predict;
#[cfg(test)]
mod tests_search;

pub use collinearity::{local_collinearity, local_collinearity_mgwr, Collinearity};
pub use family::Family;
pub use fit::{fit_full, fit_full_with, fit_lite, Criteria, Fit, Lite};
pub use kernel::Kernel;
pub use mgwr::{fit_mgwr, Inference, MgwrFit, MgwrOptions};
pub use predict::{predict, Exog, Prediction};
pub use select::{
    equal_interval, golden_section, select_bandwidth, Criterion, Search, SelectOptions, Selection,
};

/// A GWR model specification: data and kernel settings, shared by fitting and selection.
///
/// `x` is row-major `n x k` and already contains the constant column, if any.
#[derive(Debug, Clone, Copy)]
pub struct Model<'a> {
    /// Interleaved coordinates `x0, y0, x1, y1, ...` (lon, lat in degrees when `spherical`).
    pub coords: &'a [f64],
    pub y: &'a [f64],
    pub x: &'a [f64],
    pub n: usize,
    pub k: usize,
    pub kernel: Kernel,
    pub fixed: bool,
    pub spherical: bool,
    pub family: Family,
    /// Poisson exposure offsets (empty means ones); ignored by the other families.
    pub offset: &'a [f64],
    /// IWLS convergence tolerance and iteration cap (GLM families).
    pub iwls_tol: f64,
    pub iwls_max_iter: usize,
}

impl<'a> Model<'a> {
    /// Validates the shapes and values and builds the model.
    pub fn new(
        coords: &'a [f64],
        y: &'a [f64],
        x: &'a [f64],
        k: usize,
        kernel: Kernel,
        fixed: bool,
        spherical: bool,
    ) -> Result<Self, String> {
        let n = y.len();
        if n == 0 {
            return Err("GWR needs at least one observation".to_string());
        }
        if k == 0 {
            return Err("GWR needs at least one regressor".to_string());
        }
        if coords.len() != 2 * n {
            return Err(format!(
                "coords has {} values, expected {} (2 per observation)",
                coords.len(),
                2 * n
            ));
        }
        if x.len() != n * k {
            return Err(format!(
                "x has {} values, expected {n} rows x {k} columns",
                x.len()
            ));
        }
        if n <= k {
            return Err(format!(
                "GWR needs more observations than regressors (n = {n}, k = {k})"
            ));
        }
        Ok(Self {
            coords,
            y,
            x,
            n,
            k,
            kernel,
            fixed,
            spherical,
            family: Family::Gaussian,
            offset: &[],
            iwls_tol: 1.0e-5,
            iwls_max_iter: 20,
        })
    }

    /// Switches the model to a GLM family, validating the response and the offset.
    ///
    /// Poisson needs `y >= 0` and a strictly positive offset (empty means ones); Binomial needs
    /// `y` in `{0, 1}` and ignores the offset. `tol` and `max_iter` are the IWLS settings
    /// (mgwr's `fit` defaults are `1e-5` and `20`).
    pub fn with_family(
        mut self,
        family: Family,
        offset: &'a [f64],
        tol: f64,
        max_iter: usize,
    ) -> Result<Self, String> {
        if !(tol > 0.0 && tol.is_finite()) {
            return Err("iwls tolerance must be positive".to_string());
        }
        if max_iter == 0 {
            return Err("iwls max_iter must be at least 1".to_string());
        }
        match family {
            Family::Gaussian => {}
            Family::Poisson => {
                if self.y.iter().any(|v| !(*v >= 0.0 && v.is_finite())) {
                    return Err("poisson y must be non-negative and finite".to_string());
                }
                if !offset.is_empty()
                    && (offset.len() != self.n
                        || offset.iter().any(|v| !(*v > 0.0 && v.is_finite())))
                {
                    return Err(format!(
                        "offset must have {} strictly positive values",
                        self.n
                    ));
                }
                self.offset = offset;
            }
            Family::Binomial => {
                if self.y.iter().any(|v| *v != 0.0 && *v != 1.0) {
                    return Err("binomial y must be 0 or 1".to_string());
                }
            }
        }
        self.family = family;
        self.iwls_tol = tol;
        self.iwls_max_iter = max_iter;
        Ok(self)
    }

    /// Offset of observation `j` (one unless a Poisson offset was given).
    pub(crate) fn offset_at(&self, j: usize) -> f64 {
        if self.offset.is_empty() {
            1.0
        } else {
            self.offset[j]
        }
    }

    /// Row `j` of the design matrix.
    pub(crate) fn row(&self, j: usize) -> &[f64] {
        &self.x[j * self.k..(j + 1) * self.k]
    }
}

/// Student t quantile for a two-sided level: `t.ppf(1 - alpha / 2, df)`, as mgwr's
/// `critical_tval` computes it. `alpha <= 0` gives `+inf`, `alpha >= 2` gives `-inf`.
pub fn t_critical(alpha: f64, df: f64) -> f64 {
    use crate::regression::dist::t_two_sided;
    if alpha.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    let q = 1.0 - alpha / 2.0;
    if q >= 1.0 {
        return f64::INFINITY;
    }
    if q <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if q == 0.5 {
        return 0.0;
    }
    // Two-sided tail mass of the quantile magnitude: 2 (1 - q) above the median, 2 q below.
    let target = if q > 0.5 { 2.0 * (1.0 - q) } else { 2.0 * q };
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while t_two_sided(hi, df) > target && hi < 1e300 {
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_two_sided(mid, df) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    if q > 0.5 {
        t
    } else {
        -t
    }
}
