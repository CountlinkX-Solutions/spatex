//! Bounded scalar minimization, a literal port of SciPy's `_minimize_scalar_bounded`
//! (`minimize_scalar(method='bounded')`, Brent's method with golden-section fallback).
//!
//! spreg's ML models minimize their concentrated log-likelihood with this routine and a loose
//! tolerance (`xatol = 1e-7`), so the optimum is only determined to that tolerance: matching
//! SciPy's rho/lambda to ~1e-12 requires following exactly the same iteration path. Every branch,
//! the literal `sqrt(2.2e-16)` and the NaN behavior (all comparisons with NaN are false) are
//! therefore kept as in SciPy.

/// Outcome of [`minimize_bounded`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Minimum {
    /// Abscissa of the best point.
    pub x: f64,
    /// Objective value at `x`.
    pub fun: f64,
    /// Number of objective evaluations (`nfev`).
    pub nfev: usize,
    /// SciPy's `status`: 0 converged, 1 evaluation limit reached, 2 NaN encountered.
    pub flag: u8,
}

impl Minimum {
    /// Whether the search converged (SciPy's `success`).
    pub fn converged(&self) -> bool {
        self.flag == 0
    }
}

/// NumPy's `sign` plus one when the argument is zero: the step direction used by SciPy.
fn direction(v: f64) -> f64 {
    let sign = if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        f64::NAN
    };
    sign + f64::from(u8::from(v == 0.0))
}

/// Minimizes `func` on the closed interval `[x1, x2]` with absolute tolerance `xatol`, using at
/// most `maxfun` evaluations (SciPy's default is 500).
pub fn minimize_bounded<F: FnMut(f64) -> f64>(
    mut func: F,
    x1: f64,
    x2: f64,
    xatol: f64,
    maxfun: usize,
) -> Result<Minimum, String> {
    if !(x1.is_finite() && x2.is_finite()) {
        return Err("optimization bounds must be finite scalars".to_string());
    }
    if x1 > x2 {
        return Err("the lower bound exceeds the upper bound".to_string());
    }
    let mut flag = 0u8;
    let sqrt_eps = 2.2e-16_f64.sqrt();
    let golden_mean = 0.5 * (3.0 - 5.0_f64.sqrt());
    let (mut a, mut b) = (x1, x2);
    let mut fulc = a + golden_mean * (b - a);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat = 0.0_f64;
    let mut e = 0.0_f64;
    let mut x = xf;
    let mut fx = func(x);
    let mut num = 1usize;
    let mut fu = f64::INFINITY;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
    let mut tol2 = 2.0 * tol1;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        // Check for a parabolic fit.
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            // Check for acceptability of the parabola.
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = direction(xm - xf);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = direction(rat);
        x = xf + si * rat.abs().max(tol1);
        fu = func(x);
        num += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if num >= maxfun {
            flag = 1;
            break;
        }
    }
    if xf.is_nan() || fx.is_nan() || fu.is_nan() {
        flag = 2;
    }
    Ok(Minimum {
        x: xf,
        fun: fx,
        nfev: num,
        flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values come from scipy 1.x `minimize_scalar(f, bounds=(-1, 1), method='bounded',
    // options={'xatol': tol, 'maxiter': maxfun})`.

    #[test]
    fn matches_scipy_on_a_quadratic() {
        let m = minimize_bounded(|x| (x - 0.3) * (x - 0.3), -1.0, 1.0, 1e-7, 500).unwrap();
        assert_eq!((m.nfev, m.flag), (6, 0));
        assert_eq!(m.x, 0.3);
    }

    #[test]
    fn matches_scipy_on_a_skewed_function() {
        let m = minimize_bounded(|x| x.exp() - 2.0 * x, -1.0, 1.0, 1e-7, 500).unwrap();
        assert_eq!((m.nfev, m.flag), (11, 0));
        assert_eq!(m.x, 0.693_147_198_075_902);
    }

    #[test]
    fn stops_at_the_evaluation_limit() {
        let m = minimize_bounded(|x| (x - 0.3).abs(), -1.0, 1.0, 1e-12, 4).unwrap();
        assert_eq!((m.nfev, m.flag), (4, 1));
        assert_eq!(m.x, 0.244_567_281_339_793_44);
    }

    #[test]
    fn boundary_minimum_stays_inside_the_interval() {
        let m = minimize_bounded(|x| -x, -1.0, 1.0, 1e-7, 500).unwrap();
        assert_eq!((m.nfev, m.flag), (36, 0));
        assert_eq!(m.x, 0.999_999_951_234_752_5);
    }

    #[test]
    fn flags_nan_and_rejects_bad_bounds() {
        let m = minimize_bounded(|_| f64::NAN, -1.0, 1.0, 1e-7, 500).unwrap();
        assert_eq!(m.flag, 2);
        assert!(minimize_bounded(|x| x, 1.0, -1.0, 1e-7, 500).is_err());
        assert!(minimize_bounded(|x| x, f64::NEG_INFINITY, 1.0, 1e-7, 500).is_err());
    }
}
