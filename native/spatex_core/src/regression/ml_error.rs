//! Maximum-likelihood spatial error model `y = X beta + u`, `u = lambda W u + e`
//! (`spreg.ML_Error`, spreg 1.9.1). See [`super::ml_lag`] for the conventions and the documented
//! deviations; the results are named `f64` vectors in the same layout.

use faer::Mat;

use super::brent::minimize_bounded;
use super::dist::norm_two_sided;
use super::linalg::{col_mat, column, inverse, mat_from_cols, mean, row_major};
use super::ml_common::{check_inputs, pearson_r, std_dev, traces, Jacobian};
use super::ml_lag::Options;
use super::ols::Entries;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// The spatially filtered normal equations at `lam`: `(xs'xs)^-1` and `xs'ys`, plus `ys'ys`.
struct Filtered {
    xsxsi: Mat<f64>,
    xsys: Mat<f64>,
    ysys: f64,
}

fn filter(
    y: &[f64],
    ylag: &[f64],
    cols: &[Vec<f64>],
    xlag: &[Vec<f64>],
    lam: f64,
) -> Result<Filtered, String> {
    let n = y.len();
    let ys: Vec<f64> = y.iter().zip(ylag).map(|(a, b)| a - lam * b).collect();
    let xs: Vec<Vec<f64>> = cols
        .iter()
        .zip(xlag)
        .map(|(c, l)| c.iter().zip(l).map(|(a, b)| a - lam * b).collect())
        .collect();
    let xm = mat_from_cols(&xs, n);
    let xsxs = xm.transpose() * &xm;
    let xsxsi = inverse(&xsxs)?;
    let xsys = xm.transpose() * col_mat(&ys);
    Ok(Filtered {
        xsxsi,
        xsys,
        ysys: dot(&ys, &ys),
    })
}

/// Fits the ML spatial error model (`Options::impacts` is ignored).
pub fn ml_error(y: &[f64], cols: &[Vec<f64>], opts: &Options) -> Result<Entries, String> {
    ml_error_xlag(y, cols, None, opts)
}

/// [`ml_error`] with the spatially lagged design given instead of computed as `W X`: spreg's
/// `ML_Error_Regimes` lags the original columns and then expands them by regime, which differs
/// from lagging the regime design where neighbours lie in another regime.
pub fn ml_error_xlag(
    y: &[f64],
    cols: &[Vec<f64>],
    xlag: Option<&[Vec<f64>]>,
    opts: &Options,
) -> Result<Entries, String> {
    let n = y.len();
    let k = cols.len();
    let w = opts.w;
    check_inputs(w, n, opts.epsilon, opts.max_dense_n)?;
    if k == 0 || n <= k {
        return Err(format!("need more observations ({n}) than columns ({k})"));
    }
    let nf = n as f64;
    let ylag = w.lag(y)?;
    let xlag: Vec<Vec<f64>> = match xlag {
        Some(given) => {
            if given.len() != k || given.iter().any(|c| c.len() != n) {
                return Err(format!("xlag must hold {k} columns of {n} values"));
            }
            given.to_vec()
        }
        None => cols
            .iter()
            .map(|c| w.lag(c))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let jac = Jacobian::new(w, opts.method)?;

    // A singular filtered X'X aborts the fit (spreg raises LinAlgError).
    let failure = std::cell::RefCell::new(None::<String>);
    let objective = |lam: f64| -> f64 {
        match filter(y, &ylag, cols, &xlag, lam) {
            Ok(f) => {
                let x1 = &f.xsxsi * &f.xsys;
                let x2 = (f.xsys.transpose() * &x1)[(0, 0)];
                let sig2 = (f.ysys - x2) / nf;
                (nf / 2.0) * sig2.ln() - jac.log_det(lam)
            }
            Err(e) => {
                failure.borrow_mut().get_or_insert(e);
                f64::NAN
            }
        }
    };
    let (lam, fun, nfev, converged) = match opts.at {
        Some(lam) => (lam, objective(lam), 1, true),
        None => {
            let m = minimize_bounded(objective, -1.0, 1.0, opts.epsilon, 500)?;
            (m.x, m.fun, m.nfev, m.converged())
        }
    };
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }

    let logll = -fun - nf / 2.0 * (2.0 * std::f64::consts::PI).ln() - nf / 2.0;
    let f = filter(y, &ylag, cols, &xlag, lam)?;
    let b = column(&(&f.xsxsi * &f.xsys), 0);
    let x = mat_from_cols(cols, n);
    let predy = column(&(&x * col_mat(&b)), 0);
    let u: Vec<f64> = y.iter().zip(&predy).map(|(a, c)| a - c).collect();
    let wu = w.lag(&u)?;
    let e_filtered: Vec<f64> = u.iter().zip(&wu).map(|(a, c)| a - lam * c).collect();
    let sig2 = dot(&e_filtered, &e_filtered) / nf;

    let t = traces(w, lam)?;
    let info = Mat::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) => t.tr2 + t.tr3,
        (1, 1) => nf / (2.0 * sig2 * sig2),
        _ => t.tr1 / sig2,
    });
    let vm1 = inverse(&info)?;
    let kc = k + 1;
    let vm = Mat::from_fn(kc, kc, |i, j| {
        if i < k && j < k {
            sig2 * f.xsxsi[(i, j)]
        } else if i == k && j == k {
            vm1[(0, 0)]
        } else {
            0.0
        }
    });

    let mut betas = b;
    betas.push(lam);
    let std_err: Vec<f64> = (0..kc).map(|i| vm[(i, i)].sqrt()).collect();
    let z_stat: Vec<f64> = betas
        .iter()
        .zip(&std_err)
        .flat_map(|(b, s)| {
            let z = b / s;
            [z, norm_two_sided(z)]
        })
        .collect();
    // spreg does not count lambda in k for the information criteria.
    let kf = k as f64;

    let mut out = Entries::new();
    out.insert("lam", vec![lam]);
    out.insert("betas", betas);
    out.insert("vm", row_major(&vm));
    out.insert("vm1", row_major(&vm1));
    out.insert("std_err", std_err);
    out.insert("z_stat", z_stat);
    out.insert("logll", vec![logll]);
    out.insert("aic", vec![2.0 * kf - 2.0 * logll]);
    out.insert("schwarz", vec![kf * nf.ln() - 2.0 * logll]);
    out.insert("sig2", vec![sig2]);
    out.insert("pr2", vec![pearson_r(y, &predy).powi(2)]);
    out.insert("mean_y", vec![mean(y)]);
    out.insert("std_y", vec![std_dev(y)]);
    out.insert("iterations", vec![nfev as f64]);
    out.insert("converged", vec![f64::from(u8::from(converged))]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("e_filtered", e_filtered);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::ml_common::Method;
    use super::super::ml_lag::tests::{data, lattice, opts};
    use super::*;

    #[test]
    fn reports_a_consistent_fit() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let e = ml_error(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let lam = e["lam"][0];
        assert!(lam > -1.0 && lam < 1.0);
        assert_eq!(e["betas"].len(), 4);
        assert_eq!(e["betas"][3], lam);
        assert_eq!(e["vm"].len(), 16);
        assert_eq!(e["vm1"].len(), 4);
        // beta and lambda are uncorrelated.
        for j in 0..3 {
            assert_eq!(e["vm"][3 * 4 + j], 0.0);
            assert_eq!(e["vm"][j * 4 + 3], 0.0);
        }
        assert_eq!(e["vm"][15], e["vm1"][0]);
        // k excludes lambda: aic = 2 * 3 - 2 logll.
        assert!((e["aic"][0] - (6.0 - 2.0 * e["logll"][0])).abs() < 1e-10);
        for i in 0..25 {
            assert!((e["predy"][i] - (y[i] - e["u"][i])).abs() < 1e-12);
        }
        assert!(!e.contains_key("predy_e"));
    }

    #[test]
    fn at_parameter_reproduces_the_optimum() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let fitted = ml_error(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let mut o = opts(&w, Method::Full);
        o.at = Some(fitted["lam"][0]);
        let at = ml_error(&y, &cols, &o).unwrap();
        for key in ["logll", "sig2", "betas", "vm", "e_filtered"] {
            assert_eq!(fitted[key], at[key], "{key}");
        }
    }

    #[test]
    fn a_given_lagged_design_replaces_w_x() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let own: Vec<Vec<f64>> = cols.iter().map(|c| w.lag(c).unwrap()).collect();
        let base = ml_error(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let same = ml_error_xlag(&y, &cols, Some(&own), &opts(&w, Method::Full)).unwrap();
        assert_eq!(base["betas"], same["betas"]);
        assert_eq!(base["logll"], same["logll"]);
        // a different lagged design changes the likelihood
        let shifted: Vec<Vec<f64>> = own
            .iter()
            .map(|c| {
                c.iter()
                    .enumerate()
                    .map(|(i, v)| v + (i % 3) as f64)
                    .collect()
            })
            .collect();
        let other = ml_error_xlag(&y, &cols, Some(&shifted), &opts(&w, Method::Full)).unwrap();
        assert!((other["logll"][0] - base["logll"][0]).abs() > 1e-9);
        // the shape is checked
        let err = ml_error_xlag(&y, &cols, Some(&own[..2]), &opts(&w, Method::Full)).unwrap_err();
        assert!(err.contains("xlag"));
    }

    #[test]
    fn ord_matches_full_and_errors_propagate() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let full = ml_error(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let ord = ml_error(&y, &cols, &opts(&w, Method::Ord)).unwrap();
        assert!((full["lam"][0] - ord["lam"][0]).abs() < 1e-6);
        let dup = vec![cols[0].clone(), cols[1].clone(), cols[1].clone()];
        assert!(ml_error(&y, &dup, &opts(&w, Method::Full))
            .unwrap_err()
            .contains("singular"));
        let mut o = opts(&w, Method::Full);
        o.max_dense_n = 3;
        assert!(ml_error(&y, &cols, &o).unwrap_err().contains("max_dense_n"));
    }
}
