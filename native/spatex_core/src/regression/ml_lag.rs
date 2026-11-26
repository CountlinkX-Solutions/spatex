//! Maximum-likelihood spatial lag model `y = rho W y + X beta + e` (`spreg.ML_Lag`, spreg 1.9.1).
//!
//! `rho` maximizes the concentrated log-likelihood over (-1, 1) with SciPy's bounded Brent
//! search (see [`super::brent`]); the coefficients, residuals and the asymptotic covariance of
//! `(beta, rho, sigma^2)` follow from it. The design matrix arrives as columns that already
//! include the constant. Results are named `f64` vectors (see [`Entries`]): scalars have one
//! element, matrices are row-major, pairs are `[statistic, p]`.
//!
//! Deviations from spreg (recorded in `test/reference/known_deviations.exs`): `method = Ord`
//! uses the eigenvalues of `W` itself (spreg substitutes the binary-symmetrized matrix when the
//! pattern is symmetric, which is wrong for non-binary weights) and the variance always uses the
//! real `W`; the log-determinant is a sum of logs and cannot overflow.

use faer::Mat;

use super::brent::minimize_bounded;
use super::dist::norm_two_sided;
use super::linalg::{col_mat, column, inverse, mat_from_cols, mean, row_major};
use super::ml_common::{
    check_inputs, impact_methods, multipliers, pearson_r, power_expansion, std_dev, traces,
    Impacts, Jacobian, Method,
};
use super::ols::Entries;
use crate::graph::Csr;

/// Options of [`ml_lag`].
pub struct Options<'a> {
    /// Spatial weights, used exactly as given.
    pub w: &'a Csr,
    pub method: Method,
    /// Brent tolerance (`xatol`) and power-expansion threshold.
    pub epsilon: f64,
    pub impacts: Impacts,
    /// Evaluate every quantity at this `rho` instead of optimizing (test hook).
    pub at: Option<f64>,
    /// Largest `n` accepted (the variance needs dense `n x n` matrices).
    pub max_dense_n: usize,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

/// Fits the ML spatial lag model.
pub fn ml_lag(y: &[f64], cols: &[Vec<f64>], opts: &Options) -> Result<Entries, String> {
    let n = y.len();
    let k = cols.len();
    let w = opts.w;
    check_inputs(w, n, opts.epsilon, opts.max_dense_n)?;
    if k == 0 || n <= k {
        return Err(format!("need more observations ({n}) than columns ({k})"));
    }
    let nf = n as f64;
    let ylag = w.lag(y)?;
    let x = mat_from_cols(cols, n);
    let xt = x.transpose();
    let xtx = xt * &x;
    let xtxi = inverse(&xtx)?;
    let b0 = &xtxi * (xt * col_mat(y));
    let b1 = &xtxi * (xt * col_mat(&ylag));
    let e0: Vec<f64> = y
        .iter()
        .zip(column(&(&x * &b0), 0))
        .map(|(a, b)| a - b)
        .collect();
    let e1: Vec<f64> = ylag
        .iter()
        .zip(column(&(&x * &b1), 0))
        .map(|(a, b)| a - b)
        .collect();

    let jac = Jacobian::new(w, opts.method)?;
    let objective = |rho: f64| -> f64 {
        let er: Vec<f64> = e0.iter().zip(&e1).map(|(a, b)| a - rho * b).collect();
        let sig2 = dot(&er, &er) / nf;
        (nf / 2.0) * sig2.ln() - jac.log_det(rho)
    };
    let (rho, fun, nfev, converged) = match opts.at {
        Some(rho) => (rho, objective(rho), 1, true),
        None => {
            let m = minimize_bounded(objective, -1.0, 1.0, opts.epsilon, 500)?;
            (m.x, m.fun, m.nfev, m.converged())
        }
    };

    let logll = -fun - nf / 2.0 * (2.0 * std::f64::consts::PI).ln() - nf / 2.0;
    let b: Vec<f64> = (0..k).map(|j| b0[(j, 0)] - rho * b1[(j, 0)]).collect();
    let u: Vec<f64> = e0.iter().zip(&e1).map(|(a, c)| a - rho * c).collect();
    let predy: Vec<f64> = y.iter().zip(&u).map(|(a, c)| a - c).collect();
    let xb = column(&(&x * col_mat(&b)), 0);
    let predy_e = power_expansion(w, &xb, rho, opts.epsilon)?;
    let e_pred: Vec<f64> = y.iter().zip(&predy_e).map(|(a, c)| a - c).collect();
    let utu = dot(&u, &u);
    let sig2 = utu / nf;

    let t = traces(w, rho)?;
    let wpredy = w.lag(&predy_e)?;
    let wpy_wpy = dot(&wpredy, &wpredy);
    let xt_wpy: Vec<f64> = (0..k).map(|j| dot(cols[j].as_slice(), &wpredy)).collect();
    // Information matrix in the order beta (k), rho, sigma^2.
    let size = k + 2;
    let mut info = Mat::zeros(size, size);
    for i in 0..k {
        for j in 0..k {
            info[(i, j)] = xtx[(i, j)] / sig2;
        }
        info[(i, k)] = xt_wpy[i] / sig2;
        info[(k, i)] = xt_wpy[i] / sig2;
    }
    info[(k, k)] = t.tr2 + t.tr3 + wpy_wpy / sig2;
    info[(k, k + 1)] = t.tr1 / sig2;
    info[(k + 1, k)] = t.tr1 / sig2;
    info[(k + 1, k + 1)] = nf / (2.0 * sig2 * sig2);
    let vm1 = inverse(&info)?;
    let kc = k + 1;
    let vm = Mat::from_fn(kc, kc, |i, j| vm1[(i, j)]);

    let mut betas = b;
    betas.push(rho);
    let std_err: Vec<f64> = (0..kc).map(|i| vm[(i, i)].sqrt()).collect();
    let z_stat: Vec<f64> = betas
        .iter()
        .zip(&std_err)
        .flat_map(|(b, s)| {
            let z = b / s;
            [z, norm_two_sided(z)]
        })
        .collect();
    // spreg counts rho in k for the information criteria.
    let kf = kc as f64;

    let mut out = Entries::new();
    out.insert("rho", vec![rho]);
    out.insert("vm", row_major(&vm));
    out.insert("vm1", row_major(&vm1));
    out.insert("std_err", std_err);
    out.insert("z_stat", z_stat);
    out.insert("logll", vec![logll]);
    out.insert("aic", vec![2.0 * kf - 2.0 * logll]);
    out.insert("schwarz", vec![kf * nf.ln() - 2.0 * logll]);
    out.insert("sig2", vec![sig2]);
    out.insert("utu", vec![utu]);
    out.insert("pr2", vec![pearson_r(y, &predy).powi(2)]);
    if rho.abs() < 1.0 {
        out.insert("pr2_e", vec![pearson_r(y, &predy_e).powi(2)]);
    }
    out.insert("mean_y", vec![mean(y)]);
    out.insert("std_y", vec![std_dev(y)]);
    out.insert("iterations", vec![nfev as f64]);
    out.insert("converged", vec![f64::from(u8::from(converged))]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("predy_e", predy_e);
    out.insert("e_pred", e_pred);
    if rho.abs() < 1.0 {
        impacts(&mut out, w, rho, opts.impacts)?;
    }
    out.insert("betas", betas);
    Ok(out)
}

/// Impact multipliers `mult_<method>` = `[adi, aii, ati]` (`power_terms` for the power method).
/// Per-variable effects are the slopes times these multipliers.
fn impacts(out: &mut Entries, w: &Csr, rho: f64, choice: Impacts) -> Result<(), String> {
    for method in impact_methods(choice) {
        let m = multipliers(w, rho, method)?;
        let key = match *method {
            "simple" => "mult_simple",
            "full" => "mult_full",
            _ => "mult_power",
        };
        out.insert(key, vec![m.adi, m.aii, m.ati]);
        if *method == "power" {
            out.insert("power_terms", vec![m.pow as f64]);
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Row-standardized rook lattice on a `side x side` grid.
    pub(crate) fn lattice(side: usize) -> Csr {
        let (mut f, mut t, mut w) = (vec![], vec![], vec![]);
        for r in 0..side {
            for c in 0..side {
                let mut nb = vec![];
                if r > 0 {
                    nb.push((r - 1) * side + c);
                }
                if r + 1 < side {
                    nb.push((r + 1) * side + c);
                }
                if c > 0 {
                    nb.push(r * side + c - 1);
                }
                if c + 1 < side {
                    nb.push(r * side + c + 1);
                }
                for j in &nb {
                    f.push((r * side + c) as u32);
                    t.push(*j as u32);
                    w.push(1.0 / nb.len() as f64);
                }
            }
        }
        Csr::from_arrays(side * side, &f, &t, &w).unwrap()
    }

    /// Deterministic pseudo-data from a linear congruential generator.
    pub(crate) fn data(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        let x1: Vec<f64> = (0..n).map(|_| next()).collect();
        let x2: Vec<f64> = (0..n).map(|_| next()).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x1[i] - x2[i] + next()).collect();
        (y, vec![vec![1.0; n], x1, x2])
    }

    pub(crate) fn opts(w: &Csr, method: Method) -> Options<'_> {
        Options {
            w,
            method,
            epsilon: 1e-7,
            impacts: Impacts::Simple,
            at: None,
            max_dense_n: 100,
        }
    }

    #[test]
    fn reports_a_consistent_fit() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let e = ml_lag(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let rho = e["rho"][0];
        assert!(rho > -1.0 && rho < 1.0);
        assert_eq!(e["betas"].len(), 4);
        assert_eq!(e["betas"][3], rho);
        assert_eq!(e["vm"].len(), 16);
        assert_eq!(e["vm1"].len(), 25);
        assert_eq!(e["converged"], vec![1.0]);
        // predy = y - u and sig2 = u'u / n.
        for i in 0..25 {
            assert!((e["predy"][i] - (y[i] - e["u"][i])).abs() < 1e-12);
        }
        assert!((e["sig2"][0] - e["utu"][0] / 25.0).abs() < 1e-12);
        // rho counts in k: aic = 2 * 4 - 2 logll.
        assert!((e["aic"][0] - (8.0 - 2.0 * e["logll"][0])).abs() < 1e-10);
        assert_eq!(e["mult_simple"][0], 1.0);
        assert!((e["mult_simple"][2] - 1.0 / (1.0 - rho)).abs() < 1e-12);
        // vm is the leading block of vm1.
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(e["vm"][i * 4 + j], e["vm1"][i * 5 + j]);
            }
        }
    }

    #[test]
    fn at_parameter_reproduces_the_optimum() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let fitted = ml_lag(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let mut o = opts(&w, Method::Full);
        o.at = Some(fitted["rho"][0]);
        let at = ml_lag(&y, &cols, &o).unwrap();
        for key in ["logll", "sig2", "betas", "vm", "std_err"] {
            assert_eq!(fitted[key], at[key], "{key}");
        }
        assert_eq!(at["iterations"], vec![1.0]);
    }

    #[test]
    fn methods_agree_on_a_symmetric_lattice() {
        let w = lattice(5);
        let (y, cols) = data(25);
        let full = ml_lag(&y, &cols, &opts(&w, Method::Full)).unwrap();
        let ord = ml_lag(&y, &cols, &opts(&w, Method::Ord)).unwrap();
        let lu = ml_lag(&y, &cols, &opts(&w, Method::Lu)).unwrap();
        assert!((full["rho"][0] - ord["rho"][0]).abs() < 1e-6);
        assert_eq!(full["rho"], lu["rho"]);
        // Ord's variance uses the real W (spreg's symmetrized matrix is not reproduced).
        for (a, b) in full["std_err"].iter().zip(&ord["std_err"]) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn recovers_a_strong_spatial_lag() {
        let w = lattice(8);
        let n = 64;
        let (_, cols) = data(n);
        // y = (I - 0.6 W)^-1 (1 + 2 x1 - x2 + e) via the power series.
        let (noise, _) = data(n);
        let mu: Vec<f64> = (0..n)
            .map(|i| 1.0 + 2.0 * cols[1][i] - cols[2][i] + 0.3 * noise[i])
            .collect();
        let y = power_expansion(&w, &mu, 0.6, 1e-12).unwrap();
        let mut o = opts(&w, Method::Full);
        o.max_dense_n = 100;
        let e = ml_lag(&y, &cols, &o).unwrap();
        assert!((e["rho"][0] - 0.6).abs() < 0.15, "rho = {}", e["rho"][0]);
    }

    #[test]
    fn rejects_bad_input() {
        let w = lattice(3);
        let (y, cols) = data(9);
        let mut o = opts(&w, Method::Full);
        o.max_dense_n = 5;
        assert!(ml_lag(&y, &cols, &o).unwrap_err().contains("max_dense_n"));
        o.max_dense_n = 100;
        o.epsilon = -1.0;
        assert!(ml_lag(&y, &cols, &o).unwrap_err().contains("epsilon"));
        let (y2, cols2) = data(10);
        assert!(ml_lag(&y2, &cols2, &opts(&w, Method::Full))
            .unwrap_err()
            .contains("9 nodes"));
        let dup = vec![cols[0].clone(), cols[1].clone(), cols[1].clone()];
        assert!(ml_lag(&y, &dup, &opts(&w, Method::Full))
            .unwrap_err()
            .contains("singular"));
    }
}
