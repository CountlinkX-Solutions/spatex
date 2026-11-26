//! Two-stage least squares (`spreg.twosls.BaseTSLS` and `TSLS`, spreg 1.9.1), the building block
//! of the spatial GM estimators.
//!
//! Columns arrive as vectors that already include the constant. Instruments are `H = [X | q]`
//! unless an explicit `H` is supplied (the GM estimators reuse the first-stage `H` on spatially
//! filtered variables).

use faer::Mat;

use super::dist::norm_two_sided;
use super::gm_common::dot;
use super::linalg::{col_mat, column, inverse, mat_from_cols, row_major};
use super::ml_common::pearson_r;
use super::ols::{Entries, Robust};
use crate::graph::Csr;

/// Where the instruments come from.
#[derive(Clone, Copy)]
pub enum Instruments<'a> {
    /// Extra instruments: `H = [X | q]`.
    Q(&'a [Vec<f64>]),
    /// The full instrument matrix.
    H(&'a [Vec<f64>]),
}

/// Variance estimator settings.
#[derive(Clone, Copy)]
pub struct VarOptions<'a> {
    pub robust: Robust,
    /// Estimate sigma^2 (and the White correction) with `n - k` instead of `n`.
    pub sig2n_k: bool,
    /// Kernel weights for HAC.
    pub gwk: Option<&'a Csr>,
}

impl VarOptions<'_> {
    /// Classical variance with `sigma^2 = u'u / n`.
    pub const PLAIN: VarOptions<'static> = VarOptions {
        robust: Robust::None,
        sig2n_k: false,
        gwk: None,
    };
}

/// A fitted 2SLS model.
pub struct Tsls {
    pub n: usize,
    /// Columns of `Z = [X | yend]`.
    pub z: Vec<Vec<f64>>,
    /// Columns of the instrument matrix.
    pub h: Vec<Vec<f64>>,
    pub betas: Vec<f64>,
    /// `(Z'H (H'H)^-1 H'Z)^-1`.
    pub varb: Mat<f64>,
    /// `Z'H (H'H)^-1`.
    pub zthhthi: Mat<f64>,
    /// `(H'H)^-1`.
    pub hthi: Mat<f64>,
    pub predy: Vec<f64>,
    pub u: Vec<f64>,
    pub utu: f64,
    /// `sigma^2`: `u'u / n`, or `u'u / (n - k)` with `sig2n_k`.
    pub sig2: f64,
    pub vm: Mat<f64>,
}

impl Tsls {
    /// `n Z'H (H'H)^-1 ... `: `n * zthhthi' varb` (`h x k`), used by the heteroskedastic GM models.
    pub fn pfora1a2(&self) -> Mat<f64> {
        let t = self.zthhthi.transpose().to_owned();
        &t * &self.varb * (self.n as f64)
    }
}

/// Fits 2SLS of `y` on `[x | yend]` with the given instruments.
pub fn base_tsls(
    y: &[f64],
    x: &[Vec<f64>],
    yend: &[Vec<f64>],
    inst: Instruments,
    var: VarOptions,
) -> Result<Tsls, String> {
    let n = y.len();
    let mut z: Vec<Vec<f64>> = x.to_vec();
    z.extend(yend.iter().cloned());
    let h: Vec<Vec<f64>> = match inst {
        Instruments::Q(q) => x.iter().chain(q.iter()).cloned().collect(),
        Instruments::H(h) => h.to_vec(),
    };
    let k = z.len();
    if k == 0 {
        return Err("the model has no regressors".to_string());
    }
    if h.len() < k {
        return Err(format!(
            "under-identified: {} instruments for {k} regressors",
            h.len()
        ));
    }
    if n <= k {
        return Err(format!("need more observations ({n}) than columns ({k})"));
    }
    let hm = mat_from_cols(&h, n);
    let zm = mat_from_cols(&z, n);
    let hth = hm.transpose() * &hm;
    let hthi = inverse(&hth).map_err(|_| "H'H singular - no inverse".to_string())?;
    let zth = zm.transpose() * &hm;
    let hty = hm.transpose() * col_mat(y);
    let factor1 = &zth * &hthi;
    let factor2 = &factor1 * zth.transpose();
    let varb = inverse(&factor2).map_err(|_| {
        "Singular matrix Z'H(H'H)^-1H'Z - endogenous variable(s) may be part of X".to_string()
    })?;
    let betas_m = &varb * &factor1 * &hty;
    let betas = column(&betas_m, 0);
    let predy = column(&(&zm * &betas_m), 0);
    let u: Vec<f64> = y.iter().zip(&predy).map(|(a, b)| a - b).collect();
    let utu = dot(&u, &u);
    let kf = k as f64;
    let nf = n as f64;
    let sig2 = if var.sig2n_k {
        utu / (nf - kf)
    } else {
        utu / nf
    };
    let vm = match var.robust {
        Robust::None => &varb * sig2,
        Robust::White | Robust::Hac => {
            let xu: Vec<Vec<f64>> = h
                .iter()
                .map(|c| c.iter().zip(&u).map(|(a, b)| a * b).collect())
                .collect();
            let xum = mat_from_cols(&xu, n);
            let psi0 = match (var.robust, var.gwk) {
                (Robust::Hac, Some(g)) => {
                    let lagged = xu.iter().map(|c| g.lag(c)).collect::<Result<Vec<_>, _>>()?;
                    xum.transpose() * mat_from_cols(&lagged, n)
                }
                (Robust::Hac, None) => {
                    return Err("robust: hac requires kernel weights".to_string())
                }
                _ => {
                    let m = xum.transpose() * &xum;
                    if var.sig2n_k {
                        m * (nf / (nf - kf))
                    } else {
                        m
                    }
                }
            };
            let psi1 = &varb * &factor1;
            &psi1 * &psi0 * psi1.transpose()
        }
    };
    Ok(Tsls {
        n,
        z,
        h,
        betas,
        varb,
        zthhthi: factor1,
        hthi,
        predy,
        u,
        utu,
        sig2,
        vm,
    })
}

/// Standard errors and `[z, p]` pairs of a coefficient vector with covariance `vm` (only the
/// first `betas.len()` diagonal entries are used).
pub fn inference(betas: &[f64], vm: &Mat<f64>) -> (Vec<f64>, Vec<f64>) {
    let std_err: Vec<f64> = (0..betas.len()).map(|i| vm[(i, i)].sqrt()).collect();
    let z_stat = betas
        .iter()
        .zip(&std_err)
        .flat_map(|(b, s)| {
            let z = b / s;
            [z, norm_two_sided(z)]
        })
        .collect();
    (std_err, z_stat)
}

/// `spreg.TSLS` on columns that already include the constant: the result entries.
pub fn tsls(
    y: &[f64],
    x: &[Vec<f64>],
    yend: &[Vec<f64>],
    q: &[Vec<f64>],
    var: VarOptions,
) -> Result<Entries, String> {
    if yend.is_empty() {
        return Err("tsls needs at least one endogenous variable".to_string());
    }
    if q.is_empty() {
        return Err("tsls needs at least one instrument".to_string());
    }
    let t = base_tsls(y, x, yend, Instruments::Q(q), var)?;
    Ok(entries(y, t))
}

/// 2SLS of `y` on `[x | yend]` with the full instrument matrix `h` (no instruments are derived):
/// the second stage of models that reuse a first-stage `H`. With no `yend` and `h = x` it is OLS.
pub fn tsls_h(
    y: &[f64],
    x: &[Vec<f64>],
    yend: &[Vec<f64>],
    h: &[Vec<f64>],
) -> Result<Entries, String> {
    let t = base_tsls(y, x, yend, Instruments::H(h), VarOptions::PLAIN)?;
    Ok(entries(y, t))
}

fn entries(y: &[f64], t: Tsls) -> Entries {
    let (std_err, z_stat) = inference(&t.betas, &t.vm);
    let mut out = Entries::new();
    out.insert("pr2", vec![pearson_r(y, &t.predy).powi(2)]);
    out.insert("vm", row_major(&t.vm));
    out.insert("varb", row_major(&t.varb));
    out.insert("std_err", std_err);
    out.insert("z_stat", z_stat);
    out.insert("sig2", vec![t.sig2]);
    out.insert("utu", vec![t.utu]);
    out.insert("u", t.u);
    out.insert("predy", t.predy);
    out.insert("betas", t.betas);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = 1 + 2 x + 3 e with e endogenous (driven by an instrument), exact in the first stage.
    type Cols = Vec<Vec<f64>>;

    fn data() -> (Vec<f64>, Cols, Cols, Cols) {
        let n = 30;
        let c = vec![1.0; n];
        let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
        let q: Vec<f64> = (0..n).map(|i| ((i * 5) % 13) as f64).collect();
        let e: Vec<f64> = (0..n).map(|i| 0.5 * q[i] + 0.2 * x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + 3.0 * e[i]).collect();
        (y, vec![c, x], vec![e], vec![q])
    }

    #[test]
    fn exact_model_is_recovered_and_shapes_are_right() {
        let (y, x, yend, q) = data();
        let t = base_tsls(&y, &x, &yend, Instruments::Q(&q), VarOptions::PLAIN).unwrap();
        for (b, want) in t.betas.iter().zip([1.0, 2.0, 3.0]) {
            assert!((b - want).abs() < 1e-8, "{b}");
        }
        assert_eq!(
            (t.varb.nrows(), t.zthhthi.nrows(), t.zthhthi.ncols()),
            (3, 3, 3)
        );
        assert!(t.utu < 1e-12);
        assert_eq!((t.pfora1a2().nrows(), t.pfora1a2().ncols()), (3, 3));
    }

    #[test]
    fn explicit_h_matches_q_and_errors() {
        let (y, x, yend, q) = data();
        let a = base_tsls(&y, &x, &yend, Instruments::Q(&q), VarOptions::PLAIN).unwrap();
        let h: Vec<Vec<f64>> = x.iter().chain(&q).cloned().collect();
        let b = base_tsls(&y, &x, &yend, Instruments::H(&h), VarOptions::PLAIN).unwrap();
        assert_eq!(a.betas, b.betas);
        // endogenous variable without an instrument: under-identified
        let err = base_tsls(&y, &x, &yend, Instruments::Q(&[]), VarOptions::PLAIN).err();
        assert!(err.unwrap().contains("under-identified"));
        // an endogenous column that is part of X: singular
        let err = base_tsls(
            &y,
            &x,
            &[x[1].clone()],
            Instruments::Q(&q),
            VarOptions::PLAIN,
        )
        .err();
        assert!(err.unwrap().contains("Singular"));
    }

    #[test]
    fn robust_variances_differ_and_hac_needs_a_kernel() {
        let (mut y, x, yend, q) = data();
        for (i, v) in y.iter_mut().enumerate() {
            *v += ((i * 3) % 5) as f64 - 2.0;
        }
        let plain = base_tsls(&y, &x, &yend, Instruments::Q(&q), VarOptions::PLAIN).unwrap();
        let white = |sig2n_k| VarOptions {
            robust: Robust::White,
            sig2n_k,
            gwk: None,
        };
        let w0 = base_tsls(&y, &x, &yend, Instruments::Q(&q), white(false)).unwrap();
        let w1 = base_tsls(&y, &x, &yend, Instruments::Q(&q), white(true)).unwrap();
        assert!((w0.vm[(0, 0)] - plain.vm[(0, 0)]).abs() > 1e-9);
        assert!((w1.vm[(0, 0)] / w0.vm[(0, 0)] - 30.0 / 27.0).abs() < 1e-12);
        let hac = VarOptions {
            robust: Robust::Hac,
            sig2n_k: false,
            gwk: None,
        };
        assert!(base_tsls(&y, &x, &yend, Instruments::Q(&q), hac).is_err());
        // HAC with an identity kernel equals White
        let id = Csr::from_arrays(
            30,
            &(0..30).collect::<Vec<u32>>(),
            &(0..30).collect::<Vec<u32>>(),
            &[1.0; 30],
        )
        .unwrap();
        let hac = VarOptions {
            robust: Robust::Hac,
            sig2n_k: false,
            gwk: Some(&id),
        };
        let h = base_tsls(&y, &x, &yend, Instruments::Q(&q), hac).unwrap();
        assert!((h.vm[(1, 1)] - w0.vm[(1, 1)]).abs() < 1e-10);
    }

    #[test]
    fn public_entries_cover_the_reported_attributes() {
        let (y, x, yend, q) = data();
        let e = tsls(&y, &x, &yend, &q, VarOptions::PLAIN).unwrap();
        assert_eq!(e["betas"].len(), 3);
        assert_eq!(e["vm"].len(), 9);
        assert_eq!(e["z_stat"].len(), 6);
        assert!(tsls(&y, &x, &[], &q, VarOptions::PLAIN).is_err());
        assert!(tsls(&y, &x, &yend, &[], VarOptions::PLAIN).is_err());
    }

    #[test]
    fn tsls_h_entries_match_the_q_fit_and_cover_ols() {
        let (y, x, yend, q) = data();
        let h: Vec<Vec<f64>> = x.iter().chain(&q).cloned().collect();
        let a = tsls(&y, &x, &yend, &q, VarOptions::PLAIN).unwrap();
        let b = tsls_h(&y, &x, &yend, &h).unwrap();
        for key in ["betas", "vm", "varb", "u", "predy", "sig2", "utu"] {
            assert_eq!(a[key], b[key], "{key}");
        }
        // no endogenous columns and H = X is OLS
        let ols = tsls_h(&y, &x, &[], &x).unwrap();
        let fit = crate::regression::linalg::fit(&y, &x).unwrap();
        for (b, o) in ols["betas"].iter().zip(&fit.betas) {
            assert!((b - o).abs() < 1e-9);
        }
        assert!(tsls_h(&y, &x, &yend, &x).is_err());
    }
}
