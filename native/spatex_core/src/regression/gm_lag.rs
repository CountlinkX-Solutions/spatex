//! Spatial two-stage least squares `y = rho W y + X beta + e` (`spreg.GM_Lag`, spreg 1.9.1).
//!
//! `W y` is instrumented with spatial lags of `X` (and of the extra instruments with `lag_q`).
//! Nothing is optimized: every quantity matches spreg to rounding. The design columns already
//! include the constant (the first column when `skip_const` is set).

use super::dist::chi2_sf;
use super::gm_common::{dot, set_endog, set_endog_slx, sp_att};
use super::linalg::{row_major, trace_wtw_ww};
use super::ml_common::{impact_methods, multipliers, pearson_r, Impacts};
use super::ols::{Entries, Robust};
use super::tsls::{base_tsls, inference, Instruments, VarOptions};
use crate::graph::Csr;

/// Options of [`gm_lag`].
pub struct Options<'a> {
    pub w: &'a Csr,
    pub w_lags: usize,
    pub lag_q: bool,
    pub robust: Robust,
    pub gwk: Option<&'a Csr>,
    pub sig2n_k: bool,
    pub spat_diag: bool,
    pub impacts: Impacts,
    pub hard_bound: bool,
    /// The first design column is the constant (excluded from the spatial instruments).
    pub skip_const: bool,
    /// Number of SLX lags among the regressors (spreg's `slx_lags`; 0 for none).
    pub slx_lags: usize,
    /// With `slx_lags > 0`: the number of original non-constant regressors, which precede the
    /// SLX columns and are the ones lagged for the instruments.
    pub kx_base: usize,
}

/// Fits the spatial 2SLS lag model. `yend` / `q` are extra endogenous variables and instruments.
pub fn gm_lag(
    y: &[f64],
    x: &[Vec<f64>],
    yend: &[Vec<f64>],
    q: &[Vec<f64>],
    opts: &Options,
) -> Result<Entries, String> {
    let n = y.len();
    let w = opts.w;
    if w.n != n {
        return Err(format!("graph has {} nodes but y has {n} rows", w.n));
    }
    if yend.is_empty() && !q.is_empty() {
        return Err("q needs a matching yend".to_string());
    }
    let x_nc: &[Vec<f64>] = if opts.skip_const && !x.is_empty() {
        &x[1..]
    } else {
        x
    };
    let (yend2, q2) = if opts.slx_lags > 0 {
        let base = x_nc
            .get(..opts.kx_base)
            .ok_or("kx_base exceeds the number of regressors")?;
        set_endog_slx(w, y, base, yend, q, opts.w_lags, opts.lag_q, opts.slx_lags)?
    } else {
        set_endog(w, y, x_nc, yend, q, opts.w_lags, opts.lag_q)?
    };
    let var = VarOptions {
        robust: opts.robust,
        sig2n_k: opts.sig2n_k,
        gwk: opts.gwk,
    };
    let t = base_tsls(y, x, &yend2, Instruments::Q(&q2), var)?;
    let k = t.betas.len();
    let nf = n as f64;
    let rho = t.betas[k - 1];
    let wy = yend2.last().expect("Wy is the last endogenous column");
    let att = sp_att(w, y, &t.predy, wy, rho, opts.hard_bound)?;
    let (std_err, z_stat) = inference(&t.betas, &t.vm);

    let mut out = Entries::new();
    out.insert("rho", vec![rho]);
    out.insert("pr2", vec![pearson_r(y, &t.predy).powi(2)]);
    if !att.out_of_bounds {
        out.insert("pr2_e", vec![pearson_r(y, &att.predy_e).powi(2)]);
    } else {
        out.insert("warn_rho", vec![1.0]);
    }
    if opts.spat_diag && opts.robust != Robust::Hac {
        out.insert("ak_test", ak_test(w, &t, nf)?);
    }
    if !att.out_of_bounds {
        insert_multipliers(&mut out, w, rho, opts.impacts)?;
    }
    out.insert("vm", row_major(&t.vm));
    out.insert("varb", row_major(&t.varb));
    out.insert("std_err", std_err);
    out.insert("z_stat", z_stat);
    out.insert("sig2", vec![t.sig2]);
    out.insert("u", t.u);
    out.insert("predy", t.predy);
    out.insert("predy_e", att.predy_e);
    out.insert("e_pred", att.e_pred);
    out.insert("betas", t.betas);
    Ok(out)
}

/// Anselin-Kelejian test for residual spatial autocorrelation in the 2SLS residuals:
/// `[statistic, p]` (chi-squared with one degree of freedom).
fn ak_test(w: &Csr, t: &super::tsls::Tsls, nf: f64) -> Result<Vec<f64>, String> {
    ak_from_parts(w, &t.u, &t.z, |i, j| t.varb[(i, j)], nf)
}

/// [`ak_test`] on the residuals `u`, the regressor columns `z` and the 2SLS covariance
/// `varb(i, j)` of those regressors.
fn ak_from_parts(
    w: &Csr,
    u: &[f64],
    z: &[Vec<f64>],
    varb: impl Fn(usize, usize) -> f64,
    nf: f64,
) -> Result<Vec<f64>, String> {
    let s0 = w.s0();
    let wu = w.lag(u)?;
    let utu = dot(u, u);
    let mi = nf * dot(u, &wu) / (s0 * utu);
    let k = z.len();
    let etwz: Vec<f64> = z
        .iter()
        .map(|c| w.lag(c).map(|wc| dot(u, &wc)))
        .collect::<Result<_, _>>()?;
    let mut a = 0.0;
    for i in 0..k {
        for j in 0..k {
            a += etwz[i] * varb(i, j) * etwz[j];
        }
    }
    let sig2n = utu / nf;
    let s12 = (s0 / nf) * (s0 / nf);
    let phi2 = (trace_wtw_ww(w) + 4.0 / sig2n * a) / (s12 * nf);
    let ak = nf * mi * mi / phi2;
    Ok(vec![ak, chi2_sf(ak, 1.0)])
}

/// What [`lag_diagnostics`] reports besides the pseudo R-squared.
pub struct DiagOptions {
    pub spat_diag: bool,
    pub impacts: Impacts,
}

/// The diagnostics of a spatial lag fit assembled by a caller (the regimes models): the
/// Anselin-Kelejian test, the reduced-form pseudo R-squared `pr2_e` and the impact multipliers.
///
/// `z` are the regressor columns of the 2SLS fit, `varb` its `(Z'H (H'H)^-1 H'Z)^-1` row-major,
/// `u` its residuals and `predy_e` the reduced-form prediction. `pr2_e` and the multipliers are
/// omitted for `|rho| >= 1`, as in spreg.
#[allow(clippy::too_many_arguments)]
pub fn lag_diagnostics(
    w: &Csr,
    y: &[f64],
    predy_e: &[f64],
    z: &[Vec<f64>],
    varb: &[f64],
    u: &[f64],
    rho: f64,
    opts: &DiagOptions,
) -> Result<Entries, String> {
    let k = z.len();
    if varb.len() != k * k {
        return Err(format!(
            "varb has {} values, expected {k} x {k}",
            varb.len()
        ));
    }
    let mut out = Entries::new();
    if opts.spat_diag {
        out.insert(
            "ak_test",
            ak_from_parts(w, u, z, |i, j| varb[i * k + j], u.len() as f64)?,
        );
    }
    if rho.abs() < 1.0 {
        out.insert("pr2_e", vec![pearson_r(y, predy_e).powi(2)]);
        insert_multipliers(&mut out, w, rho, opts.impacts)?;
    }
    Ok(out)
}

fn insert_multipliers(
    out: &mut Entries,
    w: &Csr,
    rho: f64,
    impacts: Impacts,
) -> Result<(), String> {
    for method in impact_methods(impacts) {
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
mod tests {
    use super::super::ml_lag::tests::{data, lattice};
    use super::*;

    fn opts(w: &Csr) -> Options<'_> {
        Options {
            w,
            w_lags: 1,
            lag_q: true,
            robust: Robust::None,
            gwk: None,
            sig2n_k: false,
            spat_diag: true,
            impacts: Impacts::Simple,
            hard_bound: false,
            skip_const: true,
            slx_lags: 0,
            kx_base: 0,
        }
    }

    fn design(n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        data(n)
    }

    #[test]
    fn reports_a_consistent_fit() {
        let w = lattice(6);
        let (y, x) = design(36);
        let e = gm_lag(&y, &x, &[], &[], &opts(&w)).unwrap();
        let k = x.len() + 1;
        assert_eq!(e["betas"].len(), k);
        assert_eq!(e["betas"][k - 1], e["rho"][0]);
        assert_eq!(e["vm"].len(), k * k);
        assert_eq!(e["z_stat"].len(), 2 * k);
        assert_eq!(e["ak_test"].len(), 2);
        assert!(e["ak_test"][1] > 0.0 && e["ak_test"][1] <= 1.0);
        assert_eq!(e["mult_simple"][0], 1.0);
        for i in 0..36 {
            assert!((e["predy"][i] - (y[i] - e["u"][i])).abs() < 1e-10);
        }
        let sig2 = e["u"].iter().map(|v| v * v).sum::<f64>() / 36.0;
        assert!((e["sig2"][0] - sig2).abs() < 1e-12);
        assert!(e["pr2_e"][0] >= 0.0 && e["pr2"][0] <= 1.0);
    }

    #[test]
    fn options_change_the_instruments_and_variance() {
        let w = lattice(6);
        let (y, x) = design(36);
        let base = gm_lag(&y, &x, &[], &[], &opts(&w)).unwrap();
        let mut o2 = opts(&w);
        o2.w_lags = 2;
        let two = gm_lag(&y, &x, &[], &[], &o2).unwrap();
        assert!((two["rho"][0] - base["rho"][0]).abs() > 1e-9);
        let mut ow = opts(&w);
        ow.robust = Robust::White;
        let white = gm_lag(&y, &x, &[], &[], &ow).unwrap();
        assert_eq!(white["betas"], base["betas"]);
        assert_ne!(white["vm"], base["vm"]);
        let mut oh = opts(&w);
        oh.robust = Robust::Hac;
        oh.gwk = Some(&w);
        let hac = gm_lag(&y, &x, &[], &[], &oh).unwrap();
        assert!(!hac.contains_key("ak_test"));
        let mut on = opts(&w);
        on.spat_diag = false;
        on.impacts = Impacts::All;
        let plain = gm_lag(&y, &x, &[], &[], &on).unwrap();
        assert!(!plain.contains_key("ak_test"));
        for key in ["mult_simple", "mult_full", "mult_power", "power_terms"] {
            assert!(plain.contains_key(key), "{key}");
        }
    }

    #[test]
    fn endogenous_variable_gets_its_own_instruments() {
        let w = lattice(6);
        let (y, x) = design(36);
        let yend = vec![x[1]
            .iter()
            .zip(&x[2])
            .map(|(a, b)| a - 0.5 * b)
            .collect::<Vec<f64>>()];
        let q = vec![x[2].iter().map(|v| v * v).collect::<Vec<f64>>()];
        let e = gm_lag(&y, &x[..2], &yend, &q, &opts(&w)).unwrap();
        // constant, x1, yend, rho
        assert_eq!(e["betas"].len(), 4);
        assert!(gm_lag(&y, &x, &[], &q, &opts(&w)).is_err());
    }

    #[test]
    fn diagnostics_of_given_parts_match_the_fit() {
        let w = lattice(6);
        let (y, x) = design(36);
        let mut o = opts(&w);
        o.impacts = Impacts::All;
        let e = gm_lag(&y, &x, &[], &[], &o).unwrap();
        let (yend2, _) = set_endog(&w, &y, &x[1..], &[], &[], 1, true).unwrap();
        let z: Vec<Vec<f64>> = x.iter().cloned().chain(yend2.iter().cloned()).collect();
        let d = lag_diagnostics(
            &w,
            &y,
            &e["predy_e"],
            &z,
            &e["varb"],
            &e["u"],
            e["rho"][0],
            &DiagOptions {
                spat_diag: true,
                impacts: Impacts::All,
            },
        )
        .unwrap();
        for key in ["ak_test", "pr2_e", "mult_simple", "mult_full", "mult_power"] {
            assert_eq!(d[key].len(), e[key].len(), "{key}");
            for (a, b) in d[key].iter().zip(&e[key]) {
                assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b}");
            }
        }
        let off = lag_diagnostics(
            &w,
            &y,
            &e["predy_e"],
            &z,
            &e["varb"],
            &e["u"],
            1.2,
            &DiagOptions {
                spat_diag: false,
                impacts: Impacts::Simple,
            },
        )
        .unwrap();
        assert!(off.is_empty(), "|rho| >= 1 omits pr2_e and the multipliers");
        assert!(lag_diagnostics(
            &w,
            &y,
            &e["predy_e"],
            &z,
            &e["varb"][..3],
            &e["u"],
            0.1,
            &DiagOptions {
                spat_diag: true,
                impacts: Impacts::None,
            },
        )
        .is_err());
    }
}
