//! Spatial regression NIFs (spreg): thin transport over `spatex_core::regression`.

use rustler::{Binary, Env, NifMap, ResourceArc};
use spatex_core::regression::gm_common;
use spatex_core::regression::gm_error::{self, A1Kind, Combo, Estimator};
use spatex_core::regression::gm_lag::{self, DiagOptions, Options as GmLagOptions};
use spatex_core::regression::ml_common::{Impacts, Method};
use spatex_core::regression::ml_lag::Options as MlOptions;
use spatex_core::regression::ols::{self, Options, Robust};
use spatex_core::regression::tsls::{self as tsls_mod, VarOptions};
use spatex_core::regression::{
    ml_error, ml_lag, panel, panel_error_re, panel_gm_kkp, sur_error_ml,
};

use crate::{f64_binary, Graph};

/// OLS with diagnostics.
///
/// `x` holds `k` columns of `n` values (column-major) and already contains the constant.
/// `robust` is 0 (none), 1 (White) or 2 (HAC). Returns `[{name, f64_binary}]`: scalars have
/// one element, matrices are row-major, pairs are `[statistic, p]`.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn regression_ols<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    k: usize,
    robust: u8,
    sig2n_k: bool,
    flags: (bool, bool, bool, bool, bool),
    graph: Option<Graph>,
    gwk: Option<Graph>,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let flat = spatex_core::bytes_to_f64s(x.as_slice())?;
    let n = y.len();
    if k == 0 || flat.len() != n * k {
        return Err(format!(
            "x has {} values, expected {n} rows x {k} columns",
            flat.len()
        ));
    }
    let cols: Vec<Vec<f64>> = flat.chunks(n).map(<[f64]>::to_vec).collect();
    let (nonspat_diag, white_test, vif, spat_diag, moran) = flags;
    let robust = match robust {
        0 => Robust::None,
        1 => Robust::White,
        2 => Robust::Hac,
        other => return Err(format!("unknown robust code {other}")),
    };
    let w = graph.as_ref().map(|g: &ResourceArc<_>| &g.0);
    let gwk = gwk.as_ref().map(|g| &g.0);
    let entries = ols::ols(
        &y,
        &cols,
        &Options {
            robust,
            sig2n_k,
            gwk,
            nonspat_diag,
            white_test,
            vif,
            w,
            spat_diag,
            moran,
        },
    )?;
    Ok(entries
        .into_iter()
        .map(|(name, values)| (name.to_string(), f64_binary(env, &values)))
        .collect())
}

/// `{method, epsilon, impacts, at, max_dense_n}`: method 0 full / 1 ord / 2 lu, impacts 0 none ..
/// 4 all, `at` an optional parameter to evaluate at instead of optimizing.
type MlSettings = (u8, f64, u8, Option<f64>, usize);

fn columns(y: &[f64], x: Binary, k: usize) -> Result<Vec<Vec<f64>>, String> {
    let flat = spatex_core::bytes_to_f64s(x.as_slice())?;
    let n = y.len();
    if k == 0 || flat.len() != n * k {
        return Err(format!(
            "x has {} values, expected {n} rows x {k} columns",
            flat.len()
        ));
    }
    Ok(flat.chunks(n).map(<[f64]>::to_vec).collect())
}

fn transport<'a>(
    env: Env<'a>,
    entries: spatex_core::regression::ols::Entries,
) -> Vec<(String, Binary<'a>)> {
    entries
        .into_iter()
        .map(|(name, values)| (name.to_string(), f64_binary(env, &values)))
        .collect()
}

/// ML spatial lag model (`spreg.ML_Lag`).
///
/// `x` holds `k` columns of `n` values (column-major) including the constant; `opts` is an
/// [`MlSettings`] tuple. Returns `[{name, f64_binary}]` (matrices row-major).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_ml_lag<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    k: usize,
    graph: Graph,
    opts: MlSettings,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let (method, epsilon, impacts, at, max_dense_n) = opts;
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let cols = columns(&y, x, k)?;
    let entries = ml_lag::ml_lag(
        &y,
        &cols,
        &MlOptions {
            w: &graph.0,
            method: Method::from_code(method)?,
            epsilon,
            impacts: Impacts::from_code(impacts)?,
            at,
            max_dense_n,
        },
    )?;
    Ok(transport(env, entries))
}

/// Spatial panel with fixed effects (`spreg.ML_LagFE` for `model = "lag"`, `ML_ErrorFE` for
/// `"error"`). `y` and the `k` columns of `x` are stacked by period (`n * t` values each).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_panel_fe<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    k: usize,
    graph: Graph,
    (model, t, epsilon, impacts): (&str, usize, f64, u8),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let cols = columns(&y, x, k)?;
    let opts = panel::Options {
        w: &graph.0,
        t,
        epsilon,
        impacts: Impacts::from_code(impacts)?,
    };
    let entries = match model {
        "lag" => panel::ml_lag_fe(&y, &cols, &opts)?,
        "error" => panel::ml_error_fe(&y, &cols, &opts)?,
        "lag_re" => panel::ml_lag_re(&y, &cols, &opts)?,
        other => return Err(format!("unknown panel model {other:?}")),
    };
    Ok(transport(env, entries))
}

/// GM random-effects spatial error panel (`spreg.GM_ErrorRE` / `GM_KKP`). `y` and the `k` columns
/// of `x` (with the constant) are stacked by period; `settings` is `(t, full_weights, at)`: `at`
/// is empty, or spreg's five optimizer results `[lambda_1, sigma_v^2, lambda, sigma_v^2,
/// sigma_1^2]` to evaluate the formulas without the optimizer.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_panel_gm_kkp<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    k: usize,
    graph: Graph,
    (t, full_weights, at): (usize, bool, Vec<f64>),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let cols = columns(&y, x, k)?;
    let at = match at.len() {
        0 => None,
        5 => Some([at[0], at[1], at[2], at[3], at[4]]),
        other => return Err(format!("`at` needs 5 values, got {other}")),
    };
    let entries = panel_gm_kkp::gm_kkp(&y, &cols, &graph.0, t, full_weights, at)?;
    Ok(transport(env, entries))
}

/// Random-effects spatial error panel (`spreg.ML_ErrorRE`). `y` and the `k` columns of `x` (with
/// the constant) are stacked by period; `settings` is `(t, has_at, lam, phi)`: with `has_at` the
/// model is evaluated at `(lam, phi)` (the reported variance ratio) instead of optimized.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_panel_error_re<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    k: usize,
    graph: Graph,
    (t, has_at, lam, phi): (usize, bool, f64, f64),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let cols = columns(&y, x, k)?;
    let at = has_at.then_some((lam, phi));
    let entries = panel_error_re::ml_error_re(&y, &cols, &graph.0, t, at)?;
    Ok(transport(env, entries))
}

/// ML spatial error model (`spreg.ML_Error`); arguments as [`regression_ml_lag`] (impacts ignored).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_ml_error<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    k: usize,
    graph: Graph,
    opts: MlSettings,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let (method, epsilon, _impacts, at, max_dense_n) = opts;
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let cols = columns(&y, x, k)?;
    let entries = ml_error::ml_error(
        &y,
        &cols,
        &MlOptions {
            w: &graph.0,
            method: Method::from_code(method)?,
            epsilon,
            impacts: Impacts::None,
            at,
            max_dense_n,
        },
    )?;
    Ok(transport(env, entries))
}

/// [`regression_ml_error`] with the spatially lagged design `xlag` (`k` columns of `n` values,
/// like `x`) instead of `W x` (`spreg.ML_Error_Regimes`).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_ml_error_xlag<'a>(
    env: Env<'a>,
    y: Binary,
    x: Binary,
    xlag: Binary,
    k: usize,
    graph: Graph,
    opts: MlSettings,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let (method, epsilon, _impacts, at, max_dense_n) = opts;
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let cols = columns(&y, x, k)?;
    let lagged = columns(&y, xlag, k)?;
    let entries = ml_error::ml_error_xlag(
        &y,
        &cols,
        Some(&lagged),
        &MlOptions {
            w: &graph.0,
            method: Method::from_code(method)?,
            epsilon,
            impacts: Impacts::None,
            at,
            max_dense_n,
        },
    )?;
    Ok(transport(env, entries))
}

fn robust_of(code: u8) -> Result<Robust, String> {
    match code {
        0 => Ok(Robust::None),
        1 => Ok(Robust::White),
        2 => Ok(Robust::Hac),
        other => Err(format!("unknown robust code {other}")),
    }
}

/// A column-major block of `k` columns: `(f64 binary, k)`.
type Block<'a> = (Binary<'a>, usize);

/// Splits `k` column-major columns of length `n`; an empty binary with `k == 0` is allowed.
fn optional_columns(n: usize, block: Block) -> Result<Vec<Vec<f64>>, String> {
    let (data, k) = block;
    let flat = spatex_core::bytes_to_f64s(data.as_slice())?;
    if flat.len() != n * k {
        return Err(format!(
            "expected {n} rows x {k} columns, got {} values",
            flat.len()
        ));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    Ok(flat.chunks(n).map(<[f64]>::to_vec).collect())
}

/// Two-stage least squares (`spreg.TSLS`). `x`, `yend` and `q` are `{binary, columns}` blocks
/// (column-major); `x` includes the constant. `robust` is 0 (none), 1 (White) or 2 (HAC, needs
/// `gwk`).
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn regression_tsls<'a>(
    env: Env<'a>,
    y: Binary,
    x: Block,
    yend: Block,
    q: Block,
    robust: u8,
    sig2n_k: bool,
    gwk: Option<Graph>,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let n = y.len();
    let x = optional_columns(n, x)?;
    let yend = optional_columns(n, yend)?;
    let q = optional_columns(n, q)?;
    let gwk = gwk.as_ref().map(|g| &g.0);
    if let Some(g) = gwk {
        if g.n != n {
            return Err(format!("gwk has {} nodes but y has {n} rows", g.n));
        }
    }
    let var = VarOptions {
        robust: robust_of(robust)?,
        sig2n_k,
        gwk,
    };
    Ok(transport(env, tsls_mod::tsls(&y, &x, &yend, &q, var)?))
}

/// Settings of [`regression_gm_lag`].
#[derive(NifMap)]
pub struct GmLagSettings {
    w_lags: usize,
    lag_q: bool,
    robust: u8,
    sig2n_k: bool,
    spat_diag: bool,
    impacts: u8,
    hard_bound: bool,
    skip_const: bool,
    slx_lags: usize,
    kx_base: usize,
}

/// Spatial 2SLS lag model (`spreg.GM_Lag`). Blocks as in [`regression_tsls`]; `x` includes the
/// constant.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn regression_gm_lag<'a>(
    env: Env<'a>,
    y: Binary,
    x: Block,
    yend: Block,
    q: Block,
    graph: Graph,
    gwk: Option<Graph>,
    settings: GmLagSettings,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let n = y.len();
    let x = optional_columns(n, x)?;
    let yend = optional_columns(n, yend)?;
    let q = optional_columns(n, q)?;
    let gwk = gwk.as_ref().map(|g| &g.0);
    if let Some(g) = gwk {
        if g.n != n {
            return Err(format!("gwk has {} nodes but y has {n} rows", g.n));
        }
    }
    let entries = gm_lag::gm_lag(
        &y,
        &x,
        &yend,
        &q,
        &GmLagOptions {
            w: &graph.0,
            w_lags: settings.w_lags,
            lag_q: settings.lag_q,
            robust: robust_of(settings.robust)?,
            gwk,
            sig2n_k: settings.sig2n_k,
            spat_diag: settings.spat_diag,
            impacts: Impacts::from_code(settings.impacts)?,
            hard_bound: settings.hard_bound,
            skip_const: settings.skip_const,
            slx_lags: settings.slx_lags,
            kx_base: settings.kx_base,
        },
    )?;
    Ok(transport(env, entries))
}

/// The endogenous variables and instruments of a spatial lag model (`set_endog`): returns
/// `yend` (`W y` last) and `q`, both column-major. `x` excludes the constant.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn regression_lag_endog<'a>(
    env: Env<'a>,
    y: Binary,
    x: Block,
    yend: Block,
    q: Block,
    graph: Graph,
    w_lags: usize,
    lag_q: bool,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let n = y.len();
    let x = optional_columns(n, x)?;
    let yend = optional_columns(n, yend)?;
    let q = optional_columns(n, q)?;
    if graph.0.n != n {
        return Err(format!("graph has {} nodes but y has {n} rows", graph.0.n));
    }
    let (yend2, q2) = gm_common::set_endog(&graph.0, &y, &x, &yend, &q, w_lags, lag_q)?;
    Ok(vec![
        ("yend".to_string(), f64_binary(env, &yend2.concat())),
        ("q".to_string(), f64_binary(env, &q2.concat())),
    ])
}

/// Reduced-form prediction of a spatial lag fit (`sp_att`): `predy_e`, `e_pred` and, when
/// `|rho| >= 1` (nothing is computed), `warn_rho`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_sp_att<'a>(
    env: Env<'a>,
    y: Binary,
    predy: Binary,
    wy: Binary,
    graph: Graph,
    rho: f64,
    hard_bound: bool,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let predy = spatex_core::bytes_to_f64s(predy.as_slice())?;
    let wy = spatex_core::bytes_to_f64s(wy.as_slice())?;
    let n = y.len();
    if predy.len() != n || wy.len() != n || graph.0.n != n {
        return Err("y, predy, wy and the graph must have the same length".to_string());
    }
    let att = gm_common::sp_att(&graph.0, &y, &predy, &wy, rho, hard_bound)?;
    let mut out = vec![
        ("predy_e".to_string(), f64_binary(env, &att.predy_e)),
        ("e_pred".to_string(), f64_binary(env, &att.e_pred)),
    ];
    if att.out_of_bounds {
        out.push(("warn_rho".to_string(), f64_binary(env, &[1.0])));
    }
    Ok(out)
}

/// Settings of [`regression_lag_diag`].
#[derive(NifMap)]
pub struct LagDiagSettings {
    spat_diag: bool,
    impacts: u8,
}

/// Anselin-Kelejian test, `pr2_e` and impact multipliers of a spatial lag fit assembled by the
/// caller (regimes models): `z` is the regressor block of the 2SLS fit, `varb` its covariance
/// (row-major), `u` its residuals.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn regression_lag_diag<'a>(
    env: Env<'a>,
    y: Binary,
    predy_e: Binary,
    z: Block,
    varb: Binary,
    u: Binary,
    graph: Graph,
    rho: f64,
    settings: LagDiagSettings,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let n = y.len();
    let predy_e = spatex_core::bytes_to_f64s(predy_e.as_slice())?;
    let u = spatex_core::bytes_to_f64s(u.as_slice())?;
    let varb = spatex_core::bytes_to_f64s(varb.as_slice())?;
    let z = optional_columns(n, z)?;
    if predy_e.len() != n || u.len() != n || graph.0.n != n {
        return Err("y, predy_e, u and the graph must have the same length".to_string());
    }
    let entries = gm_lag::lag_diagnostics(
        &graph.0,
        &y,
        &predy_e,
        &z,
        &varb,
        &u,
        rho,
        &DiagOptions {
            spat_diag: settings.spat_diag,
            impacts: Impacts::from_code(settings.impacts)?,
        },
    )?;
    Ok(transport(env, entries))
}

/// Kelejian-Prucha lambda from first-stage residuals (`lam`, and `degenerate` when the moment
/// scale was zero): the first step of the regimes error models.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_error_lambda<'a>(
    env: Env<'a>,
    u: Binary,
    graph: Graph,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let u = spatex_core::bytes_to_f64s(u.as_slice())?;
    let o = gm_error::kp98_lambda(&graph.0, &u)?;
    Ok(vec![
        ("lam".to_string(), f64_binary(env, &[o.lam])),
        (
            "degenerate".to_string(),
            f64_binary(env, &[f64::from(u8::from(o.degenerate))]),
        ),
    ])
}

/// The SUR spatial error ML step (`spreg.SURerrorML`): the concentrated likelihood of the lambdas
/// given the `k` residual columns `e` (column-major, `n` values each), minimized from `lam0`
/// (`optimize`) or only evaluated there. Returns `lam`, `fun` (the negative concentrated
/// log-likelihood), `iterations` and `converged`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_sur_error_search<'a>(
    env: Env<'a>,
    e: Binary,
    k: usize,
    graph: Graph,
    lam0: Binary,
    (optimize, max_dense_n): (bool, usize),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let flat = spatex_core::bytes_to_f64s(e.as_slice())?;
    let lam0 = spatex_core::bytes_to_f64s(lam0.as_slice())?;
    if k == 0 || lam0.len() != k || flat.len() % k != 0 {
        return Err(format!("expected {k} residual columns and {k} lambdas"));
    }
    let cols: Vec<Vec<f64>> = flat.chunks(flat.len() / k).map(<[f64]>::to_vec).collect();
    let problem = sur_error_ml::Problem::new(&graph.0, &cols, max_dense_n)?;
    let (lam, fun, iterations, converged) = if optimize {
        let f = sur_error_ml::minimize(&problem, &lam0)?;
        (f.lam, f.fun, f.iterations, f.converged)
    } else {
        let fun = problem.objective(&lam0);
        (lam0, fun, 0, true)
    };
    Ok(vec![
        ("lam".to_string(), f64_binary(env, &lam)),
        ("fun".to_string(), f64_binary(env, &[fun])),
        (
            "iterations".to_string(),
            f64_binary(env, &[iterations as f64]),
        ),
        (
            "converged".to_string(),
            f64_binary(env, &[f64::from(u8::from(converged))]),
        ),
    ])
}

/// Trace terms of the asymptotic variance of the SUR spatial error ML model (`surerrvm`):
/// `tr_d`, `tr_dd` (one per equation) and `cross` (`k x k`, row-major).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_sur_error_traces<'a>(
    env: Env<'a>,
    graph: Graph,
    lam: Binary,
    max_dense_n: usize,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let lam = spatex_core::bytes_to_f64s(lam.as_slice())?;
    let t = sur_error_ml::trace_terms(&graph.0, &lam, max_dense_n)?;
    Ok(vec![
        ("tr_d".to_string(), f64_binary(env, &t.tr_d)),
        ("tr_dd".to_string(), f64_binary(env, &t.tr_dd)),
        ("cross".to_string(), f64_binary(env, &t.cross.concat())),
    ])
}

/// The spatial filter `v - lambda W v` of every column of a block (`get_spFilter`), column-major.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_sp_filter<'a>(
    env: Env<'a>,
    cols: Block,
    graph: Graph,
    lam: f64,
) -> Result<Binary<'a>, String> {
    let n = graph.0.n;
    let cols = optional_columns(n, cols)?;
    let out = gm_common::filter_cols(&graph.0, &cols, lam)?;
    Ok(f64_binary(env, &out.concat()))
}

/// 2SLS with an explicit instrument matrix `h` (second stage of the regimes error models; OLS
/// with no `yend` and `h = x`). Blocks as in [`regression_tsls`].
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_tsls_h<'a>(
    env: Env<'a>,
    y: Binary,
    x: Block,
    yend: Block,
    h: Block,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let n = y.len();
    let x = optional_columns(n, x)?;
    let yend = optional_columns(n, yend)?;
    let h = optional_columns(n, h)?;
    Ok(transport(env, tsls_mod::tsls_h(&y, &x, &yend, &h)?))
}

/// Regime design of [`GmErrorSettings`].
#[derive(NifMap)]
pub struct RegimesSettings {
    index: Vec<usize>,
    nr: usize,
    x_flags: Vec<bool>,
    y_flags: Vec<bool>,
}

/// Settings of [`regression_gm_error`]: `estimator` 0 KP98 / 1 Het / 2 Hom, `a1` 0 hom_sc /
/// 1 hom / 2 het, `combo` adds `W y` as an endogenous regressor, `at` replaces the optimizer,
/// `regimes` (Het and Hom) fits a pooled regimes model on the base columns given.
#[derive(NifMap)]
pub struct GmErrorSettings {
    estimator: u8,
    max_iter: usize,
    epsilon: f64,
    step1c: bool,
    a1: u8,
    hard_bound: bool,
    combo: bool,
    w_lags: usize,
    lag_q: bool,
    skip_const: bool,
    at: Option<Vec<f64>>,
    regimes: Option<RegimesSettings>,
}

/// GMM spatial error models and their combo / endogenous variants. Blocks as in
/// [`regression_tsls`]; `x` includes the constant.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn regression_gm_error<'a>(
    env: Env<'a>,
    y: Binary,
    x: Block,
    yend: Block,
    q: Block,
    graph: Graph,
    settings: GmErrorSettings,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let n = y.len();
    let x = optional_columns(n, x)?;
    let yend = optional_columns(n, yend)?;
    let q = optional_columns(n, q)?;
    let estimator = match settings.estimator {
        0 => Estimator::Kp98,
        1 => Estimator::Het,
        2 => Estimator::Hom,
        other => return Err(format!("unknown estimator code {other}")),
    };
    let a1 = match settings.a1 {
        0 => A1Kind::HomSc,
        1 => A1Kind::Hom,
        2 => A1Kind::Het,
        other => return Err(format!("unknown A1 code {other}")),
    };
    let combo = settings.combo.then_some(Combo {
        w_lags: settings.w_lags,
        lag_q: settings.lag_q,
        skip_const: settings.skip_const,
    });
    let entries = gm_error::gm_error(
        &y,
        &x,
        &yend,
        &q,
        &gm_error::Options {
            w: &graph.0,
            estimator,
            max_iter: settings.max_iter,
            epsilon: settings.epsilon,
            step1c: settings.step1c,
            a1,
            hard_bound: settings.hard_bound,
            combo,
            at: settings.at,
            regimes: settings.regimes.map(|r| gm_error::Regimes {
                index: r.index,
                nr: r.nr,
                x_flags: r.x_flags,
                y_flags: r.y_flags,
            }),
        },
    )?;
    Ok(transport(env, entries))
}
