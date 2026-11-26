//! Geographically weighted regression NIFs (`Spatex.GWR`): thin transport over
//! `spatex_core::gwr`.

use rustler::{Binary, Env};
use spatex_core::gwr::{
    self, Collinearity, Criterion, Exog, Family, Fit, Kernel, MgwrFit, MgwrOptions, Model, Search,
    SelectOptions,
};

use crate::f64_binary;

fn f64s(b: &Binary) -> Result<Vec<f64>, String> {
    spatex_core::bytes_to_f64s(b.as_slice())
}

/// Column-major (`k` columns of `n`) to row-major `n x k`.
fn row_major(flat: &[f64], n: usize, k: usize) -> Result<Vec<f64>, String> {
    if k == 0 || flat.len() != n * k {
        return Err(format!(
            "x has {} values, expected {n} rows x {k} columns",
            flat.len()
        ));
    }
    let mut out = vec![0.0; n * k];
    for (c, col) in flat.chunks(n).enumerate() {
        for (r, v) in col.iter().enumerate() {
            out[r * k + c] = *v;
        }
    }
    Ok(out)
}

fn build_model<'a>(
    coords: &'a [f64],
    y: &'a [f64],
    x: &'a [f64],
    k: usize,
    model: &(String, bool, bool, usize),
    offset: &'a [f64],
    glm: &(String, f64, usize),
) -> Result<Model<'a>, String> {
    let kernel = Kernel::parse(&model.0)?;
    Model::new(coords, y, x, k, kernel, model.1, model.2)?.with_family(
        Family::parse(&glm.0)?,
        offset,
        glm.1,
        glm.2,
    )
}

fn entries<'a>(env: Env<'a>, f: &Fit) -> Vec<(String, Binary<'a>)> {
    let c = &f.criteria;
    let scalars = |name: &str, v: f64| (name.to_string(), f64_binary(env, &[v]));
    let blocks = |name: &str, v: &[f64]| (name.to_string(), f64_binary(env, v));
    vec![
        blocks("params", &f.params),
        blocks("bse", &f.bse),
        blocks("tvalues", &f.tvalues),
        blocks("local_r2", &f.local_r2),
        blocks("deviance", &f.deviance),
        blocks("resid_deviance", &f.resid_deviance),
        blocks("p_dev", &f.p_dev),
        blocks(
            "iterations",
            &f.iterations
                .iter()
                .map(|&i| f64::from(i))
                .collect::<Vec<_>>(),
        ),
        blocks("predy", &f.predy),
        blocks("resid", &f.resid),
        blocks("influ", &f.influ),
        blocks("std_res", &f.std_res),
        blocks("cooks_d", &f.cooks_d),
        blocks("hat", &f.hat),
        blocks("adj_alpha", &f.adj_alpha),
        scalars("resid_ss", f.resid_ss),
        scalars("tr_s", f.tr_s),
        scalars("tr_sts", f.tr_sts),
        scalars("enp", f.enp),
        scalars("sigma2", f.sigma2),
        scalars("scale", f.scale),
        scalars("d2", f.d2),
        scalars("adj_d2", f.adj_d2),
        scalars("null_deviance", f.null_deviance),
        scalars("global_deviance", f.global_deviance),
        scalars("llf", c.llf),
        scalars("aic", c.aic),
        scalars("aicc", c.aicc),
        scalars("bic", c.bic),
        scalars("cv", f.cv),
        scalars("r2", f.r2),
        scalars("adj_r2", f.adj_r2),
        scalars("df_resid", f.df_resid),
        scalars("critical_tval", f.critical_tval(None)),
    ]
}

/// Full GWR fit.
///
/// `x` holds `k` columns of `n` values and already contains the intercept. `model` is
/// `{kernel, fixed, spherical}`; `glm` is `{family, iwls_tol, iwls_max_iter}` with the Poisson
/// `offset` as an f64 binary (empty for ones); `bw` is `{bandwidth, global}` where `global`
/// means the infinite bandwidth. Returns `[{name, f64_binary}]`; per-location matrices are
/// row-major `n x k`. `flags` is `{sigma2_v1, hat_matrix}`; `hat_matrix` additionally returns the row-major `n x n` hat matrix as
/// `"hat"` (empty otherwise).
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn gwr_fit<'a>(
    env: Env<'a>,
    coords: Binary,
    y: Binary,
    x: Binary,
    model: (String, bool, bool, usize),
    glm: (String, f64, usize),
    offset: Binary,
    bw: (f64, bool),
    flags: (bool, bool),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let coords = f64s(&coords)?;
    let y = f64s(&y)?;
    let offset = f64s(&offset)?;
    let k = model.3;
    let x = row_major(&f64s(&x)?, y.len(), k)?;
    let m = build_model(&coords, &y, &x, k, &model, &offset, &glm)?;
    let bandwidth = if bw.1 { f64::INFINITY } else { bw.0 };
    let (sigma2_v1, hat_matrix) = flags;
    let fit = gwr::fit_full_with(&m, bandwidth, sigma2_v1, hat_matrix)?;
    Ok(entries(env, &fit))
}

/// Bandwidth selection. `search` is `{criterion, method, step, include_global, bw_min, bw_max}`
/// with `method` one of `"golden"`, `"interval"` (needs `step`) and `"scipy"`; `shape` is
/// `{n_columns, constant, tol, max_iter}` where `n_columns` counts the local and the `X_glob`
/// columns (mgwr's `_init_section`). Returns `{[bw, score, nfev, status], history}` with
/// `history` the flat `[bw0, score0, bw1, score1, ...]` evaluation list; `nfev` and `status`
/// are the scipy optimizer's (both `-1` for the other searches).
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gwr_select<'a>(
    env: Env<'a>,
    coords: Binary,
    y: Binary,
    x: Binary,
    model: (String, bool, bool, usize),
    glm: (String, f64, usize),
    offset: Binary,
    search: (String, String, f64, bool, Option<f64>, Option<f64>),
    shape: (usize, bool, f64, usize),
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let coords = f64s(&coords)?;
    let y = f64s(&y)?;
    let offset = f64s(&offset)?;
    let k = model.3;
    let x = row_major(&f64s(&x)?, y.len(), k)?;
    let m = build_model(&coords, &y, &x, k, &model, &offset, &glm)?;
    let (criterion, method, step, include_global, bw_min, bw_max) = search;
    let bw_max = if include_global {
        Some(f64::INFINITY)
    } else {
        bw_max
    };
    let (n_loc, constant, tol, max_iter) = shape;
    let options = SelectOptions {
        criterion: Criterion::parse(&criterion)?,
        search: match method.as_str() {
            "interval" => Search::Interval(step),
            "scipy" => Search::Scipy,
            _ => Search::Golden,
        },
        bw_min,
        bw_max,
        n_loc,
        constant,
        tol,
        max_iter,
    };
    let sel = gwr::select_bandwidth(&m, &options)?;
    let history: Vec<f64> = sel.history.iter().flat_map(|(b, s)| [*b, *s]).collect();
    let (nfev, status) = sel
        .optimizer
        .map_or((-1.0, -1.0), |o| (o.nfev as f64, f64::from(o.status)));
    Ok((
        f64_binary(env, &[sel.bw, sel.score, nfev, status]),
        f64_binary(env, &history),
    ))
}

/// The row permutation of replicate `replicate` of a seeded stream, as `n` native `f64` indices.
#[rustler::nif]
pub fn gwr_permutation<'a>(env: Env<'a>, n: usize, seed: u64, replicate: u64) -> Binary<'a> {
    let rows: Vec<f64> = spatex_core::perm::permutation(n, seed, replicate)
        .into_iter()
        .map(f64::from)
        .collect();
    f64_binary(env, &rows)
}

/// `t.ppf(1 - alpha / 2, df)`.
#[rustler::nif]
pub fn gwr_t_critical(alpha: f64, df: f64) -> f64 {
    gwr::t_critical(alpha, df)
}

fn mgwr_entries<'a>(env: Env<'a>, f: &MgwrFit) -> Vec<(String, Binary<'a>)> {
    let scalars = |name: &str, v: f64| (name.to_string(), f64_binary(env, &[v]));
    let blocks = |name: &str, v: &[f64]| (name.to_string(), f64_binary(env, v));
    let history: Vec<f64> = f.bws_history.iter().flatten().copied().collect();
    // `[count, bw0, score0, ..., count, ...]` per search.
    let sel_hist: Vec<f64> = f
        .sel_hist
        .iter()
        .flat_map(|h| std::iter::once(h.len() as f64).chain(h.iter().flat_map(|(b, s)| [*b, *s])))
        .collect();
    let mut out = vec![
        blocks("bws", &f.bws),
        blocks("bws_history", &history),
        blocks("sel_hist", &sel_hist),
        blocks("scores", &f.scores),
        blocks("params", &f.params),
        blocks("predy", &f.predy),
        blocks("resid", &f.resid),
        scalars("bw_init", f.bw_init),
        scalars("resid_ss", f.resid_ss),
        scalars("converged", if f.converged { 1.0 } else { 0.0 }),
    ];
    if let Some(i) = &f.inference {
        out.extend([
            blocks("enp_j", &i.enp_j),
            blocks("cct", &i.cct),
            blocks("bse", &i.bse),
            blocks("tvalues", &i.tvalues),
            blocks("adj_alpha_j", &i.adj_alpha_j),
            blocks("critical_tval", &i.critical_tval),
            scalars("tr_s", i.tr_s),
            scalars("sigma2", i.sigma2),
            scalars("llf", i.criteria.llf),
            scalars("aic", i.criteria.aic),
            scalars("aicc", i.criteria.aicc),
            scalars("bic", i.criteria.bic),
            scalars("r2", i.r2),
            scalars("adj_r2", i.adj_r2),
        ]);
    }
    out
}

/// Multiscale GWR (Gaussian): bandwidth backfitting and optional hat-matrix inference.
///
/// `x` holds `k` columns of `n` values including the intercept. `search` is
/// `{criterion, interval_step, tol, max_iter}` (step `0` is the golden section), `bounds` the
/// per-column `{min, max}` lists and the optional starting bandwidth, `outer` is
/// `{tol_multi, max_iter_multi, rss_score, bws_same_times}` and `inference` is
/// `{enabled, chunk_size}`. Matrices are row-major `n x k`; `bws_history` is the flat
/// `iterations x k` list.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn mgwr_fit<'a>(
    env: Env<'a>,
    coords: Binary,
    y: Binary,
    x: Binary,
    model: (String, bool, bool, usize),
    search: (String, f64, f64, usize),
    bounds: (Vec<Option<f64>>, Vec<Option<f64>>, Option<f64>),
    outer: (f64, usize, bool, usize),
    inference: (bool, usize),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let coords = f64s(&coords)?;
    let y = f64s(&y)?;
    let k = model.3;
    let x = row_major(&f64s(&x)?, y.len(), k)?;
    let kernel = Kernel::parse(&model.0)?;
    let m = Model::new(&coords, &y, &x, k, kernel, model.1, model.2)?;
    let options = MgwrOptions {
        criterion: Criterion::parse(&search.0)?,
        search: if search.1 > 0.0 {
            Search::Interval(search.1)
        } else {
            Search::Golden
        },
        tol: search.2,
        max_iter: search.3,
        multi_bw_min: bounds.0,
        multi_bw_max: bounds.1,
        init_bw: bounds.2,
        tol_multi: outer.0,
        max_iter_multi: outer.1,
        rss_score: outer.2,
        bws_same_times: outer.3,
        inference: inference.0,
        chunk_size: inference.1,
    };
    let fit = gwr::fit_mgwr(&m, &options)?;
    Ok(mgwr_entries(env, &fit))
}

/// GWR prediction at new locations (`GWR.predict`).
///
/// The calibration data and model settings are as in `gwr_fit` (`bw` is `{bandwidth, global,
/// sigma2_v1}`); `new` is `{points, p, exog}` with the `m` prediction points interleaved, `p`
/// holding `k` columns of `m` regressor values (the intercept column included) and `exog` the
/// optional `{scale, resid}` of the calibration sample. Returns `[{name, f64_binary}]`; matrices are row-major `m x k`.
#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn gwr_predict<'a>(
    env: Env<'a>,
    coords: Binary,
    y: Binary,
    x: Binary,
    model: (String, bool, bool, usize),
    glm: (String, f64, usize),
    offset: Binary,
    bw: (f64, bool, bool),
    new: (Binary, Binary, Option<(f64, Binary)>),
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let coords = f64s(&coords)?;
    let y = f64s(&y)?;
    let offset = f64s(&offset)?;
    let k = model.3;
    let x = row_major(&f64s(&x)?, y.len(), k)?;
    let m = build_model(&coords, &y, &x, k, &model, &offset, &glm)?;
    let points = f64s(&new.0)?;
    let rows = points.len() / 2;
    let p = row_major(&f64s(&new.1)?, rows, k)?;
    let exog = new.2;
    let resid = match &exog {
        Some((_, r)) => f64s(r)?,
        None => Vec::new(),
    };
    let exog = exog.as_ref().map(|(scale, _)| Exog {
        scale: *scale,
        resid: &resid,
    });
    let bandwidth = if bw.1 { f64::INFINITY } else { bw.0 };
    let pred = gwr::predict(&m, bandwidth, bw.2, &points, &p, exog)?;
    Ok(vec![
        ("params".to_string(), f64_binary(env, &pred.params)),
        ("bse".to_string(), f64_binary(env, &pred.bse)),
        ("tvalues".to_string(), f64_binary(env, &pred.tvalues)),
        (
            "predictions".to_string(),
            f64_binary(env, &pred.predictions),
        ),
        ("local_r2".to_string(), f64_binary(env, &pred.local_r2)),
        ("scale".to_string(), f64_binary(env, &[pred.scale])),
    ])
}

fn collinearity_entries<'a>(env: Env<'a>, c: &Collinearity) -> Vec<(String, Binary<'a>)> {
    vec![
        ("corr".to_string(), f64_binary(env, &c.corr)),
        ("vif".to_string(), f64_binary(env, &c.vif)),
        (
            "condition_number".to_string(),
            f64_binary(env, &c.condition_number),
        ),
        ("vdp".to_string(), f64_binary(env, &c.vdp)),
        (
            "shape".to_string(),
            f64_binary(env, &[c.ncor as f64, c.nvif as f64]),
        ),
    ]
}

/// Local collinearity of the geographically weighted design (`GWRResults.local_collinearity`).
///
/// `x` holds `k` columns of `n` values, the intercept included; `bw` is `nil` for the global
/// bandwidth; `constant` says that the first column is the intercept. The response plays no
/// part, so none is passed.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn gwr_collinearity<'a>(
    env: Env<'a>,
    coords: Binary,
    x: Binary,
    model: (String, bool, bool, usize),
    bw: Option<f64>,
    constant: bool,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let coords = f64s(&coords)?;
    let n = coords.len() / 2;
    let k = model.3;
    let x = row_major(&f64s(&x)?, n, k)?;
    let y = vec![0.0; n];
    let kernel = Kernel::parse(&model.0)?;
    let m = Model::new(&coords, &y, &x, k, kernel, model.1, model.2)?;
    let c = gwr::local_collinearity(&m, bw.unwrap_or(f64::INFINITY), constant)?;
    Ok(collinearity_entries(env, &c))
}

/// Local collinearity of an MGWR design (`MGWRResults.local_collinearity`): one bandwidth per
/// column, `nil` for the global one.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn mgwr_collinearity<'a>(
    env: Env<'a>,
    coords: Binary,
    x: Binary,
    model: (String, bool, bool, usize),
    bws: Vec<Option<f64>>,
) -> Result<Vec<(String, Binary<'a>)>, String> {
    let coords = f64s(&coords)?;
    let n = coords.len() / 2;
    let k = model.3;
    let x = row_major(&f64s(&x)?, n, k)?;
    let y = vec![0.0; n];
    let kernel = Kernel::parse(&model.0)?;
    let m = Model::new(&coords, &y, &x, k, kernel, model.1, model.2)?;
    let bws: Vec<f64> = bws.iter().map(|b| b.unwrap_or(f64::INFINITY)).collect();
    let c = gwr::local_collinearity_mgwr(&m, &bws)?;
    Ok(collinearity_entries(env, &c))
}
