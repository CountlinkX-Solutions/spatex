//! NIFs for local Geary (univariate and multivariate), local join counts and LOSH
//! (`Spatex.ESDA.geary_local/3`, `geary_local_mv/3`, `join_counts_local/3`, `losh/3`).
//!
//! Per-site results travel as one native-endian `f64` binary of consecutive blocks of `n`
//! values (NaN-safe); the reference distributions (`n x P`, site-major) are a second binary,
//! empty unless requested.

use rustler::{Binary, Env};
use spatex_core::esda::{geary_local, geary_local_mv, join_counts_local, losh};
use spatex_core::perm::crand::{Crand, Settings};
use spatex_core::perm::Alternative;

use crate::{f64_binary, Graph};

fn settings(
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
    island_weight: f64,
) -> Result<Settings, String> {
    Ok(Settings {
        permutations,
        seed,
        alternative: Alternative::parse(alternative)?,
        island_weight,
        keep,
    })
}

fn sims<'a>(env: Env<'a>, c: Option<&Crand>) -> Binary<'a> {
    f64_binary(env, c.and_then(|c| c.sims.as_deref()).unwrap_or(&[]))
}

/// Local Geary on a row-standardised graph. Returns `{blocks, sims}`; blocks are `local_g`,
/// then (when `permutations > 0`) `p_sim` and (when `sig > 0`) `labs`.
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_geary_local<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
    sig: f64,
    island_weight: f64,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let x = spatex_core::bytes_to_f64s(x.as_slice())?;
    let m = geary_local::geary_local(&x, &graph.0)?;
    let mut out = m.local_g.clone();
    let mut crand = None;
    if permutations > 0 {
        let s = settings(permutations, seed, alternative, keep, island_weight)?;
        let c = geary_local::geary_local_permutations(&x, &graph.0, &s)?;
        out.extend(&c.p_sim);
        if sig > 0.0 {
            out.extend(geary_local::labels(&m.local_g, &x, &c.p_sim, sig));
        }
        crand = Some(c);
    }
    Ok((f64_binary(env, &out), sims(env, crand.as_ref())))
}

/// Multivariate local Geary on a row-standardised graph; `vars` holds the `k` variables one
/// after the other. Returns `{blocks, sims}`; blocks are `local_g` then (when
/// `permutations > 0`) `p_sim` (esda's folded, directed p-value).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_geary_local_mv<'a>(
    env: Env<'a>,
    graph: Graph,
    vars: Binary,
    k: usize,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let flat = spatex_core::bytes_to_f64s(vars.as_slice())?;
    let n = graph.0.n;
    if k == 0 || flat.len() != k * n {
        return Err(format!(
            "expected {k} variables of length {n} (length mismatch), got {} values",
            flat.len()
        ));
    }
    let vars: Vec<Vec<f64>> = flat.chunks(n).map(<[f64]>::to_vec).collect();
    let mut out = geary_local_mv::geary_local_mv(&vars, &graph.0)?;
    let mut crand = None;
    if permutations > 0 {
        let s = settings(permutations, seed, "directed", keep, 0.0)?;
        let c = geary_local_mv::geary_local_mv_permutations(&vars, &graph.0, &s)?;
        out.extend(&c.p_sim);
        crand = Some(c);
    }
    Ok((f64_binary(env, &out), sims(env, crand.as_ref())))
}

/// Local join counts. Returns `{blocks, sims}`; blocks are `ljc` then (when
/// `permutations > 0`) `p_sim`.
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_join_counts_local<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
    island_weight: f64,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let mut out = join_counts_local::join_counts_local(&y, &graph.0)?;
    let mut crand = None;
    if permutations > 0 {
        let s = settings(permutations, seed, alternative, keep, island_weight)?;
        let c = join_counts_local::join_counts_local_permutations(&y, &graph.0, &s)?;
        out.extend(&c.p_sim);
        crand = Some(c);
    }
    Ok((f64_binary(env, &out), sims(env, crand.as_ref())))
}

/// Bivariate local join counts, `case="BJC"`. Returns `{blocks, sims}` like
/// [`esda_join_counts_local`].
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_join_counts_local_bjc<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    z: Binary,
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
    island_weight: f64,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let x = spatex_core::bytes_to_f64s(x.as_slice())?;
    let z = spatex_core::bytes_to_f64s(z.as_slice())?;
    let mut out = join_counts_local::join_counts_local_bjc(&x, &z, &graph.0)?;
    let mut crand = None;
    if permutations > 0 {
        let s = settings(permutations, seed, alternative, keep, island_weight)?;
        let c = join_counts_local::join_counts_local_bjc_permutations(&x, &z, &graph.0, &s)?;
        out.extend(&c.p_sim);
        crand = Some(c);
    }
    Ok((f64_binary(env, &out), sims(env, crand.as_ref())))
}

/// LOSH. Returns blocks `hi, ylag, yresid, var_hi` and, with `chi2`, `pval`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_losh<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    a: f64,
    chi2: bool,
) -> Result<Binary<'a>, String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let l = losh::losh(&y, &graph.0, a)?;
    let mut out = Vec::new();
    for block in [&l.hi, &l.ylag, &l.yresid, &l.var_hi] {
        out.extend(block);
    }
    if chi2 {
        out.extend(losh::chi2_pvalues(&l.hi, &l.var_hi));
    }
    Ok(f64_binary(env, &out))
}
