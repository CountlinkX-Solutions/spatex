//! NIFs for the local ESDA statistics (`Spatex.ESDA.moran_local/3` and friends).
//!
//! Per-site results travel as one native-endian `f64` binary holding consecutive blocks of `n`
//! values (NaN-safe, see `esda_global`); the Elixir side slices it with the documented block
//! order. The reference distributions (`n x P`, site-major) come back as a second binary,
//! empty unless requested.

use rustler::{Binary, Env};
use spatex_core::esda::fdr;
use spatex_core::esda::g_local::{self, GTransform, Star};
use spatex_core::esda::local_moran::{self, LocalMoran};
use spatex_core::perm::crand::{Crand, Settings};
use spatex_core::perm::Alternative;

use crate::{f64_binary, Graph};

fn settings(
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
) -> Result<Settings, String> {
    settings_with(permutations, seed, alternative, 0.0, keep)
}

fn settings_with(
    permutations: usize,
    seed: u64,
    alternative: &str,
    island_weight: f64,
    keep: bool,
) -> Result<Settings, String> {
    Ok(Settings {
        permutations,
        seed,
        alternative: Alternative::parse(alternative)?,
        island_weight,
        keep,
    })
}

/// Observed blocks `is, q, z, lag` of a local Moran result.
fn push_observed(out: &mut Vec<f64>, m: &LocalMoran) {
    out.extend(&m.is);
    out.extend(m.q.iter().map(|&q| f64::from(q)));
    out.extend(&m.z);
    out.extend(&m.lag);
}

/// Permutation blocks `p_sim, ei_sim, se_i_sim, vi_sim, z_sim, p_z_sim`.
fn push_perm(out: &mut Vec<f64>, c: &Crand) {
    out.extend(&c.p_sim);
    out.extend(c.summary.iter().map(|s| s.ei_sim));
    out.extend(c.summary.iter().map(|s| s.se_sim));
    out.extend(c.summary.iter().map(|s| s.vi_sim));
    out.extend(c.summary.iter().map(|s| s.z_sim));
    out.extend(c.summary.iter().map(|s| s.p_z_sim));
}

fn sims<'a>(env: Env<'a>, c: &Crand) -> Binary<'a> {
    f64_binary(env, c.sims.as_deref().unwrap_or(&[]))
}

/// Local Moran's I. Returns `{blocks, sims}` with blocks
/// `is, q, z, lag, ei, vi, eic, vic` followed (when `permutations > 0`) by the permutation blocks.
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_moran_local<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    geoda_quads: bool,
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let m = local_moran::local_moran(&y, &graph.0, geoda_quads)?;
    let mut out = Vec::new();
    push_observed(&mut out, &m);
    let moments = m.moments.as_ref().ok_or("missing analytic moments")?;
    for block in [&moments.ei, &moments.vi, &moments.eic, &moments.vic] {
        out.extend(block);
    }
    let mut sim = f64_binary(env, &[]);
    if permutations > 0 {
        let s = settings(permutations, seed, alternative, keep)?;
        let c = local_moran::local_moran_permutations(&y, &graph.0, &s)?;
        push_perm(&mut out, &c);
        sim = sims(env, &c);
    }
    Ok((f64_binary(env, &out), sim))
}

/// Bivariate local Moran's I. Returns `{blocks, sims}` with blocks `is, q, z, lag, zy`
/// followed (when `permutations > 0`) by the permutation blocks.
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_moran_local_bv<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    y: Binary,
    geoda_quads: bool,
    permutations: usize,
    seed: u64,
    alternative: &str,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let x = spatex_core::bytes_to_f64s(x.as_slice())?;
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let m = local_moran::local_moran_bv(&x, &y, &graph.0, geoda_quads)?;
    let mut out = Vec::new();
    push_observed(&mut out, &m);
    out.extend(m.zy.as_ref().ok_or("missing zy")?);
    let mut sim = f64_binary(env, &[]);
    if permutations > 0 {
        let s = settings(permutations, seed, alternative, keep)?;
        let c = local_moran::local_moran_bv_permutations(&x, &y, &graph.0, &s)?;
        push_perm(&mut out, &c);
        sim = sims(env, &c);
    }
    Ok((f64_binary(env, &out), sim))
}

/// Local Getis-Ord `G` / `G*`. The graph is the raw input graph (its diagonal and transform
/// matter). `prep` is `{input_binary, star_kind, star_weight, transform}` where `star_kind` is
/// `infer | off | on | weight` (with `star_weight` for the last one).
/// Returns `{blocks, sims, star, y_sum}` with blocks `gs, egs, vgs, zs, p_norm` followed (when
/// `permutations > 0`) by the permutation blocks.
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_g_local<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    prep: (bool, &str, f64, &str),
    permutations: usize,
    seed: u64,
    alternative: &str,
    island_weight: f64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>, bool, f64), String> {
    let (input_binary, star_kind, star_weight, transform) = prep;
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let opts = g_local::Options {
        star: Star::parse(star_kind, star_weight)?,
        transform: GTransform::parse(transform)?,
        input_binary,
    };
    let r = g_local::g_local(&y, &graph.0, &opts)?;
    let mut out = Vec::new();
    for block in [&r.gs, &r.egs, &r.vgs, &r.zs, &r.p_norm] {
        out.extend(block);
    }
    let mut sim = f64_binary(env, &[]);
    if permutations > 0 {
        let s = settings_with(permutations, seed, alternative, island_weight, keep)?;
        let c = g_local::g_local_permutations(&y, &r, &s)?;
        push_perm(&mut out, &c);
        sim = sims(env, &c);
    }
    Ok((f64_binary(env, &out), sim, r.star, r.y_sum))
}

/// The `esda.fdr` cut-off of a vector of p-values.
#[rustler::nif]
pub fn esda_fdr(p_values: Binary, alpha: f64) -> Result<f64, String> {
    let p = spatex_core::bytes_to_f64s(p_values.as_slice())?;
    fdr::fdr(&p, alpha)
}
