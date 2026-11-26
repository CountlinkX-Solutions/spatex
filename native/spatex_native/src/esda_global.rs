//! NIFs for the global ESDA statistics (`Spatex.ESDA`).
//!
//! Scalar results travel as one native-endian `f64` binary because Erlang floats cannot hold
//! NaN or infinity; the Elixir side decodes them with `Spatex.Input.from_f64_binary/1` and
//! names the fields. The field order of each binary is documented on the NIF.

use rustler::{Binary, Env};
use spatex_core::esda::{geary, getis_ord, join_counts, moran};
use spatex_core::perm::PermTest;

use crate::{f64_binary, Graph};

/// Scalars of a permutation test in the order `p_sim, ei_sim, se_i_sim, vi_sim, z_sim, p_z_sim`.
fn push_perm(out: &mut Vec<f64>, t: &PermTest) {
    out.extend([
        t.p_sim,
        t.summary.ei_sim,
        t.summary.se_sim,
        t.summary.vi_sim,
        t.summary.z_sim,
        t.summary.p_z_sim,
    ]);
}

fn sim_binary<'a>(env: Env<'a>, keep: bool, t: &PermTest) -> Binary<'a> {
    f64_binary(env, if keep { &t.sim } else { &[] })
}

/// Moran's I. Returns `{scalars, sim}` where `scalars` is
/// `i, ei, vi_norm, se_i_norm, z_norm, p_norm, vi_rand, se_i_rand, z_rand, p_rand`
/// followed (when `permutations > 0`) by the permutation scalars.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_moran<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    permutations: usize,
    seed: u64,
    two_tailed: bool,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let a = moran::moran(&y, &graph.0, two_tailed)?;
    let mut out = vec![
        a.i,
        a.ei,
        a.vi_norm,
        a.se_i_norm,
        a.z_norm,
        a.p_norm,
        a.vi_rand,
        a.se_i_rand,
        a.z_rand,
        a.p_rand,
    ];
    let mut sim = f64_binary(env, &[]);
    if permutations > 0 {
        let t = moran::moran_permutations(&y, &graph.0, permutations, seed)?;
        push_perm(&mut out, &t);
        sim = sim_binary(env, keep, &t);
    }
    Ok((f64_binary(env, &out), sim))
}

/// Bivariate Moran's I. Returns `{scalars, sim}` with `scalars` = `i` followed by the
/// permutation scalars when `permutations > 0`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_moran_bv<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    y: Binary,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let x = spatex_core::bytes_to_f64s(x.as_slice())?;
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    if permutations == 0 {
        let i = moran::moran_bv(&x, &y, &graph.0)?;
        return Ok((f64_binary(env, &[i]), f64_binary(env, &[])));
    }
    let t = moran::moran_bv_permutations(&x, &y, &graph.0, permutations, seed)?;
    let i = moran::moran_bv(&x, &y, &graph.0)?;
    let mut out = vec![i];
    push_perm(&mut out, &t);
    Ok((f64_binary(env, &out), sim_binary(env, keep, &t)))
}

/// Rates `events / base`, empirical-Bayes standardised when `adjusted`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_rate<'a>(
    env: Env<'a>,
    events: Binary,
    base: Binary,
    adjusted: bool,
) -> Result<Binary<'a>, String> {
    let events = spatex_core::bytes_to_f64s(events.as_slice())?;
    let base = spatex_core::bytes_to_f64s(base.as_slice())?;
    moran::rate(&events, &base, adjusted).map(|y| f64_binary(env, &y))
}

/// Geary's C. Returns `{scalars, sim}` where `scalars` is
/// `c, ec, vc_norm, se_c_norm, z_norm, p_norm, vc_rand, se_c_rand, z_rand, p_rand`
/// followed (when `permutations > 0`) by `p_sim, ec_sim, se_c_sim, vc_sim, z_sim, p_z_sim`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_geary<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let a = geary::geary(&y, &graph.0)?;
    let mut out = vec![
        a.c,
        a.ec,
        a.vc_norm,
        a.se_c_norm,
        a.z_norm,
        a.p_norm,
        a.vc_rand,
        a.se_c_rand,
        a.z_rand,
        a.p_rand,
    ];
    let mut sim = f64_binary(env, &[]);
    if permutations > 0 {
        let t = geary::geary_permutations(&y, &graph.0, permutations, seed)?;
        push_perm(&mut out, &t);
        sim = sim_binary(env, keep, &t);
    }
    Ok((f64_binary(env, &out), sim))
}

/// Getis-Ord General G (weights are binarized in core). Returns `{scalars, sim}` where
/// `scalars` is `g, eg, vg, se_g, z_norm, p_norm, den_sum` followed (when `permutations > 0`)
/// by `p_sim, eg_sim, se_g_sim, vg_sim, z_sim, p_z_sim`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_getis_ord_g<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let a = getis_ord::general_g(&y, &graph.0)?;
    let mut out = vec![a.g, a.eg, a.vg, a.se_g, a.z_norm, a.p_norm, a.den_sum];
    let mut sim = f64_binary(env, &[]);
    if permutations > 0 {
        let t = getis_ord::general_g_permutations(&y, &graph.0, permutations, seed)?;
        push_perm(&mut out, &t);
        sim = sim_binary(env, keep, &t);
    }
    Ok((f64_binary(env, &out), sim))
}

/// Join counts. Returns `{scalars, sims}`. `scalars` is
/// `j, bb, ww, bw, chi2, chi2_p, chi2_dof`, the crosstab `ww, wb, bw, bb`, the expected table in
/// the same order, `autocorr_pos, autocorr_neg`, followed (when `permutations > 0`) by
/// `min_bb, mean_bb, max_bb, min_bw, mean_bw, max_bw, p_sim_bb, p_sim_bw, p_sim_chi2,
/// p_sim_autocorr_pos, p_sim_autocorr_neg, sim_autocorr_chi2`. `sims` concatenates the
/// simulated `bb`, `ww`, `bw` and `chi2` vectors (empty unless `keep`).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn esda_join_counts<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    drop_islands: bool,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = spatex_core::bytes_to_f64s(y.as_slice())?;
    let a = join_counts::join_counts(&y, &graph.0, drop_islands)?;
    let mut out = vec![a.j, a.bb, a.ww, a.bw, a.chi2, a.chi2_p, a.chi2_dof];
    out.extend(a.crosstab);
    out.extend(a.expected);
    out.extend([a.autocorr_pos, a.autocorr_neg]);
    let mut sims = f64_binary(env, &[]);
    if permutations > 0 {
        let t =
            join_counts::join_counts_permutations(&y, &graph.0, drop_islands, permutations, seed)?;
        out.extend([
            t.min_bb,
            t.mean_bb,
            t.max_bb,
            t.min_bw,
            t.mean_bw,
            t.max_bw,
            t.p_sim_bb,
            t.p_sim_bw,
            t.p_sim_chi2,
            t.p_sim_autocorr_pos,
            t.p_sim_autocorr_neg,
            t.sim_autocorr_chi2,
        ]);
        if keep {
            let all: Vec<f64> = [&t.sim_bb, &t.sim_ww, &t.sim_bw, &t.sim_chi2]
                .into_iter()
                .flatten()
                .copied()
                .collect();
            sims = f64_binary(env, &all);
        }
    }
    Ok((f64_binary(env, &out), sims))
}
