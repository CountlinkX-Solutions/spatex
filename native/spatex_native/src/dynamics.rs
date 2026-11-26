//! NIFs for spatial dynamics (`Spatex.Dynamics`). Numeric blocks travel as native-endian `f64`
//! binaries (NaN and infinity safe); class indices as `u32` binaries.

use rustler::{Binary, Env};
use spatex_core::dynamics::markov::{chi2_test, count_transitions, homogeneity, kullback};
use spatex_core::dynamics::spatial::{spatial_markov, SpatialMarkovSettings};
use spatex_core::dynamics::special::chi2_sf;
use spatex_core::dynamics::{chain, local_tau, rank};

use crate::{f64_binary, Graph};

fn f64s(b: &Binary) -> Result<Vec<f64>, String> {
    spatex_core::bytes_to_f64s(b.as_slice())
}

/// Classic transition counts `k x k` of an `n x t` u32 class panel.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_transitions<'a>(
    env: Env<'a>,
    ids: Binary,
    n: usize,
    t: usize,
    k: usize,
) -> Result<Binary<'a>, String> {
    let ids = spatex_core::bytes_to_u32s(ids.as_slice())?;
    count_transitions(&ids, n, t, k).map(|c| f64_binary(env, &c))
}

type Chain<'a> = (
    Binary<'a>,
    Binary<'a>,
    Binary<'a>,
    (
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
        Vec<usize>,
    ),
);

/// Ergodic summary of a (possibly non-stochastic) probability matrix; empty rows are filled
/// with absorbing states first. Returns `{steady (rows x k), mfpt (k x k), sojourn (k),
/// {communication, recurrent, transient, absorbing}}`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_chain<'a>(env: Env<'a>, p: Binary, k: usize) -> Result<Chain<'a>, String> {
    let mut p = f64s(&p)?;
    if p.len() != k * k {
        return Err(format!("expected {k} x {k} values, got {}", p.len()));
    }
    chain::fill_empty_diagonals(&mut p, k);
    let steady: Vec<f64> = chain::steady_state(&p, k)?.into_iter().flatten().collect();
    let mfpt = chain::mfpt(&p, k)?;
    let sojourn = chain::sojourn_time(&p, k);
    let s = chain::structure(&p, k);
    Ok((
        f64_binary(env, &steady),
        f64_binary(env, &mfpt),
        f64_binary(env, &sojourn),
        (s.communication, s.recurrent, s.transient, s.absorbing),
    ))
}

/// `[statistic, p-value, dof]` of `giddy.markov.chi2(t1, t2)`.
#[rustler::nif]
pub fn dynamics_chi2_test<'a>(
    env: Env<'a>,
    t1: Binary,
    t2: Binary,
    k: usize,
) -> Result<Binary<'a>, String> {
    let (t1, t2) = (f64s(&t1)?, f64s(&t2)?);
    if t1.len() != k * k || t2.len() != k * k {
        return Err(format!("expected two {k} x {k} matrices"));
    }
    let (stat, p, dof) = chi2_test(&t1, &t2, k);
    Ok(f64_binary(env, &[stat, p, dof]))
}

/// `1 - chi2.cdf(x, dof)` as a one-element f64 binary (NaN-safe).
#[rustler::nif]
pub fn dynamics_chi2_sf<'a>(env: Env<'a>, x: f64, dof: f64) -> Binary<'a> {
    f64_binary(env, &[chi2_sf(x, dof)])
}

/// `[q, q_p_value, lr, lr_p_value, dof]` of the homogeneity test of `m` stacked `k x k` counts.
#[rustler::nif]
pub fn dynamics_homogeneity<'a>(
    env: Env<'a>,
    counts: Binary,
    m: usize,
    k: usize,
) -> Result<Binary<'a>, String> {
    let counts = f64s(&counts)?;
    if counts.len() != m * k * k || m == 0 {
        return Err(format!("expected {m} stacked {k} x {k} matrices"));
    }
    let mats: Vec<Vec<f64>> = counts.chunks(k * k).map(<[f64]>::to_vec).collect();
    let h = homogeneity(&mats, k);
    Ok(f64_binary(env, &[h.q, h.q_p, h.lr, h.lr_p, h.dof as f64]))
}

/// `[statistic, p_value, dof]` of the Kullback homogeneity test of `s` stacked `r x r` counts.
#[rustler::nif]
pub fn dynamics_kullback<'a>(
    env: Env<'a>,
    counts: Binary,
    dims: (usize, usize),
    giddy_totals: bool,
) -> Result<Binary<'a>, String> {
    let (s, r) = dims;
    let k = kullback(&f64s(&counts)?, s, r, giddy_totals)?;
    Ok(f64_binary(env, &[k.statistic, k.p_value, k.dof as f64]))
}

/// Spatial Markov. Returns `{meta, blocks}`. `meta` is `[k, m, has_cutoffs, cutoffs_len,
/// has_lag_cutoffs, lag_cutoffs_len, sims_len, has_rpvalue]`; `blocks` concatenates
/// `class_ids (n t), lclass_ids (n t), cutoffs, lag_cutoffs, transitions (k k), t_cond (m k k),
/// [x2], x2_sims, [x2_rpvalue]`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_spatial_markov<'a>(
    env: Env<'a>,
    graph: Graph,
    y: Binary,
    dims: (usize, usize),
    classes: (usize, usize, bool, bool),
    cutoffs: (Option<Binary>, Option<Binary>),
    draws: (usize, u64),
) -> Result<(Vec<usize>, Binary<'a>), String> {
    let y = f64s(&y)?;
    let (n, t) = dims;
    let (k, m, fixed, discrete) = classes;
    let (permutations, seed) = draws;
    let settings = SpatialMarkovSettings {
        k,
        m,
        fixed,
        discrete,
        cutoffs: cutoffs.0.as_ref().map(f64s).transpose()?,
        lag_cutoffs: cutoffs.1.as_ref().map(f64s).transpose()?,
        permutations,
        seed,
    };
    let r = spatial_markov(&y, n, t, &graph.0, &settings)?;
    let mut blocks: Vec<f64> = Vec::new();
    blocks.extend(r.class_ids.iter().map(|&v| f64::from(v)));
    blocks.extend(r.lclass_ids.iter().map(|&v| f64::from(v)));
    let cut = r.cutoffs.as_deref().unwrap_or(&[]);
    let lag = r.lag_cutoffs.as_deref().unwrap_or(&[]);
    blocks.extend(cut);
    blocks.extend(lag);
    blocks.extend(&r.transitions);
    blocks.extend(&r.t_cond);
    blocks.push(r.x2);
    blocks.extend(&r.x2_sims);
    if let Some(p) = r.x2_rpvalue {
        blocks.push(p);
    }
    let meta = vec![
        r.k,
        r.m,
        usize::from(r.cutoffs.is_some()),
        cut.len(),
        usize::from(r.lag_cutoffs.is_some()),
        lag.len(),
        r.x2_sims.len(),
        usize::from(r.x2_rpvalue.is_some()),
    ];
    Ok((meta, f64_binary(env, &blocks)))
}

/// `[tau, p, concordant, discordant, extra_x, extra_y]`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_tau<'a>(env: Env<'a>, x: Binary, y: Binary) -> Result<Binary<'a>, String> {
    let (x, y) = (f64s(&x)?, f64s(&y)?);
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    let t = rank::tau(&x, &y);
    Ok(f64_binary(
        env,
        &[t.tau, t.p, t.concordant, t.discordant, t.extra_x, t.extra_y],
    ))
}

/// `{[tau, tau_p, c, d, ex, ey, tau_spatial, pairs, cs, ds, psim], taus}`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_spatial_tau<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    y: Binary,
    permutations: usize,
    seed: u64,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let (x, y) = (f64s(&x)?, f64s(&y)?);
    let s = rank::spatial_tau(&x, &y, &graph.0, permutations, seed)?;
    let scalars = [
        s.tau.tau,
        s.tau.p,
        s.tau.concordant,
        s.tau.discordant,
        s.tau.extra_x,
        s.tau.extra_y,
        s.tau_spatial,
        s.pairs_spatial as f64,
        s.concordant_spatial,
        s.discordant_spatial,
        s.psim,
    ];
    Ok((f64_binary(env, &scalars), f64_binary(env, &s.taus)))
}

/// `{blocks, ranks}` with blocks `theta (t-1), total (t-1), [max_total], pvalue_left,
/// pvalue_right, z` (the last three empty without permutations).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_theta<'a>(
    env: Env<'a>,
    y: Binary,
    n: usize,
    t: usize,
    regime: Binary,
    permutations: usize,
    seed: u64,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let y = f64s(&y)?;
    let regime = spatex_core::bytes_to_u32s(regime.as_slice())?;
    let r = rank::theta(&y, n, t, &regime, permutations, seed)?;
    let mut blocks = r.theta.clone();
    blocks.extend(&r.total);
    blocks.push(r.max_total);
    blocks.extend(&r.pvalue_left);
    blocks.extend(&r.pvalue_right);
    blocks.extend(&r.z);
    Ok((f64_binary(env, &blocks), f64_binary(env, &r.ranks)))
}

/// `Tau_Local`: `{tau, tau_local}`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_tau_local<'a>(
    env: Env<'a>,
    x: Binary,
    y: Binary,
) -> Result<(f64, Binary<'a>), String> {
    let (tau, local) = local_tau::tau_local(&f64s(&x)?, &f64s(&y)?)?;
    Ok((tau, f64_binary(env, &local)))
}

fn local_tau_out<'a>(
    env: Env<'a>,
    mut blocks: Vec<f64>,
    r: local_tau::LocalTau,
) -> (Binary<'a>, Binary<'a>) {
    blocks.extend(&r.value);
    if let Some(p) = &r.p {
        blocks.extend(p);
    }
    (
        f64_binary(env, &blocks),
        f64_binary(env, r.sims.as_deref().unwrap_or(&[])),
    )
}

/// `Tau_Local_Neighbor`: `{blocks, sims}`, blocks `weights`, `tau_ln` and (with permutations) `p`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_tau_local_neighbor<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    y: Binary,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let (r, weights) =
        local_tau::tau_local_neighbor(&f64s(&x)?, &f64s(&y)?, &graph.0, permutations, seed, keep)?;
    Ok(local_tau_out(env, weights, r))
}

/// `Tau_Local_Neighborhood`: `{blocks, sims}`, blocks `tau_lnhood` and (with permutations) `p`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_tau_local_neighborhood<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    y: Binary,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let r = local_tau::tau_local_neighborhood(
        &f64s(&x)?,
        &f64s(&y)?,
        &graph.0,
        permutations,
        seed,
        keep,
    )?;
    Ok(local_tau_out(env, Vec::new(), r))
}

/// `Tau_Regional`: `{regimes, tau_reg, p, sims}` (`k x k` row-major; `p`/`sims` empty unless
/// computed).
#[rustler::nif(schedule = "DirtyCpu")]
pub fn dynamics_tau_regional<'a>(
    env: Env<'a>,
    x: Binary,
    y: Binary,
    regime: Vec<i64>,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(Vec<i64>, Binary<'a>, Binary<'a>, Binary<'a>), String> {
    let r = local_tau::tau_regional(&f64s(&x)?, &f64s(&y)?, &regime, permutations, seed, keep)?;
    Ok((
        r.regimes,
        f64_binary(env, &r.tau_reg),
        f64_binary(env, r.p.as_deref().unwrap_or(&[])),
        f64_binary(env, r.sims.as_deref().unwrap_or(&[])),
    ))
}
