//! NIFs for inequality measures (`Spatex.Inequality`). Results travel as native-endian `f64`
//! binaries (NaN-safe).

use rustler::{Binary, Env};
use spatex_core::inequality;

use crate::{f64_binary, Graph};

fn f64s(b: &Binary) -> Result<Vec<f64>, String> {
    spatex_core::bytes_to_f64s(b.as_slice())
}

/// One-element binary holding the Gini coefficient.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_gini<'a>(env: Env<'a>, x: Binary) -> Result<Binary<'a>, String> {
    let x = f64s(&x)?;
    Ok(f64_binary(env, &[inequality::gini(&x)]))
}

/// Theil's T per column of an `n x t` matrix.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_theil<'a>(
    env: Env<'a>,
    y: Binary,
    n: usize,
    t: usize,
) -> Result<Binary<'a>, String> {
    let y = f64s(&y)?;
    if y.len() != n * t {
        return Err(format!("expected {n} x {t} values, got {}", y.len()));
    }
    Ok(f64_binary(env, &inequality::theil(&y, n, t)))
}

/// Theil decomposition: blocks `t, bg, wg` (each `t` long) followed, with permutations, by the
/// `bg` pseudo p-values.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_theil_d<'a>(
    env: Env<'a>,
    y: Binary,
    dims: (usize, usize),
    partition: Binary,
    draws: (usize, u64),
) -> Result<Binary<'a>, String> {
    let y = f64s(&y)?;
    let part = spatex_core::bytes_to_u32s(partition.as_slice())?;
    let (n, t) = dims;
    let (permutations, seed) = draws;
    let d = inequality::theil_d(&y, n, t, &part)?;
    let mut blocks = d.t;
    blocks.extend(&d.bg);
    blocks.extend(&d.wg);
    if permutations > 0 {
        blocks.extend(inequality::theil_d_sim(
            &y,
            n,
            t,
            &part,
            permutations,
            seed,
        )?);
    }
    Ok(f64_binary(env, &blocks))
}

/// Spatial Gini: `{scalars, wcgp, polarization_sim}` with scalars
/// `g, wg, wcg, wcg_share, dtotal, den, polarization, p_sim, e_wcg, s_wcg, z_wcg, p_z_sim,
/// polarization_p_sim`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_spatial_gini<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    permutations: usize,
    seed: u64,
) -> Result<(Binary<'a>, Binary<'a>, Binary<'a>), String> {
    let x = f64s(&x)?;
    let s = inequality::spatial_gini(&x, &graph.0, permutations, seed)?;
    let scalars = [
        s.g,
        s.wg,
        s.wcg,
        s.wcg_share,
        s.dtotal,
        s.den,
        s.polarization,
        s.p_sim,
        s.e_wcg,
        s.s_wcg,
        s.z_wcg,
        s.p_z_sim,
        s.polarization_p_sim,
    ];
    Ok((
        f64_binary(env, &scalars),
        f64_binary(env, &s.wcgp),
        f64_binary(env, &s.polarization_sim),
    ))
}

/// `[a, ede]` of the Atkinson index.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_atkinson<'a>(
    env: Env<'a>,
    y: Binary,
    epsilon: f64,
) -> Result<Binary<'a>, String> {
    let r = inequality::lorenz::atkinson(&f64s(&y)?, epsilon)?;
    Ok(f64_binary(env, &[r.a, r.ede]))
}

/// `[distance, intersection_point, coefficient]` of the Schutz measures.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_schutz<'a>(env: Env<'a>, y: Binary) -> Result<Binary<'a>, String> {
    let r = inequality::lorenz::schutz(&f64s(&y)?)?;
    Ok(f64_binary(
        env,
        &[r.distance, r.intersection_point, r.coefficient],
    ))
}

/// `{population, income_share}` of the Lorenz curve, `n + 1` points each.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_lorenz_curve<'a>(
    env: Env<'a>,
    y: Binary,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let (pop, share) = inequality::lorenz::lorenz_curve(&f64s(&y)?);
    Ok((f64_binary(env, &pop), f64_binary(env, &share)))
}

/// One-element binary holding the Wolfson bipolarization index.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_wolfson<'a>(env: Env<'a>, y: Binary) -> Result<Binary<'a>, String> {
    Ok(f64_binary(env, &[inequality::lorenz::wolfson(&f64s(&y)?)?]))
}

/// Spatial polarization: `{scalars, labels, sim}` with scalars
/// `statistic, p_value, n_a_components, n_g_components, n_i_components` and `labels` the
/// `i_labels`, `a_labels` and `g_labels` vectors (`-1` for none) one after the other. `bins`
/// empty means `k` quantile classes; `sim` is empty unless `keep`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn inequality_polarization<'a>(
    env: Env<'a>,
    graph: Graph,
    x: Binary,
    grouping: (usize, Binary),
    draws: (usize, u64, bool),
) -> Result<(Binary<'a>, Binary<'a>, Binary<'a>), String> {
    let x = f64s(&x)?;
    let (k, bins) = grouping;
    let bins = f64s(&bins)?;
    let grouping = if bins.is_empty() {
        inequality::polarization::Grouping::Quantiles(k)
    } else {
        inequality::polarization::Grouping::Bins(bins)
    };
    let (permutations, seed, keep) = draws;
    let r = inequality::polarization::polarization(&x, &graph.0, &grouping, permutations, seed)?;
    let scalars = [
        r.statistic,
        r.p_value,
        r.n_a_components as f64,
        r.n_g_components as f64,
        r.n_i_components as f64,
    ];
    let labels: Vec<f64> = r
        .i_labels
        .iter()
        .chain(&r.a_labels)
        .chain(&r.g_labels)
        .map(|&l| l as f64)
        .collect();
    let sim: &[f64] = if keep { &r.sim } else { &[] };
    Ok((
        f64_binary(env, &scalars),
        f64_binary(env, &labels),
        f64_binary(env, sim),
    ))
}
