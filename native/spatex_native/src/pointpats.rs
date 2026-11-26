//! NIFs for `Spatex.PointPats`. Points cross as interleaved `x, y` f64 binaries; results with
//! possible NaN/Inf come back as f64 binaries.

use rustler::{Binary, Env, OwnedBinary};
use spatex_core::pointpats::{
    alpha, centrography as cg, distance as dist, hull, pairs, quadrat as quad, random, Window,
};

use crate::f64_binary;

fn support_spec(kind: &str, values: Vec<f64>) -> Result<dist::SupportSpec, String> {
    match (kind, values.as_slice()) {
        ("int", [m]) if *m >= 2.0 && m.fract() == 0.0 => Ok(dist::SupportSpec::Int(*m as usize)),
        ("lin", [start, stop, num])
            if start.is_finite() && stop.is_finite() && *num >= 2.0 && num.fract() == 0.0 =>
        {
            Ok(dist::SupportSpec::Linspace {
                start: *start,
                stop: *stop,
                num: *num as usize,
            })
        }
        ("list", v) => Ok(dist::SupportSpec::Explicit(v.to_vec())),
        _ => Err(format!("invalid support specification {kind:?}")),
    }
}

fn window(kind: &str, data: Binary) -> Result<Window, String> {
    let v = spatex_core::bytes_to_f64s(data.as_slice())?;
    match kind {
        "bbox" if v.len() == 4 => Ok(Window::Bbox([v[0], v[1], v[2], v[3]])),
        "convex" => {
            let pts = pairs(&v)?;
            if pts.len() < 3 {
                return Err("a convex window needs at least 3 vertices".into());
            }
            Ok(Window::Convex(pts))
        }
        _ => Err(format!("invalid window {kind:?}")),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_centrography<'a>(
    env: Env<'a>,
    op: &str,
    points: Binary,
    aux: Binary,
    flags: Vec<bool>,
) -> Result<Binary<'a>, String> {
    let p = pairs(&spatex_core::bytes_to_f64s(points.as_slice())?)?;
    let aux = spatex_core::bytes_to_f64s(aux.as_slice())?;
    let out: Vec<f64> = match op {
        "mean_center" => cg::mean_center(&p)?.to_vec(),
        "weighted_mean_center" => cg::weighted_mean_center(&p, &aux)?.to_vec(),
        "manhattan_median" => cg::manhattan_median(&p)?.to_vec(),
        "euclidean_median" => cg::euclidean_median(&p)?.to_vec(),
        "std_distance" => vec![cg::std_distance(&p)?],
        "dtot" => match aux.as_slice() {
            [x, y] => vec![cg::dtot([*x, *y], &p)],
            _ => return Err("dtot needs a center".into()),
        },
        "ellipse" => {
            let method = match flags.as_slice() {
                [false, ..] => cg::EllipseMethod::Crimestat,
                [true, c, d] => cg::EllipseMethod::Yuill {
                    crimestat_corr: *c,
                    degfreed_corr: *d,
                },
                _ => return Err("invalid ellipse flags".into()),
            };
            let w = (!aux.is_empty()).then_some(aux.as_slice());
            let (a, b, c) = cg::ellipse(&p, w, method)?;
            vec![a, b, c]
        }
        "mbr" => cg::mbr(&p)?.to_vec(),
        "hull" => spatex_core::pointpats::flatten(&hull::convex_hull(&p)?),
        "mrr" => {
            let r = hull::minimum_rotated_rectangle(&p)?;
            let mut v = vec![r.angle, r.area];
            v.extend(spatex_core::pointpats::flatten(&r.corners));
            v
        }
        "mbc" => {
            let (c, r) = hull::minimum_bounding_circle(&p)?;
            vec![c[0], c[1], r]
        }
        other => return Err(format!("unknown centrography operation {other:?}")),
    };
    Ok(f64_binary(env, &out))
}

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn pp_distance<'a>(
    env: Env<'a>,
    kind: &str,
    points: Binary,
    spec_kind: &str,
    spec: Vec<f64>,
    empty: Binary,
    truncate: bool,
    linearized: bool,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let p = pairs(&spatex_core::bytes_to_f64s(points.as_slice())?)?;
    let e = pairs(&spatex_core::bytes_to_f64s(empty.as_slice())?)?;
    let spec = support_spec(spec_kind, spec)?;
    let (s, v) = match kind {
        "g" => dist::g(&p, &spec)?,
        "f" => dist::f(&p, &e, &spec)?,
        "j" => dist::j(&p, &e, &spec, truncate)?,
        "k" => dist::k(&p, &spec)?,
        "l" => dist::l(&p, &spec, linearized)?,
        other => return Err(format!("unknown distance statistic {other:?}")),
    };
    Ok((f64_binary(env, &s), f64_binary(env, &v)))
}

type RipleyOut<'a> = (
    (Binary<'a>, Binary<'a>, Binary<'a>),
    (Binary<'a>, Binary<'a>, Binary<'a>),
);

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn pp_ripley_test<'a>(
    env: Env<'a>,
    kind: &str,
    points: Binary,
    spec_kind: &str,
    spec: Vec<f64>,
    window_kind: &str,
    window_data: Binary,
    options: (usize, u64, bool, bool, bool, usize),
) -> Result<RipleyOut<'a>, String> {
    let (n_simulations, seed, keep_simulations, truncate, linearized, n_empty) = options;
    let p = pairs(&spatex_core::bytes_to_f64s(points.as_slice())?)?;
    let kind = match kind {
        "f" => dist::Kind::F,
        "g" => dist::Kind::G,
        "j" => dist::Kind::J,
        "k" => dist::Kind::K,
        "l" => dist::Kind::L,
        other => return Err(format!("unknown distance statistic {other:?}")),
    };
    let r = dist::ripley_test(
        &p,
        &dist::TestOptions {
            kind,
            spec: support_spec(spec_kind, spec)?,
            window: window(window_kind, window_data)?,
            n_simulations,
            seed,
            keep_simulations,
            truncate,
            linearized,
            n_empty,
        },
    )?;
    let sims = r.simulations.unwrap_or_default();
    Ok((
        (
            f64_binary(env, &r.support),
            f64_binary(env, &r.statistic),
            f64_binary(env, &r.pvalue),
        ),
        (
            f64_binary(env, &sims),
            f64_binary(env, &r.envelope_low),
            f64_binary(env, &r.envelope_high),
        ),
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_poisson<'a>(
    env: Env<'a>,
    window_kind: &str,
    window_data: Binary,
    intensity: Option<f64>,
    sizes: (Option<usize>, Option<usize>),
    seed: u64,
) -> Result<(Binary<'a>, usize, usize), String> {
    let w = window(window_kind, window_data)?;
    w.validate()?;
    let (n_obs, n_sims, _) =
        random::parse_size_and_intensity(w.area(), intensity, sizes.0, sizes.1)?;
    if n_obs == 0 || n_sims == 0 {
        return Err("size must be positive".into());
    }
    let sims = random::poisson(&w, n_obs, n_sims, seed)?;
    let flat: Vec<f64> = sims
        .iter()
        .flat_map(|s| spatex_core::pointpats::flatten(s))
        .collect();
    Ok((f64_binary(env, &flat), n_obs, n_sims))
}

#[rustler::nif(schedule = "DirtyCpu")]
#[allow(clippy::too_many_arguments)]
pub fn pp_cluster_poisson<'a>(
    env: Env<'a>,
    window_kind: &str,
    window_data: Binary,
    intensity: Option<f64>,
    sizes: (Option<usize>, Option<usize>),
    cluster: (usize, Vec<f64>),
    seed: u64,
) -> Result<(Binary<'a>, usize, usize), String> {
    let w = window(window_kind, window_data)?;
    w.validate()?;
    let (n_obs, n_sims, _) =
        random::parse_size_and_intensity(w.area(), intensity, sizes.0, sizes.1)?;
    if n_obs == 0 || n_sims == 0 {
        return Err("size must be positive".into());
    }
    let sims = random::cluster_poisson(&w, n_obs, n_sims, cluster.0, &cluster.1, seed)?;
    let flat: Vec<f64> = sims
        .iter()
        .flat_map(|s| spatex_core::pointpats::flatten(s))
        .collect();
    Ok((f64_binary(env, &flat), n_obs, n_sims))
}

fn flatten_sims<'a>(env: Env<'a>, sims: &[Vec<spatex_core::pointpats::Pt>]) -> Binary<'a> {
    let flat: Vec<f64> = sims
        .iter()
        .flat_map(|s| spatex_core::pointpats::flatten(s))
        .collect();
    f64_binary(env, &flat)
}

fn cov_matrix(cov: Vec<f64>) -> Result<Option<[f64; 4]>, String> {
    match cov.len() {
        0 => Ok(None),
        4 => Ok(Some([cov[0], cov[1], cov[2], cov[3]])),
        _ => Err("cov must be a 2 x 2 matrix".into()),
    }
}

fn centre(center: Vec<f64>) -> Result<Option<[f64; 2]>, String> {
    match center.len() {
        0 => Ok(None),
        2 => Ok(Some([center[0], center[1]])),
        _ => Err("center must be a pair {x, y}".into()),
    }
}

/// Resolves `size` (no intensity): `(n_obs, n_sims)`, positive.
fn counts(w: &Window, sizes: (Option<usize>, Option<usize>)) -> Result<(usize, usize), String> {
    let (n_obs, n_sims, _) = random::parse_size_and_intensity(w.area(), None, sizes.0, sizes.1)?;
    if n_obs == 0 || n_sims == 0 {
        return Err("size must be positive".into());
    }
    Ok((n_obs, n_sims))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_normal<'a>(
    env: Env<'a>,
    window_kind: &str,
    window_data: Binary,
    sizes: (Option<usize>, Option<usize>),
    center: Vec<f64>,
    cov: Vec<f64>,
    seed: u64,
) -> Result<(Binary<'a>, usize, usize), String> {
    let w = window(window_kind, window_data)?;
    w.validate()?;
    let (n_obs, n_sims) = counts(&w, sizes)?;
    let sims = random::normal(&w, n_obs, n_sims, centre(center)?, cov_matrix(cov)?, seed)?;
    Ok((flatten_sims(env, &sims), n_obs, n_sims))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_cluster_normal<'a>(
    env: Env<'a>,
    window_kind: &str,
    window_data: Binary,
    sizes: (Option<usize>, Option<usize>),
    n_seeds: usize,
    cov: Vec<f64>,
    seed: u64,
) -> Result<(Binary<'a>, usize, usize), String> {
    let w = window(window_kind, window_data)?;
    w.validate()?;
    let (n_obs, n_sims) = counts(&w, sizes)?;
    let sims = random::cluster_normal(&w, n_obs, n_sims, n_seeds, cov_matrix(cov)?, seed)?;
    Ok((flatten_sims(env, &sims), n_obs, n_sims))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_strauss<'a>(
    env: Env<'a>,
    window_kind: &str,
    window_data: Binary,
    intensity: Option<f64>,
    sizes: (Option<usize>, Option<usize>),
    params: (f64, f64, usize, usize),
    seed: u64,
) -> Result<(Binary<'a>, usize, usize), String> {
    let w = window(window_kind, window_data)?;
    w.validate()?;
    let (n_obs, n_sims, _) =
        random::parse_size_and_intensity(w.area(), intensity, sizes.0, sizes.1)?;
    if n_obs == 0 || n_sims == 0 {
        return Err("size must be positive".into());
    }
    let (gamma, r, n_iter, max_iter) = params;
    let sims = random::strauss(&w, n_obs, n_sims, gamma, r, n_iter, max_iter, seed)?;
    Ok((flatten_sims(env, &sims), n_obs, n_sims))
}

type QuadratOut<'a> = (
    (Vec<u64>, Vec<u64>),
    (usize, usize, usize, usize, usize),
    (Binary<'a>, Binary<'a>, Binary<'a>),
);

#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_quadrat<'a>(
    env: Env<'a>,
    points: Binary,
    shape: &str,
    geometry: (usize, usize, f64, f64, f64),
    realizations: usize,
    seed: u64,
    max_cells: usize,
) -> Result<QuadratOut<'a>, String> {
    let p = pairs(&spatex_core::bytes_to_f64s(points.as_slice())?)?;
    let (nx, ny, width, height, lh) = geometry;
    let shape = match shape {
        "rectangle" => quad::Shape::Rectangle {
            nx,
            ny,
            width,
            height,
        },
        "hexagon" => quad::Shape::Hexagon { lh },
        other => return Err(format!("unknown shape {other:?}")),
    };
    let t = quad::quadrat_test(&p, shape, realizations, seed, max_cells)?;
    let g = &t.grid;
    let scalars = vec![
        t.chi2,
        t.chi2_pvalue,
        g.rectangle_width,
        g.rectangle_height,
        g.mbb[0],
        g.mbb[1],
        g.mbb[2],
        g.mbb[3],
        t.chi2_r_pvalue.unwrap_or(f64::NAN),
    ];
    Ok((
        (
            g.cell_ids.iter().map(|v| *v as u64).collect(),
            g.counts.iter().map(|v| *v as u64).collect(),
        ),
        (
            g.count_column,
            g.count_row,
            g.count_row_even,
            g.count_row_odd,
            t.df,
        ),
        (
            f64_binary(env, &scalars),
            f64_binary(env, &t.chi2_contrib),
            f64_binary(env, &t.chi2_realizations),
        ),
    ))
}

fn bytes_binary<'a>(env: Env<'a>, bytes: &[u8]) -> Binary<'a> {
    let mut owned = OwnedBinary::new(bytes.len());
    owned.as_mut_slice().copy_from_slice(bytes);
    Binary::from_owned(owned, env)
}

/// Alpha shape polygons as little-endian WKB, one binary per polygon.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_alpha_shape<'a>(
    env: Env<'a>,
    points: Binary,
    alpha_value: f64,
) -> Result<Vec<Binary<'a>>, String> {
    let p = pairs(&spatex_core::bytes_to_f64s(points.as_slice())?)?;
    let polys = alpha::alpha_shape(&p, alpha_value)?;
    Ok(polys
        .iter()
        .map(|q| bytes_binary(env, &alpha::polygon_wkb(q)))
        .collect())
}

/// Automatic alpha shape: `(wkb, radius, circle centres as interleaved x, y)`; the centres are
/// empty unless `circles`.
#[rustler::nif(schedule = "DirtyCpu")]
pub fn pp_alpha_shape_auto<'a>(
    env: Env<'a>,
    points: Binary,
    step: usize,
    circles: bool,
) -> Result<(Binary<'a>, f64, Binary<'a>), String> {
    let p = pairs(&spatex_core::bytes_to_f64s(points.as_slice())?)?;
    let a = alpha::alpha_shape_auto(&p, step)?;
    let centers = if circles {
        spatex_core::pointpats::flatten(&alpha::bounding_circle_centers(&a.polygon, a.radius))
    } else {
        Vec::new()
    };
    Ok((
        bytes_binary(env, &alpha::polygon_wkb(&a.polygon)),
        a.radius,
        f64_binary(env, &centers),
    ))
}
