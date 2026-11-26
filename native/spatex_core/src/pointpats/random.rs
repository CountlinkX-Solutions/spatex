//! Random point processes (`pointpats/random.py`): conditional Poisson (CSR), the Poisson
//! cluster process, the truncated bivariate normal and normal cluster processes, and the Strauss
//! process. Replication `r` draws from the stream `(seed, r)`.

use super::{unit, Pt, Window};
use crate::limits;
use crate::perm::{replicate_rng, uniform_below};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// `parse_size_and_intensity`: returns `(n_obs, n_sims, intensity)`.
///
/// `n_obs`/`n_sims` are `None` when the caller did not fix them (`size` nil, or an integer
/// `size` combined with `intensity`, which fixes only `n_sims`). More than
/// [`limits::MAX_SIMULATED_POINTS`] points in total is an error raised before sampling.
pub fn parse_size_and_intensity(
    area: f64,
    intensity: Option<f64>,
    n_obs: Option<usize>,
    n_sims: Option<usize>,
) -> Result<(usize, usize, f64), String> {
    let (n, s, i) = match (n_obs, n_sims, intensity) {
        (Some(_), Some(_), Some(_)) => {
            return Err(
                "intensity cannot be given together with a size tuple {n_obs, n_sims}".into(),
            )
        }
        (Some(n), Some(s), None) => (n, s, n as f64 / area),
        (Some(n), None, None) => (n, 1, n as f64 / area),
        (Some(_), None, Some(_)) => {
            return Err("n_obs and intensity cannot both be given for a single simulation".into())
        }
        (None, s, Some(i)) => {
            let expected = area * i;
            if !(expected.is_finite() && expected <= limits::MAX_SIMULATED_POINTS as f64) {
                return Err(format!(
                    "simulated points: area * intensity = {expected:e} exceeds the limit of {}; {HINT}",
                    limits::MAX_SIMULATED_POINTS
                ));
            }
            (expected as usize, s.unwrap_or(1), i)
        }
        (None, s, None) => (100, s.unwrap_or(1), 100.0 / area),
    };
    let total = limits::checked_product("simulated points", n, s)?;
    limits::ensure_within(
        "simulated points",
        total,
        limits::MAX_SIMULATED_POINTS,
        HINT,
    )?;
    Ok((n, s, i))
}

const HINT: &str = "simulate fewer points or fewer patterns per call";

/// `n` uniform points of the window drawn from `rng`.
pub fn poisson_one(window: &Window, n: usize, rng: &mut ChaCha8Rng) -> Vec<Pt> {
    (0..n).map(|_| window.sample(rng)).collect()
}

/// `n_sims` independent patterns of `n_obs` points uniform in `window`.
pub fn poisson(
    window: &Window,
    n_obs: usize,
    n_sims: usize,
    seed: u64,
) -> Result<Vec<Vec<Pt>>, String> {
    window.validate()?;
    Ok((0..n_sims)
        .into_par_iter()
        .map(|r| poisson_one(window, n_obs, &mut replicate_rng(seed, r as u64)))
        .collect())
}

fn min_pair_distance(p: &[Pt]) -> f64 {
    let mut m = f64::INFINITY;
    for i in 0..p.len() {
        for j in (i + 1)..p.len() {
            m = m.min((p[i][0] - p[j][0]).hypot(p[i][1] - p[j][1]));
        }
    }
    m
}

/// `n_obs` points in `n_seeds` clusters: the seed itself followed by points uniform in the disc
/// of the cluster radius around it, inside the window. `radii` holds one radius, one per seed,
/// or is empty for the default of half the minimum seed distance.
pub fn cluster_poisson(
    window: &Window,
    n_obs: usize,
    n_sims: usize,
    n_seeds: usize,
    radii: &[f64],
    seed: u64,
) -> Result<Vec<Vec<Pt>>, String> {
    window.validate()?;
    if n_seeds == 0 || n_seeds > n_obs {
        return Err("too many clusters for the requested number of points".into());
    }
    if radii.is_empty() && n_seeds < 2 {
        return Err("cluster_radius is required when n_seeds is 1".into());
    }
    if !(radii.is_empty() || radii.len() == 1 || radii.len() == n_seeds) {
        return Err("cluster_radius must be a scalar or have one value per seed".into());
    }
    if radii.iter().any(|r| !(r.is_finite() && *r > 0.0)) {
        return Err("cluster_radius must be positive".into());
    }
    (0..n_sims)
        .into_par_iter()
        .map(|r| {
            let mut rng = replicate_rng(seed, r as u64);
            let seeds = poisson_one(window, n_seeds, &mut rng);
            let radii: Vec<f64> = match radii.len() {
                0 => vec![0.5 * min_pair_distance(&seeds); n_seeds],
                1 => vec![radii[0]; n_seeds],
                _ => radii.to_vec(),
            };
            if radii.iter().any(|r| r.is_nan() || *r <= 0.0) {
                return Err("coincident seeds give a zero cluster radius".to_string());
            }
            let (base, extra) = (n_obs / n_seeds, n_obs % n_seeds);
            let mut out = Vec::with_capacity(n_obs);
            for (c, s) in seeds.iter().enumerate() {
                let size = base + usize::from(c < extra);
                out.push(*s);
                let rad = radii[c];
                let mut left = size - 1;
                while left > 0 {
                    let x = s[0] + rad * (2.0 * unit(&mut rng) - 1.0);
                    let y = s[1] + rad * (2.0 * unit(&mut rng) - 1.0);
                    let (dx, dy) = (x - s[0], y - s[1]);
                    if dx * dx + dy * dy <= rad * rad && window.contains(x, y) {
                        out.push([x, y]);
                        left -= 1;
                    }
                }
            }
            Ok(out)
        })
        .collect()
}

/// Rejections in a row after which a normal sampler gives up: the centre is then effectively
/// outside the window (pointpats loops forever).
const MAX_REJECTIONS: usize = 1_000_000;

/// Standard normal pair by Box-Muller.
fn std_normal_pair(rng: &mut ChaCha8Rng) -> (f64, f64) {
    let u1 = 1.0 - unit(rng); // (0, 1]
    let u2 = unit(rng);
    let rad = (-2.0 * u1.ln()).sqrt();
    let ang = std::f64::consts::TAU * u2;
    (rad * ang.cos(), rad * ang.sin())
}

/// Lower Cholesky factor `[a, 0, b, c]` (row major) of a symmetric positive-definite `cov`
/// given row major as `[xx, xy, yx, yy]`.
fn cholesky2(cov: [f64; 4]) -> Result<[f64; 3], String> {
    if cov.iter().any(|v| !v.is_finite()) {
        return Err("cov must be finite".into());
    }
    let scale = cov[0].abs().max(cov[3].abs()).max(1e-300);
    if (cov[1] - cov[2]).abs() > 1e-9 * scale {
        return Err("cov must be a symmetric 2 x 2 matrix".into());
    }
    if cov[0] <= 0.0 {
        return Err("cov must be positive definite".into());
    }
    let a = cov[0].sqrt();
    let b = cov[1] / a;
    let d = cov[3] - b * b;
    if d <= 0.0 {
        return Err("cov must be positive definite".into());
    }
    Ok([a, b, d.sqrt()])
}

/// Window centre as pointpats' `centroid`: the middle of a box, the vertex mean of a polygon.
fn window_centre(window: &Window) -> Pt {
    match window {
        Window::Bbox(b) => [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0],
        Window::Convex(v) => {
            let n = v.len() as f64;
            [
                v.iter().map(|p| p[0]).sum::<f64>() / n,
                v.iter().map(|p| p[1]).sum::<f64>() / n,
            ]
        }
    }
}

/// `n` normal draws around `center` with Cholesky factor `l`, rejecting those outside the window.
fn normal_one(
    window: &Window,
    n: usize,
    center: Pt,
    l: [f64; 3],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<Pt>, String> {
    let mut out = Vec::with_capacity(n);
    let mut misses = 0usize;
    while out.len() < n {
        let (z0, z1) = std_normal_pair(rng);
        let (x, y) = (center[0] + l[0] * z0, center[1] + l[1] * z0 + l[2] * z1);
        if window.contains(x, y) {
            out.push([x, y]);
            misses = 0;
        } else {
            misses += 1;
            if misses > MAX_REJECTIONS {
                return Err(
                    "the normal distribution almost never falls inside the window; \
                            move center into the window or increase cov"
                        .into(),
                );
            }
        }
    }
    Ok(out)
}

/// Row-major covariance `[xx, xy, yx, yy]`; the default is `(max(width, height) / 2)^2` times
/// the identity.
fn default_normal_cov(window: &Window) -> [f64; 4] {
    let b = window.bbox();
    let v = ((b[2] - b[0]).max(b[3] - b[1]) / 2.0).powi(2);
    [v, 0.0, 0.0, v]
}

/// `n_sims` patterns of `n_obs` bivariate normal points around `center` (default: the window
/// centre) with covariance `cov`, redrawn until inside the window.
pub fn normal(
    window: &Window,
    n_obs: usize,
    n_sims: usize,
    center: Option<Pt>,
    cov: Option<[f64; 4]>,
    seed: u64,
) -> Result<Vec<Vec<Pt>>, String> {
    window.validate()?;
    let l = cholesky2(cov.unwrap_or_else(|| default_normal_cov(window)))?;
    let center = center.unwrap_or_else(|| window_centre(window));
    if !center.iter().all(|v| v.is_finite()) {
        return Err("center must be finite".into());
    }
    (0..n_sims)
        .into_par_iter()
        .map(|r| normal_one(window, n_obs, center, l, &mut replicate_rng(seed, r as u64)))
        .collect()
}

/// `n_obs` points in `n_seeds` normal clusters: each cluster is its centre (uniform in the
/// window) followed by normal points around it inside the window. The default covariance is
/// the squared mean distance between the centres of that replication.
pub fn cluster_normal(
    window: &Window,
    n_obs: usize,
    n_sims: usize,
    n_seeds: usize,
    cov: Option<[f64; 4]>,
    seed: u64,
) -> Result<Vec<Vec<Pt>>, String> {
    window.validate()?;
    if n_seeds == 0 || n_seeds > n_obs {
        return Err("too many clusters for the requested number of points".into());
    }
    if cov.is_none() && n_seeds < 2 {
        return Err("cov is required when n_seeds is 1".into());
    }
    let fixed = cov.map(cholesky2).transpose()?;
    (0..n_sims)
        .into_par_iter()
        .map(|r| {
            let mut rng = replicate_rng(seed, r as u64);
            let centers = poisson_one(window, n_seeds, &mut rng);
            let l = match fixed {
                Some(l) => l,
                None => {
                    let v = mean_pair_distance(&centers).powi(2);
                    if !(v.is_finite() && v > 0.0) {
                        return Err("coincident seeds give a zero cluster covariance".to_string());
                    }
                    cholesky2([v, 0.0, 0.0, v])?
                }
            };
            let (base, extra) = (n_obs / n_seeds, n_obs % n_seeds);
            let mut out = Vec::with_capacity(n_obs);
            for (c, centre) in centers.iter().enumerate() {
                let size = base + usize::from(c < extra);
                out.push(*centre);
                out.extend(normal_one(window, size - 1, *centre, l, &mut rng)?);
            }
            Ok(out)
        })
        .collect()
}

fn mean_pair_distance(p: &[Pt]) -> f64 {
    let (mut total, mut count) = (0.0, 0usize);
    for i in 0..p.len() {
        for j in (i + 1)..p.len() {
            total += (p[i][0] - p[j][0]).hypot(p[i][1] - p[j][1]);
            count += 1;
        }
    }
    total / count as f64
}

const STRAUSS_FAILURE: &str = "the number of points requested is too large for the specified \
     gamma and r; reduce the size or intensity, or increase gamma, r or max_iter";

/// Points of `p` within `r` of `q`, skipping index `skip` (`usize::MAX` for none): the change
/// in the number of close pairs when `q` is added to or removed from the pattern.
fn neighbours_within(p: &[Pt], q: Pt, r2: f64, skip: usize) -> i32 {
    p.iter()
        .enumerate()
        .filter(|(i, a)| *i != skip && (a[0] - q[0]).powi(2) + (a[1] - q[1]).powi(2) <= r2)
        .count() as i32
}

/// One Strauss pattern by pointpats' birth-death Metropolis-Hastings sampler: start from
/// `1.5 n` uniform points, run `n_iter` steps (births with probability `min(1, n gamma^dk /
/// (len + 1))`, deaths while the pattern is larger than `n` with `min(1, len / (n gamma^dk))`,
/// `dk` the number of pairs within `r` the point forms) and keep the first `n` points.
/// Pair counts are updated from the moved point alone instead of recounting every pair.
fn strauss_one(
    window: &Window,
    n: usize,
    params: (f64, f64, usize, usize),
    rng: &mut ChaCha8Rng,
) -> Result<Vec<Pt>, String> {
    let (gamma, r, n_iter, max_iter) = params;
    let r2 = r * r;
    let target = n as f64;
    for _ in 0..max_iter {
        let mut points = poisson_one(window, (n as f64 * 1.5) as usize, rng);
        for _ in 0..n_iter {
            if points.len() < n || unit(rng) < 0.5 {
                let q = window.sample(rng);
                let dk = neighbours_within(&points, q, r2, usize::MAX);
                let ratio = target * gamma.powi(dk) / (points.len() + 1) as f64;
                if unit(rng) < ratio.min(1.0) {
                    points.push(q);
                }
            } else if points.len() > n {
                let idx = uniform_below(rng, points.len() as u64) as usize;
                let dk = neighbours_within(&points, points[idx], r2, idx);
                let den = target * gamma.powi(dk);
                if den == 0.0 {
                    return Err(STRAUSS_FAILURE.into());
                }
                let ratio = points.len() as f64 / den;
                if unit(rng) < ratio.min(1.0) {
                    points.remove(idx);
                }
            }
        }
        if points.len() >= n {
            points.truncate(n);
            return Ok(points);
        }
    }
    Err(STRAUSS_FAILURE.into())
}

/// `n_sims` Strauss patterns of `n_obs` points: `gamma < 1` inhibits pairs closer than `r`,
/// `gamma = 1` is Poisson, `gamma > 1` clusters. `max_iter` restarts bound the attempts per
/// replication.
#[allow(clippy::too_many_arguments)]
pub fn strauss(
    window: &Window,
    n_obs: usize,
    n_sims: usize,
    gamma: f64,
    r: f64,
    n_iter: usize,
    max_iter: usize,
    seed: u64,
) -> Result<Vec<Vec<Pt>>, String> {
    window.validate()?;
    if !(gamma.is_finite() && gamma >= 0.0) {
        return Err("gamma must be a non-negative number".into());
    }
    if !(r.is_finite() && r > 0.0) {
        return Err("r must be a positive number".into());
    }
    if max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    (0..n_sims)
        .into_par_iter()
        .map(|i| {
            let mut rng = replicate_rng(seed, i as u64);
            strauss_one(window, n_obs, (gamma, r, n_iter, max_iter), &mut rng)
        })
        .collect()
}
