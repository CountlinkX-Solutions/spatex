//! Distance statistics (`pointpats/distance_statistics.py`): Ripley's G, F, J, K, L and their
//! simulation envelopes. All estimators are uncorrected for edge effects, as in pointpats.

use super::random::poisson_one;
use super::{bounds, Pt, Window};
use crate::perm::replicate_rng;
use rayon::prelude::*;
use rstar::{primitives::GeomWithData, PointDistance, RTree};

/// How the distance grid is chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum SupportSpec {
    /// `linspace(0, max_nn, m)` over the pattern's maximum nearest-neighbour distance.
    Int(usize),
    /// `linspace(start, stop, num)`.
    Linspace {
        /// First value.
        start: f64,
        /// Last value.
        stop: f64,
        /// Number of values.
        num: usize,
    },
    /// The values as given (ascending).
    Explicit(Vec<f64>),
}

/// numpy's `linspace(start, stop, num)` arithmetic (`i * step + start`, last forced to `stop`).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    match num {
        0 => vec![],
        1 => vec![start],
        _ => {
            let step = (stop - start) / (num - 1) as f64;
            let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
            v[num - 1] = stop;
            v
        }
    }
}

type Tree = RTree<GeomWithData<[f64; 2], usize>>;

fn build_tree(p: &[Pt]) -> Tree {
    RTree::bulk_load(
        p.iter()
            .enumerate()
            .map(|(i, q)| GeomWithData::new(*q, i))
            .collect(),
    )
}

/// Nearest-neighbour distance of every point to the others (duplicates give 0).
fn nn_self(p: &[Pt], tree: &Tree) -> Vec<f64> {
    p.iter()
        .enumerate()
        .map(|(i, q)| {
            tree.nearest_neighbor_iter(*q)
                .take(2)
                .find(|c| c.data != i)
                .map_or(f64::NAN, |c| c.distance_2(q).sqrt())
        })
        .collect()
}

/// Distance from each `from` point to its nearest pattern point.
fn nn_to(from: &[Pt], tree: &Tree) -> Vec<f64> {
    from.iter()
        .map(|q| {
            tree.nearest_neighbor(*q)
                .map_or(f64::NAN, |c| c.distance_2(q).sqrt())
        })
        .collect()
}

fn resolve_support(spec: &SupportSpec, nn: &[f64]) -> Result<Vec<f64>, String> {
    let support = match spec {
        SupportSpec::Int(m) => {
            let max_nn = nn.iter().copied().fold(0.0_f64, f64::max);
            linspace(0.0, max_nn, *m)
        }
        SupportSpec::Linspace { start, stop, num } => linspace(*start, *stop, *num),
        SupportSpec::Explicit(v) => v.clone(),
    };
    if support.len() < 2 {
        return Err("support needs at least 2 values".into());
    }
    if support.iter().any(|v| !v.is_finite()) || support.windows(2).any(|w| w[0] > w[1]) {
        return Err("support must be finite and ascending".into());
    }
    Ok(support)
}

/// `numpy.histogram(values, bins=support)` followed by pointpats' normalization:
/// `[0, cumsum(counts) / sum(counts)]` (bins half-open, the last closed; out-of-range dropped).
fn cdf_on(values: &[f64], support: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(f64::total_cmp);
    let m = support.len();
    let mut cum = Vec::with_capacity(m);
    for e in &support[..m - 1] {
        cum.push(sorted.partition_point(|v| v < e));
    }
    cum.push(sorted.partition_point(|v| *v <= support[m - 1]));
    let total = (cum[m - 1] - cum[0]) as f64;
    let mut out = Vec::with_capacity(m);
    out.push(0.0);
    for i in 1..m {
        out.push((cum[i] - cum[0]) as f64 / total);
    }
    out
}

fn bbox_area(p: &[Pt]) -> f64 {
    let b = bounds(p);
    ((b[2] - b[0]) * (b[3] - b[1])).abs()
}

/// Pair counts: `out[k] = #{pairs with distance < support[k]}` (strict).
fn pair_counts(p: &[Pt], support: &[f64], parallel: bool) -> Vec<u64> {
    let m = support.len();
    let row = |i: usize, hist: &mut Vec<u64>| {
        for j in (i + 1)..p.len() {
            let d = ((p[i][0] - p[j][0]).powi(2) + (p[i][1] - p[j][1]).powi(2)).sqrt();
            let k = support.partition_point(|e| *e <= d);
            if k < m {
                hist[k] += 1;
            }
        }
    };
    let hist = if parallel {
        (0..p.len())
            .into_par_iter()
            .fold(
                || vec![0u64; m],
                |mut h, i| {
                    row(i, &mut h);
                    h
                },
            )
            .reduce(
                || vec![0u64; m],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    } else {
        let mut h = vec![0u64; m];
        (0..p.len()).for_each(|i| row(i, &mut h));
        h
    };
    let mut run = 0;
    hist.into_iter()
        .map(|c| {
            run += c;
            run
        })
        .collect()
}

fn k_stat(p: &[Pt], support: &[f64], parallel: bool) -> Vec<f64> {
    let n = p.len() as f64;
    let intensity = n / bbox_area(p);
    pair_counts(p, support, parallel)
        .into_iter()
        .map(|c| c as f64 * 2.0 / n / intensity)
        .collect()
}

fn l_from_k(k: &[f64], support: &[f64], linearized: bool) -> Vec<f64> {
    k.iter()
        .zip(support)
        .map(|(v, s)| {
            let l = (v / std::f64::consts::PI).sqrt();
            if linearized {
                l - s
            } else {
                l
            }
        })
        .collect()
}

fn j_from(g: &[f64], f: &[f64]) -> Vec<f64> {
    g.iter()
        .zip(f)
        .map(|(g, f)| (1.0 - g) / (1.0 - f))
        .collect()
}

fn require(p: &[Pt]) -> Result<(), String> {
    if p.len() < 2 {
        Err("distance statistics need at least 2 points".into())
    } else {
        Ok(())
    }
}

/// A support grid and its statistic.
pub type Curve = (Vec<f64>, Vec<f64>);

/// Nearest-neighbour distribution `G`.
pub fn g(p: &[Pt], spec: &SupportSpec) -> Result<Curve, String> {
    require(p)?;
    let nn = nn_self(p, &build_tree(p));
    let support = resolve_support(spec, &nn)?;
    let stat = cdf_on(&nn, &support);
    Ok((support, stat))
}

/// Empty-space distribution `F` against the given empty-space points.
pub fn f(p: &[Pt], empty: &[Pt], spec: &SupportSpec) -> Result<Curve, String> {
    require(p)?;
    if empty.is_empty() {
        return Err("empty_points must not be empty".into());
    }
    let tree = build_tree(p);
    let support = resolve_support(spec, &nn_self(p, &tree))?;
    let stat = cdf_on(&nn_to(empty, &tree), &support);
    Ok((support, stat))
}

/// Cuts at the first infinite or NaN value (exclusive); `None` when all are finite.
fn first_invalid(v: &[f64]) -> Option<usize> {
    v.iter().position(|x| !x.is_finite())
}

/// `J = (1 - G) / (1 - F)`, optionally truncated at the first infinite/NaN value.
pub fn j(p: &[Pt], empty: &[Pt], spec: &SupportSpec, truncate: bool) -> Result<Curve, String> {
    require(p)?;
    if empty.is_empty() {
        return Err("empty_points must not be empty".into());
    }
    let tree = build_tree(p);
    let nn = nn_self(p, &tree);
    let support = resolve_support(spec, &nn)?;
    let mut stat = j_from(
        &cdf_on(&nn, &support),
        &cdf_on(&nn_to(empty, &tree), &support),
    );
    let mut support = support;
    if truncate {
        if let Some(c) = first_invalid(&stat) {
            stat.truncate(c);
            support.truncate(c);
        }
    }
    Ok((support, stat))
}

/// Ripley's K, uncorrected, with the bounding-box area of the pattern.
pub fn k(p: &[Pt], spec: &SupportSpec) -> Result<Curve, String> {
    require(p)?;
    let nn = nn_self(p, &build_tree(p));
    let support = resolve_support(spec, &nn)?;
    let stat = k_stat(p, &support, true);
    Ok((support, stat))
}

/// `L = sqrt(K / pi)`, or `L - d` when `linearized`.
pub fn l(p: &[Pt], spec: &SupportSpec, linearized: bool) -> Result<Curve, String> {
    let (support, kv) = k(p, spec)?;
    let stat = l_from_k(&kv, &support, linearized);
    Ok((support, stat))
}

/// Which function a simulation test evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Empty-space function.
    F,
    /// Nearest-neighbour function.
    G,
    /// `(1 - G) / (1 - F)`.
    J,
    /// Ripley's K.
    K,
    /// `sqrt(K / pi)`.
    L,
}

/// Options of a simulation test.
#[derive(Clone, Debug)]
pub struct TestOptions {
    /// The function.
    pub kind: Kind,
    /// The support grid.
    pub spec: SupportSpec,
    /// Simulation window.
    pub window: Window,
    /// Number of simulated patterns.
    pub n_simulations: usize,
    /// Seed.
    pub seed: u64,
    /// Return every simulated curve.
    pub keep_simulations: bool,
    /// Truncate J at its first infinite/NaN value.
    pub truncate: bool,
    /// Linearize L.
    pub linearized: bool,
    /// Number of empty-space points for F and J.
    pub n_empty: usize,
}

/// Result of [`ripley_test`].
#[derive(Clone, Debug)]
pub struct TestResult {
    /// Support grid.
    pub support: Vec<f64>,
    /// Observed statistic.
    pub statistic: Vec<f64>,
    /// `min(p, 1 - p)` with `p = (#{sim >= obs} + 1) / (N + 1)`; NaN without simulations.
    pub pvalue: Vec<f64>,
    /// Row-major `n_simulations x len(support)` when kept.
    pub simulations: Option<Vec<f64>>,
    /// Per-support minimum over simulations (NaN without simulations).
    pub envelope_low: Vec<f64>,
    /// Per-support maximum over simulations (NaN without simulations).
    pub envelope_high: Vec<f64>,
}

const EMPTY_STREAM: u64 = u64::MAX;

/// Seeded Monte Carlo test of a distance function against CSR (`_ripley_test`).
///
/// Replicated pointpats behavior: empty-space points are always drawn from the bounding box of
/// the observed pattern; K/L use the bounding-box area of each (simulated) pattern, never the
/// window area; the p-value folds as `min(p, 1 - p)`.
pub fn ripley_test(p: &[Pt], o: &TestOptions) -> Result<TestResult, String> {
    require(p)?;
    o.window.validate()?;
    let n = p.len();
    let tree = build_tree(p);
    let nn = nn_self(p, &tree);
    let support = resolve_support(&o.spec, &nn)?;
    let uses_empty = matches!(o.kind, Kind::F | Kind::J);
    let empty: Vec<Pt> = if uses_empty {
        if o.n_empty == 0 {
            return Err("n_empty must be positive".into());
        }
        let b = bounds(p);
        let w = Window::Bbox(b);
        w.validate()?;
        poisson_one(&w, o.n_empty, &mut replicate_rng(o.seed, EMPTY_STREAM))
    } else {
        vec![]
    };

    let curve = |pat: &[Pt], tree: Option<&Tree>, nn: Option<&[f64]>| -> Vec<f64> {
        match o.kind {
            Kind::G => cdf_on(nn.unwrap_or(&[]), &support),
            Kind::F => cdf_on(&nn_to(&empty, tree.expect("tree")), &support),
            Kind::J => j_from(
                &cdf_on(nn.unwrap_or(&[]), &support),
                &cdf_on(&nn_to(&empty, tree.expect("tree")), &support),
            ),
            Kind::K => k_stat(pat, &support, false),
            Kind::L => l_from_k(&k_stat(pat, &support, false), &support, o.linearized),
        }
    };

    let mut observed = if matches!(o.kind, Kind::K | Kind::L) {
        let kv = k_stat(p, &support, true);
        if o.kind == Kind::L {
            l_from_k(&kv, &support, o.linearized)
        } else {
            kv
        }
    } else {
        curve(p, Some(&tree), Some(nn.as_slice()))
    };
    let mut support_out = support.clone();
    if o.kind == Kind::J && o.truncate {
        if let Some(c) = first_invalid(&observed) {
            observed.truncate(c);
            support_out.truncate(c);
        }
    }
    let m = observed.len();

    let simulate = |i: usize| -> Vec<f64> {
        let pat = poisson_one(&o.window, n, &mut replicate_rng(o.seed, i as u64));
        let mut v = if matches!(o.kind, Kind::K | Kind::L) {
            curve(&pat, None, None)
        } else {
            let t = build_tree(&pat);
            let d = if matches!(o.kind, Kind::G | Kind::J) {
                nn_self(&pat, &t)
            } else {
                vec![]
            };
            curve(&pat, Some(&t), Some(d.as_slice()))
        };
        v.truncate(m);
        v
    };

    let ns = o.n_simulations;
    if ns == 0 {
        return Ok(TestResult {
            support: support_out,
            statistic: observed,
            pvalue: vec![f64::NAN; m],
            simulations: o.keep_simulations.then(Vec::new),
            envelope_low: vec![f64::NAN; m],
            envelope_high: vec![f64::NAN; m],
        });
    }

    let (counts, low, high, kept) = if o.keep_simulations {
        let rows: Vec<Vec<f64>> = (0..ns).into_par_iter().map(simulate).collect();
        let mut counts = vec![0usize; m];
        let mut low = vec![f64::INFINITY; m];
        let mut high = vec![f64::NEG_INFINITY; m];
        for r in &rows {
            for j in 0..m {
                counts[j] += usize::from(r[j] >= observed[j]);
                low[j] = low[j].min(r[j]);
                high[j] = high[j].max(r[j]);
            }
        }
        (counts, low, high, Some(rows.concat()))
    } else {
        let init = || {
            (
                vec![0usize; m],
                vec![f64::INFINITY; m],
                vec![f64::NEG_INFINITY; m],
            )
        };
        let (c, l, h) = (0..ns)
            .into_par_iter()
            .fold(init, |mut acc, i| {
                let r = simulate(i);
                for j in 0..m {
                    acc.0[j] += usize::from(r[j] >= observed[j]);
                    acc.1[j] = acc.1[j].min(r[j]);
                    acc.2[j] = acc.2[j].max(r[j]);
                }
                acc
            })
            .reduce(init, |mut a, b| {
                for j in 0..m {
                    a.0[j] += b.0[j];
                    a.1[j] = a.1[j].min(b.1[j]);
                    a.2[j] = a.2[j].max(b.2[j]);
                }
                a
            });
        (c, l, h, None)
    };
    let pvalue = counts
        .iter()
        .map(|&c| {
            let p = (c as f64 + 1.0) / (ns as f64 + 1.0);
            p.min(1.0 - p)
        })
        .collect();
    Ok(TestResult {
        support: support_out,
        statistic: observed,
        pvalue,
        simulations: kept,
        envelope_low: low,
        envelope_high: high,
    })
}
