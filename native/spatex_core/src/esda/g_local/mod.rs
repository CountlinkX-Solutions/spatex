//! Local Getis-Ord `G_i` and `G*_i` (`esda.G_Local`).
//!
//! The input graph is *not* pre-transformed: [`g_local`] first decides the diagonal (the
//! `star` rules of esda's `_infer_star_and_structure_w`) on the graph as given, and only then
//! applies the requested transformation (`R` or `B`). Inference by conditional randomization
//! draws neighbors from the raw `y` ([`crate::perm::crand`]).

use crate::esda::moran::{check_finite, check_graph};
use crate::graph::{Csr, Transform};
use crate::perm::crand::{self, Crand, Settings};
use crate::perm::norm_sf;

/// How the diagonal (self weight) is chosen, esda's `star` argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Star {
    /// `star=None`: `G*` exactly when the input graph has a nonzero diagonal.
    Infer,
    /// `star=False`: any diagonal is removed.
    Off,
    /// `star=True`: keep a nonzero diagonal; otherwise assign 1 (binary input or transform) or
    /// the row maximum of the input weights (row-standardization).
    On,
    /// A number: assign it as the self weight of every node.
    Weight(f64),
}

impl Star {
    /// Parses the NIF encoding `"infer" | "off" | "on" | "weight"` plus the weight.
    pub fn parse(kind: &str, weight: f64) -> Result<Star, String> {
        match kind {
            "infer" => Ok(Star::Infer),
            "off" => Ok(Star::Off),
            "on" => Ok(Star::On),
            "weight" if weight.is_finite() => Ok(Star::Weight(weight)),
            "weight" => Err("the star self weight must be finite".into()),
            other => Err(format!(
                "unknown star mode {other:?}, expected infer, off, on or weight"
            )),
        }
    }
}

/// The two transformations `G_Local` supports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GTransform {
    /// Row-standardized.
    R,
    /// Binary.
    B,
}

impl GTransform {
    /// Parses `"r"` or `"b"`, case-insensitively (esda asserts on anything else).
    pub fn parse(s: &str) -> Result<GTransform, String> {
        match s.to_ascii_lowercase().as_str() {
            "r" => Ok(GTransform::R),
            "b" => Ok(GTransform::B),
            _ => Err(format!(
                "G_Local supports only the transforms r and b, got {s:?}"
            )),
        }
    }
}

/// Options of [`g_local`].
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Diagonal policy.
    pub star: Star,
    /// Transformation applied after the diagonal is settled.
    pub transform: GTransform,
    /// The input graph carries the binary transform (esda then assigns self weight 1).
    pub input_binary: bool,
}

/// Observed local G statistics with their normality inference.
#[derive(Clone, Debug)]
pub struct GLocal {
    /// Number of sites.
    pub n: usize,
    /// `G*` (nonzero diagonal after preparation) rather than `G`.
    pub star: bool,
    /// `G_i` / `G*_i`.
    pub gs: Vec<f64>,
    /// Expectation `card_i / N`.
    pub egs: Vec<f64>,
    /// Variance under the normality assumption.
    pub vgs: Vec<f64>,
    /// `(Gs - EGs) / sqrt(VGs)` (NaN for islands).
    pub zs: Vec<f64>,
    /// `sf(|Zs|)`, one-tailed on `|Z|` as in esda.
    pub p_norm: Vec<f64>,
    /// `sum(y)`.
    pub y_sum: f64,
    /// The prepared (diagonal settled, transformed) weights used by the statistic.
    pub weights: Csr,
}

/// Rebuilds `g` with every diagonal entry replaced by `self_weight(i)`; a zero self weight
/// removes the entry (nodes left without entries keep a zero-weight self-loop island).
fn assign_self_weight(g: &Csr, self_weight: impl Fn(usize) -> f64) -> Csr {
    let mut offsets = Vec::with_capacity(g.n + 1);
    let mut neighbors = Vec::with_capacity(g.nnz() + g.n);
    let mut weights = Vec::with_capacity(g.nnz() + g.n);
    offsets.push(0);
    for i in 0..g.n {
        let w_self = self_weight(i);
        let mut placed = w_self == 0.0;
        let row_start = neighbors.len();
        for k in g.offsets[i]..g.offsets[i + 1] {
            let j = g.neighbors[k] as usize;
            if j == i {
                continue;
            }
            if !placed && j > i {
                neighbors.push(i as u32);
                weights.push(w_self);
                placed = true;
            }
            neighbors.push(j as u32);
            weights.push(g.weights[k]);
        }
        if !placed {
            neighbors.push(i as u32);
            weights.push(w_self);
        }
        if neighbors.len() == row_start {
            neighbors.push(i as u32);
            weights.push(0.0);
        }
        offsets.push(neighbors.len());
    }
    Csr {
        n: g.n,
        offsets,
        neighbors,
        weights,
    }
}

fn diagonal(g: &Csr) -> Vec<f64> {
    (0..g.n)
        .map(|i| {
            (g.offsets[i]..g.offsets[i + 1])
                .find(|&k| g.neighbors[k] as usize == i)
                .map_or(0.0, |k| g.weights[k])
        })
        .collect()
}

fn row_max(g: &Csr, i: usize) -> f64 {
    let row = &g.weights[g.offsets[i]..g.offsets[i + 1]];
    // An island's row holds only the zero self-loop, so its maximum is 0.
    row.iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
}

/// Settles the diagonal and transforms (`_infer_star_and_structure_w`).
fn prepare(g: &Csr, opts: &Options) -> Csr {
    let diag = diagonal(g);
    let zero_diag = diag.iter().all(|d| *d == 0.0);
    let star = match opts.star {
        Star::Infer => Some(!zero_diag),
        Star::Off => Some(false),
        Star::On => Some(true),
        Star::Weight(_) => None,
    };
    let settled = match (star, zero_diag) {
        (Some(false), false) => assign_self_weight(g, |_| 0.0),
        (Some(true), true) => {
            if opts.transform == GTransform::B || opts.input_binary {
                assign_self_weight(g, |_| 1.0)
            } else {
                assign_self_weight(g, |i| row_max(g, i))
            }
        }
        (Some(_), _) => g.clone(),
        (None, _) => match opts.star {
            Star::Weight(w) => assign_self_weight(g, |_| w),
            _ => unreachable!("only a numeric star leaves the mode undecided"),
        },
    };
    let t = match opts.transform {
        GTransform::R => Transform::R,
        GTransform::B => Transform::B,
    };
    settled.transform(t)
}

/// Local `G` / `G*` of `y` over `g`.
///
/// `y` must be finite, non-negative and have a positive sum (esda validates none of this).
pub fn g_local(y: &[f64], g: &Csr, opts: &Options) -> Result<GLocal, String> {
    check_graph(y.len(), g)?;
    if y.len() < 3 {
        return Err("at least 3 observations are required".into());
    }
    check_finite(y, "y")?;
    if let Some(i) = y.iter().position(|v| *v < 0.0) {
        return Err(format!(
            "y must be non-negative for the Getis-Ord statistics (index {i})"
        ));
    }
    let y_sum: f64 = y.iter().sum();
    if y_sum <= 0.0 {
        return Err("the sum of y must be positive: Getis-Ord G is undefined".into());
    }
    let w = prepare(g, opts);
    let diag = diagonal(&w);
    let star = diag.iter().any(|d| *d > 0.0);
    let remove_self = if star { 0.0 } else { 1.0 };
    let n = y.len() as f64;
    let big_n = n - remove_self;
    let sum_sq: f64 = y.iter().map(|v| v * v).sum();
    let lag = w.lag(y)?;

    let len = y.len();
    let mut out = GLocal {
        n: len,
        star,
        gs: Vec::with_capacity(len),
        egs: Vec::with_capacity(len),
        vgs: Vec::with_capacity(len),
        zs: Vec::with_capacity(len),
        p_norm: Vec::with_capacity(len),
        y_sum,
        weights: Csr {
            n: 0,
            offsets: Vec::new(),
            neighbors: Vec::new(),
            weights: Vec::new(),
        },
    };
    for (i, &yi) in y.iter().enumerate() {
        let gs = lag[i] / (y_sum - yi * remove_self);
        let mean = (y_sum - yi * remove_self) / big_n;
        let mean_sq = (sum_sq - yi * yi * remove_self) / big_n;
        let var = mean_sq - mean * mean;
        let card: f64 = w.weights[w.offsets[i]..w.offsets[i + 1]].iter().sum();
        let eg = card / big_n;
        let mut vg = card * (big_n - card);
        vg /= big_n - 1.0;
        vg *= 1.0 / (big_n * big_n);
        vg *= var / (mean * mean);
        let z = (gs - eg) / vg.sqrt();
        out.gs.push(gs);
        out.egs.push(eg);
        out.vgs.push(vg);
        out.zs.push(z);
        out.p_norm.push(norm_sf(z.abs()));
    }
    out.weights = w;
    Ok(out)
}

/// Conditional randomization of `G` / `G*` (neighbors drawn from the raw `y`).
///
/// Observed values are re-evaluated with the replicate formula (see
/// [`crand::observed_with`]) except for sites without neighbors, whose observed statistic is the
/// analytic `Gs` while every replicate is `0` (the esda quirk behind their smallest p-value).
pub fn g_local_permutations(y: &[f64], r: &GLocal, settings: &Settings) -> Result<Crand, String> {
    if y.len() != r.n {
        return Err(format!(
            "y has length {} but the statistic was computed for {} sites (length mismatch)",
            y.len(),
            r.n
        ));
    }
    let g = &r.weights;
    let y_sum = r.y_sum;
    let star = r.star;
    let stat = |i: usize, self_w: f64, other: &[f64], draw: &[f64]| {
        let lag: f64 = other.iter().zip(draw).map(|(w, d)| w * d).sum();
        if star {
            (lag + self_w * y[i]) / y_sum
        } else {
            lag / (y_sum - y[i])
        }
    };
    let mut observed = crand::observed_with(g, y, settings.island_weight, stat);
    for (i, obs) in observed.iter_mut().enumerate() {
        let has_neighbor = (g.offsets[i]..g.offsets[i + 1])
            .any(|k| g.neighbors[k] as usize != i && g.weights[k] != 0.0);
        if !has_neighbor {
            *obs = r.gs[i];
        }
    }
    crand::crand(g, y, &observed, settings, stat)
}

#[cfg(test)]
mod tests;
