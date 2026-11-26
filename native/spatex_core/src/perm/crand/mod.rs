//! Conditional randomization (`esda.crand`): the permutation engine of the local statistics.
//!
//! For every site `i` the focal value stays fixed and only its neighbors' values are redrawn,
//! sampled without replacement from the other `n - 1` sites. As in esda, ONE shared matrix of
//! permutation indices (`P x max_card`) serves every site, so replicates are correlated across
//! sites by design.
//!
//! # Determinism contract
//!
//! Row `r` of the index matrix is drawn from [`replicate_rng`]`(seed, r)` with a partial
//! Fisher-Yates shuffle, so it depends only on `(seed, r, n, max_card)`. Sites are then
//! processed independently (rayon) from that read-only matrix, hence the output is bit-for-bit
//! the same for any thread count. numba's Mersenne Twister is not reproduced; the statistics
//! agree with PySAL up to Monte Carlo error.
//!
//! # Genericity
//!
//! The engine knows nothing about the statistic. A local statistic supplies the vector the
//! neighbors are drawn from (`values`), its observed value per site, and a closure
//! `stat(i, self_weight, other_weights, draw)` evaluating one replicate, where `draw[c]` is the
//! value assigned to the `c`-th neighbor (column order) of site `i`.

use super::{replicate_rng, significance, summarize, uniform_below, Alternative, SimSummary};
use crate::graph::Csr;
use crate::limits;
use rayon::prelude::*;

/// Options of a conditional randomization run.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Number of permutations `P` (at least 1).
    pub permutations: usize,
    /// Seed of the shared index matrix.
    pub seed: u64,
    /// Which tail(s) the pseudo p-value measures.
    pub alternative: Alternative,
    /// Weight of the single pseudo-neighbor of an island (esda's `island_weight`).
    pub island_weight: f64,
    /// Keep the `n x P` reference distributions.
    pub keep: bool,
}

/// Result of [`crand`].
#[derive(Clone, Debug)]
pub struct Crand {
    /// Pseudo p-value per site (`f64`; esda stores float32).
    pub p_sim: Vec<f64>,
    /// `EI_sim`, `seI_sim`, ... per site.
    pub summary: Vec<SimSummary>,
    /// Site-major `n x P` simulated statistics, only when `keep` was set.
    pub sims: Option<Vec<f64>>,
}

/// Per-site weights with the self weight split off (`W` diagonal) and exact zeros dropped.
struct Sites {
    self_weight: Vec<f64>,
    offsets: Vec<usize>,
    other: Vec<f64>,
}

impl Sites {
    fn new(g: &Csr) -> Sites {
        let mut self_weight = vec![0.0; g.n];
        let mut offsets = Vec::with_capacity(g.n + 1);
        let mut other = Vec::with_capacity(g.nnz());
        offsets.push(0);
        for (i, self_w) in self_weight.iter_mut().enumerate() {
            for k in g.offsets[i]..g.offsets[i + 1] {
                let w = g.weights[k];
                if g.neighbors[k] as usize == i {
                    *self_w = w;
                } else if w != 0.0 {
                    other.push(w);
                }
            }
            offsets.push(other.len());
        }
        Sites {
            self_weight,
            offsets,
            other,
        }
    }

    fn cardinality(&self, i: usize) -> usize {
        self.offsets[i + 1] - self.offsets[i]
    }

    fn max_cardinality(&self) -> usize {
        (0..self.self_weight.len())
            .map(|i| self.cardinality(i))
            .max()
            .unwrap_or(0)
    }
}

/// The statistic evaluated on every site's actual neighbors (column order, self weight split
/// off, islands as in [`crand`] with a zero draw).
///
/// Passing these as `observed` to [`crand`] makes the observed value and the replicates come
/// from the very same floating-point expression, so a replicate that redraws exactly the
/// observed neighbors ties the observed statistic bit for bit instead of differing by an ulp.
pub fn observed_with<F>(g: &Csr, values: &[f64], island_weight: f64, stat: F) -> Vec<f64>
where
    F: Fn(usize, f64, &[f64], &[f64]) -> f64,
{
    let sites = Sites::new(g);
    let mut draw = Vec::new();
    (0..g.n)
        .map(|i| {
            if sites.cardinality(i) == 0 {
                return stat(i, 0.0, &[island_weight], &[0.0]);
            }
            let range = g.offsets[i]..g.offsets[i + 1];
            draw.clear();
            for k in range {
                let j = g.neighbors[k] as usize;
                if j != i && g.weights[k] != 0.0 {
                    draw.push(values[j]);
                }
            }
            let other = &sites.other[sites.offsets[i]..sites.offsets[i + 1]];
            stat(i, sites.self_weight[i], other, &draw)
        })
        .collect()
}

/// The shared index matrix, row-major `permutations x max_card`; every entry indexes the
/// `n - 1` sites other than the focal one.
///
/// Row `r` holds `max_card` distinct ids drawn with a partial Fisher-Yates shuffle from the
/// stream `(seed, r)`. `max_card` must not exceed `n - 1`.
pub fn permuted_ids(n: usize, max_card: usize, permutations: usize, seed: u64) -> Vec<u32> {
    let m = n.saturating_sub(1);
    assert!(max_card <= m, "max_card {max_card} exceeds n - 1 = {m}");
    let mut out = vec![0u32; permutations * max_card];
    if max_card == 0 {
        return out;
    }
    out.par_chunks_mut(max_card).enumerate().for_each_init(
        || {
            (
                (0..m as u32).collect::<Vec<u32>>(),
                Vec::with_capacity(max_card),
            )
        },
        |(buf, swaps), (r, row)| {
            let mut rng = replicate_rng(seed, r as u64);
            swaps.clear();
            for k in 0..max_card {
                let j = k + uniform_below(&mut rng, (m - k) as u64) as usize;
                buf.swap(k, j);
                swaps.push(j);
            }
            row.copy_from_slice(&buf[..max_card]);
            // Undo the swaps so the buffer is the identity again for the next row.
            for (k, &j) in swaps.iter().enumerate().rev() {
                buf.swap(k, j);
            }
        },
    );
    out
}

fn validate(g: &Csr, values: &[f64], observed: &[f64], permutations: usize) -> Result<(), String> {
    if g.n < 2 {
        return Err("conditional randomization needs at least 2 sites".into());
    }
    if values.len() != g.n || observed.len() != g.n {
        return Err(format!(
            "values ({}) and observed ({}) must both have one entry per graph node ({})",
            values.len(),
            observed.len(),
            g.n
        ));
    }
    if permutations == 0 {
        return Err("at least 1 permutation is required".into());
    }
    Ok(())
}

/// Conditional randomization of a local statistic.
///
/// `values` is the vector the neighbors are drawn from, `observed` the statistic per site and
/// `stat(i, self_weight, other_weights, draw)` evaluates one replicate. Islands (no neighbor
/// besides itself) get the pseudo-neighbor weights `[island_weight]` and a zero self weight.
pub fn crand<F>(
    g: &Csr,
    values: &[f64],
    observed: &[f64],
    settings: &Settings,
    stat: F,
) -> Result<Crand, String>
where
    F: Fn(usize, f64, &[f64], &[f64]) -> f64 + Sync,
{
    validate(g, values, observed, settings.permutations)?;
    let sites = Sites::new(g);
    let max_card = sites.max_cardinality().max(1);
    let entries = limits::checked_product(
        "permutations x max cardinality",
        settings.permutations,
        max_card,
    )?;
    limits::ensure_within(
        "permutations x max cardinality",
        entries,
        limits::MAX_PERMUTATION_ENTRIES,
        "use fewer permutations",
    )?;
    let ids = permuted_ids(g.n, max_card, settings.permutations, settings.seed);
    run(&sites, values, observed, &ids, max_card, settings, stat)
}

/// [`crand`] with an explicit index matrix (`permutations x max_card`, row-major), used to pin
/// the column mapping in tests and to share a matrix across statistics.
pub fn crand_with_ids<F>(
    g: &Csr,
    values: &[f64],
    observed: &[f64],
    ids: &[u32],
    max_card: usize,
    settings: &Settings,
    stat: F,
) -> Result<Crand, String>
where
    F: Fn(usize, f64, &[f64], &[f64]) -> f64 + Sync,
{
    validate(g, values, observed, settings.permutations)?;
    let sites = Sites::new(g);
    if max_card < sites.max_cardinality().max(1) {
        return Err(format!(
            "max_card {max_card} is below the graph's maximum cardinality"
        ));
    }
    if ids.len() != settings.permutations * max_card {
        return Err(format!(
            "expected {} ids ({} permutations x {max_card}), got {}",
            settings.permutations * max_card,
            settings.permutations,
            ids.len()
        ));
    }
    if ids.iter().any(|&m| m as usize >= g.n - 1) {
        return Err("permutation id out of range".into());
    }
    run(&sites, values, observed, ids, max_card, settings, stat)
}

fn run<F>(
    sites: &Sites,
    values: &[f64],
    observed: &[f64],
    ids: &[u32],
    max_card: usize,
    settings: &Settings,
    stat: F,
) -> Result<Crand, String>
where
    F: Fn(usize, f64, &[f64], &[f64]) -> f64 + Sync,
{
    let p = settings.permutations;
    let island = [settings.island_weight];
    let per_site: Vec<(f64, SimSummary, Vec<f64>)> = (0..values.len())
        .into_par_iter()
        .map(|i| {
            let (self_weight, other): (f64, &[f64]) = if sites.cardinality(i) == 0 {
                (0.0, &island)
            } else {
                (
                    sites.self_weight[i],
                    &sites.other[sites.offsets[i]..sites.offsets[i + 1]],
                )
            };
            let k = other.len();
            let mut draw = vec![0.0; k];
            let mut rstats = Vec::with_capacity(p);
            for row in ids.chunks_exact(max_card) {
                for (slot, &m) in draw.iter_mut().zip(&row[..k]) {
                    let m = m as usize;
                    *slot = values[if m < i { m } else { m + 1 }];
                }
                rstats.push(stat(i, self_weight, other, &draw));
            }
            let p_sim = significance(settings.alternative, observed[i], &rstats);
            let summary = summarize(observed[i], &rstats);
            (
                p_sim,
                summary,
                if settings.keep { rstats } else { Vec::new() },
            )
        })
        .collect();
    let mut p_sim = Vec::with_capacity(per_site.len());
    let mut summary = Vec::with_capacity(per_site.len());
    let mut sims = settings
        .keep
        .then(|| Vec::with_capacity(per_site.len() * p));
    for (p_i, s_i, rstats) in per_site {
        p_sim.push(p_i);
        summary.push(s_i);
        if let Some(all) = sims.as_mut() {
            all.extend(rstats);
        }
    }
    Ok(Crand {
        p_sim,
        summary,
        sims,
    })
}

#[cfg(test)]
mod tests;
