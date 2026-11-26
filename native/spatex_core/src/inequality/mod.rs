//! Inequality measures (`inequality` 1.2.0): Gini, Theil, Theil decomposition, spatial Gini,
//! Atkinson, Schutz and Wolfson (see [`lorenz`]) and the spatial polarization index
//! (see [`polarization`]).
//!
//! Permutations draw from `perm::replicate_rng`, so results are identical at any thread count.

use rayon::prelude::*;

use crate::graph::Csr;
use crate::perm::{norm_sf, replicate_rng, shuffle, simulate};

/// `numpy.finfo(float).tiny`: zeros are lifted to this before taking logarithms.
const SMALL: f64 = f64::MIN_POSITIVE;

/// Gini coefficient in relative mean difference form (`inequality.gini._gini`).
///
/// `NaN` when the values sum to zero.
pub fn gini(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().sum();
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let weighted: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, v)| 2.0 * (i + 1) as f64 * v)
        .sum();
    (weighted - n * sum - sum) / (n * sum)
}

fn column(y: &[f64], t: usize, n: usize, s: usize) -> impl Iterator<Item = f64> + '_ {
    (0..n).map(move |i| y[i * t + s])
}

/// Theil's T of every column of an `n x t` row-major matrix.
pub fn theil(y: &[f64], n: usize, t: usize) -> Vec<f64> {
    (0..t)
        .map(|s| {
            let col: Vec<f64> = column(y, t, n, s)
                .map(|v| if v == 0.0 { v + SMALL } else { v })
                .collect();
            let total: f64 = col.iter().sum();
            let nf = n as f64;
            col.iter()
                .map(|v| {
                    let share = v / total;
                    share * (nf * share).ln()
                })
                .sum()
        })
        .collect()
}

/// Decomposition of Theil's T by an exhaustive partition.
#[derive(Clone, Debug, PartialEq)]
pub struct TheilD {
    /// Total inequality per column.
    pub t: Vec<f64>,
    /// Between-group inequality per column.
    pub bg: Vec<f64>,
    /// Within-group inequality per column (`t - bg`).
    pub wg: Vec<f64>,
}

fn between_group(y: &[f64], rows: &[usize], n: usize, t: usize, part: &[u32]) -> Vec<f64> {
    let groups = part.iter().max().map_or(0, |&g| g as usize + 1);
    let mut counts = vec![0.0; groups];
    let mut totals = vec![0.0; groups * t];
    let mut overall = vec![0.0; t];
    for (i, &g) in part.iter().enumerate() {
        counts[g as usize] += 1.0;
        for s in 0..t {
            let v = y[rows[i] * t + s];
            totals[g as usize * t + s] += v;
            overall[s] += v;
        }
    }
    (0..t)
        .map(|s| {
            (0..groups)
                .filter(|&g| counts[g] > 0.0)
                .map(|g| {
                    let share = totals[g * t + s] / overall[s];
                    let share = if share == 0.0 { SMALL } else { share };
                    share * (n as f64 / counts[g] * share).ln()
                })
                .sum()
        })
        .collect()
}

fn check_panel(y: &[f64], n: usize, t: usize, part: &[u32]) -> Result<(), String> {
    if y.len() != n * t || part.len() != n {
        return Err(format!(
            "y has {} values for {n} x {t}, partition has {} entries",
            y.len(),
            part.len()
        ));
    }
    Ok(())
}

/// Theil's T and its between / within group decomposition.
pub fn theil_d(y: &[f64], n: usize, t: usize, part: &[u32]) -> Result<TheilD, String> {
    check_panel(y, n, t, part)?;
    let total = theil(y, n, t);
    let rows: Vec<usize> = (0..n).collect();
    let bg = between_group(y, &rows, n, t, part);
    let wg = total.iter().zip(&bg).map(|(a, b)| a - b).collect();
    Ok(TheilD { t: total, bg, wg })
}

/// Permutation p-values `(1 + #{bg* >= bg}) / (P + 1)` per column of the between-group term
/// (`inequality.theil.TheilDSim`): rows of `y` are shuffled across locations.
pub fn theil_d_sim(
    y: &[f64],
    n: usize,
    t: usize,
    part: &[u32],
    permutations: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    check_panel(y, n, t, part)?;
    let identity: Vec<usize> = (0..n).collect();
    let observed = between_group(y, &identity, n, t, part);
    let sims: Vec<Vec<f64>> = (0..permutations)
        .into_par_iter()
        .map(|r| {
            let mut rows = identity.clone();
            shuffle(&mut rows, &mut replicate_rng(seed, r as u64));
            between_group(y, &rows, n, t, part)
        })
        .collect();
    Ok((0..t)
        .map(|s| {
            let ge = sims.iter().filter(|v| v[s] >= observed[s]).count();
            (1.0 + ge as f64) / (permutations as f64 + 1.0)
        })
        .collect())
}

/// Result of [`spatial_gini`].
#[derive(Clone, Debug)]
pub struct SpatialGini {
    /// Gini coefficient.
    pub g: f64,
    /// Sum of absolute differences over neighbor pairs (each pair counted both ways).
    pub wg: f64,
    /// The same over non-neighbor pairs.
    pub wcg: f64,
    /// `wcg / (2 n^2 mean)`.
    pub wcg_share: f64,
    /// Sum of absolute differences over all ordered pairs, `g * den`.
    pub dtotal: f64,
    /// `2 n^2 mean(x)`.
    pub den: f64,
    /// `(wcg / wg) (n_neighbor_pairs / n_distant_pairs)`, expected 1 under randomness.
    pub polarization: f64,
    /// Folded pseudo p-value of `wcg` (`NaN` without permutations).
    pub p_sim: f64,
    /// Mean of the simulated `wcg`.
    pub e_wcg: f64,
    /// Standard deviation (ddof 0) of the simulated `wcg`.
    pub s_wcg: f64,
    /// `(wcg - e_wcg) / s_wcg`.
    pub z_wcg: f64,
    /// `1 - Phi(z_wcg)`.
    pub p_z_sim: f64,
    /// Upper-tail pseudo p-value of the polarization.
    pub polarization_p_sim: f64,
    /// Simulated `wcg` per permutation.
    pub wcgp: Vec<f64>,
    /// Simulated polarization per permutation.
    pub polarization_sim: Vec<f64>,
}

/// Sum of |x_i - x_j| over the non-zero neighbor entries of the graph.
fn neighbor_sad(x: &[f64], g: &Csr) -> f64 {
    let mut sum = 0.0;
    for i in 0..g.n {
        for e in g.offsets[i]..g.offsets[i + 1] {
            if g.weights[e] != 0.0 {
                sum += (x[i] - x[g.neighbors[e] as usize]).abs();
            }
        }
    }
    sum
}

/// Spatial Gini (`inequality.gini.Gini_Spatial`): how much of the inequality lies between
/// neighbors versus non-neighbors, with a permutation test on the non-neighbor part.
///
/// The graph is read as binary: every non-zero entry is a neighbor.
pub fn spatial_gini(
    x: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
) -> Result<SpatialGini, String> {
    if x.len() != g.n {
        return Err(format!(
            "x has {} values but the graph has {} nodes",
            x.len(),
            g.n
        ));
    }
    let n = x.len() as f64;
    let gi = gini(x);
    let mean = x.iter().sum::<f64>() / n;
    let den = mean * 2.0 * n * n;
    let d = gi * den;
    let wg = neighbor_sad(x, g);
    let wcg = d - wg;
    let n_pairs = n * (n - 1.0) / 2.0;
    let n_n = g.nonzero_count() as f64 / 2.0;
    let n_d = n_pairs - n_n;
    let polarization = (wcg / wg) * (n_n / n_d);
    let mut out = SpatialGini {
        g: gi,
        wg,
        wcg,
        wcg_share: wcg / den,
        dtotal: d,
        den,
        polarization,
        p_sim: f64::NAN,
        e_wcg: f64::NAN,
        s_wcg: f64::NAN,
        z_wcg: f64::NAN,
        p_z_sim: f64::NAN,
        polarization_p_sim: f64::NAN,
        wcgp: Vec::new(),
        polarization_sim: Vec::new(),
    };
    if permutations == 0 {
        return Ok(out);
    }
    let scale = n_n / n_d;
    let wcgp: Vec<f64> = simulate(x, permutations, seed, |xp| d - neighbor_sad(xp, g));
    let p = permutations as f64;
    let larger = wcgp.iter().filter(|&&v| v >= wcg).count();
    let larger = larger.min(permutations - larger);
    out.p_sim = (larger as f64 + 1.0) / (p + 1.0);
    out.e_wcg = wcgp.iter().sum::<f64>() / p;
    out.s_wcg = (wcgp.iter().map(|v| (v - out.e_wcg).powi(2)).sum::<f64>() / p).sqrt();
    out.z_wcg = (wcg - out.e_wcg) / out.s_wcg;
    out.p_z_sim = norm_sf(out.z_wcg);
    out.polarization_sim = wcgp.iter().map(|w| w / (d - w) * scale).collect();
    let above = out
        .polarization_sim
        .iter()
        .filter(|&&v| v >= polarization)
        .count();
    out.polarization_p_sim = (above as f64 + 1.0) / (p + 1.0);
    out.wcgp = wcgp;
    Ok(out)
}

pub mod lorenz;
pub mod polarization;

#[cfg(test)]
mod lorenz_tests;
#[cfg(test)]
mod polarization_tests;
#[cfg(test)]
mod tests;
