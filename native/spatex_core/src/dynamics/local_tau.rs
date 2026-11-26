//! Local and regional Kendall tau (`giddy.rank`: `Tau_Local`, `Tau_Local_Neighbor`,
//! `Tau_Local_Neighborhood`, `Tau_Regional`).
//!
//! All four build on the pair concordance `S_ij = sign((x_i - x_j)(y_i - y_j))`. Weights are
//! binary (giddy sets `transform = "b"`); self pairs and zero weights are ignored. Islands have
//! an undefined (`NaN`) neighbor statistic, as giddy's `0 / 0`. Inference relabels sites with
//! seeded ChaCha8 streams (giddy uses numpy's global generator); the local p-value is giddy's
//! `(min(#sim >= obs, #sim <= obs) + 1) / (P + 1)`.

use rayon::prelude::*;

use crate::graph::Csr;
use crate::perm::crand::{self, Settings};
use crate::perm::{replicate_rng, shuffle, Alternative};

fn concordance(x: &[f64], y: &[f64], i: usize, j: usize) -> f64 {
    let dxdy = (x[i] - x[j]) * (y[i] - y[j]);
    if dxdy > 0.0 {
        1.0
    } else if dxdy < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn check(x: &[f64], y: &[f64]) -> Result<(), String> {
    if x.len() != y.len() {
        return Err(format!(
            "x and y must have the same length (length mismatch), got {} and {}",
            x.len(),
            y.len()
        ));
    }
    if x.len() < 2 {
        return Err("at least two observations are required".into());
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err("x and y must be finite".into());
    }
    Ok(())
}

/// `Tau_Local`: `(tau, tau_local)` with `tau = sum S / (n (n - 1))` and
/// `tau_local_i = sum_j S_ij / (n - 1)`.
pub fn tau_local(x: &[f64], y: &[f64]) -> Result<(f64, Vec<f64>), String> {
    check(x, y)?;
    let n = x.len();
    let rows: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| (0..n).map(|j| concordance(x, y, i, j)).sum())
        .collect();
    let total: f64 = rows.iter().sum();
    let nf = n as f64;
    Ok((
        total / (nf * (nf - 1.0)),
        rows.iter().map(|r| r / (nf - 1.0)).collect(),
    ))
}

/// Binary, self-free neighbor lists of `g`.
fn neighbor_lists(g: &Csr) -> Vec<Vec<usize>> {
    (0..g.n)
        .map(|i| {
            g.row(i)
                .filter(|&k| g.weights[k] != 0.0 && g.neighbors[k] as usize != i)
                .map(|k| g.neighbors[k] as usize)
                .collect()
        })
        .collect()
}

/// A local statistic with its permutation inference.
#[derive(Clone, Debug)]
pub struct LocalTau {
    /// The observed statistic per site (`NaN` for islands).
    pub value: Vec<f64>,
    /// giddy's pseudo p-value per site (`NaN` for islands), when permutations ran.
    pub p: Option<Vec<f64>>,
    /// Site-major `n x P` simulations, when requested.
    pub sims: Option<Vec<f64>>,
}

/// giddy's local pseudo p-value from one site's simulations.
fn local_p(observed: f64, sims: &[f64]) -> f64 {
    if observed.is_nan() {
        return f64::NAN;
    }
    let larger = sims.iter().filter(|&&s| s >= observed).count();
    let smaller = sims.iter().filter(|&&s| s <= observed).count();
    (larger.min(smaller) as f64 + 1.0) / (sims.len() as f64 + 1.0)
}

/// Conditional randomization of a neighbor-set statistic: each replicate gives site `i`
/// `|N(i)|` distinct random other sites (crand on site indices).
fn randomize<F>(
    g: &Csr,
    observed: &[f64],
    permutations: usize,
    seed: u64,
    keep: bool,
    stat: F,
) -> Result<LocalTau, String>
where
    F: Fn(usize, &[usize]) -> f64 + Sync,
{
    let settings = Settings {
        permutations,
        seed,
        alternative: Alternative::Directed,
        island_weight: 0.0,
        keep: true,
    };
    let ids: Vec<f64> = (0..g.n).map(|i| i as f64).collect();
    let crand_stat = |i: usize, _self_w: f64, _other: &[f64], draw: &[f64]| {
        let drawn: Vec<usize> = draw.iter().map(|&d| d as usize).collect();
        stat(i, &drawn)
    };
    // crand needs a finite observation; islands are reported as NaN afterwards.
    let finite: Vec<f64> = observed
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    let c = crand::crand(g, &ids, &finite, &settings, crand_stat)?;
    let sims = c.sims.unwrap_or_default();
    let p = (0..g.n)
        .map(|i| local_p(observed[i], &sims[i * permutations..(i + 1) * permutations]))
        .collect();
    let sims = if keep {
        Some(
            sims.iter()
                .enumerate()
                .map(|(e, s)| {
                    if observed[e / permutations].is_nan() {
                        f64::NAN
                    } else {
                        *s
                    }
                })
                .collect(),
        )
    } else {
        None
    };
    Ok(LocalTau {
        value: observed.to_vec(),
        p: Some(p),
        sims,
    })
}

/// `Tau_Local_Neighbor`: `(result, weights)` with `tau_ln_i = sum_{j in N(i)} S_ij / |N(i)|` and
/// `weights_i = |N(i)| / s0`.
pub fn tau_local_neighbor(
    x: &[f64],
    y: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<(LocalTau, Vec<f64>), String> {
    check(x, y)?;
    if g.n != x.len() {
        return Err(format!(
            "x has length {} but the graph has {} nodes (length mismatch)",
            x.len(),
            g.n
        ));
    }
    let nb = neighbor_lists(g);
    let s0: usize = nb.iter().map(Vec::len).sum();
    let stat = |i: usize, set: &[usize]| {
        set.iter().map(|&j| concordance(x, y, i, j)).sum::<f64>() / set.len() as f64
    };
    let observed: Vec<f64> = (0..g.n).map(|i| stat(i, &nb[i])).collect();
    let weights = nb.iter().map(|s| s.len() as f64 / s0 as f64).collect();
    let result = if permutations > 0 {
        randomize(g, &observed, permutations, seed, keep, stat)?
    } else {
        LocalTau {
            value: observed,
            p: None,
            sims: None,
        }
    };
    Ok((result, weights))
}

/// `Tau_Local_Neighborhood`: the tau of the neighborhood `{i} + N(i)`,
/// `sum_{a, b} S_ab / (m (m - 1))` over its `m` members.
pub fn tau_local_neighborhood(
    x: &[f64],
    y: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<LocalTau, String> {
    check(x, y)?;
    if g.n != x.len() {
        return Err(format!(
            "x has length {} but the graph has {} nodes (length mismatch)",
            x.len(),
            g.n
        ));
    }
    let nb = neighbor_lists(g);
    let stat = |i: usize, set: &[usize]| {
        let members: Vec<usize> = std::iter::once(i).chain(set.iter().copied()).collect();
        let m = members.len() as f64;
        let mut total = 0.0;
        for &a in &members {
            for &b in &members {
                total += concordance(x, y, a, b);
            }
        }
        total / (m * (m - 1.0))
    };
    let observed: Vec<f64> = (0..g.n).map(|i| stat(i, &nb[i])).collect();
    if permutations > 0 {
        randomize(g, &observed, permutations, seed, keep, stat)
    } else {
        Ok(LocalTau {
            value: observed,
            p: None,
            sims: None,
        })
    }
}

/// Result of [`tau_regional`]: `k x k` matrices, row-major.
#[derive(Clone, Debug)]
pub struct RegionalTau {
    /// The distinct regime labels, ascending.
    pub regimes: Vec<i64>,
    pub tau_reg: Vec<f64>,
    pub p: Option<Vec<f64>>,
    /// Replicate-major `P x k x k` simulations, when requested.
    pub sims: Option<Vec<f64>>,
}

fn regional(x: &[f64], y: &[f64], region: &[usize], k: usize, size: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut sum = vec![0.0; k * k];
    for a in 0..n {
        for b in 0..n {
            sum[region[a] * k + region[b]] += concordance(x, y, a, b);
        }
    }
    (0..k * k)
        .map(|e| {
            let (r, s) = (e / k, e % k);
            let pairs = if r == s {
                size[r] * (size[r] - 1.0)
            } else {
                size[r] * size[s]
            };
            sum[e] / pairs
        })
        .collect()
}

/// `Tau_Regional`: `T_rs = sum_{a in r, b in s} S_ab / #pairs` (`#pairs = n_r (n_r - 1)` on the
/// diagonal). Permutations shuffle all sites over the fixed regimes; giddy's p-value takes the
/// smaller of the two tail counts.
pub fn tau_regional(
    x: &[f64],
    y: &[f64],
    regime: &[i64],
    permutations: usize,
    seed: u64,
    keep: bool,
) -> Result<RegionalTau, String> {
    check(x, y)?;
    if regime.len() != x.len() {
        return Err(format!(
            "regime has length {} but x has {} (length mismatch)",
            regime.len(),
            x.len()
        ));
    }
    let mut regimes = regime.to_vec();
    regimes.sort_unstable();
    regimes.dedup();
    let k = regimes.len();
    let region: Vec<usize> = regime
        .iter()
        .map(|r| regimes.binary_search(r).unwrap())
        .collect();
    let mut size = vec![0.0; k];
    for &r in &region {
        size[r] += 1.0;
    }
    let tau_reg = regional(x, y, &region, k, &size);
    if permutations == 0 {
        return Ok(RegionalTau {
            regimes,
            tau_reg,
            p: None,
            sims: None,
        });
    }
    let sims: Vec<f64> = (0..permutations)
        .into_par_iter()
        .flat_map_iter(|r| {
            let mut ids: Vec<usize> = (0..x.len()).collect();
            shuffle(&mut ids, &mut replicate_rng(seed, r as u64));
            let (xr, yr): (Vec<f64>, Vec<f64>) = ids.iter().map(|&i| (x[i], y[i])).unzip();
            regional(&xr, &yr, &region, k, &size)
        })
        .collect();
    let p = (0..k * k)
        .map(|e| {
            let obs = tau_reg[e];
            let column = (0..permutations).map(|r| sims[r * k * k + e]);
            let (mut larger, mut smaller) = (0.0, 0.0);
            for s in column {
                if s >= obs {
                    larger += 1.0;
                }
                if s <= obs {
                    smaller += 1.0;
                }
            }
            let side = if smaller < larger { smaller } else { larger };
            (1.0 + side) / (1.0 + permutations as f64)
        })
        .collect();
    Ok(RegionalTau {
        regimes,
        tau_reg,
        p: Some(p),
        sims: keep.then_some(sims),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const X: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    const Y: [f64; 5] = [2.0, 1.0, 4.0, 3.0, 5.0];

    /// Path 0-1-2-3 plus island 4.
    fn path() -> Csr {
        Csr::from_arrays(5, &[0, 1, 1, 2, 2, 3], &[1, 0, 2, 1, 3, 2], &[1.0; 6]).unwrap()
    }

    #[test]
    fn tau_local_sums_concordances() {
        let (tau, local) = tau_local(&X, &Y).unwrap();
        // Discordant pairs: (0, 1) and (2, 3); 8 of 10 pairs concordant.
        assert!((tau - 0.6).abs() < 1e-15);
        assert!((local[0] - 0.5).abs() < 1e-15 && (local[4] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn neighbor_and_neighborhood_statistics() {
        let (r, w) = tau_local_neighbor(&X, &Y, &path(), 0, 0, false).unwrap();
        assert_eq!(&r.value[..4], &[-1.0, 0.0, 0.0, -1.0]);
        assert!(r.value[4].is_nan());
        assert_eq!(w, vec![1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 0.0]);
        let h = tau_local_neighborhood(&X, &Y, &path(), 0, 0, false).unwrap();
        // {0, 1}: one discordant pair; {0, 1, 2}: S_01 = -1, S_02 = S_12 = 1.
        assert_eq!(h.value[0], -1.0);
        assert!((h.value[1] - 1.0 / 3.0).abs() < 1e-15);
        assert!(h.value[4].is_nan());
    }

    #[test]
    fn permutations_are_seeded_and_islands_stay_nan() {
        let a = tau_local_neighbor(&X, &Y, &path(), 99, 4, true).unwrap().0;
        let b = tau_local_neighbor(&X, &Y, &path(), 99, 4, true).unwrap().0;
        let bits = |v: &Option<Vec<f64>>| {
            v.as_ref()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&a.p), bits(&b.p));
        let p = a.p.unwrap();
        assert!(p[4].is_nan() && p[..4].iter().all(|v| *v > 0.0 && *v <= 1.0));
        assert_eq!(a.sims.unwrap().len(), 5 * 99);
    }

    #[test]
    fn regional_tau_over_two_regimes() {
        let r = tau_regional(&X, &Y, &[1, 1, 2, 2, 2], 49, 1, false).unwrap();
        assert_eq!(r.regimes, vec![1, 2]);
        // Regime 1 = {0, 1}: the discordant pair, T_11 = -2 / 2.
        assert_eq!(r.tau_reg[0], -1.0);
        assert_eq!(r.p.unwrap().len(), 4);
    }
}
