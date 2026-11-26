//! Rank mobility (`giddy.rank`): Kendall's tau, spatial tau and the regime measure theta.

use rayon::prelude::*;

use crate::graph::Csr;
use crate::perm::{erfc, replicate_rng, shuffle};

/// giddy divides by this 8-digit approximation of the square root of two ("follow scipy");
/// kept for parity.
#[allow(clippy::approx_constant)]
const GIDDY_SQRT_2: f64 = 1.414_213_6;

/// Result of [`tau`].
#[derive(Clone, Debug)]
pub struct TauResult {
    /// Kendall's tau-b.
    pub tau: f64,
    /// Asymptotic two-sided p-value (giddy's `erfc(|z| / 1.4142136)`).
    pub p: f64,
    /// Concordant pairs.
    pub concordant: f64,
    /// Discordant pairs.
    pub discordant: f64,
    /// Pairs tied in `y` only.
    pub extra_x: f64,
    /// Pairs tied in `x` only.
    pub extra_y: f64,
}

/// Kendall's tau-b of `x` and `y` (ties handled; classic tau when there are none).
///
/// Knight's O(n log n) algorithm: sort by `(x, y)`, count the joint and the `x` ties, then
/// count the discordant pairs as the inversions of a merge sort by `y`. The pair counts equal
/// giddy's O(n^2) loop ([`tau_naive`]) for finite inputs.
pub fn tau(x: &[f64], y: &[f64]) -> TauResult {
    let n = x.len();
    // `+ 0.0` folds -0.0 into 0.0 so that the total order agrees with `==`.
    let x: Vec<f64> = x.iter().map(|v| v + 0.0).collect();
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        x[a].total_cmp(&x[b])
            .then(y_of(y, a).total_cmp(&y_of(y, b)))
    });
    let (mut tied_x, mut tied_xy) = (0u64, 0u64);
    let (mut run_x, mut run_xy) = (1u64, 1u64);
    for w in 0..n {
        let i = order[w];
        ys.push(y_of(y, i));
        if w == 0 {
            continue;
        }
        let j = order[w - 1];
        if x[i] == x[j] {
            tied_x += run_x;
            run_x += 1;
            if y_of(y, i) == y_of(y, j) {
                tied_xy += run_xy;
                run_xy += 1;
            } else {
                run_xy = 1;
            }
        } else {
            (run_x, run_xy) = (1, 1);
        }
    }
    let discordant = merge_inversions(&mut ys);
    // After the merge sort `ys` is ascending, so the `y` ties are adjacent runs.
    let (mut tied_y, mut run_y) = (0u64, 1u64);
    for w in 1..n {
        if ys[w] == ys[w - 1] {
            tied_y += run_y;
            run_y += 1;
        } else {
            run_y = 1;
        }
    }
    let all = (n as u64) * (n as u64).saturating_sub(1) / 2;
    let untied = all + tied_xy - tied_x - tied_y;
    finish(
        n,
        (untied - discordant) as f64,
        discordant as f64,
        (tied_y - tied_xy) as f64,
        (tied_x - tied_xy) as f64,
    )
}

fn y_of(y: &[f64], i: usize) -> f64 {
    y[i] + 0.0
}

/// Sorts `v` ascending (stable) and returns the number of strict inversions `i < j, v[i] > v[j]`.
fn merge_inversions(v: &mut [f64]) -> u64 {
    let n = v.len();
    let mut buf = v.to_vec();
    let mut inversions = 0u64;
    let mut width = 1;
    while width < n {
        let mut lo = 0;
        while lo < n {
            let mid = (lo + width).min(n);
            let hi = (lo + 2 * width).min(n);
            let (mut i, mut j, mut k) = (lo, mid, lo);
            while i < mid && j < hi {
                if v[j] < v[i] {
                    buf[k] = v[j];
                    inversions += (mid - i) as u64;
                    j += 1;
                } else {
                    buf[k] = v[i];
                    i += 1;
                }
                k += 1;
            }
            buf[k..k + mid - i].copy_from_slice(&v[i..mid]);
            k += mid - i;
            buf[k..k + hi - j].copy_from_slice(&v[j..hi]);
            lo = hi;
        }
        v.copy_from_slice(&buf);
        width *= 2;
    }
    inversions
}

/// giddy's O(n^2) pair loop, kept as the reference for [`tau`].
#[cfg(test)]
pub(super) fn tau_naive(x: &[f64], y: &[f64]) -> TauResult {
    let n = x.len();
    let (mut c, mut d, mut ex, mut ey) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        for j in i + 1..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let dxdy = dx * dy;
            if dxdy > 0.0 {
                c += 1.0;
            } else if dxdy < 0.0 {
                d += 1.0;
            } else if dx == 0.0 && dy != 0.0 {
                ey += 1.0;
            } else if dy == 0.0 && dx != 0.0 {
                ex += 1.0;
            }
        }
    }
    finish(n, c, d, ex, ey)
}

/// tau-b and giddy's asymptotic p-value from the pair counts.
fn finish(n: usize, c: f64, d: f64, ex: f64, ey: f64) -> TauResult {
    let cd = c + d;
    let tau = (c - d) / ((cd + ex) * (cd + ey)).sqrt();
    let nf = n as f64;
    let v = (4.0 * nf + 10.0) / (9.0 * nf * (nf - 1.0));
    let z = tau / v.sqrt();
    TauResult {
        tau,
        p: erfc(z.abs() / GIDDY_SQRT_2),
        concordant: c,
        discordant: d,
        extra_x: ex,
        extra_y: ey,
    }
}

/// Result of [`spatial_tau`].
#[derive(Clone, Debug)]
pub struct SpatialTau {
    /// Classic tau-b over all pairs.
    pub tau: TauResult,
    /// Tau over neighbor pairs only.
    pub tau_spatial: f64,
    /// Number of neighbor pairs, `floor(s0 / 2)` for the binary graph.
    pub pairs_spatial: usize,
    /// Concordant neighbor pairs.
    pub concordant_spatial: f64,
    /// Discordant neighbor pairs.
    pub discordant_spatial: f64,
    /// Simulated `tau_spatial` under random relabelling of locations.
    pub taus: Vec<f64>,
    /// One-sided folded pseudo p-value (`NaN` without permutations).
    pub psim: f64,
}

fn pairs_stat(x: &[f64], y: &[f64], pairs: &[(u32, u32)], perm: Option<&[usize]>) -> [f64; 3] {
    let (mut n1, mut n2, mut i_s, mut gc) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for &(i, j) in pairs {
        let (i, j) = (i as usize, j as usize);
        let (pi, pj) = perm.map_or((i, j), |p| (p[i], p[j]));
        let dx = x[pi] - x[pj];
        let dy = y[pi] - y[pj];
        let dxdy = dx * dy;
        if dxdy != 0.0 {
            n1 += 1.0;
            n2 += 1.0;
            if dxdy > 0.0 {
                gc += 1.0;
                i_s += 1.0;
            } else {
                i_s -= 1.0;
            }
        } else {
            if dx != 0.0 {
                n1 += 1.0;
            }
            if dy != 0.0 {
                n2 += 1.0;
            }
        }
    }
    [i_s / (n1.sqrt() * n2.sqrt()), gc, gc - i_s]
}

/// Spatial tau of `x` and `y` over the neighbor pairs of `g` (weights ignored).
pub fn spatial_tau(
    x: &[f64],
    y: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
) -> Result<SpatialTau, String> {
    if x.len() != y.len() || x.len() != g.n {
        return Err(format!(
            "x ({}), y ({}) and the graph ({} nodes) must have the same length",
            x.len(),
            y.len(),
            g.n
        ));
    }
    let mut pairs = Vec::new();
    for i in 0..g.n {
        for e in g.offsets[i]..g.offsets[i + 1] {
            let j = g.neighbors[e];
            if g.weights[e] != 0.0 && (i as u32) < j {
                pairs.push((i as u32, j));
            }
        }
    }
    let [tau_spatial, gc, gd] = pairs_stat(x, y, &pairs, None);
    let taus: Vec<f64> = (0..permutations)
        .into_par_iter()
        .map(|r| {
            let mut ids: Vec<usize> = (0..g.n).collect();
            shuffle(&mut ids, &mut replicate_rng(seed, r as u64));
            pairs_stat(x, y, &pairs, Some(&ids))[0]
        })
        .collect();
    let psim = if permutations > 0 {
        let larger = taus.iter().filter(|&&v| v >= tau_spatial).count() as f64;
        let p = permutations as f64;
        let psim = (larger + 1.0) / (p + 1.0);
        if psim > 0.5 {
            (p - larger + 1.0) / (p + 1.0)
        } else {
            psim
        }
    } else {
        f64::NAN
    };
    Ok(SpatialTau {
        tau: tau(x, y),
        tau_spatial,
        pairs_spatial: g.nonzero_count() / 2,
        concordant_spatial: gc,
        discordant_spatial: gd,
        taus,
        psim,
    })
}

/// Result of [`theta`].
#[derive(Clone, Debug)]
pub struct Theta {
    /// `theta` per interval (`t - 1`).
    pub theta: Vec<f64>,
    /// Total absolute rank change per interval.
    pub total: Vec<f64>,
    /// Theoretical maximum of the total for `n` locations.
    pub max_total: f64,
    /// `n x t` ranks (average ties) per period.
    pub ranks: Vec<f64>,
    /// `P(sim <= observed)` per interval (empty without permutations).
    pub pvalue_left: Vec<f64>,
    /// `P(sim > observed)` per interval.
    pub pvalue_right: Vec<f64>,
    /// `(observed - mean) / std` per interval.
    pub z: Vec<f64>,
}

fn rank_average(col: &[f64]) -> Vec<f64> {
    let n = col.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| col[a].total_cmp(&col[b]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && col[order[j + 1]] == col[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &o in &order[i..=j] {
            ranks[o] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Regime mobility measure of an `n x t` panel (`giddy.rank.Theta`).
pub fn theta(
    y: &[f64],
    n: usize,
    t: usize,
    regime: &[u32],
    permutations: usize,
    seed: u64,
) -> Result<Theta, String> {
    if y.len() != n * t || regime.len() != n {
        return Err(format!(
            "y has {} values for {n} x {t}, regime has {} entries",
            y.len(),
            regime.len()
        ));
    }
    if t < 2 {
        return Err("at least two time periods are needed".to_string());
    }
    let mut ranks = vec![0.0; n * t];
    for s in 0..t {
        let col: Vec<f64> = (0..n).map(|i| y[i * t + s]).collect();
        for (i, r) in rank_average(&col).into_iter().enumerate() {
            ranks[i * t + s] = r;
        }
    }
    let q = t - 1;
    let d: Vec<f64> = (0..n * q)
        .map(|x| ranks[(x / q) * t + x % q + 1] - ranks[(x / q) * t + x % q])
        .collect();
    let total: Vec<f64> = (0..q)
        .map(|s| (0..n).map(|i| d[i * q + s].abs()).sum())
        .collect();
    let max_total = (1..=n)
        .map(|i| (2.0 * i as f64 - n as f64 - 1.0).abs())
        .sum();
    let groups = regime.iter().max().map_or(0, |&g| g as usize + 1);
    let calc = |reg: &[u32]| -> Vec<f64> {
        let mut sums = vec![0.0; groups * q];
        for i in 0..n {
            for s in 0..q {
                sums[reg[i] as usize * q + s] += d[i * q + s];
            }
        }
        (0..q)
            .map(|s| (0..groups).map(|g| sums[g * q + s].abs()).sum::<f64>() / total[s])
            .collect()
    };
    let observed = calc(regime);
    let (mut left, mut right, mut z) = (Vec::new(), Vec::new(), Vec::new());
    if permutations > 0 {
        let sims: Vec<Vec<f64>> = (0..permutations)
            .into_par_iter()
            .map(|r| {
                let mut reg = regime.to_vec();
                shuffle(&mut reg, &mut replicate_rng(seed, r as u64));
                calc(&reg)
            })
            .collect();
        let rows = permutations as f64 + 1.0;
        for s in 0..q {
            let obs = observed[s];
            let le = 1 + sims.iter().filter(|v| v[s] <= obs).count();
            let gt = sims.iter().filter(|v| v[s] > obs).count();
            left.push(le as f64 / rows);
            right.push(gt as f64 / rows);
            let mean = (obs + sims.iter().map(|v| v[s]).sum::<f64>()) / rows;
            let var = ((obs - mean).powi(2)
                + sims.iter().map(|v| (v[s] - mean).powi(2)).sum::<f64>())
                / rows;
            z.push((obs - mean) / var.sqrt());
        }
    }
    Ok(Theta {
        theta: observed,
        total,
        max_total,
        ranks,
        pvalue_left: left,
        pvalue_right: right,
        z,
    })
}
