//! Spatial Markov (`giddy.markov.Spatial_Markov`): transitions conditioned on the class of the
//! spatial lag.

use rayon::prelude::*;

use super::markov::{check_classes, chi2_test, count_transitions};
use crate::classify::bins::{bin1d, quantile_bins};
use crate::graph::Csr;
use crate::limits;
use crate::perm::{replicate_rng, shuffle};

/// Options of [`spatial_markov`].
#[derive(Clone, Debug)]
pub struct SpatialMarkovSettings {
    /// Number of classes of the variable (the number of labels when `discrete`).
    pub k: usize,
    /// Number of classes of the spatial lag.
    pub m: usize,
    /// Classify all periods together (`true`) or period by period.
    pub fixed: bool,
    /// `y` already holds class indices `0..k`; the lag is the categorical lag.
    pub discrete: bool,
    /// Upper class bounds of the variable (the last, infinite, bound is implied).
    pub cutoffs: Option<Vec<f64>>,
    /// Upper class bounds of the lag.
    pub lag_cutoffs: Option<Vec<f64>>,
    /// Row permutations for the `x2` pseudo p-value (0 skips it).
    pub permutations: usize,
    /// Seed of the permutation streams.
    pub seed: u64,
}

/// Result of [`spatial_markov`].
#[derive(Clone, Debug)]
pub struct SpatialMarkov {
    /// Number of classes of the variable.
    pub k: usize,
    /// Number of classes of the lag.
    pub m: usize,
    /// `n x t` class indices of the variable.
    pub class_ids: Vec<u32>,
    /// `n x t` class indices of the lag.
    pub lclass_ids: Vec<u32>,
    /// Class upper bounds without the final infinite one (`None` when classified per period).
    pub cutoffs: Option<Vec<f64>>,
    /// Same for the lag.
    pub lag_cutoffs: Option<Vec<f64>>,
    /// Classic transition counts `k x k`.
    pub transitions: Vec<f64>,
    /// Conditional counts `m x k x k`.
    pub t_cond: Vec<f64>,
    /// Observed `x2 = sum_l chi2(T_l, transitions)`.
    pub x2: f64,
    /// Simulated `x2` per permutation.
    pub x2_sims: Vec<f64>,
    /// `(#{x2* >= x2} + 1) / (P + 1)` when permutations were requested.
    pub x2_rpvalue: Option<f64>,
}

/// Classifies an `n x t` row-major panel. Returns `(ids, cutoffs, number of classes)`.
fn classify_panel(
    vals: &[f64],
    n: usize,
    t: usize,
    k: usize,
    fixed: bool,
    cutoffs: Option<&[f64]>,
) -> (Vec<u32>, Option<Vec<f64>>, usize) {
    if let Some(c) = cutoffs {
        let mut bins = c.to_vec();
        bins.push(f64::INFINITY);
        let (ids, _) = bin1d(vals, &bins);
        return (ids, Some(c.to_vec()), bins.len());
    }
    if fixed {
        let (bins, _) = quantile_bins(vals, k);
        let (ids, _) = bin1d(vals, &bins);
        let cut = bins[..bins.len() - 1].to_vec();
        return (ids, Some(cut), bins.len());
    }
    let mut ids = vec![0u32; n * t];
    for s in 0..t {
        let col: Vec<f64> = (0..n).map(|i| vals[i * t + s]).collect();
        let (bins, _) = quantile_bins(&col, k);
        let (yb, _) = bin1d(&col, &bins);
        for i in 0..n {
            ids[i * t + s] = yb[i];
        }
    }
    (ids, None, k)
}

/// Weighted mode of the neighbors' classes (`libpysal.weights.lag_categorical`, `ties="tryself"`).
///
/// Ties are retried with the focal class added at the mean neighbor weight; a remaining tie goes
/// to the lowest class (libpysal draws at random). Islands keep their own class.
pub fn lag_categorical(g: &Csr, ids: &[u32], k: usize) -> Vec<u32> {
    (0..g.n)
        .map(|i| {
            let mut tally = vec![0.0; k];
            let (mut sum, mut count) = (0.0, 0usize);
            for e in g.offsets[i]..g.offsets[i + 1] {
                let w = g.weights[e];
                if w != 0.0 {
                    tally[ids[g.neighbors[e] as usize] as usize] += w;
                    sum += w;
                    count += 1;
                }
            }
            if count == 0 {
                return ids[i];
            }
            let winners = |tally: &[f64]| -> Vec<usize> {
                let max = tally.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (0..k).filter(|&c| tally[c] == max).collect()
            };
            let mut w = winners(&tally);
            if w.len() > 1 {
                tally[ids[i] as usize] += sum / count as f64;
                w = winners(&tally);
            }
            w[0] as u32
        })
        .collect()
}

/// Categorical lag of every column of an `n x t` panel.
fn lag_categorical_panel(g: &Csr, ids: &[u32], n: usize, t: usize, k: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * t];
    for s in 0..t {
        let col: Vec<u32> = (0..n).map(|i| ids[i * t + s]).collect();
        for (i, v) in lag_categorical(g, &col, k).into_iter().enumerate() {
            out[i * t + s] = v;
        }
    }
    out
}

/// Spatial lag `W y` of every column; `rows[i]` is the location whose values sit at row `i`.
fn lag_panel(g: &Csr, y: &[f64], rows: &[usize], t: usize) -> Vec<f64> {
    let mut out = vec![0.0; g.n * t];
    for i in 0..g.n {
        for e in g.offsets[i]..g.offsets[i + 1] {
            let src = rows[g.neighbors[e] as usize] * t;
            let w = g.weights[e];
            for s in 0..t {
                out[i * t + s] += w * y[src + s];
            }
        }
    }
    out
}

fn conditional_counts(lc: &[u32], ids: &[u32], n: usize, t: usize, m: usize, k: usize) -> Vec<f64> {
    let mut counts = vec![0.0; m * k * k];
    for i in 0..n {
        for s in 0..t.saturating_sub(1) {
            let l = lc[i * t + s] as usize;
            let a = ids[i * t + s] as usize;
            let b = ids[i * t + s + 1] as usize;
            counts[(l * k + a) * k + b] += 1.0;
        }
    }
    counts
}

fn x2_of(t_cond: &[f64], transitions: &[f64], m: usize, k: usize) -> f64 {
    (0..m)
        .map(|l| chi2_test(&t_cond[l * k * k..(l + 1) * k * k], transitions, k).0)
        .sum()
}

/// Spatial Markov of an `n x t` panel `y` (row-major) over the graph `g` (used as given).
pub fn spatial_markov(
    y: &[f64],
    n: usize,
    t: usize,
    g: &Csr,
    s: &SpatialMarkovSettings,
) -> Result<SpatialMarkov, String> {
    if y.len() != n * t || g.n != n {
        return Err(format!(
            "y has {} values for {n} x {t}, graph has {} nodes",
            y.len(),
            g.n
        ));
    }
    if t < 2 {
        return Err("at least two time periods are needed".to_string());
    }
    // Requested sizes bound the classified ones, so the tables are checked before any work.
    let (k_max, m_max) = (s.k, if s.discrete { s.k } else { s.m });
    check_classes(k_max)?;
    check_classes(m_max)?;
    let cells = limits::checked_product("m * k * k", m_max, k_max)
        .and_then(|mk| limits::checked_product("m * k * k", mk, k_max))?;
    limits::ensure_within(
        "conditional transition table (m * k * k)",
        cells,
        limits::MAX_CONDITIONAL_CELLS,
        "use fewer classes k or lag classes m",
    )?;
    let identity: Vec<usize> = (0..n).collect();
    let (class_ids, cutoffs, k, m, lclass_ids, lag_cutoffs);
    if s.discrete {
        k = s.k;
        m = k;
        class_ids = y
            .iter()
            .map(|&v| {
                if v >= 0.0 && v.fract() == 0.0 && (v as usize) < k {
                    Ok(v as u32)
                } else {
                    Err(format!("class index {v} is out of range for {k} classes"))
                }
            })
            .collect::<Result<Vec<u32>, String>>()?;
        cutoffs = None;
        lag_cutoffs = None;
        lclass_ids = lag_categorical_panel(g, &class_ids, n, t, k);
    } else {
        let (ids, cut, kk) = classify_panel(y, n, t, s.k, s.fixed, s.cutoffs.as_deref());
        class_ids = ids;
        cutoffs = cut;
        k = kk;
        let ly = lag_panel(g, y, &identity, t);
        let (lc, lcut, mm) = classify_panel(&ly, n, t, s.m, s.fixed, s.lag_cutoffs.as_deref());
        lclass_ids = lc;
        lag_cutoffs = lcut;
        m = mm;
    }
    let transitions = count_transitions(&class_ids, n, t, k)?;
    let t_cond = conditional_counts(&lclass_ids, &class_ids, n, t, m, k);
    let x2 = x2_of(&t_cond, &transitions, m, k);
    let mut x2_sims = Vec::new();
    let mut x2_rpvalue = None;
    if s.permutations > 0 {
        x2_sims = (0..s.permutations)
            .into_par_iter()
            .map(|r| {
                let mut rows = identity.clone();
                shuffle(&mut rows, &mut replicate_rng(s.seed, r as u64));
                let lc = if s.discrete {
                    let mut pc = vec![0u32; n * t];
                    for i in 0..n {
                        pc[i * t..(i + 1) * t]
                            .copy_from_slice(&class_ids[rows[i] * t..(rows[i] + 1) * t]);
                    }
                    lag_categorical_panel(g, &pc, n, t, k)
                } else {
                    let ly = lag_panel(g, y, &rows, t);
                    classify_panel(&ly, n, t, s.m, s.fixed, lag_cutoffs.as_deref()).0
                };
                let tc = conditional_counts(&lc, &class_ids, n, t, m, k);
                x2_of(&tc, &transitions, m, k)
            })
            .collect();
        let larger = x2_sims.iter().filter(|&&v| v >= x2).count();
        x2_rpvalue = Some((larger as f64 + 1.0) / (s.permutations as f64 + 1.0));
    }
    Ok(SpatialMarkov {
        k,
        m,
        class_ids,
        lclass_ids,
        cutoffs,
        lag_cutoffs,
        transitions,
        t_cond,
        x2,
        x2_sims,
        x2_rpvalue,
    })
}
