//! Seeded, thread-count independent permutation engine shared by every permutation test.
//!
//! # Determinism contract
//!
//! Replicate `r` draws from its own `ChaCha8` stream: `seed_from_u64(seed)` with the stream id
//! set to `r` ([`replicate_rng`]). A replicate therefore depends only on `(seed, r)`, never on
//! how replicates are scheduled across rayon threads, and replicate `r` is the same whether
//! 10 or 10 000 permutations are requested. Bit-for-bit equality with numpy's stream is not a
//! goal; the statistics are, up to Monte Carlo error.
//!
//! The p-value helpers replicate `esda` 2.10.0 (`Moran.p_sim`, `esda/significance.py`).

pub mod crand;

use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};
use rayon::prelude::*;

/// The RNG of replicate `replicate` for a user `seed`: an independent ChaCha8 stream.
pub fn replicate_rng(seed: u64, replicate: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(replicate);
    rng
}

/// Unbiased uniform integer in `0..bound` (Lemire's multiply-shift with rejection).
///
/// `bound` must be non-zero.
pub fn uniform_below<R: Rng + ?Sized>(rng: &mut R, bound: u64) -> u64 {
    debug_assert!(bound > 0);
    let mut product = u128::from(rng.next_u64()) * u128::from(bound);
    let mut low = product as u64;
    if low < bound {
        let threshold = bound.wrapping_neg() % bound;
        while low < threshold {
            product = u128::from(rng.next_u64()) * u128::from(bound);
            low = product as u64;
        }
    }
    (product >> 64) as u64
}

/// In-place Fisher-Yates shuffle.
pub fn shuffle<T, R: Rng + ?Sized>(slice: &mut [T], rng: &mut R) {
    for i in (1..slice.len()).rev() {
        let j = uniform_below(rng, i as u64 + 1) as usize;
        slice.swap(i, j);
    }
}

/// The row permutation of replicate `replicate`: a shuffle of `0..n` from that replicate's
/// stream (so it depends only on `(seed, replicate)`). Used by `Spatex.GWR.spatial_variability`,
/// which re-orders the coordinates of the observations.
pub fn permutation(n: usize, seed: u64, replicate: u64) -> Vec<u32> {
    let mut rows: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut rows, &mut replicate_rng(seed, replicate));
    rows
}

/// Runs `permutations` replicates: shuffle a copy of `base` with the replicate's stream and
/// evaluate `stat` on it. The result is in replicate order and independent of the thread count.
pub fn simulate<F>(base: &[f64], permutations: usize, seed: u64, stat: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    (0..permutations)
        .into_par_iter()
        .map_init(
            || base.to_vec(),
            |buf, r| {
                buf.copy_from_slice(base);
                shuffle(buf, &mut replicate_rng(seed, r as u64));
                stat(buf)
            },
        )
        .collect()
}

/// Folded pseudo p-value used by Moran, Moran_BV, Geary and G in esda:
/// `larger = #{sim >= obs}`, flipped to `P - larger` when that is smaller,
/// `p = (larger + 1) / (P + 1)`. Ties are asymmetric on purpose (`P - larger` counts `sim < obs`).
pub fn folded_p(observed: f64, sim: &[f64]) -> f64 {
    let p = sim.len();
    let mut larger = sim.iter().filter(|&&s| s >= observed).count();
    if p - larger < larger {
        larger = p - larger;
    }
    (larger as f64 + 1.0) / (p as f64 + 1.0)
}

/// The significance alternatives of `esda.significance.calculate_significance`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alternative {
    /// Tail chosen from the data and folded (esda's default for local statistics).
    Directed,
    /// Percentile-mirror two-sided test; may exceed 1 exactly as in esda.
    TwoSided,
    /// Upper tail.
    Greater,
    /// Lower tail.
    Lesser,
    /// Distance from the reference mean, folded.
    Folded,
}

impl Alternative {
    /// Parses `"directed"`, `"two_sided"` (or `"two-sided"`), `"greater"`, `"lesser"`, `"folded"`.
    pub fn parse(s: &str) -> Result<Alternative, String> {
        match s {
            "directed" => Ok(Alternative::Directed),
            "two_sided" | "two-sided" => Ok(Alternative::TwoSided),
            "greater" => Ok(Alternative::Greater),
            "lesser" => Ok(Alternative::Lesser),
            "folded" => Ok(Alternative::Folded),
            other => Err(format!(
                "unknown alternative {other:?}, expected one of directed, two_sided, greater, lesser, folded"
            )),
        }
    }
}

/// numpy `percentile` (linear interpolation) of an ascending-sorted slice; `q` in `[0, 100]`.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = (n - 1) as f64 * (q / 100.0);
    let lo = virtual_index.floor();
    let t = virtual_index - lo;
    let lo = lo as usize;
    let hi = (lo + 1).min(n - 1);
    let (a, b) = (sorted[lo], sorted[hi]);
    // numpy's _lerp, symmetric around t = 0.5.
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// Pseudo p-value of `observed` against the reference distribution `sim`
/// (`esda._permutation_significance`).
pub fn significance(alt: Alternative, observed: f64, sim: &[f64]) -> f64 {
    let p = sim.len() as f64;
    let count = |f: &dyn Fn(f64) -> bool| sim.iter().filter(|&&s| f(s)).count() as f64;
    match alt {
        Alternative::Directed => folded_p(observed, sim),
        Alternative::Greater => (count(&|s| s >= observed) + 1.0) / (p + 1.0),
        Alternative::Lesser => (count(&|s| s <= observed) + 1.0) / (p + 1.0),
        Alternative::TwoSided => {
            let pct = count(&|s| s <= observed) / p * 100.0;
            let p_low = pct.min(100.0 - pct);
            let mut sorted = sim.to_vec();
            sorted.sort_by(f64::total_cmp);
            let low = percentile(&sorted, p_low);
            let high = percentile(&sorted, 100.0 - p_low);
            let outside = count(&|s| s <= low) + count(&|s| s >= high);
            (outside + 1.0) / (p + 1.0)
        }
        Alternative::Folded => {
            let mean = sim.iter().sum::<f64>() / p;
            let folded = (observed - mean).abs();
            (count(&|s| (s - mean).abs() >= folded) + 1.0) / (p + 1.0)
        }
    }
}

/// Complementary error function, accurate to about 1e-15 relative error.
///
/// Below 2 it uses the all-positive series for `erf` (no cancellation); from 2 up it uses the
/// continued fraction for `erfc`, which stays accurate deep into the tail.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let sqrt_pi = std::f64::consts::PI.sqrt();
    if x < 2.0 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0.0;
        while term > 1e-18 * sum {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
        }
        1.0 - 2.0 / sqrt_pi * (-x2).exp() * sum
    } else if x > 27.0 {
        0.0
    } else {
        let mut t = x;
        for k in (1..=120).rev() {
            t = x + (f64::from(k) / 2.0) / t;
        }
        (-x * x).exp() / (sqrt_pi * t)
    }
}

/// Standard normal CDF.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Standard normal survival function, accurate in the far tail.
pub fn norm_sf(z: f64) -> f64 {
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Moments of a simulated reference distribution, as esda reports them.
#[derive(Clone, Copy, Debug)]
pub struct SimSummary {
    /// `sum(sim) / P`.
    pub ei_sim: f64,
    /// Population standard deviation (ddof = 0).
    pub se_sim: f64,
    /// `se_sim^2`.
    pub vi_sim: f64,
    /// `(observed - ei_sim) / se_sim`.
    pub z_sim: f64,
    /// One-tailed normal p of `z_sim` (survival when `z_sim > 0`, else CDF).
    pub p_z_sim: f64,
}

/// `EI_sim`, `seI_sim`, `VI_sim`, `z_sim`, `p_z_sim` for `observed` against `sim`.
pub fn summarize(observed: f64, sim: &[f64]) -> SimSummary {
    let p = sim.len() as f64;
    let ei_sim = sim.iter().sum::<f64>() / p;
    let vi_sim = sim.iter().map(|s| (s - ei_sim) * (s - ei_sim)).sum::<f64>() / p;
    let se_sim = vi_sim.sqrt();
    let z_sim = (observed - ei_sim) / se_sim;
    let p_z_sim = if z_sim > 0.0 {
        norm_sf(z_sim)
    } else {
        norm_cdf(z_sim)
    };
    SimSummary {
        ei_sim,
        se_sim,
        vi_sim,
        z_sim,
        p_z_sim,
    }
}

/// A finished permutation test: the reference distribution, folded pseudo p-value and moments.
#[derive(Clone, Debug)]
pub struct PermTest {
    /// Statistic of every replicate, in replicate order.
    pub sim: Vec<f64>,
    /// Folded pseudo p-value ([`folded_p`]).
    pub p_sim: f64,
    /// `EI_sim`, `seI_sim`, `VI_sim`, `z_sim`, `p_z_sim`.
    pub summary: SimSummary,
}

impl PermTest {
    /// Builds the p-value and moments of `observed` against `sim`.
    pub fn new(observed: f64, sim: Vec<f64>) -> PermTest {
        let p_sim = folded_p(observed, &sim);
        let summary = summarize(observed, &sim);
        PermTest {
            sim,
            p_sim,
            summary,
        }
    }
}

#[cfg(test)]
mod tests;
